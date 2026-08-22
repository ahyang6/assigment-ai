"""Gmail OAuth connection and inbox risk-scanning feature (the Gmail Scan
page). Kept separate from app.py so the Gmail-specific OAuth/PKCE/token
persistence logic can be edited on its own. Imports `detector` from
detection.py (not from app.py) to avoid a circular import - app.py needs
`dashboard` from this file, so this file can't import anything back from
app.py."""
import base64
import html
import json
import re

import pandas as pd
import plotly.express as px
import streamlit as st
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2.credentials import Credentials as GoogleCredentials
from google_auth_oauthlib.flow import Flow as GoogleOAuthFlow
from googleapiclient.discovery import build as build_google_service
from googleapiclient.errors import HttpError as GoogleApiError

from design import RISK_COLORS, RISK_ICONS, page_header, stat_card, style_fig
from detection import BASE_DIR, detector, metrics

GMAIL_TOKEN_PATH = BASE_DIR / "gmail_token.json"
GMAIL_PKCE_PATH = BASE_DIR / "gmail_pkce_verifier.txt"


GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
GMAIL_SETUP_HELP = (
    "Add `google_client_id`, `google_client_secret`, and `google_redirect_uri` "
    "to this app's Streamlit secrets to enable Gmail scanning."
)


def highlight_keywords(text: str, indicators: dict) -> str:
    """Escape the body for safe HTML rendering, then wrap every matched
    keyword/suspicious URL in a highlighted <mark> span. URLs are matched
    first and claim their full span so a keyword that happens to appear
    inside a URL (e.g. 'verify' inside fake-verify.example) doesn't get
    nested/double-highlighted; overlapping spans are merged into one."""
    spans: list[tuple[int, int]] = []

    for url in indicators.get("suspicious_urls", []):
        for m in re.finditer(re.escape(url), text, re.IGNORECASE):
            spans.append((m.start(), m.end()))

    keyword_terms = sorted(
        {w for values in indicators.get("keywords", {}).values() for w in values},
        key=len, reverse=True,
    )
    for term in keyword_terms:
        for m in re.finditer(re.escape(term), text, re.IGNORECASE):
            start, end = m.start(), m.end()
            if any(s <= start < e or s < end <= e for s, e in spans):
                continue  # already covered by a claimed (e.g. URL) span
            spans.append((start, end))

    spans.sort()
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    pieces = []
    cursor = 0
    for start, end in merged:
        pieces.append(html.escape(text[cursor:start]))
        pieces.append(
            f'<mark style="background:#ff3b5c66; color:#fff; border-radius:3px; padding:0 3px;">'
            f'{html.escape(text[start:end])}</mark>'
        )
        cursor = end
    pieces.append(html.escape(text[cursor:]))
    return "".join(pieces).replace("\n", "<br>")


def gmail_oauth_configured() -> bool:
    """Whether the required Google OAuth secrets have been set up."""
    return all(key in st.secrets for key in ("google_client_id", "google_client_secret", "google_redirect_uri"))


def build_gmail_oauth_flow() -> GoogleOAuthFlow:
    """Build the OAuth flow using credentials from Streamlit secrets."""
    client_config = {
        "web": {
            "client_id": st.secrets["google_client_id"],
            "client_secret": st.secrets["google_client_secret"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [st.secrets["google_redirect_uri"]],
        }
    }
    flow = GoogleOAuthFlow.from_client_config(client_config, scopes=GMAIL_SCOPES)
    flow.redirect_uri = st.secrets["google_redirect_uri"]
    return flow


def save_gmail_credentials(credentials: GoogleCredentials) -> None:
    """Persist credentials to a shared file (not just session_state), so a
    connection completed in one browser tab is picked up by other tabs too
    - each tab is its own independent Streamlit session and can't see
    another tab's session_state directly."""
    GMAIL_TOKEN_PATH.write_text(credentials.to_json(), encoding="utf-8")


def load_gmail_credentials() -> GoogleCredentials | None:
    """Load previously-saved credentials from the shared file, if any."""
    if not GMAIL_TOKEN_PATH.exists():
        return None
    try:
        return GoogleCredentials.from_authorized_user_info(
            json.loads(GMAIL_TOKEN_PATH.read_text(encoding="utf-8")), scopes=GMAIL_SCOPES
        )
    except Exception:
        return None


def clear_gmail_credentials() -> None:
    """Remove the saved credentials file (used on disconnect)."""
    if GMAIL_TOKEN_PATH.exists():
        GMAIL_TOKEN_PATH.unlink()


def _extract_gmail_body(payload: dict) -> str:
    """Recursively find and decode the plain-text (or HTML, as fallback)
    body from a Gmail API message payload."""
    if payload.get("mimeType") == "text/plain" and payload.get("body", {}).get("data"):
        return base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="ignore")

    html_fallback = ""
    for part in payload.get("parts", []) or []:
        if part.get("mimeType") == "text/plain" and part.get("body", {}).get("data"):
            return base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8", errors="ignore")
        if part.get("mimeType") == "text/html" and part.get("body", {}).get("data") and not html_fallback:
            html_fallback = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8", errors="ignore")
        if part.get("parts"):
            nested = _extract_gmail_body(part)
            if nested:
                return nested

    if html_fallback:
        return re.sub(r"<[^>]+>", " ", html_fallback)  # strip HTML tags as a simple fallback
    return ""


def _extract_gmail_headers(payload: dict) -> dict:
    """Pull the Subject/From/Date headers out of a Gmail message payload,
    so scan results can show which real email each verdict belongs to."""
    headers = {h.get("name", ""): h.get("value", "") for h in payload.get("headers", [])}
    return {
        "subject": headers.get("Subject") or "(no subject)",
        "from": headers.get("From") or "(unknown sender)",
        "date": headers.get("Date") or "",
    }


def fetch_gmail_messages(credentials, max_results, progress_callback=None) -> list[dict]:
    """Fetch messages in the user's inbox (paginated through the full
    inbox), up to max_results if given (None/0 = no limit, scan
    everything). Returns a list of dicts with subject/from/date/body for
    each message, so callers can show which real email a result belongs
    to - not just an anonymous risk count."""
    service = build_google_service("gmail", "v1", credentials=credentials)
    message_refs = []
    page_token = None
    while True:
        response = service.users().messages().list(
            userId="me", labelIds=["INBOX"], maxResults=500, pageToken=page_token
        ).execute()
        message_refs.extend(response.get("messages", []))
        page_token = response.get("nextPageToken")
        if not page_token or (max_results and len(message_refs) >= max_results):
            break
    if max_results:
        message_refs = message_refs[:max_results]

    emails = []
    total = len(message_refs)
    for i, msg_ref in enumerate(message_refs):
        full_message = service.users().messages().get(userId="me", id=msg_ref["id"], format="full").execute()
        payload = full_message.get("payload", {})
        body = _extract_gmail_body(payload)
        if body.strip():
            emails.append({**_extract_gmail_headers(payload), "body": body})
        if progress_callback:
            progress_callback(i + 1, total)
    return emails


def dashboard() -> None:
    page_header(
        "📊", "Gmail Scan", "root@messageguard:~$ connect a gmail inbox to scan its risk",
        extra_style="<style>.block-container { max-width: 1280px !important; }</style>",
    )

    if not gmail_oauth_configured():
        st.warning(f"Gmail integration isn't configured yet. {GMAIL_SETUP_HELP}")
        return

    if "gmail_credentials" not in st.session_state:
        loaded = load_gmail_credentials()
        if loaded:
            st.session_state.gmail_credentials = loaded

    if "gmail_credentials" not in st.session_state:
        auth_code = st.query_params.get("code")
        if auth_code:
            try:
                flow = build_gmail_oauth_flow()
                # The auth URL was generated in a separate tab/session, so
                # the PKCE code_verifier that belongs to it has to be
                # recovered from the shared file rather than regenerated
                # here - it must match exactly what was sent to Google.
                if GMAIL_PKCE_PATH.exists():
                    flow.code_verifier = GMAIL_PKCE_PATH.read_text(encoding="utf-8").strip()
                flow.fetch_token(code=auth_code)
                # Only consume the verifier file once we know the exchange
                # actually succeeded - deleting it unconditionally on read
                # meant a Streamlit rerun that reached this code again after
                # a failed attempt (with the same stale ?code= still in the
                # URL) would find it already gone and fail with a confusing
                # "missing code verifier" error instead of the real problem.
                if GMAIL_PKCE_PATH.exists():
                    GMAIL_PKCE_PATH.unlink()
                st.session_state.gmail_credentials = flow.credentials
                save_gmail_credentials(flow.credentials)
                st.query_params.clear()
                st.session_state.gmail_just_connected = True
                st.rerun()
            except Exception as e:
                st.error(f"Gmail authorization failed: {e}")
                # Authorization codes are single-use - clearing the code
                # from the URL here prevents a later rerun from silently
                # retrying (and failing on) this same already-spent code.
                st.query_params.clear()
        elif st.session_state.get("gmail_just_connected"):
            # This tab just finished the OAuth exchange (opened as the
            # separate authorization tab) - the credentials are already
            # saved to the shared file, so the original tab will pick them
            # up on its next auto-refresh. Nothing more to do in this one.
            st.success("✅ Gmail account connected! You can close this tab now and go back to your original tab.")
            st.html("<script>setTimeout(function(){ window.close(); }, 2500);</script>")
        else:
            st.write("Connect your Gmail account to scan your inbox and see how many emails are Low, Medium, or High risk.")
            flow = build_gmail_oauth_flow()
            flow.autogenerate_code_verifier = True
            auth_url, _ = flow.authorization_url(access_type="offline", include_granted_scopes="true", prompt="consent")
            # Persist the verifier this Flow instance just generated, so
            # whichever session ends up exchanging the code (this tab or a
            # separate new one) can retrieve the matching value.
            GMAIL_PKCE_PATH.write_text(flow.code_verifier, encoding="utf-8")
            # Streamlit's own iframe sandboxing doesn't include
            # allow-top-navigation (a known, still-open Streamlit platform
            # limitation - see github.com/streamlit/streamlit/issues/6922),
            # so target="_top" links reliably get blocked rather than
            # navigating the tab. st.link_button (opens a new tab) is the
            # only approach that's actually worked end-to-end.
            st.link_button("🔗 Connect Gmail Account (opens a new tab)", auth_url, type="primary", use_container_width=True)
            st.caption("After authorizing in the new tab, this page will pick up the connection automatically within a few seconds.")
            # Poll for the connection completing in the other tab by
            # reloading this tab periodically - session_state can't be
            # shared across tabs directly, so this is what lets the
            # *original* tab notice the shared credentials file appearing.
            st.html("<script>setTimeout(function(){ window.location.reload(); }, 3000);</script>")
        return

    credentials = st.session_state.gmail_credentials
    if credentials.expired and credentials.refresh_token:
        credentials.refresh(GoogleAuthRequest())
        st.session_state.gmail_credentials = credentials
        save_gmail_credentials(credentials)

    status_col, disconnect_col = st.columns([3, 1])
    with status_col:
        st.success("✅ Gmail account connected.")
    with disconnect_col:
        if st.button("Disconnect Gmail", use_container_width=True):
            del st.session_state.gmail_credentials
            st.session_state.pop("gmail_scan_results", None)
            st.session_state.pop("gmail_scan_emails", None)
            st.session_state.pop("gmail_last_algorithm", None)
            st.session_state.pop("gmail_just_connected", None)
            clear_gmail_credentials()
            st.rerun()

    info = metrics()
    algorithm_names = info.get("candidate_names") or list(info.get("models", {}).keys()) or ["Support Vector Machine"]
    default_algorithm = "Support Vector Machine" if "Support Vector Machine" in algorithm_names else algorithm_names[0]
    selected_algorithm = st.selectbox(
        "Detection algorithm",
        algorithm_names,
        index=algorithm_names.index(default_algorithm),
        key="gmail_selected_algorithm",
        help="Choose which trained algorithm analyzes your inbox.",
    )

    limit_input = st.number_input(
        "Max emails to scan (0 = entire inbox)", min_value=0, value=0, step=50,
    )

    def _run_detection(emails: list[dict], algorithm_name: str) -> list[dict]:
        results = []
        for email in emails:
            try:
                result = detector(algorithm_name).analyze(email["body"])
                results.append({
                    "Subject": email["subject"],
                    "From": email["from"],
                    "Date": email["date"],
                    "Body": email["body"],
                    "Prediction": result["prediction"],
                    "Risk Score": result["risk_score"],
                    "Confidence": result["confidence"],
                    "Category": result["category"],
                    "Explanation": result["explanation"],
                    "Indicators": result["indicators"],
                })
            except ValueError:
                continue  # empty/unanalyzable message body
        return results

    if st.button("🔍 Scan Inbox", type="primary", use_container_width=True):
        try:
            progress_bar = st.progress(0.0, text="Fetching inbox...")

            def update_progress(done, total):
                progress_bar.progress(done / total if total else 0.0, text=f"Analysing email {done}/{total}...")

            emails = fetch_gmail_messages(credentials, max_results=limit_input or None, progress_callback=update_progress)
            # Cache the raw fetched emails (not just the analysis results) so
            # switching the algorithm afterwards can re-analyze them directly
            # without hitting the Gmail API again.
            st.session_state.gmail_scan_emails = emails
            st.session_state.gmail_scan_results = _run_detection(emails, selected_algorithm)
            st.session_state.gmail_last_algorithm = selected_algorithm
            progress_bar.empty()
            st.success(f"Scanned {len(emails)} emails.")
        except GoogleApiError as e:
            st.error(f"Gmail API error: {e}")
        except Exception as e:
            st.error(f"Scan failed: {e}")

    # If the user picks a different algorithm after already scanning once,
    # automatically re-run detection on the same already-fetched emails -
    # no need to re-fetch from Gmail just to try a different algorithm.
    cached_emails = st.session_state.get("gmail_scan_emails")
    if cached_emails and st.session_state.get("gmail_last_algorithm") != selected_algorithm:
        with st.spinner(f"Re-analysing {len(cached_emails)} emails with {selected_algorithm}..."):
            st.session_state.gmail_scan_results = _run_detection(cached_emails, selected_algorithm)
        st.session_state.gmail_last_algorithm = selected_algorithm

    results = st.session_state.get("gmail_scan_results")
    if results:
        results_df = pd.DataFrame(results)
        counts = results_df["Prediction"].value_counts()
        chart_df = pd.DataFrame({"Risk": counts.index, "Count": counts.values})

        st.html("<div style='margin-top:1rem;'></div>")
        kpi = st.columns(4)
        with kpi[0]:
            stat_card(len(results), "Emails Scanned", color="#00e5ff")
        with kpi[1]:
            stat_card(int(counts.get("low", 0)), "Low Risk", color=RISK_COLORS["low"])
        with kpi[2]:
            stat_card(int(counts.get("medium", 0)), "Medium Risk", color=RISK_COLORS["medium"])
        with kpi[3]:
            stat_card(int(counts.get("high", 0)), "High Risk", color=RISK_COLORS["high"])

        st.html("<div style='margin-top:1rem;'></div>")
        st.html('<div class="mg-panel-title">Inbox Risk Breakdown</div>')
        st.plotly_chart(
            style_fig(px.pie(chart_df, names="Risk", values="Count", color="Risk", color_discrete_map=RISK_COLORS)),
            use_container_width=True,
        )

        st.html("<div style='margin-top:1rem;'></div>")
        st.html(f'<div class="mg-panel-title">Scanned Emails - Details (Algorithm: {selected_algorithm})</div>')

        col_widths = [3, 2, 1.1, 1, 1.6, 1]
        header_cols = st.columns(col_widths)
        for col, label in zip(header_cols, ["Subject", "From", "Risk", "Score", "Category", ""]):
            col.markdown(f"**{label}**")
        st.html("<hr style='margin:0.3rem 0 0.6rem 0;'>")

        display_df = results_df.sort_values("Risk Score", ascending=False).reset_index(drop=True)

        @st.dialog("Email Content", width="large")
        def show_email_dialog(row: pd.Series) -> None:
            color = RISK_COLORS.get(row["Prediction"], "")
            icon = RISK_ICONS.get(row["Prediction"], "")
            st.markdown(f"### {row['Subject']}")
            st.caption(f"From: {row['From']}  |  {row['Date'] or '—'}")
            st.markdown(
                f"<span style='color:{color}; font-weight:700; font-size:1.05rem;'>"
                f"{icon} {row['Prediction'].upper()} — {row['Risk Score']}/100</span>"
                f"<br>**Category:** {row['Category']}",
                unsafe_allow_html=True,
            )
            st.write(f"**Why flagged:** {row['Explanation']}")
            st.divider()
            st.caption("Highlighted text was detected as a risk indicator (keyword or suspicious link).")
            st.html(
                f"<div style='line-height:1.7; white-space:pre-wrap; word-break:break-word;'>"
                f"{highlight_keywords(row['Body'], row.get('Indicators', {}))}</div>"
            )

        for i, row in display_df.iterrows():
            row_cols = st.columns(col_widths)
            row_cols[0].write(row["Subject"])
            row_cols[1].write(row["From"])
            icon = RISK_ICONS.get(row["Prediction"], "")
            color = RISK_COLORS.get(row["Prediction"], "")
            row_cols[2].markdown(
                f"<span style='color:{color}; font-weight:700;'>{icon} {row['Prediction'].upper()}</span>",
                unsafe_allow_html=True,
            )
            row_cols[3].write(f"{row['Risk Score']}/100")
            row_cols[4].write(row["Category"])
            if row_cols[5].button("查看", key=f"view_email_{i}", use_container_width=True):
                show_email_dialog(row)