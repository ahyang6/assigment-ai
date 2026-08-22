"""Gmail OAuth connection and inbox risk-scanning feature (the Dashboard
page). Kept separate from app.py so the Gmail-specific OAuth/PKCE/token
persistence logic can be edited on its own. Imports `detector` from
detection.py (not from app.py) to avoid a circular import - app.py needs
`dashboard` from this file, so this file can't import anything back from
app.py."""
import base64
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
from detection import BASE_DIR, detector

GMAIL_TOKEN_PATH = BASE_DIR / "gmail_token.json"
GMAIL_PKCE_PATH = BASE_DIR / "gmail_pkce_verifier.txt"


GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
GMAIL_SETUP_HELP = (
    "Add `google_client_id`, `google_client_secret`, and `google_redirect_uri` "
    "to this app's Streamlit secrets to enable Gmail scanning."
)


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
        "📊", "Statistics Dashboard", "root@messageguard:~$ connect a gmail inbox to scan its risk",
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
            st.session_state.pop("gmail_just_connected", None)
            clear_gmail_credentials()
            st.rerun()

    limit_input = st.number_input(
        "Max emails to scan (0 = entire inbox)", min_value=0, value=0, step=50,
    )

    if st.button("🔍 Scan Inbox", type="primary", use_container_width=True):
        try:
            progress_bar = st.progress(0.0, text="Fetching inbox...")

            def update_progress(done, total):
                progress_bar.progress(done / total if total else 0.0, text=f"Analysing email {done}/{total}...")

            emails = fetch_gmail_messages(credentials, max_results=limit_input or None, progress_callback=update_progress)
            results = []
            for email in emails:
                try:
                    result = detector().analyze(email["body"])
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
                    })
                except ValueError:
                    continue  # empty/unanalyzable message body
            progress_bar.empty()
            st.session_state.gmail_scan_results = results
            st.success(f"Scanned {len(emails)} emails.")
        except GoogleApiError as e:
            st.error(f"Gmail API error: {e}")
        except Exception as e:
            st.error(f"Scan failed: {e}")

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
        st.html('<div class="mg-panel-title">Scanned Emails - Details</div>')
        st.caption("Click any email below to see its full content.")

        display_df = results_df.sort_values("Risk Score", ascending=False).reset_index(drop=True)
        for _, row in display_df.iterrows():
            icon = RISK_ICONS.get(row["Prediction"], "")
            color = RISK_COLORS.get(row["Prediction"], "")
            header = f"{icon} {row['Risk Score']}/100 · {row['Subject']} — {row['From']}"
            with st.expander(header):
                st.markdown(
                    f"<span style='color:{color}; font-weight:700;'>{row['Prediction'].upper()}</span>"
                    f" &nbsp;|&nbsp; **Category:** {row['Category']}"
                    f" &nbsp;|&nbsp; **Date:** {row['Date'] or '—'}",
                    unsafe_allow_html=True,
                )
                st.write(f"**Why:** {row['Explanation']}")
                st.html("<div style='margin-top:0.6rem;'></div>")
                st.html('<div class="mg-panel-title">Full Email Content</div>')
                st.text(row["Body"])