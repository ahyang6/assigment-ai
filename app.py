"""Streamlit interface for the AI-powered spam and phishing detector.

This is the UI/routing layer: page functions, navigation, file
translation, and PDF export. The machine-learning logic lives in
detection.py and the Gmail scanning feature lives in gmail_integration.py
- both are imported here, not duplicated."""
from io import BytesIO
from pathlib import Path

import docx
import pandas as pd
import plotly.express as px
import streamlit as st
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from pypdf import PdfReader
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from detection import (
    append_history, categorize_message, clear_history, delete_history_rows,
    detector, ensure_trained, load_history, metrics,
)
from design import (
    CATEGORY_ICONS, RISK_COLORS, RISK_ICONS,
    apply_theme, page_header, render_hero,
    stat_card, style_fig,
)
from detection import (
    append_history, categorize_message, clear_history, delete_history_rows,
    detector, load_history, metrics,
)
from gmail_integration import dashboard


if "started" not in st.session_state:
    st.session_state.started = False
if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Home"
if "sidebar_visible" not in st.session_state:
    st.session_state.sidebar_visible = True

# The "Get Started" button lives inside the hero's HTML component (so it's
# visually one piece with the background) rather than as a normal Streamlit
# button. A component iframe can't call Streamlit callbacks directly, so the
# button is a plain link that navigates the *parent* page to `?start=1`;
# we catch that here and translate it into normal session-state navigation.
if st.query_params.get("start") == "1":
    st.session_state.started = True
    st.session_state.nav_page = "Analyze Message"
    st.query_params.clear()
elif st.query_params.get("code"):
    # Returning from the Google OAuth redirect (Dashboard's Gmail connect
    # flow). This is a fresh page load, so without this check the user
    # would land back on the Home hero instead - and clicking "Get Started"
    # from there would wipe the ?code= param (it navigates to a bare
    # "?start=1") before Dashboard ever gets a chance to exchange it for a
    # token. Skip straight to Dashboard and leave the param in place; its
    # own OAuth handling reads and clears it after a successful exchange.
    st.session_state.started = True
    st.session_state.nav_page = "Dashboard"

st.set_page_config(
    page_title="Message Guard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = ["Home", "Analyze Message", "Dashboard", "History", "File Translation"]


def result_pdf(message: str, result: dict) -> bytes:
    """Make a small downloadable report for one result."""
    buffer = BytesIO(); pdf = canvas.Canvas(buffer, pagesize=letter)
    text = pdf.beginText(48, 740); text.setFont("Helvetica", 11)
    category = result.get("category") or categorize_message(result["prediction"], result["indicators"])
    lines = ["Message Guard - Analysis Report", "", f"Prediction: {result['prediction'].title()}",
             f"Category: {category}",
             f"Confidence: {result['confidence']:.1%}", f"Risk: {result['risk_score']}/100 ({result['risk_level']})", "",
             "Explanation:", result['explanation'], "",
             "Probability Distribution:"]
    lines += [f"  {cls.title()}: {prob:.1%}" for cls, prob in result["probabilities"].items()]
    words = [w.upper() for values in result["indicators"]["keywords"].values() for w in values]
    lines += ["", "Detected Keywords: " + (", ".join(words) if words else "None")]
    if result["indicators"]["urls"]:
        lines.append("Suspicious URLs: " + ", ".join(result["indicators"]["urls"]))
    lines += ["", "Message:"]
    for line in lines + [message[i:i+90] for i in range(0, len(message), 90)]:
        text.textLine(line)
    pdf.drawText(text); pdf.save(); return buffer.getvalue()


TEXT_DECODABLE_EXTENSIONS = {".txt", ".csv", ".json", ".html", ".htm", ".md", ".log", ".xml", ".rtf"}
SUPPORTED_UPLOAD_FORMATS_MESSAGE = "Supported formats: .txt, .csv, .json, .html, .md, .log, .xml, .eml, .pdf, .docx"


def extract_text_from_upload(filename: str, raw_bytes: bytes) -> str:
    """Extract plain text from an uploaded file of (almost) any common
    format, for use either as an analyzable message or as input to the
    File Translation converter. Raises ValueError with a clear message for
    formats that don't contain extractable text (images, old .doc, other
    unrecognised binaries, scanned/image-only PDFs, etc.)."""
    suffix = Path(filename).suffix.lower()

    if suffix in TEXT_DECODABLE_EXTENSIONS:
        return raw_bytes.decode("utf-8", errors="ignore")

    if suffix == ".eml":
        email_message = BytesParser(policy=policy.default).parse(BytesIO(raw_bytes))
        if email_message.is_multipart():
            body = email_message.get_body(preferencelist=("plain",))
            return body.get_content() if body else ""
        return email_message.get_content()

    if suffix == ".pdf":
        reader = PdfReader(BytesIO(raw_bytes))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if not text.strip():
            raise ValueError("This PDF has no extractable text (it may be a scanned/image-only PDF).")
        return text

    if suffix == ".docx":
        document = docx.Document(BytesIO(raw_bytes))
        text = "\n".join(paragraph.text for paragraph in document.paragraphs)
        if not text.strip():
            raise ValueError("This Word document appears to be empty.")
        return text

    if suffix == ".doc":
        raise ValueError("Old-format .doc files aren't supported - only modern .docx.")

    raise ValueError(f"Unsupported file format: {suffix or '(no extension)'}. {SUPPORTED_UPLOAD_FORMATS_MESSAGE}")


def build_eml_bytes(subject: str, body_text: str) -> bytes:
    """Wrap plain text into a minimal, valid .eml file (RFC822 email)."""
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = "converted@message-guard.local"
    message["To"] = "recipient@message-guard.local"
    message.set_content(body_text)
    return message.as_bytes()


@st.dialog("Unsupported File Format")
def unsupported_format_dialog(filename: str) -> None:
    """Shown when Analyze Message receives a file that isn't .txt/.eml."""
    suffix = Path(filename).suffix or "(no extension)"
    st.error(f"**{filename}** — the file type `{suffix}` isn't supported here.")
    st.write("Analyze Message only accepts **.txt** and **.eml** files directly.")
    st.write("Use **File Translation** in the sidebar to convert this file to .txt or .eml first, then upload the converted file here.")
    if st.button("Got it", type="primary", use_container_width=True):
        st.rerun()


def go_to(page_name: str) -> None:
    """Central helper: change page + rerun (avoids duplicated rerun logic)."""
    st.session_state.nav_page = page_name
    st.rerun()


def home() -> None:
    # Full-bleed home page: strip the block-container's padding/max-width
    # just for this render so the hero can fill the entire browser viewport
    # edge-to-edge instead of sitting inside a small centered card. The
    # sidebar itself is always "expanded" per set_page_config (Streamlit's
    # initial_sidebar_state only applies once at true first load and can't
    # be reliably re-toggled dynamically across reruns), so it's hidden here
    # via plain CSS instead — fully within our own control.
    st.html(
        """
        <style>
        .block-container {
            padding-top: 0 !important;
            padding-bottom: 0 !important;
            padding-left: 0 !important;
            padding-right: 0 !important;
            max-width: 100% !important;
        }
        section[data-testid="stSidebar"] {
            display: none;
        }
        </style>
        """
    )

    render_hero("EMAIL DETECTION", dark=True)


def analyze() -> None:
    page_header(
        "🔍", "Analyze Message",
        "root@messageguard:~$ paste a message or email to classify its risk",
        extra_style="<style>.block-container { max-width: 1280px !important; }</style>",
    )

    info = metrics()
    algorithm_names = info.get("candidate_names") or list(info.get("models", {}).keys()) or ["Support Vector Machine"]
    best_algorithm = info.get("best_model")
    default_index = algorithm_names.index(best_algorithm) if best_algorithm in algorithm_names else 0
    selected_algorithm = st.selectbox(
        "Detection algorithm",
        algorithm_names,
        index=default_index,
        help="Choose which trained algorithm analyzes the message below.",
    )

    sample = "URGENT! Verify your account now at https://secure-check.example or it will be suspended!!"

    uploaded_message = ""
    with st.expander("Or drag and drop a file to analyze (.txt / .eml — other formats can be converted first via File Translation)"):
        uploaded_file = st.file_uploader("Upload file", type=None, label_visibility="collapsed")

        if uploaded_file is not None:
            suffix = Path(uploaded_file.name).suffix.lower()
            if suffix not in (".txt", ".eml"):
                if st.session_state.get("_last_invalid_upload") != (uploaded_file.name, uploaded_file.size):
                    st.session_state._last_invalid_upload = (uploaded_file.name, uploaded_file.size)
                    unsupported_format_dialog(uploaded_file.name)
            else:
                try:
                    uploaded_message = extract_text_from_upload(uploaded_file.name, uploaded_file.read())
                except Exception as e:
                    st.error(f"Unable to read file: {e}")
                    return

    if "message_input" not in st.session_state:
        st.session_state.message_input = ""
    if uploaded_message and st.session_state.get("_last_upload_id") != (uploaded_file.name, uploaded_file.size):
        st.session_state.message_input = uploaded_message
        st.session_state._last_upload_id = (uploaded_file.name, uploaded_file.size)

    # Text area
    message = st.text_area(
        "Paste a text message or email",
        key="message_input",
        height=120,
        placeholder=sample
    )

    if st.button("Analyze Message", type="primary", use_container_width=True):
        try:
            with st.spinner("Checking language patterns and risk indicators..."):
                result = detector(selected_algorithm).analyze(message)

            category = result.get("category") or categorize_message(result["prediction"], result["indicators"])
            append_history(
                message,
                result["prediction"],
                result["confidence"],
                result["risk_score"],
                category
            )

            st.session_state.result = result
            st.session_state.message = message

        except (FileNotFoundError, ValueError) as error:
            st.error(str(error))
            return

    result = st.session_state.get("result")

    if not result:
        return

    risk_color = RISK_COLORS[result["prediction"]]
    icon = RISK_ICONS[result["prediction"]]

    row1 = st.columns([1.4, 1])

    # --- panel 1: verdict + gauge -------------------------------------------------
    category = result.get("category") or categorize_message(result["prediction"], result["indicators"])
    category_icon = CATEGORY_ICONS.get(category, "⚠️")
    with row1[0]:
        st.html(
            f"""
            <div class="mg-terminal-card cf-fade" style="border-color:{risk_color}55; box-shadow:0 0 26px {risk_color}22; height:230px;">
                <div class="mg-terminal-card-bar">
                    <span class="dot r"></span><span class="dot y"></span><span class="dot g"></span>
                    <span class="label">verdict.log</span>
                </div>
                <div class="mg-terminal-card-body" style="display:flex; align-items:center; gap:1rem;">
                    <div class="mg-gauge" style="--gauge-pct:{result['risk_score']}; --gauge-color:{risk_color}; --gauge-glow:{risk_color}33;">
                        <div class="mg-gauge-inner">
                            <div class="mg-gauge-value">{result['risk_score']}</div>
                            <div class="mg-gauge-label">/ 100</div>
                        </div>
                    </div>
                    <div>
                        <div style="color:{risk_color}; font-size:1.35rem; font-weight:800; letter-spacing:0.02em;">
                            {icon}&nbsp;{result['prediction'].upper()}
                        </div>
                        <div style="color:var(--mg-text-dim); font-size:0.78rem; margin-top:0.3rem;">
                            RISK: {result['risk_level']}<br>CONF: {result['confidence']:.1%}<br>ALGO: {result.get('algorithm', '—')}
                        </div>
                        <div style="margin-top:0.5rem;">
                            <span class="mg-badge">{category_icon}&nbsp;{category}</span>
                        </div>
                    </div>
                </div>
            </div>
            """
        )

    # --- panel 2: detected keywords / suspicious URLs -----------------------------
    with row1[1]:
        words = [
            word.upper()
            for values in result["indicators"]["keywords"].values()
            for word in values
        ]
        badges = "".join(f'<span class="mg-badge">{w}</span>' for w in words) or '<span class="mg-panel-title">None detected</span>'
        url_badges = "".join(f'<span class="mg-badge danger">{u}</span>' for u in result["indicators"]["urls"])
        st.html(
            f"""
            <div class="mg-terminal-card cf-fade" style="height:230px;">
                <div class="mg-terminal-card-bar">
                    <span class="dot r"></span><span class="dot y"></span><span class="dot g"></span>
                    <span class="label">indicators.log</span>
                </div>
                <div class="mg-terminal-card-body">
                    <div class="mg-panel-title">Keywords</div>
                    <div>{badges}</div>
                    {"<div class='mg-panel-title' style='margin-top:0.7rem;'>Suspicious URLs</div><div>" + url_badges + "</div>" if url_badges else ""}
                </div>
            </div>
            """
        )

    row2 = st.columns([2, 1])

    # --- panel 4: explanation ------------------------------------------------------
    with row2[0]:
        st.html(
            f"""
            <div class="mg-terminal-card cf-fade" style="height:120px;">
                <div class="mg-terminal-card-bar">
                    <span class="dot r"></span><span class="dot y"></span><span class="dot g"></span>
                    <span class="label">explanation.log</span>
                </div>
                <div class="mg-terminal-card-body" style="color:var(--mg-text); font-size:0.88rem;">
                    {result['explanation']}
                </div>
            </div>
            """
        )

    # --- panel 5: export -------------------------------------------------------------
    with row2[1]:
        st.download_button(
            "Download Result as PDF",
            result_pdf(st.session_state.message, result),
            "message-analysis.pdf",
            "application/pdf",
            use_container_width=True,
        )


def history_page() -> None:
    page_header(
        "🕘", "Prediction History", "root@messageguard:~$ cat prediction_history.csv",
        extra_style="<style>.block-container { max-width: 1280px !important; }</style>",
    )
    history = load_history()

    counts = history["Prediction"].str.lower().value_counts() if not history.empty else pd.Series(dtype=int)
    kpi = st.columns(4)
    with kpi[0]:
        stat_card(len(history), "Total Analyses", color="#00e5ff")
    with kpi[1]:
        stat_card(int(counts.get("low", 0)), "Low Risk", color=RISK_COLORS["low"])
    with kpi[2]:
        stat_card(int(counts.get("medium", 0)), "Medium Risk", color=RISK_COLORS["medium"])
    with kpi[3]:
        stat_card(int(counts.get("high", 0)), "High Risk", color=RISK_COLORS["high"])

    st.html("<div style='margin-top:1rem;'></div>")

    # Track each row's position in the underlying saved file (via a hidden
    # column) so deletion still targets the right rows even when a search
    # filter has changed which rows are currently displayed.
    history = history.reset_index(drop=True)
    history["_orig_idx"] = history.index

    search = st.text_input("Search messages or predictions")
    if search:
        history = history[history.astype(str).apply(lambda row: row.str.contains(search, case=False).any(), axis=1)]

    display_history = history.copy()
    display_history.insert(0, "Select", False)

    if st.session_state.get("_select_all_history"):
        display_history["Select"] = True
        st.session_state.pop("_select_all_history", None)
        st.session_state.pop("history_editor", None)  # force the editor to reinit with all rows checked

    log_label_col, select_all_col = st.columns([4, 1])
    with log_label_col:
        st.html('<div class="mg-panel-title" style="margin-top:0.4rem;">Prediction Log</div>')
    with select_all_col:
        if st.button("Select All", use_container_width=True, disabled=display_history.empty):
            st.session_state._select_all_history = True
            st.rerun()

    edited = st.data_editor(
        display_history,
        key="history_editor",
        hide_index=True,
        use_container_width=True,
        disabled=["Date", "Message", "Prediction", "Category", "Confidence", "Risk Score"],
        column_order=["Select", "Date", "Message", "Prediction", "Category", "Confidence", "Risk Score"],
        column_config={"Select": st.column_config.CheckboxColumn("", width="small")},
    )
    selected_rows = edited[edited["Select"]]

    export_col, delete_selected_col, delete_all_col = st.columns(3)
    with export_col:
        st.download_button(
            "Export history as CSV", history.drop(columns=["_orig_idx"]).to_csv(index=False).encode(),
            "prediction-history.csv", "text/csv", use_container_width=True,
        )
    with delete_selected_col:
        if st.button(
            f"Delete Selected ({len(selected_rows)})",
            use_container_width=True, disabled=selected_rows.empty,
        ):
            confirm_delete_dialog(selected_rows["_orig_idx"].tolist(), f"{len(selected_rows)} selected record(s)")
    with delete_all_col:
        if st.button("Delete All History", use_container_width=True, disabled=load_history().empty):
            confirm_delete_dialog(None, f"all {len(load_history())} record(s)")


@st.dialog("Confirm Deletion")
def confirm_delete_dialog(row_positions: list[int] | None, description: str) -> None:
    """Modal confirmation shown before any history deletion — row_positions=None means delete everything."""
    st.warning(f"This will permanently delete {description}. This action cannot be undone.")
    cancel_col, confirm_col = st.columns(2)
    with cancel_col:
        if st.button("Cancel", use_container_width=True):
            st.rerun()
    with confirm_col:
        if st.button("Yes, Delete", type="primary", use_container_width=True):
            if row_positions is None:
                clear_history()
            else:
                delete_history_rows(row_positions)
            st.session_state.pop("history_editor", None)
            st.rerun()


def file_translation() -> None:
    page_header(
        "🔄", "File Translation",
        "root@messageguard:~$ convert any file into .txt or .eml",
        extra_style="<style>.block-container { max-width: 1280px !important; }</style>",
    )

    st.write(
        "Analyze Message only accepts .txt and .eml files directly. Upload a file in any "
        "supported format below, pick a target format, and convert it — you'll get a "
        "download button for the converted file, which you can then upload to Analyze Message."
    )
    st.caption(SUPPORTED_UPLOAD_FORMATS_MESSAGE)

    uploaded_file = st.file_uploader("Upload a file to convert", type=None)
    target_format = st.radio("Convert to", [".txt", ".eml"], horizontal=True)

    if st.button("Convert", type="primary", use_container_width=True, disabled=uploaded_file is None):
        try:
            with st.spinner("Extracting text and converting..."):
                text = extract_text_from_upload(uploaded_file.name, uploaded_file.read())
            base_name = Path(uploaded_file.name).stem
            if target_format == ".txt":
                converted_bytes = text.encode("utf-8")
                out_name = f"{base_name}.txt"
                mime = "text/plain"
            else:
                converted_bytes = build_eml_bytes(subject=base_name, body_text=text)
                out_name = f"{base_name}.eml"
                mime = "message/rfc822"
            st.session_state.converted_file = {"bytes": converted_bytes, "name": out_name, "mime": mime}
            st.success(f"Converted successfully — {out_name} is ready to download below.")
        except Exception as e:
            st.error(f"Conversion failed: {e}")
            st.session_state.pop("converted_file", None)

    converted = st.session_state.get("converted_file")
    if converted:
        st.download_button(
            f"Download {converted['name']}",
            converted["bytes"],
            converted["name"],
            converted["mime"],
            use_container_width=True,
        )

def algorithm_comparison() -> None:
    page_header(
        "📈", "Algorithm Comparison",
        "root@messageguard:~$ cat algorithm_accuracy_report.log",
        extra_style="<style>.block-container { max-width: 1280px !important; }</style>",
    )

    info = metrics()
    models = info.get("models", {})
    if not models:
        st.info("No trained models found yet.")
        return

    best_algorithm = info.get("best_model")

    comparison_df = pd.DataFrame([
        {
            "Algorithm": name,
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1 Score": m["f1"],
            "CV Accuracy": m.get("cv_accuracy"),
            "CV F1": m.get("cv_f1"),
        }
        for name, m in models.items()
    ]).sort_values("Accuracy", ascending=False).reset_index(drop=True)

    st.html('<div class="mg-panel-title">Accuracy Comparison — All Algorithms</div>')

    percent_cols = ["Accuracy", "Precision", "Recall", "F1 Score", "CV Accuracy", "CV F1"]
    st.dataframe(
        comparison_df.style.format({col: "{:.1%}" for col in percent_cols}).apply(
            lambda row: [
                "background-color: rgba(246,130,31,0.18); color:#ffb066; font-weight:700;"
                if row["Algorithm"] == best_algorithm else ""
                for _ in row
            ],
            axis=1,
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(f"🏆 Best performing algorithm (by cross-validated F1): **{best_algorithm}**")

    st.html("<div style='margin-top:1.2rem;'></div>")
    fig = px.bar(
        comparison_df, x="Algorithm", y="Accuracy", color="Algorithm",
        text=comparison_df["Accuracy"].apply(lambda v: f"{v:.1%}"),
    )
    fig = style_fig(fig)
    fig.update_layout(showlegend=False, yaxis=dict(tickformat=".0%", range=[0, 1]))
    st.html('<div class="mg-panel-title">Accuracy Chart</div>')
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


apply_theme()

pages = {
    "Home": home,
    "Analyze Message": analyze,
    "Dashboard": dashboard,
    "History": history_page,
    "File Translation": file_translation,
    "Algorithm Comparison": algorithm_comparison,
}

NAV_ICONS = {
    "Analyze Message": "🔍",
    "Dashboard": "📊",
    "History": "🕘",
    "File Translation": "🔄",
    "Algorithm Comparison": "📈",   # 新增
}
NAV_ITEMS = ["Analyze Message", "Dashboard", "History", "File Translation", "Algorithm Comparison"]

if st.session_state.started:
    ensure_trained() 

    if not st.session_state.sidebar_visible:
        # Force-hide Streamlit's native sidebar via our own CSS (not relying
        # on Streamlit's collapse mechanism, which only reliably applies once
        # at first load) and give a normal, always-reachable button in the
        # main content area to bring it back.
        st.html("<style>section[data-testid='stSidebar']{display:none !important;}</style>")
        if st.button("☰  Show Sidebar", type="primary"):
            st.session_state.sidebar_visible = True
            st.rerun()
    else:
        st.sidebar.html(
            """
            <div class="mg-sidebar-titlebar">
                <span class="dot r"></span><span class="dot y"></span><span class="dot g"></span>
                <span class="brand">MESSAGE_GUARD</span>
            </div>
            """
        )

        if st.sidebar.button("✕ Hide Sidebar", use_container_width=True):
            st.session_state.sidebar_visible = False
            st.rerun()

        if st.sidebar.button("← Back to Home", use_container_width=True):
            st.session_state.started = False
            go_to("Home")

        st.sidebar.html('<div class="mg-sidebar-label">&gt; Navigate</div>')
        if st.session_state.nav_page not in pages:
            st.session_state.nav_page = NAV_ITEMS[0]

        for item in NAV_ITEMS:
            label = f"{NAV_ICONS.get(item, '')}  {item}"
            is_active = item == st.session_state.nav_page
            if st.sidebar.button(
                label, key=f"nav_btn_{item}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
            ) and not is_active:
                st.session_state.nav_page = item
                st.rerun()

        best_model = metrics().get("best_model", "Not trained yet")
        st.sidebar.html('<div class="mg-sidebar-spacer"></div>')
        st.sidebar.html(
            f"""
            <div class="mg-sidebar-footer">
                <span class="pulse"></span>
                <div class="meta">
                    <div class="model">{best_model}</div>
                    <div class="status">MODEL ONLINE</div>
                </div>
            </div>
            """
        )
# else: nothing rendered in the sidebar on Home — it stays hidden

pages[st.session_state.nav_page]()