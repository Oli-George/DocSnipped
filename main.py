# pyrefly: ignore [missing-import]
import streamlit as st
import re
from summarizer.summarize import process_and_summarize_text, process_and_summarize_doc_with_progress
from summarizer.qa import answer_question
from summarizer.models import load_sentiment_model

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocSnipped",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Session state initialisation ───────────────────────────────────────────────
def _init_state():
    defaults = {
        "summary": "",
        "source_text": "",
        "current_choice": None,
        "max_words": 150,
        "sentiment_label": None,
        "sentiment_score": None,
        "theme": "dark",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_state()

# ── Theme tokens ───────────────────────────────────────────────────────────────
_DARK = {
    "app_bg":          "#0F1117",
    "secondary_bg":    "#1A1D27",
    "border":          "#2d2f45",
    "hdr_from":        "#1e1b4b",
    "hdr_mid":         "#312e81",
    "hdr_border":      "#4338ca55",
    "hdr_h1":          "#e0e7ff",
    "hdr_p":           "#a5b4fc",
    "sent_bg":         "#1A1D27",
    "sent_border":     "#2d2f45",
    "sent_label":      "#a5b4fc",
    "sent_score":      "#6b7280",
    "toggle_bg":       "#ffffff",
    "toggle_border":   "#e2e8f0",
    "toggle_hover_bg": "#f3f4f6",
    # Buttons (secondary)
    "btn_bg":          "#1e1b4b",
    "btn_border":      "#4338ca",
    "btn_text":        "#c7d2fe",
    "btn_hover_bg":    "#312e81",
    "btn_hover_text":  "#e0e7ff",
    # SVG icon shown when in dark mode = sun (to switch to light)
    # Stroke colour encoded: #f59e0b (amber) → %23f59e0b
    "toggle_svg_uri":  "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%23f59e0b' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='4'/%3E%3Cline x1='12' y1='2' x2='12' y2='6'/%3E%3Cline x1='12' y1='18' x2='12' y2='22'/%3E%3Cline x1='4.93' y1='4.93' x2='7.76' y2='7.76'/%3E%3Cline x1='16.24' y1='16.24' x2='19.07' y2='19.07'/%3E%3Cline x1='2' y1='12' x2='6' y2='12'/%3E%3Cline x1='18' y1='12' x2='22' y2='12'/%3E%3Cline x1='4.93' y1='19.07' x2='7.76' y2='16.24'/%3E%3Cline x1='16.24' y1='7.76' x2='19.07' y2='4.93'/%3E%3C/svg%3E",
    "toggle_hover_svg_uri": "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%23f59e0b' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='4'/%3E%3Cline x1='12' y1='2' x2='12' y2='6'/%3E%3Cline x1='12' y1='18' x2='12' y2='22'/%3E%3Cline x1='4.93' y1='4.93' x2='7.76' y2='7.76'/%3E%3Cline x1='16.24' y1='16.24' x2='19.07' y2='19.07'/%3E%3Cline x1='2' y1='12' x2='6' y2='12'/%3E%3Cline x1='18' y1='12' x2='22' y2='12'/%3E%3Cline x1='4.93' y1='19.07' x2='7.76' y2='16.24'/%3E%3Cline x1='16.24' y1='7.76' x2='19.07' y2='4.93'/%3E%3C/svg%3E",
    "toggle_title":    "Switch to light mode",
    "text":            "#E8E8F0",
    "text_muted":      "#9ca3af",
    "stApp_bg":        "#0F1117",
    "widget_bg":       "#1A1D27",
    "widget_border":   "#2d2f45",
}

_LIGHT = {
    "app_bg":          "#f8f9fc",
    "secondary_bg":    "#ffffff",
    "border":          "#e2e8f0",
    "hdr_from":        "#ede9fe",
    "hdr_mid":         "#ddd6fe",
    "hdr_border":      "#8b5cf633",
    "hdr_h1":          "#1e1b4b",
    "hdr_p":           "#4338ca",
    "sent_bg":         "#ffffff",
    "sent_border":     "#e2e8f0",
    "sent_label":      "#4338ca",
    "sent_score":      "#6b7280",
    "toggle_bg":       "#ffffff",
    "toggle_border":   "#e2e8f0",
    "toggle_hover_bg": "#f3f4f6",
    # Buttons (secondary)
    "btn_bg":          "#ede9fe",
    "btn_border":      "#8b5cf6",
    "btn_text":        "#3730a3",
    "btn_hover_bg":    "#7C6FCD",
    "btn_hover_text":  "#ffffff",
    # SVG icon shown when in light mode = moon (to switch to dark)
    # Stroke colour encoded: #7C6FCD → %237C6FCD
    "toggle_svg_uri":  "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%237C6FCD' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z'/%3E%3C/svg%3E",
    "toggle_hover_svg_uri": "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%237C6FCD' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z'/%3E%3C/svg%3E",
    "toggle_title":    "Switch to dark mode",
    "text":            "#111827",
    "text_muted":      "#6b7280",
    "stApp_bg":        "#f8f9fc",
    "widget_bg":       "#ffffff",
    "widget_border":   "#e2e8f0",
}

# ── Theme CSS injection ────────────────────────────────────────────────────────
def _inject_theme(t: dict):
    st.markdown(f"""
<style>
/* ── App shell ── */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {{
    background-color: {t['stApp_bg']} !important;
}}
[data-testid="stHeader"] {{
    background-color: {t['stApp_bg']} !important;
}}
[data-testid="stSidebar"] {{
    background-color: {t['widget_bg']} !important;
}}

/* ── All widget labels & body text ── */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label,
label,
.stSelectbox label,
.stTextInput label,
.stTextArea label,
.stSlider label,
.stFileUploader label,
.stMarkdown p,
[data-testid="stCaptionContainer"] p,
[data-testid="stText"] p,
p {{
    color: {t['text']} !important;
}}
[data-testid="stCaptionContainer"] p {{
    color: {t['text_muted']} !important;
    opacity: 1 !important;
}}

/* ── Header card ── */
.ds-header {{
    background: linear-gradient(135deg, {t['hdr_from']} 0%, {t['hdr_mid']} 50%, {t['hdr_from']} 100%);
    border: 1px solid {t['hdr_border']};
    border-radius: 16px;
    padding: 2rem 2.5rem 1.75rem;
    margin-bottom: 2rem;
    text-align: center;
}}
.ds-header h1 {{
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: {t['hdr_h1']};
    margin: 0 0 0.3rem;
}}
.ds-header p {{
    color: {t['hdr_p']};
    font-size: 1.05rem;
    margin: 0;
}}

/* ── Primary buttons ── */
[data-testid="stBaseButton-primary"] {{
    background-color: #7C6FCD !important;
    color: #ffffff !important;
    border: none !important;
    font-weight: 600 !important;
}}
[data-testid="stBaseButton-primary"]:hover {{
    background-color: #6d60c4 !important;
    color: #ffffff !important;
}}

/* ── Secondary buttons ── */
[data-testid="stBaseButton-secondary"] {{
    background-color: {t['btn_bg']} !important;
    border: 1px solid {t['btn_border']} !important;
    font-weight: 500 !important;
}}
/* Target both the button element and the inner <p> Streamlit renders the label into */
[data-testid="stBaseButton-secondary"] p {{
    color: {t['btn_text']} !important;
}}
[data-testid="stBaseButton-secondary"]:hover {{
    background-color: {t['btn_hover_bg']} !important;
    border-color: {t['btn_hover_bg']} !important;
}}
[data-testid="stBaseButton-secondary"]:hover p {{
    color: {t['btn_hover_text']} !important;
}}

/* ── Theme toggle button ──────────────────────────────────────────────────────
   Targeted via #ds-toggle-col marker injected inside the toggle column.
   Scoped tightly so it ONLY matches the button in the same column — no
   sibling combinator that could leak to other buttons on the page.
── */
[data-testid="column"]:has(#ds-toggle-col) button,
[data-testid="stColumn"]:has(#ds-toggle-col) button,
[data-testid="column"]:has(#ds-toggle-col) [data-testid="stBaseButton-secondary"],
[data-testid="stColumn"]:has(#ds-toggle-col) [data-testid="stBaseButton-secondary"] {{
    width:  38px !important;
    height: 38px !important;
    min-height: 0 !important;
    padding: 0 !important;
    background-color: {t['toggle_bg']} !important;
    border: 1px solid {t['toggle_border']} !important;
    border-radius: 8px !important;
    background-image: url("{t['toggle_svg_uri']}") !important;
    background-repeat: no-repeat !important;
    background-position: center !important;
    background-size: 18px 18px !important;
    color: transparent !important;
    transition: all 0.2s ease !important;
    box-shadow: none !important;
}}
[data-testid="column"]:has(#ds-toggle-col) button p,
[data-testid="stColumn"]:has(#ds-toggle-col) button p {{
    color: transparent !important;
    display: none !important;
}}
[data-testid="column"]:has(#ds-toggle-col) button:hover,
[data-testid="stColumn"]:has(#ds-toggle-col) button:hover,
[data-testid="column"]:has(#ds-toggle-col) [data-testid="stBaseButton-secondary"]:hover,
[data-testid="stColumn"]:has(#ds-toggle-col) [data-testid="stBaseButton-secondary"]:hover {{
    background-color: {t['toggle_hover_bg']} !important;
    border-color: {t['toggle_hover_bg']} !important;
    background-image: url("{t['toggle_hover_svg_uri']}") !important;
    background-repeat: no-repeat !important;
    background-position: center !important;
    background-size: 18px 18px !important;
}}
/* Pull the toggle row up and flush-right */
[data-testid="stHorizontalBlock"]:has(#ds-toggle-col) {{
    margin-bottom: 0.5rem !important;
    justify-content: flex-end;
}}
[data-testid="stHorizontalBlock"]:has(#ds-toggle-col) [data-testid="column"]:first-child,
[data-testid="stHorizontalBlock"]:has(#ds-toggle-col) [data-testid="stColumn"]:first-child {{
    visibility: hidden !important;
    height: 0 !important;
    min-height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
}}

/* ── Selectbox (override dark-mode base for light theme) ── */
[data-testid="stSelectbox"] [data-baseweb="select"] > div:first-child {{
    background-color: {t['widget_bg']} !important;
    border-color: {t['border']} !important;
}}
[data-testid="stSelectbox"] [data-baseweb="select"] span,
[data-testid="stSelectbox"] [data-baseweb="select"] div {{
    color: {t['text']} !important;
}}
/* Dropdown list */
[data-baseweb="popover"] [data-baseweb="menu"] {{
    background-color: {t['widget_bg']} !important;
}}
[data-baseweb="popover"] [role="option"] {{
    background-color: {t['widget_bg']} !important;
    color: {t['text']} !important;
}}
[data-baseweb="popover"] [role="option"]:hover {{
    background-color: {t['btn_bg']} !important;
}}

/* ── Sentiment card ── */
.ds-sentiment {{
    display: flex;
    align-items: center;
    gap: 1rem;
    background: {t['sent_bg']};
    border: 1px solid {t['sent_border']};
    border-radius: 12px;
    padding: 0.9rem 1.3rem;
    margin-top: 0.75rem;
    width: fit-content;
}}
.ds-sentiment-label {{
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {t['sent_label']};
    margin-bottom: 0.15rem;
}}
.ds-sentiment-value {{
    font-size: 1.6rem;
    font-weight: 800;
    line-height: 1;
}}
.ds-sentiment-score {{
    font-size: 0.85rem;
    color: {t['sent_score']};
    margin-top: 0.2rem;
}}
.positive {{ color: #16a34a; }}
.negative {{ color: #dc2626; }}
.neutral  {{ color: #6b7280; }}
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
def capitalize_sentences(text):
    if not text:
        return text
    return re.sub(r'(^|[.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)

def TextInference(snippet, max_words):
    return process_and_summarize_text(snippet, max_words=max_words)

def DocInference(document, max_words):
    summary, text = process_and_summarize_doc_with_progress(document, max_words=max_words)
    return summary, text

def compute_sentiment(summary_text: str):
    """Run sentiment only once per summary and cache in session state."""
    pipeline = load_sentiment_model()
    result = pipeline(summary_text[:512])[0]
    st.session_state["sentiment_label"] = result["label"].upper()
    st.session_state["sentiment_score"] = result["score"]

def render_sentiment():
    """Render the sentiment result stored in session state."""
    label = st.session_state["sentiment_label"]
    score = st.session_state["sentiment_score"]
    if label is None:
        return
    css_class = {"POSITIVE": "positive", "NEGATIVE": "negative"}.get(label, "neutral")
    st.markdown(f"""
    <div class="ds-sentiment">
        <div>
            <div class="ds-sentiment-label">Sentiment</div>
            <div class="ds-sentiment-value {css_class}">{label}</div>
            <div class="ds-sentiment-score">Confidence: {score:.0%}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Main app ───────────────────────────────────────────────────────────────────
def main():
    is_dark = st.session_state["theme"] == "dark"
    tokens  = _DARK if is_dark else _LIGHT

    # Inject theme CSS first
    _inject_theme(tokens)

    col_spacer, col_toggle = st.columns([10, 1])
    with col_toggle:
        # Marker used by CSS :has(#ds-toggle-col) to reliably target this button
        st.markdown('<div id="ds-toggle-col"></div>', unsafe_allow_html=True)
        if st.button("\u00a0", key="theme_toggle", help=tokens["toggle_title"]):
            st.session_state["theme"] = "light" if is_dark else "dark"
            st.rerun()

    # ── Header card ───────────────────────────────────────────────────────────
    st.markdown("""
    <div class="ds-header">
        <h1>DocSnipped</h1>
        <p>Paste text or upload a document — Get an instant AI-powered summary.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Input mode selector ───────────────────────────────────────────────────
    options = ["Select an Option", "Paste Text", "Upload Document"]
    choice = st.selectbox("How would you like to provide your content?", options)
    choice_index = options.index(choice)

    if st.session_state["current_choice"] != choice_index:
        st.session_state["current_choice"] = choice_index
        st.session_state["summary"] = ""
        st.session_state["source_text"] = ""
        st.session_state["sentiment_label"] = None
        st.session_state["sentiment_score"] = None

    # ── Paste Text path ───────────────────────────────────────────────────────
    if choice_index == 1:
        st.caption("Paste your text below and click the button to generate a summary.")
        text_snippet = st.text_area("Paste your text snippet here:", height=200)

        st.session_state["max_words"] = st.slider(
            "Summary length (words)",
            min_value=10, max_value=500,
            value=st.session_state["max_words"], step=10,
            help="Approximate word count for the generated summary.",
            key="slider_text",
        )

        if st.button("Generate Summary", type="primary", use_container_width=True):
            if text_snippet:
                with st.spinner("Generating summary..."):
                    text_summary = TextInference(text_snippet, st.session_state["max_words"])
                    st.session_state["summary"] = text_summary
                    st.session_state["source_text"] = text_snippet
                    st.session_state["sentiment_label"] = None
                    st.session_state["sentiment_score"] = None
            else:
                st.warning("Please paste some text first.")

    # ── Upload Document path ──────────────────────────────────────────────────
    elif choice_index == 2:
        try:
            doc = st.file_uploader(
                "Upload your document (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"],
            )
        except Exception as e:
            st.error(f"Error uploading document: {e}")
            st.caption("Please try uploading again.")
            doc = None

        if doc is not None:
            st.success("Document uploaded successfully! Click below to generate a summary.")

            st.session_state["max_words"] = st.slider(
                "Summary length (words)",
                min_value=10, max_value=500,
                value=st.session_state["max_words"], step=10,
                help="Approximate word count for the generated summary.",
                key="slider_doc",
            )

            if st.button("Generate Summary", type="primary", use_container_width=True):
                doc_summary, doc_text = DocInference(doc, st.session_state["max_words"])
                st.session_state["summary"] = doc_summary
                st.session_state["source_text"] = doc_text
                st.session_state["sentiment_label"] = None
                st.session_state["sentiment_score"] = None

    # ── Placeholder ───────────────────────────────────────────────────────────
    else:
        st.info("Select an option above to get started.")

    # ── Results section ───────────────────────────────────────────────────────
    if st.session_state["summary"]:
        st.divider()
        
        # Conditionally render copy widget at the top-right of the summary section for Paste Text mode
        if choice_index == 1:
            col_hdr, col_copy = st.columns([5, 1])
            with col_hdr:
                st.subheader("Summary")
            with col_copy:
                # pyrefly: ignore [missing-import]
                import streamlit.components.v1 as components
                copy_text_escaped = st.session_state["summary"].replace("'", "\\'").replace("\n", "\\n")
                icon_color = "#7C6FCD" if is_dark else "#4338ca"
                bg_color = "#1A1D27" if is_dark else "#ffffff"
                border_color = "#2d2f45" if is_dark else "#e2e8f0"
                hover_bg = "#2d2f45" if is_dark else "#f3f4f6"
                
                html_code = f"""
                <div style="display: flex; justify-content: flex-end; align-items: center; height: 38px;">
                    <button id="copy-btn" onclick="copyToClipboard()" style="
                        background-color: {bg_color};
                        border: 1px solid {border_color};
                        border-radius: 6px;
                        width: 100px;
                        height: 32px;
                        cursor: pointer;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        gap: 6px;
                        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
                        font-size: 11.5px;
                        font-weight: 600;
                        color: {icon_color};
                        transition: all 0.2s ease;
                        padding: 0 8px;
                        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
                    ">
                        <svg id="copy-icon" xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                        </svg>
                        <span id="btn-text">Copy</span>
                    </button>
                </div>
                <script>
                    function copyToClipboard() {{
                        const text = `{copy_text_escaped}`;
                        navigator.clipboard.writeText(text).then(() => {{
                            const btn = document.getElementById('copy-btn');
                            const icon = document.getElementById('copy-icon');
                            const txt = document.getElementById('btn-text');
                            
                            btn.style.borderColor = '#16a34a';
                            btn.style.color = '#16a34a';
                            txt.textContent = 'Copied!';
                            icon.innerHTML = '<polyline points="20 6 9 17 4 12"></polyline>';
                            icon.setAttribute('stroke', '#16a34a');
                            
                            setTimeout(() => {{
                                btn.style.borderColor = '{border_color}';
                                btn.style.color = '{icon_color}';
                                txt.textContent = 'Copy';
                                icon.innerHTML = '<rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>';
                                icon.setAttribute('stroke', '{icon_color}');
                            }}, 2000);
                        }}).catch(err => {{
                            console.error('Failed to copy: ', err);
                        }});
                    }}
                    const btn = document.getElementById('copy-btn');
                    btn.addEventListener('mouseover', () => {{
                        if (btn.style.borderColor !== 'rgb(22, 163, 74)') {{
                            btn.style.backgroundColor = '{hover_bg}';
                        }}
                    }});
                    btn.addEventListener('mouseout', () => {{
                        btn.style.backgroundColor = '{bg_color}';
                    }});
                </script>
                """
                components.html(html_code, height=38, scrolling=False)
        else:
            st.subheader("Summary")
            
        st.write(st.session_state["summary"])

        # Show optional "Download Document" (PDF, DOCX, TXT options) below the summary for Upload Document mode
        if choice_index == 2:
            st.write("")  # spacing
            st.markdown("<p style='font-size:0.9rem; font-weight:600; margin-bottom: 0.5rem;'>Download Document Summary</p>", unsafe_allow_html=True)
            
            from summarizer.export import generate_txt_report, generate_docx_report, generate_pdf_report
            
            summary_txt = st.session_state["summary"]
            sent_lbl = st.session_state["sentiment_label"] or "NEUTRAL"
            sent_scr = st.session_state["sentiment_score"] or 0.0
            
            col_pdf, col_docx, col_txt = st.columns(3)
            
            with col_pdf:
                pdf_data = generate_pdf_report(summary_txt, sent_lbl, sent_scr, "Uploaded Document")
                st.download_button(
                    label="📄 Download PDF",
                    data=pdf_data,
                    file_name="doc_summary.pdf",
                    mime="application/pdf",
                    key="dl_pdf",
                    use_container_width=True
                )
                
            with col_docx:
                docx_data = generate_docx_report(summary_txt, sent_lbl, sent_scr, "Uploaded Document")
                st.download_button(
                    label="📝 Download DOCX",
                    data=docx_data,
                    file_name="doc_summary.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key="dl_docx",
                    use_container_width=True
                )
                
            with col_txt:
                txt_data = generate_txt_report(summary_txt, sent_lbl, sent_scr, "Uploaded Document")
                st.download_button(
                    label="📄 Download TXT",
                    data=txt_data,
                    file_name="doc_summary.txt",
                    mime="text/plain",
                    key="dl_txt",
                    use_container_width=True
                )

        if st.session_state["sentiment_label"] is None:
            with st.spinner("Analyzing sentiment..."):
                compute_sentiment(st.session_state["summary"])

        render_sentiment()

        st.divider()
        st.subheader("Ask a Follow-up Question")
        st.caption("Ask anything about the text or document you provided.")

        question = st.text_input("Your question:")
        if st.button("Get Answer", use_container_width=True):
            if question:
                with st.spinner("Searching for the answer..."):
                    answer = answer_question(question, st.session_state["source_text"])
                    if answer:
                        answer = capitalize_sentences(answer)
                st.markdown("**Answer:**")
                st.write(answer)
            else:
                st.warning("Please enter a question.")

if __name__ == "__main__":
    main()