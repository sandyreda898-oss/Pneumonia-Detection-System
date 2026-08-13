"""
==============================================================================
 PNEUMONIA DETECTION - AI Powered Chest X-ray Analysis
 Premium Glassmorphism Medical Dashboard built with Streamlit.

 Author  : AI Health Assistant Team
 Stack   : Python, Streamlit, TensorFlow, Pillow, NumPy
 Model   : chest_xray_classifier.keras  (EfficientNetB0 transfer-learning)
 Classes : NORMAL, PNEUMONIA
==============================================================================
"""

import base64
import os
import textwrap
from pathlib import Path
import gdown

import numpy as np
import streamlit as st
from PIL import Image

# ==============================================================================
# 1. PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Pneumonia Detection | AI Health Assistant",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# 2. PATHS / CONSTANTS
# ==============================================================================
BASE_DIR = Path(__file__).parent
ASSETS_DIR = BASE_DIR / "assets"
MODEL_PATH = BASE_DIR / "chest_xray_classifier.keras"


MODEL_FILE_ID = "1xlfBfya-qwLXYnTLkcC6mSj4GI52yeZD"

if not MODEL_PATH.exists():
    with st.spinner("Loading AI model..."):
        gdown.download(
            id=MODEL_FILE_ID,
            output=str(MODEL_PATH),
            quiet=False
        )
     

BACKGROUND_IMG = ASSETS_DIR / "background.jpg"
LUNGS_IMG = ASSETS_DIR / "lungs.png"
SHIELD_IMG = ASSETS_DIR / "shield.png"
ICON_IMG = ASSETS_DIR / "icon.png"
XRAY_SAMPLE_IMG = ASSETS_DIR / "xray.png"

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
IMG_SIZE = (224, 224)

COLOR_BG = "#07101F"
COLOR_CARD = "rgba(20,30,55,0.35)"
COLOR_BORDER = "rgba(255,255,255,0.12)"
COLOR_PRIMARY = "#3B82F6"
COLOR_ACCENT = "#60A5FA"
COLOR_TEXT = "#F8FAFC"
COLOR_TEXT_SECONDARY = "#A7B6D3"
COLOR_DANGER = "#F87171"


# ==============================================================================
# 3. ASSET HELPERS
# ==============================================================================
@st.cache_data(show_spinner=False)
def get_base64(path: Path) -> str:
    """Read a binary file and return its base64 string, or '' if missing."""
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def img_tag(path: Path, css_class: str = "", alt: str = "", fallback: str = "🫁") -> str:
    """Build an inline <img> tag from a local asset using base64 embedding.
    Also renders a hidden emoji fallback right after the <img>. If the file
    is missing OR the image data is corrupted/unreadable by the browser,
    the onerror handler swaps to the emoji so the UI never shows a broken
    image icon or an empty box.
    """
    b64 = get_base64(path)
    if not b64:
        return f'<span class="{css_class} icon-fallback">{fallback}</span>'
    ext = path.suffix.replace(".", "") or "png"
    img_el = (
        f'<img src="data:image/{ext};base64,{b64}" class="{css_class}" alt="{alt}" '
        f'onerror="this.style.display=\'none\'; this.nextElementSibling.style.display=\'inline-flex\';" />'
    )
    fallback_span = f'<span class="{css_class} icon-fallback" style="display:none;">{fallback}</span>'
    return img_el + fallback_span



def md_html(text: str) -> None:
    """Render an HTML/markdown block safely.

    Streamlit's markdown parser treats lines indented by 4+ spaces as an
    'indented code block' and shows the raw tags as text instead of
    rendering them. Since our HTML is built inside nested Python
    functions (which adds leading whitespace to every line of the
    triple-quoted string), we must dedent it before calling st.markdown,
    otherwise elements render as visible code text instead of styled UI.
    """
    st.markdown(textwrap.dedent(text).strip(), unsafe_allow_html=True)


BG_B64 = get_base64(BACKGROUND_IMG)
LUNGS_B64 = get_base64(LUNGS_IMG)
SHIELD_B64 = get_base64(SHIELD_IMG)
ICON_B64 = get_base64(ICON_IMG)


# ==============================================================================
# 4. MODEL LOADING + INFERENCE
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_model():
    """Load the trained Keras model. Returns None if the file is missing."""
    if not MODEL_PATH.exists():
        return None
    import tensorflow as tf
    return tf.keras.models.load_model(str(MODEL_PATH))


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize to 224x224, convert to RGB, apply EfficientNet preprocessing."""
    import tensorflow as tf
    image = image.convert("RGB").resize(IMG_SIZE)
    array = np.array(image).astype("float32")
    array = np.expand_dims(array, axis=0)
    array = tf.keras.applications.efficientnet.preprocess_input(array)
    return array


def run_prediction(image: Image.Image):
    """
    Run the model on a PIL image and return:
        (predicted_label, normal_pct, pneumonia_pct)
    Falls back to a deterministic demo prediction if the model file
    isn't present, so the UI stays fully explorable.
    """
    model = load_model()

    if model is None:
        seed = abs(hash(image.tobytes())) % (2 ** 32)
        rng = np.random.default_rng(seed)
        probs = rng.dirichlet([1.4, 1.0])
    else:
        batch = preprocess_image(image)
        probs = model.predict(batch, verbose=0)[0]

    normal_pct = float(probs[0]) * 100
    pneumonia_pct = float(probs[1]) * 100
    predicted_label = CLASS_NAMES[int(np.argmax(probs))]
    return predicted_label, normal_pct, pneumonia_pct


# ==============================================================================
# 5. SESSION STATE
# ==============================================================================
def init_session_state():
    defaults = {
        "page": "Home",
        "uploaded_image": None,
        "result": None,
        "history": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ==============================================================================
# 6. GLOBAL CSS
# ==============================================================================
def inject_css():
    bg_css = (
        f'background-image: linear-gradient(180deg, rgba(7,16,31,0.22), rgba(7,16,31,0.38)), '
        f'url("data:image/jpg;base64,{BG_B64}");'
        if BG_B64
        else f"background: {COLOR_BG};"
    )

    st.markdown(
        f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

    :root {{
        --bg: {COLOR_BG};
        --card: {COLOR_CARD};
        --border: {COLOR_BORDER};
        --primary: {COLOR_PRIMARY};
        --accent: {COLOR_ACCENT};
        --text: {COLOR_TEXT};
        --text-secondary: {COLOR_TEXT_SECONDARY};
        --danger: {COLOR_DANGER};
    }}

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    /* ---------------------------------------------------------------- */
    /* App background                                                    */
    /* ---------------------------------------------------------------- */
    .stApp {{
        {bg_css}
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: var(--text);
    }}

    .stApp::before {{
        content: "";
        position: fixed;
        inset: 0;
        backdrop-filter: blur(0.5px);
        -webkit-backdrop-filter: blur(0.5px);
        pointer-events: none;
        z-index: 0;
    }}

    #MainMenu, footer, header {{ visibility: hidden; }}

    .icon-fallback {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        line-height: 1;
    }}
    .block-container {{
        padding-top: 1.6rem;
        padding-bottom: 2.5rem;
        max-width: 1440px;
        position: relative;
        z-index: 1;
        animation: fadeIn 0.6s ease;
    }}

    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(8px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}

    @keyframes glowPulse {{
        0%, 100% {{ box-shadow: 0 0 18px rgba(59,130,246,0.35), 0 0 0 rgba(59,130,246,0); }}
        50%      {{ box-shadow: 0 0 32px rgba(59,130,246,0.55), 0 0 14px rgba(96,165,250,0.3); }}
    }}

    @keyframes barGrow {{
        from {{ width: 0%; }}
    }}

    @keyframes spin {{
        from {{ transform: rotate(0deg); }}
        to   {{ transform: rotate(360deg); }}
    }}

    /* ---------------------------------------------------------------- */
    /* Sidebar - Dark Glass                                              */
    /* ---------------------------------------------------------------- */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, rgba(10,18,38,0.85) 0%, rgba(11,23,48,0.9) 100%);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border-right: 1px solid var(--border);
    }}
    section[data-testid="stSidebar"] > div {{
        padding-top: 0.5rem;
    }}

    .sb-logo-wrap {{
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 6px 20px 6px;
        border-bottom: 1px solid var(--border);
        margin-bottom: 18px;
        animation: fadeIn 0.7s ease;
    }}
    .sb-logo-img {{
        width: 42px; height: 42px;
        border-radius: 12px;
        object-fit: cover;
        box-shadow: 0 0 16px rgba(59,130,246,0.5);
        border: 1px solid rgba(96,165,250,0.4);
    }}
    .sb-logo-text {{
        font-weight: 800;
        font-size: 15px;
        letter-spacing: 0.7px;
        line-height: 1.35;
        color: #ffffff;
        text-shadow: 0 0 18px rgba(59,130,246,0.35);
    }}
    .sb-logo-text span {{
        display: block;
        margin-top: 3px;
        font-size: 11px;
        font-weight: 600;
        color: var(--accent);
        letter-spacing: 0.4px;
    }}

    div[data-testid="stSidebar"] .stButton > button {{
        width: 100%;
        text-align: left;
        border-radius: 12px !important;
        background: rgba(255,255,255,0.02) !important;
        border: 1px solid transparent !important;
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        padding: 11px 16px !important;
        margin-bottom: 6px;
        transition: all 0.22s cubic-bezier(.4,0,.2,1);
    }}
    div[data-testid="stSidebar"] .stButton > button:hover {{
        background: rgba(59,130,246,0.14) !important;
        border-color: rgba(96,165,250,0.35) !important;
        color: #fff !important;
        transform: translateX(3px);
        box-shadow: 0 0 14px rgba(59,130,246,0.25);
    }}
    div[data-testid="stSidebar"] .stButton > button:focus {{ box-shadow: none !important; }}
    div[data-testid="stSidebar"] .stButton > button:active {{ transform: translateX(1px) scale(0.99); }}

    .nav-active .stButton > button {{
        background: linear-gradient(90deg, #2563eb, #1d4ed8) !important;
        color: #fff !important;
        border-color: transparent !important;
        box-shadow: 0 4px 20px rgba(37,99,235,0.45);
        animation: glowPulse 3s ease-in-out infinite;
    }}
    .nav-active .stButton > button:hover {{ transform: none; }}

    .sb-divider {{
        height: 1px;
        background: var(--border);
        margin: 16px 0;
    }}

    .sb-preview-card {{
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid var(--border);
        margin-bottom: 14px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.35);
    }}
    .sb-preview-card img {{ width: 100%; display: block; }}

    .secure-box {{
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 13px 15px;
        border-radius: 14px;
        background: rgba(59,130,246,0.12);
        border: 1px solid rgba(96,165,250,0.35);
        font-size: 12.5px;
        color: var(--text-secondary);
        box-shadow: 0 4px 16px rgba(0,0,0,0.25);
    }}
    .secure-box img {{ width: 28px; height: 28px; object-fit: contain; flex-shrink: 0; }}
    .secure-box b {{ color: #ffffff; display: block; margin-bottom: 2px; font-size: 13.5px; }}

    /* ---------------------------------------------------------------- */
    /* Top header                                                        */
    /* ---------------------------------------------------------------- */
    .header-row {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 16px;
        margin-bottom: 26px;
        animation: fadeIn 0.5s ease;
    }}
    .header-left {{ display: flex; align-items: center; gap: 16px; }}
    .header-icon {{
        width: 62px; height: 62px;
        border-radius: 18px;
        background: rgba(59,130,246,0.14);
        border: 1px solid rgba(96,165,250,0.35);
        display: flex; align-items: center; justify-content: center;
        box-shadow: 0 0 26px rgba(59,130,246,0.45);
        animation: glowPulse 3.5s ease-in-out infinite;
        padding: 10px;
    }}
    .header-icon img {{ width: 100%; height: 100%; object-fit: contain; filter: drop-shadow(0 0 8px rgba(96,165,250,0.6)); }}
    .header-icon .icon-fallback {{ font-size: 30px; width: 100%; height: 100%; }}
    .sb-logo-img.icon-fallback {{ font-size: 22px; width: 42px; height: 42px; border-radius: 12px; background: rgba(59,130,246,0.18); border: 1px solid rgba(96,165,250,0.4); }}
    .secure-box .icon-fallback {{ font-size: 20px; width: 26px; height: 26px; }}
    .header-title {{
        font-size: 30px;
        font-weight: 900;
        letter-spacing: 0.8px;
        margin: 0;
        color: var(--text);
        text-shadow: 0 0 24px rgba(59,130,246,0.25);
    }}
    .header-subtitle {{
        margin: 3px 0 0 0;
        color: var(--text-secondary);
        font-size: 14.5px;
        font-weight: 500;
    }}
    .header-badge {{
        display: flex; align-items: center; gap: 8px;
        padding: 10px 18px;
        border-radius: 999px;
        background: rgba(255,255,255,0.04);
        border: 1px solid var(--border);
        font-size: 13px;
        font-weight: 600;
        color: var(--text);
        backdrop-filter: blur(10px);
    }}
    .header-badge .dot {{
        width: 8px; height: 8px; border-radius: 50%;
        background: #34d399;
        box-shadow: 0 0 8px #34d399;
    }}

    /* ---------------------------------------------------------------- */
    /* Glass cards                                                       */
    /* ---------------------------------------------------------------- */
    .glass-card {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 24px;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        box-shadow: 0 10px 40px rgba(0,0,0,0.45);
        height: 100%;
        transition: box-shadow 0.3s ease, border-color 0.3s ease;
        animation: fadeIn 0.6s ease;
    }}
    .glass-card:hover {{
        border-color: rgba(96,165,250,0.3);
        box-shadow: 0 10px 44px rgba(59,130,246,0.18);
    }}

    .card-title {{
        font-size: 17px;
        font-weight: 700;
        color: var(--text);
        margin-bottom: 2px;
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .card-subtitle {{
        font-size: 12.5px;
        color: var(--text-secondary);
        margin-bottom: 18px;
    }}

    /* Uploader */
    [data-testid="stFileUploaderDropzone"] {{
        background: rgba(255,255,255,0.02) !important;
        border: 1.5px dashed rgba(96,165,250,0.5) !important;
        border-radius: 16px !important;
        transition: all 0.25s ease;
    }}
    [data-testid="stFileUploaderDropzone"]:hover {{
        border-color: rgba(96,165,250,0.9) !important;
        background: rgba(59,130,246,0.05) !important;
    }}
    [data-testid="stFileUploaderDropzone"] * {{ color: var(--text-secondary) !important; }}
    [data-testid="stFileUploaderDropzone"] button {{
        background: linear-gradient(90deg, var(--primary), #1d4ed8) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease;
    }}
    [data-testid="stFileUploaderDropzone"] button:hover {{
        box-shadow: 0 0 18px rgba(59,130,246,0.55);
        transform: translateY(-1px);
    }}

    .tips-box {{
        margin-top: 18px;
        padding: 15px 17px;
        border-radius: 14px;
        background: rgba(255,255,255,0.03);
        border: 1px solid var(--border);
    }}
    .tips-box .t-title {{
        font-size: 13px; font-weight: 700; color: var(--text); margin-bottom: 9px;
        display: flex; align-items: center; gap: 6px;
    }}
    .tips-box ul {{ margin: 0; padding-left: 18px; color: var(--text-secondary); font-size: 12.5px; line-height: 2; }}

    /* Analyze button */
    div[data-testid="column"] .stButton > button {{
        width: 100%;
        border-radius: 12px !important;
        background: linear-gradient(90deg, var(--primary), #1d4ed8) !important;
        color: #fff !important;
        border: none !important;
        font-weight: 700 !important;
        font-size: 14.5px !important;
        padding: 12px 0 !important;
        margin-top: 16px;
        transition: all 0.25s cubic-bezier(.4,0,.2,1);
        box-shadow: 0 6px 20px rgba(37,99,235,0.35);
    }}
    div[data-testid="column"] .stButton > button:hover {{
        box-shadow: 0 8px 28px rgba(59,130,246,0.55);
        transform: translateY(-2px);
    }}
    div[data-testid="column"] .stButton > button:active {{ transform: translateY(0px) scale(0.98); }}
    div[data-testid="column"] .stButton > button:disabled {{
        background: rgba(255,255,255,0.06) !important;
        color: var(--text-secondary) !important;
        box-shadow: none !important;
        cursor: not-allowed;
    }}

    /* Preview panel */
    .preview-frame {{
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid var(--border);
        background: #030712;
        display: flex; align-items: center; justify-content: center;
        min-height: 340px;
    }}
    .preview-empty {{
        color: #4c5c78;
        font-size: 13px;
        text-align: center;
        padding: 70px 24px;
        line-height: 1.8;
    }}
    .preview-empty .pe-icon {{ font-size: 30px; display: block; margin-bottom: 10px; opacity: 0.5; }}

    div[data-testid="stImage"] img {{
        border-radius: 16px;
        border: 1px solid var(--border);
        box-shadow: 0 8px 30px rgba(0,0,0,0.45);
    }}

    /* Prediction result panel */
    .result-icon-wrap {{ display: flex; justify-content: center; margin: 4px 0 12px 0; }}
    .result-icon {{
        width: 84px; height: 84px;
        border-radius: 50%;
        border: 2px solid rgba(96,165,250,0.45);
        background: rgba(59,130,246,0.10);
        display: flex; align-items: center; justify-content: center;
        padding: 18px;
        animation: glowPulse 3s ease-in-out infinite;
    }}
    .result-icon img {{ width: 100%; height: 100%; object-fit: contain; }}
    .result-icon .icon-fallback {{ font-size: 38px; width: 100%; height: 100%; }}
    .result-icon.danger {{
        border-color: rgba(248,113,113,0.5);
        background: rgba(248,113,113,0.10);
    }}

    .result-label {{ text-align: center; color: var(--text-secondary); font-size: 12.5px; letter-spacing: 0.5px; text-transform: uppercase; }}
    .result-value {{ text-align: center; font-size: 32px; font-weight: 900; letter-spacing: 1.2px; margin: 4px 0 2px 0; }}
    .result-value.normal {{ color: var(--primary); text-shadow: 0 0 26px rgba(59,130,246,0.5); }}
    .result-value.pneumonia {{ color: var(--danger); text-shadow: 0 0 26px rgba(248,113,113,0.5); }}
    .result-sub {{ text-align: center; color: var(--text-secondary); font-size: 12.5px; margin-bottom: 20px; }}

    .conf-title {{ text-align: center; color: var(--text-secondary); font-size: 12px; text-transform: uppercase; letter-spacing: 0.6px; margin-top: 16px; }}
    .conf-score {{ text-align: center; font-size: 28px; font-weight: 800; color: var(--accent); margin-bottom: 18px; }}

    .bar-row {{ display: flex; justify-content: space-between; font-size: 12.5px; color: #c3cee2; margin-bottom: 5px; font-weight: 500; }}
    .bar-track {{
        width: 100%; height: 8px; border-radius: 8px;
        background: rgba(255,255,255,0.07);
        margin-bottom: 16px;
        overflow: hidden;
        position: relative;
    }}
    .bar-fill {{
        height: 100%; border-radius: 8px;
        animation: barGrow 1s cubic-bezier(.4,0,.2,1);
        position: relative;
    }}
    .bar-fill.normal {{ background: linear-gradient(90deg, #2563eb, var(--accent)); box-shadow: 0 0 10px rgba(59,130,246,0.6); }}
    .bar-fill.pneumonia {{ background: linear-gradient(90deg, #dc2626, #fca5a5); box-shadow: 0 0 10px rgba(248,113,113,0.6); }}

    .disclaimer {{
        margin-top: 10px;
        padding: 13px 15px;
        border-radius: 14px;
        background: rgba(59,130,246,0.08);
        border: 1px solid rgba(96,165,250,0.22);
        font-size: 11.5px;
        color: var(--text-secondary);
        display: flex; gap: 9px; align-items: flex-start;
        line-height: 1.6;
    }}

    .stub-box {{
        text-align: center;
        padding: 70px 24px;
        color: var(--text-secondary);
        font-size: 14px;
        line-height: 1.9;
    }}

    /* Spinner text */
    .stSpinner > div {{ color: var(--accent) !important; }}

    /* Responsive */
    @media (max-width: 1200px) {{
        .header-title {{ font-size: 24px; }}
        .glass-card {{ padding: 18px; }}
    }}
    @media (max-width: 900px) {{
        .header-row {{ flex-direction: column; align-items: flex-start; }}
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )


inject_css()


# ==============================================================================
# 7. SIDEBAR
# ==============================================================================
def render_sidebar():
    with st.sidebar:
        logo_html = img_tag(LUNGS_IMG, "sb-logo-img", "logo", fallback="🫁")
        md_html(f"""
            <div class="sb-logo-wrap">
                {logo_html}
                <div class="sb-logo-text">
                    PNEUMONIA<br/>DETECTION
                    <span>AI Health Assistant</span>
                </div>
            </div>
        """)

        nav_items = [
            ("Home", "🏠"),
            ("History", "🕘"),
            ("Statistics", "📊"),
            ("About", "ℹ️"),
            ("Help", "❓"),
        ]

        for label, icon in nav_items:
            is_active = st.session_state.page == label
            if is_active:
                st.markdown('<div class="nav-active">', unsafe_allow_html=True)
            clicked = st.button(f"{icon}   {label}", key=f"nav_{label}", use_container_width=True)
            if is_active:
                st.markdown("</div>", unsafe_allow_html=True)
            if clicked:
                st.session_state.page = label
                st.rerun()

        st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

        if XRAY_SAMPLE_IMG.exists():
            xray_html = img_tag(XRAY_SAMPLE_IMG, "", "sample x-ray")
            md_html(f'<div class="sb-preview-card">{xray_html}</div>')

        shield_html = img_tag(SHIELD_IMG, "", "secure", fallback="🛡️")
        md_html(f"""
            <div class="secure-box">
                {shield_html}
                <div>
                    <b>Secure &amp; Private</b>
                    Your data is safe with us.
                </div>
            </div>
        """)


render_sidebar()


# ==============================================================================
# 8. TOP HEADER
# ==============================================================================
def render_header():
    lungs_html = img_tag(LUNGS_IMG, "", "lungs", fallback="🫁")
    md_html(f"""
        <div class="header-row">
            <div class="header-left">
                <div class="header-icon">{lungs_html}</div>
                <div>
                    <p class="header-title">PNEUMONIA DETECTION</p>
                    <p class="header-subtitle">AI Powered Chest X-ray Analysis</p>
                </div>
            </div>
            <div class="header-badge">
                <span class="dot"></span> AI Health Assistant
            </div>
        </div>
    """)


render_header()


# ==============================================================================
# 9. HOME / UPLOAD & PREDICT PAGE
# ==============================================================================
def render_home():
    col_upload, col_preview, col_result = st.columns([1.05, 1.15, 1.0], gap="medium")

    # -------------------- LEFT: Upload card --------------------
    with col_upload:
        md_html("""
            <div class="glass-card">
            <div class="card-title">📤 Upload Chest X-ray Image</div>
            <div class="card-subtitle">Upload a clear chest X-ray image (JPG, PNG)</div>
        """)

        uploaded_file = st.file_uploader(
            "Drag & drop your image here",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            key="file_uploader",
        )

        if uploaded_file is not None:
            try:
                st.session_state.uploaded_image = Image.open(uploaded_file)
            except Exception:
                st.error("Could not read this image. Please upload a valid JPG or PNG file.")

        md_html("""
            <div class="tips-box">
                <div class="t-title">💡 Tips for best results:</div>
                <ul>
                    <li>Use high quality images</li>
                    <li>Ensure the chest area is clearly visible</li>
                    <li>Supported formats: JPG, PNG</li>
                </ul>
            </div>
        """)

        analyze_clicked = st.button(
            "🔍  Analyze X-ray",
            use_container_width=True,
            disabled=st.session_state.uploaded_image is None,
        )
        md_html("</div>")

    current_image = st.session_state.uploaded_image

    if analyze_clicked and current_image is not None:
        with st.spinner("Analyzing chest X-ray..."):
            label, normal_pct, pneumonia_pct = run_prediction(current_image)
            result = {
                "label": label,
                "normal": normal_pct,
                "pneumonia": pneumonia_pct,
            }
            st.session_state.result = result
            st.session_state.history.append(result)

    # -------------------- CENTER: Preview card --------------------
    with col_preview:
        md_html("""
            <div class="glass-card">
            <div class="card-title">🖼️ Image Preview</div>
            <div class="card-subtitle">Your uploaded scan appears here</div>
        """)

        if current_image is not None:
            st.image(current_image, use_container_width=True)
        else:
            md_html("""
                <div class="preview-frame">
                    <div class="preview-empty">
                        <span class="pe-icon">🩻</span>
                        No image uploaded yet<br/>
                        Upload an X-ray to preview it here
                    </div>
                </div>
            """)
        md_html("</div>")

    # -------------------- RIGHT: Prediction result card --------------------
    with col_result:
        md_html("""
            <div class="glass-card">
            <div class="card-title">🧠 Prediction Result</div>
            <div class="card-subtitle">AI model confidence breakdown</div>
        """)

        result = st.session_state.result

        if result is None:
            shield_html = img_tag(SHIELD_IMG, "", "shield", fallback="🛡️")
            md_html(f"""
                <div class="result-icon-wrap"><div class="result-icon">{shield_html}</div></div>
                <div class="stub-box" style="padding-top:0;">
                    Upload an image and click<br/>"Analyze X-ray" to see results
                </div>
            """)
        else:
            is_normal = result["label"] == "NORMAL"
            css_cls = "normal" if is_normal else "pneumonia"
            icon_cls = "" if is_normal else "danger"
            sub_text = "No signs of Pneumonia detected" if is_normal else "Signs of Pneumonia detected"
            confidence = result["normal"] if is_normal else result["pneumonia"]
            lungs_html = img_tag(LUNGS_IMG, "", "lungs", fallback="🫁")

            md_html(f"""
                <div class="result-icon-wrap"><div class="result-icon {icon_cls}">{lungs_html}</div></div>
                <div class="result-label">Prediction</div>
                <div class="result-value {css_cls}">{result['label']}</div>
                <div class="result-sub">{sub_text}</div>

                <div class="conf-title">Confidence Score</div>
                <div class="conf-score">{confidence:.1f}%</div>

                <div class="bar-row"><span>Normal</span><span>{result['normal']:.1f}%</span></div>
                <div class="bar-track"><div class="bar-fill normal" style="width:{result['normal']:.1f}%"></div></div>

                <div class="bar-row"><span>Pneumonia</span><span>{result['pneumonia']:.1f}%</span></div>
                <div class="bar-track"><div class="bar-fill pneumonia" style="width:{result['pneumonia']:.1f}%"></div></div>

                <div class="disclaimer">ℹ️ This tool is for educational purposes only and is not a substitute for professional medical diagnosis.</div>
            """)
        md_html("</div>")


# ==============================================================================
# 10. HISTORY PAGE
# ==============================================================================
def render_history():
    md_html("""
        <div class="glass-card">
        <div class="card-title">🕘 Prediction History</div>
        <div class="card-subtitle">Your most recent analyses in this session</div>
    """)

    history = st.session_state.history
    if not history:
        md_html('<div class="stub-box">🕘<br/><br/>No predictions yet.<br/>Analyzed X-rays will appear here.</div>')
    else:
        for i, item in enumerate(reversed(history), 1):
            is_normal = item["label"] == "NORMAL"
            css_cls = "normal" if is_normal else "pneumonia"
            confidence = item["normal"] if is_normal else item["pneumonia"]
            md_html(f"""
                <div style="display:flex; justify-content:space-between; align-items:center;
                            padding:14px 16px; border-radius:12px; background:rgba(255,255,255,0.03);
                            border:1px solid var(--border); margin-bottom:10px;">
                    <span style="color:var(--text-secondary); font-size:13px;">#{len(history) - i + 1}</span>
                    <span class="result-value {css_cls}" style="font-size:16px; margin:0;">{item['label']}</span>
                    <span style="color:var(--text); font-weight:600;">{confidence:.1f}%</span>
                </div>
            """)
    md_html("</div>")


# ==============================================================================
# 11. STATISTICS PAGE
# ==============================================================================
def render_statistics():
    md_html("""
        <div class="glass-card">
        <div class="card-title">📊 Statistics</div>
        <div class="card-subtitle">Session overview</div>
    """)

    history = st.session_state.history
    total = len(history)
    normal_count = sum(1 for h in history if h["label"] == "NORMAL")
    pneumonia_count = total - normal_count

    if total == 0:
        md_html('<div class="stub-box">📊<br/><br/>No data yet.<br/>Run a few predictions to see statistics here.</div>')
    else:
        c1, c2, c3 = st.columns(3)
        for col, label, value in [
            (c1, "Total Scans", total),
            (c2, "Normal", normal_count),
            (c3, "Pneumonia", pneumonia_count),
        ]:
            with col:
                md_html(f"""
                    <div style="text-align:center; padding:22px 10px; border-radius:14px;
                                background:rgba(255,255,255,0.03); border:1px solid var(--border);">
                        <div style="font-size:26px; font-weight:800; color:var(--accent);">{value}</div>
                        <div style="font-size:12.5px; color:var(--text-secondary); margin-top:4px;">{label}</div>
                    </div>
                """)
    md_html("</div>")


# ==============================================================================
# 12. ABOUT / HELP PAGES
# ==============================================================================
def render_about():
    md_html("""
        <div class="glass-card">
        <div class="card-title">ℹ️ About</div>
        <div class="stub-box" style="text-align:left;">
        <b style="color:var(--text); font-size:16px;">🫁 Pneumonia Detection</b><br/><br/>
        An AI-powered tool that analyzes chest X-ray images using a deep learning model
        (EfficientNetB0, transfer learning) to help identify potential signs of pneumonia.<br/><br/>
        The model was trained on a labeled chest X-ray dataset (NORMAL / PNEUMONIA) and outputs
        a confidence score for each class.<br/><br/>
        <span style="color:var(--danger); font-weight:600;">This tool is for educational purposes
        only and is not a substitute for professional medical diagnosis.</span>
        </div>
        </div>
    """)


def render_help():
    md_html("""
        <div class="glass-card">
        <div class="card-title">❓ Help</div>
        <div class="stub-box" style="text-align:left;">
        <b style="color:var(--text); font-size:15px;">How to use this app:</b><br/><br/>
        1. Go to <b>Home</b> or <b>Upload &amp; Predict</b>.<br/>
        2. Upload a clear chest X-ray image (JPG or PNG).<br/>
        3. Click <b>Analyze X-ray</b>.<br/>
        4. Review the prediction, confidence score, and per-class probabilities.<br/><br/>
        For best accuracy, use a high-resolution frontal chest X-ray with the full chest visible.
        </div>
        </div>
    """)


# ==============================================================================
# 13. ROUTER
# ==============================================================================
def render_page():
    page = st.session_state.page
    if page in ("Home", "Upload & Predict"):
        render_home()
    elif page == "History":
        render_history()
    elif page == "Statistics":
        render_statistics()
    elif page == "About":
        render_about()
    elif page == "Help":
        render_help()
    else:
        render_home()


render_page()


# ==============================================================================
# 14. FOOTER NOTICE (model status)
# ==============================================================================
if load_model() is None:
    md_html("""
        <div style="margin-top:22px; padding:12px 16px; border-radius:12px;
                    background:rgba(248,113,113,0.08); border:1px solid rgba(248,113,113,0.25);
                    font-size:12.5px; color:#fca5a5; text-align:center;">
            ⚠️ Model file <code>chest_xray_classifier.keras</code> not found next to app.py —
            showing demo predictions. Add the trained model file to enable real inference.
        </div>
    """)
