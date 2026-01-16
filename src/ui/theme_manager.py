"""
Theme Manager - Light/Dark Mode Support
Provides tasteful, accessible themes with proper contrast ratios
"""

from dataclasses import dataclass
from typing import Dict
import streamlit as st


@dataclass
class ThemeColors:
    """Color scheme for a theme"""
    # Primary brand colors
    primary: str
    secondary: str
    accent: str

    # Semantic colors
    success: str
    warning: str
    danger: str
    info: str

    # Base colors
    background: str
    surface: str
    card: str

    # Text colors
    text_primary: str
    text_secondary: str
    text_muted: str
    text_inverse: str

    # Border colors
    border: str
    border_light: str

    # Interactive states
    hover: str
    active: str
    disabled: str

    # Chart colors
    chart_positive: str
    chart_negative: str
    chart_neutral: str


# Vive Health Brand Colors (Base)
VIVE_BRAND = {
    'turquoise': '#23b2be',  # Pantone P 121-6 C
    'navy': '#004366',        # Pantone P 111-16 C
    'red_orange': '#EB3300',  # Pantone P 2028 C
    'yellow_gold': '#F0B323', # Pantone 7409 C
    'gray': '#777473'         # Pantone P 172-9 C
}


def get_light_theme() -> ThemeColors:
    """
    Light theme with high contrast for readability
    WCAG AAA compliant (7:1 contrast ratio for body text)
    """
    return ThemeColors(
        # Primary brand colors
        primary=VIVE_BRAND['turquoise'],
        secondary=VIVE_BRAND['navy'],
        accent=VIVE_BRAND['red_orange'],

        # Semantic colors
        success='#10b981',       # Green-500
        warning=VIVE_BRAND['yellow_gold'],
        danger=VIVE_BRAND['red_orange'],
        info=VIVE_BRAND['turquoise'],

        # Base colors
        background='#ffffff',    # Pure white
        surface='#f8f9fa',       # Off-white
        card='#ffffff',          # White cards

        # Text colors (high contrast on white)
        text_primary='#1a1a1a',      # Near black
        text_secondary='#4a4a4a',    # Dark gray
        text_muted='#6b7280',        # Gray-500
        text_inverse='#ffffff',       # White text for dark backgrounds

        # Border colors
        border='#e5e7eb',        # Gray-200
        border_light='#f3f4f6',  # Gray-100

        # Interactive states
        hover='#f3f4f6',         # Light gray hover
        active='#e5e7eb',        # Slightly darker active
        disabled='#d1d5db',      # Gray-300

        # Chart colors
        chart_positive='#10b981',
        chart_negative=VIVE_BRAND['red_orange'],
        chart_neutral=VIVE_BRAND['gray']
    )


def get_dark_theme() -> ThemeColors:
    """
    Dark theme with comfortable contrast for extended viewing
    WCAG AA compliant (4.5:1 contrast ratio minimum)
    """
    return ThemeColors(
        # Primary brand colors (adjusted for dark background)
        primary='#34c5d3',       # Lighter turquoise
        secondary='#5b9cd4',     # Lighter blue
        accent='#ff6b4a',        # Softer red-orange

        # Semantic colors
        success='#34d399',       # Green-400
        warning='#fbbf24',       # Amber-400
        danger='#ff6b4a',        # Coral red
        info='#34c5d3',          # Light turquoise

        # Base colors
        background='#0f172a',    # Slate-900
        surface='#1e293b',       # Slate-800
        card='#2d3a52',          # Lighter slate for cards

        # Text colors (optimized for dark background)
        text_primary='#f1f5f9',      # Slate-100
        text_secondary='#cbd5e1',    # Slate-300
        text_muted='#94a3b8',        # Slate-400
        text_inverse='#0f172a',       # Dark text for light backgrounds

        # Border colors
        border='#334155',        # Slate-700
        border_light='#475569',  # Slate-600

        # Interactive states
        hover='#334155',         # Slate-700
        active='#475569',        # Slate-600
        disabled='#475569',      # Slate-600

        # Chart colors
        chart_positive='#34d399',
        chart_negative='#ff6b4a',
        chart_neutral='#94a3b8'
    )


def get_current_theme() -> ThemeColors:
    """Get the currently active theme"""
    if 'theme_mode' not in st.session_state:
        st.session_state.theme_mode = 'light'

    return get_light_theme() if st.session_state.theme_mode == 'light' else get_dark_theme()


def toggle_theme():
    """Toggle between light and dark themes"""
    if 'theme_mode' not in st.session_state:
        st.session_state.theme_mode = 'light'

    st.session_state.theme_mode = 'dark' if st.session_state.theme_mode == 'light' else 'light'


def inject_theme_css():
    """Inject CSS for the current theme into Streamlit"""
    theme = get_current_theme()
    mode = st.session_state.get('theme_mode', 'light')

    css = f"""
    <style>
    /* Theme Variables */
    :root {{
        --primary: {theme.primary};
        --secondary: {theme.secondary};
        --accent: {theme.accent};
        --success: {theme.success};
        --warning: {theme.warning};
        --danger: {theme.danger};
        --info: {theme.info};

        --background: {theme.background};
        --surface: {theme.surface};
        --card: {theme.card};

        --text-primary: {theme.text_primary};
        --text-secondary: {theme.text_secondary};
        --text-muted: {theme.text_muted};
        --text-inverse: {theme.text_inverse};

        --border: {theme.border};
        --border-light: {theme.border_light};

        --hover: {theme.hover};
        --active: {theme.active};
        --disabled: {theme.disabled};
    }}

    /* Override Streamlit's default styles */
    .stApp {{
        background-color: var(--background);
        color: var(--text-primary);
    }}

    /* Main content area */
    .main .block-container {{
        background-color: var(--background);
    }}

    /* Headers */
    h1, h2, h3, h4, h5, h6 {{
        color: var(--text-primary) !important;
    }}

    /* Paragraphs and text */
    p, span, div, label {{
        color: var(--text-primary);
    }}

    /* Code blocks */
    code {{
        background-color: var(--surface);
        color: var(--text-primary);
        border: 1px solid var(--border);
    }}

    pre {{
        background-color: var(--surface);
        border: 1px solid var(--border);
    }}

    /* Cards and containers */
    .element-container {{
        color: var(--text-primary);
    }}

    /* Expanders */
    .streamlit-expanderHeader {{
        background-color: var(--surface) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border);
    }}

    .streamlit-expanderHeader:hover {{
        background-color: var(--hover) !important;
    }}

    .streamlit-expanderContent {{
        background-color: var(--card);
        border: 1px solid var(--border);
        border-top: none;
    }}

    /* Buttons */
    .stButton > button {{
        color: var(--text-primary);
        border: 1px solid var(--border);
        background-color: var(--surface);
    }}

    .stButton > button:hover {{
        background-color: var(--hover);
        border-color: var(--primary);
    }}

    .stButton > button[kind="primary"] {{
        background-color: var(--primary);
        color: var(--text-inverse);
    }}

    .stButton > button[kind="primary"]:hover {{
        background-color: var(--secondary);
    }}

    /* Input fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div,
    .stMultiSelect > div > div {{
        background-color: var(--surface);
        color: var(--text-primary);
        border-color: var(--border);
    }}

    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {{
        border-color: var(--primary);
    }}

    /* DataFrames and tables */
    .dataframe {{
        background-color: var(--card);
        color: var(--text-primary);
    }}

    .dataframe th {{
        background-color: var(--surface);
        color: var(--text-primary);
        border-color: var(--border);
    }}

    .dataframe td {{
        background-color: var(--card);
        color: var(--text-primary);
        border-color: var(--border-light);
    }}

    /* Sidebar */
    .css-1d391kg, [data-testid="stSidebar"] {{
        background-color: var(--surface);
    }}

    [data-testid="stSidebar"] * {{
        color: var(--text-primary);
    }}

    /* Metrics */
    .css-1xarl3l {{
        color: var(--text-primary);
    }}

    [data-testid="stMetricValue"] {{
        color: var(--text-primary);
    }}

    [data-testid="stMetricLabel"] {{
        color: var(--text-secondary);
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        background-color: var(--surface);
        border-bottom: 2px solid var(--border);
    }}

    .stTabs [data-baseweb="tab"] {{
        color: var(--text-secondary);
        background-color: transparent;
    }}

    .stTabs [data-baseweb="tab"]:hover {{
        background-color: var(--hover);
        color: var(--text-primary);
    }}

    .stTabs [aria-selected="true"] {{
        background-color: var(--card);
        color: var(--primary);
        border-bottom: 3px solid var(--primary);
    }}

    /* Alerts */
    .stAlert {{
        background-color: var(--surface);
        color: var(--text-primary);
        border: 1px solid var(--border);
    }}

    /* Success messages */
    .stSuccess {{
        background-color: var(--success);
        color: var(--text-inverse);
    }}

    /* Warning messages */
    .stWarning {{
        background-color: var(--warning);
        color: var(--text-inverse);
    }}

    /* Error messages */
    .stError {{
        background-color: var(--danger);
        color: var(--text-inverse);
    }}

    /* Info messages */
    .stInfo {{
        background-color: var(--info);
        color: var(--text-inverse);
    }}

    /* Markdown */
    .stMarkdown {{
        color: var(--text-primary);
    }}

    /* Links */
    a {{
        color: var(--primary);
    }}

    a:hover {{
        color: var(--secondary);
    }}

    /* Divider */
    hr {{
        border-color: var(--border);
    }}

    /* File uploader */
    .stFileUploader {{
        background-color: var(--surface);
        border: 2px dashed var(--border);
    }}

    /* Progress bar */
    .stProgress > div > div {{
        background-color: var(--primary);
    }}

    /* Spinner */
    .stSpinner > div {{
        border-color: var(--primary);
    }}

    /* Toast notifications */
    .stToast {{
        background-color: var(--card);
        color: var(--text-primary);
        border: 1px solid var(--border);
    }}

    /* Better contrast for small text */
    small, .small {{
        color: var(--text-secondary);
    }}

    /* Caption text */
    .caption {{
        color: var(--text-muted);
    }}

    /* Theme toggle button styling */
    .theme-toggle {{
        position: fixed;
        top: 4.5rem;
        right: 1rem;
        z-index: 999;
        background-color: var(--card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }}

    /* Better markdown rendering */
    .stMarkdown code {{
        background-color: var(--surface);
        color: var(--primary);
        padding: 0.2em 0.4em;
        border-radius: 3px;
    }}

    /* Better table styling */
    table {{
        background-color: var(--card);
        border: 1px solid var(--border);
    }}

    th {{
        background-color: var(--surface);
        color: var(--text-primary);
        font-weight: 600;
    }}

    td {{
        color: var(--text-primary);
    }}
    </style>
    """

    st.markdown(css, unsafe_allow_html=True)


def render_theme_toggle():
    """Render theme toggle button"""
    theme = get_current_theme()
    current_mode = st.session_state.get('theme_mode', 'light')

    # Icon and text based on current mode
    icon = '🌙' if current_mode == 'light' else '☀️'
    text = 'Dark Mode' if current_mode == 'light' else 'Light Mode'

    col1, col2, col3 = st.columns([6, 2, 1])

    with col2:
        if st.button(f"{icon} {text}", key='theme_toggle_btn', use_container_width=True):
            toggle_theme()
            st.rerun()


def get_color(color_name: str) -> str:
    """Get a specific color from the current theme"""
    theme = get_current_theme()
    return getattr(theme, color_name, theme.primary)


def get_status_color(status: str) -> str:
    """Get appropriate color for a status"""
    theme = get_current_theme()
    status_lower = status.lower()

    if any(word in status_lower for word in ['success', 'complete', 'pass', 'good', 'excellent']):
        return theme.success
    elif any(word in status_lower for word in ['warning', 'risk', 'review', 'caution']):
        return theme.warning
    elif any(word in status_lower for word in ['error', 'fail', 'danger', 'critical', 'high']):
        return theme.danger
    elif any(word in status_lower for word in ['info', 'pending', 'processing']):
        return theme.info
    else:
        return theme.text_muted
