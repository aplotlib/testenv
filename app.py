"""
Vive Health Quality Suite - Version 19.0
Enhanced Quality Case Screening with Statistical Rigor

Tab 1: Return Categorizer (PRESERVED)
Tab 2: B2B Report Generator (PRESERVED)  
Tab 3: Quality Case Screening (REBUILT)

Features:
- ANOVA/MANOVA with p-values and post-hoc testing
- SPC Control Charting (CUSUM, Shewhart)
- Weighted Risk Scoring
- AI-powered cross-case correlation
- Fuzzy threshold matching
- Vendor email generation
- Investigation plan generation
- State persistence (session-based)
- Custom threshold profiles
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import io
from typing import Dict, List, Any, Optional, Tuple
import time
from collections import Counter, defaultdict
import re
import os
import gc
import json

# Visualization
try:
    import altair as alt
    ALTAIR_AVAILABLE = True
except ImportError:
    ALTAIR_AVAILABLE = False

# --- Custom Modules ---
try:
    from enhanced_ai_analysis import (
        EnhancedAIAnalyzer, AIProvider, FBA_REASON_MAP,
        MEDICAL_DEVICE_CATEGORIES
    )
    from quality_analytics import (
        QualityAnalytics, QualityStatistics, SPCAnalysis, TrendAnalysis,
        RiskScoring, ActionDetermination, VendorEmailGenerator,
        InvestigationPlanGenerator, DataValidation,
        SOP_THRESHOLDS, parse_numeric, parse_percentage,
        fuzzy_match_category, generate_methodology_markdown
    )
    AI_AVAILABLE = True
except ImportError as e:
    AI_AVAILABLE = False
    print(f"Module Missing: {e}")

# Check optional imports
try:
    import xlsxwriter
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- App Configuration ---
st.set_page_config(
    page_title="Vive Health Quality Suite",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

APP_CONFIG = {
    'title': 'Vive Health Quality Suite',
    'version': '19.0 (Enhanced Screening)',
    'chunk_sizes': [100, 250, 500, 1000],
    'default_chunk': 500,
}

# Colors
COLORS = {
    'primary': '#00D9FF',
    'secondary': '#FF006E',
    'accent': '#FFB700',
    'success': '#00F5A0',
    'warning': '#FF6B35',
    'danger': '#FF0054',
    'dark': '#0A0A0F',
    'light': '#1A1A2E',
    'text': '#E0E0E0',
    'muted': '#666680',
    'cost': '#50C878'
}

# Quality categories (For Tab 1 Analysis)
QUALITY_CATEGORIES = [
    'Product Defects/Quality',
    'Performance/Effectiveness',
    'Missing Components',
    'Design/Material Issues',
    'Stability/Positioning Issues',
    'Medical/Health Concerns'
]

# AI Provider options - OpenAI default for Tab 3
AI_PROVIDER_OPTIONS = {
    'OpenAI GPT-3.5 (Default)': AIProvider.OPENAI,
    'Claude Haiku (Fast)': AIProvider.FASTEST,
    'Claude Sonnet': AIProvider.CLAUDE,
    'Both (Consensus)': AIProvider.BOTH
}

# Default category thresholds (from SOPs)
DEFAULT_CATEGORY_THRESHOLDS = {
    'B2B': 0.025,
    'INS': 0.07,
    'RHB': 0.075,
    'LVA': 0.095,
    'MOB': 0.10,
    'CSH': 0.105,
    'SUP': 0.11,
    'All Others': 0.10
}

# --- Initialization & Styling ---

def inject_custom_css():
    """Inject custom CSS for modern UI"""
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    :root {{
        --primary: {COLORS['primary']};
        --secondary: {COLORS['secondary']};
        --accent: {COLORS['accent']};
        --success: {COLORS['success']};
        --warning: {COLORS['warning']};
        --danger: {COLORS['danger']};
        --dark: {COLORS['dark']};
        --light: {COLORS['light']};
        --text: {COLORS['text']};
    }}
    
    html, body, .stApp {{
        font-family: 'Inter', sans-serif;
    }}
    
    .main-header {{
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 5px 20px rgba(0, 217, 255, 0.3);
    }}
    
    .main-title {{
        font-size: 2.2em;
        font-weight: 700;
        color: white;
        margin: 0;
    }}
    
    .info-box {{
        background: rgba(26, 26, 46, 0.8);
        border: 1px solid var(--primary);
        border-radius: 8px;
        padding: 1.2rem;
        margin: 0.8rem 0;
    }}
    
    .stButton > button {{
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        color: white;
        border: none;
        padding: 0.7rem 1.5rem;
        border-radius: 6px;
        font-weight: 600;
    }}
    
    .risk-critical {{
        background-color: #ff4b4b !important;
        color: white !important;
    }}
    
    .risk-warning {{
        background-color: #ffa500 !important;
        color: black !important;
    }}
    
    .risk-monitor {{
        background-color: #ffff00 !important;
        color: black !important;
    }}
    
    .risk-ok {{
        background-color: #00ff00 !important;
        color: black !important;
    }}
    
    .processing-log {{
        background: #1a1a2e;
        border: 1px solid #333;
        border-radius: 5px;
        padding: 10px;
        max-height: 200px;
        overflow-y: auto;
        font-family: monospace;
        font-size: 12px;
    }}
    
    .methodology-box {{
        background: #f8f9fa;
        border-left: 4px solid #4facfe;
        padding: 15px;
        margin: 10px 0;
    }}
    
    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    </style>
    """, unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables for all tabs"""
    defaults = {
        # General AI
        'ai_analyzer': None,
        'ai_provider': AIProvider.OPENAI,  # Default to OpenAI for Tab 3
        
        # Tab 1: Categorizer
        'categorized_data': None,
        'processing_complete': False,
        'reason_summary': {},
        'product_summary': {},
        'total_cost': 0.0,
        'batch_size': 20,
        'chunk_size': APP_CONFIG['default_chunk'],
        'processing_errors': [],
        'column_mapping': {},
        'show_product_analysis': False,
        'processing_speed': 0.0,
        'export_data': None,
        'export_filename': None,
        
        # Tab 2: B2B Reports
        'b2b_processed_data': None,
        'b2b_processing_complete': False,
        'b2b_export_data': None,
        'b2b_export_filename': None,
        'b2b_perf_mode': 'Small (< 500 rows)',
        
        # Tab 3: Quality Screening - NEW
        'qc_mode': 'Lite',
        'qc_results': None,
        'qc_results_df': None,
        'processing_log': [],
        'anova_result': None,
        'manova_result': None,
        'statistical_suggestion': None,
        
        # Threshold profiles
        'threshold_profiles': {
            'Standard Review': DEFAULT_CATEGORY_THRESHOLDS.copy(),
            'Aggressive Q4 Review': {k: v * 0.8 for k, v in DEFAULT_CATEGORY_THRESHOLDS.items()}
        },
        'active_profile': 'Standard Review',
        'custom_thresholds': None,
        
        # User-uploaded threshold data
        'user_threshold_data': None,
        
        # AI Chat state
        'ai_chat_history': [],
        'ai_needs_clarification': False,
        'ai_clarification_question': '',
        
        # Session persistence
        'saved_sessions': {},
        'current_session_id': None,
        
        # Manual entry data (Lite mode)
        'lite_entries': [],
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def check_api_keys():
    """Check for API keys and set environment variables"""
    keys_found = {}
    try:
        if hasattr(st, 'secrets'):
            # Check OpenAI
            for key in ['OPENAI_API_KEY', 'openai_api_key', 'openai']:
                if key in st.secrets:
                    val = str(st.secrets[key]).strip()
                    if val:
                        keys_found['openai'] = val
                        os.environ['OPENAI_API_KEY'] = val
                        break
            
            # Check Claude
            for key in ['ANTHROPIC_API_KEY', 'anthropic_api_key', 'claude_api_key', 'claude']:
                if key in st.secrets:
                    val = str(st.secrets[key]).strip()
                    if val:
                        keys_found['claude'] = val
                        os.environ['ANTHROPIC_API_KEY'] = val
                        break
    except Exception as e:
        logger.warning(f"Error checking secrets: {e}")
    
    return keys_found


def get_ai_analyzer(provider: AIProvider = None, max_workers: int = 5):
    """Get or create AI analyzer instance"""
    if provider is None:
        provider = st.session_state.ai_provider
    
    if st.session_state.ai_analyzer is None or st.session_state.ai_analyzer.provider != provider:
        try:
            check_api_keys()
            st.session_state.ai_analyzer = EnhancedAIAnalyzer(provider, max_workers=max_workers)
            logger.info(f"Created AI analyzer: {provider.value}, Workers: {max_workers}")
        except Exception as e:
            st.error(f"Error initializing AI: {str(e)}")
    
    return st.session_state.ai_analyzer


def log_process(message: str, msg_type: str = 'info'):
    """Adds message to the Processing Transparency Log"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    entry = f"[{timestamp}] [{msg_type.upper()}] {message}"
    st.session_state.processing_log.append(entry)
    if msg_type == 'error':
        logger.error(message)
    else:
        logger.info(message)


def render_api_health_check():
    """Render API health check status in sidebar"""
    keys = check_api_keys()
    
    st.sidebar.markdown("### 🔌 API Status")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if keys.get('openai'):
            st.success("OpenAI ✓")
        else:
            st.error("OpenAI ✗")
    
    with col2:
        if keys.get('claude'):
            st.success("Claude ✓")
        else:
            st.warning("Claude ✗")
    
    return keys


# -------------------------
# TAB 1 LOGIC: Categorizer (PRESERVED)
# -------------------------

def process_file_preserve_structure(file_content, filename):
    """Process file while preserving original structure"""
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_content), dtype=str)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(file_content), dtype=str)
        elif filename.endswith('.txt'):
            df = pd.read_csv(io.BytesIO(file_content), sep='\t', dtype=str)
        else:
            return None, None
        
        column_mapping = {}
        cols = df.columns.tolist()
        
        if len(cols) >= 11:
            if len(cols) > 8: column_mapping['complaint'] = cols[8]
            if len(cols) > 1: column_mapping['sku'] = cols[1]
            
            while len(df.columns) < 11:
                df[f'Column_{len(df.columns)}'] = ''
            column_mapping['category'] = df.columns[10]
            df[column_mapping['category']] = ''
        else:
            st.error("File structure not recognized. Need at least 11 columns (A-K).")
            return None, None
            
        return df, column_mapping
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None, None


def process_in_chunks(df, analyzer, column_mapping):
    """Process categorization in chunks"""
    chunk_size = st.session_state.chunk_size
    complaint_col = column_mapping['complaint']
    category_col = column_mapping['category']
    
    valid_indices = df[df[complaint_col].notna() & (df[complaint_col].str.strip() != '')].index
    total_valid = len(valid_indices)
    
    progress_bar = st.progress(0)
    stats_container = st.container()
    processed_count = 0
    start_time = time.time()
    
    for chunk_start in range(0, total_valid, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_valid)
        chunk_indices = valid_indices[chunk_start:chunk_end]
        
        batch_data = []
        for idx in chunk_indices:
            batch_data.append({
                'index': idx,
                'complaint': str(df.at[idx, complaint_col]).strip(),
                'fba_reason': str(df.at[idx, 'reason']) if 'reason' in df.columns else None
            })
            
        sub_batch_size = st.session_state.batch_size
        for i in range(0, len(batch_data), sub_batch_size):
            sub_batch = batch_data[i:i+sub_batch_size]
            results = analyzer.categorize_batch(sub_batch, mode='standard')
            
            for result in results:
                df.at[result['index'], category_col] = result.get('category', 'Other/Miscellaneous')
                processed_count += 1
            
            progress = processed_count / total_valid
            progress_bar.progress(progress)
            
            elapsed = time.time() - start_time
            speed = processed_count / elapsed if elapsed > 0 else 0
            
            with stats_container:
                c1, c2 = st.columns(2)
                c1.metric("Processed", f"{processed_count}/{total_valid}")
                c2.metric("Speed", f"{speed:.1f}/sec")
            
            gc.collect()
            
    return df


def generate_statistics(df, column_mapping):
    """Generate summary stats for dashboard"""
    cat_col = column_mapping.get('category')
    sku_col = column_mapping.get('sku')
    
    if not cat_col: return
    
    categorized = df[df[cat_col].notna() & (df[cat_col] != '')]
    st.session_state.reason_summary = categorized[cat_col].value_counts().to_dict()
    
    if sku_col:
        prod_summary = defaultdict(lambda: defaultdict(int))
        for _, row in categorized.iterrows():
            if pd.notna(row.get(sku_col)):
                prod_summary[str(row[sku_col])][row[cat_col]] += 1
        st.session_state.product_summary = dict(prod_summary)


def export_with_column_k(df):
    """Export to Excel preserving format"""
    output = io.BytesIO()
    if EXCEL_AVAILABLE:
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Returns')
            workbook = writer.book
            worksheet = writer.sheets['Returns']
            fmt = workbook.add_format({'bg_color': '#E6F5E6', 'font_color': '#006600', 'bold': True})
            worksheet.set_column(10, 10, 20, fmt)
    else:
        df.to_csv(output, index=False)
    output.seek(0)
    return output.getvalue()


def display_results_dashboard(df, column_mapping):
    """Render Tab 1 Dashboard"""
    st.markdown("### 📊 Analysis Results")
    total = len(df)
    cat_col = column_mapping.get('category')
    categorized = len(df[df[cat_col].notna() & (df[cat_col] != '')])
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Rows", total)
    c2.metric("Categorized", categorized)
    
    quality_count = sum(c for k, c in st.session_state.reason_summary.items() if k in QUALITY_CATEGORIES)
    quality_pct = (quality_count / categorized * 100) if categorized else 0
    c3.metric("Quality Issues", f"{quality_pct:.1f}%")
    
    st.markdown("#### Top Categories")
    st.bar_chart(pd.Series(st.session_state.reason_summary).head(10))


# -------------------------
# TAB 2 LOGIC: B2B Reports (PRESERVED)
# -------------------------

def extract_main_sku(text):
    """Extract 3 Caps + 4 Digits (e.g., MOB1027)"""
    if not isinstance(text, str): return None
    match = re.search(r'\b([A-Z]{3}\d{4})', text)
    return match.group(1) if match else None


def find_sku_in_row(row):
    """Heuristic to find SKU in row"""
    for col in ['Main SKU', 'Main SKU/Display Name', 'SKU', 'Product', 'Internal Reference']:
        if col in row.index and pd.notna(row[col]):
            sku = extract_main_sku(str(row[col]))
            if sku: return sku
    
    for col in ['Display Name', 'Subject', 'Name', 'Description', 'Body']:
        if col in row.index and pd.notna(row[col]):
            sku = extract_main_sku(str(row[col]))
            if sku: return sku
    return "Unknown"


def process_b2b_file(file_content, filename):
    try:
        if filename.endswith('.csv'):
            return pd.read_csv(io.BytesIO(file_content), dtype=str)
        else:
            return pd.read_excel(io.BytesIO(file_content), dtype=str)
    except Exception as e:
        st.error(f"Error: {e}")
        return None


def generate_b2b_report(df, analyzer, batch_size):
    """Run AI Summary and SKU extraction"""
    display_col = 'Display Name' if 'Display Name' in df.columns else df.columns[0]
    desc_col = 'Description' if 'Description' in df.columns else None
    
    items = []
    for idx, row in df.iterrows():
        items.append({
            'index': idx,
            'subject': str(row.get(display_col, '')),
            'details': str(row.get(desc_col, ''))[:1000],
            'full_description': str(row.get(desc_col, '')),
            'sku': find_sku_in_row(row)
        })
        
    results = []
    progress_bar = st.progress(0)
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        results.extend(analyzer.summarize_batch(batch))
        progress_bar.progress(min((i + batch_size) / len(items), 1.0))
        
    final = []
    for item in results:
        final.append({
            'Display Name': item['subject'],
            'Description': item['full_description'],
            'SKU': item['sku'],
            'Reason': item.get('summary', 'Error')
        })
    return pd.DataFrame(final)


# -------------------------
# TAB 3 LOGIC: Quality Screening (REBUILT)
# -------------------------

def render_quality_screening_tab():
    """Render the completely rebuilt Quality Case Screening tab"""
    
    st.markdown("### 🧪 Quality Case Screening")
    st.caption("AI-powered quality screening compliant with ISO 13485, FDA 21 CFR 820, EU MDR, UK MDR")
    
    # --- MODE SELECTION ---
    col1, col2, col3 = st.columns([2, 2, 2])
    
    with col1:
        mode = st.radio(
            "Screening Mode",
            ["Lite (1-5 Products)", "Pro (Mass Analysis)"],
            horizontal=True,
            help="Lite: Manual entry for quick screening. Pro: Upload CSV/Excel for batch analysis."
        )
        st.session_state.qc_mode = "Lite" if "Lite" in mode else "Pro"
    
    with col2:
        # AI Provider selection - OpenAI default
        ai_provider = st.selectbox(
            "AI Provider",
            options=list(AI_PROVIDER_OPTIONS.keys()),
            index=0,  # OpenAI default
            help="Select AI provider. OpenAI is default. Claude available for additional review."
        )
        st.session_state.ai_provider = AI_PROVIDER_OPTIONS[ai_provider]
    
    with col3:
        # Threshold profile selection
        profile = st.selectbox(
            "Threshold Profile",
            options=list(st.session_state.threshold_profiles.keys()) + ["+ Create New"],
            index=list(st.session_state.threshold_profiles.keys()).index(st.session_state.active_profile) 
                  if st.session_state.active_profile in st.session_state.threshold_profiles else 0
        )
        if profile != "+ Create New":
            st.session_state.active_profile = profile
    
    st.divider()
    
    # --- SIDEBAR CONFIGURATION ---
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📋 Tab 3 Config")
        
        # Custom threshold file upload
        st.markdown("#### Custom Thresholds")
        threshold_file = st.file_uploader(
            "Upload threshold CSV",
            type=['csv', 'xlsx'],
            help="Upload file with Category and Return Rate Threshold columns",
            key="threshold_upload"
        )
        
        if threshold_file:
            try:
                if threshold_file.name.endswith('.csv'):
                    threshold_df = pd.read_csv(threshold_file)
                else:
                    threshold_df = pd.read_excel(threshold_file)
                st.session_state.user_threshold_data = threshold_df
                st.success(f"Loaded {len(threshold_df)} threshold rules")
            except Exception as e:
                st.error(f"Error loading thresholds: {e}")
        
        # Processing log
        with st.expander("📜 Processing Log", expanded=False):
            if st.session_state.processing_log:
                log_text = "\n".join(st.session_state.processing_log[-20:])
                st.code(log_text, language="")
            else:
                st.caption("No logs yet")
            
            if st.button("Clear Log", key="clear_log"):
                st.session_state.processing_log = []
    
    # --- MAIN CONTENT ---
    
    if st.session_state.qc_mode == "Lite":
        render_lite_mode()
    else:
        render_pro_mode()
    
    # --- RESULTS DISPLAY ---
    if st.session_state.qc_results_df is not None:
        render_screening_results()


def render_lite_mode():
    """Render Lite mode - manual entry for 1-5 products"""
    
    st.info("ℹ️ **Lite Mode**: Enter product details manually for quick screening (1-5 products)")
    
    # Date range selection
    col1, col2 = st.columns(2)
    with col1:
        date_range = st.selectbox(
            "Data Date Range",
            ["Last 30 days", "Last 60 days", "Last 90 days", "Last 180 days", "Last 365 days", "Custom Range"],
            index=0
        )
    with col2:
        if date_range == "Custom Range":
            date_start = st.date_input("Start Date", datetime.now() - timedelta(days=30))
            date_end = st.date_input("End Date", datetime.now())
        else:
            days = int(re.search(r'\d+', date_range).group())
            date_start = datetime.now() - timedelta(days=days)
            date_end = datetime.now()
    
    st.markdown("---")
    
    # Product entry form
    with st.form("lite_entry_form"):
        st.markdown("#### Product Information (Required)")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            product_name = st.text_input("Product Name*", placeholder="e.g., Knee Walker")
        with col2:
            product_sku = st.text_input("Product SKU*", placeholder="e.g., MOB1027")
        with col3:
            category = st.selectbox(
                "Category*",
                options=['Select...'] + list(DEFAULT_CATEGORY_THRESHOLDS.keys()),
                index=0
            )
        
        col4, col5, col6 = st.columns(3)
        with col4:
            units_sold = st.number_input("Units Sold*", min_value=1, value=100)
        with col5:
            units_returned = st.number_input("Units Returned*", min_value=0, value=0)
        with col6:
            return_rate_calc = (units_returned / units_sold * 100) if units_sold > 0 else 0
            st.metric("Calculated Return Rate", f"{return_rate_calc:.1f}%")
        
        # Complaint reasons
        complaint_reasons = st.text_area(
            "Top Return Reasons (comma-separated)*",
            placeholder="e.g., Uncomfortable, Too small, Defective wheel, Missing parts",
            help="Enter the most common return reasons separated by commas"
        )
        
        st.markdown("#### Optional Fields")
        
        col7, col8, col9 = st.columns(3)
        with col7:
            primary_channel = st.selectbox(
                "Primary Sales Channel",
                ["Select...", "Amazon", "B2B", "Website", "Other"]
            )
        with col8:
            channel_distribution = st.text_input(
                "Channel Distribution",
                placeholder="e.g., Amazon 60%, B2B 40%"
            )
        with col9:
            packaging_method = st.selectbox(
                "Packaging Method",
                ["Select...", "Standard Box", "Poly Bag", "Custom", "Other"]
            )
        
        col10, col11, col12 = st.columns(3)
        with col10:
            unit_cost = st.number_input("Landed Cost ($)", min_value=0.0, value=0.0, step=0.01)
        with col11:
            b2b_price = st.number_input("B2B Sales Price ($)", min_value=0.0, value=0.0, step=0.01)
        with col12:
            b2c_price = st.number_input("B2C/Amazon Price ($)", min_value=0.0, value=0.0, step=0.01)
        
        col13, col14, col15 = st.columns(3)
        with col13:
            b2b_returns = st.number_input("B2B Returns", min_value=0, value=0)
        with col14:
            amazon_returns = st.number_input("Amazon Returns", min_value=0, value=0)
        with col15:
            st.write("")  # Spacer
        
        # Feedback fields
        st.markdown("#### Feedback & Context")
        col16, col17 = st.columns(2)
        with col16:
            b2b_feedback = st.text_area("B2B Customer Feedback", placeholder="Enter any B2B-specific feedback...")
        with col17:
            amazon_feedback = st.text_area("Amazon Customer Feedback", placeholder="Enter Amazon reviews/feedback...")
        
        # Safety and manual context
        manual_context = st.text_area(
            "Additional Context (Manuals, Known Issues, etc.)",
            placeholder="Enter any additional context that might help the AI analysis...",
            help="Include any relevant background information, manual excerpts, or known issues"
        )
        
        col18, col19 = st.columns(2)
        with col18:
            safety_risk = st.checkbox("⚠️ Potential Safety Risk?", help="Check if there's any indication of safety concern or injury")
        with col19:
            is_new_product = st.checkbox("🆕 New Product (< 90 days)?", help="Check if this is a recently launched product")
        
        submitted = st.form_submit_button("🔍 Run AI Screening", type="primary", use_container_width=True)
    
    if submitted:
        if not product_name or not product_sku or category == 'Select...':
            st.error("Please fill in all required fields (Product Name, SKU, Category)")
            return
        
        # Build entry dictionary
        entry = {
            'SKU': product_sku,
            'Name': product_name,
            'Category': category,
            'Sold': units_sold,
            'Returned': units_returned,
            'Return_Rate': units_returned / units_sold if units_sold > 0 else 0,
            'Landed Cost': unit_cost,
            'Complaint_Text': complaint_reasons,
            'Manual_Context': manual_context,
            'Safety Risk': 'Yes' if safety_risk else 'No',
            'Is_New_Product': is_new_product,
            'Primary_Channel': primary_channel if primary_channel != 'Select...' else '',
            'B2B_Feedback': b2b_feedback,
            'Amazon_Feedback': amazon_feedback,
            'B2B_Returns': b2b_returns,
            'Amazon_Returns': amazon_returns,
            'Date_Range': f"{date_start} to {date_end}"
        }
        
        # Create DataFrame and process
        df_input = pd.DataFrame([entry])
        process_screening(df_input)


def render_pro_mode():
    """Render Pro mode - mass upload analysis"""
    
    st.info("🚀 **Pro Mode**: Upload CSV/Excel for mass analysis (up to 500+ products)")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Product Data",
        type=['csv', 'xlsx', 'xls'],
        help="Upload file with columns: SKU, Category, Sold, Returned (required). Optional: Name, Landed Cost, Complaint_Text, etc.",
        key="qc_pro_upload"
    )
    
    if uploaded_file:
        try:
            # Load file
            if uploaded_file.name.endswith('.csv'):
                df_input = pd.read_csv(uploaded_file)
            else:
                df_input = pd.read_excel(uploaded_file)
            
            log_process(f"Loaded file: {uploaded_file.name} ({len(df_input)} rows)")
            
            # Validate
            validation = DataValidation.validate_upload(df_input)
            
            # Show validation report
            with st.expander("📋 Data Validation Report", expanded=True):
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Rows", validation['total_rows'])
                col2.metric("Columns Found", len(validation['found_cols']))
                col3.metric("Validation", "✅ Passed" if validation['valid'] else "❌ Issues Found")
                
                if validation['warnings']:
                    for warning in validation['warnings']:
                        st.warning(warning)
                
                if validation['numeric_issues']:
                    st.markdown("**Numeric Issues:**")
                    for issue in validation['numeric_issues']:
                        st.caption(f"- {issue['column']}: {issue['issue']}")
                
                if validation['column_mapping']:
                    st.markdown("**Column Mapping:**")
                    mapping_df = pd.DataFrame([
                        {'Standard': k, 'Your Column': v}
                        for k, v in validation['column_mapping'].items()
                    ])
                    st.dataframe(mapping_df, hide_index=True)
            
            if not validation['valid']:
                st.error("Please fix the validation issues before proceeding.")
                return
            
            # Preview data
            st.markdown("#### Data Preview")
            st.dataframe(df_input.head(10), use_container_width=True)
            
            # Statistical analysis suggestion
            st.markdown("#### 📊 Statistical Analysis")
            
            # Prepare numeric columns for suggestion
            numeric_cols = []
            for col in ['Return_Rate', 'Landed Cost', 'Sold', 'Returned']:
                mapped_col = validation['column_mapping'].get(col)
                if mapped_col and mapped_col in df_input.columns:
                    numeric_cols.append(col)
            
            # Get AI suggestion
            suggestion = QualityStatistics.suggest_analysis_type(
                df_input.rename(columns={v: k for k, v in validation['column_mapping'].items()}),
                [c for c in numeric_cols if c not in ['Sold', 'Returned']]
            )
            st.session_state.statistical_suggestion = suggestion
            
            # Display suggestion
            with st.expander("🤖 AI Analysis Suggestion", expanded=True):
                st.markdown(f"**Recommended:** {suggestion['recommended']}")
                st.markdown(f"**Reason:** {suggestion['reason']}")
                
                if suggestion['alternatives']:
                    st.markdown("**Alternatives:**")
                    for alt in suggestion['alternatives']:
                        st.caption(f"- {alt['test']}: {alt['when']}")
                
                if suggestion['warnings']:
                    for warning in suggestion['warnings']:
                        st.warning(warning)
            
            # User override
            col1, col2 = st.columns(2)
            with col1:
                analysis_type = st.selectbox(
                    "Select Analysis Type",
                    ["Auto (AI Suggested)", "ANOVA", "MANOVA", "Kruskal-Wallis", "Descriptive Only"],
                    index=0
                )
            with col2:
                include_claude_review = st.checkbox(
                    "Request Claude AI Review",
                    help="Get additional analysis from Claude AI (slower but more thorough)"
                )
            
            # Run analysis button
            if st.button("🔍 Run Full Screening Analysis", type="primary", use_container_width=True):
                # Rename columns to standard names
                df_renamed = df_input.rename(columns={v: k for k, v in validation['column_mapping'].items()})
                
                # Determine analysis type
                if analysis_type == "Auto (AI Suggested)":
                    actual_analysis = suggestion['recommended']
                else:
                    actual_analysis = analysis_type
                
                process_screening(df_renamed, analysis_type=actual_analysis, include_claude=include_claude_review)
        
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            log_process(f"File load error: {e}", 'error')


def process_screening(df: pd.DataFrame, analysis_type: str = "ANOVA", include_claude: bool = False):
    """Process screening analysis"""
    
    log_process("Starting Quality Case Screening Analysis")
    progress = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Data Preparation
        status_text.text("Step 1/6: Preparing data...")
        log_process("Preparing data...")
        
        # Parse numeric columns
        if 'Sold' in df.columns:
            df['Sold'] = parse_numeric(df['Sold'])
        else:
            df['Sold'] = 100  # Default
        
        if 'Returned' in df.columns:
            df['Returned'] = parse_numeric(df['Returned'])
        else:
            df['Returned'] = 0
        
        if 'Return_Rate' not in df.columns or df['Return_Rate'].isna().all():
            df['Return_Rate'] = df['Returned'] / df['Sold'].replace(0, 1)
        else:
            df['Return_Rate'] = parse_percentage(df['Return_Rate']).abs()
        
        if 'Landed Cost' in df.columns:
            df['Landed Cost'] = parse_numeric(df['Landed Cost'])
        else:
            df['Landed Cost'] = 0
        
        progress.progress(15)
        
        # Step 2: Get thresholds
        status_text.text("Step 2/6: Applying thresholds...")
        log_process("Applying category thresholds...")
        
        active_thresholds = st.session_state.threshold_profiles.get(
            st.session_state.active_profile, 
            DEFAULT_CATEGORY_THRESHOLDS
        )
        
        # Apply fuzzy matching if user threshold data available
        if st.session_state.user_threshold_data is not None:
            df['Category_Threshold'] = df.apply(
                lambda row: fuzzy_match_category(
                    str(row.get('Name', '')),
                    str(row.get('Category', '')),
                    st.session_state.user_threshold_data
                )[1],
                axis=1
            )
        else:
            df['Category_Threshold'] = df['Category'].apply(
                lambda x: active_thresholds.get(x, active_thresholds.get('All Others', 0.10))
            )
        
        progress.progress(30)
        
        # Step 3: Statistical Analysis
        status_text.text("Step 3/6: Running statistical analysis...")
        log_process(f"Running {analysis_type} analysis...")
        
        if analysis_type == "MANOVA" and len(df) > 10:
            metrics = ['Return_Rate']
            if 'Landed Cost' in df.columns and df['Landed Cost'].sum() > 0:
                metrics.append('Landed Cost')
            
            manova_result = QualityStatistics.perform_manova(df, 'Category', metrics)
            st.session_state.manova_result = manova_result
            log_process(f"MANOVA p-value: {manova_result.p_value:.4f}")
        
        elif analysis_type in ["ANOVA", "Auto"] and len(df) > 5:
            anova_result = QualityStatistics.perform_anova(df, 'Category', 'Return_Rate')
            st.session_state.anova_result = anova_result
            log_process(f"ANOVA F={anova_result.statistic:.2f}, p={anova_result.p_value:.4f}")
        
        elif analysis_type == "Kruskal-Wallis" and len(df) > 5:
            kw_result = QualityStatistics.perform_kruskal_wallis(df, 'Category', 'Return_Rate')
            st.session_state.anova_result = kw_result
            log_process(f"Kruskal-Wallis H={kw_result.statistic:.2f}, p={kw_result.p_value:.4f}")
        
        progress.progress(45)
        
        # Step 4: Calculate Risk Scores and Determine Actions
        status_text.text("Step 4/6: Calculating risk scores...")
        log_process("Calculating weighted risk scores...")
        
        results = []
        for idx, row in df.iterrows():
            # Risk score
            risk_score, risk_components = RiskScoring.calculate_risk_score(
                return_rate=row['Return_Rate'],
                category_threshold=row['Category_Threshold'],
                landed_cost=row.get('Landed Cost', 0),
                safety_risk=str(row.get('Safety Risk', '')).lower() in ['yes', 'true', '1'],
                complaint_count=len(str(row.get('Complaint_Text', '')).split(',')) if row.get('Complaint_Text') else 0,
                units_sold=row['Sold']
            )
            
            # SPC Signal
            cat_std = df[df['Category'] == row['Category']]['Return_Rate'].std()
            cat_mean = df[df['Category'] == row['Category']]['Return_Rate'].mean()
            spc_result = SPCAnalysis.detect_signal(row['Return_Rate'], cat_mean, cat_std if cat_std > 0 else 0.01)
            
            # Determine action
            action, triggers = ActionDetermination.determine_action(
                return_rate=row['Return_Rate'],
                category_threshold=row['Category_Threshold'],
                landed_cost=row.get('Landed Cost', 0),
                safety_risk=str(row.get('Safety Risk', '')).lower() in ['yes', 'true', '1'],
                is_new_product=row.get('Is_New_Product', False),
                complaint_count=len(str(row.get('Complaint_Text', '')).split(',')) if row.get('Complaint_Text') else 0,
                risk_score=risk_score
            )
            
            # Build result row
            result_row = row.to_dict()
            result_row.update({
                'Risk_Score': risk_score,
                'Risk_Components': json.dumps(risk_components),
                'SPC_Signal': spc_result.signal_type,
                'SPC_Z_Score': spc_result.z_score,
                'Action': action,
                'Triggers': '; '.join(triggers) if triggers else 'None'
            })
            results.append(result_row)
        
        results_df = pd.DataFrame(results)
        progress.progress(60)
        
        # Step 5: AI Analysis
        status_text.text("Step 5/6: Running AI analysis...")
        log_process("Running AI-powered analysis...")
        
        analyzer = get_ai_analyzer()
        
        # AI analysis for high-risk items
        high_risk_items = results_df[results_df['Risk_Score'] >= 50]
        
        if len(high_risk_items) > 0 and analyzer:
            ai_recommendations = []
            for idx, row in high_risk_items.head(10).iterrows():  # Limit to top 10
                prompt = f"""Analyze this medical device quality issue:
Product: {row.get('Name', row.get('SKU', 'Unknown'))} (SKU: {row.get('SKU', 'N/A')})
Category: {row.get('Category', 'Unknown')}
Return Rate: {row['Return_Rate']:.1%} (Category threshold: {row['Category_Threshold']:.1%})
Risk Score: {row['Risk_Score']:.0f}/100
Main Complaints: {row.get('Complaint_Text', 'N/A')}
Safety Concern: {row.get('Safety Risk', 'No')}

Based on ISO 13485 and FDA QSR requirements, provide:
1. Brief assessment (2-3 sentences)
2. Primary investigation area
3. Recommended immediate action"""
                
                system_prompt = "You are a medical device quality expert. Be concise and action-oriented."
                
                try:
                    recommendation = analyzer.generate_text(prompt, system_prompt, mode='chat')
                    ai_recommendations.append({
                        'SKU': row.get('SKU', 'Unknown'),
                        'AI_Recommendation': recommendation or "AI analysis unavailable"
                    })
                except Exception as e:
                    log_process(f"AI analysis error for {row.get('SKU')}: {e}", 'error')
                    ai_recommendations.append({
                        'SKU': row.get('SKU', 'Unknown'),
                        'AI_Recommendation': f"Error: {str(e)}"
                    })
            
            # Merge AI recommendations
            if ai_recommendations:
                ai_df = pd.DataFrame(ai_recommendations)
                results_df = results_df.merge(ai_df, on='SKU', how='left')
        
        progress.progress(80)
        
        # Step 6: Claude Review (if requested)
        if include_claude and st.session_state.ai_provider != AIProvider.CLAUDE:
            status_text.text("Step 6/6: Running Claude review...")
            log_process("Requesting Claude AI additional review...")
            
            try:
                claude_analyzer = EnhancedAIAnalyzer(AIProvider.CLAUDE, max_workers=3)
                
                # Get Claude's overall assessment
                summary_prompt = f"""Review this quality screening batch:
- Total products: {len(results_df)}
- Products requiring action: {len(results_df[results_df['Action'].str.contains('Escalat|Case')])}
- Highest risk score: {results_df['Risk_Score'].max():.0f}
- Categories with issues: {', '.join(results_df[results_df['Risk_Score'] > 50]['Category'].unique()[:5])}

Provide a brief executive summary and any patterns you notice."""
                
                claude_review = claude_analyzer.generate_text(
                    summary_prompt,
                    "You are a senior quality director reviewing screening results.",
                    mode='chat'
                )
                
                st.session_state.ai_chat_history.append({
                    'role': 'claude_review',
                    'content': claude_review
                })
                log_process("Claude review completed")
            except Exception as e:
                log_process(f"Claude review error: {e}", 'error')
        
        progress.progress(100)
        status_text.text("Analysis complete!")
        
        # Store results
        st.session_state.qc_results_df = results_df
        log_process(f"Analysis complete. {len(results_df)} products screened.")
        
        st.rerun()
        
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        log_process(f"Processing error: {e}", 'error')
        raise


def render_screening_results():
    """Render the screening results dashboard"""
    
    df = st.session_state.qc_results_df
    
    st.markdown("---")
    st.markdown("### 📊 Screening Results")
    
    # Summary Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total = len(df)
    escalations = len(df[df['Action'].str.contains('Escalat', na=False)])
    cases = len(df[df['Action'].str.contains('Case', na=False)])
    monitors = len(df[df['Action'].str.contains('Monitor', na=False)])
    
    col1.metric("Total Analyzed", total)
    col2.metric("Immediate Escalations", escalations, delta_color="inverse")
    col3.metric("Quality Cases", cases, delta_color="inverse")
    col4.metric("Monitor", monitors)
    
    # Statistical Results
    if st.session_state.anova_result or st.session_state.manova_result:
        with st.expander("📈 Statistical Analysis Results", expanded=True):
            if st.session_state.manova_result:
                result = st.session_state.manova_result
                st.markdown(f"**MANOVA Results**")
                col1, col2, col3 = st.columns(3)
                col1.metric("F-Statistic", f"{result.statistic:.3f}")
                col2.metric("p-value", f"{result.p_value:.4f}")
                col3.metric("Significant", "Yes ✓" if result.significant else "No")
                st.info(result.recommendation)
            
            elif st.session_state.anova_result:
                result = st.session_state.anova_result
                st.markdown(f"**{result.test_type} Results**")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("F-Statistic" if 'ANOVA' in result.test_type else "H-Statistic", 
                           f"{result.statistic:.3f}")
                col2.metric("p-value", f"{result.p_value:.4f}")
                col3.metric("Effect Size (η²)", f"{result.effect_size:.3f}" if result.effect_size else "N/A")
                col4.metric("Significant", "Yes ✓" if result.significant else "No")
                
                st.info(result.recommendation)
                
                if result.outlier_categories:
                    st.warning(f"⚠️ Outlier Categories: {', '.join(str(c) for c in result.outlier_categories)}")
    
    # Risk Heatmap
    if ALTAIR_AVAILABLE and 'Landed Cost' in df.columns and df['Landed Cost'].sum() > 0:
        with st.expander("🔥 Risk Heatmap (Return Rate vs Cost)", expanded=True):
            chart_df = df[['SKU', 'Return_Rate', 'Landed Cost', 'Risk_Score', 'Action', 'Category']].copy()
            chart_df['Return_Rate_Pct'] = chart_df['Return_Rate'] * 100
            
            chart = alt.Chart(chart_df).mark_circle(size=100).encode(
                x=alt.X('Landed Cost:Q', title='Landed Cost ($)', scale=alt.Scale(zero=False)),
                y=alt.Y('Return_Rate_Pct:Q', title='Return Rate (%)', scale=alt.Scale(zero=False)),
                color=alt.Color('Risk_Score:Q', 
                               scale=alt.Scale(scheme='redyellowgreen', reverse=True, domain=[0, 100]),
                               legend=alt.Legend(title='Risk Score')),
                size=alt.Size('Risk_Score:Q', scale=alt.Scale(range=[50, 500]), legend=None),
                tooltip=['SKU', 'Category', 'Return_Rate_Pct', 'Landed Cost', 'Risk_Score', 'Action']
            ).interactive().properties(height=400)
            
            st.altair_chart(chart, use_container_width=True)
    
    # Claude Review (if available)
    claude_reviews = [c for c in st.session_state.ai_chat_history if c.get('role') == 'claude_review']
    if claude_reviews:
        with st.expander("🤖 Claude AI Review", expanded=True):
            st.markdown(claude_reviews[-1]['content'])
    
    # Results Table
    st.markdown("#### Detailed Results")
    
    # Add color coding based on action
    def highlight_action(row):
        if 'Immediate' in str(row.get('Action', '')):
            return ['background-color: #ff4b4b'] * len(row)
        elif 'Case' in str(row.get('Action', '')):
            return ['background-color: #ffa500'] * len(row)
        elif 'Monitor' in str(row.get('Action', '')):
            return ['background-color: #ffff99'] * len(row)
        return [''] * len(row)
    
    # Select display columns
    display_cols = ['SKU', 'Name', 'Category', 'Return_Rate', 'Category_Threshold', 
                   'Landed Cost', 'Risk_Score', 'SPC_Signal', 'Action', 'Triggers']
    display_cols = [c for c in display_cols if c in df.columns]
    
    # Format display
    display_df = df[display_cols].copy()
    if 'Return_Rate' in display_df.columns:
        display_df['Return_Rate'] = display_df['Return_Rate'].apply(lambda x: f"{x:.1%}")
    if 'Category_Threshold' in display_df.columns:
        display_df['Category_Threshold'] = display_df['Category_Threshold'].apply(lambda x: f"{x:.1%}")
    if 'Risk_Score' in display_df.columns:
        display_df['Risk_Score'] = display_df['Risk_Score'].apply(lambda x: f"{x:.0f}")
    
    st.dataframe(
        display_df.style.apply(highlight_action, axis=1),
        use_container_width=True,
        height=400
    )
    
    # Action Items Section
    st.markdown("---")
    st.markdown("### 🎯 Action Items")
    
    # Filter for items needing action
    action_items = df[df['Action'].str.contains('Escalat|Case', na=False)]
    
    if len(action_items) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📧 Generate Vendor Email")
            selected_sku = st.selectbox(
                "Select SKU",
                options=action_items['SKU'].unique(),
                key="email_sku_select"
            )
            
            email_type = st.selectbox(
                "Email Type",
                ["CAPA Request", "RCA Request", "Inspection Notice"]
            )
            
            if st.button("Generate Email", key="gen_email"):
                row = action_items[action_items['SKU'] == selected_sku].iloc[0]
                
                if email_type == "CAPA Request":
                    email = VendorEmailGenerator.generate_capa_request(
                        sku=row['SKU'],
                        product_name=row.get('Name', row['SKU']),
                        issue_summary=row.get('Complaint_Text', 'Quality concerns identified'),
                        return_rate=row['Return_Rate'],
                        defect_description=row.get('Triggers', ''),
                        units_affected=int(row.get('Returned', 0))
                    )
                elif email_type == "RCA Request":
                    email = VendorEmailGenerator.generate_rca_request(
                        sku=row['SKU'],
                        product_name=row.get('Name', row['SKU']),
                        defect_type=row['Action'],
                        occurrence_rate=row['Return_Rate'],
                        sample_complaints=str(row.get('Complaint_Text', '')).split(',')[:5]
                    )
                else:
                    email = VendorEmailGenerator.generate_inspection_notice(
                        sku=row['SKU'],
                        product_name=row.get('Name', row['SKU']),
                        special_focus=str(row.get('Triggers', '')).split(';')
                    )
                
                st.text_area("Generated Email", email, height=400)
                st.download_button(
                    "📥 Download Email",
                    email,
                    file_name=f"vendor_email_{selected_sku}_{datetime.now().strftime('%Y%m%d')}.txt"
                )
        
        with col2:
            st.markdown("#### 📋 Generate Investigation Plan")
            plan_sku = st.selectbox(
                "Select SKU",
                options=action_items['SKU'].unique(),
                key="plan_sku_select"
            )
            
            if st.button("Generate Plan", key="gen_plan"):
                row = action_items[action_items['SKU'] == plan_sku].iloc[0]
                
                plan = InvestigationPlanGenerator.generate_plan(
                    sku=row['SKU'],
                    product_name=row.get('Name', row['SKU']),
                    category=row.get('Category', 'Unknown'),
                    issue_type=row.get('Action', 'Quality Issue'),
                    complaint_summary=row.get('Complaint_Text', 'See triggers'),
                    return_rate=row['Return_Rate'],
                    risk_score=row['Risk_Score']
                )
                
                plan_md = InvestigationPlanGenerator.format_plan_markdown(plan)
                st.markdown(plan_md)
                
                st.download_button(
                    "📥 Download Plan",
                    plan_md,
                    file_name=f"investigation_plan_{plan_sku}_{datetime.now().strftime('%Y%m%d')}.md"
                )
    else:
        st.success("✅ No immediate action items. All products within acceptable thresholds.")
    
    # Safety Disclaimer
    st.markdown("---")
    st.warning("""
    ⚠️ **Important Safety Notice**: Any safety concern or potential/confirmed injury requires a Quality Issue 
    to be opened immediately in Odoo. This can be opened and closed same day as long as an investigation took place.
    Refer to Quality Incident Response SOP (QMS-SOP-001-9) for full procedures.
    """)
    
    # Methodology
    with st.expander("📐 Methodology & Math", expanded=False):
        st.markdown(generate_methodology_markdown())
    
    # Export
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        # Excel export
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Screening Results')
            
            # Add metadata sheet
            metadata = pd.DataFrame([
                ['Analysis Date', datetime.now().strftime('%Y-%m-%d %H:%M')],
                ['AI Provider', st.session_state.ai_provider.value],
                ['Threshold Profile', st.session_state.active_profile],
                ['Total Products', len(df)],
                ['Escalations', len(df[df['Action'].str.contains('Escalat', na=False)])],
                ['Quality Cases', len(df[df['Action'].str.contains('Case', na=False)])]
            ], columns=['Parameter', 'Value'])
            metadata.to_excel(writer, index=False, sheet_name='Metadata')
        
        output.seek(0)
        st.download_button(
            "📥 Download Full Report (Excel)",
            output.getvalue(),
            file_name=f"quality_screening_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    with col2:
        # Clear results
        if st.button("🗑️ Clear Results", type="secondary"):
            st.session_state.qc_results_df = None
            st.session_state.anova_result = None
            st.session_state.manova_result = None
            st.session_state.ai_chat_history = []
            st.rerun()


# --- Interactive Help Guide ---
def render_help_guide():
    """Render interactive help guide"""
    with st.expander("📚 Interactive Help Guide", expanded=False):
        st.markdown("""
        ### Quality Case Screening - Quick Start Guide
        
        #### Lite Mode (1-5 Products)
        1. Fill in **required fields**: Product Name, SKU, Category, Sales, Returns
        2. Add **complaint reasons** (comma-separated)
        3. Check **Safety Risk** if any safety concerns exist
        4. Click **Run AI Screening**
        
        #### Pro Mode (Mass Analysis)
        1. **Upload** your CSV/Excel file
        2. Review the **Data Validation Report**
        3. Check the **AI Analysis Suggestion**
        4. Optionally select a different analysis type
        5. Click **Run Full Screening Analysis**
        
        #### Understanding Results
        - **Risk Score**: 0-100 composite score (higher = more urgent)
        - **SPC Signal**: Statistical process control status
        - **Action**: Recommended next step based on SOPs
        
        #### Color Coding
        - 🔴 **Red**: Immediate escalation required
        - 🟠 **Orange**: Open Quality Case
        - 🟡 **Yellow**: Monitor closely
        - ⬜ **White**: No action required
        
        #### Required CSV Columns
        - `SKU` or `Product_SKU`
        - `Category`
        - `Sold` or `Units_Sold`
        - `Returned` or `Units_Returned`
        
        #### Optional Columns
        - `Name` or `Product_Name`
        - `Landed Cost` or `Cost`
        - `Complaint_Text` or `Complaints`
        - `Return_Rate` (calculated if not provided)
        """)
        
        # Example data download
        example_data = pd.DataFrame([
            {'SKU': 'MOB1027', 'Name': 'Knee Walker', 'Category': 'MOB', 'Sold': 1000, 'Returned': 120, 'Landed Cost': 85.00, 'Complaint_Text': 'Wheel squeaks, uncomfortable padding'},
            {'SKU': 'SUP1036', 'Name': 'Post Op Shoe', 'Category': 'SUP', 'Sold': 500, 'Returned': 45, 'Landed Cost': 12.00, 'Complaint_Text': 'Wrong size, poor fit'},
            {'SKU': 'LVA1004', 'Name': 'Pressure Mattress', 'Category': 'LVA', 'Sold': 800, 'Returned': 150, 'Landed Cost': 145.00, 'Complaint_Text': 'Pump failure, air leak'},
        ])
        
        csv_buffer = io.StringIO()
        example_data.to_csv(csv_buffer, index=False)
        
        st.download_button(
            "📥 Download Example Data",
            csv_buffer.getvalue(),
            file_name="example_screening_data.csv",
            mime="text/csv"
        )


# --- MAIN APP ---

def main():
    initialize_session_state()
    inject_custom_css()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">VIVE HEALTH QUALITY SUITE</h1>
        <p style="color: white; margin: 0.5rem 0;">AI-Powered Returns Analysis & Quality Screening (v19.0)</p>
    </div>
    """, unsafe_allow_html=True)

    if not AI_AVAILABLE:
        st.error("❌ AI Modules Missing. Please check deployment.")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Tab 1 & 2 AI Provider (original behavior)
        provider_t12 = st.selectbox(
            "🤖 AI Provider (Tab 1 & 2)",
            options=['Fastest (Claude Haiku)', 'OpenAI GPT-3.5', 'Claude Sonnet', 'Both (Consensus)'],
            index=0
        )
        
        # Map to enum for tabs 1 & 2
        provider_map_t12 = {
            'Fastest (Claude Haiku)': AIProvider.FASTEST,
            'OpenAI GPT-3.5': AIProvider.OPENAI,
            'Claude Sonnet': AIProvider.CLAUDE,
            'Both (Consensus)': AIProvider.BOTH
        }
        
        # API Health Check
        render_api_health_check()
        
        # Help guide
        render_help_guide()

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Return Categorizer", "📑 B2B Report Generator", "🧪 Quality Screening"])
    
    # --- TAB 1: Categorizer (PRESERVED) ---
    with tab1:
        # Use Tab 1/2 provider
        st.session_state.ai_provider = provider_map_t12[provider_t12]
        
        st.markdown("### 📁 Return Categorization (Column I → K)")
        st.info("Upload file with **Column I** (Complaint) and **Column B** (SKU). Output will be in **Column K**.")
        
        file = st.file_uploader("Upload Data", type=['csv', 'xlsx'], key="t1_up")
        if file:
            df, mapping = process_file_preserve_structure(file.read(), file.name)
            if df is not None:
                st.session_state.column_mapping = mapping
                st.write(f"Loaded {len(df)} rows.")
                if st.button("Start Categorization"):
                    analyzer = get_ai_analyzer()
                    with st.spinner("Processing..."):
                        res = process_in_chunks(df, analyzer, mapping)
                        st.session_state.categorized_data = res
                        st.session_state.processing_complete = True
                        generate_statistics(res, mapping)
                        
                        st.session_state.export_data = export_with_column_k(res)
                        st.session_state.export_filename = f"Categorized_{datetime.now().strftime('%Y%m%d')}.xlsx"
                        st.rerun()

        if st.session_state.processing_complete and st.session_state.categorized_data is not None:
            display_results_dashboard(st.session_state.categorized_data, st.session_state.column_mapping)
            if st.session_state.export_data:
                st.download_button("⬇️ Download Result", st.session_state.export_data, 
                                 file_name=st.session_state.export_filename, 
                                 mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # --- TAB 2: B2B Reports (PRESERVED) ---
    with tab2:
        # Use Tab 1/2 provider
        st.session_state.ai_provider = provider_map_t12[provider_t12]
        
        st.markdown("### 📑 B2B Report Automation")
        st.info("Convert Odoo exports into B2B reports with AI summaries and SKU extraction.")
        
        perf = st.select_slider("Dataset Size / Speed", options=['Small', 'Medium', 'Large'], value='Small')
        batch_map = {'Small': (10, 3), 'Medium': (25, 6), 'Large': (50, 10)}
        batch_size, workers = batch_map[perf]
        
        file_b2b = st.file_uploader("Upload Odoo Export", type=['csv', 'xlsx'], key="t2_up")
        if file_b2b:
            df_b2b = process_b2b_file(file_b2b.read(), file_b2b.name)
            if df_b2b is not None:
                st.write(f"Loaded {len(df_b2b)} tickets.")
                if st.button("Generate Report"):
                    analyzer = get_ai_analyzer(max_workers=workers)
                    with st.spinner("Generating Summaries..."):
                        final_df = generate_b2b_report(df_b2b, analyzer, batch_size)
                        st.session_state.b2b_processed_data = final_df
                        st.session_state.b2b_processing_complete = True
                        
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            final_df.to_excel(writer, index=False, sheet_name='Report')
                        st.session_state.b2b_export_data = output.getvalue()
                        st.session_state.b2b_export_filename = "B2B_Report.xlsx"
                        st.rerun()

        if st.session_state.b2b_processing_complete and st.session_state.b2b_processed_data is not None:
            st.markdown("#### Preview")
            st.dataframe(st.session_state.b2b_processed_data.head())
            st.download_button("⬇️ Download B2B Report", st.session_state.b2b_export_data,
                             file_name=st.session_state.b2b_export_filename,
                             mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # --- TAB 3: Quality Screening (REBUILT) ---
    with tab3:
        render_quality_screening_tab()


if __name__ == "__main__":
    main()
