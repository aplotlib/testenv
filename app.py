import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import io
import json
from typing import Dict, List, Any, Optional, Tuple
import time
from collections import Counter, defaultdict
import re
import os
import gc
import altair as alt
from difflib import get_close_matches

# --- Custom Modules ---
# Try to import custom modules (ensure these are in the same folder)
try:
    from enhanced_ai_analysis import (
        EnhancedAIAnalyzer, AIProvider, FBA_REASON_MAP,
        MEDICAL_DEVICE_CATEGORIES
    )
    from quality_analytics import QualityAnalytics, parse_numeric, SOP_THRESHOLDS
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
    'version': '18.0 (Unified)',
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

# AI Provider options
AI_PROVIDER_OPTIONS = {
    'Fastest (Claude Haiku)': AIProvider.FASTEST,
    'OpenAI GPT-3.5': AIProvider.OPENAI,
    'Claude Sonnet': AIProvider.CLAUDE,
    'Both (Consensus)': AIProvider.BOTH
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
        'ai_provider': AIProvider.FASTEST,
        
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
        
        # Tab 3: Quality Screening
        'qc_mode': 'Lite', # Lite or Pro
        'qc_results': None,
        'processing_log': [],
        'anova_result': None,
        'manova_result': None,
        'tukey_result': None,
        'qc_threshold_profiles': {'Default SOP': SOP_THRESHOLDS.copy()},
        'qc_active_profile': 'Default SOP',
        'qc_product_thresholds': [],
        'qc_manual_context': '',
        'qc_history_cases': None,
        'qc_validation_report': None,
        'qc_export_data': None,
        'qc_export_filename': None,
        'qc_ai_summary': None
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
                    keys_found['openai'] = val
                    os.environ['OPENAI_API_KEY'] = val
                    break
            
            # Check Claude
            for key in ['ANTHROPIC_API_KEY', 'anthropic_api_key', 'claude_api_key', 'claude']:
                if key in st.secrets:
                    val = str(st.secrets[key]).strip()
                    keys_found['claude'] = val
                    os.environ['ANTHROPIC_API_KEY'] = val
                    break
    except Exception as e:
        logger.warning(f"Error checking secrets: {e}")
    
    return keys_found

def get_ai_analyzer(max_workers=5):
    """Get or create AI analyzer instance"""
    if st.session_state.ai_analyzer is None or st.session_state.ai_analyzer.max_workers != max_workers:
        try:
            check_api_keys() # Ensure env vars are set
            st.session_state.ai_analyzer = EnhancedAIAnalyzer(st.session_state.ai_provider, max_workers=max_workers)
            logger.info(f"Created AI analyzer: {st.session_state.ai_provider.value}, Workers: {max_workers}")
        except Exception as e:
            st.error(f"Error initializing AI: {str(e)}")
    
    return st.session_state.ai_analyzer

def log_process(message: str, type: str = 'info'):
    """Adds message to the Processing Transparency Log (Tab 3)"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    entry = f"[{timestamp}] {message}"
    st.session_state.processing_log.append(entry)
    if type == 'error':
        logger.error(message)
    else:
        logger.info(message)

def ai_call_with_retry(callable_fn, *args, max_retries: int = 3, backoff: float = 2.0, **kwargs):
    """AI call wrapper with exponential backoff for rate limits."""
    for attempt in range(1, max_retries + 1):
        try:
            return callable_fn(*args, **kwargs)
        except Exception as exc:
            msg = str(exc).lower()
            if "rate limit" in msg or "429" in msg:
                wait = backoff ** attempt
                log_process(f"AI rate limited, waiting {wait:.0f}s (attempt {attempt}/{max_retries})", "error")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("AI call failed after retries.")

def build_effective_thresholds(profile_thresholds: Dict[str, float]) -> Dict[str, float]:
    """Merge SOP thresholds with profile, keeping higher threshold when conflicts exist."""
    effective = SOP_THRESHOLDS.copy()
    for key, value in profile_thresholds.items():
        if isinstance(value, (int, float)):
            effective[key] = max(effective.get(key, 0), value)
        else:
            effective[key] = value
    return effective

def parse_threshold_upload(df: pd.DataFrame) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Parse uploaded threshold overrides."""
    category_thresholds = {}
    product_thresholds = []
    if df is None or df.empty:
        return category_thresholds, product_thresholds

    normalized_cols = {col.lower().strip(): col for col in df.columns}
    cat_col = normalized_cols.get('category')
    rate_col = normalized_cols.get('return rate threshold') or normalized_cols.get('threshold') or normalized_cols.get('return_rate_threshold')
    product_col = normalized_cols.get('product name') or normalized_cols.get('product name pattern') or normalized_cols.get('product')

    if cat_col and rate_col:
        for _, row in df[[cat_col, rate_col]].dropna().iterrows():
            category_thresholds[str(row[cat_col]).strip()] = float(row[rate_col])

    if product_col and rate_col:
        for _, row in df[[product_col, rate_col]].dropna().iterrows():
            product_thresholds.append({
                'pattern': str(row[product_col]).strip().lower(),
                'threshold': float(row[rate_col])
            })

    return category_thresholds, product_thresholds

def apply_product_thresholds(product_name: str, product_thresholds: List[Dict[str, Any]]) -> Optional[float]:
    """Apply fuzzy logic to match product name patterns and return the highest threshold match."""
    if not product_name or not product_thresholds:
        return None
    matches = []
    name_lower = str(product_name).lower()
    for item in product_thresholds:
        pattern = item.get('pattern', '')
        if pattern and pattern in name_lower:
            matches.append(item.get('threshold', 0))
    return max(matches) if matches else None

def generate_example_dataset() -> bytes:
    """Generate example data for onboarding."""
    sample = pd.DataFrame([
        {
            'SKU': 'MOB1027',
            'Category': 'MOB - Walkers',
            'Sold': 1000,
            'Returned': 140,
            'Landed Cost': 120,
            'Complaint_Text': 'Walker frame cracked after two weeks. Battery charger failed.',
            'Return_Rate_30D': 0.14,
            'Return_Rate_6M': 0.09,
            'Return_Rate_12M': 0.07
        },
        {
            'SKU': 'LVA2201',
            'Category': 'LVA',
            'Sold': 500,
            'Returned': 20,
            'Landed Cost': 220,
            'Complaint_Text': 'Lift was unstable and caused a near fall.',
            'Return_Rate_30D': 0.04,
            'Return_Rate_6M': 0.03,
            'Return_Rate_12M': 0.025
        }
    ])
    output = io.BytesIO()
    sample.to_csv(output, index=False)
    return output.getvalue()

def summarize_language_trends(df: pd.DataFrame, text_col: str) -> Dict[str, int]:
    """Simple keyword frequency analysis for complaint text."""
    if text_col not in df.columns:
        return {}
    stopwords = {'the', 'and', 'with', 'that', 'this', 'from', 'after', 'when', 'was', 'were', 'have', 'has'}
    words = []
    for text in df[text_col].dropna().astype(str):
        tokens = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        words.extend([t for t in tokens if t not in stopwords])
    counts = Counter(words)
    return dict(counts.most_common(15))

def build_qc_export(df: pd.DataFrame, metadata: Dict[str, Any]) -> bytes:
    """Export QC results with metadata sheet."""
    output = io.BytesIO()
    if EXCEL_AVAILABLE:
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Screening Results')
            meta_df = pd.DataFrame(list(metadata.items()), columns=['Key', 'Value'])
            meta_df.to_excel(writer, index=False, sheet_name='Metadata')
    else:
        df.to_csv(output, index=False)
    output.seek(0)
    return output.getvalue()

# -------------------------
# TAB 1 LOGIC: Categorizer
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
        
        # Map specific columns based on requirements (B=SKU, I=Complaint, K=Category)
        # 0-indexed: B=1, I=8, K=10
        if len(cols) >= 11:
            if len(cols) > 8: column_mapping['complaint'] = cols[8] # Column I
            if len(cols) > 1: column_mapping['sku'] = cols[1] # Column B
            
            # Ensure Column K exists
            while len(df.columns) < 11:
                df[f'Column_{len(df.columns)}'] = ''
            column_mapping['category'] = df.columns[10] # Column K
            
            # Reset Column K to empty to avoid confusion
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
            
        # Process in sub-batches
        sub_batch_size = st.session_state.batch_size
        for i in range(0, len(batch_data), sub_batch_size):
            sub_batch = batch_data[i:i+sub_batch_size]
            results = analyzer.categorize_batch(sub_batch, mode='standard')
            
            for result in results:
                df.at[result['index'], category_col] = result.get('category', 'Other/Miscellaneous')
                processed_count += 1
            
            # Update Progress
            progress = processed_count / total_valid
            progress_bar.progress(progress)
            
            elapsed = time.time() - start_time
            speed = processed_count / elapsed if elapsed > 0 else 0
            
            with stats_container:
                c1, c2 = st.columns(2)
                c1.metric("Processed", f"{processed_count}/{total_valid}")
                c2.metric("Speed", f"{speed:.1f}/sec")
            
            # Force GC
            gc.collect()
            
    return df

def generate_statistics(df, column_mapping):
    """Generate summary stats for dashboard"""
    cat_col = column_mapping.get('category')
    sku_col = column_mapping.get('sku')
    
    if not cat_col: return
    
    categorized = df[df[cat_col].notna() & (df[cat_col] != '')]
    st.session_state.reason_summary = categorized[cat_col].value_counts().to_dict()
    
    # SKU Summary
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
            worksheet.set_column(10, 10, 20, fmt) # Col K
    else:
        df.to_csv(output, index=False)
    output.seek(0)
    return output.getvalue()

def display_results_dashboard(df, column_mapping):
    """Render Tab 1 Dashboard"""
    st.markdown("### 📊 Analysis Results")
    total = len(df)
    cat_col = column_mapping.get('category')
    if not cat_col or cat_col not in df.columns:
        st.warning("Category column not detected. Unable to render summary metrics.")
        return
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
# TAB 2 LOGIC: B2B Reports
# -------------------------

def extract_main_sku(text):
    """Extract 3 Caps + 4 Digits (e.g., MOB1027)"""
    if not isinstance(text, str): return None
    match = re.search(r'\b([A-Z]{3}\d{4})', text)
    return match.group(1) if match else None

def find_sku_in_row(row):
    """Heuristic to find SKU in row"""
    # Check specific columns
    for col in ['Main SKU', 'Main SKU/Display Name', 'SKU', 'Product', 'Internal Reference']:
        if col in row.index and pd.notna(row[col]):
            sku = extract_main_sku(str(row[col]))
            if sku: return sku
    
    # Check subject/description
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
            'details': str(row.get(desc_col, ''))[:1000], # Truncate for AI context
            'full_description': str(row.get(desc_col, '')),
            'sku': find_sku_in_row(row)
        })
        
    # Batch AI
    results = []
    progress_bar = st.progress(0)
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        results.extend(analyzer.summarize_batch(batch))
        progress_bar.progress(min((i + batch_size) / len(items), 1.0))
        
    # Build Result DF
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
# TAB 3 LOGIC: Screening
# -------------------------

def render_quality_screening_tab():
    st.markdown("### 🧪 Quality Case Screening")
    keys = check_api_keys()
    with st.sidebar:
        st.markdown("### 🩺 API Health Check")
        openai_status = "✅ Connected" if keys.get('openai') else "⚠️ Disconnected"
        claude_status = "✅ Connected" if keys.get('claude') else "⚠️ Disconnected"
        st.markdown(f"OpenAI: {openai_status}")
        st.markdown(f"Claude: {claude_status}")

    with st.expander("🧭 Interactive Help Guide", expanded=False):
        st.markdown(
            "- **Step 1:** Choose Lite (1–5 manual entries) or Pro (upload). \n"
            "- **Step 2:** Confirm thresholds or upload overrides. \n"
            "- **Step 3:** Run screening and review heatmap, risk scores, and AI insights. \n"
            "- **Step 4:** Export the report with metadata."
        )
        st.download_button(
            "⬇️ Download Example Data",
            generate_example_dataset(),
            file_name="quality_screening_example.csv",
            mime="text/csv"
        )

    st.info(
        "⚠️ **Regulatory Reminder:** Any potential or confirmed injury must be logged as a Quality Issue (Odoo ticket) "
        "at minimum, even if it can be opened and closed the same day after investigation."
    )

    # Configuration
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        mode = st.radio("Mode", ["Lite (Manual/Small Batch)", "Pro (Mass Analysis/Upload)"], horizontal=True)
        st.session_state.qc_mode = "Lite" if "Lite" in mode else "Pro"
    with col2:
        st.session_state.qc_manual_context = st.text_area(
            "Global Manual Context (Optional)",
            value=st.session_state.qc_manual_context,
            help="Shared context applied to all SKUs in this run."
        )
    with col3:
        if keys.get('openai') or keys.get('claude'):
            st.success("AI Connected")
        else:
            st.error("AI Disconnected")

    base_categories = [key for key in SOP_THRESHOLDS.keys() if 'Critical' not in key and key != 'New_Launch_Days']

    with st.expander("🎚️ Threshold Profiles & Overrides", expanded=False):
        profile_names = list(st.session_state.qc_threshold_profiles.keys())
        profile_index = profile_names.index(st.session_state.qc_active_profile) if st.session_state.qc_active_profile in profile_names else 0
        st.session_state.qc_active_profile = st.selectbox(
            "Active Threshold Profile",
            profile_names,
            index=profile_index
        )
        active_profile = st.session_state.qc_threshold_profiles.get(st.session_state.qc_active_profile, {})
        profile_df = pd.DataFrame({
            "Category": base_categories,
            "Return Rate Threshold": [active_profile.get(cat, SOP_THRESHOLDS.get(cat, 0.1)) for cat in base_categories]
        })
        edited_df = st.data_editor(profile_df, num_rows="fixed", use_container_width=True)
        new_profile_name = st.text_input("Save as New Profile Name")
        col_save, col_new = st.columns(2)
        if col_save.button("Save Profile Updates"):
            st.session_state.qc_threshold_profiles[st.session_state.qc_active_profile] = dict(
                zip(edited_df["Category"], edited_df["Return Rate Threshold"])
            )
            st.success("Profile updated.")
        if col_new.button("Save as New Profile") and new_profile_name:
            st.session_state.qc_threshold_profiles[new_profile_name] = dict(
                zip(edited_df["Category"], edited_df["Return Rate Threshold"])
            )
            st.session_state.qc_active_profile = new_profile_name
            st.success("New profile saved.")

        uploaded_thresholds = st.file_uploader(
            "Upload Threshold Overrides (CSV/Excel)",
            type=['csv', 'xlsx'],
            key="threshold_upload"
        )
        if uploaded_thresholds:
            try:
                if uploaded_thresholds.name.endswith('.csv'):
                    overrides_df = pd.read_csv(uploaded_thresholds)
                else:
                    overrides_df = pd.read_excel(uploaded_thresholds)
                cat_overrides, prod_overrides = parse_threshold_upload(overrides_df)
                if cat_overrides:
                    active_profile.update(cat_overrides)
                    st.session_state.qc_threshold_profiles[st.session_state.qc_active_profile] = active_profile
                if prod_overrides:
                    st.session_state.qc_product_thresholds = prod_overrides
                st.success("Threshold overrides applied.")
            except Exception as e:
                st.error(f"Threshold upload error: {e}")
            if keys.get('openai') or keys.get('claude'):
                if st.button("Ask AI to Normalize Thresholds"):
                    analyzer = get_ai_analyzer()
                    prompt = (
                        "Normalize the uploaded threshold list against these categories: "
                        f"{', '.join(base_categories)}. "
                        "Return a mapping of upload categories to the closest SOP category."
                    )
                    with st.spinner("Analyzing thresholds..."):
                        resp = ai_call_with_retry(analyzer.generate_text, prompt, "You are a Quality Analyst.")
                        st.markdown(resp)

        st.markdown("**Product Name Thresholds (fuzzy match)**")
        product_thresholds_df = pd.DataFrame(st.session_state.qc_product_thresholds)
        if product_thresholds_df.empty:
            product_thresholds_df = pd.DataFrame([{'pattern': '', 'threshold': 0.0}])
        updated_product_thresholds = st.data_editor(
            product_thresholds_df,
            num_rows="dynamic",
            use_container_width=True
        )
        if st.button("Save Product Thresholds"):
            st.session_state.qc_product_thresholds = updated_product_thresholds.dropna().to_dict(orient="records")
            st.success("Product thresholds saved.")

    with st.expander("💾 State Persistence", expanded=False):
        session_payload = {
            "qc_results": st.session_state.qc_results.to_dict(orient="records") if st.session_state.qc_results is not None else None,
            "qc_threshold_profiles": st.session_state.qc_threshold_profiles,
            "qc_active_profile": st.session_state.qc_active_profile,
            "qc_product_thresholds": st.session_state.qc_product_thresholds,
            "qc_manual_context": st.session_state.qc_manual_context
        }
        st.download_button(
            "⬇️ Save Session",
            json.dumps(session_payload, indent=2),
            file_name="quality_screening_session.json",
            mime="application/json"
        )
        uploaded_session = st.file_uploader("Load Saved Session", type=['json'], key="qc_session_upload")
        if uploaded_session:
            try:
                loaded = json.loads(uploaded_session.read().decode('utf-8'))
                st.session_state.qc_threshold_profiles = loaded.get("qc_threshold_profiles", st.session_state.qc_threshold_profiles)
                st.session_state.qc_active_profile = loaded.get("qc_active_profile", st.session_state.qc_active_profile)
                st.session_state.qc_product_thresholds = loaded.get("qc_product_thresholds", st.session_state.qc_product_thresholds)
                st.session_state.qc_manual_context = loaded.get("qc_manual_context", st.session_state.qc_manual_context)
                if loaded.get("qc_results"):
                    st.session_state.qc_results = pd.DataFrame(loaded["qc_results"])
                st.success("Session restored.")
            except Exception as e:
                st.error(f"Session load error: {e}")

    st.divider()

    # INPUT
    df_input = None
    ai_review_enabled = False
    ai_severity_enabled = False

    if st.session_state.qc_mode == "Lite":
        st.info("ℹ️ Lite Mode: Analyze 1–5 products manually.")
        lite_defaults = pd.DataFrame([
            {
                'SKU': '',
                'Category': base_categories[0] if base_categories else 'All Others',
                'Sold': 1,
                'Returned': 0,
                'Landed Cost': 0.0,
                'Complaint_Text': '',
                'Manual_Context': '',
                'Safety Risk': 'No',
                'Date Range': 'Last 30 Days',
                'Primary Sales Channel': '',
                'Sales Channel Distribution': '',
                'Packaging Method': '',
                'B2B Feedback': '',
                'Amazon Feedback': '',
                'Unit Cost': 0.0,
                'B2B Sales Price': 0.0,
                'B2C/AMZ Sales Price': 0.0,
                'B2B Returns': 0,
                'Amazon Returns': 0
            }
        ])
        lite_df = st.data_editor(
            lite_defaults,
            num_rows="dynamic",
            use_container_width=True
        )
        lite_df = lite_df.head(5)
        ai_review_enabled = st.checkbox("Enable AI Screening Review (Lite)", value=False)
        ai_severity_enabled = st.checkbox("Enable AI Severity Scoring (Lite)", value=False)
        if st.button("🔍 Run Screening Analysis", type="primary"):
            df_input = lite_df[lite_df['SKU'].astype(str).str.strip() != '']

    else:  # PRO MODE
        st.info("🚀 Pro Mode: Mass analysis with statistical rigor.")
        uploaded = st.file_uploader("Upload Data (CSV/Excel)", type=['csv', 'xlsx'], key="qc_upload")
        if uploaded:
            try:
                file_size_mb = uploaded.size / (1024 * 1024)
                if uploaded.name.endswith('.csv'):
                    file_bytes = uploaded.getvalue()
                    if file_size_mb > 50:
                        log_process("Large file detected; applying chunked processing.")
                        chunks = pd.read_csv(io.BytesIO(file_bytes), chunksize=APP_CONFIG['default_chunk'])
                        df_input = pd.concat(chunks, ignore_index=True)
                    else:
                        df_input = pd.read_csv(io.BytesIO(file_bytes))
                else:
                    df_input = pd.read_excel(uploaded)

                report = QualityAnalytics.validate_upload(
                    df_input,
                    ['SKU', 'Category', 'Sold', 'Returned'],
                    numeric_cols=['Sold', 'Returned', 'Landed Cost']
                )
                st.session_state.qc_validation_report = report
                if not report['valid']:
                    st.error(f"Missing columns: {report['missing_cols']}")
                else:
                    st.success(f"Loaded {len(df_input)} rows.")
            except Exception as e:
                st.error(f"Error: {e}")

        with st.expander("🧾 Data Validation Report", expanded=False):
            report = st.session_state.qc_validation_report
            if report:
                st.write(report)
            else:
                st.info("Upload a file to generate the validation report.")

        historical_upload = st.file_uploader(
            "Upload Historical Quality Case Summaries (Optional)",
            type=['csv', 'xlsx'],
            key="qc_history_upload"
        )
        if historical_upload:
            if historical_upload.name.endswith('.csv'):
                historical_cases = pd.read_csv(historical_upload)
            else:
                historical_cases = pd.read_excel(historical_upload)
            st.session_state.qc_history_cases = historical_cases

        ai_review_enabled = st.checkbox("Enable AI Screening Review (Pro)", value=False)
        ai_severity_enabled = st.checkbox("Enable AI Severity Scoring (Pro)", value=False)

        if st.button("🔍 Run Screening Analysis", type="primary") and df_input is not None:
            df_input = df_input.copy()

    # PROCESSING
    if df_input is not None and df_input.empty:
        st.warning("Please enter at least one product before running analysis.")
    if df_input is not None and not df_input.empty:
        log_process("Started Screening Analysis")
        progress = st.progress(0)
        active_profile = st.session_state.qc_threshold_profiles.get(st.session_state.qc_active_profile, {})
        effective_thresholds = build_effective_thresholds(active_profile)

        # 1. Clean & Calc
        numeric_columns = ['Sold', 'Returned', 'Landed Cost', 'Return_Rate_30D', 'Return_Rate_6M', 'Return_Rate_12M']
        for col in numeric_columns:
            if col in df_input.columns:
                df_input[col] = parse_numeric(df_input[col])
        if 'Landed Cost' not in df_input.columns:
            df_input['Landed Cost'] = 0.0
        df_input['Return_Rate'] = np.where(
            df_input['Sold'] > 0,
            df_input['Returned'] / df_input['Sold'],
            0
        )
        if 'Manual_Context' not in df_input.columns:
            df_input['Manual_Context'] = ''
        if st.session_state.qc_manual_context:
            df_input['Manual_Context'] = df_input['Manual_Context'].fillna('') + "\n" + st.session_state.qc_manual_context
        if 'Complaint_Text' not in df_input.columns:
            df_input['Complaint_Text'] = ''
        missing_context = df_input['Complaint_Text'].astype(str).str.len() < 5
        if missing_context.any():
            st.warning("AI needs more context for some rows. Please add complaint details or manual context.")
            log_process("Missing complaint context detected; AI may request clarification.", "error")

        # 2. ANOVA + Tukey
        if len(df_input) > 5 and df_input['Category'].nunique() > 1:
            anova = QualityAnalytics.perform_anova(df_input, 'Category', 'Return_Rate')
            if anova.get('p_value') is not None:
                st.session_state.anova_result = anova
                log_process(f"ANOVA P-Value: {anova.get('p_value'):.4f}")
            if anova.get('significant'):
                st.session_state.tukey_result = QualityAnalytics.perform_tukey_hsd(df_input, 'Category', 'Return_Rate')
                log_process("Tukey HSD computed for significant ANOVA.")

        # 3. MANOVA
        metric_cols = [col for col in ['Return_Rate', 'Landed Cost'] if col in df_input.columns]
        if df_input['Category'].nunique() > 1 and len(metric_cols) > 1:
            st.session_state.manova_result = QualityAnalytics.perform_manova(df_input, 'Category', metric_cols)
            if st.session_state.manova_result.get('p_value') is not None:
                log_process(f"MANOVA P-Value: {st.session_state.manova_result['p_value']:.4f}")

        # 4. Category Stats
        category_stats = df_input.groupby('Category')['Return_Rate'].agg(['mean', 'std']).fillna(0)

        analyzer = None
        if (ai_review_enabled or ai_severity_enabled) and (keys.get('openai') or keys.get('claude')):
            analyzer = get_ai_analyzer()

        # 5. Row Logic
        results = []
        for _, row in df_input.iterrows():
            category = row.get('Category', 'All Others')
            cat_avg = category_stats.at[category, 'mean'] if category in category_stats.index else SOP_THRESHOLDS.get(category, 0.10)
            cat_std = category_stats.at[category, 'std'] if category in category_stats.index else 0.0

            applied_threshold = effective_thresholds.get(category, SOP_THRESHOLDS['All Others'])
            product_threshold = apply_product_thresholds(row.get('SKU'), st.session_state.qc_product_thresholds)
            if product_threshold is not None:
                applied_threshold = max(applied_threshold, product_threshold)

            manual_text = f"{row.get('Complaint_Text', '')} {row.get('Manual_Context', '')}".lower()
            safety_keywords = ['injury', 'injured', 'hospital', 'unsafe', 'hazard', 'burn', 'fall']
            if any(word in manual_text for word in safety_keywords):
                row['Safety Risk'] = 'Yes'

            ai_severity_score = 0.0
            if ai_severity_enabled and analyzer is not None:
                prompt = (
                    "Assess complaint severity for medical device quality screening. "
                    "Return a severity score from 0-40 and label (low/medium/high/critical). "
                    f"Complaint: {row.get('Complaint_Text', '')}"
                )
                try:
                    response = ai_call_with_retry(analyzer.generate_text, prompt, "You are a Quality Engineer.")
                    match = re.search(r'(\d{1,2})', response)
                    if match:
                        ai_severity_score = min(float(match.group(1)), 40)
                    row['AI_Severity'] = response
                except Exception as exc:
                    log_process(f"AI severity scoring failed: {exc}", "error")

            risk = QualityAnalytics.calculate_weighted_risk_score(row, cat_avg, cat_std, ai_severity_score)
            action = QualityAnalytics.determine_action(row, effective_thresholds)
            trend = QualityAnalytics.analyze_trend(row)

            history_values = [row.get('Return_Rate_6M'), row.get('Return_Rate_12M')]
            history_values = [val for val in history_values if pd.notna(val)]
            history_mean = float(np.mean(history_values)) if history_values else float(cat_avg)
            history_std = float(np.std(history_values)) if history_values else float(cat_std if cat_std > 0 else cat_avg * 0.2)
            spc = QualityAnalytics.detect_spc_signals(row, history_mean, history_std)

            row['Category_Threshold'] = applied_threshold
            row['Trend_Status'] = trend['trend']
            row['Risk_Score'] = risk
            row['Recommended_Action'] = action
            row['SPC_Signal'] = spc
            results.append(row)

        results_df = pd.DataFrame(results)

        if ai_review_enabled and analyzer is not None and len(results_df) <= 20:
            ai_actions = []
            for _, row in results_df.iterrows():
                prompt = (
                    "Using SOP thresholds, classify this SKU as one of: "
                    "Quality Case, Quality Issue, Monitor, or Dismiss. "
                    f"SKU: {row.get('SKU')}, Return Rate: {row.get('Return_Rate'):.1%}, "
                    f"Category Threshold: {row.get('Category_Threshold'):.1%}, "
                    f"Landed Cost: {row.get('Landed Cost')}, "
                    f"Safety Risk: {row.get('Safety Risk')}, "
                    f"Manual Context: {row.get('Manual_Context')}"
                )
                try:
                    response = ai_call_with_retry(analyzer.generate_text, prompt, "You are a Quality Manager.")
                    ai_actions.append(response)
                except Exception as exc:
                    log_process(f"AI screening failed: {exc}", "error")
                    ai_actions.append("AI Screening Error")
            results_df['AI_Recommendation'] = ai_actions
        elif ai_review_enabled and len(results_df) > 20:
            st.warning("AI Screening Review skipped: dataset too large for safe batch processing.")

        st.session_state.qc_results = results_df
        metadata = {
            "Analysis Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "AI Provider": st.session_state.ai_provider.value,
            "Threshold Profile": st.session_state.qc_active_profile
        }
        st.session_state.qc_export_data = build_qc_export(results_df, metadata)
        st.session_state.qc_export_filename = f"QC_Screening_{datetime.now().strftime('%Y%m%d')}.xlsx"
        progress.progress(100)
        log_process("Analysis Complete")

    # RESULTS
    if st.session_state.qc_results is not None:
        df = st.session_state.qc_results
        st.markdown("### 📊 Screening Results")

        # Heatmap
        with st.expander("🔥 Risk Heatmap", expanded=True):
            chart = alt.Chart(df).mark_circle(size=60).encode(
                x=alt.X('Landed Cost', title='Landed Cost ($)'),
                y=alt.Y('Return_Rate', title='Return Rate', axis=alt.Axis(format='%')),
                color=alt.Color('Risk_Score', scale=alt.Scale(scheme='redyellowgreen', reverse=True)),
                tooltip=['SKU', 'Category', 'Return_Rate', 'Recommended_Action', 'Trend_Status', 'SPC_Signal']
            ).interactive()
            st.altair_chart(chart, use_container_width=True)

        # Metrics
        c1, c2, c3, c4 = st.columns(4)
        escalations = len(df[df['Recommended_Action'].str.contains("Escalate")])
        c1.metric("Total Analyzed", len(df))
        c2.metric("Escalations Needed", escalations, delta_color="inverse")
        if st.session_state.anova_result:
            p = st.session_state.anova_result.get('p_value', 1.0)
            c3.metric("ANOVA p-value", f"{p:.4f}", delta="Significant" if p < 0.05 else "Not Significant")
        if st.session_state.manova_result:
            p = st.session_state.manova_result.get('p_value', 1.0)
            c4.metric("MANOVA p-value", f"{p:.4f}", delta="Significant" if p < 0.05 else "Not Significant")

        # Table
        st.dataframe(
            df.style.apply(
                lambda x: ['background-color: #ff4b4b' if 'Escalate' in str(v) else '' for v in x],
                subset=['Recommended_Action']
            )
        )

        if st.session_state.qc_export_data:
            st.download_button(
                "⬇️ Download Screening Report",
                st.session_state.qc_export_data,
                file_name=st.session_state.qc_export_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        # ANOVA + Post-Hoc Details
        with st.expander("📐 ANOVA & Post-Hoc Details", expanded=False):
            if st.session_state.anova_result:
                st.write(st.session_state.anova_result)
            if st.session_state.tukey_result:
                tukey = st.session_state.tukey_result
                if tukey.get('results') is not None:
                    st.dataframe(tukey['results'])
            else:
                st.info("No Tukey HSD results available.")

        # Language Trends
        with st.expander("🗣️ Language Trends", expanded=False):
            trends = summarize_language_trends(df, 'Complaint_Text')
            if trends:
                st.bar_chart(pd.Series(trends))
            else:
                st.info("No complaint text available for trend analysis.")

        # AI Investigation + Emails
        st.markdown("#### 🤖 AI Investigation & Vendor Communication")
        escalated = df[df['Recommended_Action'].str.contains("Escalate")]
        targets = escalated['SKU'].unique()
        if len(targets) > 0:
            sel_sku = st.selectbox("Select SKU to Investigate", targets)
            row = df[df['SKU'] == sel_sku].iloc[0]
            if st.button("Generate Draft Investigation Plan"):
                if keys.get('openai') or keys.get('claude'):
                    prompt = (
                        f"Create a draft investigation plan for SKU {sel_sku}. "
                        f"Return Rate: {row['Return_Rate']:.1%}. "
                        f"Issue: {row.get('Complaint_Text', 'N/A')}. "
                        "Suggest device areas to inspect and immediate containment steps."
                    )
                    analyzer = get_ai_analyzer()
                    with st.spinner("Thinking..."):
                        resp = ai_call_with_retry(analyzer.generate_text, prompt, "You are a Quality Engineer.")
                        st.markdown(resp)
                else:
                    st.warning("AI not connected. Unable to generate investigation plan.")

            if st.button("Generate Vendor Email Draft"):
                email_prompt = (
                    f"Draft a vendor email requesting CAPA/RCA for SKU {sel_sku}. "
                    f"Return Rate: {row['Return_Rate']:.1%}. "
                    f"Key complaint: {row.get('Complaint_Text', 'N/A')}. "
                    "Include request for timelines and corrective actions."
                )
                if keys.get('openai') or keys.get('claude'):
                    analyzer = get_ai_analyzer()
                    with st.spinner("Drafting email..."):
                        resp = ai_call_with_retry(analyzer.generate_text, email_prompt, "You are a Quality Manager.")
                        st.markdown(resp)
                else:
                    st.markdown(
                        f"Subject: CAPA/RCA Request for {sel_sku}\n\n"
                        "Hello,\n\n"
                        "We have identified an elevated return rate and quality signal on this SKU. "
                        "Please provide CAPA/RCA details and a corrective action timeline.\n\n"
                        "Thank you."
                    )
        else:
            st.info("No escalations detected.")

        # Cross-Case Correlation
        with st.expander("🔗 Cross-Case Correlation", expanded=False):
            if st.session_state.qc_history_cases is not None and (keys.get('openai') or keys.get('claude')):
                sku_list = df['SKU'].unique()
                selected = st.selectbox("Select SKU", sku_list, key="cross_case_sku")
                if st.button("Analyze Correlation"):
                    case_summaries = st.session_state.qc_history_cases.head(50).to_string(index=False)
                    prompt = (
                        f"Compare SKU {selected} complaint to historical case summaries and report correlations. "
                        f"Complaint: {df[df['SKU'] == selected]['Complaint_Text'].iloc[0]}\n"
                        f"Historical Cases:\n{case_summaries}"
                    )
                    analyzer = get_ai_analyzer()
                    with st.spinner("Analyzing..."):
                        resp = ai_call_with_retry(analyzer.generate_text, prompt, "You are a Quality Analyst.")
                        st.markdown(resp)
            else:
                st.info("Upload historical case summaries and connect AI to enable correlation.")

        # Similar Product Names
        with st.expander("🧩 Similar Product Names", expanded=False):
            sku_list = df['SKU'].unique()
            selected = st.selectbox("Select SKU", sku_list, key="similar_sku")
            names = list(df['SKU'].astype(str).unique())
            similar = get_close_matches(selected, names, n=5)
            st.write(similar)
            if keys.get('openai') or keys.get('claude'):
                if st.button("Ask AI for Similar Products"):
                    analyzer = get_ai_analyzer()
                    prompt = (
                        f"Given this SKU name: {selected}, identify similar product names from this list: "
                        f"{', '.join(names[:50])}. "
                        "Return likely related products and why they are similar."
                    )
                    with st.spinner("Identifying similar products..."):
                        resp = ai_call_with_retry(analyzer.generate_text, prompt, "You are a Quality Analyst.")
                        st.markdown(resp)

        # Processing Transparency
        with st.expander("🧾 Processing Transparency Log", expanded=False):
            if st.session_state.processing_log:
                st.code("\n".join(st.session_state.processing_log[-50:]))
            else:
                st.info("No log entries yet.")

        with st.expander("Methodology & Math"):
            st.markdown(QualityAnalytics.generate_methodology_markdown())

# --- MAIN APP ---

def main():
    initialize_session_state()
    inject_custom_css()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">VIVE HEALTH QUALITY SUITE</h1>
        <p style="color: white; margin: 0.5rem 0;">AI-Powered Returns Analysis & Reporting (v18.0)</p>
    </div>
    """, unsafe_allow_html=True)

    if not AI_AVAILABLE:
        st.error("❌ AI Modules Missing. Please check deployment.")
        st.stop()
        
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        provider = st.selectbox("🤖 AI Provider", options=list(AI_PROVIDER_OPTIONS.keys()))
        st.session_state.ai_provider = AI_PROVIDER_OPTIONS[provider]
        
        keys = check_api_keys()
        if not keys:
            st.warning("⚠️ No API Keys found in secrets.")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Return Categorizer", "📑 B2B Report Generator", "🧪 Quality Screening"])
    
    # --- TAB 1: Categorizer ---
    with tab1:
        st.markdown("### 📁 Return Categorization (Column I → K)")
        st.info("Upload file with **Column I** (Complaint) and **Column B** (SKU). Output will be in **Column K**.")
        
        file = st.file_uploader("Upload Data", type=['csv', 'xlsx'], key="t1_up")
        if file:
            df, mapping = process_file_preserve_structure(file.read(), file.name)
            if df is not None:
                st.write(f"Loaded {len(df)} rows.")
                if st.button("Start Categorization"):
                    analyzer = get_ai_analyzer()
                    with st.spinner("Processing..."):
                        res = process_in_chunks(df, analyzer, mapping)
                        st.session_state.categorized_data = res
                        st.session_state.processing_complete = True
                        generate_statistics(res, mapping)
                        
                        # Export
                        st.session_state.export_data = export_with_column_k(res)
                        st.session_state.export_filename = f"Categorized_{datetime.now().strftime('%Y%m%d')}.xlsx"
                        st.rerun()

        if st.session_state.processing_complete and st.session_state.categorized_data is not None:
            display_results_dashboard(st.session_state.categorized_data, st.session_state.column_mapping)
            if st.session_state.export_data:
                st.download_button("⬇️ Download Result", st.session_state.export_data, 
                                 file_name=st.session_state.export_filename, 
                                 mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # --- TAB 2: B2B Reports ---
    with tab2:
        st.markdown("### 📑 B2B Report Automation")
        st.info("Convert Odoo exports into B2B reports with AI summaries and SKU extraction.")
        
        # Performance Slider
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
                        
                        # Prepare Export
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

    # --- TAB 3: Quality Screening ---
    with tab3:
        render_quality_screening_tab()

if __name__ == "__main__":
    main()
