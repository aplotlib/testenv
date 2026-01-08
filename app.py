import streamlit as st
import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime, timedelta
import io
from typing import Dict, List, Any, Optional, Tuple
import time
from collections import Counter, defaultdict
import re
import os
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config with increased limits
st.set_page_config(
    page_title="Vive Health Quality & B2B Reports",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import AI module
try:
    from enhanced_ai_analysis import EnhancedAIAnalyzer, AIProvider, FBA_REASON_MAP, MEDICAL_DEVICE_CATEGORIES
    AI_AVAILABLE = True
except ImportError as e:
    AI_AVAILABLE = False
    logger.error(f"AI module not available: {str(e)}")

# Check optional imports
try:
    import xlsxwriter
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False

# App Configuration
APP_CONFIG = {
    'title': 'Vive Health Quality & B2B Report Generator',
    'version': '17.2',
    'company': 'Vive Health',
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
    'cost': '#50C878',
    'claude': '#9B59B6',
    'openai': '#00D9FF'
}

# Quality categories (For Tab 1)
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

def inject_custom_css():
    """Inject custom CSS"""
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
        --muted: {COLORS['muted']};
        --cost: {COLORS['cost']};
        --claude: {COLORS['claude']};
        --openai: {COLORS['openai']};
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
    
    .metric-card {{
        background: rgba(26, 26, 46, 0.9);
        border: 1px solid rgba(0, 217, 255, 0.3);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }}
    
    .stButton > button {{
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        color: white;
        border: none;
        padding: 0.7rem 1.5rem;
        border-radius: 6px;
        font-weight: 600;
    }}
    
    .stProgress > div > div {{
        background: linear-gradient(90deg, var(--primary), var(--accent));
    }}
    
    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        'original_data': None,  # Store original file structure
        'processed_data': None,
        'categorized_data': None,
        'ai_analyzer': None,
        'processing_complete': False,
        'reason_summary': {},
        'product_summary': {},
        'total_cost': 0.0,
        'api_calls_made': 0,
        'processing_time': 0.0,
        'ai_provider': AIProvider.FASTEST,
        'batch_size': 20,
        'chunk_size': APP_CONFIG['default_chunk'],
        'processing_errors': [],
        'total_rows_processed': 0,
        'column_mapping': {},  # Track original column positions
        'show_product_analysis': False,
        'processing_speed': 0.0,
        'auto_download_triggered': False,  # Track if auto-download was triggered
        'export_data': None,  # Store export data
        'export_filename': None,  # Store filename
        
        # B2B Report specific state
        'b2b_original_data': None,
        'b2b_processed_data': None,
        'b2b_processing_complete': False,
        'b2b_export_data': None,
        'b2b_export_filename': None,
        'b2b_perf_mode': 'Small (< 500 rows)',

        # Quality Case Screening state
        'qc_manual_entries': [],
        'qc_screened_data': None,
        'qc_export_data': None,
        'qc_export_filename': None,
        'qc_ai_review': None,
        'qc_chat_history': [],
        'qc_combined_data': None,
        'qc_stat_manual_entries': []
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def check_api_keys():
    """Check for API keys"""
    keys_found = {}
    
    try:
        if hasattr(st, 'secrets'):
            # Check OpenAI
            for key in ['OPENAI_API_KEY', 'openai_api_key', 'openai']:
                if key in st.secrets:
                    keys_found['openai'] = str(st.secrets[key]).strip()
                    break
            
            # Check Claude
            for key in ['ANTHROPIC_API_KEY', 'anthropic_api_key', 'claude_api_key', 'claude']:
                if key in st.secrets:
                    keys_found['claude'] = str(st.secrets[key]).strip()
                    break
    except Exception as e:
        logger.warning(f"Error checking secrets: {e}")
    
    return keys_found

def get_ai_analyzer(max_workers=5):
    """Get or create AI analyzer with specific workers"""
    # Always re-initialize if workers change or not exists
    if st.session_state.ai_analyzer is None or st.session_state.ai_analyzer.max_workers != max_workers:
        try:
            keys = check_api_keys()
            
            if 'openai' in keys:
                os.environ['OPENAI_API_KEY'] = keys['openai']
            if 'claude' in keys:
                os.environ['ANTHROPIC_API_KEY'] = keys['claude']
            
            st.session_state.ai_analyzer = EnhancedAIAnalyzer(st.session_state.ai_provider, max_workers=max_workers)
            logger.info(f"Created AI analyzer with provider: {st.session_state.ai_provider.value}, Workers: {max_workers}")
        except Exception as e:
            logger.error(f"Error creating AI analyzer: {e}")
            st.error(f"Error initializing AI: {str(e)}")
    
    return st.session_state.ai_analyzer

# -------------------------
# TAB 1: CATEGORIZER LOGIC
# -------------------------

def process_file_preserve_structure(file_content, filename):
    """Process file while preserving original structure (For Categorizer)"""
    try:
        # Read file
        if filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_content), dtype=str)  # Read as strings to preserve format
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(file_content), dtype=str)
        elif filename.endswith('.txt'):
            df = pd.read_csv(io.BytesIO(file_content), sep='\t', dtype=str)
        else:
            st.error(f"Unsupported file type: {filename}")
            return None, None
        
        # Store original structure
        original_df = df.copy()
        
        # Create column mapping based on your file structure
        column_mapping = {}
        
        # Check if columns are labeled A, B, C, etc. or by position
        cols = df.columns.tolist()
        
        # For your specific file structure:
        # Column B = Product Identifier Tag (SKU)
        # Column I = Categorizing/Investigator Complaint (complaint text)
        # Column K = Category (where AI results go)
        
        if len(cols) >= 11:  # Need at least columns A through K
            # Try to map by position (0-indexed)
            # Column I = index 8, Column B = index 1, Column K = index 10
            
            # Find complaint column (Column I - 9th column, index 8)
            if len(cols) > 8:
                column_mapping['complaint'] = cols[8]  # Column I
            else:
                st.error("File doesn't have enough columns. Expected complaint data in Column I.")
                return None, None
            
            # Find SKU/Product column (Column B - 2nd column, index 1)
            if len(cols) > 1:
                column_mapping['sku'] = cols[1]  # Column B
            
            # Category column (Column K - 11th column, index 10)
            if len(cols) > 10:
                column_mapping['category'] = cols[10]  # Column K
            else:
                # Add columns if needed to reach K
                while len(df.columns) < 11:
                    df[f'Column_{len(df.columns)}'] = ''
                column_mapping['category'] = df.columns[10]
        else:
            st.error("File structure not recognized. Need at least 11 columns (A-K).")
            return None, None
        
        # Ensure column K exists and is empty
        if column_mapping.get('category'):
            df[column_mapping['category']] = ''
        
        # Store mapping
        st.session_state.column_mapping = column_mapping
        
        # Validate we have complaint data
        complaint_col = column_mapping['complaint']
        valid_complaints = df[df[complaint_col].notna() & (df[complaint_col].str.strip() != '')].copy()
        
        logger.info(f"File structure: {len(df)} total rows, {len(valid_complaints)} with complaints")
        logger.info(f"Column mapping: Complaint={complaint_col} (Col I), SKU={column_mapping.get('sku')} (Col B), Category={column_mapping.get('category')} (Col K)")
        
        return df, column_mapping
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        logger.error(f"File processing error: {e}")
        return None, None

def process_in_chunks(df, analyzer, column_mapping, chunk_size=None):
    """Process large datasets in chunks"""
    if chunk_size is None:
        chunk_size = st.session_state.chunk_size
    
    complaint_col = column_mapping['complaint']
    category_col = column_mapping['category']
    
    # Get rows with complaints
    valid_indices = df[df[complaint_col].notna() & (df[complaint_col].str.strip() != '')].index
    total_valid = len(valid_indices)
    
    if total_valid == 0:
        st.warning("No valid complaints found in Column I to process")
        return df
    
    # Clear messaging about processing
    st.info(f"""
    📊 **Processing Details:**
    - Total complaints to categorize (from Column I): **{total_valid:,}**
    - Processing chunk size: **{chunk_size}** rows at a time
    - API batch size: **{st.session_state.batch_size}** items per call
    """)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    stats_container = st.container()
    
    processed_count = 0
    start_time = time.time()
    
    # Process in chunks
    for chunk_start in range(0, total_valid, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_valid)
        chunk_indices = valid_indices[chunk_start:chunk_end]
        chunk_num = (chunk_start // chunk_size) + 1
        total_chunks = (total_valid + chunk_size - 1) // chunk_size
        
        # Prepare batch data
        batch_data = []
        for idx in chunk_indices:
            complaint = str(df.at[idx, complaint_col]).strip()
            
            # Check for FBA reason code
            fba_reason = None
            if 'reason' in df.columns:
                fba_reason = str(df.at[idx, 'reason'])
            
            batch_data.append({
                'index': idx,
                'complaint': complaint,
                'fba_reason': fba_reason
            })
        
        try:
            # Process batch with smaller sub-batches
            sub_batch_size = st.session_state.batch_size
            
            for i in range(0, len(batch_data), sub_batch_size):
                sub_batch = batch_data[i:i+sub_batch_size]
                
                # Categorize sub-batch
                results = analyzer.categorize_batch(sub_batch, mode='standard')
                
                # Update dataframe
                for result in results:
                    idx = result['index']
                    category = result.get('category', 'Other/Miscellaneous')
                    df.at[idx, category_col] = category
                    
                    # Track stats
                    processed_count += 1
                
                # Update progress
                progress = processed_count / total_valid
                progress_bar.progress(progress)
                status_text.text(f"Processing chunk {chunk_num}/{total_chunks}... ({processed_count}/{total_valid})")
                
                # Display ETA and speed
                elapsed_time = time.time() - start_time
                if elapsed_time > 0:
                    processing_speed = processed_count / elapsed_time
                    remaining = total_valid - processed_count
                    eta_seconds = remaining / processing_speed if processing_speed > 0 else 0
                    
                    with stats_container:
                        st.write(f"⏱️ Speed: {processing_speed:.1f} items/sec | ETA: {eta_seconds/60:.1f} min")
        
        except Exception as e:
            st.error(f"Error processing batch: {str(e)}")
            st.session_state.processing_errors.append(str(e))
            continue
    
    # Cleanup
    progress_bar.empty()
    status_text.empty()
    
    return df

def generate_statistics(df, column_mapping):
    """Generate statistics for dashboard"""
    category_col = column_mapping['category']
    
    # Basic stats
    total_categorized = df[df[category_col].notna() & (df[category_col] != '')].shape[0]
    category_counts = df[category_col].value_counts().to_dict()
    
    # Store in session state
    st.session_state.reason_summary = category_counts
    st.session_state.total_rows_processed = total_categorized
    
def display_results_dashboard(df, column_mapping):
    """Display results in a dashboard"""
    category_col = column_mapping['category']
    
    st.markdown("### 📊 Results Dashboard")
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Processed", st.session_state.total_rows_processed)
    with col2:
        st.metric("Categories Found", len(st.session_state.reason_summary))
    with col3:
        st.metric("Processing Time", f"{st.session_state.processing_time:.1f}s")
    
    # Category breakdown
    st.markdown("#### 📈 Category Distribution")
    if st.session_state.reason_summary:
        category_df = pd.DataFrame(
            list(st.session_state.reason_summary.items()),
            columns=['Category', 'Count']
        ).sort_values('Count', ascending=False)
        
        st.dataframe(category_df, use_container_width=True)
    
    # Preview results
    st.markdown("#### 🔍 Preview Results (Top 20)")
    display_cols = [column_mapping['sku'], column_mapping['complaint'], category_col]
    display_cols = [col for col in display_cols if col in df.columns]
    st.dataframe(df[display_cols].head(20), use_container_width=True)

def export_with_column_k(df):
    """Export file with Column K filled"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

# -------------------------
# TAB 2: B2B REPORT LOGIC
# -------------------------

def extract_main_sku(text):
    """
    Extracts the Main SKU (3 Caps + 4 Digits) from text.
    Ignores variants (e.g., matches MOB1027 from MOB1027BLU).
    Strictly follows the pattern: 3 Uppercase Letters + 4 Digits.
    """
    if not isinstance(text, str):
        return None
    
    # Pattern: 3 uppercase letters followed by 4 digits.
    # We check for this pattern anywhere in the string.
    # The capturing group extracts just the Main SKU.
    match = re.search(r'\b([A-Z]{3}\d{4})', text)
    
    if match:
        return match.group(1)
    return None

def find_sku_in_row(row):
    """
    Attempts to find the Main SKU in various columns.
    Priority: 
    1. Explicit SKU columns (Product, SKU, Main SKU)
    2. Subject/Display Name
    3. Description/Body
    """
    # 1. Check explicit columns first
    sku_cols = ['Main SKU', 'Main SKU/Display Name', 'SKU', 'Product', 'Internal Reference']
    for col in sku_cols:
        if col in row.index and pd.notna(row[col]):
            sku = extract_main_sku(str(row[col]))
            if sku: return sku
    
    # 2. Check Display Name / Subject
    subject_cols = ['Display Name', 'Subject', 'Name']
    for col in subject_cols:
        if col in row.index and pd.notna(row[col]):
            sku = extract_main_sku(str(row[col]))
            if sku: return sku
            
    # 3. Check Description (Last resort)
    desc_cols = ['Description', 'Body']
    for col in desc_cols:
        if col in row.index and pd.notna(row[col]):
            sku = extract_main_sku(str(row[col]))
            if sku: return sku
            
    return "Unknown"

def strip_html(text):
    """Remove HTML tags from description for cleaner AI processing"""
    if not text or not isinstance(text, str):
        return ""
    # Simple regex to remove tags
    clean = re.compile('<.*?>')
    return re.sub(clean, ' ', text).strip()

def process_b2b_file(file_content, filename):
    """Process raw Odoo export for B2B Report"""
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_content), dtype=str)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(file_content), dtype=str)
        else:
            st.error("Unsupported file format")
            return None

        # NO FILTERING based on 'Type' as per requirements.
        # We process all rows in the upload.
        
        logger.info(f"B2B Processing: {len(df)} rows loaded (No filtering)")
        return df
        
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

def generate_b2b_report(df, analyzer, batch_size):
    """Generate the B2B report with AI summaries and Cleaned SKUs"""
    
    # Columns to use for display/AI
    display_col = 'Display Name' if 'Display Name' in df.columns else df.columns[0]
    description_col = 'Description' if 'Description' in df.columns else df.columns[1]

    # Create output dataframe
    output_data = []

    # Process in batches
    total = len(df)
    
    for i in range(0, total, batch_size):
        batch = df.iloc[i:i+batch_size]
        
        # Prepare AI input
        batch_texts = []
        for _, row in batch.iterrows():
            description = strip_html(str(row[description_col]))
            batch_texts.append(description)
        
        # Get AI summaries
        ai_results = analyzer.summarize_batch(batch_texts)
        
        # Process each row
        for idx, (_, row) in enumerate(batch.iterrows()):
            sku = find_sku_in_row(row)
            
            output_data.append({
                'Display Name': row.get(display_col, ''),
                'Description': row.get(description_col, ''),
                'SKU': sku,
                'Reason': ai_results[idx] if idx < len(ai_results) else 'Unknown'
            })
        
    return pd.DataFrame(output_data)

# -------------------------
# TAB 3: QUALITY CASE SCREENING
# -------------------------

QUALITY_CASE_COLUMNS = [
    {"name": "SKU", "required": True, "type": "text", "example": "CRD1001"},
    {"name": "Category", "required": True, "type": "text", "example": "Compression"},
    {"name": "Sold", "required": True, "type": "number", "example": "1000"},
    {"name": "Returned", "required": True, "type": "number", "example": "80"},
    {"name": "Landed Cost", "required": False, "type": "number", "example": "25.50"},
    {"name": "Retail Price", "required": False, "type": "number", "example": "75.00"},
    {"name": "Launch Date", "required": False, "type": "date", "example": "2024-06-01"},
    {"name": "Safety Risk", "required": False, "type": "flag", "example": "Yes"},
    {"name": "Zero Tolerance Component", "required": False, "type": "flag", "example": "Yes"},
    {"name": "AQL Fail", "required": False, "type": "flag", "example": "Yes"},
    {"name": "Unique Complaint Count (30d)", "required": False, "type": "number", "example": "6"},
    {"name": "Sales Units (30d)", "required": False, "type": "number", "example": "120"},
    {"name": "Sales Value (30d)", "required": False, "type": "number", "example": "4800"},
    {"name": "Input Source", "required": False, "type": "text", "example": "Manual"}
]

QUALITY_CASE_STANDARD_COLUMNS = [col['name'] for col in QUALITY_CASE_COLUMNS]

CATEGORY_BENCHMARKS = {
    "Compression": 0.05,
    "Mobility": 0.08,
    "Pain Relief": 0.07,
    "Sleep": 0.06,
    "All Others": 0.08
}

def quality_case_requirements_df():
    return pd.DataFrame(QUALITY_CASE_COLUMNS)

def build_quality_case_template():
    template_df = pd.DataFrame(columns=QUALITY_CASE_STANDARD_COLUMNS)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        template_df.to_excel(writer, index=False, sheet_name='Quality Case Template')
    output.seek(0)
    return output

def normalize_quality_case_data(df, mapping, source_label):
    normalized = pd.DataFrame()

    for standard_col in QUALITY_CASE_STANDARD_COLUMNS:
        mapped_col = mapping.get(standard_col, '')
        if mapped_col and mapped_col in df.columns:
            normalized[standard_col] = df[mapped_col]
        else:
            normalized[standard_col] = ''

    normalized['Input Source'] = source_label
    return normalized

def build_quality_case_summary(screened_df):
    action_counts = screened_df['Recommended_Action'].value_counts().to_dict()
    top_escalations = screened_df[screened_df['Recommended_Action'].str.contains('Escalate')]

    top_rows = top_escalations[['SKU', 'Category', 'Return_Rate_Display', 'Recommended_Action']].to_dict('records')
    category_alerts = (
        screened_df.groupby('Category')['Return_Rate']
        .mean()
        .sort_values(ascending=False)
        .head(5)
        .reset_index()
        .assign(Return_Rate=lambda x: (x['Return_Rate'] * 100).round(2).astype(str) + '%')
        .to_dict('records')
    )

    return {
        'total_rows': len(screened_df),
        'action_counts': action_counts,
        'top_escalations': top_rows,
        'top_categories': category_alerts
    }

def display_quality_case_dashboard(screened_df):
    st.markdown("### 📊 Quality Case Dashboard")

    action_counts = screened_df['Recommended_Action'].value_counts().to_dict()
    total_count = len(screened_df)
    escalations = action_counts.get('Escalate to Quality Case', 0) + action_counts.get('Escalate to Quality Case (Immediate)', 0)
    monitor_count = action_counts.get('Monitor', 0)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Products", total_count)
    with col2:
        st.metric("Escalations", escalations)
    with col3:
        st.metric("Monitoring", monitor_count)

    st.markdown("#### 📌 Action Breakdown")
    breakdown_df = pd.DataFrame(
        list(action_counts.items()),
        columns=['Recommended Action', 'Count']
    ).sort_values('Count', ascending=False)
    st.dataframe(breakdown_df, use_container_width=True)

    st.markdown("#### 🔎 Highest Return Rates")
    st.dataframe(
        screened_df[['SKU', 'Category', 'Return_Rate_Display', 'Recommended_Action']]
        .assign(Return_Rate_Sort=screened_df['Return_Rate'])
        .sort_values('Return_Rate_Sort', ascending=False)
        .drop(columns=['Return_Rate_Sort'])
        .head(10),
        use_container_width=True
    )

def parse_numeric(series):
    return pd.to_numeric(series, errors='coerce').fillna(0)

def load_quality_case_file(file_content, filename):
    if filename.endswith('.csv'):
        return pd.read_csv(io.BytesIO(file_content))
    if filename.endswith(('.xlsx', '.xls')):
        return pd.read_excel(io.BytesIO(file_content))
    if filename.endswith('.txt'):
        return pd.read_csv(io.BytesIO(file_content), sep='\t')
    st.error(f"Unsupported file type: {filename}")
    return None

def one_way_anova(groups):
    cleaned_groups = []
    for group in groups:
        values = np.asarray(group, dtype=float)
        values = values[~np.isnan(values)]
        if values.size > 0:
            cleaned_groups.append(values)

    if len(cleaned_groups) < 2:
        return None

    all_values = np.concatenate(cleaned_groups)
    if all_values.size == 0:
        return None

    grand_mean = np.mean(all_values)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in cleaned_groups)
    ss_within = sum(np.sum((g - np.mean(g)) ** 2) for g in cleaned_groups)

    df_between = len(cleaned_groups) - 1
    df_within = len(all_values) - len(cleaned_groups)
    if df_within <= 0:
        return None

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    if ms_within == 0:
        return None

    return {
        'f_stat': ms_between / ms_within,
        'df_between': df_between,
        'df_within': df_within,
        'group_sizes': [len(g) for g in cleaned_groups],
        'grand_mean': grand_mean
    }

def manova_wilks_lambda(groups, metric_cols):
    valid_groups = []
    for group in groups:
        values = np.asarray(group, dtype=float)
        values = values[~np.isnan(values).any(axis=1)]
        if values.shape[0] > 1:
            valid_groups.append(values)

    if len(valid_groups) < 2:
        return None

    overall = np.vstack(valid_groups)
    if overall.shape[1] != len(metric_cols):
        return None

    overall_mean = np.mean(overall, axis=0)
    w_matrix = np.zeros((len(metric_cols), len(metric_cols)))
    b_matrix = np.zeros((len(metric_cols), len(metric_cols)))

    for g in valid_groups:
        group_mean = np.mean(g, axis=0)
        centered = g - group_mean
        w_matrix += centered.T @ centered
        mean_diff = (group_mean - overall_mean).reshape(-1, 1)
        b_matrix += len(g) * (mean_diff @ mean_diff.T)

    try:
        det_w = np.linalg.det(w_matrix)
        det_t = np.linalg.det(w_matrix + b_matrix)
    except np.linalg.LinAlgError:
        return None

    if det_t == 0:
        return None

    return {
        'wilks_lambda': det_w / det_t,
        'within_matrix': w_matrix,
        'between_matrix': b_matrix,
        'group_sizes': [len(g) for g in valid_groups]
    }

def compute_quality_case_screening(df, config):
    data = df.copy()
    sold_col = config['sold_col']
    returned_col = config['returned_col']
    category_col = config['category_col']

    data['Sold'] = parse_numeric(data[sold_col])
    data['Returned'] = parse_numeric(data[returned_col])
    data['Return_Rate'] = np.where(data['Sold'] > 0, data['Returned'] / data['Sold'], 0)

    category_stats = data.groupby(category_col)['Return_Rate'].agg(['mean', 'std']).reset_index()
    category_stats.rename(columns={'mean': 'Cat_Avg', 'std': 'Cat_Std'}, inplace=True)
    category_stats['Cat_Std'] = category_stats['Cat_Std'].fillna(0)

    data = data.merge(category_stats, on=category_col, how='left')

    if config['use_benchmarks']:
        data['Benchmark_Avg'] = data[category_col].map(CATEGORY_BENCHMARKS)
        data['Benchmark_Avg'] = data['Benchmark_Avg'].fillna(CATEGORY_BENCHMARKS['All Others'])
        data['Cat_Avg'] = data['Benchmark_Avg']

    immediate = pd.Series(False, index=data.index)
    if config['safety_col']:
        immediate |= data[config['safety_col']].astype(str).str.lower().isin(['yes', 'true', '1', 'y'])
    if config['zero_tolerance_col']:
        immediate |= data[config['zero_tolerance_col']].astype(str).str.lower().isin(['yes', 'true', '1', 'y'])
    if config['landed_cost_col']:
        immediate |= parse_numeric(data[config['landed_cost_col']]) >= config['landed_cost_threshold']
    if config['retail_price_col']:
        immediate |= parse_numeric(data[config['retail_price_col']]) >= config['retail_price_threshold']
    if config['launch_date_col']:
        launch_dates = pd.to_datetime(data[config['launch_date_col']], errors='coerce', utc=True)
        now = pd.Timestamp.now(tz="UTC")
        recent = launch_dates.notna() & ((now - launch_dates).dt.days <= config['launch_days'])
        immediate |= recent
    if config['aql_fail_col']:
        immediate |= data[config['aql_fail_col']].astype(str).str.lower().isin(['yes', 'true', '1', 'y'])

    condition_critical = data['Return_Rate'] >= config['return_rate_cap']
    condition_relative = data['Return_Rate'] > (data['Cat_Avg'] * config['relative_threshold'])
    condition_outlier = data['Return_Rate'] > (data['Cat_Avg'] + data['Cat_Std'])
    condition_sop_delta = data['Return_Rate'] > (data['Cat_Avg'] + config['cat_avg_delta'])

    qualitative = pd.Series(False, index=data.index)
    if config['complaint_count_col']:
        complaints = parse_numeric(data[config['complaint_count_col']])
        sales_units = parse_numeric(data[config['sales_units_30d_col']]) if config['sales_units_30d_col'] else 0
        sales_value = parse_numeric(data[config['sales_value_30d_col']]) if config['sales_value_30d_col'] else 0
        low_impact = (sales_units > 0) & ((complaints / sales_units) < 0.05) & (sales_value < 1000)
        qualitative = (complaints >= 3) & ~low_impact

    low_volume = data['Sold'] < config['low_volume_cutoff']

    action = np.select(
        [
            immediate,
            condition_critical | condition_relative | condition_outlier | condition_sop_delta | qualitative,
            low_volume
        ],
        [
            'Escalate to Quality Case (Immediate)',
            'Escalate to Quality Case',
            'Dismiss (Low Volume)'
        ],
        default='Monitor'
    )

    review_date = pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=config['review_days'])
    data['Recommended_Action'] = action
    data['Review_By'] = np.where(data['Recommended_Action'] == 'Monitor', review_date.date().isoformat(), '')

    data['Return_Rate_Display'] = (data['Return_Rate'] * 100).round(2).astype(str) + '%'
    data['Cat_Avg_Display'] = (data['Cat_Avg'] * 100).round(2).astype(str) + '%'

    return data

def main():
    """Main application function"""
    initialize_session_state()
    inject_custom_css()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">VIVE HEALTH QUALITY SUITE</h1>
        <p style="color: white; margin: 0.5rem 0; font-size: 1.1em;">
            AI-Powered Returns Analysis & Reporting
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check AI status
    if not AI_AVAILABLE:
        st.error("❌ AI Module not available. Please check enhanced_ai_analysis.py")
        st.stop()
    
    keys = check_api_keys()
    if not keys:
        st.error("❌ No API keys found. Please add API keys to Streamlit secrets.")
        st.stop()

    # Sidebar Configuration
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        provider = st.selectbox("🤖 AI Provider", options=list(AI_PROVIDER_OPTIONS.keys()))
        st.session_state.ai_provider = AI_PROVIDER_OPTIONS[provider]
        
        st.markdown("---")
        st.caption(f"Version {APP_CONFIG['version']}")

    # TABS
    tab1, tab2, tab3 = st.tabs(["📊 Return Categorizer", "📑 B2B Report Generator", "🧪 Quality Case Screening"])

    # -------------------------
    # TAB 1: Original Categorizer
    # -------------------------
    with tab1:
        st.markdown("### 📁 Return Categorization (Column I → K)")
        st.markdown("""
        <div style="background: rgba(255, 183, 0, 0.1); border: 1px solid var(--accent); 
                    border-radius: 8px; padding: 0.8rem; margin-bottom: 1rem;">
            <strong>📌 Goal:</strong> Categorize complaints into standardized Quality Categories.
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Upload Return Data (Excel/CSV)", type=['csv', 'xlsx', 'xls', 'txt'], key="tab1_uploader")
        
        if uploaded_file:
            with st.spinner(f"Reading {uploaded_file.name}..."):
                file_content = uploaded_file.read()
                df, column_mapping = process_file_preserve_structure(file_content, uploaded_file.name)
            
            if df is not None and column_mapping:
                # Show file info
                valid_complaints = df[df[column_mapping['complaint']].notna() & (df[column_mapping['complaint']].str.strip() != '')].shape[0]
                st.info(f"Found {valid_complaints:,} complaints to categorize.")
                
                if st.button("🚀 Start Categorization", type="primary"):
                    analyzer = get_ai_analyzer()
                    with st.spinner("Categorizing..."):
                        categorized_df = process_in_chunks(df, analyzer, column_mapping)
                        st.session_state.categorized_data = categorized_df
                        st.session_state.processing_complete = True
                        generate_statistics(categorized_df, column_mapping)
                        
                        # Export
                        st.session_state.export_data = export_with_column_k(categorized_df)
                        st.session_state.export_filename = f"categorized_{datetime.now().strftime('%Y%m%d')}.xlsx"
                        st.rerun()
        
        # Results Display (Tab 1)
        if st.session_state.processing_complete and st.session_state.categorized_data is not None:
            display_results_dashboard(st.session_state.categorized_data, st.session_state.column_mapping)
            
            if st.session_state.export_data:
                st.download_button(
                    label="⬇️ Download Categorized File",
                    data=st.session_state.export_data,
                    file_name=st.session_state.export_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary",
                    use_container_width=True
                )

    # -------------------------
    # TAB 2: B2B Report Generator
    # -------------------------
    with tab2:
        st.markdown("### 📑 B2B Report Automation")
        st.markdown("""
        <div style="background: rgba(0, 217, 255, 0.1); border: 1px solid var(--primary); 
                    border-radius: 8px; padding: 0.8rem; margin-bottom: 1rem;">
            <strong>📌 Goal:</strong> Convert raw Odoo Helpdesk export into a compliant B2B Report.
            <ul style="margin-bottom:0;">
                <li><strong>Format:</strong> Matches standard B2B Report columns (Display Name, Description, SKU, Reason)</li>
                <li><strong>SKU Logic:</strong> Auto-extracts Main SKU (e.g., <code>MOB1027</code>)</li>
                <li><strong>AI Summary:</strong> Generates detailed Reason summaries for every ticket.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Performance / File Size Selection
        st.markdown("#### ⚙️ Data Volume / Processing Speed")
        perf_mode = st.select_slider(
            "Select Dataset Size to optimize API performance:",
            options=['Small (< 500 rows)', 'Medium (500-2,000 rows)', 'Large (2,000+ rows)'],
            value=st.session_state.b2b_perf_mode,
            key='perf_selector'
        )
        st.session_state.b2b_perf_mode = perf_mode
        
        # Map selection to performance settings
        if perf_mode == 'Small (< 500 rows)':
            batch_size = 10
            max_workers = 3
            st.caption("Settings: Conservative batching for max reliability.")
        elif perf_mode == 'Medium (500-2,000 rows)':
            batch_size = 25
            max_workers = 6
            st.caption("Settings: Balanced speed and concurrency.")
        else: # Large
            batch_size = 50
            max_workers = 10
            st.caption("Settings: Aggressive parallel processing for high volume.")

        st.divider()
        
        b2b_file = st.file_uploader("Upload Odoo Export (CSV/Excel)", type=['csv', 'xlsx'], key="b2b_uploader")
        
        if b2b_file:
            # 1. Read & Preview
            b2b_df = process_b2b_file(b2b_file.read(), b2b_file.name)
            
            if b2b_df is not None:
                st.markdown(f"**Total Tickets Found:** {len(b2b_df):,}")
                
                # 2. Process Button
                if st.button("⚡ Generate B2B Report", type="primary"):
                    # Update analyzer with new worker settings based on user choice
                    analyzer = get_ai_analyzer(max_workers=max_workers)
                    
                    with st.spinner("Running AI Analysis & SKU Extraction..."):
                        # Run the B2B pipeline
                        final_b2b = generate_b2b_report(b2b_df, analyzer, batch_size)
                        
                        # Save to session
                        st.session_state.b2b_processed_data = final_b2b
                        st.session_state.b2b_processing_complete = True
                        
                        # Prepare Download
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            final_b2b.to_excel(writer, index=False, sheet_name='B2B Report')
                            
                            # Formatting
                            workbook = writer.book
                            worksheet = writer.sheets['B2B Report']
                            
                            # Add simple formatting
                            header_fmt = workbook.add_format({'bold': True, 'bg_color': '#00D9FF', 'font_color': 'white'})
                            for col_num, value in enumerate(final_b2b.columns.values):
                                worksheet.write(0, col_num, value, header_fmt)
                                worksheet.set_column(col_num, col_num, 30) # Wider columns for description

                        st.session_state.b2b_export_data = output.getvalue()
                        st.session_state.b2b_export_filename = f"B2B_Report_{datetime.now().strftime('%Y-%m-%d')}.xlsx"
                        
                        st.rerun()

        # 3. B2B Dashboard Results
        if st.session_state.b2b_processing_complete and st.session_state.b2b_processed_data is not None:
            df_res = st.session_state.b2b_processed_data
            
            st.markdown("### 🏁 Report Dashboard")
            
            # Dashboard Metrics
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Total Processed", len(df_res))
            with c2:
                sku_found_count = len(df_res[df_res['SKU'] != 'Unknown'])
                st.metric("SKUs Identified", f"{sku_found_count}", delta=f"{sku_found_count/len(df_res)*100:.1f}% coverage")
            with c3:
                unique_skus = df_res[df_res['SKU'] != 'Unknown']['SKU'].nunique()
                st.metric("Unique Products", unique_skus)
            
            # Preview Table
            st.markdown("#### Preview (Top 10)")
            st.dataframe(df_res.head(10), use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="⬇️ Download B2B Report (.xlsx)",
                    data=st.session_state.b2b_export_data,
                    file_name=st.session_state.b2b_export_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary",
                    use_container_width=True
                )
            with col2:
                if st.button("🔄 Clear / Start Over", use_container_width=True):
                    st.session_state.b2b_processed_data = None
                    st.session_state.b2b_processing_complete = False
                    st.rerun()

    # -------------------------
    # TAB 3: Quality Case Screening
    # -------------------------
    with tab3:
        st.markdown("### 🧪 Quality Case Screening & Escalation")
        st.markdown("""
        <div style="background: rgba(255, 0, 110, 0.1); border: 1px solid var(--secondary); 
                    border-radius: 8px; padding: 0.8rem; margin-bottom: 1rem;">
            <strong>📌 Goal:</strong> Screen SKUs for Quality Case escalation using SOP thresholds,
            ANOVA/MANOVA checks, and calendar-based monitoring.
        </div>
        """, unsafe_allow_html=True)

        intake_tab, thresholds_tab, results_tab = st.tabs(
            ["📥 Intake", "⚙️ Thresholds", "📊 Results & Stats"]
        )

        with intake_tab:
            with st.expander("📋 Excel Column Requirements", expanded=False):
                st.dataframe(quality_case_requirements_df(), use_container_width=True)
            st.download_button(
                label="⬇️ Download Quality Case Template",
                data=build_quality_case_template(),
                file_name="quality_case_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

            st.divider()

            st.markdown("#### ✍️ Manual Entry (One Product at a Time)")
            with st.form("qc_manual_entry_form", clear_on_submit=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    manual_sku = st.text_input("SKU")
                    manual_category = st.text_input("Category")
                with col2:
                    manual_sold = st.number_input("Units Sold", min_value=0, step=1)
                    manual_returned = st.number_input("Units Returned", min_value=0, step=1)
                with col3:
                    manual_landed_cost = st.number_input("Landed Cost ($)", min_value=0.0, step=1.0)
                    manual_retail_price = st.number_input("Retail Price ($)", min_value=0.0, step=1.0)

                with st.expander("Optional Risk Signals", expanded=False):
                    col4, col5, col6 = st.columns(3)
                    with col4:
                        manual_launch_date = st.text_input("Launch Date (YYYY-MM-DD)")
                        manual_safety_risk = st.selectbox("Safety Risk", options=["", "Yes", "No"])
                    with col5:
                        manual_zero_tolerance = st.selectbox("Zero Tolerance Component", options=["", "Yes", "No"])
                        manual_aql_fail = st.selectbox("AQL Fail", options=["", "Yes", "No"])
                    with col6:
                        manual_complaint_count = st.number_input("Unique Complaints (30d)", min_value=0, step=1)
                        manual_sales_units_30d = st.number_input("Sales Units (30d)", min_value=0, step=1)
                        manual_sales_value_30d = st.number_input("Sales Value $ (30d)", min_value=0.0, step=10.0)

                submitted = st.form_submit_button("➕ Add Product")
                if submitted:
                    if not manual_sku or not manual_category:
                        st.error("SKU and Category are required for manual entry.")
                    else:
                        st.session_state.qc_manual_entries.append({
                            'SKU': manual_sku,
                            'Category': manual_category,
                            'Sold': manual_sold,
                            'Returned': manual_returned,
                            'Landed Cost': manual_landed_cost,
                            'Retail Price': manual_retail_price,
                            'Launch Date': manual_launch_date.strip() if manual_launch_date else '',
                            'Safety Risk': manual_safety_risk,
                            'Zero Tolerance Component': manual_zero_tolerance,
                            'AQL Fail': manual_aql_fail,
                            'Unique Complaint Count (30d)': manual_complaint_count,
                            'Sales Units (30d)': manual_sales_units_30d,
                            'Sales Value (30d)': manual_sales_value_30d,
                            'Input Source': 'Manual'
                        })
                        st.success(f"Added {manual_sku} to the manual intake list.")

            if st.session_state.qc_manual_entries:
                st.markdown("#### ✅ Manual Entries")
                st.dataframe(pd.DataFrame(st.session_state.qc_manual_entries), use_container_width=True)
                if st.button("🧹 Clear Manual Entries", use_container_width=True):
                    st.session_state.qc_manual_entries = []
                    st.rerun()

            st.divider()

            st.markdown("#### 📤 Upload Intake File")
            qc_file = st.file_uploader(
                "Upload Quality Screening Data (CSV/XLSX)",
                type=['csv', 'xlsx', 'xls', 'txt'],
                key="tab3_uploader"
            )

            upload_mapping = {}
            normalized_upload = None
            if qc_file:
                qc_df = load_quality_case_file(qc_file.read(), qc_file.name)

                if qc_df is not None:
                    st.markdown("#### 🧭 Column Mapping")
                    columns = qc_df.columns.tolist()

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        upload_mapping['SKU'] = st.selectbox("SKU Column", options=columns, index=0)
                        upload_mapping['Category'] = st.selectbox(
                            "Category Column",
                            options=columns,
                            index=min(1, len(columns) - 1)
                        )
                        upload_mapping['Sold'] = st.selectbox("Sold Column", options=columns)
                    with col2:
                        upload_mapping['Returned'] = st.selectbox("Returned Column", options=columns)
                        upload_mapping['Landed Cost'] = st.selectbox("Landed Cost Column (Optional)", options=[""] + columns)
                        upload_mapping['Retail Price'] = st.selectbox("Retail Price Column (Optional)", options=[""] + columns)
                    with col3:
                        upload_mapping['Launch Date'] = st.selectbox("Launch Date Column (Optional)", options=[""] + columns)
                        upload_mapping['Safety Risk'] = st.selectbox("Safety Risk Column (Optional)", options=[""] + columns)
                        upload_mapping['Zero Tolerance Component'] = st.selectbox(
                            "Zero Tolerance Component Column (Optional)",
                            options=[""] + columns
                        )

                    st.markdown("#### 🔎 Additional Optional Columns")
                    col7, col8, col9 = st.columns(3)
                    with col7:
                        upload_mapping['AQL Fail'] = st.selectbox("AQL Fail Column (Optional)", options=[""] + columns)
                    with col8:
                        upload_mapping['Unique Complaint Count (30d)'] = st.selectbox(
                            "Unique Complaint Count (30d) (Optional)", options=[""] + columns
                        )
                        upload_mapping['Sales Units (30d)'] = st.selectbox(
                            "Sales Units (30d) (Optional)", options=[""] + columns
                        )
                    with col9:
                        upload_mapping['Sales Value (30d)'] = st.selectbox(
                            "Sales Value $ (30d) (Optional)", options=[""] + columns
                        )
                    normalized_upload = normalize_quality_case_data(qc_df, upload_mapping, "Upload")

        manual_df = pd.DataFrame(st.session_state.qc_manual_entries)
        if manual_df.empty:
            manual_df = pd.DataFrame(columns=QUALITY_CASE_STANDARD_COLUMNS)

        combined_frames = []
        if normalized_upload is not None:
            combined_frames.append(normalized_upload)
        if not manual_df.empty:
            combined_frames.append(manual_df)

        if combined_frames:
            combined_df = pd.concat(combined_frames, ignore_index=True)
            st.session_state.qc_combined_data = combined_df
        else:
            combined_df = None

        with thresholds_tab:
            st.markdown("#### ⚙️ Screening Thresholds")
            col4, col5, col6 = st.columns(3)
            with col4:
                return_rate_cap = st.number_input("Return Rate Hard Cap", value=0.25, step=0.01, format="%.2f")
                relative_threshold = st.number_input("Relative Threshold (x Category Avg)", value=1.20, step=0.05)
            with col5:
                cat_avg_delta = st.number_input("Category Avg + Delta (5% = 0.05)", value=0.05, step=0.01)
                low_volume_cutoff = st.number_input("Low Volume Cutoff (Units Sold)", value=10, step=1)
            with col6:
                use_benchmarks = st.checkbox("Use SOP Category Benchmarks", value=True)
                review_days = st.number_input("Monitor Review in X Days", value=14, step=1)

            st.markdown("#### 🚨 Immediate Escalation Criteria")
            col7, col8, col9 = st.columns(3)
            with col7:
                landed_cost_threshold = st.number_input("Landed Cost Threshold ($)", value=150.0, step=10.0)
                retail_price_threshold = st.number_input("Retail Price Threshold ($)", value=0.0, step=10.0)
            with col8:
                launch_days = st.number_input("New Launch Window (Days)", value=90, step=5)
            with col9:
                st.caption("Optional flags are mapped in the intake tab.")

            if combined_df is not None and not combined_df.empty:
                st.markdown("#### 📥 Combined Intake Preview")
                st.dataframe(combined_df.head(10), use_container_width=True)
            else:
                st.info("Add manual entries or upload a file to preview intake data.")

        def resolve_optional_column(name, data_frame):
            if name in data_frame.columns:
                series = data_frame[name].replace('', np.nan)
                if series.notna().any():
                    return name
            return None

        with results_tab:
            if combined_df is not None and not combined_df.empty:
                if st.button("🔍 Run Quality Case Screening", type="primary"):
                    config = {
                        'sold_col': 'Sold',
                        'returned_col': 'Returned',
                        'category_col': 'Category',
                        'use_benchmarks': use_benchmarks,
                        'return_rate_cap': return_rate_cap,
                        'relative_threshold': relative_threshold,
                        'cat_avg_delta': cat_avg_delta,
                        'low_volume_cutoff': int(low_volume_cutoff),
                        'review_days': int(review_days),
                        'safety_col': resolve_optional_column('Safety Risk', combined_df),
                        'zero_tolerance_col': resolve_optional_column('Zero Tolerance Component', combined_df),
                        'landed_cost_col': resolve_optional_column('Landed Cost', combined_df),
                        'retail_price_col': resolve_optional_column('Retail Price', combined_df),
                        'launch_date_col': resolve_optional_column('Launch Date', combined_df),
                        'launch_days': int(launch_days),
                        'aql_fail_col': resolve_optional_column('AQL Fail', combined_df),
                        'landed_cost_threshold': landed_cost_threshold,
                        'retail_price_threshold': retail_price_threshold,
                        'complaint_count_col': resolve_optional_column('Unique Complaint Count (30d)', combined_df),
                        'sales_units_30d_col': resolve_optional_column('Sales Units (30d)', combined_df),
                        'sales_value_30d_col': resolve_optional_column('Sales Value (30d)', combined_df),
                    }

                    with st.spinner("Analyzing quality triggers..."):
                        screened = compute_quality_case_screening(combined_df, config)

                    st.session_state.qc_screened_data = screened
                    st.session_state.qc_ai_review = None

                    export_buffer = io.BytesIO()
                    with pd.ExcelWriter(export_buffer, engine='xlsxwriter') as writer:
                        screened.to_excel(writer, index=False, sheet_name='Quality Screening')
                    export_buffer.seek(0)
                    st.session_state.qc_export_data = export_buffer
                    st.session_state.qc_export_filename = f"quality_screening_{datetime.now().strftime('%Y%m%d')}.xlsx"
            else:
                st.info("Add manual entries or upload a file to run screening.")

        if st.session_state.qc_screened_data is not None:
            screened = st.session_state.qc_screened_data
            display_quality_case_dashboard(screened)

            with results_tab:
                st.markdown("#### ✅ Screening Results")
                st.dataframe(
                    screened[
                        [
                            'SKU',
                            'Category',
                            'Return_Rate_Display',
                            'Cat_Avg_Display',
                            'Recommended_Action',
                            'Review_By',
                            'Input Source'
                        ]
                    ],
                    use_container_width=True
                )

                st.markdown("#### 📊 ANOVA / MANOVA Summary")
                st.markdown(
                    """
                    **What these tests do**
                    - **ANOVA** compares average return rates across categories to detect statistically meaningful differences.
                    - **MANOVA** evaluates multiple metrics at once (e.g., return rate + complaint count) to spot multi-signal
                      shifts that might indicate systemic device issues.

                    **Formulas**
                    - ANOVA: \\(F = MS_{between} / MS_{within}\\), where
                      \\(MS_{between} = SS_{between}/df_{between}\\) and
                      \\(MS_{within} = SS_{within}/df_{within}\\).
                    - MANOVA (Wilks' Lambda): \\(\\Lambda = \\det(W) / \\det(W + B)\\).

                    **Why this matters for medical devices**
                    - A high **F** suggests a category is behaving differently than expected (possible design or quality issues).
                    - A low **Wilks' Lambda** suggests multiple risk signals are shifting together, warranting closer review.
                    """
                )

                categories = sorted(screened['Category'].dropna().unique().tolist())
                selected_categories = st.multiselect(
                    "Categories to include",
                    options=categories,
                    default=categories
                )
                if not selected_categories:
                    st.warning("Select at least one category to run ANOVA/MANOVA.")
                    selected_categories = []

                metric_defaults = [
                    col for col in [
                        'Return_Rate',
                        'Sold',
                        'Returned',
                        'Landed Cost',
                        'Retail Price',
                        'Unique Complaint Count (30d)',
                        'Sales Units (30d)',
                        'Sales Value (30d)'
                    ]
                    if col in screened.columns
                ]
                numeric_extra = [
                    col for col in screened.columns
                    if pd.api.types.is_numeric_dtype(screened[col]) and col not in metric_defaults
                ]
                manova_options = metric_defaults + numeric_extra

                if len(manova_options) < 2:
                    st.warning("MANOVA requires at least two numeric metrics in the data.")
                    manova_metric_1 = None
                    manova_metric_2 = None
                else:
                    manova_metric_1 = st.selectbox("MANOVA Metric 1", options=manova_options, index=0)
                    metric_2_options = [col for col in manova_options if col != manova_metric_1]
                    manova_metric_2 = st.selectbox("MANOVA Metric 2", options=metric_2_options, index=0)

                with st.expander("➕ Add manual data for ANOVA/MANOVA"):
                    st.caption(
                        "These rows are used only for the statistical checks below and do not change the screening results."
                    )
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        manual_stat_category = st.text_input("Category (Stats Only)")
                        manual_stat_return_rate = st.number_input(
                            "Return Rate (0.00 - 1.00)",
                            min_value=0.0,
                            max_value=1.0,
                            step=0.01,
                            format="%.2f"
                        )
                    with col2:
                        manual_stat_metric_1 = st.number_input(
                            f"{manova_metric_1 or 'Metric 1'} Value",
                            step=1.0
                        )
                    with col3:
                        manual_stat_metric_2 = st.number_input(
                            f"{manova_metric_2 or 'Metric 2'} Value",
                            step=1.0
                        )

                    if st.button("Add Manual Stats Row"):
                        if not manual_stat_category:
                            st.error("Category is required for manual stats entry.")
                        else:
                            entry = {
                                'Category': manual_stat_category,
                                'Return_Rate': manual_stat_return_rate
                            }
                            if manova_metric_1:
                                entry[manova_metric_1] = manual_stat_metric_1
                            if manova_metric_2:
                                entry[manova_metric_2] = manual_stat_metric_2
                            st.session_state.qc_stat_manual_entries.append(entry)
                            st.success("Manual stats entry added.")

                    if st.session_state.qc_stat_manual_entries:
                        st.dataframe(
                            pd.DataFrame(st.session_state.qc_stat_manual_entries),
                            use_container_width=True
                        )
                        if st.button("Clear Manual Stats Entries"):
                            st.session_state.qc_stat_manual_entries = []
                            st.rerun()

                analysis_df = screened[['Category', 'Return_Rate']].copy()
                if manova_metric_1:
                    analysis_df[manova_metric_1] = (
                        parse_numeric(screened[manova_metric_1]) if manova_metric_1 in screened.columns else np.nan
                    )
                if manova_metric_2:
                    analysis_df[manova_metric_2] = (
                        parse_numeric(screened[manova_metric_2]) if manova_metric_2 in screened.columns else np.nan
                    )

                if st.session_state.qc_stat_manual_entries:
                    manual_stats_df = pd.DataFrame(st.session_state.qc_stat_manual_entries)
                    needed_cols = ['Category', 'Return_Rate', manova_metric_1, manova_metric_2]
                    for col in needed_cols:
                        if col and col not in manual_stats_df.columns:
                            manual_stats_df[col] = np.nan
                    manual_stats_df = manual_stats_df[[col for col in needed_cols if col]]
                    analysis_df = pd.concat([analysis_df, manual_stats_df], ignore_index=True)

                if selected_categories:
                    analysis_df = analysis_df[analysis_df['Category'].isin(selected_categories)]

                if selected_categories:
                    summary_table = (
                        analysis_df.groupby('Category')['Return_Rate']
                        .agg(['count', 'mean'])
                        .rename(columns={'count': 'Sample Size', 'mean': 'Avg Return Rate'})
                        .reset_index()
                    )
                    summary_table['Avg Return Rate'] = (
                        summary_table['Avg Return Rate'] * 100
                    ).round(2).astype(str) + '%'
                    st.markdown("**ANOVA Group Summary**")
                    st.dataframe(summary_table, use_container_width=True)

                grouped = [
                    parse_numeric(analysis_df.loc[analysis_df['Category'] == cat, 'Return_Rate']).values
                    for cat in selected_categories
                ]
                anova_results = one_way_anova(grouped)
                if anova_results:
                    st.info(
                        f"ANOVA F={anova_results['f_stat']:.4f} "
                        f"(df={anova_results['df_between']}, {anova_results['df_within']})"
                    )
                    st.caption(
                        f"Group sizes: {anova_results['group_sizes']}. "
                        "Higher F suggests category-level return rate differences worth investigating."
                    )
                else:
                    st.warning("ANOVA requires at least two categories with data.")

                if manova_metric_1 and manova_metric_2:
                    manova_groups = []
                    for cat in selected_categories:
                        subset = analysis_df.loc[
                            analysis_df['Category'] == cat,
                            [manova_metric_1, manova_metric_2]
                        ]
                        subset = subset.apply(parse_numeric).dropna()
                        if len(subset) > 1:
                            manova_groups.append(subset.values)

                    manova_results = manova_wilks_lambda(manova_groups, [manova_metric_1, manova_metric_2])
                    if manova_results:
                        st.info(
                            f"MANOVA Wilks' Lambda={manova_results['wilks_lambda']:.4f} "
                            f"(groups: {manova_results['group_sizes']})"
                        )
                        st.caption(
                            "Lower Wilks' Lambda indicates stronger multivariate separation across categories."
                        )
                    else:
                        st.warning("MANOVA requires at least two categories with 2+ rows.")

                st.markdown("#### ⬇️ Export Screening Data")
                st.download_button(
                    label="Download Screening Results",
                    data=st.session_state.qc_export_data,
                    file_name=st.session_state.qc_export_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary"
                )

                with st.expander("💬 More (AI Review + Chat)"):
                    st.markdown("#### 🤖 AI Quality Review")
                    if st.button("🧠 Generate AI Review", use_container_width=True):
                        analyzer = get_ai_analyzer()
                        summary = build_quality_case_summary(screened)
                        system_prompt = (
                            "You are a quality analyst reviewing screening results. "
                            "Summarize key risks, escalations, and recommended next actions. "
                            "Keep the response structured and concise."
                        )
                        prompt = (
                            "Analyze the screening summary and provide a short executive readout.\n\n"
                            f"Summary:\n{json.dumps(summary, indent=2)}"
                        )
                        with st.spinner("Generating AI review..."):
                            response = analyzer.generate_text(prompt, system_prompt, mode='summary')
                        st.session_state.qc_ai_review = response or "AI review unavailable."

                    if st.session_state.qc_ai_review:
                        st.write(st.session_state.qc_ai_review)

                    st.markdown("#### 💬 Quality Case Chat")
                    for message in st.session_state.qc_chat_history:
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])

                    chat_prompt = st.chat_input("Ask about screening results or column requirements...")
                    if chat_prompt:
                        st.session_state.qc_chat_history.append({"role": "user", "content": chat_prompt})
                        context_summary = build_quality_case_summary(screened)
                        system_prompt = (
                            "You are a quality case screening assistant. Answer using the provided summary and "
                            "column requirements. If the user asks for details not present, say that you do not have "
                            "that data. Provide actionable, concise answers."
                        )
                        prompt = (
                            f"Column requirements:\n{json.dumps(QUALITY_CASE_COLUMNS, indent=2)}\n\n"
                            f"Screening summary:\n{json.dumps(context_summary, indent=2)}\n\n"
                            f"User question: {chat_prompt}"
                        )
                        analyzer = get_ai_analyzer()
                        with st.spinner("Thinking..."):
                            response = analyzer.generate_text(prompt, system_prompt, mode='chat')
                        assistant_message = response or "I'm unable to answer that right now."
                        st.session_state.qc_chat_history.append({"role": "assistant", "content": assistant_message})
                        st.rerun()

if __name__ == "__main__":
    main()
