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
import altair as alt

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
        'anova_result': None
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
    
    # Configuration
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        mode = st.radio("Mode", ["Lite (Manual/Small Batch)", "Pro (Mass Analysis/Upload)"], horizontal=True)
        st.session_state.qc_mode = "Lite" if "Lite" in mode else "Pro"
    
    with col3:
        # API Status
        keys = check_api_keys()
        if keys.get('openai') or keys.get('claude'):
            st.success("AI Connected")
        else:
            st.error("AI Disconnected")

    st.divider()

    # INPUT
    df_input = None
    
    if st.session_state.qc_mode == "Lite":
        st.info("ℹ️ Lite Mode: Analyze 1-5 products manually.")
        with st.form("lite_entry"):
            c1, c2, c3 = st.columns(3)
            sku = c1.text_input("Product SKU/Name")
            category = c2.selectbox("Category", list(SOP_THRESHOLDS.keys()))
            cost = c3.number_input("Landed Cost ($)", min_value=0.0)
            
            c4, c5, c6 = st.columns(3)
            sold = c4.number_input("Units Sold", min_value=1)
            returned = c5.number_input("Units Returned", min_value=0)
            complaint = c6.text_area("Top Complaint Reasons")
            
            manual_text = st.text_area("Context (Manuals/Feedback) - Optional")
            safety = st.checkbox("Potential Safety Risk?")
            
            if st.form_submit_button("Run Analysis") and sku:
                df_input = pd.DataFrame([{
                    'SKU': sku, 'Category': category, 'Landed Cost': cost,
                    'Sold': sold, 'Returned': returned, 'Complaint_Text': complaint,
                    'Manual_Context': manual_text, 'Safety Risk': 'Yes' if safety else 'No'
                }])

    else: # PRO MODE
        st.info("🚀 Pro Mode: Mass analysis with statistical rigor.")
        uploaded = st.file_uploader("Upload Data (CSV/Excel)", type=['csv', 'xlsx'], key="qc_upload")
        if uploaded:
            try:
                if uploaded.name.endswith('.csv'): df_input = pd.read_csv(uploaded)
                else: df_input = pd.read_excel(uploaded)
                
                report = QualityAnalytics.validate_upload(df_input, ['SKU', 'Category', 'Sold', 'Returned'])
                if not report['valid']:
                    st.error(f"Missing columns: {report['missing_cols']}")
                    df_input = None
                else:
                    st.success(f"Loaded {len(df_input)} rows.")
            except Exception as e:
                st.error(f"Error: {e}")

    # PROCESSING
    if df_input is not None:
        if st.button("🔍 Run Screening Analysis", type="primary"):
            log_process("Started Screening Analysis")
            progress = st.progress(0)
            
            # 1. Clean & Calc
            df_input['Sold'] = parse_numeric(df_input['Sold'])
            df_input['Returned'] = parse_numeric(df_input['Returned'])
            df_input['Landed Cost'] = parse_numeric(df_input.get('Landed Cost', pd.Series(0)))
            df_input['Return_Rate'] = df_input['Returned'] / df_input['Sold']
            
            # 2. ANOVA (if enough data)
            if len(df_input) > 5:
                anova = QualityAnalytics.perform_anova(df_input, 'Category', 'Return_Rate')
                if anova.get('p_value'):
                    st.session_state.anova_result = anova
                    log_process(f"ANOVA P-Value: {anova['p_value']:.4f}")

            # 3. Row Logic
            results = []
            for _, row in df_input.iterrows():
                cat_avg = SOP_THRESHOLDS.get(row['Category'], 0.10)
                
                # Logic from QualityAnalytics
                risk = QualityAnalytics.calculate_risk_score(row, cat_avg)
                action = QualityAnalytics.determine_action(row, SOP_THRESHOLDS)
                spc = QualityAnalytics.detect_spc_signals(row, cat_avg, cat_avg*0.2)
                
                row['Risk_Score'] = risk
                row['Recommended_Action'] = action
                row['SPC_Signal'] = spc
                results.append(row)
                
            st.session_state.qc_results = pd.DataFrame(results)
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
                tooltip=['SKU', 'Category', 'Return_Rate', 'Recommended_Action']
            ).interactive()
            st.altair_chart(chart, use_container_width=True)

        # Metrics
        c1, c2, c3 = st.columns(3)
        escalations = len(df[df['Recommended_Action'].str.contains("Escalate")])
        c1.metric("Total Analyzed", len(df))
        c2.metric("Escalations Needed", escalations, delta_color="inverse")
        if st.session_state.anova_result:
            p = st.session_state.anova_result.get('p_value', 1.0)
            c3.metric("ANOVA p-value", f"{p:.4f}", delta="Significant" if p < 0.05 else "Not Significant")

        # Table
        st.dataframe(df.style.apply(lambda x: ['background-color: #ff4b4b' if 'Escalate' in str(v) else '' for v in x], subset=['Recommended_Action']))
        
        # AI Investigation
        st.markdown("#### 🤖 AI Investigation Plan")
        targets = df[df['Recommended_Action'].str.contains("Escalate")]['SKU'].unique()
        if len(targets) > 0:
            sel_sku = st.selectbox("Select SKU to Investigate", targets)
            if st.button("Generate CAPA Plan"):
                row = df[df['SKU'] == sel_sku].iloc[0]
                prompt = f"Create investigation plan for SKU {sel_sku}. Return Rate: {row['Return_Rate']:.1%}. Issue: {row.get('Complaint_Text', 'N/A')}"
                
                analyzer = get_ai_analyzer()
                with st.spinner("Thinking..."):
                    resp = analyzer.generate_text(prompt, "You are a Quality Engineer.")
                    st.markdown(resp)
        else:
            st.info("No escalations detected.")

        with st.expander("Methodology"):
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
