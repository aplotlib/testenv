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
                
                elapsed = time.time() - start_time
                speed = processed_count / elapsed if elapsed > 0 else 0
                remaining = (total_valid - processed_count) / speed if speed > 0 else 0
                
                # Update status with clear information
                with stats_container:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Progress", f"{processed_count:,}/{total_valid:,}")
                    with col2:
                        st.metric("Speed", f"{speed:.1f}/sec")
                    with col3:
                        st.metric("Chunk", f"{chunk_num}/{total_chunks}")
                    with col4:
                        if remaining > 0:
                            st.metric("ETA", f"{int(remaining)}s")
                        else:
                            st.metric("ETA", "Complete")
                
                # Small delay to prevent overwhelming
                time.sleep(0.05)
                
        except Exception as e:
            logger.error(f"Chunk processing error: {e}")
            st.session_state.processing_errors.append(f"Chunk {chunk_num}: {str(e)}")
            
            # Fill failed items with default category
            for item in batch_data:
                if pd.isna(df.at[item['index'], category_col]):
                    df.at[item['index'], category_col] = 'Other/Miscellaneous'
        
        # Force garbage collection after each chunk
        gc.collect()
    
    # Final update
    progress_bar.progress(1.0)
    elapsed = time.time() - start_time
    st.session_state.processing_speed = processed_count / elapsed if elapsed > 0 else 0
    
    # Clear the stats container and show final message
    stats_container.empty()
    status_text.success(f"✅ Complete! Processed {processed_count:,} returns in {elapsed:.1f} seconds at {st.session_state.processing_speed:.1f} returns/second")
    
    return df


def generate_statistics(df, column_mapping):
    """Generate statistics from categorized data"""
    category_col = column_mapping.get('category')
    sku_col = column_mapping.get('sku')
    
    if not category_col:
        logger.warning("No category column in mapping, cannot generate statistics")
        return
    
    # Category statistics
    categorized_df = df[df[category_col].notna() & (df[category_col] != '')]
    if len(categorized_df) == 0:
        logger.warning("No categorized data found")
        return
    
    category_counts = categorized_df[category_col].value_counts()
    st.session_state.reason_summary = category_counts.to_dict()
    
    # SKU statistics
    if sku_col and sku_col in df.columns:
        product_summary = defaultdict(lambda: defaultdict(int))
        
        for _, row in categorized_df.iterrows():
            if pd.notna(row.get(sku_col)):
                sku = str(row[sku_col]).strip()
                if sku and sku != 'nan':
                    category = row[category_col]
                    product_summary[sku][category] += 1
        
        st.session_state.product_summary = dict(product_summary)
        logger.info(f"Generated product summary for {len(product_summary)} SKUs")
    
    logger.info(f"Statistics generated: {len(st.session_state.reason_summary)} categories")


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
    """Display enhanced results dashboard (Tab 1)"""
    st.markdown("### 📊 Analysis Results")
    
    # Validate column mapping
    category_col = column_mapping.get('category')
    sku_col = column_mapping.get('sku')
    
    if not category_col:
        st.error("Category column not detected. Unable to render summary metrics.")
        return
    
    # Calculate metrics
    total_rows = len(df)
    categorized_rows = len(df[df[category_col].notna() & (df[category_col] != '')])
    
    # Key Metrics Row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Rows", f"{total_rows:,}")
    
    with col2:
        st.metric("Categorized", f"{categorized_rows:,}")
    
    with col3:
        success_rate = categorized_rows / total_rows * 100 if total_rows > 0 else 0
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    with col4:
        quality_count = sum(count for cat, count in st.session_state.reason_summary.items() 
                          if cat in QUALITY_CATEGORIES)
        quality_rate = quality_count / categorized_rows * 100 if categorized_rows > 0 else 0
        st.metric("Quality Issues", f"{quality_rate:.1f}%", 
                 help=f"{quality_count:,} quality-related returns")
    
    with col5:
        cost_per_return = st.session_state.total_cost / categorized_rows if categorized_rows > 0 else 0
        st.metric("Cost/Return", f"${cost_per_return:.4f}")
    
    # Category Distribution
    st.markdown("---")
    st.markdown("#### 📈 Category Distribution")
    
    if st.session_state.reason_summary:
        # Create two columns for better layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Show top categories with visual bars
            top_categories = sorted(st.session_state.reason_summary.items(), 
                                  key=lambda x: x[1], reverse=True)[:10]
            
            for i, (cat, count) in enumerate(top_categories):
                pct = count / categorized_rows * 100 if categorized_rows > 0 else 0
                
                # Determine color based on category type
                if cat in QUALITY_CATEGORIES:
                    color = COLORS['danger']
                    icon = "🔴"
                else:
                    color = COLORS['primary']
                    icon = "🔵"
                
                # Create visual bar
                st.markdown(f"""
                <div style="margin: 0.8rem 0;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.3rem;">
                        <span style="font-weight: 500;">{icon} {cat}</span>
                        <span style="color: {COLORS['muted']};">{count:,} ({pct:.1f}%)</span>
                    </div>
                    <div style="background: rgba(255,255,255,0.1); height: 20px; border-radius: 10px; overflow: hidden;">
                        <div style="background: {color}; width: {pct}%; height: 100%; 
                                    border-radius: 10px; transition: width 0.5s ease;">
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Summary stats box
            top_pct = (top_categories[0][1] / categorized_rows * 100) if top_categories and categorized_rows > 0 else 0
            st.markdown(f"""
            <div class="info-box" style="text-align: center;">
                <h4 style="color: var(--primary); margin: 0;">Summary</h4>
                <div style="margin-top: 1rem;">
                    <div style="font-size: 2em; font-weight: 700; color: var(--danger);">
                        {quality_rate:.0f}%
                    </div>
                    <div style="color: var(--muted);">Quality Issues</div>
                </div>
                <hr style="opacity: 0.2; margin: 1rem 0;">
                <div style="font-size: 0.9em;">
                    <div>Categories: {len(st.session_state.reason_summary)}</div>
                    <div style="color: var(--muted); margin-top: 0.5rem;">
                        Top category accounts for {top_pct:.0f}% of returns
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Product Analysis (if enabled)
    if st.session_state.show_product_analysis and st.session_state.product_summary:
        st.markdown("---")
        st.markdown("#### 📦 Product/SKU Analysis")
        
        # Product metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Unique SKUs", f"{len(st.session_state.product_summary):,}")
        
        with col2:
            avg_returns_per_sku = categorized_rows / len(st.session_state.product_summary) if st.session_state.product_summary else 0
            st.metric("Avg Returns/SKU", f"{avg_returns_per_sku:.1f}")
        
        with col3:
            # Find SKUs with high quality issues
            high_quality_skus = 0
            for sku, issues in st.session_state.product_summary.items():
                quality_count_sku = sum(count for cat, count in issues.items() if cat in QUALITY_CATEGORIES)
                total_count = sum(issues.values())
                if total_count > 0 and quality_count_sku / total_count > 0.5:
                    high_quality_skus += 1
            st.metric("High Risk SKUs", f"{high_quality_skus:,}", 
                     help="SKUs with >50% quality issues")
        
        # Top problematic products
        st.markdown("##### 🚨 Top 10 Products by Return Volume (from Column B)")
        
        # Calculate product metrics
        product_data = []
        for sku, issues in st.session_state.product_summary.items():
            total = sum(issues.values())
            quality = sum(count for cat, count in issues.items() if cat in QUALITY_CATEGORIES)
            quality_pct_prod = quality / total * 100 if total > 0 else 0
            top_issue = max(issues.items(), key=lambda x: x[1])[0] if issues else 'N/A'
            
            product_data.append({
                'SKU': sku,
                'Total Returns': total,
                'Quality Issues': quality,
                'Quality %': quality_pct_prod,
                'Top Issue': top_issue,
                'Risk Score': quality * (quality_pct_prod / 100)
            })
        
        # Sort by total returns
        product_data.sort(key=lambda x: x['Total Returns'], reverse=True)
        
        # Display top products
        for i, product in enumerate(product_data[:10]):
            if i < 5:
                if product['Quality %'] > 50:
                    risk_color = COLORS['danger']
                    risk_label = "High Risk"
                elif product['Quality %'] > 25:
                    risk_color = COLORS['warning']
                    risk_label = "Medium Risk"
                else:
                    risk_color = COLORS['success']
                    risk_label = "Low Risk"
                
                st.markdown(f"""
                <div class="info-box" style="border-left: 4px solid {risk_color}; margin: 0.5rem 0;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong>{i+1}. SKU: {product['SKU'][:40]}{'...' if len(product['SKU']) > 40 else ''}</strong>
                            <div style="color: {COLORS['muted']}; font-size: 0.9em; margin-top: 0.2rem;">
                                Top issue: {product['Top Issue']}
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 1.2em; font-weight: 600;">{product['Total Returns']:,} returns</div>
                            <div style="color: {risk_color}; font-size: 0.9em;">
                                {product['Quality %']:.0f}% quality ({risk_label})
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"{i+1}. **{product['SKU'][:30]}...**: {product['Total Returns']} returns ({product['Quality %']:.0f}% quality)")
        
        # Option to export full product analysis
        if st.button("📥 Export Full SKU Analysis"):
            export_data = []
            for sku, issues in st.session_state.product_summary.items():
                for category, count in issues.items():
                    export_data.append({
                        'SKU': sku,
                        'Category': category,
                        'Count': count,
                        'Is_Quality_Issue': 'Yes' if category in QUALITY_CATEGORIES else 'No'
                    })
            
            export_df = pd.DataFrame(export_data)
            csv = export_df.to_csv(index=False)
            
            st.download_button(
                label="Download SKU Analysis CSV",
                data=csv,
                file_name=f"sku_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )


# -------------------------
# TAB 2 LOGIC: B2B Reports (PRESERVED)
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

        logger.info(f"B2B Processing: {len(df)} rows loaded")
        return df
        
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None


def generate_b2b_report(df, analyzer, batch_size):
    """Generate the B2B report with AI summaries and Cleaned SKUs"""
    
    # Columns to use for display/AI
    display_col = 'Display Name' if 'Display Name' in df.columns else df.columns[0]
    desc_col = 'Description' if 'Description' in df.columns else None
    
    # Pre-process data for AI batching
    items_to_process = []
    
    for idx, row in df.iterrows():
        subject = str(row.get(display_col, ''))
        description = str(row.get(desc_col, '')) if desc_col else ''
        
        # Extract Main SKU using strict logic
        main_sku = find_sku_in_row(row)
        
        items_to_process.append({
            'index': idx,
            'subject': subject,
            'details': strip_html(description)[:1000],
            'full_description': description,
            'sku': main_sku
        })
    
    # Batch Process with AI
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_items = len(items_to_process)
    processed_results = []
    
    for i in range(0, total_items, batch_size):
        batch = items_to_process[i:i+batch_size]
        
        # Use the summarize_batch method
        batch_results = analyzer.summarize_batch(batch)
        processed_results.extend(batch_results)
        
        # Update progress
        progress = min((i + batch_size) / total_items, 1.0)
        progress_bar.progress(progress)
        status_text.text(f"⏳ Generating summaries: {min(i + batch_size, total_items)}/{total_items}")
        
    status_text.success("✅ AI Summarization Complete!")
    time.sleep(1)
    status_text.empty()
    progress_bar.empty()
    
    # Construct Final DataFrame
    final_rows = []
    for item in processed_results:
        final_rows.append({
            'Display Name': item['subject'],
            'Description': item['full_description'],
            'SKU': item['sku'],
            'Reason': item.get('summary', 'Summary Unavailable')
        })
        
    return pd.DataFrame(final_rows)


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
    """Render Lite mode - manual entry for 1-5 products with dynamic forms"""
    
    st.info("ℹ️ **Lite Mode**: Enter product details manually for quick screening (1-5 products)")
    
    # Initialize lite entries in session state if not exists
    if 'lite_entries' not in st.session_state or not st.session_state.lite_entries:
        st.session_state.lite_entries = [{'id': 0}]  # Start with one empty entry
    
    # Date range selection (applies to all products)
    st.markdown("#### 📅 Date Range (applies to all products)")
    col1, col2 = st.columns(2)
    with col1:
        date_range = st.selectbox(
            "Data Date Range",
            ["Last 30 days", "Last 60 days", "Last 90 days", "Last 180 days", "Last 365 days", "Custom Range"],
            index=0,
            key="lite_date_range"
        )
    with col2:
        if date_range == "Custom Range":
            date_start = st.date_input("Start Date", datetime.now() - timedelta(days=30), key="lite_date_start")
            date_end = st.date_input("End Date", datetime.now(), key="lite_date_end")
        else:
            days = int(re.search(r'\d+', date_range).group())
            date_start = datetime.now() - timedelta(days=days)
            date_end = datetime.now()
    
    st.markdown("---")
    
    # Product count controls
    col_add, col_remove, col_count = st.columns([1, 1, 2])
    with col_add:
        if st.button("➕ Add Product", disabled=len(st.session_state.lite_entries) >= 5):
            new_id = max([e['id'] for e in st.session_state.lite_entries]) + 1
            st.session_state.lite_entries.append({'id': new_id})
            st.rerun()
    with col_remove:
        if st.button("➖ Remove Last", disabled=len(st.session_state.lite_entries) <= 1):
            st.session_state.lite_entries.pop()
            st.rerun()
    with col_count:
        st.markdown(f"**Products to screen:** {len(st.session_state.lite_entries)} of 5 max")
    
    st.markdown("---")
    
    # Collect all product data
    all_entries = []
    all_valid = True
    
    # Create expandable sections for each product
    for idx, entry in enumerate(st.session_state.lite_entries):
        entry_id = entry['id']
        with st.expander(f"📦 Product {idx + 1}", expanded=(idx == 0 or idx == len(st.session_state.lite_entries) - 1)):
            
            # Required fields
            st.markdown("**Required Fields**")
            col1, col2, col3 = st.columns(3)
            with col1:
                product_name = st.text_input("Product Name*", placeholder="e.g., Knee Walker", key=f"name_{entry_id}")
            with col2:
                product_sku = st.text_input("Product SKU*", placeholder="e.g., MOB1027", key=f"sku_{entry_id}")
            with col3:
                category = st.selectbox(
                    "Category*",
                    options=['Select...'] + list(DEFAULT_CATEGORY_THRESHOLDS.keys()),
                    index=0,
                    key=f"cat_{entry_id}"
                )
            
            col4, col5, col6 = st.columns(3)
            with col4:
                units_sold = st.number_input("Units Sold*", min_value=1, value=100, key=f"sold_{entry_id}")
            with col5:
                units_returned = st.number_input("Units Returned*", min_value=0, value=0, key=f"ret_{entry_id}")
            with col6:
                return_rate_calc = (units_returned / units_sold * 100) if units_sold > 0 else 0
                st.metric("Return Rate", f"{return_rate_calc:.1f}%")
            
            # Complaint reasons
            complaint_reasons = st.text_input(
                "Top Return Reasons (comma-separated)*",
                placeholder="e.g., Uncomfortable, Too small, Defective wheel",
                key=f"complaints_{entry_id}"
            )
            
            # Optional fields in a collapsed section
            with st.container():
                show_optional = st.checkbox("Show optional fields", key=f"show_opt_{entry_id}")
                
                if show_optional:
                    st.markdown("**Optional Fields**")
                    col7, col8, col9 = st.columns(3)
                    with col7:
                        unit_cost = st.number_input("Landed Cost ($)", min_value=0.0, value=0.0, step=0.01, key=f"cost_{entry_id}")
                    with col8:
                        primary_channel = st.selectbox(
                            "Primary Channel",
                            ["Select...", "Amazon", "B2B", "Website", "Other"],
                            key=f"channel_{entry_id}"
                        )
                    with col9:
                        packaging_method = st.selectbox(
                            "Packaging",
                            ["Select...", "Standard Box", "Poly Bag", "Custom", "Other"],
                            key=f"pack_{entry_id}"
                        )
                    
                    col10, col11 = st.columns(2)
                    with col10:
                        b2b_feedback = st.text_area("B2B Feedback", placeholder="Optional B2B feedback...", height=68, key=f"b2b_fb_{entry_id}")
                    with col11:
                        amazon_feedback = st.text_area("Amazon Feedback", placeholder="Optional Amazon feedback...", height=68, key=f"amz_fb_{entry_id}")
                    
                    manual_context = st.text_area(
                        "Additional Context",
                        placeholder="Any relevant background info, manual excerpts, known issues...",
                        height=68,
                        key=f"context_{entry_id}"
                    )
                else:
                    unit_cost = 0.0
                    primary_channel = "Select..."
                    packaging_method = "Select..."
                    b2b_feedback = ""
                    amazon_feedback = ""
                    manual_context = ""
            
            # Safety flags
            col_safe, col_new = st.columns(2)
            with col_safe:
                safety_risk = st.checkbox("⚠️ Safety Risk?", key=f"safety_{entry_id}")
            with col_new:
                is_new_product = st.checkbox("🆕 New Product?", key=f"new_{entry_id}")
            
            # Validate this entry
            entry_valid = product_name and product_sku and category != 'Select...'
            if not entry_valid:
                all_valid = False
            
            # Build entry dict
            all_entries.append({
                'SKU': product_sku,
                'Name': product_name,
                'Category': category if category != 'Select...' else '',
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
                'Date_Range': f"{date_start} to {date_end}",
                '_valid': entry_valid
            })
    
    st.markdown("---")
    
    # Summary before processing
    valid_count = sum(1 for e in all_entries if e.get('_valid', False))
    if valid_count < len(all_entries):
        st.warning(f"⚠️ {len(all_entries) - valid_count} product(s) missing required fields (Name, SKU, Category)")
    
    # Process button
    col_btn, col_clear = st.columns([3, 1])
    with col_btn:
        if st.button("🔍 Run AI Screening", type="primary", use_container_width=True, disabled=valid_count == 0):
            # Filter to valid entries only
            valid_entries = [e for e in all_entries if e.get('_valid', False)]
            
            # Remove internal _valid flag
            for e in valid_entries:
                e.pop('_valid', None)
            
            # Create DataFrame and process
            df_input = pd.DataFrame(valid_entries)
            process_screening(df_input)
    
    with col_clear:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.lite_entries = [{'id': 0}]
            st.rerun()


def render_pro_mode():
    """Render Pro mode - mass upload analysis"""
    
    st.info("🚀 **Pro Mode**: Upload CSV/Excel for mass analysis (up to 500+ products)")
    
    # Template download section
    st.markdown("#### 📋 Download Template")
    col_template, col_example = st.columns(2)
    
    with col_template:
        # Create blank template
        template_df = pd.DataFrame(columns=[
            'SKU', 'Name', 'Category', 'Sold', 'Returned', 'Landed Cost',
            'Complaint_Text', 'Safety Risk', 'Primary_Channel',
            'B2B_Feedback', 'Amazon_Feedback', 'Manual_Context'
        ])
        
        # Add one example row with instructions
        template_df.loc[0] = [
            'MOB1027', 'Knee Walker Deluxe', 'MOB', 1000, 120, 85.00,
            'Wheel squeaks, uncomfortable padding, hard to fold',
            'No', 'Amazon',
            '', '', ''
        ]
        
        template_csv = template_df.to_csv(index=False)
        st.download_button(
            "📥 Download Blank Template",
            template_csv,
            file_name="quality_screening_template.csv",
            mime="text/csv",
            help="Download a blank CSV template with all columns"
        )
    
    with col_example:
        # Create example with multiple rows
        example_df = pd.DataFrame([
            {
                'SKU': 'MOB1027', 'Name': 'Knee Walker Deluxe', 'Category': 'MOB',
                'Sold': 1000, 'Returned': 120, 'Landed Cost': 85.00,
                'Complaint_Text': 'Wheel squeaks, uncomfortable padding',
                'Safety Risk': 'No', 'Primary_Channel': 'Amazon',
                'B2B_Feedback': '', 'Amazon_Feedback': '3 star avg', 'Manual_Context': ''
            },
            {
                'SKU': 'SUP1036', 'Name': 'Post Op Shoe', 'Category': 'SUP',
                'Sold': 500, 'Returned': 125, 'Landed Cost': 12.00,
                'Complaint_Text': 'Wrong size, poor fit, runs small',
                'Safety Risk': 'No', 'Primary_Channel': 'Amazon',
                'B2B_Feedback': '', 'Amazon_Feedback': '', 'Manual_Context': ''
            },
            {
                'SKU': 'LVA1004', 'Name': 'Alternating Pressure Mattress', 'Category': 'LVA',
                'Sold': 800, 'Returned': 150, 'Landed Cost': 145.00,
                'Complaint_Text': 'Pump failure, air leak, motor noise',
                'Safety Risk': 'Yes', 'Primary_Channel': 'B2B',
                'B2B_Feedback': 'Multiple facilities reporting pump issues',
                'Amazon_Feedback': '', 'Manual_Context': 'Known batch issue from Q3'
            },
            {
                'SKU': 'CSH1006', 'Name': 'Knee Walker Pad Cover', 'Category': 'CSH',
                'Sold': 2000, 'Returned': 180, 'Landed Cost': 8.50,
                'Complaint_Text': 'Velcro wears out, material pilling',
                'Safety Risk': 'No', 'Primary_Channel': 'Amazon',
                'B2B_Feedback': '', 'Amazon_Feedback': '', 'Manual_Context': ''
            },
            {
                'SKU': 'RHB1022', 'Name': 'Shoulder Pulley', 'Category': 'RHB',
                'Sold': 3000, 'Returned': 90, 'Landed Cost': 6.00,
                'Complaint_Text': 'Rope fraying, door bracket weak',
                'Safety Risk': 'No', 'Primary_Channel': 'Amazon',
                'B2B_Feedback': '', 'Amazon_Feedback': '', 'Manual_Context': ''
            }
        ])
        
        example_csv = example_df.to_csv(index=False)
        st.download_button(
            "📥 Download Example Data",
            example_csv,
            file_name="quality_screening_example.csv",
            mime="text/csv",
            help="Download example data with 5 sample products"
        )
    
    # Column reference
    with st.expander("📖 Column Reference Guide"):
        st.markdown("""
        | Column | Required | Description | Example |
        |--------|----------|-------------|---------|
        | `SKU` | ✅ Yes | Product SKU/identifier | MOB1027 |
        | `Name` | No | Product name | Knee Walker Deluxe |
        | `Category` | ✅ Yes | Product category code | MOB, SUP, LVA, CSH, RHB, INS |
        | `Sold` | ✅ Yes | Units sold in period | 1000 |
        | `Returned` | ✅ Yes | Units returned | 120 |
        | `Landed Cost` | No | Unit cost in USD | 85.00 |
        | `Complaint_Text` | No | Top complaints (comma-separated) | Wheel squeaks, hard to fold |
        | `Safety Risk` | No | Safety concern flag | Yes / No |
        | `Primary_Channel` | No | Main sales channel | Amazon, B2B, Website |
        | `B2B_Feedback` | No | B2B customer feedback | Facilities reporting issues |
        | `Amazon_Feedback` | No | Amazon reviews/feedback | 3 star average |
        | `Manual_Context` | No | Additional context | Known batch issue |
        
        **Category Codes:** B2B, INS, RHB, LVA, MOB, CSH, SUP
        
        **Notes:**
        - Return rate is auto-calculated from Sold/Returned
        - If your file has different column names, the system will attempt to map them
        - Blank optional fields are fine
        """)
    
    st.markdown("---")
    
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
                # Store mapping immediately
                st.session_state.column_mapping = column_mapping
                
                # Show file info
                complaint_col = column_mapping.get('complaint')
                if complaint_col:
                    valid_complaints = df[df[complaint_col].notna() & (df[complaint_col].str.strip() != '')].shape[0]
                    st.info(f"Found {valid_complaints:,} complaints to categorize in Column I.")
                else:
                    st.warning("Complaint column not found in expected position.")
                
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

    # --- TAB 2: B2B Reports (PRESERVED) ---
    with tab2:
        # Use Tab 1/2 provider
        st.session_state.ai_provider = provider_map_t12[provider_t12]
        
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
        else:  # Large
            batch_size = 50
            max_workers = 10
            st.caption("Settings: Aggressive parallel processing for high volume.")

        st.divider()
        
        b2b_file = st.file_uploader("Upload Odoo Export (CSV/Excel)", type=['csv', 'xlsx'], key="b2b_uploader")
        
        if b2b_file:
            # Read & Preview
            b2b_df = process_b2b_file(b2b_file.read(), b2b_file.name)
            
            if b2b_df is not None:
                st.markdown(f"**Total Tickets Found:** {len(b2b_df):,}")
                
                # Process Button
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
                                worksheet.set_column(col_num, col_num, 30)

                        st.session_state.b2b_export_data = output.getvalue()
                        st.session_state.b2b_export_filename = f"B2B_Report_{datetime.now().strftime('%Y-%m-%d')}.xlsx"
                        
                        st.rerun()

        # B2B Dashboard Results
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

    # --- TAB 3: Quality Screening (REBUILT) ---
    with tab3:
        render_quality_screening_tab()


if __name__ == "__main__":
    main()
