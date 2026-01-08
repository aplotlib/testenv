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
    from enhanced_ai_analysis import (
        EnhancedAIAnalyzer, AIProvider, FBA_REASON_MAP,
        MEDICAL_DEVICE_CATEGORIES
    )
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
        'b2b_perf_mode': 'Small (< 500 rows)'
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
        return
    
    # Category statistics
    categorized_df = df[df[category_col].notna() & (df[category_col] != '')]
    if len(categorized_df) == 0:
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

def export_with_column_k(df):
    """Export data with categories in column K, preserving original format"""
    output = io.BytesIO()
    
    # Ensure we have at least 11 columns (up to K)
    while len(df.columns) < 11:
        df[f'Col_{len(df.columns)}'] = ''
    
    # Save based on format preference
    if EXCEL_AVAILABLE:
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Write with string format to preserve original data
            df.to_excel(writer, index=False, sheet_name='Returns')
            
            # Get workbook and worksheet
            workbook = writer.book
            worksheet = writer.sheets['Returns']
            
            # Format column K (11th column, index 10) with categories
            category_format = workbook.add_format({
                'bg_color': '#E6F5E6',
                'font_color': '#006600',
                'bold': True
            })
            
            # Apply format to column K
            worksheet.set_column(10, 10, 20, category_format)  # Column K
    else:
        # CSV fallback
        df.to_csv(output, index=False)
    
    output.seek(0)
    return output.getvalue()

def display_results_dashboard(df, column_mapping):
    """Display enhanced results dashboard (Tab 1)"""
    st.markdown("### 📊 Analysis Results")
    
    # Calculate metrics
    total_rows = len(df)
    category_col = column_mapping.get('category')
    sku_col = column_mapping.get('sku')
    
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
            st.markdown("""
            <div class="info-box" style="text-align: center;">
                <h4 style="color: var(--primary); margin: 0;">Summary</h4>
                <div style="margin-top: 1rem;">
                    <div style="font-size: 2em; font-weight: 700; color: var(--danger);">
                        {quality_pct:.0f}%
                    </div>
                    <div style="color: var(--muted);">Quality Issues</div>
                </div>
                <hr style="opacity: 0.2; margin: 1rem 0;">
                <div style="font-size: 0.9em;">
                    <div>Categories: {total_cats}</div>
                    <div style="color: var(--muted); margin-top: 0.5rem;">
                        Top category accounts for {top_pct:.0f}% of returns
                    </div>
                </div>
            </div>
            """.format(
                quality_pct=quality_rate,
                total_cats=len(st.session_state.reason_summary),
                top_pct=(top_categories[0][1] / categorized_rows * 100) if top_categories else 0
            ), unsafe_allow_html=True)
    
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
                quality_count = sum(count for cat, count in issues.items() if cat in QUALITY_CATEGORIES)
                total_count = sum(issues.values())
                if total_count > 0 and quality_count / total_count > 0.5:
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
            quality_pct = quality / total * 100 if total > 0 else 0
            top_issue = max(issues.items(), key=lambda x: x[1])[0] if issues else 'N/A'
            
            product_data.append({
                'SKU': sku,
                'Total Returns': total,
                'Quality Issues': quality,
                'Quality %': quality_pct,
                'Top Issue': top_issue,
                'Risk Score': quality * (quality_pct / 100)  # Weighted risk
            })
        
        # Sort by total returns
        product_data.sort(key=lambda x: x['Total Returns'], reverse=True)
        
        # Display top products
        for i, product in enumerate(product_data[:10]):
            if i < 5:  # Show first 5 in detail
                # Determine risk color
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
                # Show remaining as simple list
                st.markdown(f"{i+1}. **{product['SKU'][:30]}...**: {product['Total Returns']} returns ({product['Quality %']:.0f}% quality)")
        
        # Option to export full product analysis
        if st.button("📥 Export Full SKU Analysis"):
            # Create detailed product export
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
    desc_col = 'Description' if 'Description' in df.columns else None
    
    # Pre-process data for AI batching
    items_to_process = []
    
    for idx, row in df.iterrows():
        subject = str(row.get(display_col, ''))
        description = str(row.get(desc_col, ''))
        
        # Extract Main SKU using strict logic
        main_sku = find_sku_in_row(row)
        
        items_to_process.append({
            'index': idx,
            'subject': subject,
            'details': strip_html(description)[:1000], # Increased text limit for better context
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
        
        # Use the new summarize_batch method in EnhancedAIAnalyzer
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
    
    # Construct Final DataFrame - STRICTLY matching requested format
    # Display Name, Description, SKU, Reason
    final_rows = []
    for item in processed_results:
        final_rows.append({
            'Display Name': item['subject'],
            'Description': item['full_description'],
            'SKU': item['sku'],
            'Reason': item['summary']
        })
        
    return pd.DataFrame(final_rows)

# -------------------------
# TAB 3: QUALITY CASE SCREENING
# -------------------------

CATEGORY_BENCHMARKS = {
    'B2B Products (All)': 0.025,
    'INS': 0.07,
    'RHB': 0.075,
    'LVA': 0.095,
    'MOB - Power Scooters': 0.095,
    'MOB - Walkers/Rollators/other': 0.10,
    'MOB - Wheelchairs (manual)': 0.105,
    'CSH': 0.105,
    'SUP': 0.11,
    'MOB - Wheelchairs (power)': 0.115,
    'All Others': 0.10,
}

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
    all_values = np.concatenate([g for g in groups if len(g) > 0])
    if len(all_values) == 0 or len(groups) < 2:
        return None

    grand_mean = np.mean(all_values)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups if len(g) > 0)
    ss_within = sum(np.sum((g - np.mean(g)) ** 2) for g in groups if len(g) > 0)

    df_between = len(groups) - 1
    df_within = len(all_values) - len(groups)
    if df_within <= 0:
        return None

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    if ms_within == 0:
        return None

    return {
        'f_stat': ms_between / ms_within,
        'df_between': df_between,
        'df_within': df_within
    }

def manova_wilks_lambda(groups, metric_cols):
    valid_groups = [g for g in groups if len(g) > 1]
    if len(valid_groups) < 2:
        return None

    overall = np.vstack(valid_groups)
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
        'between_matrix': b_matrix
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
        launch_dates = pd.to_datetime(data[config['launch_date_col']], errors='coerce')
        recent = launch_dates.notna() & ((pd.Timestamp.utcnow() - launch_dates).dt.days <= config['launch_days'])
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

    review_date = pd.Timestamp.utcnow().normalize() + pd.Timedelta(days=config['review_days'])
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
                
@@ -1047,27 +1210,179 @@ def main():
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

        qc_file = st.file_uploader(
            "Upload Quality Screening Data (CSV/XLSX)",
            type=['csv', 'xlsx', 'xls', 'txt'],
            key="tab3_uploader"
        )

        if qc_file:
            qc_df = load_quality_case_file(qc_file.read(), qc_file.name)

            if qc_df is not None:
                st.markdown("#### 🧭 Column Mapping")
                columns = qc_df.columns.tolist()

                col1, col2, col3 = st.columns(3)
                with col1:
                    sku_col = st.selectbox("SKU Column", options=columns, index=0)
                    category_col = st.selectbox("Category Column", options=columns, index=min(1, len(columns) - 1))
                    sold_col = st.selectbox("Sold Column", options=columns)
                with col2:
                    returned_col = st.selectbox("Returned Column", options=columns)
                    landed_cost_col = st.selectbox("Landed Cost Column (Optional)", options=[""] + columns)
                    retail_price_col = st.selectbox("Retail Price Column (Optional)", options=[""] + columns)
                with col3:
                    launch_date_col = st.selectbox("Launch Date Column (Optional)", options=[""] + columns)
                    safety_col = st.selectbox("Safety Risk Column (Optional)", options=[""] + columns)
                    zero_tolerance_col = st.selectbox("Zero Tolerance Component Column (Optional)", options=[""] + columns)

                st.markdown("#### 📈 Trend-Based Thresholds")
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
                    aql_fail_col = st.selectbox("AQL Fail Column (Optional)", options=[""] + columns)
                with col9:
                    complaint_count_col = st.selectbox("Unique Complaint Count (30d) (Optional)", options=[""] + columns)
                    sales_units_30d_col = st.selectbox("Sales Units (30d) (Optional)", options=[""] + columns)
                    sales_value_30d_col = st.selectbox("Sales Value $ (30d) (Optional)", options=[""] + columns)

                st.markdown("#### 📊 ANOVA / MANOVA Inputs")
                manova_metric_1 = st.selectbox("MANOVA Metric 1", options=columns, index=columns.index(returned_col))
                manova_metric_2 = st.selectbox("MANOVA Metric 2", options=columns, index=columns.index(sold_col))

                if st.button("🔍 Run Quality Case Screening", type="primary"):
                    config = {
                        'sold_col': sold_col,
                        'returned_col': returned_col,
                        'category_col': category_col,
                        'use_benchmarks': use_benchmarks,
                        'return_rate_cap': return_rate_cap,
                        'relative_threshold': relative_threshold,
                        'cat_avg_delta': cat_avg_delta,
                        'low_volume_cutoff': int(low_volume_cutoff),
                        'review_days': int(review_days),
                        'safety_col': safety_col or None,
                        'zero_tolerance_col': zero_tolerance_col or None,
                        'landed_cost_col': landed_cost_col or None,
                        'retail_price_col': retail_price_col or None,
                        'launch_date_col': launch_date_col or None,
                        'launch_days': int(launch_days),
                        'aql_fail_col': aql_fail_col or None,
                        'landed_cost_threshold': landed_cost_threshold,
                        'retail_price_threshold': retail_price_threshold,
                        'complaint_count_col': complaint_count_col or None,
                        'sales_units_30d_col': sales_units_30d_col or None,
                        'sales_value_30d_col': sales_value_30d_col or None,
                    }

                    with st.spinner("Analyzing quality triggers..."):
                        screened = compute_quality_case_screening(qc_df, config)

                    st.markdown("#### ✅ Screening Results")
                    st.dataframe(
                        screened[
                            [
                                sku_col,
                                category_col,
                                'Return_Rate_Display',
                                'Cat_Avg_Display',
                                'Recommended_Action',
                                'Review_By'
                            ]
                        ],
                        use_container_width=True
                    )

                    st.markdown("#### 📊 ANOVA / MANOVA Summary")
                    grouped = [
                        parse_numeric(screened.loc[screened[category_col] == cat, 'Return_Rate']).values
                        for cat in screened[category_col].dropna().unique()
                    ]
                    anova_results = one_way_anova(grouped)
                    if anova_results:
                        st.info(
                            f"ANOVA F={anova_results['f_stat']:.4f} "
                            f"(df={anova_results['df_between']}, {anova_results['df_within']})"
                        )
                    else:
                        st.warning("ANOVA requires at least two categories with data.")

                    manova_groups = []
                    for cat in screened[category_col].dropna().unique():
                        subset = screened[screened[category_col] == cat][[manova_metric_1, manova_metric_2]]
                        subset = subset.apply(parse_numeric)
                        if len(subset) > 1:
                            manova_groups.append(subset.values)

                    manova_results = manova_wilks_lambda(manova_groups, [manova_metric_1, manova_metric_2])
                    if manova_results:
                        st.info(f"MANOVA Wilks' Lambda={manova_results['wilks_lambda']:.4f}")
                    else:
                        st.warning("MANOVA requires at least two categories with 2+ rows.")

                    st.markdown("#### ⬇️ Export Screening Data")
                    export_buffer = io.BytesIO()
                    with pd.ExcelWriter(export_buffer, engine='xlsxwriter') as writer:
                        screened.to_excel(writer, index=False, sheet_name='Quality Screening')
                    export_buffer.seek(0)
                    st.download_button(
                        label="Download Screening Results",
                        data=export_buffer,
                        file_name=f"quality_screening_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary"
                    )

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
