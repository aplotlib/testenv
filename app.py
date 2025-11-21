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
    'version': '17.1',
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
        'batch_size': 50,
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
        'b2b_export_filename': None
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

def get_ai_analyzer():
    """Get or create AI analyzer"""
    if st.session_state.ai_analyzer is None and AI_AVAILABLE:
        try:
            keys = check_api_keys()
            
            if 'openai' in keys:
                os.environ['OPENAI_API_KEY'] = keys['openai']
            if 'claude' in keys:
                os.environ['ANTHROPIC_API_KEY'] = keys['claude']
            
            st.session_state.ai_analyzer = EnhancedAIAnalyzer(st.session_state.ai_provider)
            logger.info(f"Created AI analyzer with provider: {st.session_state.ai_provider.value}")
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

def generate_b2b_report(df, analyzer):
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
            'details': strip_html(description)[:500], # Limit text for API speed
            'full_description': description,
            'sku': main_sku
        })
    
    # Batch Process with AI
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    batch_size = st.session_state.batch_size
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
    
    # Construct Final DataFrame
    final_rows = []
    for item in processed_results:
        final_rows.append({
            'Main SKU': item['sku'],
            'Issue Subject': item['subject'],
            'AI Summary': item['summary'],
            'Full Description': item['full_description'],
            'SKU Found': 'Yes' if item['sku'] != 'Unknown' else 'No'
        })
        
    return pd.DataFrame(final_rows)

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
    tab1, tab2 = st.tabs(["📊 Return Categorizer", "📑 B2B Report Generator"])

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
                <li><strong>SKU Logic:</strong> Auto-extracts Main SKU (e.g., <code>MOB1027</code>) from Subject/Description using "3 Caps + 4 Digits" rule.</li>
                <li><strong>Filtering:</strong> Processes ALL tickets in upload (No pre-filtering).</li>
                <li><strong>AI Summary:</strong> Generates 10-word "Reason" summaries for every ticket.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        b2b_file = st.file_uploader("Upload Odoo Export (CSV/Excel)", type=['csv', 'xlsx'], key="b2b_uploader")
        
        if b2b_file:
            # 1. Read & Preview
            b2b_df = process_b2b_file(b2b_file.read(), b2b_file.name)
            
            if b2b_df is not None:
                st.markdown(f"**Total Tickets Found:** {len(b2b_df):,}")
                
                # 2. Process Button
                if st.button("⚡ Generate B2B Report", type="primary"):
                    analyzer = get_ai_analyzer()
                    
                    with st.spinner("Running AI Analysis & SKU Extraction..."):
                        # Run the B2B pipeline
                        final_b2b = generate_b2b_report(b2b_df, analyzer)
                        
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
                                worksheet.set_column(col_num, col_num, 20)

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
                sku_found_count = len(df_res[df_res['Main SKU'] != 'Unknown'])
                st.metric("SKUs Identified", f"{sku_found_count}", delta=f"{sku_found_count/len(df_res)*100:.1f}% coverage")
            with c3:
                unique_skus = df_res[df_res['Main SKU'] != 'Unknown']['Main SKU'].nunique()
                st.metric("Unique Products", unique_skus)
            
            # Preview Table
            st.markdown("#### Preview (Top 10)")
            st.dataframe(df_res.head(10), use_container_width=True)
            
            # Warnings if SKUs missing
            if sku_found_count < len(df_res):
                st.warning(f"⚠️ {len(df_res) - sku_found_count} tickets have 'Unknown' SKU. Please check the 'Full Description' column in the export.")
            
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

if __name__ == "__main__":
    main()
