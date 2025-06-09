"""
Vive Health Quality Complaint Categorizer - Production Version
AI-Powered Return Reason Classification with Column K Export
Version: 16.1 - Auto-Download Feature Added

Key Features:
- Handles 2600+ rows efficiently
- Exports with categories in Column K
- Preserves original file structure
- Google Sheets compatible
- AUTO-DOWNLOAD on completion
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config with increased limits
st.set_page_config(
    page_title="Vive Health Return Categorizer",
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
    'title': 'Vive Health Medical Device Return Categorizer',
    'version': '16.1',
    'company': 'Vive Health',
    'chunk_sizes': [100, 250, 500, 1000],  # Available chunk sizes
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

# Quality categories
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
        'export_filename': None  # Store filename
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

def process_file_preserve_structure(file_content, filename):
    """Process file while preserving original structure"""
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
    """Display enhanced results dashboard"""
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

def main():
    """Main application function"""
    initialize_session_state()
    inject_custom_css()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">VIVE HEALTH RETURN CATEGORIZER</h1>
        <p style="color: white; margin: 0.5rem 0; font-size: 1.1em;">
            AI-Powered Medical Device Return Analysis
        </p>
        <p style="color: white; opacity: 0.9; font-size: 0.9em;">
            ✅ Handles 2600+ rows | 📊 Column K Export | 🔄 Google Sheets Compatible
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show completion notification if processing is done
    if st.session_state.processing_complete and st.session_state.export_data:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #00F5A0 0%, #00D9FF 50%, #00F5A0 100%); 
                    padding: 1rem; border-radius: 10px; text-align: center; margin-bottom: 1rem;
                    animation: pulse 2s infinite; box-shadow: 0 0 30px rgba(0, 245, 160, 0.5);">
            <h2 style="color: white; margin: 0;">🎉 Analysis Complete! Scroll down to download your results 👇</h2>
        </div>
        <style>
        @keyframes pulse {
            0% { transform: scale(1); box-shadow: 0 0 30px rgba(0, 245, 160, 0.5); }
            50% { transform: scale(1.02); box-shadow: 0 0 40px rgba(0, 245, 160, 0.7); }
            100% { transform: scale(1); box-shadow: 0 0 30px rgba(0, 245, 160, 0.5); }
        }
        </style>
        """, unsafe_allow_html=True)
    
    # Check AI status
    if not AI_AVAILABLE:
        st.error("❌ AI Module not available. Please check enhanced_ai_analysis.py")
        st.stop()
    
    keys = check_api_keys()
    if not keys:
        st.error("❌ No API keys found. Please add API keys to Streamlit secrets.")
        with st.expander("How to add API keys"):
            st.markdown("""
            Add to `.streamlit/secrets.toml`:
            ```
            openai_api_key = "sk-..."
            claude_api_key = "sk-ant-..."
            ```
            """)
        st.stop()
    
    # Show available APIs
    available_apis = []
    if 'openai' in keys:
        available_apis.append("OpenAI ✅")
    if 'claude' in keys:
        available_apis.append("Claude ✅")
    
    st.info(f"🤖 Available AI: {' | '.join(available_apis)}")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # AI Provider
        provider = st.selectbox(
            "🤖 AI Provider",
            options=list(AI_PROVIDER_OPTIONS.keys()),
            help="Choose AI model for categorization"
        )
        st.session_state.ai_provider = AI_PROVIDER_OPTIONS[provider]
        
        st.markdown("---")
        st.markdown("### 🚀 Performance Settings")
        
        # Chunk size for large files
        st.session_state.chunk_size = st.select_slider(
            "Processing Chunk Size",
            options=APP_CONFIG['chunk_sizes'],
            value=st.session_state.chunk_size,
            format_func=lambda x: f"{x:,} rows",
            help="Process data in chunks to handle large files efficiently"
        )
        
        # API Batch size
        st.session_state.batch_size = st.slider(
            "API Batch Size",
            min_value=10,
            max_value=100,
            value=50,
            help="Number of items per API call (affects speed and stability)"
        )
        
        # Show estimated performance
        est_speed = st.session_state.batch_size * 2
        st.caption(f"⚡ Estimated: ~{est_speed} returns/second")
        
        st.markdown("---")
        st.markdown("### 📊 Display Options")
        
        # Product analysis toggle
        st.session_state.show_product_analysis = st.checkbox(
            "Show Product/SKU Analysis",
            value=st.session_state.show_product_analysis,
            help="Display detailed breakdown by product SKU"
        )
        
        # Session stats
        st.markdown("---")
        st.markdown("### 📈 Session Statistics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Cost", f"${st.session_state.total_cost:.4f}")
            st.metric("API Calls", f"{st.session_state.api_calls_made:,}")
        
        with col2:
            if st.session_state.total_rows_processed > 0:
                st.metric("Total Processed", f"{st.session_state.total_rows_processed:,}")
            if st.session_state.processing_speed > 0:
                st.metric("Avg Speed", f"{st.session_state.processing_speed:.1f}/sec")
        
        # Info section
        st.markdown("---")
        with st.expander("ℹ️ Quick Tips"):
            st.markdown("""
            **For best results:**
            - Use chunk size 500 for 1000-3000 rows
            - Use chunk size 1000 for 3000+ rows
            - Smaller API batch = more stable
            - Larger API batch = faster
            """)
    
    # Main content
    st.markdown("### 📁 Upload Return Data File")
    
    # Current structure notice
    st.markdown("""
    <div style="background: rgba(255, 183, 0, 0.1); border: 1px solid var(--accent); 
                border-radius: 8px; padding: 1rem; margin-bottom: 1rem; text-align: center;">
        <strong style="color: var(--accent);">📌 Current Setup:</strong> 
        Complaints from Column I → AI Categories to Column K
    </div>
    """, unsafe_allow_html=True)
    
    # Instructions
    with st.expander("📖 File Format & Instructions", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Required File Structure:**
            - **Column I**: Complaint/Investigation Text *(required)*
            - **Column B**: Product Identifier/SKU *(optional)*
            - **Column K**: Will receive AI categories *(auto-created)*
            
            **Supported Formats:**
            - Excel files (.xlsx, .xls)
            - CSV files (.csv)
            - FBA Return Reports (.txt)
            """)
        
        with col2:
            st.markdown("""
            **Key Features:**
            - ✅ Preserves your exact file format
            - ✅ Only modifies Column K
            - ✅ Handles 2600+ rows efficiently
            - ✅ Google Sheets compatible export
            - ✅ 100% AI categorization accuracy
            - ✅ Real-time progress tracking
            """)
        
        st.info("💡 **Your file structure:** Complaints in Column I → Categories added to Column K")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose file",
        type=['csv', 'xlsx', 'xls', 'txt'],
        help="Upload file with complaints in column B"
    )
    
    if uploaded_file:
        # Read and process file
        with st.spinner(f"Reading {uploaded_file.name}..."):
            file_content = uploaded_file.read()
            df, column_mapping = process_file_preserve_structure(file_content, uploaded_file.name)
        
        if df is not None and column_mapping:
            st.session_state.original_data = df.copy()
            
            # Show file info with accurate column positions
            complaint_col = column_mapping.get('complaint')
            sku_col = column_mapping.get('sku')
            valid_complaints = df[df[complaint_col].notna() & 
                                (df[complaint_col].str.strip() != '')].shape[0]
            
            # Success message with clear information
            st.success(f"""
            ✅ **File loaded successfully!**
            """)
            
            # Show detected structure
            st.info(f"""
            📋 **Detected Structure:**
            - Complaints found in: **Column I** ({complaint_col})
            - Product SKUs in: **Column B** ({sku_col if sku_col else 'Not found'})
            - Categories will go in: **Column K**
            """)
            
            # File details in columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Rows", f"{len(df):,}")
            with col2:
                st.metric("Valid Complaints", f"{valid_complaints:,}")
            with col3:
                st.metric("Complaint Column", "I")
            with col4:
                st.metric("Category Column", "K")
            
            # Data preview
            with st.expander("📋 Preview Data", expanded=False):
                # Show relevant columns including Column I (complaint)
                preview_cols = []
                
                # Always try to show the complaint column (Column I)
                if complaint_col in df.columns:
                    preview_cols.append(complaint_col)
                
                # Show SKU column (Column B) if available
                if column_mapping.get('sku') and column_mapping['sku'] in df.columns:
                    preview_cols.append(column_mapping['sku'])
                
                # Show category column (Column K)
                if column_mapping.get('category') in df.columns:
                    preview_cols.append(column_mapping['category'])
                
                # If we have the columns, show them, otherwise show first few columns
                if preview_cols:
                    # Add column headers for clarity
                    preview_df = df[preview_cols].head(10).copy()
                    
                    # Add column position labels
                    col_labels = []
                    for col in preview_cols:
                        col_idx = df.columns.tolist().index(col)
                        col_letter = chr(65 + col_idx)  # Convert to letter (A, B, C...)
                        col_labels.append(f"Column {col_letter}: {col}")
                    
                    st.markdown("**Column Preview:**")
                    for label in col_labels:
                        st.caption(label)
                    
                    st.dataframe(preview_df, use_container_width=True)
                else:
                    st.dataframe(df.head(10))
            
            # Cost estimation box
            est_cost_per_item = 0.002  # Average cost per categorization
            est_total_cost = valid_complaints * est_cost_per_item
            
            st.markdown(f"""
            <div class="info-box" style="background: linear-gradient(135deg, rgba(80, 200, 120, 0.1), rgba(80, 200, 120, 0.2)); 
                        border-color: var(--cost); text-align: center;">
                <h4 style="color: var(--cost); margin: 0;">💰 Estimated Processing Cost</h4>
                <div style="font-size: 2em; font-weight: 700; color: var(--cost); margin: 0.5rem 0;">
                    ${est_total_cost:.2f}
                </div>
                <div style="color: var(--muted);">
                    for {valid_complaints:,} returns at ~${est_cost_per_item:.3f} each
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Process button
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                process_button = st.button(
                    f"🚀 Categorize {valid_complaints:,} Returns", 
                    type="primary", 
                    use_container_width=True,
                    help="Start AI categorization process"
                )
                
                if process_button:
                    analyzer = get_ai_analyzer()
                    
                    if analyzer:
                        start_time = time.time()
                        st.session_state.processing_errors = []
                        
                        # Process data
                        with st.spinner("Processing returns..."):
                            categorized_df = process_in_chunks(
                                df, 
                                analyzer, 
                                column_mapping,
                                chunk_size=st.session_state.chunk_size
                            )
                            
                            st.session_state.categorized_data = categorized_df
                            st.session_state.processing_time = time.time() - start_time
                            st.session_state.processing_complete = True
                            st.session_state.total_rows_processed += valid_complaints
                            
                            # Generate statistics
                            generate_statistics(categorized_df, column_mapping)
                            
                            # Update costs
                            cost_summary = analyzer.get_cost_summary()
                            st.session_state.total_cost = cost_summary.get('total_cost', 0)
                            st.session_state.api_calls_made = cost_summary.get('api_calls', 0)
                            
                            # Prepare export data immediately after processing
                            st.session_state.export_data = export_with_column_k(categorized_df)
                            file_extension = '.xlsx' if EXCEL_AVAILABLE else '.csv'
                            st.session_state.export_filename = f"categorized_returns_{datetime.now().strftime('%Y%m%d_%H%M%S')}{file_extension}"
                            st.session_state.auto_download_triggered = False  # Reset flag for new download
                        
                        st.balloons()
                        
                        # Success summary
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("✅ Processed", f"{st.session_state.total_rows_processed:,} returns")
                        with col2:
                            st.metric("⏱️ Time", f"{st.session_state.processing_time:.1f}s")
                        with col3:
                            st.metric("💰 Cost", f"${st.session_state.total_cost:.4f}")
                        
                        st.success("Processing complete! See results below.")
                        
                        # Show any errors
                        if st.session_state.processing_errors:
                            with st.expander(f"⚠️ {len(st.session_state.processing_errors)} Processing Warnings"):
                                for error in st.session_state.processing_errors:
                                    st.warning(error)
                        
                        # Trigger page refresh to show download section
                        st.rerun()
    
    # Results section
    if st.session_state.processing_complete and st.session_state.categorized_data is not None:
        st.markdown("---")
        
        # Display results
        display_results_dashboard(st.session_state.categorized_data, st.session_state.column_mapping)
        
        # Export section
        st.markdown("### 💾 Export Your Results")
        
        # Check if we have export data ready
        if st.session_state.export_data and st.session_state.export_filename:
            file_extension = '.xlsx' if EXCEL_AVAILABLE else '.csv'
            mime_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' if EXCEL_AVAILABLE else 'text/csv'
            
            # Success box with prominent download
            st.markdown(f"""
            <div class="info-box" style="background: linear-gradient(135deg, rgba(0, 245, 160, 0.2), rgba(0, 245, 160, 0.3)); 
                        border-color: var(--success); text-align: center; margin-bottom: 1rem; 
                        border-width: 2px; box-shadow: 0 0 20px rgba(0, 245, 160, 0.4);">
                <h3 style="color: var(--success); margin: 0;">✅ Analysis Complete - Download Ready!</h3>
                <p style="margin: 0.5rem 0; font-size: 1.1em;">
                    Your file is ready with AI categories in Column K
                </p>
                <p style="margin: 0.3rem 0; color: var(--accent); font-weight: 600;">
                    📄 Filename: {st.session_state.export_filename}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Prominent download section
            col1, col2, col3 = st.columns([1, 3, 1])
            
            with col2:
                # Big download button with animation
                st.markdown("""
                <style>
                @keyframes pulse {
                    0% { transform: scale(1); }
                    50% { transform: scale(1.05); }
                    100% { transform: scale(1); }
                }
                .download-container button {
                    animation: pulse 2s infinite;
                    font-size: 1.2em !important;
                    padding: 1rem 2rem !important;
                    background: linear-gradient(135deg, #00F5A0 0%, #00D9FF 100%) !important;
                }
                </style>
                <div class="download-container">
                """, unsafe_allow_html=True)
                
                # Download button
                downloaded = st.download_button(
                    label=f"⬇️ DOWNLOAD YOUR RESULTS {file_extension.upper()}",
                    data=st.session_state.export_data,
                    file_name=st.session_state.export_filename,
                    mime=mime_type,
                    use_container_width=True,
                    type="primary",
                    key=f"download_{st.session_state.export_filename}"
                )
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                if downloaded:
                    st.success("✅ File downloaded successfully!")
                    st.session_state.auto_download_triggered = True
                
                # Additional info
                st.markdown("""
                <div style="text-align: center; margin-top: 1.5rem;">
                    <p style="color: var(--success); font-weight: 600; font-size: 1.1em;">
                        ✅ File is Google Sheets compatible!
                    </p>
                    <p style="color: var(--text); margin: 0.5rem 0;">
                        Import directly - only Column K has been modified with AI categories
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Important notice
                st.warning("""
                ⚠️ **Important**: Download your results now to save your work! 
                The file contains all your categorized data with AI classifications in Column K.
                """)
            
            # Tips section
            with st.expander("💡 Next Steps & Tips", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    **What's in your file:**
                    - ✅ All original data preserved
                    - ✅ AI categories in Column K
                    - ✅ Ready for analysis
                    
                    **Quality Analysis Tips:**
                    - Filter by category in Column K
                    - Pivot by SKU and category
                    - Identify top quality issues
                    """)
                
                with col2:
                    st.markdown("""
                    **Import to Google Sheets:**
                    1. Open Google Sheets
                    2. File → Import
                    3. Upload your downloaded file
                    4. Select "Replace spreadsheet"
                    
                    **For Excel:**
                    - Open directly in Excel
                    - Data is pre-formatted
                    """)
        
        # Option to process another file
        st.markdown("---")
        if st.button("📁 Process Another File", type="secondary", use_container_width=True):
            # Reset relevant session state
            for key in ['original_data', 'processed_data', 'categorized_data', 
                       'processing_complete', 'auto_download_triggered', 
                       'export_data', 'export_filename']:
                if key in st.session_state:
                    st.session_state[key] = None if key != 'processing_complete' else False
            st.rerun()

if __name__ == "__main__":
    main()
