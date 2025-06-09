"""
Vive Health Quality Complaint Categorizer - Production Version
AI-Powered Return Reason Classification with Column K Export
Version: 16.0 - Handles Large Datasets & Preserves File Format

Key Features:
- Handles 2600+ rows efficiently
- Exports with categories in Column K
- Preserves original file structure
- Google Sheets compatible
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
    'version': '16.0',
    'company': 'Vive Health',
    'max_batch_size': 100,  # Larger batches for efficiency
    'chunk_size': 500,  # Process in chunks for large files
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
        'processing_errors': [],
        'total_rows_processed': 0,
        'column_mapping': {}  # Track original column positions
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
        
        # Create column mapping
        column_mapping = {}
        
        # Check if columns are labeled A, B, C, etc.
        excel_style_columns = all(col in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'] 
                                 for col in df.columns[:11] if col in df.columns)
        
        if excel_style_columns:
            # Excel-style columns
            column_mapping['complaint'] = 'B'
            column_mapping['sku'] = 'C'
            column_mapping['category'] = 'K'
        else:
            # Try to identify columns by position or name
            cols = df.columns.tolist()
            
            # Find complaint column (usually 2nd column, index 1)
            if len(cols) > 1:
                complaint_col = None
                for col in ['Complaint', 'complaint', 'Return Reason', 'Reason', 'Comments']:
                    if col in cols:
                        complaint_col = col
                        break
                
                if not complaint_col and len(cols) > 1:
                    complaint_col = cols[1]  # Use second column
                
                column_mapping['complaint'] = complaint_col
            
            # Find SKU column (usually 3rd column, index 2)
            if len(cols) > 2:
                sku_col = None
                for col in ['SKU', 'sku', 'Product SKU', 'Product Identifier Tag', 'ASIN']:
                    if col in cols:
                        sku_col = col
                        break
                
                if not sku_col and len(cols) > 2:
                    sku_col = cols[2]  # Use third column
                
                column_mapping['sku'] = sku_col
            
            # Find or create category column (column K or 11th column)
            if len(cols) >= 11:
                column_mapping['category'] = cols[10]  # 11th column (index 10) = column K
            else:
                # Need to add columns up to K
                while len(df.columns) < 11:
                    df[f'Column_{len(df.columns)}'] = ''
                column_mapping['category'] = df.columns[10]
        
        # Ensure column K exists and is empty
        if column_mapping.get('category'):
            df[column_mapping['category']] = ''
        
        # Store mapping
        st.session_state.column_mapping = column_mapping
        
        # Validate we have complaint data
        if 'complaint' not in column_mapping or column_mapping['complaint'] not in df.columns:
            st.error("Cannot find complaint/return reason column (expected in column B)")
            return None, None
        
        # Count valid complaints
        complaint_col = column_mapping['complaint']
        valid_complaints = df[df[complaint_col].notna() & (df[complaint_col].str.strip() != '')].copy()
        
        logger.info(f"File structure: {len(df)} total rows, {len(valid_complaints)} with complaints")
        logger.info(f"Column mapping: {column_mapping}")
        
        return df, column_mapping
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        logger.error(f"File processing error: {e}")
        return None, None

def process_in_chunks(df, analyzer, column_mapping, chunk_size=500):
    """Process large datasets in chunks"""
    complaint_col = column_mapping['complaint']
    category_col = column_mapping['category']
    
    # Get rows with complaints
    valid_indices = df[df[complaint_col].notna() & (df[complaint_col].str.strip() != '')].index
    total_valid = len(valid_indices)
    
    if total_valid == 0:
        st.warning("No valid complaints found to process")
        return df
    
    st.info(f"Processing {total_valid} returns in chunks of {chunk_size}...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    processed_count = 0
    start_time = time.time()
    
    # Process in chunks
    for chunk_start in range(0, total_valid, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_valid)
        chunk_indices = valid_indices[chunk_start:chunk_end]
        
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
            sub_batch_size = min(st.session_state.batch_size, 50)
            
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
                
                status_text.text(
                    f"Processed: {processed_count}/{total_valid} | "
                    f"Speed: {speed:.1f}/sec | "
                    f"ETA: {remaining:.0f}s"
                )
                
                # Small delay to prevent overwhelming
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Chunk processing error: {e}")
            st.session_state.processing_errors.append(f"Chunk {chunk_start//chunk_size + 1}: {str(e)}")
            
            # Fill failed items with default category
            for item in batch_data:
                if pd.isna(df.at[item['index'], category_col]):
                    df.at[item['index'], category_col] = 'Other/Miscellaneous'
        
        # Force garbage collection after each chunk
        gc.collect()
    
    # Final update
    progress_bar.progress(1.0)
    elapsed = time.time() - start_time
    status_text.text(f"✅ Complete! Processed {processed_count} returns in {elapsed:.1f}s")
    
    return df

def generate_statistics(df, column_mapping):
    """Generate statistics from categorized data"""
    category_col = column_mapping.get('category')
    sku_col = column_mapping.get('sku')
    
    if not category_col:
        return
    
    # Category statistics
    category_counts = df[category_col].value_counts()
    category_counts = category_counts[category_counts.index != '']  # Remove empty
    
    st.session_state.reason_summary = category_counts.to_dict()
    
    # SKU statistics
    if sku_col and sku_col in df.columns:
        product_summary = defaultdict(lambda: defaultdict(int))
        
        for _, row in df.iterrows():
            if pd.notna(row[category_col]) and row[category_col] != '':
                sku = str(row[sku_col]) if pd.notna(row[sku_col]) else 'Unknown'
                category = row[category_col]
                product_summary[sku][category] += 1
        
        st.session_state.product_summary = dict(product_summary)

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
    """Display results dashboard"""
    st.markdown("### 📊 Processing Results")
    
    # Calculate metrics
    total_rows = len(df)
    complaint_col = column_mapping.get('complaint')
    category_col = column_mapping.get('category')
    
    categorized_rows = len(df[df[category_col].notna() & (df[category_col] != '')])
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rows", f"{total_rows:,}")
    
    with col2:
        st.metric("Categorized", f"{categorized_rows:,}")
    
    with col3:
        success_rate = categorized_rows / total_rows * 100 if total_rows > 0 else 0
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    with col4:
        st.metric("Processing Time", f"{st.session_state.processing_time:.1f}s")
    
    # Category breakdown
    if st.session_state.reason_summary:
        st.markdown("#### Category Distribution")
        
        # Show top categories
        top_categories = sorted(st.session_state.reason_summary.items(), 
                              key=lambda x: x[1], reverse=True)[:10]
        
        for cat, count in top_categories:
            pct = count / categorized_rows * 100 if categorized_rows > 0 else 0
            
            color = COLORS['danger'] if cat in QUALITY_CATEGORIES else COLORS['primary']
            st.markdown(f"""
            <div style="margin: 0.5rem 0;">
                <div style="display: flex; justify-content: space-between;">
                    <span>{cat}</span>
                    <span>{count} ({pct:.1f}%)</span>
                </div>
                <div style="background: #333; height: 10px; border-radius: 5px;">
                    <div style="background: {color}; width: {pct}%; height: 100%; border-radius: 5px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

def main():
    """Main application function"""
    initialize_session_state()
    inject_custom_css()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">VIVE HEALTH RETURN CATEGORIZER</h1>
        <p style="color: white; margin: 0.5rem 0;">
            AI-Powered Categorization | Column K Export | Google Sheets Ready
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check AI status
    if not AI_AVAILABLE:
        st.error("❌ AI Module not available. Please check enhanced_ai_analysis.py")
        st.stop()
    
    keys = check_api_keys()
    if not keys:
        st.error("❌ No API keys found. Add to Streamlit secrets.")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        # AI Provider
        provider = st.selectbox(
            "AI Provider",
            options=list(AI_PROVIDER_OPTIONS.keys())
        )
        st.session_state.ai_provider = AI_PROVIDER_OPTIONS[provider]
        
        # Batch size
        st.session_state.batch_size = st.slider(
            "Batch Size",
            min_value=10,
            max_value=100,
            value=50,
            help="Larger = faster but uses more memory"
        )
        
        # Session stats
        st.markdown("---")
        st.markdown("### 📊 Session Stats")
        st.metric("Total Cost", f"${st.session_state.total_cost:.4f}")
        st.metric("API Calls", f"{st.session_state.api_calls_made:,}")
        
        if st.session_state.total_rows_processed > 0:
            st.metric("Rows Processed", f"{st.session_state.total_rows_processed:,}")
    
    # Main content
    st.markdown("### 📁 Upload Return Data File")
    
    # Instructions
    with st.expander("📖 Important: File Format Requirements", expanded=True):
        st.markdown("""
        **Your file structure will be preserved!**
        
        - **Column B**: Must contain complaint/return reason
        - **Column C**: Should contain SKU (optional)
        - **Column K**: Will be populated with AI categories
        - All other columns remain unchanged
        
        **Supported formats:** CSV, Excel (.xlsx, .xls), FBA Reports (.txt)
        
        **✅ The exported file will match your original format with only Column K modified**
        """)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose file",
        type=['csv', 'xlsx', 'xls', 'txt'],
        help="Upload file with complaints in column B"
    )
    
    if uploaded_file:
        # Read and process file
        file_content = uploaded_file.read()
        df, column_mapping = process_file_preserve_structure(file_content, uploaded_file.name)
        
        if df is not None and column_mapping:
            st.session_state.original_data = df.copy()
            
            # Show file info
            complaint_col = column_mapping.get('complaint')
            valid_complaints = df[df[complaint_col].notna() & 
                                (df[complaint_col].str.strip() != '')].shape[0]
            
            st.success(f"""
            ✅ File loaded successfully!
            - Total rows: {len(df):,}
            - Rows with complaints: {valid_complaints:,}
            - Complaint column: {complaint_col}
            - Categories will be added to: Column K
            """)
            
            # Preview
            with st.expander("Preview Data"):
                st.dataframe(df.head(10))
            
            # Cost estimate
            est_cost = valid_complaints * 0.002  # Rough estimate
            st.info(f"💰 Estimated cost: ${est_cost:.2f} for {valid_complaints:,} returns")
            
            # Process button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 Categorize Returns", type="primary", use_container_width=True):
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
                                chunk_size=APP_CONFIG['chunk_size']
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
                        
                        st.balloons()
                        st.success(f"""
                        ✅ Processing complete!
                        - Time: {st.session_state.processing_time:.1f}s
                        - Speed: {valid_complaints/st.session_state.processing_time:.1f} returns/sec
                        - Cost: ${st.session_state.total_cost:.4f}
                        """)
                        
                        # Show any errors
                        if st.session_state.processing_errors:
                            with st.expander(f"⚠️ {len(st.session_state.processing_errors)} Errors"):
                                for error in st.session_state.processing_errors:
                                    st.warning(error)
    
    # Results section
    if st.session_state.processing_complete and st.session_state.categorized_data is not None:
        st.markdown("---")
        
        # Display results
        display_results_dashboard(st.session_state.categorized_data, st.session_state.column_mapping)
        
        # Export section
        st.markdown("### 💾 Export Results")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            export_data = export_with_column_k(st.session_state.categorized_data)
            
            file_extension = '.xlsx' if EXCEL_AVAILABLE else '.csv'
            mime_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' if EXCEL_AVAILABLE else 'text/csv'
            
            st.download_button(
                label=f"📥 Download with Categories in Column K {file_extension.upper()}",
                data=export_data,
                file_name=f"categorized_returns_{datetime.now().strftime('%Y%m%d_%H%M%S')}{file_extension}",
                mime=mime_type,
                use_container_width=True,
                help="Download file with AI categories in Column K - ready for Google Sheets import"
            )
            
            st.success("""
            ✅ File ready for Google Sheets import!
            - Original structure preserved
            - Categories added to Column K
            - All other data unchanged
            """)

if __name__ == "__main__":
    main()
