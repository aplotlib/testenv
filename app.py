"""
Vive Health Quality Complaint Categorizer - Optimized Version
AI-Powered Return Reason Classification with OpenAI + Claude
Version: 15.0 - Timeout Prevention & Stability Focus

Key Features:
- Dual AI support (OpenAI + Claude)
- Timeout prevention with smaller batches
- Error recovery and retry logic
- 100% AI categorization
- Column C SKU tracking
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import io
from typing import Dict, List, Any, Optional, Tuple
import json
import time
from collections import Counter, defaultdict
import re
import os
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Vive Health Return Categorizer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import AI module with error handling
try:
    from enhanced_ai_analysis import (
        EnhancedAIAnalyzer, AIProvider, FBA_REASON_MAP,
        MEDICAL_DEVICE_CATEGORIES, process_dataframe_in_batches
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

try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# App Configuration
APP_CONFIG = {
    'title': 'Vive Health Medical Device Return Categorizer',
    'version': '15.0',
    'company': 'Vive Health',
    'batch_size': 10,  # Smaller batches to prevent timeouts
    'timeout': 25,  # Timeout for processing
    'max_retries': 3
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
    'Both (Highest Accuracy)': AIProvider.BOTH
}

def inject_custom_css():
    """Inject custom CSS styling"""
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
        background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #16213e 100%);
        color: var(--text);
        font-family: 'Inter', sans-serif;
    }}
    
    .main-header {{
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2.5rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 217, 255, 0.3);
    }}
    
    .main-title {{
        font-size: 2.5em;
        font-weight: 700;
        color: white;
        text-shadow: 0 0 20px rgba(255, 255, 255, 0.5);
        margin: 0;
    }}
    
    .info-box {{
        background: rgba(26, 26, 46, 0.8);
        border: 1px solid var(--primary);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }}
    
    .metric-card {{
        background: rgba(26, 26, 46, 0.9);
        border: 2px solid rgba(0, 217, 255, 0.3);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        transition: all 0.3s ease;
    }}
    
    .metric-card:hover {{
        transform: translateY(-3px);
        border-color: var(--primary);
        box-shadow: 0 8px 20px rgba(0, 217, 255, 0.3);
    }}
    
    .cost-box {{
        background: linear-gradient(135deg, rgba(80, 200, 120, 0.1) 0%, rgba(80, 200, 120, 0.2) 100%);
        border: 2px solid var(--cost);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 0 15px rgba(80, 200, 120, 0.3);
    }}
    
    .ai-badge {{
        display: inline-block;
        padding: 0.4rem 0.8rem;
        border-radius: 15px;
        font-weight: 600;
        margin: 0.2rem;
        font-size: 0.9em;
    }}
    
    .ai-badge.openai {{
        background: linear-gradient(135deg, var(--openai), rgba(0, 217, 255, 0.7));
        color: white;
    }}
    
    .ai-badge.claude {{
        background: linear-gradient(135deg, var(--claude), rgba(155, 89, 182, 0.7));
        color: white;
    }}
    
    .stButton > button {{
        font-weight: 600;
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        color: white;
        border: none;
        padding: 0.8rem 1.5rem;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0, 217, 255, 0.4);
        transition: all 0.3s ease;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 217, 255, 0.5);
    }}
    
    .stProgress > div > div {{
        background: linear-gradient(90deg, var(--primary), var(--accent));
        height: 8px;
        border-radius: 4px;
    }}
    
    #MainMenu, footer, header {{
        visibility: hidden;
    }}
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        'processed_data': None,
        'categorized_data': None,
        'ai_analyzer': None,
        'processing_complete': False,
        'reason_summary': {},
        'product_summary': {},
        'severity_counts': {},
        'quality_insights': None,
        'total_cost': 0.0,
        'api_calls_made': 0,
        'processing_time': 0.0,
        'ai_provider': AIProvider.FASTEST,
        'batch_size': APP_CONFIG['batch_size'],
        'date_filter_enabled': False,
        'date_range_start': datetime.now() - timedelta(days=30),
        'date_range_end': datetime.now(),
        'processing_errors': []
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def check_api_keys():
    """Check for API keys in Streamlit secrets"""
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
            
            # Set environment variables
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

def display_header():
    """Display application header"""
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">VIVE HEALTH RETURN CATEGORIZER</h1>
        <p style="font-size: 1.1em; color: white; margin: 0.5rem 0;">
            AI-Powered Medical Device Quality Management
        </p>
        <p style="font-size: 0.9em; color: white; opacity: 0.9;">
            ⚡ Dual AI Support | 📄 PDF Processing | 💰 Cost Tracking
        </p>
    </div>
    """, unsafe_allow_html=True)

def display_ai_status():
    """Display AI status"""
    if not AI_AVAILABLE:
        st.error("❌ AI Module Not Available - Please check enhanced_ai_analysis.py")
        return False
    
    keys = check_api_keys()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'openai' in keys:
            st.success("✅ OpenAI Connected")
        else:
            st.error("❌ OpenAI Not Configured")
    
    with col2:
        if 'claude' in keys:
            st.success("✅ Claude Connected")
        else:
            st.error("❌ Claude Not Configured")
    
    with col3:
        if keys:
            st.info(f"💰 Total Cost: ${st.session_state.total_cost:.4f}")
        else:
            st.warning("⚠️ No API Keys Found")
    
    if not keys:
        st.markdown("""
        <div class="info-box" style="border-color: var(--danger);">
            <h4 style="color: var(--danger);">Configuration Required</h4>
            <p>Add API keys to Streamlit secrets:</p>
            <pre>openai_api_key = "sk-..."
claude_api_key = "sk-ant-..."</pre>
        </div>
        """, unsafe_allow_html=True)
        return False
    
    return True

def process_file(file_content, filename, date_filter=None):
    """Process uploaded file"""
    try:
        # Determine file type and read
        if filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_content))
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(file_content))
        elif filename.endswith('.txt'):
            # Try tab-delimited first (FBA reports)
            df = pd.read_csv(io.BytesIO(file_content), sep='\t', encoding='utf-8')
        else:
            st.error(f"Unsupported file type: {filename}")
            return None
        
        # Handle column mapping
        if 'Complaint' not in df.columns:
            # Try to find complaint column
            if 'B' in df.columns:
                df['Complaint'] = df['B']
            elif 'customer-comments' in df.columns:
                df['Complaint'] = df['customer-comments']
            elif len(df.columns) > 1:
                # Use second column as complaint
                df['Complaint'] = df.iloc[:, 1]
            else:
                st.error("Cannot find complaint/return reason column")
                return None
        
        # Handle SKU column (Column C)
        if 'Product Identifier Tag' not in df.columns:
            if 'C' in df.columns:
                df['Product Identifier Tag'] = df['C']
            elif 'SKU' in df.columns:
                df['Product Identifier Tag'] = df['SKU']
            elif 'sku' in df.columns:
                df['Product Identifier Tag'] = df['sku']
            elif len(df.columns) > 2:
                # Use third column as SKU
                df['Product Identifier Tag'] = df.iloc[:, 2]
        
        # Handle FBA reason codes
        if 'reason' in df.columns:
            df['FBA_Reason_Code'] = df['reason']
        
        # Apply date filter if enabled
        if date_filter and st.session_state.date_filter_enabled:
            date_cols = ['Date', 'date', 'return-date', 'A']
            date_col = None
            
            for col in date_cols:
                if col in df.columns:
                    date_col = col
                    break
            
            if date_col:
                try:
                    df[date_col] = pd.to_datetime(df[date_col])
                    mask = (df[date_col] >= date_filter[0]) & (df[date_col] <= date_filter[1])
                    df = df[mask]
                except:
                    pass
        
        # Clean data
        df = df[df['Complaint'].notna() & (df['Complaint'].str.strip() != '')]
        
        # Add Category column
        df['Category'] = ''
        
        return df
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        logger.error(f"File processing error: {traceback.format_exc()}")
        return None

def categorize_with_timeout(df, analyzer, batch_size=10):
    """Categorize data with timeout protection"""
    total_rows = len(df)
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Process in smaller batches to prevent timeouts
    for i in range(0, total_rows, batch_size):
        batch_start = time.time()
        
        # Get batch
        batch = df.iloc[i:i+batch_size]
        batch_data = []
        
        for idx, row in batch.iterrows():
            batch_data.append({
                'index': idx,
                'complaint': str(row.get('Complaint', '')),
                'fba_reason': str(row.get('FBA_Reason_Code', '')) if 'FBA_Reason_Code' in row else None
            })
        
        try:
            # Process batch with timeout
            batch_results = analyzer.categorize_batch(batch_data, mode='standard')
            results.extend(batch_results)
            
            # Update progress
            progress = min((i + batch_size) / total_rows, 1.0)
            progress_bar.progress(progress)
            
            elapsed = time.time() - batch_start
            status_text.text(f"Processing: {min(i + batch_size, total_rows)}/{total_rows} | Batch time: {elapsed:.1f}s")
            
            # Small delay to prevent overwhelming
            if elapsed < 0.5:
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            st.session_state.processing_errors.append(f"Batch {i//batch_size + 1}: {str(e)}")
            
            # Add default results for failed batch
            for item in batch_data:
                results.append({
                    'index': item['index'],
                    'category': 'Other/Miscellaneous',
                    'confidence': 0.1,
                    'severity': 'none'
                })
    
    return results

def process_data():
    """Main processing function"""
    df = st.session_state.processed_data
    if df is None or df.empty:
        return
    
    analyzer = get_ai_analyzer()
    if not analyzer:
        st.error("AI analyzer not available")
        return
    
    start_time = time.time()
    st.session_state.processing_errors = []
    
    try:
        # Get results with timeout protection
        results = categorize_with_timeout(df, analyzer, st.session_state.batch_size)
        
        # Update dataframe
        category_counts = Counter()
        product_issues = defaultdict(lambda: defaultdict(int))
        
        for result in results:
            idx = result['index']
            category = result['category']
            
            df.at[idx, 'Category'] = category
            category_counts[category] += 1
            
            # Track by SKU
            if 'Product Identifier Tag' in df.columns:
                sku = str(df.at[idx, 'Product Identifier Tag']).strip()
                if sku and sku != 'nan':
                    product_issues[sku][category] += 1
        
        # Update session state
        st.session_state.categorized_data = df
        st.session_state.reason_summary = dict(category_counts)
        st.session_state.product_summary = dict(product_issues)
        st.session_state.processing_time = time.time() - start_time
        st.session_state.processing_complete = True
        
        # Update costs
        if analyzer:
            cost_summary = analyzer.get_cost_summary()
            st.session_state.total_cost = cost_summary.get('total_cost', 0)
            st.session_state.api_calls_made = cost_summary.get('api_calls', 0)
        
        # Show results
        st.success(f"""
        ✅ Processing Complete!
        - Time: {st.session_state.processing_time:.1f} seconds
        - Processed: {len(df)} returns
        - Cost: ${st.session_state.total_cost:.4f}
        - Errors: {len(st.session_state.processing_errors)}
        """)
        
        if st.session_state.processing_errors:
            with st.expander("⚠️ Processing Errors"):
                for error in st.session_state.processing_errors:
                    st.warning(error)
                    
    except Exception as e:
        st.error(f"Processing failed: {str(e)}")
        logger.error(f"Processing error: {traceback.format_exc()}")

def display_results():
    """Display results"""
    df = st.session_state.categorized_data
    if df is None:
        return
    
    st.markdown("### 📊 Results")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Returns", len(df))
    
    with col2:
        quality_count = sum(count for cat, count in st.session_state.reason_summary.items() 
                          if cat in QUALITY_CATEGORIES)
        st.metric("Quality Issues", f"{quality_count} ({quality_count/len(df)*100:.1f}%)")
    
    with col3:
        st.metric("Processing Time", f"{st.session_state.processing_time:.1f}s")
    
    with col4:
        st.metric("Total Cost", f"${st.session_state.total_cost:.4f}")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Categories", "📦 Products", "💾 Export"])
    
    with tab1:
        # Category breakdown
        st.markdown("#### Return Categories")
        
        category_df = pd.DataFrame([
            {
                'Category': cat,
                'Count': count,
                'Percentage': f"{count/len(df)*100:.1f}%",
                'Type': 'Quality' if cat in QUALITY_CATEGORIES else 'Other'
            }
            for cat, count in sorted(st.session_state.reason_summary.items(), 
                                    key=lambda x: x[1], reverse=True)
        ])
        
        st.dataframe(category_df, use_container_width=True, hide_index=True)
    
    with tab2:
        # Product analysis
        st.markdown("#### Product Analysis (by SKU)")
        
        if st.session_state.product_summary:
            product_data = []
            for sku, issues in st.session_state.product_summary.items():
                total = sum(issues.values())
                quality = sum(count for cat, count in issues.items() if cat in QUALITY_CATEGORIES)
                
                product_data.append({
                    'SKU': sku,
                    'Total Returns': total,
                    'Quality Issues': quality,
                    'Quality %': f"{quality/total*100:.1f}%" if total > 0 else "0%",
                    'Top Issue': max(issues.items(), key=lambda x: x[1])[0] if issues else 'N/A'
                })
            
            product_df = pd.DataFrame(product_data)
            product_df = product_df.sort_values('Total Returns', ascending=False)
            
            st.dataframe(product_df.head(50), use_container_width=True, hide_index=True)
            
            if len(product_data) > 50:
                st.caption(f"Showing top 50 of {len(product_data)} SKUs")
        else:
            st.info("No SKU data available")
    
    with tab3:
        # Export options
        st.markdown("#### Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV export
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv,
                f"categorized_returns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col2:
            # Excel export if available
            if EXCEL_AVAILABLE:
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, sheet_name='Returns', index=False)
                    
                    # Add summary sheet
                    summary_df = category_df
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                excel_data = output.getvalue()
                
                st.download_button(
                    "📥 Download Excel",
                    excel_data,
                    f"categorized_returns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            else:
                st.info("Install xlsxwriter for Excel export")

def main():
    """Main application function"""
    initialize_session_state()
    inject_custom_css()
    display_header()
    
    # Check AI status
    if not display_ai_status():
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        # AI Provider selection
        provider_name = st.selectbox(
            "AI Provider",
            options=list(AI_PROVIDER_OPTIONS.keys()),
            help="Choose AI provider for categorization"
        )
        st.session_state.ai_provider = AI_PROVIDER_OPTIONS[provider_name]
        
        # Batch size
        st.session_state.batch_size = st.slider(
            "Batch Size",
            min_value=5,
            max_value=20,
            value=10,
            help="Smaller = more stable, Larger = faster"
        )
        
        # Date filter
        st.markdown("### 📅 Date Filter")
        st.session_state.date_filter_enabled = st.checkbox("Enable date filtering")
        
        if st.session_state.date_filter_enabled:
            st.session_state.date_range_start = st.date_input("Start Date", st.session_state.date_range_start)
            st.session_state.date_range_end = st.date_input("End Date", st.session_state.date_range_end)
        
        # Stats
        st.markdown("### 📊 Session Stats")
        st.metric("API Calls", st.session_state.api_calls_made)
        st.metric("Total Cost", f"${st.session_state.total_cost:.4f}")
    
    # Main content
    st.markdown("### 📁 Upload Files")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload return files",
        type=['csv', 'xlsx', 'xls', 'txt'],
        accept_multiple_files=True,
        help="Upload CSV/Excel files with complaints in column B and SKUs in column C"
    )
    
    if uploaded_files:
        all_data = []
        
        # Process each file
        for file in uploaded_files:
            with st.spinner(f"Processing {file.name}..."):
                file_content = file.read()
                
                date_filter = None
                if st.session_state.date_filter_enabled:
                    date_filter = (st.session_state.date_range_start, st.session_state.date_range_end)
                
                df = process_file(file_content, file.name, date_filter)
                
                if df is not None:
                    all_data.append(df)
                    st.success(f"✅ {file.name}: {len(df)} returns loaded")
        
        if all_data:
            # Combine all data
            combined_df = pd.concat(all_data, ignore_index=True) if len(all_data) > 1 else all_data[0]
            st.session_state.processed_data = combined_df
            
            st.info(f"📊 Total returns ready for processing: {len(combined_df)}")
            
            # Preview
            with st.expander("Preview Data"):
                preview_cols = ['Complaint', 'Product Identifier Tag']
                if 'FBA_Reason_Code' in combined_df.columns:
                    preview_cols.append('FBA_Reason_Code')
                
                st.dataframe(combined_df[preview_cols].head(10))
            
            # Cost estimate
            avg_cost_per_item = 0.002  # Rough estimate
            estimated_cost = len(combined_df) * avg_cost_per_item
            
            st.markdown(f"""
            <div class="cost-box">
                <h4 style="margin: 0;">💰 Estimated Cost</h4>
                <p style="font-size: 1.5em; margin: 0.5rem 0;">${estimated_cost:.4f}</p>
                <p style="color: var(--muted); margin: 0;">for {len(combined_df)} returns</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Process button
            if st.button("🚀 Categorize Returns", type="primary", use_container_width=True):
                process_data()
    
    # Display results if available
    if st.session_state.processing_complete:
        st.markdown("---")
        display_results()

if __name__ == "__main__":
    main()
