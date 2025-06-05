"""
Amazon Return Analysis Tool - Quality Management Edition
Vive Health | Streamlined Return Reason Categorization
"""

import streamlit as st

# Streamlit page config must be first
st.set_page_config(
    page_title="Vive Health Return Analysis Tool",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import io
from typing import Dict, List, Any, Optional, Tuple
import re
from collections import Counter, defaultdict
from io import BytesIO
import requests
from bs4 import BeautifulSoup
import time
import json

# Import handling with fallbacks
try:
    import xlsxwriter
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False
    st.warning("xlsxwriter not installed. Excel export will be limited.")

try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    try:
        import PyPDF2
        PDF_AVAILABLE = True
        PDF_LIBRARY = 'PyPDF2'
    except ImportError:
        PDF_AVAILABLE = False
        st.error("Please install pdfplumber or PyPDF2: pip install pdfplumber")
else:
    PDF_LIBRARY = 'pdfplumber'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import enhanced_ai_analysis
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    logger.warning("AI module not available")

# Configuration
APP_CONFIG = {
    'title': 'Vive Health Return Analysis Tool',
    'version': '1.0',
    'company': 'Vive Health',
    'purpose': 'Quality Management Return Analysis'
}

COLORS = {
    'primary': '#00D9FF', 'secondary': '#FF006E', 'accent': '#FFB700',
    'success': '#00F5A0', 'warning': '#FF6B35', 'danger': '#FF0054',
    'dark': '#0A0A0F', 'light': '#1A1A2E', 'text': '#E0E0E0', 'muted': '#666680'
}

# Return reason categories for quality management
RETURN_CATEGORIES = {
    'SIZE_FIT_ISSUES': {
        'keywords': ['too small', 'too large', 'doesnt fit', "doesn't fit", 'wrong size', 'size', 'fit', 'tight', 'loose', 'big', 'little'],
        'color': '#FF6B35',
        'icon': '📏'
    },
    'QUALITY_DEFECTS': {
        'keywords': ['defective', 'broken', 'damaged', 'doesnt work', "doesn't work", 'poor quality', 'defect', 'malfunction', 'faulty', 'dead', 'not working'],
        'color': '#FF0054',
        'icon': '⚠️'
    },
    'WRONG_PRODUCT': {
        'keywords': ['wrong item', 'not as described', 'inaccurate', 'different', 'not what', 'incorrect', 'mislabeled'],
        'color': '#FF006E',
        'icon': '📦'
    },
    'BUYER_MISTAKE': {
        'keywords': ['bought by mistake', 'accidentally', 'wrong order', 'my mistake', 'ordered wrong', 'accident'],
        'color': '#666680',
        'icon': '🤷'
    },
    'NO_LONGER_NEEDED': {
        'keywords': ['no longer needed', 'changed mind', 'dont need', "don't need", 'not needed', 'patient died', 'cancelled'],
        'color': '#666680',
        'icon': '❌'
    },
    'FUNCTIONALITY_ISSUES': {
        'keywords': ['not comfortable', 'hard to use', 'unstable', 'difficult', 'uncomfortable', 'awkward', 'complicated'],
        'color': '#FFB700',
        'icon': '🔧'
    },
    'COMPATIBILITY_ISSUES': {
        'keywords': ['doesnt fit toilet', "doesn't fit", 'not compatible', 'incompatible', 'wont fit', "won't fit", 'wrong type'],
        'color': '#00D9FF',
        'icon': '🔌'
    },
    'UNCATEGORIZED': {
        'keywords': [],
        'color': '#1A1A2E',
        'icon': '❓'
    }
}

def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        'return_data': None,
        'pdf_data': None,
        'fba_data': None,
        'ai_analyzer': None,
        'current_view': 'upload',
        'analysis_results': None,
        'selected_asin': None,
        'categorized_returns': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def inject_cyberpunk_css():
    """Inject cyberpunk CSS styling"""
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&display=swap');
    
    :root {{
        --primary: {COLORS['primary']}; --secondary: {COLORS['secondary']};
        --accent: {COLORS['accent']}; --success: {COLORS['success']};
        --warning: {COLORS['warning']}; --danger: {COLORS['danger']};
        --dark: {COLORS['dark']}; --light: {COLORS['light']};
        --text: {COLORS['text']}; --muted: {COLORS['muted']};
    }}
    
    html, body, .stApp {{
        background: linear-gradient(135deg, var(--dark) 0%, var(--light) 100%);
        color: var(--text); font-family: 'Rajdhani', sans-serif;
    }}
    
    h1, h2, h3 {{ font-family: 'Orbitron', sans-serif; text-transform: uppercase; letter-spacing: 0.1em; }}
    
    h1 {{
        background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(0, 217, 255, 0.4);
    }}
    
    .neon-box {{
        background: rgba(10, 10, 15, 0.9); border: 1px solid var(--primary);
        border-radius: 10px; padding: 1.5rem;
        box-shadow: 0 0 20px rgba(0, 217, 255, 0.4), inset 0 0 20px rgba(0, 217, 255, 0.1);
    }}
    
    .category-card {{
        background: rgba(26, 26, 46, 0.8); border: 1px solid rgba(0, 217, 255, 0.4);
        border-radius: 10px; padding: 1.5rem; margin: 0.5rem 0;
        transition: all 0.3s ease; cursor: pointer;
    }}
    
    .category-card:hover {{ transform: translateY(-5px) scale(1.02); }}
    
    .quality-alert {{
        background: rgba(255, 0, 84, 0.1); border: 2px solid var(--danger);
        border-radius: 10px; padding: 1rem;
        box-shadow: 0 0 15px rgba(255, 0, 84, 0.3);
    }}
    
    .success-box {{
        background: rgba(0, 245, 160, 0.1); border: 1px solid var(--success);
        border-radius: 10px; padding: 1rem;
        box-shadow: 0 0 15px rgba(0, 245, 160, 0.2);
    }}
    
    .stButton > button {{
        font-family: 'Rajdhani', sans-serif; font-weight: 600;
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        color: var(--dark); border: none; padding: 0.75rem 2rem;
        border-radius: 5px; transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 217, 255, 0.4);
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px); box-shadow: 0 6px 25px rgba(0, 217, 255, 0.6);
    }}
    
    #MainMenu, footer, header {{ visibility: hidden; }}
    </style>
    """, unsafe_allow_html=True)

def check_ai_status():
    """Check AI availability"""
    if not AI_AVAILABLE:
        return False
    try:
        if st.session_state.ai_analyzer is None:
            st.session_state.ai_analyzer = enhanced_ai_analysis.EnhancedAIAnalyzer()
        status = st.session_state.ai_analyzer.get_api_status()
        return status.get('available', False)
    except Exception as e:
        logger.error(f"Error checking AI status: {e}")
        return False

def parse_pdf_returns(pdf_file) -> Dict[str, Any]:
    """Parse Amazon Manage Returns PDF"""
    try:
        returns_data = []
        
        if PDF_LIBRARY == 'pdfplumber':
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    returns = extract_return_entries_from_text(text)
                    returns_data.extend(returns)
        else:
            # PyPDF2 fallback
            reader = PyPDF2.PdfReader(pdf_file)
            for page in reader.pages:
                text = page.extract_text()
                returns = extract_return_entries_from_text(text)
                returns_data.extend(returns)
        
        return {
            'success': True,
            'returns': returns_data,
            'total_count': len(returns_data),
            'error': None
        }
        
    except Exception as e:
        logger.error(f"PDF parsing error: {e}")
        return {
            'success': False,
            'returns': [],
            'total_count': 0,
            'error': str(e)
        }

def extract_return_entries_from_text(text: str) -> List[Dict]:
    """Extract individual return records from PDF text"""
    returns = []
    
    # Pattern to identify return blocks
    order_pattern = r'Order ID:\s*(\d{3}-\d{7}-\d{7})'
    
    # Split text by order IDs
    order_matches = list(re.finditer(order_pattern, text))
    
    for i, match in enumerate(order_matches):
        start = match.start()
        end = order_matches[i+1].start() if i+1 < len(order_matches) else len(text)
        
        return_block = text[start:end]
        
        # Extract fields from return block
        return_data = {
            'order_id': match.group(1),
            'buyer': extract_field(return_block, r'Buyer:\s*(.+?)(?:Marketplace|$)'),
            'product_name': extract_field(return_block, r'([^\\n]+?)Return Quantity'),
            'asin': extract_field(return_block, r'ASIN:\s*([A-Z0-9]{10})'),
            'sku': extract_field(return_block, r'SKU:\s*([A-Z0-9-]+)'),
            'return_reason': extract_field(return_block, r'Return Reason:\s*(.+?)(?:Buyer Comment|$)'),
            'buyer_comment': extract_field(return_block, r'Buyer Comment:\s*(.+?)(?:Request Date|$)'),
            'request_date': extract_field(return_block, r'Request Date:\s*(\d{2}/\d{2}/\d{4})'),
            'quantity': extract_field(return_block, r'Return Quantity:\s*(\d+)') or '1'
        }
        
        # Clean extracted data
        for key, value in return_data.items():
            if value:
                return_data[key] = value.strip()
        
        returns.append(return_data)
    
    return returns

def extract_field(text: str, pattern: str) -> str:
    """Extract field using regex pattern"""
    match = re.search(pattern, text, re.DOTALL | re.MULTILINE)
    return match.group(1).strip() if match else ""

def parse_fba_return_report(file_content: str) -> Dict[str, Any]:
    """Parse FBA Return Report TSV file"""
    try:
        # Try to parse as TSV
        df = pd.read_csv(io.StringIO(file_content), sep='\t', encoding='utf-8')
        
        # Verify expected columns
        expected_cols = ['return-date', 'order-id', 'sku', 'asin', 'product-name', 'reason', 'customer-comments']
        missing_cols = [col for col in expected_cols if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"Missing columns in FBA report: {missing_cols}")
        
        # Convert to standard format
        returns_data = []
        for _, row in df.iterrows():
            return_data = {
                'order_id': row.get('order-id', ''),
                'asin': row.get('asin', ''),
                'sku': row.get('sku', ''),
                'product_name': row.get('product-name', ''),
                'return_reason': row.get('reason', ''),
                'buyer_comment': row.get('customer-comments', ''),
                'return_date': row.get('return-date', ''),
                'quantity': row.get('quantity', 1)
            }
            returns_data.append(return_data)
        
        return {
            'success': True,
            'returns': returns_data,
            'total_count': len(returns_data),
            'dataframe': df,
            'error': None
        }
        
    except Exception as e:
        logger.error(f"FBA report parsing error: {e}")
        return {
            'success': False,
            'returns': [],
            'total_count': 0,
            'dataframe': None,
            'error': str(e)
        }

def categorize_return(return_data: Dict, use_ai: bool = False) -> str:
    """Categorize a return based on reason and comment"""
    reason = str(return_data.get('return_reason', '')).lower()
    comment = str(return_data.get('buyer_comment', '')).lower()
    combined_text = f"{reason} {comment}"
    
    # First try keyword matching
    for category, info in RETURN_CATEGORIES.items():
        if category == 'UNCATEGORIZED':
            continue
        for keyword in info['keywords']:
            if keyword.lower() in combined_text:
                return category
    
    # If no match and AI is available, use AI
    if use_ai and check_ai_status():
        try:
            result = st.session_state.ai_analyzer.api_client.call_api(
                messages=[
                    {
                        "role": "system",
                        "content": f"Categorize this Amazon return into one of these categories: {', '.join([k for k in RETURN_CATEGORIES.keys() if k != 'UNCATEGORIZED'])}. Respond with ONLY the category name."
                    },
                    {
                        "role": "user",
                        "content": f"Return reason: {return_data.get('return_reason', 'None')}\nCustomer comment: {return_data.get('buyer_comment', 'None')}"
                    }
                ],
                max_tokens=10,
                temperature=0
            )
            
            if result['success']:
                ai_category = result['result'].strip().upper().replace(' ', '_')
                if ai_category in RETURN_CATEGORIES:
                    return ai_category
        except Exception as e:
            logger.error(f"AI categorization error: {e}")
    
    return 'UNCATEGORIZED'

def process_returns_data(returns: List[Dict], use_ai: bool = False) -> Dict[str, Any]:
    """Process and categorize all returns"""
    categorized = defaultdict(list)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, return_data in enumerate(returns):
        status_text.text(f"Categorizing return {i+1} of {len(returns)}...")
        progress_bar.progress((i + 1) / len(returns))
        
        category = categorize_return(return_data, use_ai)
        return_data['category'] = category
        categorized[category].append(return_data)
    
    progress_bar.empty()
    status_text.empty()
    
    # Calculate statistics
    stats = {}
    for category, returns_list in categorized.items():
        stats[category] = {
            'count': len(returns_list),
            'percentage': (len(returns_list) / len(returns)) * 100 if returns else 0,
            'returns': returns_list
        }
    
    return {
        'categorized': dict(categorized),
        'stats': stats,
        'total_returns': len(returns)
    }

def display_header():
    """Display application header"""
    st.markdown("""
    <div class="neon-box" style="text-align: center;">
        <h1 style="font-size: 2.5em; margin: 0;">VIVE HEALTH RETURN ANALYSIS TOOL</h1>
        <p style="color: var(--primary); text-transform: uppercase; letter-spacing: 3px;">
            Quality Management Return Categorization System
        </p>
        <p style="color: var(--accent); font-size: 0.9em;">✨ Upload PDF or FBA Returns → Get Categorized Insights in Minutes</p>
    </div>
    """, unsafe_allow_html=True)

def display_upload_section():
    """Display file upload section"""
    st.markdown("""
    <div class="neon-box">
        <h2 style="color: var(--primary);">📤 UPLOAD RETURN DATA</h2>
        <p>Upload Amazon return data from either source:</p>
        <ul>
            <li><strong>PDF Export:</strong> Manage Returns page from Seller Central</li>
            <li><strong>FBA Report:</strong> Return report in .txt (TSV) format</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📄 PDF Returns (Manage Returns Page)")
        pdf_file = st.file_uploader(
            "Upload PDF",
            type=['pdf'],
            key="pdf_upload",
            help="Export from Seller Central > Returns > Manage Returns > Print as PDF"
        )
        
        if pdf_file:
            with st.spinner("🔍 Parsing PDF..."):
                result = parse_pdf_returns(pdf_file)
                
                if result['success']:
                    st.session_state.pdf_data = result
                    st.success(f"✅ Found {result['total_count']} returns in PDF")
                else:
                    st.error(f"❌ Error: {result['error']}")
    
    with col2:
        st.markdown("### 📊 FBA Return Report")
        fba_file = st.file_uploader(
            "Upload FBA Report",
            type=['txt', 'tsv', 'csv'],
            key="fba_upload",
            help="Export from Seller Central > Reports > Fulfillment > FBA Returns"
        )
        
        if fba_file:
            with st.spinner("🔍 Parsing FBA report..."):
                content = fba_file.read().decode('utf-8')
                result = parse_fba_return_report(content)
                
                if result['success']:
                    st.session_state.fba_data = result
                    st.success(f"✅ Found {result['total_count']} returns in FBA report")
                else:
                    st.error(f"❌ Error: {result['error']}")
    
    # Combine data if both are uploaded
    if st.session_state.pdf_data and st.session_state.fba_data:
        st.info("📊 Both data sources loaded. Combining for comprehensive analysis...")
        
        # Combine returns from both sources
        all_returns = []
        if st.session_state.pdf_data:
            all_returns.extend(st.session_state.pdf_data['returns'])
        if st.session_state.fba_data:
            all_returns.extend(st.session_state.fba_data['returns'])
        
        st.session_state.return_data = all_returns
    elif st.session_state.pdf_data:
        st.session_state.return_data = st.session_state.pdf_data['returns']
    elif st.session_state.fba_data:
        st.session_state.return_data = st.session_state.fba_data['returns']
    
    # Process button
    if st.session_state.return_data:
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            use_ai = st.checkbox("🤖 Use AI for enhanced categorization", value=check_ai_status())
            
            if st.button("🚀 ANALYZE RETURNS", type="primary", use_container_width=True):
                with st.spinner("🔍 Categorizing returns..."):
                    results = process_returns_data(st.session_state.return_data, use_ai)
                    st.session_state.categorized_returns = results
                    st.session_state.current_view = 'analysis'
                    st.rerun()

def display_analysis_results():
    """Display categorized return analysis"""
    if not st.session_state.categorized_returns:
        st.error("No analysis results available")
        return
    
    results = st.session_state.categorized_returns
    
    # Summary metrics
    st.markdown("""
    <div class="neon-box">
        <h2 style="color: var(--success);">✅ ANALYSIS COMPLETE</h2>
        <p>Total Returns Analyzed: <strong>{}</strong></p>
    </div>
    """.format(results['total_returns']), unsafe_allow_html=True)
    
    # Quality alerts
    quality_defects = results['stats'].get('QUALITY_DEFECTS', {})
    if quality_defects.get('percentage', 0) > 20:
        st.markdown("""
        <div class="quality-alert">
            <h3>⚠️ QUALITY ALERT</h3>
            <p>{}% of returns are due to quality defects. Immediate action recommended.</p>
        </div>
        """.format(round(quality_defects['percentage'], 1)), unsafe_allow_html=True)
    
    # Category breakdown
    st.markdown("### 📊 Return Categories Overview")
    
    # Sort categories by count
    sorted_categories = sorted(
        results['stats'].items(),
        key=lambda x: x[1]['count'],
        reverse=True
    )
    
    # Display categories
    cols = st.columns(2)
    for i, (category, stats) in enumerate(sorted_categories):
        if category == 'UNCATEGORIZED' and stats['count'] == 0:
            continue
            
        col = cols[i % 2]
        with col:
            info = RETURN_CATEGORIES[category]
            st.markdown(f"""
            <div class="category-card" style="border-left: 4px solid {info['color']};">
                <h4>{info['icon']} {category.replace('_', ' ').title()}</h4>
                <p style="font-size: 2em; margin: 0; color: {info['color']};">{stats['count']}</p>
                <p style="margin: 0;">{stats['percentage']:.1f}% of total returns</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Detailed view tabs
    st.markdown("---")
    tabs = st.tabs(["📋 By Category", "📦 By Product", "📈 Trends", "💾 Export"])
    
    with tabs[0]:
        # Category details
        category_filter = st.selectbox(
            "Select Category",
            options=[cat for cat, stats in sorted_categories if stats['count'] > 0]
        )
        
        if category_filter:
            category_returns = results['categorized'][category_filter]
            st.markdown(f"### {RETURN_CATEGORIES[category_filter]['icon']} {category_filter.replace('_', ' ').title()} Details")
            
            # Show returns in this category
            for i, ret in enumerate(category_returns[:10]):  # Show first 10
                st.markdown(f"""
                <div style="background: rgba(26, 26, 46, 0.6); padding: 1rem; margin: 0.5rem 0; border-radius: 8px;">
                    <strong>Order:</strong> {ret.get('order_id', 'N/A')} | 
                    <strong>ASIN:</strong> {ret.get('asin', 'N/A')} | 
                    <strong>SKU:</strong> {ret.get('sku', 'N/A')}<br>
                    <strong>Product:</strong> {ret.get('product_name', 'N/A')[:100]}...<br>
                    <strong>Reason:</strong> {ret.get('return_reason', 'N/A')}<br>
                    <strong>Comment:</strong> {ret.get('buyer_comment', 'N/A')}
                </div>
                """, unsafe_allow_html=True)
            
            if len(category_returns) > 10:
                st.info(f"Showing 10 of {len(category_returns)} returns in this category")
    
    with tabs[1]:
        # Product analysis
        st.markdown("### 📦 Returns by Product")
        
        # Aggregate by ASIN
        product_returns = defaultdict(lambda: defaultdict(int))
        for category, returns_list in results['categorized'].items():
            for ret in returns_list:
                asin = ret.get('asin', 'Unknown')
                product_name = ret.get('product_name', 'Unknown Product')
                product_returns[asin]['name'] = product_name
                product_returns[asin]['total'] += 1
                product_returns[asin][category] += 1
        
        # Sort by total returns
        sorted_products = sorted(
            product_returns.items(),
            key=lambda x: x[1]['total'],
            reverse=True
        )
        
        # Display top products
        for asin, data in sorted_products[:10]:
            st.markdown(f"""
            <div class="category-card">
                <h4>ASIN: {asin}</h4>
                <p style="font-size: 0.9em; color: var(--muted);">{data['name'][:80]}...</p>
                <p><strong>Total Returns:</strong> {data['total']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show category breakdown for this product
            categories = [(k, v) for k, v in data.items() if k not in ['name', 'total'] and v > 0]
            if categories:
                cols = st.columns(len(categories))
                for i, (cat, count) in enumerate(categories):
                    with cols[i]:
                        pct = (count / data['total']) * 100
                        st.metric(cat.replace('_', ' ').title(), f"{count} ({pct:.0f}%)")
    
    with tabs[2]:
        # Trends visualization
        st.markdown("### 📈 Return Trends")
        
        # Create simple bar chart data
        chart_data = pd.DataFrame([
            {'Category': cat.replace('_', ' ').title(), 'Count': stats['count']}
            for cat, stats in sorted_categories
            if stats['count'] > 0
        ])
        
        st.bar_chart(chart_data.set_index('Category'))
        
        # Key insights
        st.markdown("### 💡 Key Insights")
        
        # Find top issues
        top_category = sorted_categories[0] if sorted_categories else None
        if top_category:
            st.info(f"🎯 Top return reason: **{top_category[0].replace('_', ' ').title()}** ({top_category[1]['percentage']:.1f}% of returns)")
        
        # Quality specific insights
        if quality_defects.get('count', 0) > 0:
            st.warning(f"⚠️ {quality_defects['count']} quality-related returns require investigation")
        
        # Size/fit insights
        size_issues = results['stats'].get('SIZE_FIT_ISSUES', {})
        if size_issues.get('count', 0) > 0:
            st.info(f"📏 {size_issues['count']} size/fit issues - consider updating size guides")
    
    with tabs[3]:
        # Export options
        st.markdown("### 💾 Export Options")
        
        if EXCEL_AVAILABLE:
            excel_buffer = generate_excel_report(results)
            st.download_button(
                "📊 Download Excel Report",
                data=excel_buffer,
                file_name=f"return_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        # CSV export
        csv_buffer = generate_csv_report(results)
        st.download_button(
            "📄 Download CSV Report",
            data=csv_buffer,
            file_name=f"return_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Quick summary text
        summary = generate_text_summary(results)
        st.download_button(
            "📝 Download Summary",
            data=summary,
            file_name=f"return_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )

def generate_excel_report(results: Dict) -> BytesIO:
    """Generate comprehensive Excel report with all return details"""
    buffer = BytesIO()
    
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # Define formats
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#1A1A2E',
            'font_color': '#00D9FF',
            'border': 1
        })
        
        # 1. Summary sheet
        summary_data = []
        for category, stats in sorted(results['stats'].items(), key=lambda x: x[1]['count'], reverse=True):
            if stats['count'] > 0:
                summary_data.append({
                    'Category': category.replace('_', ' ').title(),
                    'Count': stats['count'],
                    'Percentage': f"{stats['percentage']:.1f}%"
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Format summary sheet
        worksheet = writer.sheets['Summary']
        worksheet.set_column('A:A', 30)
        worksheet.set_column('B:C', 15)
        
        # 2. All Returns sheet - Most important for quality analyst
        all_returns_data = []
        for category, returns_list in results['categorized'].items():
            for ret in returns_list:
                all_returns_data.append({
                    'Order ID': ret.get('order_id', ''),
                    'ASIN': ret.get('asin', ''),
                    'SKU': ret.get('sku', ''),
                    'Product Name': ret.get('product_name', ''),
                    'Category': category.replace('_', ' ').title(),
                    'Return Reason': ret.get('return_reason', ''),
                    'Customer Comment': ret.get('buyer_comment', ''),
                    'Date': ret.get('return_date', ret.get('request_date', '')),
                    'Quantity': ret.get('quantity', 1)
                })
        
        all_returns_df = pd.DataFrame(all_returns_data)
        all_returns_df.to_excel(writer, sheet_name='All Returns', index=False)
        
        # Format all returns sheet
        worksheet = writer.sheets['All Returns']
        worksheet.set_column('A:A', 20)  # Order ID
        worksheet.set_column('B:B', 12)  # ASIN
        worksheet.set_column('C:C', 15)  # SKU
        worksheet.set_column('D:D', 40)  # Product Name
        worksheet.set_column('E:E', 20)  # Category
        worksheet.set_column('F:F', 25)  # Return Reason
        worksheet.set_column('G:G', 40)  # Customer Comment
        worksheet.set_column('H:H', 12)  # Date
        worksheet.set_column('I:I', 10)  # Quantity
        
        # 3. Category sheets with full details
        for category, returns_list in results['categorized'].items():
            if returns_list:
                cat_data = []
                for ret in returns_list:
                    cat_data.append({
                        'Order ID': ret.get('order_id', ''),
                        'ASIN': ret.get('asin', ''),
                        'SKU': ret.get('sku', ''),
                        'Product': ret.get('product_name', ''),
                        'Return Reason': ret.get('return_reason', ''),
                        'Customer Comment': ret.get('buyer_comment', ''),
                        'Date': ret.get('return_date', ret.get('request_date', '')),
                        'Quantity': ret.get('quantity', 1)
                    })
                
                cat_df = pd.DataFrame(cat_data)
                sheet_name = category.replace('_', ' ')[:31]  # Excel sheet name limit
                cat_df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Format category sheet
                worksheet = writer.sheets[sheet_name]
                worksheet.set_column('A:A', 20)  # Order ID
                worksheet.set_column('B:B', 12)  # ASIN
                worksheet.set_column('C:C', 15)  # SKU
                worksheet.set_column('D:D', 40)  # Product
                worksheet.set_column('E:E', 25)  # Return Reason
                worksheet.set_column('F:F', 40)  # Customer Comment
        
        # 4. Product analysis sheet - Group by ASIN
        product_data = []
        product_returns = defaultdict(lambda: defaultdict(int))
        
        # Collect all unique ASINs with their data
        asin_details = {}
        for category, returns_list in results['categorized'].items():
            for ret in returns_list:
                asin = ret.get('asin', 'Unknown')
                if asin not in asin_details:
                    asin_details[asin] = {
                        'product_name': ret.get('product_name', 'Unknown'),
                        'sku': ret.get('sku', '')
                    }
                product_returns[asin]['total'] += 1
                product_returns[asin][category] += 1
        
        # Create product summary
        for asin, counts in product_returns.items():
            row = {
                'ASIN': asin,
                'SKU': asin_details.get(asin, {}).get('sku', ''),
                'Product Name': asin_details.get(asin, {}).get('product_name', 'Unknown'),
                'Total Returns': counts['total']
            }
            
            # Add category breakdowns
            for cat in RETURN_CATEGORIES.keys():
                if cat != 'UNCATEGORIZED':
                    count = counts.get(cat, 0)
                    percentage = (count / counts['total'] * 100) if counts['total'] > 0 else 0
                    row[cat.replace('_', ' ').title()] = f"{count} ({percentage:.1f}%)" if count > 0 else "0"
            
            # Identify top issue for this product
            top_issue = max(
                [(k, v) for k, v in counts.items() if k != 'total' and k in RETURN_CATEGORIES],
                key=lambda x: x[1],
                default=('None', 0)
            )
            row['Top Issue'] = top_issue[0].replace('_', ' ').title() if top_issue[1] > 0 else 'None'
            
            product_data.append(row)
        
        # Sort by total returns
        product_df = pd.DataFrame(product_data)
        product_df = product_df.sort_values('Total Returns', ascending=False)
        product_df.to_excel(writer, sheet_name='By Product', index=False)
        
        # Format product sheet
        worksheet = writer.sheets['By Product']
        worksheet.set_column('A:A', 12)  # ASIN
        worksheet.set_column('B:B', 15)  # SKU
        worksheet.set_column('C:C', 40)  # Product Name
        worksheet.set_column('D:D', 15)  # Total Returns
        
        # 5. Quality Focus sheet - Only quality defects
        quality_defects = results['categorized'].get('QUALITY_DEFECTS', [])
        if quality_defects:
            quality_data = []
            for ret in quality_defects:
                quality_data.append({
                    'ASIN': ret.get('asin', ''),
                    'SKU': ret.get('sku', ''),
                    'Product': ret.get('product_name', ''),
                    'Order ID': ret.get('order_id', ''),
                    'Return Reason': ret.get('return_reason', ''),
                    'Customer Comment': ret.get('buyer_comment', ''),
                    'Date': ret.get('return_date', ret.get('request_date', ''))
                })
            
            quality_df = pd.DataFrame(quality_data)
            quality_df.to_excel(writer, sheet_name='Quality Defects', index=False)
            
            # Format quality sheet
            worksheet = writer.sheets['Quality Defects']
            worksheet.set_column('A:A', 12)  # ASIN
            worksheet.set_column('B:B', 15)  # SKU
            worksheet.set_column('C:C', 40)  # Product
            worksheet.set_column('D:D', 20)  # Order ID
            worksheet.set_column('E:E', 25)  # Return Reason
            worksheet.set_column('F:F', 40)  # Customer Comment
    
    buffer.seek(0)
    return buffer

def generate_csv_report(results: Dict) -> str:
    """Generate CSV report"""
    rows = []
    
    # Header
    rows.append(['Return Analysis Report', datetime.now().strftime('%Y-%m-%d %H:%M')])
    rows.append([])
    rows.append(['Category', 'Count', 'Percentage'])
    
    # Summary
    for category, stats in results['stats'].items():
        if stats['count'] > 0:
            rows.append([
                category.replace('_', ' ').title(),
                stats['count'],
                f"{stats['percentage']:.1f}%"
            ])
    
    rows.append([])
    rows.append(['Detailed Returns'])
    rows.append(['Order ID', 'ASIN', 'SKU', 'Product', 'Category', 'Return Reason', 'Customer Comment'])
    
    # All returns
    for category, returns_list in results['categorized'].items():
        for ret in returns_list:
            rows.append([
                ret.get('order_id', ''),
                ret.get('asin', ''),
                ret.get('sku', ''),
                ret.get('product_name', '')[:100],
                category.replace('_', ' ').title(),
                ret.get('return_reason', ''),
                ret.get('buyer_comment', '')
            ])
    
    # Convert to CSV
    output = io.StringIO()
    import csv
    writer = csv.writer(output)
    writer.writerows(rows)
    
    return output.getvalue()

def generate_text_summary(results: Dict) -> str:
    """Generate text summary for quality team"""
    summary = f"""VIVE HEALTH RETURN ANALYSIS SUMMARY
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

EXECUTIVE SUMMARY
Total Returns Analyzed: {results['total_returns']}

CATEGORY BREAKDOWN:
"""
    
    # Add categories
    for category, stats in sorted(results['stats'].items(), key=lambda x: x[1]['count'], reverse=True):
        if stats['count'] > 0:
            summary += f"\n{category.replace('_', ' ').title()}: {stats['count']} ({stats['percentage']:.1f}%)"
    
    # Quality alerts
    quality_defects = results['stats'].get('QUALITY_DEFECTS', {})
    if quality_defects.get('percentage', 0) > 20:
        summary += f"\n\n⚠️ QUALITY ALERT: {quality_defects['percentage']:.1f}% of returns are quality-related!"
    
    # Top products
    summary += "\n\nTOP PRODUCTS BY RETURNS:"
    product_returns = defaultdict(lambda: defaultdict(int))
    
    for category, returns_list in results['categorized'].items():
        for ret in returns_list:
            asin = ret.get('asin', 'Unknown')
            product_returns[asin]['name'] = ret.get('product_name', 'Unknown')
            product_returns[asin]['total'] += 1
    
    sorted_products = sorted(product_returns.items(), key=lambda x: x[1]['total'], reverse=True)
    
    for i, (asin, data) in enumerate(sorted_products[:5]):
        summary += f"\n{i+1}. ASIN {asin}: {data['total']} returns"
        summary += f"\n   {data['name'][:60]}..."
    
    # Recommendations
    summary += "\n\nRECOMMENDATIONS FOR QUALITY TEAM:"
    
    if quality_defects.get('count', 0) > 0:
        summary += "\n• Investigate quality issues - review manufacturing and QC processes"
    
    size_issues = results['stats'].get('SIZE_FIT_ISSUES', {})
    if size_issues.get('count', 0) > 0:
        summary += "\n• Update product sizing guides and descriptions"
    
    functionality_issues = results['stats'].get('FUNCTIONALITY_ISSUES', {})
    if functionality_issues.get('count', 0) > 0:
        summary += "\n• Review product usability and consider design improvements"
    
    return summary

# Streamlit page config must be first
st.set_page_config(
    page_title="Vive Health Return Analysis Tool",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def main():
    initialize_session_state()
    inject_cyberpunk_css()
    display_header()
    
    # Navigation
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        if st.button("🔄 New Analysis", use_container_width=True):
            for key in ['return_data', 'pdf_data', 'fba_data', 'categorized_returns']:
                st.session_state[key] = None
            st.session_state.current_view = 'upload'
            st.rerun()
    
    # Main content
    if st.session_state.current_view == 'upload':
        display_upload_section()
    elif st.session_state.current_view == 'analysis':
        display_analysis_results()

if __name__ == "__main__":
    main()
