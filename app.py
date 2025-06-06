"""
Vive Health Quality Complaint Categorizer
AI-Powered Return Reason Classification Tool
Version: 5.0 - Enhanced with PDF Support & Unified Analysis

This enhanced version supports:
- PDF files from Amazon Seller Central Manage Returns page
- FBA Return Reports (.txt tab-separated files)
- Product Complaints Ledger (Excel files)
- Cross-reference analysis between all data sources
- Medical device-specific categorization
- Quality management insights
"""

import streamlit as st

# MUST be the first Streamlit command
st.set_page_config(
    page_title='Vive Health Medical Device Return Categorizer',
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Now safe to import everything else
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import io
from typing import Dict, List, Any, Optional, Tuple
import json
import time
from collections import Counter, defaultdict
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for required modules
try:
    from enhanced_ai_analysis import EnhancedAIAnalyzer, APIClient
    AI_AVAILABLE = True
    api_error_message = None
except ImportError as e:
    AI_AVAILABLE = False
    api_error_message = f"AI module not available: {str(e)}"
    logger.error(api_error_message)

try:
    import xlsxwriter
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False
    logger.warning("xlsxwriter not available - Excel export will use basic format")

try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False
    logger.warning("openpyxl not available")

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.warning("pdfplumber not available - PDF parsing will be disabled")

# App Configuration
APP_CONFIG = {
    'title': 'Vive Health Medical Device Return Categorizer',
    'version': '5.0',
    'company': 'Vive Health',
    'description': 'AI-Powered Medical Device Return Classification with PDF Support'
}

# Cyberpunk color scheme
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
    'muted': '#666680'
}

# Medical Device Return Categories
MEDICAL_DEVICE_CATEGORIES = [
    'Size/Fit Issues',
    'Comfort Issues',
    'Product Defects/Quality',
    'Performance/Effectiveness',
    'Stability/Positioning Issues',
    'Equipment Compatibility',
    'Design/Material Issues',
    'Wrong Product/Misunderstanding',
    'Missing Components',
    'Customer Error/Changed Mind',
    'Shipping/Fulfillment Issues',
    'Assembly/Usage Difficulty',
    'Medical/Health Concerns',
    'Price/Value',
    'Other/Miscellaneous'
]

# FBA reason code mapping to categories
FBA_REASON_MAP = {
    'NOT_COMPATIBLE': 'Equipment Compatibility',
    'DAMAGED_BY_FC': 'Product Defects/Quality',
    'DAMAGED_BY_CARRIER': 'Shipping/Fulfillment Issues',
    'DEFECTIVE': 'Product Defects/Quality',
    'NOT_AS_DESCRIBED': 'Wrong Product/Misunderstanding',
    'WRONG_ITEM': 'Wrong Product/Misunderstanding',
    'MISSING_PARTS': 'Missing Components',
    'QUALITY_NOT_ADEQUATE': 'Performance/Effectiveness',
    'UNWANTED_ITEM': 'Customer Error/Changed Mind',
    'UNAUTHORIZED_PURCHASE': 'Customer Error/Changed Mind',
    'CUSTOMER_DAMAGED': 'Customer Error/Changed Mind',
    'SWITCHEROO': 'Wrong Product/Misunderstanding',
    'EXPIRED_ITEM': 'Product Defects/Quality',
    'DAMAGED_GLASS_VIAL': 'Product Defects/Quality',
    'DIFFERENT_PRODUCT': 'Wrong Product/Misunderstanding',
    'MISSING_ITEM': 'Missing Components',
    'NOT_DELIVERED': 'Shipping/Fulfillment Issues',
    'ORDERED_WRONG_ITEM': 'Customer Error/Changed Mind',
    'UNNEEDED_ITEM': 'Customer Error/Changed Mind',
    'BAD_GIFT': 'Customer Error/Changed Mind',
    'INACCURATE_WEBSITE_DESCRIPTION': 'Wrong Product/Misunderstanding',
    'BETTER_PRICE_AVAILABLE': 'Price/Value',
    'DOES_NOT_FIT': 'Size/Fit Issues',
    'NOT_COMPATIBLE_WITH_DEVICE': 'Equipment Compatibility',
    'UNSATISFACTORY_PRODUCT': 'Performance/Effectiveness',
    'ARRIVED_LATE': 'Shipping/Fulfillment Issues'
}

def inject_cyberpunk_css():
    """Inject cyberpunk-themed CSS"""
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&display=swap');
    
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
    }}
    
    html, body, .stApp {{
        background: linear-gradient(135deg, var(--dark) 0%, var(--light) 100%);
        color: var(--text);
        font-family: 'Rajdhani', sans-serif;
    }}
    
    h1, h2, h3 {{
        font-family: 'Orbitron', sans-serif;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }}
    
    h1 {{
        background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(0, 217, 255, 0.4);
        font-size: 2.5em;
        text-align: center;
        margin-bottom: 0.5em;
    }}
    
    .pdf-upload-box {{
        background: rgba(255, 183, 0, 0.1);
        border: 2px solid var(--accent);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 0 30px rgba(255, 183, 0, 0.3);
    }}
    
    .neon-box {{
        background: rgba(10, 10, 15, 0.9);
        border: 1px solid var(--primary);
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 0 20px rgba(0, 217, 255, 0.4);
        margin: 1rem 0;
    }}
    
    .success-box {{
        background: rgba(0, 245, 160, 0.1);
        border: 1px solid var(--success);
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 0 15px rgba(0, 245, 160, 0.2);
    }}
    
    .error-box {{
        background: rgba(255, 0, 84, 0.1);
        border: 1px solid var(--danger);
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 0 15px rgba(255, 0, 84, 0.2);
    }}
    
    .metric-card {{
        background: rgba(26, 26, 46, 0.8);
        border: 1px solid rgba(0, 217, 255, 0.4);
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }}
    
    .metric-card:hover {{
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 10px 30px rgba(0, 217, 255, 0.4);
    }}
    
    .category-badge {{
        display: inline-block;
        padding: 0.4rem 0.8rem;
        border-radius: 5px;
        font-weight: 600;
        margin: 0.25rem;
    }}
    
    .stButton > button {{
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        color: var(--dark);
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 217, 255, 0.4);
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(0, 217, 255, 0.6);
    }}
    
    .data-source-indicator {{
        display: inline-block;
        padding: 0.3rem 0.6rem;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: 600;
        margin: 0.2rem;
    }}
    
    .source-pdf {{
        background: rgba(255, 183, 0, 0.2);
        border: 1px solid var(--accent);
        color: var(--accent);
    }}
    
    .source-fba {{
        background: rgba(0, 217, 255, 0.2);
        border: 1px solid var(--primary);
        color: var(--primary);
    }}
    
    .source-ledger {{
        background: rgba(255, 0, 110, 0.2);
        border: 1px solid var(--secondary);
        color: var(--secondary);
    }}
    
    .stProgress > div > div > div > div {{
        background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
    }}
    
    #MainMenu, footer, header {{
        visibility: hidden;
    }}
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        'uploaded_files': [],
        'processed_data': None,
        'categorized_data': None,
        'ai_analyzer': None,
        'processing_complete': False,
        'file_types': {},
        'reason_summary': {},
        'product_summary': {},
        'data_sources': set(),
        'pdf_data': None,
        'fba_data': None,
        'ledger_data': None,
        'unified_data': None,
        'cost_tracking': {'session_cost': 0.0}
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def parse_pdf_returns(pdf_file) -> pd.DataFrame:
    """Parse Amazon Seller Central returns PDF"""
    if not PDFPLUMBER_AVAILABLE:
        st.error("PDF parsing requires pdfplumber. Install with: pip install pdfplumber")
        return None
        
    try:
        import pdfplumber
        
        returns_data = []
        
        with pdfplumber.open(pdf_file) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                
                if not text:
                    continue
                
                # Pattern for Amazon return entries
                # Order ID: XXX-XXXXXXX-XXXXXXX
                order_pattern = r'Order ID:\s*(\d{3}-\d{7}-\d{7})'
                
                # Split by order IDs
                order_matches = list(re.finditer(order_pattern, text))
                
                for i, match in enumerate(order_matches):
                    start = match.start()
                    end = order_matches[i+1].start() if i+1 < len(order_matches) else len(text)
                    
                    return_block = text[start:end]
                    
                    # Extract fields
                    order_id = match.group(1)
                    
                    # Extract other fields using patterns
                    asin_match = re.search(r'ASIN:\s*([A-Z0-9]{10})', return_block)
                    sku_match = re.search(r'SKU:\s*([A-Z0-9-]+)', return_block)
                    product_match = re.search(r'(Vive[^\\n]+?)(?:Return Quantity|Return Reason)', return_block, re.DOTALL)
                    reason_match = re.search(r'Return Reason:\s*(.+?)(?:Buyer Comment|Request Date|$)', return_block, re.DOTALL)
                    comment_match = re.search(r'Buyer Comment:\s*(.+?)(?:Request Date|Order Date|$)', return_block, re.DOTALL)
                    date_match = re.search(r'Request Date:\s*(\d{2}/\d{2}/\d{4})', return_block)
                    quantity_match = re.search(r'Return Quantity:\s*(\d+)', return_block)
                    
                    return_data = {
                        'order_id': order_id,
                        'asin': asin_match.group(1) if asin_match else '',
                        'sku': sku_match.group(1) if sku_match else '',
                        'product_name': product_match.group(1).strip() if product_match else '',
                        'return_reason': reason_match.group(1).strip() if reason_match else '',
                        'buyer_comment': comment_match.group(1).strip() if comment_match else '',
                        'request_date': date_match.group(1) if date_match else '',
                        'quantity': int(quantity_match.group(1)) if quantity_match else 1,
                        'data_source': 'PDF',
                        'page_number': page_num + 1
                    }
                    
                    # Clean up extracted text
                    for key in ['product_name', 'return_reason', 'buyer_comment']:
                        if return_data[key]:
                            # Remove extra whitespace and newlines
                            return_data[key] = ' '.join(return_data[key].split())
                    
                    returns_data.append(return_data)
        
        if returns_data:
            df = pd.DataFrame(returns_data)
            # Standardize column names
            df = df.rename(columns={
                'request_date': 'Date',
                'order_id': 'Order #',
                'sku': 'Imported SKU',
                'product_name': 'Product Identifier Tag',
                'buyer_comment': 'Complaint'
            })
            return df
        else:
            st.warning("No return data found in PDF. Please check the file format.")
            return None
            
    except Exception as e:
        logger.error(f"Error parsing PDF: {e}")
        st.error(f"Error parsing PDF: {str(e)}")
        return None

def process_fba_returns(file_content, filename: str) -> pd.DataFrame:
    """Process FBA return report"""
    try:
        # Read tab-separated file
        df = pd.read_csv(io.BytesIO(file_content), sep='\t')
        
        # Standardize columns
        column_mapping = {
            'return-date': 'Date',
            'order-id': 'Order #',
            'sku': 'Imported SKU',
            'asin': 'ASIN',
            'product-name': 'Product Identifier Tag',
            'quantity': 'Quantity',
            'reason': 'FBA_Reason_Code',
            'customer-comments': 'Complaint'
        }
        
        df = df.rename(columns=column_mapping)
        df['data_source'] = 'FBA'
        
        return df
        
    except Exception as e:
        logger.error(f"Error processing FBA returns: {e}")
        st.error(f"Error processing FBA returns: {str(e)}")
        return None

def process_complaints_ledger(file_content, filename: str) -> pd.DataFrame:
    """Process complaints ledger Excel file"""
    try:
        df = pd.read_excel(io.BytesIO(file_content))
        
        # Check for complaint column
        if 'Complaint' not in df.columns:
            complaint_cols = [col for col in df.columns if 'complaint' in col.lower() or 'comment' in col.lower()]
            if complaint_cols:
                df = df.rename(columns={complaint_cols[0]: 'Complaint'})
            else:
                st.error("Could not find 'Complaint' column in the file")
                st.info(f"Available columns: {', '.join(df.columns)}")
                return None
        
        df['data_source'] = 'Ledger'
        return df
        
    except Exception as e:
        logger.error(f"Error processing complaints ledger: {e}")
        st.error(f"Error processing ledger: {str(e)}")
        return None

def unify_data_sources() -> pd.DataFrame:
    """Unify data from all sources and identify cross-references"""
    unified_data = []
    
    # Collect all data sources
    if st.session_state.pdf_data is not None:
        unified_data.append(st.session_state.pdf_data)
    if st.session_state.fba_data is not None:
        unified_data.append(st.session_state.fba_data)
    if st.session_state.ledger_data is not None:
        unified_data.append(st.session_state.ledger_data)
    
    if not unified_data:
        return None
    
    # Combine all data
    df_combined = pd.concat(unified_data, ignore_index=True)
    
    # Identify cross-references (same order ID appearing in multiple sources)
    if 'Order #' in df_combined.columns:
        # Count how many sources each order appears in
        order_source_counts = df_combined.groupby('Order #')['data_source'].nunique()
        multi_source_orders = order_source_counts[order_source_counts > 1].index.tolist()
        
        # Mark cross-referenced entries
        df_combined['cross_referenced'] = df_combined['Order #'].isin(multi_source_orders)
        
        # Add source count
        df_combined['source_count'] = df_combined['Order #'].map(order_source_counts)
    else:
        df_combined['cross_referenced'] = False
        df_combined['source_count'] = 1
    
    return df_combined

def get_ai_analyzer():
    """Get or create AI analyzer"""
    if st.session_state.ai_analyzer is None and AI_AVAILABLE:
        st.session_state.ai_analyzer = EnhancedAIAnalyzer()
    return st.session_state.ai_analyzer

def categorize_return_with_ai(complaint: str, return_reason: str = None, fba_reason: str = None) -> str:
    """Use AI to categorize a return into medical device categories"""
    
    analyzer = get_ai_analyzer()
    if not analyzer or not analyzer.api_client.is_available():
        # Fallback to rule-based categorization
        return fallback_categorization(complaint, return_reason, fba_reason)
    
    try:
        # Use the analyzer's categorize_return method
        category = analyzer.categorize_return(complaint, return_reason, fba_reason)
        return category
    except Exception as e:
        logger.error(f"AI categorization error: {e}")
        return fallback_categorization(complaint, return_reason, fba_reason)

def fallback_categorization(complaint: str, return_reason: str = None, fba_reason: str = None) -> str:
    """Rule-based fallback categorization"""
    
    # First check FBA reason code mapping
    if fba_reason and fba_reason in FBA_REASON_MAP:
        return FBA_REASON_MAP[fba_reason]
    
    # Combine all text for analysis
    text = f"{complaint} {return_reason or ''} {fba_reason or ''}".lower()
    
    # Category keyword mappings
    keyword_map = {
        'Size/Fit Issues': ['small', 'large', 'size', 'fit', 'tight', 'loose', 'narrow', 'wide'],
        'Comfort Issues': ['uncomfortable', 'comfort', 'hurts', 'painful', 'pressure', 'sore'],
        'Product Defects/Quality': ['defective', 'broken', 'damaged', 'quality', 'malfunction', 'faulty', 'poor quality'],
        'Performance/Effectiveness': ['not work', 'ineffective', 'useless', 'performance', 'doesn\'t work'],
        'Stability/Positioning Issues': ['unstable', 'slides', 'moves', 'position', 'falls', 'tips'],
        'Equipment Compatibility': ['compatible', 'fit toilet', 'fit wheelchair', 'walker', 'doesn\'t fit'],
        'Design/Material Issues': ['heavy', 'bulky', 'material', 'design', 'flimsy', 'thin'],
        'Wrong Product/Misunderstanding': ['wrong', 'different', 'not as described', 'expected', 'not what'],
        'Missing Components': ['missing', 'incomplete', 'no instructions', 'parts missing'],
        'Customer Error/Changed Mind': ['mistake', 'changed mind', 'no longer', 'patient died', 'don\'t need'],
        'Shipping/Fulfillment Issues': ['shipping', 'damaged arrival', 'late', 'package', 'delivery'],
        'Assembly/Usage Difficulty': ['difficult', 'hard to', 'confusing', 'complicated', 'instructions'],
        'Medical/Health Concerns': ['doctor', 'medical', 'health', 'allergic', 'reaction'],
        'Price/Value': ['price', 'expensive', 'value', 'cheaper', 'cost']
    }
    
    # Score each category
    scores = {}
    for category, keywords in keyword_map.items():
        score = sum(1 for keyword in keywords if keyword in text)
        if score > 0:
            scores[category] = score
    
    # Return highest scoring category
    if scores:
        return max(scores.items(), key=lambda x: x[1])[0]
    
    return 'Other/Miscellaneous'

def categorize_all_data(df: pd.DataFrame) -> pd.DataFrame:
    """Categorize all returns from unified data"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    df_copy = df.copy()
    df_copy['Return_Category'] = ''
    
    total_rows = len(df_copy)
    category_counts = Counter()
    product_issues = defaultdict(lambda: defaultdict(int))
    
    for idx, row in df_copy.iterrows():
        # Get relevant fields
        complaint = str(row.get('Complaint', '')) if pd.notna(row.get('Complaint')) else ""
        return_reason = str(row.get('return_reason', '')) if pd.notna(row.get('return_reason')) else ""
        fba_reason = str(row.get('FBA_Reason_Code', '')) if pd.notna(row.get('FBA_Reason_Code')) else ""
        
        # Categorize
        if complaint or return_reason or fba_reason:
            category = categorize_return_with_ai(complaint, return_reason, fba_reason)
            df_copy.at[idx, 'Return_Category'] = category
            category_counts[category] += 1
            
            # Track by product
            product = row.get('Product Identifier Tag', 'Unknown')
            if product and str(product).strip() and product != 'Unknown':
                product_issues[product][category] += 1
        else:
            df_copy.at[idx, 'Return_Category'] = 'Other/Miscellaneous'
            category_counts['Other/Miscellaneous'] += 1
        
        # Update progress
        progress = (idx + 1) / total_rows
        progress_bar.progress(progress)
        status_text.text(f"Processing: {idx + 1}/{total_rows} returns categorized...")
        
        # Small delay every 10 items to avoid rate limiting
        if (idx + 1) % 10 == 0:
            time.sleep(0.1)
    
    status_text.text("✅ Categorization complete!")
    
    # Store summaries
    st.session_state.reason_summary = dict(category_counts)
    st.session_state.product_summary = dict(product_issues)
    
    return df_copy

def display_unified_results(df: pd.DataFrame):
    """Display comprehensive results from all data sources"""
    
    st.markdown("""
    <div class="neon-box">
        <h2 style="color: var(--primary); text-align: center;">📊 UNIFIED QUALITY ANALYSIS</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Data source summary
    if 'data_source' in df.columns:
        source_counts = df['data_source'].value_counts()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: var(--primary);">{len(df)}</h3>
                <p>Total Returns</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            pdf_count = source_counts.get('PDF', 0)
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: var(--accent);">{pdf_count}</h3>
                <p>PDF Returns</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            fba_count = source_counts.get('FBA', 0)
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: var(--primary);">{fba_count}</h3>
                <p>FBA Returns</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            ledger_count = source_counts.get('Ledger', 0)
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: var(--secondary);">{ledger_count}</h3>
                <p>Complaints</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Category breakdown
    st.markdown("---")
    st.markdown("### 📈 Return Categories Analysis")
    
    # Sort categories by count
    sorted_categories = sorted(st.session_state.reason_summary.items(), key=lambda x: x[1], reverse=True)
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Top Return Categories")
        
        for category, count in sorted_categories[:10]:
            if count > 0:
                percentage = (count / len(df)) * 100
                
                # Determine priority color
                if category in ['Product Defects/Quality', 'Medical/Health Concerns']:
                    color = COLORS['danger']
                    icon = "🚨"
                elif category in ['Performance/Effectiveness', 'Missing Components', 'Design/Material Issues']:
                    color = COLORS['warning']
                    icon = "⚠️"
                elif category in ['Customer Error/Changed Mind', 'Price/Value']:
                    color = COLORS['success']
                    icon = "✅"
                else:
                    color = COLORS['primary']
                    icon = "ℹ️"
                
                st.markdown(f"""
                <div style="margin: 1rem 0;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: {color};">
                            {icon} <strong>{category}</strong>
                        </span>
                        <span>{count} ({percentage:.1f}%)</span>
                    </div>
                    <div style="background: rgba(255,255,255,0.1); border-radius: 10px; height: 10px; margin-top: 5px;">
                        <div style="background: {color}; width: {percentage}%; height: 100%; border-radius: 10px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        # Quality metrics
        st.markdown("#### 🎯 Quality Management Metrics")
        
        # Calculate quality-related returns
        quality_categories = [
            'Product Defects/Quality', 'Performance/Effectiveness', 
            'Missing Components', 'Design/Material Issues'
        ]
        quality_count = sum(st.session_state.reason_summary.get(cat, 0) for cat in quality_categories)
        quality_pct = (quality_count / len(df) * 100) if len(df) > 0 else 0
        
        # Critical categories requiring immediate attention
        critical_categories = ['Product Defects/Quality', 'Medical/Health Concerns']
        critical_count = sum(st.session_state.reason_summary.get(cat, 0) for cat in critical_categories)
        
        st.markdown(f"""
        <div class="metric-card" style="background: rgba(255, 0, 84, 0.1); border-color: var(--danger);">
            <h3 style="color: var(--danger);">{quality_pct:.1f}%</h3>
            <p>Quality-Related Returns</p>
            <small>({quality_count} total)</small>
        </div>
        """, unsafe_allow_html=True)
        
        if critical_count > 0:
            st.error(f"⚠️ {critical_count} returns require immediate quality investigation (FDA reportable)")
        
        # Group categories by type
        st.markdown("#### Category Groups")
        
        category_groups = {
            'Quality Issues': ['Product Defects/Quality', 'Performance/Effectiveness', 'Missing Components', 'Design/Material Issues'],
            'User Experience': ['Size/Fit Issues', 'Comfort Issues', 'Equipment Compatibility', 'Stability/Positioning Issues', 'Assembly/Usage Difficulty'],
            'Fulfillment': ['Wrong Product/Misunderstanding', 'Shipping/Fulfillment Issues'],
            'Customer': ['Customer Error/Changed Mind', 'Medical/Health Concerns', 'Price/Value']
        }
        
        for group_name, categories in category_groups.items():
            group_count = sum(st.session_state.reason_summary.get(cat, 0) for cat in categories)
            if group_count > 0:
                group_pct = (group_count / len(df)) * 100
                st.markdown(f"""
                <div style="background: rgba(26, 26, 46, 0.8); border-radius: 10px; 
                            padding: 0.75rem; margin: 0.5rem 0;">
                    <strong>{group_name}:</strong> {group_count} returns ({group_pct:.1f}%)
                </div>
                """, unsafe_allow_html=True)
    
    # Cross-referenced returns
    if 'cross_referenced' in df.columns:
        cross_ref_count = df['cross_referenced'].sum()
        if cross_ref_count > 0:
            st.markdown("---")
            st.markdown(f"### 🔗 Cross-Referenced Returns: {cross_ref_count}")
            st.info(f"{cross_ref_count} returns found in multiple data sources (same Order ID)")
            
            # Show examples
            cross_ref_df = df[df['cross_referenced'] == True].head(5)
            if not cross_ref_df.empty:
                with st.expander("View Cross-Referenced Examples"):
                    display_cols = ['Order #', 'Product Identifier Tag', 'Return_Category', 'data_source', 'source_count']
                    available_cols = [col for col in display_cols if col in cross_ref_df.columns]
                    st.dataframe(cross_ref_df[available_cols], use_container_width=True)
    
    # Product-specific insights
    if st.session_state.product_summary:
        st.markdown("---")
        st.markdown("### 📦 Top Products by Return Volume")
        
        # Calculate total returns per product
        product_totals = {
            product: sum(categories.values()) 
            for product, categories in st.session_state.product_summary.items()
        }
        
        # Get top 10 products
        top_products = sorted(product_totals.items(), key=lambda x: x[1], reverse=True)[:10]
        
        for product, total in top_products:
            if product and str(product).strip() and product != 'Unknown':
                product_display = str(product)[:60] + "..." if len(str(product)) > 60 else str(product)
                
                # Get top issue for this product
                product_categories = st.session_state.product_summary[product]
                top_issue = max(product_categories.items(), key=lambda x: x[1])
                
                # Check if it's a quality issue
                is_quality = top_issue[0] in ['Product Defects/Quality', 'Performance/Effectiveness', 'Missing Components']
                
                st.markdown(f"""
                <div style="background: rgba(26, 26, 46, 0.8); border-radius: 10px; 
                            padding: 0.75rem; margin: 0.5rem 0;
                            border-left: 4px solid {'var(--danger)' if is_quality else 'var(--primary)'};">
                    <strong>{product_display}</strong><br>
                    Returns: {total} | Top Issue: {top_issue[0]} ({top_issue[1]} returns)
                </div>
                """, unsafe_allow_html=True)

def generate_quality_insights(df: pd.DataFrame) -> str:
    """Generate quality management insights using AI"""
    
    analyzer = get_ai_analyzer()
    if not analyzer or not analyzer.api_client.is_available():
        return generate_fallback_insights(df)
    
    try:
        # Get data sources
        data_sources = list(st.session_state.data_sources) if st.session_state.data_sources else ['Unknown']
        
        # Use the analyzer's generate_quality_insights method
        insights = analyzer.generate_quality_insights(
            category_summary=st.session_state.reason_summary,
            product_summary=st.session_state.product_summary,
            total_returns=len(df),
            data_sources=data_sources
        )
        
        return insights
        
    except Exception as e:
        logger.error(f"AI insights generation error: {e}")
        return generate_fallback_insights(df)

def generate_fallback_insights(df: pd.DataFrame) -> str:
    """Generate basic insights when AI is unavailable"""
    
    total_returns = len(df)
    
    # Calculate quality-related percentage
    quality_categories = ['Product Defects/Quality', 'Performance/Effectiveness', 'Missing Components', 'Design/Material Issues']
    quality_count = sum(st.session_state.reason_summary.get(cat, 0) for cat in quality_categories)
    quality_pct = (quality_count / total_returns * 100) if total_returns > 0 else 0
    
    # Get top category
    top_category = max(st.session_state.reason_summary.items(), key=lambda x: x[1]) if st.session_state.reason_summary else ('Unknown', 0)
    
    insights = f"""## QUALITY MANAGEMENT SUMMARY

**Analysis Date:** {datetime.now().strftime('%B %d, %Y')}
**Total Returns Analyzed:** {total_returns}
**Quality-Related Returns:** {quality_count} ({quality_pct:.1f}%)
**Data Sources:** {', '.join(st.session_state.data_sources) if st.session_state.data_sources else 'Unknown'}

## KEY FINDINGS

1. **Primary Return Category:** {top_category[0]} ({top_category[1]} returns, {top_category[1]/total_returns*100:.1f}%)
2. **Quality Impact:** {quality_pct:.1f}% of returns are quality-related
3. **Multi-Source Analysis:** Data unified from PDF, FBA, and Ledger sources

## IMMEDIATE ACTIONS REQUIRED

1. **Quality Investigation**
   - Review all {quality_count} quality-related returns
   - Identify potential patterns for MDR reporting
   - Document findings in quality system

2. **Product Focus**
   - Prioritize top products with highest return rates
   - Conduct root cause analysis
   - Update inspection criteria

3. **Customer Safety**
   - Review Medical/Health Concerns category immediately
   - Assess need for customer notifications
   - Document in complaint files

## RECOMMENDATIONS

- Implement enhanced incoming inspection for top return categories
- Update IFUs to address usage difficulty issues  
- Consider design modifications for comfort/fit problems
- Track improvement metrics after interventions

## NEXT STEPS

1. Schedule quality review meeting within 48 hours
2. Assign CAPA owners for top 3 issues
3. Report findings to management
4. Monitor trends weekly for 30 days

*Note: This is an automated analysis. Please review with quality team for final decisions.*
"""
    
    return insights

def export_comprehensive_report(df: pd.DataFrame, insights: str) -> bytes:
    """Export comprehensive Excel report"""
    
    output = io.BytesIO()
    
    try:
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Sheet 1: Categorized Returns
            export_df = df.copy()
            
            # Reorder columns for clarity
            priority_cols = ['Date', 'Order #', 'Product Identifier Tag', 'Imported SKU', 
                            'Return_Category', 'Complaint', 'data_source', 'cross_referenced']
            other_cols = [col for col in export_df.columns if col not in priority_cols]
            available_priority = [col for col in priority_cols if col in export_df.columns]
            ordered_cols = available_priority + other_cols
            export_df = export_df[ordered_cols]
            
            export_df.to_excel(writer, sheet_name='Categorized Returns', index=False)
            
            # Sheet 2: Category Summary
            category_summary = pd.DataFrame(
                list(st.session_state.reason_summary.items()),
                columns=['Category', 'Count']
            ).sort_values('Count', ascending=False)
            
            category_summary['Percentage'] = (category_summary['Count'] / len(df) * 100).round(1)
            category_summary['Quality Impact'] = category_summary['Category'].apply(
                lambda x: 'High' if x in ['Product Defects/Quality', 'Medical/Health Concerns']
                else 'Medium' if x in ['Performance/Effectiveness', 'Missing Components', 'Design/Material Issues']
                else 'Low'
            )
            
            category_summary.to_excel(writer, sheet_name='Category Summary', index=False)
            
            # Sheet 3: Product Analysis
            if st.session_state.product_summary:
                product_data = []
                for product, issues in st.session_state.product_summary.items():
                    if product and str(product).strip():
                        total = sum(issues.values())
                        top_issue = max(issues.items(), key=lambda x: x[1]) if issues else ('Unknown', 0)
                        
                        # Count quality issues
                        quality_issues = sum(
                            count for cat, count in issues.items() 
                            if cat in ['Product Defects/Quality', 'Performance/Effectiveness', 'Missing Components']
                        )
                        
                        product_data.append({
                            'Product': str(product)[:100],
                            'Total Returns': total,
                            'Top Issue': top_issue[0],
                            'Top Issue Count': top_issue[1],
                            'Quality Issues': quality_issues,
                            'Issue Diversity': len(issues)
                        })
                
                if product_data:
                    product_df = pd.DataFrame(product_data).sort_values('Total Returns', ascending=False)
                    product_df.to_excel(writer, sheet_name='Product Analysis', index=False)
            
            # Sheet 4: Data Source Analysis
            if 'data_source' in df.columns:
                source_pivot = pd.crosstab(df['Return_Category'], df['data_source'], margins=True, margins_name='Total')
                source_pivot.to_excel(writer, sheet_name='Source Analysis')
            
            # Sheet 5: Quality Insights
            insights_df = pd.DataFrame({
                'Quality Management Insights': [insights]
            })
            insights_df.to_excel(writer, sheet_name='Quality Insights', index=False)
            
            # Format workbook
            workbook = writer.book
            
            # Add formats
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#1A1A2E',
                'font_color': '#00D9FF',
                'border': 1
            })
            
            # Apply formatting to each sheet
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                
                # Set column widths based on sheet
                if sheet_name == 'Categorized Returns':
                    column_widths = {
                        'A:A': 15,  # Date
                        'B:B': 20,  # Order
                        'C:C': 40,  # Product
                        'D:D': 15,  # SKU
                        'E:E': 25,  # Category
                        'F:F': 50,  # Complaint
                        'G:G': 12,  # Source
                        'H:H': 15   # Cross-ref
                    }
                    
                    for col_range, width in column_widths.items():
                        worksheet.set_column(col_range, width)
                
                elif sheet_name == 'Quality Insights':
                    worksheet.set_column('A:A', 100)
                    
        output.seek(0)
        return output.getvalue()
        
    except Exception as e:
        logger.error(f"Error generating Excel report: {e}")
        st.error(f"Error generating report: {str(e)}")
        return None

def main():
    """Main application function"""
    
    # Show any critical errors
    if not AI_AVAILABLE:
        st.error(f"❌ Critical Error: {api_error_message}")
        st.info("Please ensure the enhanced_ai_analysis.py file is in the same directory as this app.")
        st.stop()
    
    if not PDFPLUMBER_AVAILABLE:
        st.warning("⚠️ PDF processing not available. Install pdfplumber: pip install pdfplumber")
    
    initialize_session_state()
    inject_cyberpunk_css()
    
    # Header
    st.markdown(f"""
    <h1>{APP_CONFIG['title']}</h1>
    <p style="text-align: center; color: var(--primary); font-size: 1.2em; margin-bottom: 2rem;">
        {APP_CONFIG['description']} - Unified Analysis from PDF, FBA & Ledger Data
    </p>
    """, unsafe_allow_html=True)
    
    # API Status
    analyzer = get_ai_analyzer()
    if analyzer:
        status = analyzer.get_api_status()
        if status['available']:
            st.success(f"✅ AI Service Connected - {status.get('model', 'Ready')}")
        else:
            st.warning(f"⚠️ AI Service Issue: {status.get('message', 'Configuration needed')}")
            st.info("The tool will use rule-based categorization as fallback.")
    
    # Instructions
    with st.expander("📖 How to Use This Tool - Quality Analyst 5-Minute Workflow", expanded=False):
        st.markdown("""
        ### 🎯 Quick Workflow for Quality Analysts
        
        **From Data to Insights in Under 5 Minutes:**
        
        1. **📄 Export from Amazon Seller Central:**
           - **PDF**: Go to Manage Returns → Print/Save as PDF
           - **FBA**: Reports → Fulfillment → Customer Returns → Export (.txt)
           - **Ledger**: Use your existing Excel complaints tracking file
        
        2. **📤 Upload Files (Any Combination):**
           - Drag & drop or browse for files
           - Tool automatically detects file types
           - Processes all three sources simultaneously
        
        3. **🤖 Automatic Processing:**
           - AI categorizes into 15 medical device categories
           - Cross-references by Order ID
           - Identifies quality vs non-quality issues
           - Highlights FDA reportable events
        
        4. **📊 Instant Insights:**
           - **🚨 Red**: Product defects (FDA reportable)
           - **⚠️ Yellow**: Design/usability issues
           - **✅ Green**: Customer errors (not quality)
           - **📈 Trends**: By product and category
        
        5. **📥 Export & Action:**
           - Excel report with all categorized data
           - Quality insights for management
           - CAPA recommendations
           - Track improvements over time
        
        ### 🔗 Cross-Reference Magic:
        
        The tool automatically links returns across sources:
        - **Same Order ID** in PDF + FBA = Complete picture
        - **Product patterns** across all sources
        - **Validation** of return reasons
        
        ### 💡 Pro Tips for Quality Teams:
        
        1. **Upload all three types** for comprehensive analysis
        2. **Focus on products** with >5% return rate
        3. **Prioritize** Product Defects/Quality category
        4. **Track monthly** to show improvement
        5. **Share insights** in quality meetings
        
        ### 📊 Medical Device Categories:
        
        **Critical (FDA Reportable):**
        - Product Defects/Quality
        - Medical/Health Concerns
        
        **High Priority:**
        - Performance/Effectiveness
        - Missing Components
        - Design/Material Issues
        
        **Medium Priority:**
        - Size/Fit Issues
        - Comfort Issues
        - Equipment Compatibility
        - Stability/Positioning
        
        **Lower Priority:**
        - Customer Error/Changed Mind
        - Shipping/Fulfillment Issues
        - Price/Value
        """)
    
    # File Upload Section
    st.markdown("""
    <div class="neon-box">
        <h3 style="color: var(--accent);">📁 UPLOAD RETURN DATA FILES</h3>
        <p>Upload one or more file types for comprehensive analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="pdf-upload-box">
            <h4 style="color: var(--accent); margin-top: 0;">📄 PDF Returns</h4>
            <p style="font-size: 0.9em;">Seller Central Manage Returns</p>
        </div>
        """, unsafe_allow_html=True)
        
        pdf_file = st.file_uploader(
            "Upload PDF",
            type=['pdf'],
            key="pdf_upload",
            label_visibility="collapsed",
            help="PDF from Amazon Seller Central Manage Returns page"
        )
        
        if pdf_file:
            with st.spinner("Parsing PDF..."):
                pdf_data = parse_pdf_returns(pdf_file)
                if pdf_data is not None:
                    st.session_state.pdf_data = pdf_data
                    st.session_state.data_sources.add('PDF')
                    st.success(f"✅ Parsed {len(pdf_data)} returns from PDF")
    
    with col2:
        st.markdown("""
        <div class="neon-box">
            <h4 style="color: var(--primary); margin-top: 0;">📊 FBA Returns</h4>
            <p style="font-size: 0.9em;">Tab-separated .txt file</p>
        </div>
        """, unsafe_allow_html=True)
        
        fba_file = st.file_uploader(
            "Upload FBA Report",
            type=['txt', 'tsv'],
            key="fba_upload",
            label_visibility="collapsed",
            help="FBA Return Report from Seller Central"
        )
        
        if fba_file:
            with st.spinner("Processing FBA returns..."):
                file_content = fba_file.read()
                fba_data = process_fba_returns(file_content, fba_file.name)
                if fba_data is not None:
                    st.session_state.fba_data = fba_data
                    st.session_state.data_sources.add('FBA')
                    st.success(f"✅ Processed {len(fba_data)} FBA returns")
    
    with col3:
        st.markdown("""
        <div class="neon-box">
            <h4 style="color: var(--secondary); margin-top: 0;">📋 Complaints Ledger</h4>
            <p style="font-size: 0.9em;">Excel file with complaints</p>
        </div>
        """, unsafe_allow_html=True)
        
        ledger_file = st.file_uploader(
            "Upload Ledger",
            type=['xlsx', 'xls'],
            key="ledger_upload",
            label_visibility="collapsed",
            help="Product Complaints Ledger Excel file"
        )
        
        if ledger_file:
            with st.spinner("Reading complaints ledger..."):
                file_content = ledger_file.read()
                ledger_data = process_complaints_ledger(file_content, ledger_file.name)
                if ledger_data is not None:
                    st.session_state.ledger_data = ledger_data
                    st.session_state.data_sources.add('Ledger')
                    st.success(f"✅ Loaded {len(ledger_data)} complaints")
    
    # Show uploaded data summary
    if st.session_state.data_sources:
        st.markdown("---")
        st.markdown("### 📊 Data Summary")
        
        col1, col2, col3 = st.columns(3)
        
        total_records = 0
        if st.session_state.pdf_data is not None:
            total_records += len(st.session_state.pdf_data)
        if st.session_state.fba_data is not None:
            total_records += len(st.session_state.fba_data)
        if st.session_state.ledger_data is not None:
            total_records += len(st.session_state.ledger_data)
        
        with col1:
            st.metric("Total Records", total_records)
        with col2:
            st.metric("Data Sources", len(st.session_state.data_sources))
        with col3:
            sources_text = ", ".join(sorted(st.session_state.data_sources))
            st.info(f"Sources: {sources_text}")
        
        # Process button
        if st.button("🚀 ANALYZE & CATEGORIZE ALL DATA", type="primary", use_container_width=True):
            with st.spinner("🤖 Unifying data sources and categorizing returns..."):
                # Unify all data sources
                unified_df = unify_data_sources()
                
                if unified_df is not None:
                    st.session_state.unified_data = unified_df
                    
                    # Categorize all returns
                    categorized_df = categorize_all_data(unified_df)
                    st.session_state.categorized_data = categorized_df
                    st.session_state.processing_complete = True
                    
                    st.balloons()
                    st.success("✅ Analysis complete!")
                else:
                    st.error("Error unifying data sources")
    
    # Display results
    if st.session_state.processing_complete and st.session_state.categorized_data is not None:
        display_unified_results(st.session_state.categorized_data)
        
        # Generate insights
        st.markdown("---")
        st.markdown("### 🎯 Quality Management Insights")
        
        with st.spinner("🤖 Generating quality insights..."):
            insights = generate_quality_insights(st.session_state.categorized_data)
        
        # Display insights in a nice box
        st.markdown("""
        <div class="neon-box">
            <h3 style="color: var(--success); margin-top: 0;">📋 AI-Generated Quality Report</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(insights)
        
        # Export section
        st.markdown("---")
        st.markdown("""
        <div class="success-box">
            <h3 style="color: var(--success);">📥 EXPORT COMPREHENSIVE REPORT</h3>
            <p>Download categorized data with quality insights</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Generate export
        excel_data = export_comprehensive_report(st.session_state.categorized_data, insights)
        
        if excel_data:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.download_button(
                    label="📥 DOWNLOAD QUALITY REPORT",
                    data=excel_data,
                    file_name=f"quality_returns_analysis_{timestamp}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
        
        # Quick actions for quality team
        st.markdown("---")
        st.markdown("""
        <div class="neon-box">
            <h3 style="color: var(--primary);">💡 QUICK ACTIONS FOR QUALITY TEAM</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Identify critical issues
        critical_categories = ['Product Defects/Quality', 'Medical/Health Concerns']
        critical_count = sum(st.session_state.reason_summary.get(cat, 0) for cat in critical_categories)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Immediate Actions:**")
            if critical_count > 0:
                st.error(f"🚨 {critical_count} returns require quality investigation")
            
            # Top defective products
            if st.session_state.product_summary:
                st.markdown("**Products Requiring Review:**")
                defective_products = []
                for product, issues in st.session_state.product_summary.items():
                    defect_count = issues.get('Product Defects/Quality', 0)
                    if defect_count > 0:
                        defective_products.append((product, defect_count))
                
                defective_products.sort(key=lambda x: x[1], reverse=True)
                for product, count in defective_products[:5]:
                    if product and str(product).strip():
                        product_short = str(product)[:40] + "..." if len(str(product)) > 40 else str(product)
                        st.markdown(f"- {product_short}: {count} quality issues")
        
        with col2:
            st.markdown("**Next Steps:**")
            st.markdown("""
            1. ✅ Review categorized returns in Excel
            2. 📊 Create CAPA for top quality issues
            3. 📧 Share report with engineering team
            4. 📈 Track improvement trends
            5. 🔍 Update quality inspection criteria
            """)

if __name__ == "__main__":
    main()
