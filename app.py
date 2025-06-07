"""
Vive Health Quality Complaint Categorizer - Enhanced UI with Cost Tracking
AI-Powered Return Reason Classification Tool with PDF Support
Version: 13.0 - OpenAI Only with Cost Estimation & Modern UI

Key Features:
- Real-time cost estimation and tracking
- Modern, animated UI with better visual feedback
- PDF parsing for Amazon Seller Central returns
- FBA Return Report (.txt) support
- Quality insights and root cause analysis
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config first
st.set_page_config(
    page_title="Vive Health Return Categorizer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check for required modules
try:
    from enhanced_ai_analysis import EnhancedAIAnalyzer, AIProvider, FBA_REASON_MAP
    AI_AVAILABLE = True
except ImportError as e:
    AI_AVAILABLE = False
    logger.error(f"AI module not available: {str(e)}")

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
    logger.warning("pdfplumber not available - PDF support disabled")

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

# App Configuration
APP_CONFIG = {
    'title': 'Vive Health Medical Device Return Categorizer',
    'version': '13.0',
    'company': 'Vive Health'
}

# Enhanced Colors with gradients
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
    'gradient1': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    'gradient2': 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
    'gradient3': 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
    'cost': '#50C878'  # Emerald green for cost
}

# Quality-related categories
QUALITY_CATEGORIES = [
    'Product Defects/Quality',
    'Performance/Effectiveness',
    'Missing Components',
    'Design/Material Issues',
    'Stability/Positioning Issues',
    'Medical/Health Concerns'
]

# OpenAI Pricing (per 1K tokens)
OPENAI_PRICING = {
    'gpt-3.5-turbo': {'input': 0.0005, 'output': 0.0015},
    'gpt-4': {'input': 0.03, 'output': 0.06}
}

def inject_enhanced_css():
    """Inject enhanced CSS with animations and modern styling"""
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@400;700&display=swap');
    
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
    }}
    
    * {{
        transition: all 0.3s ease;
    }}
    
    html, body, .stApp {{
        background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #16213e 100%);
        color: var(--text); 
        font-family: 'Inter', sans-serif;
    }}
    
    /* Animated background */
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(circle at 20% 80%, rgba(0, 217, 255, 0.05) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(255, 0, 110, 0.05) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(255, 183, 0, 0.05) 0%, transparent 50%);
        animation: pulse 10s ease-in-out infinite;
        pointer-events: none;
        z-index: 0;
    }}
    
    @keyframes pulse {{
        0%, 100% {{ opacity: 0.8; }}
        50% {{ opacity: 1; }}
    }}
    
    /* Enhanced header with animation */
    .main-header {{
        background: {COLORS['gradient3']};
        padding: 3rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 10px 40px rgba(0, 217, 255, 0.3);
        animation: slideIn 0.8s ease-out;
    }}
    
    @keyframes slideIn {{
        from {{
            transform: translateY(-50px);
            opacity: 0;
        }}
        to {{
            transform: translateY(0);
            opacity: 1;
        }}
    }}
    
    .main-header::before {{
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        animation: shine 3s infinite;
    }}
    
    @keyframes shine {{
        0% {{ transform: translateX(-100%) translateY(-100%) rotate(45deg); }}
        100% {{ transform: translateX(100%) translateY(100%) rotate(45deg); }}
    }}
    
    .main-title {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3em;
        font-weight: 700;
        color: white;
        text-shadow: 0 0 30px rgba(255, 255, 255, 0.5);
        margin: 0;
        position: relative;
        z-index: 1;
    }}
    
    /* Cost estimation box */
    .cost-box {{
        background: linear-gradient(135deg, rgba(80, 200, 120, 0.1) 0%, rgba(80, 200, 120, 0.2) 100%);
        border: 2px solid var(--cost);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 0 20px rgba(80, 200, 120, 0.3);
        animation: fadeIn 0.5s ease-out;
    }}
    
    .cost-title {{
        color: var(--cost);
        font-size: 1.2em;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }}
    
    .cost-value {{
        font-size: 2em;
        font-weight: 700;
        color: var(--cost);
        text-shadow: 0 0 10px rgba(80, 200, 120, 0.5);
    }}
    
    /* Enhanced boxes with hover effects */
    .info-box {{
        background: rgba(26, 26, 46, 0.8);
        border: 1px solid var(--primary);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }}
    
    .info-box:hover {{
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0, 217, 255, 0.4);
        border-color: var(--accent);
    }}
    
    .info-box::before {{
        content: "";
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0, 217, 255, 0.2), transparent);
        transition: left 0.5s ease;
    }}
    
    .info-box:hover::before {{
        left: 100%;
    }}
    
    /* Animated buttons */
    .stButton > button {{
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0, 217, 255, 0.4);
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(0, 217, 255, 0.6);
    }}
    
    .stButton > button::before {{
        content: "";
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: left 0.3s ease;
    }}
    
    .stButton > button:hover::before {{
        left: 100%;
    }}
    
    /* Progress indicator */
    .stProgress > div > div {{
        background: linear-gradient(90deg, var(--primary), var(--accent));
        height: 10px;
        border-radius: 5px;
        box-shadow: 0 0 10px var(--primary);
    }}
    
    /* Metric cards with animation */
    .metric-card {{
        background: rgba(26, 26, 46, 0.9);
        border: 2px solid rgba(0, 217, 255, 0.3);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }}
    
    .metric-card:hover {{
        transform: translateY(-5px) scale(1.05);
        border-color: var(--primary);
        box-shadow: 0 10px 30px rgba(0, 217, 255, 0.4);
    }}
    
    .metric-card h3 {{
        font-size: 2.5em;
        margin: 0.5rem 0;
        background: linear-gradient(45deg, var(--primary), var(--accent));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    
    /* File upload area */
    .uploadedFile {{
        background: rgba(0, 217, 255, 0.1);
        border: 2px dashed var(--primary);
        border-radius: 15px;
        transition: all 0.3s ease;
    }}
    
    .uploadedFile:hover {{
        background: rgba(0, 217, 255, 0.2);
        border-color: var(--accent);
    }}
    
    /* Success/Error messages with animation */
    .stSuccess, .stError, .stWarning, .stInfo {{
        border-radius: 10px;
        padding: 1rem;
        animation: slideInRight 0.5s ease-out;
    }}
    
    @keyframes slideInRight {{
        from {{
            transform: translateX(50px);
            opacity: 0;
        }}
        to {{
            transform: translateX(0);
            opacity: 1;
        }}
    }}
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {{
        background: rgba(26, 26, 46, 0.8);
        border-radius: 10px;
        padding: 0.5rem;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px;
        color: var(--text);
        font-weight: 500;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        color: white;
    }}
    
    /* Dataframe styling */
    .dataframe {{
        background: rgba(26, 26, 46, 0.8);
        border-radius: 10px;
        overflow: hidden;
    }}
    
    /* Loading spinner */
    .stSpinner > div {{
        border-color: var(--primary);
    }}
    
    /* Fade in animation for elements */
    @keyframes fadeIn {{
        from {{
            opacity: 0;
            transform: translateY(20px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}
    
    .fade-in {{
        animation: fadeIn 0.6s ease-out;
    }}
    
    /* Hide Streamlit branding */
    #MainMenu, footer, header {{
        visibility: hidden;
    }}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {{
        width: 10px;
        height: 10px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: var(--dark);
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: var(--primary);
        border-radius: 5px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: var(--accent);
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
        'reason_summary': {},
        'product_summary': {},
        'date_filter_enabled': False,
        'date_range_start': None,
        'date_range_end': None,
        'severity_counts': {'critical': 0, 'major': 0, 'minor': 0},
        'quality_insights': None,
        'ai_failed': False,
        'manual_mode': False,
        'pdf_extracted_data': None,
        'fba_return_data': None,
        'total_cost': 0.0,
        'cost_breakdown': {},
        'estimated_cost': 0.0,
        'model_choice': 'gpt-3.5-turbo',
        'api_calls_made': 0
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def check_api_keys():
    """Check for API keys in Streamlit secrets"""
    keys_found = {}
    
    try:
        if hasattr(st, 'secrets'):
            # Check for OpenAI key
            openai_keys = ['OPENAI_API_KEY', 'openai_api_key', 'openai']
            for key in openai_keys:
                if key in st.secrets:
                    keys_found['openai'] = str(st.secrets[key]).strip()
                    break
    except Exception as e:
        logger.warning(f"Error checking secrets: {e}")
    
    return keys_found

def estimate_cost(num_complaints: int, model: str = 'gpt-3.5-turbo') -> float:
    """Estimate cost for processing complaints"""
    # Average tokens per complaint (input + output)
    avg_tokens_per_complaint = 150  # Conservative estimate
    
    pricing = OPENAI_PRICING.get(model, OPENAI_PRICING['gpt-3.5-turbo'])
    
    # Calculate estimated cost
    total_tokens = num_complaints * avg_tokens_per_complaint
    input_cost = (total_tokens * 0.7 / 1000) * pricing['input']  # 70% input
    output_cost = (total_tokens * 0.3 / 1000) * pricing['output']  # 30% output
    
    return input_cost + output_cost

def display_cost_estimation(num_complaints: int):
    """Display cost estimation UI"""
    st.markdown("""
    <div class="cost-box fade-in">
        <div class="cost-title">💰 Estimated Processing Cost</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        model = st.selectbox(
            "AI Model",
            options=['gpt-3.5-turbo', 'gpt-4'],
            key='model_selector',
            help="GPT-3.5 is faster and cheaper, GPT-4 is more accurate"
        )
        st.session_state.model_choice = model
    
    with col2:
        cost = estimate_cost(num_complaints, model)
        st.session_state.estimated_cost = cost
        
        st.markdown(f"""
        <div style="text-align: center;">
            <div class="cost-value">${cost:.4f}</div>
            <div style="color: var(--muted); font-size: 0.9em;">
                for {num_complaints} complaints
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        per_complaint = cost / num_complaints if num_complaints > 0 else 0
        st.markdown(f"""
        <div style="text-align: center;">
            <div style="font-size: 1.5em; color: var(--cost);">${per_complaint:.5f}</div>
            <div style="color: var(--muted); font-size: 0.9em;">
                per complaint
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Model comparison
    with st.expander("💡 Model Comparison", expanded=False):
        comparison_df = pd.DataFrame({
            'Model': ['GPT-3.5 Turbo', 'GPT-4'],
            'Speed': ['Fast ⚡', 'Slower 🐢'],
            'Accuracy': ['Good (85-90%)', 'Excellent (95%+)'],
            'Cost per 1K tokens': ['$0.002', '$0.09'],
            'Best For': ['Most returns', 'Complex cases']
        })
        st.dataframe(comparison_df, hide_index=True, use_container_width=True)

def get_ai_analyzer():
    """Get or create AI analyzer (OpenAI only)"""
    if st.session_state.ai_analyzer is None and AI_AVAILABLE:
        try:
            # Set up environment variables from secrets
            keys = check_api_keys()
            
            if 'openai' in keys:
                os.environ['OPENAI_API_KEY'] = keys['openai']
            
            # Create analyzer with OpenAI only
            st.session_state.ai_analyzer = EnhancedAIAnalyzer(AIProvider.OPENAI)
            logger.info("Created AI analyzer with OpenAI")
        except Exception as e:
            logger.error(f"Error creating AI analyzer: {e}")
            st.error(f"Error initializing AI: {str(e)}")
    
    return st.session_state.ai_analyzer

def display_required_format():
    """Display the required file format with enhanced UI"""
    st.markdown("""
    <div class="info-box fade-in">
        <h4 style="color: var(--primary); margin-bottom: 1rem;">📋 Supported File Formats</h4>
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; margin-top: 1rem;">
            
            <div style="background: rgba(255, 183, 0, 0.1); padding: 1rem; border-radius: 10px; border: 1px solid var(--accent);">
                <h5 style="color: var(--accent); margin: 0;">📄 PDF Files</h5>
                <ul style="margin: 0.5rem 0 0 0; font-size: 0.9em;">
                    <li>Amazon Seller Central Returns</li>
                    <li>Auto-extracts all return data</li>
                    <li>AI categorizes reason + comments</li>
                </ul>
            </div>
            
            <div style="background: rgba(0, 217, 255, 0.1); padding: 1rem; border-radius: 10px; border: 1px solid var(--primary);">
                <h5 style="color: var(--primary); margin: 0;">📦 FBA Reports (.txt)</h5>
                <ul style="margin: 0.5rem 0 0 0; font-size: 0.9em;">
                    <li>Tab-delimited FBA returns</li>
                    <li>Maps reason codes automatically</li>
                    <li>Includes customer comments</li>
                </ul>
            </div>
            
            <div style="background: rgba(0, 245, 160, 0.1); padding: 1rem; border-radius: 10px; border: 1px solid var(--success);">
                <h5 style="color: var(--success); margin: 0;">📊 CSV/Excel</h5>
                <ul style="margin: 0.5rem 0 0 0; font-size: 0.9em;">
                    <li><strong>Complaint</strong> column (Required)</li>
                    <li><strong>Product/SKU</strong> (Recommended)</li>
                    <li><strong>Date</strong> (Optional for filtering)</li>
                </ul>
            </div>
        </div>
        
        <p style="color: var(--accent); margin-top: 1.5rem; text-align: center; font-weight: 500;">
            💡 AI automatically detects file type and processes accordingly
        </p>
    </div>
    """, unsafe_allow_html=True)

def display_ai_status():
    """Display AI status with cost tracking"""
    if not AI_AVAILABLE:
        st.markdown("""
        <div class="info-box" style="border-color: var(--danger);">
            <h3 style="color: var(--danger); margin: 0;">❌ AI Module Not Available</h3>
            <p>The enhanced_ai_analysis.py module is required for this tool to function.</p>
        </div>
        """, unsafe_allow_html=True)
        return False
    
    # Check API keys
    keys = check_api_keys()
    
    st.markdown("""
    <div class="info-box fade-in">
        <h3 style="color: var(--primary); margin-top: 0; text-align: center;">🤖 AI Configuration & Cost Tracking</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Display API status and cost info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'openai' in keys:
            st.markdown("""
            <div class="metric-card">
                <div style="color: var(--success); font-size: 2em;">✅</div>
                <h4 style="margin: 0.5rem 0;">OpenAI API</h4>
                <p style="margin: 0; color: var(--muted);">Ready to process</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card" style="border-color: var(--danger);">
                <div style="color: var(--danger); font-size: 2em;">❌</div>
                <h4 style="margin: 0.5rem 0;">OpenAI API</h4>
                <p style="margin: 0; color: var(--muted);">Key not found</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>${st.session_state.total_cost:.4f}</h3>
            <p>Total Cost Today</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{st.session_state.api_calls_made}</h3>
            <p>API Calls Made</p>
        </div>
        """, unsafe_allow_html=True)
    
    if not keys:
        st.markdown("""
        <div class="info-box" style="border-color: var(--danger); margin-top: 1rem;">
            <h4 style="color: var(--danger); margin-top: 0;">⚠️ No API Key Found</h4>
            <p>Add your OpenAI API key to Streamlit secrets:</p>
            <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 5px;">
openai_api_key = "sk-..."</pre>
        </div>
        """, unsafe_allow_html=True)
        return False
    
    return True

def parse_amazon_returns_pdf(pdf_content) -> pd.DataFrame:
    """Parse Amazon Seller Central Manage Returns PDF"""
    if not PDF_AVAILABLE:
        st.error("PDF processing not available. Please install pdfplumber: pip install pdfplumber")
        return None
    
    try:
        returns_data = []
        
        with st.spinner("🔍 Extracting data from PDF..."):
            with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
                total_pages = len(pdf.pages)
                progress_bar = st.progress(0)
                
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    
                    if not text:
                        continue
                    
                    # Extract individual return entries
                    returns = extract_return_entries_from_text(text)
                    returns_data.extend(returns)
                    
                    # Update progress
                    progress_bar.progress((page_num + 1) / total_pages)
        
        if returns_data:
            df = pd.DataFrame(returns_data)
            # Add Complaint column from buyer comment and reason
            df['Complaint'] = df.apply(lambda row: f"{row.get('return_reason', '')}. {row.get('buyer_comment', '')}".strip(), axis=1)
            df['Product Identifier Tag'] = df.get('product_name', '')
            df['Order #'] = df.get('order_id', '')
            df['Category'] = ''  # Will be filled by AI
            
            st.success(f"✅ Extracted {len(df)} returns from PDF")
            return df
        else:
            st.warning("No return data found in PDF")
            return None
            
    except Exception as e:
        st.error(f"Error parsing PDF: {str(e)}")
        logger.error(f"PDF parsing error: {e}")
        return None

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
            'product_name': extract_field(return_block, r'(Vive[^\\n]+?)(?:Return Quantity|ASIN|$)'),
            'asin': extract_field(return_block, r'ASIN:\s*([A-Z0-9]{10})'),
            'sku': extract_field(return_block, r'SKU:\s*([A-Z0-9\-]+)'),
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

def parse_fba_return_report(file_content) -> pd.DataFrame:
    """Parse FBA Return Report .txt file"""
    try:
        # FBA reports are typically tab-delimited
        df = pd.read_csv(io.BytesIO(file_content), sep='\t', encoding='utf-8')
        
        # Expected columns in FBA return report
        expected_cols = ['return-date', 'order-id', 'sku', 'asin', 'product-name', 
                        'reason', 'customer-comments', 'quantity']
        
        # Check if this looks like an FBA return report
        if not any(col in df.columns for col in expected_cols):
            return None
        
        # Standardize column names
        df = df.rename(columns={
            'customer-comments': 'Complaint',
            'product-name': 'Product Identifier Tag',
            'order-id': 'Order #',
            'return-date': 'Date'
        })
        
        # Create complaint from reason + customer comments
        if 'reason' in df.columns:
            df['FBA_Reason_Code'] = df['reason']
            # Map FBA reason codes to categories if available
            if AI_AVAILABLE and hasattr(FBA_REASON_MAP, '__getitem__'):
                df['Suggested_Category'] = df['reason'].map(FBA_REASON_MAP).fillna('Other/Miscellaneous')
            
            # Combine reason and comments for AI analysis
            df['Complaint'] = df.apply(
                lambda row: f"Reason: {row.get('reason', '')}. Customer comment: {row.get('Complaint', '')}".strip(), 
                axis=1
            )
        
        df['Category'] = ''  # Will be filled by AI
        
        st.success(f"✅ Loaded FBA Return Report: {len(df)} returns")
        return df
        
    except Exception as e:
        logger.error(f"Error parsing FBA report: {e}")
        return None

def process_complaints_file(file_content, filename: str, date_filter=None) -> pd.DataFrame:
    """Process complaints file with date filtering"""
    try:
        # Try PDF first
        if filename.endswith('.pdf'):
            return parse_amazon_returns_pdf(file_content)
        
        # Try FBA return report
        if filename.endswith('.txt'):
            fba_df = parse_fba_return_report(file_content)
            if fba_df is not None:
                return fba_df
        
        # Standard CSV/Excel processing
        if filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_content))
        else:
            df = pd.read_excel(io.BytesIO(file_content))
        
        logger.info(f"Columns in uploaded file: {df.columns.tolist()}")
        
        # Check for Complaint column
        if 'Complaint' not in df.columns:
            st.error("❌ No 'Complaint' column found.")
            return None
        
        # Apply date filter if requested
        if date_filter and 'Date' in df.columns:
            try:
                df['Date'] = pd.to_datetime(df['Date'])
                start_date, end_date = date_filter
                mask = (df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))
                df = df[mask]
                logger.info(f"Applied date filter: {len(df)} rows after filtering")
            except Exception as e:
                logger.warning(f"Could not apply date filter: {e}")
        
        # Remove empty complaints
        initial_count = len(df)
        df = df[df['Complaint'].notna() & (df['Complaint'].str.strip() != '')]
        
        if initial_count > len(df):
            st.info(f"📋 Filtered out {initial_count - len(df)} empty rows")
        
        # Add Category column if not present
        if 'Category' not in df.columns:
            df['Category'] = ''
        
        return df
            
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        st.error(f"Error processing file: {str(e)}")
        return None

def categorize_all_data_ai(df: pd.DataFrame) -> pd.DataFrame:
    """Categorize all complaints using AI with cost tracking"""
    
    analyzer = get_ai_analyzer()
    
    if not analyzer:
        st.session_state.ai_failed = True
        raise Exception("AI analyzer not initialized")
    
    # Check API status
    try:
        api_status = analyzer.get_api_status()
        if not api_status['available']:
            st.session_state.ai_failed = True
            raise Exception("AI not available - API key issues")
    except Exception as e:
        st.session_state.ai_failed = True
        raise Exception(f"AI status check failed: {str(e)}")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    cost_text = st.empty()
    
    total_rows = len(df)
    category_counts = Counter()
    product_issues = defaultdict(lambda: defaultdict(int))
    severity_counts = Counter()
    successful_categorizations = 0
    failed_categorizations = 0
    
    # Cost tracking
    session_cost = 0.0
    tokens_used = 0
    
    for idx, row in df.iterrows():
        complaint = str(row['Complaint']).strip() if pd.notna(row['Complaint']) else ""
        
        if not complaint:
            continue
        
        # Get FBA reason if available
        fba_reason = str(row.get('FBA_Reason_Code', '')) if pd.notna(row.get('FBA_Reason_Code')) else ""
        if not fba_reason:
            fba_reason = str(row.get('reason', '')) if pd.notna(row.get('reason')) else ""
        
        try:
            # Categorize using AI
            if hasattr(analyzer, 'categorize_return'):
                category, confidence, severity, language = analyzer.categorize_return(
                    complaint, fba_reason, mode='standard' if st.session_state.model_choice == 'gpt-3.5-turbo' else 'enhanced'
                )
            else:
                # Fallback
                category = 'Other/Miscellaneous'
                confidence, severity, language = 0.8, 'none', 'en'
            
            # Update dataframe
            df.at[idx, 'Category'] = category
            
            # Track statistics
            category_counts[category] += 1
            severity_counts[severity] += 1
            
            if category != 'Other/Miscellaneous':
                successful_categorizations += 1
            
            # Track by product
            if 'Product Identifier Tag' in df.columns and pd.notna(row.get('Product Identifier Tag')):
                product = str(row['Product Identifier Tag']).strip()
                if product:
                    product_issues[product][category] += 1
            
            # Update cost (rough estimate)
            estimated_tokens = len(complaint.split()) * 3  # Rough token estimate
            tokens_used += estimated_tokens
            
            pricing = OPENAI_PRICING[st.session_state.model_choice]
            call_cost = (estimated_tokens / 1000) * (pricing['input'] + pricing['output'])
            session_cost += call_cost
            
            st.session_state.api_calls_made += 1
                    
        except Exception as e:
            logger.error(f"AI categorization failed for row {idx}: {e}")
            df.at[idx, 'Category'] = 'Other/Miscellaneous'
            failed_categorizations += 1
        
        # Update progress
        progress = (idx + 1) / total_rows
        progress_bar.progress(progress)
        status_text.text(f"🤖 Processing: {idx + 1}/{total_rows} | Success: {successful_categorizations} | Failed: {failed_categorizations}")
        cost_text.text(f"💰 Session cost: ${session_cost:.4f} | Tokens: ~{tokens_used:,}")
    
    # Update total cost
    st.session_state.total_cost += session_cost
    st.session_state.cost_breakdown['categorization'] = session_cost
    
    # Calculate success rate
    success_rate = (successful_categorizations / total_rows * 100) if total_rows > 0 else 0
    
    if failed_categorizations > total_rows * 0.5:  # More than 50% failed
        st.session_state.ai_failed = True
        raise Exception(f"Too many AI failures: {failed_categorizations}/{total_rows}")
    
    status_text.text(f"✅ Complete! Categorized: {successful_categorizations}/{total_rows} ({success_rate:.1f}% success)")
    cost_text.text(f"💰 Total cost for this batch: ${session_cost:.4f}")
    
    # Store summaries
    st.session_state.reason_summary = dict(category_counts)
    st.session_state.product_summary = dict(product_issues)
    st.session_state.severity_counts = dict(severity_counts)
    
    # Generate quality insights
    try:
        st.session_state.quality_insights = generate_quality_insights(
            df, 
            st.session_state.reason_summary,
            st.session_state.product_summary
        )
    except Exception as e:
        logger.error(f"Error generating quality insights: {e}")
        st.session_state.quality_insights = None
    
    return df

def generate_quality_insights(df: pd.DataFrame, reason_summary: dict, product_summary: dict) -> dict:
    """Generate quality insights"""
    
    total_returns = len(df)
    quality_issues = {cat: count for cat, count in reason_summary.items() 
                     if cat in QUALITY_CATEGORIES}
    total_quality = sum(quality_issues.values())
    quality_rate = (total_quality / total_returns * 100) if total_returns > 0 else 0
    
    # Determine risk level
    if quality_rate > 30:
        risk_level = "HIGH"
    elif quality_rate > 15:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
    
    # Find top quality issues
    top_quality_issues = sorted(quality_issues.items(), key=lambda x: x[1], reverse=True)[:3]
    
    # Generate action items
    action_items = []
    for category, count in top_quality_issues:
        if count > 0:
            severity = "HIGH" if count > total_returns * 0.1 else "MEDIUM"
            action_items.append({
                'severity': severity,
                'issue': category,
                'frequency': count,
                'recommendation': f"Investigate root cause of {category.lower()} issues",
                'affected_products': list(product_summary.keys())[:3]
            })
    
    # Find high-risk products
    top_risk_products = []
    for product, issues in product_summary.items():
        quality_issues_count = sum(count for cat, count in issues.items() if cat in QUALITY_CATEGORIES)
        if quality_issues_count > 0:
            total_issues = sum(issues.values())
            primary_issue = max(issues.items(), key=lambda x: x[1])[0] if issues else "Unknown"
            
            top_risk_products.append({
                'product': product,
                'total_issues': total_issues,
                'quality_issues': quality_issues_count,
                'safety_issues': 0,
                'primary_root_cause': primary_issue
            })
    
    # Sort by total issues
    top_risk_products.sort(key=lambda x: x['total_issues'], reverse=True)
    
    return {
        'risk_assessment': {
            'overall_risk_level': risk_level,
            'quality_rate': quality_rate,
            'safety_critical_count': 0,
            'top_risk_products': top_risk_products[:5]
        },
        'root_cause_distribution': {
            category: {
                'count': count,
                'products': [p for p in product_summary.keys() if category in product_summary[p]],
                'examples': [f"Example {category.lower()} complaint"]
            }
            for category, count in top_quality_issues
        },
        'action_items': action_items
    }

def display_product_analysis(df: pd.DataFrame):
    """Display product-specific analysis with enhanced UI"""
    
    if st.session_state.product_summary:
        st.markdown("#### 📦 Product Analysis")
        
        # Create product data for display
        product_data = []
        for product, issues in st.session_state.product_summary.items():
            total_returns = sum(issues.values())
            quality_issues = sum(count for cat, count in issues.items() if cat in QUALITY_CATEGORIES)
            top_issue = max(issues.items(), key=lambda x: x[1]) if issues else ('Unknown', 0)
            
            product_data.append({
                'Product': product[:50] + "..." if len(product) > 50 else product,
                'Total Returns': total_returns,
                'Quality Issues': quality_issues,
                'Quality %': f"{(quality_issues/total_returns*100):.1f}%" if total_returns > 0 else "0%",
                'Top Issue': top_issue[0],
                'Count': top_issue[1]
            })
        
        # Sort by total returns
        product_data.sort(key=lambda x: x['Total Returns'], reverse=True)
        
        # Display as styled dataframe
        if product_data:
            product_df = pd.DataFrame(product_data[:20])
            
            # Apply color coding to dataframe
            def highlight_quality(row):
                quality_pct = float(row['Quality %'].rstrip('%'))
                if quality_pct > 50:
                    return ['background-color: rgba(255, 0, 84, 0.2)'] * len(row)
                elif quality_pct > 25:
                    return ['background-color: rgba(255, 107, 53, 0.2)'] * len(row)
                else:
                    return [''] * len(row)
            
            styled_df = product_df.style.apply(highlight_quality, axis=1)
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            if len(st.session_state.product_summary) > 20:
                st.caption(f"Showing top 20 of {len(st.session_state.product_summary)} products.")
    else:
        st.info("No product information available. Ensure your file has a 'Product Identifier Tag' column.")

def display_results(df: pd.DataFrame):
    """Display categorization results with enhanced UI"""
    
    st.markdown("""
    <div class="info-box fade-in" style="background: linear-gradient(135deg, rgba(0, 245, 160, 0.1), rgba(0, 245, 160, 0.2)); border-color: var(--success);">
        <h2 style="color: var(--success); text-align: center; margin: 0;">📊 CATEGORIZATION RESULTS</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics with enhanced styling
    col1, col2, col3, col4 = st.columns(4)
    
    total_categorized = len(df[df['Category'].notna() & (df['Category'] != '')])
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(df)}</h3>
            <p>Total Returns</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        quality_count = sum(
            count for cat, count in st.session_state.reason_summary.items()
            if cat in QUALITY_CATEGORIES
        )
        quality_pct = (quality_count / total_categorized * 100) if total_categorized > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <h3>{quality_pct:.1f}%</h3>
            <p>Quality Issues</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        severity_critical = st.session_state.severity_counts.get('critical', 0)
        st.markdown(f"""
        <div class="metric-card" style="border-color: var(--danger);">
            <h3 style="color: var(--danger);">{severity_critical}</h3>
            <p>Critical Issues</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card" style="border-color: var(--cost);">
            <h3 style="color: var(--cost);">${st.session_state.cost_breakdown.get('categorization', 0):.4f}</h3>
            <p>Processing Cost</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Create tabs with better styling
    tab_list = st.tabs(["📈 Categories", "🔍 Quality Insights", "📦 Products", "💰 Cost Analysis"])
    
    # Category distribution tab
    with tab_list[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Top Return Categories")
            
            # Create visual bar chart
            sorted_reasons = sorted(st.session_state.reason_summary.items(), 
                                  key=lambda x: x[1], reverse=True)[:10]
            
            for reason, count in sorted_reasons:
                percentage = (count / total_categorized) * 100 if total_categorized > 0 else 0
                
                # Color based on quality category
                if reason in QUALITY_CATEGORIES:
                    bar_color = COLORS['danger']
                    icon = "🔴"
                else:
                    bar_color = COLORS['primary']
                    icon = "🔵"
                
                st.markdown(f"""
                <div style="margin: 1rem 0;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                        <span>{icon} {reason}</span>
                        <span style="font-weight: 600;">{count} ({percentage:.1f}%)</span>
                    </div>
                    <div style="background: rgba(255,255,255,0.1); border-radius: 5px; overflow: hidden; height: 20px;">
                        <div style="background: {bar_color}; width: {percentage}%; height: 100%; 
                                    box-shadow: 0 0 10px {bar_color}; transition: width 0.5s ease;">
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Quality vs Non-Quality pie chart representation
            st.markdown("#### Quality vs Other Returns")
            quality_returns = sum(count for cat, count in st.session_state.reason_summary.items() 
                                if cat in QUALITY_CATEGORIES)
            other_returns = total_categorized - quality_returns
            
            # Visual representation
            quality_pct = (quality_returns / total_categorized * 100) if total_categorized > 0 else 0
            other_pct = 100 - quality_pct
            
            st.markdown(f"""
            <div style="text-align: center; margin: 2rem 0;">
                <div style="width: 200px; height: 200px; margin: 0 auto; position: relative;">
                    <svg viewBox="0 0 100 100" style="transform: rotate(-90deg);">
                        <circle cx="50" cy="50" r="40" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="20"/>
                        <circle cx="50" cy="50" r="40" fill="none" stroke="{COLORS['danger']}" 
                                stroke-width="20" stroke-dasharray="{quality_pct * 2.51} 251" stroke-linecap="round"/>
                    </svg>
                    <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); 
                                font-size: 2em; font-weight: 700; color: {COLORS['danger']};">
                        {quality_pct:.0f}%
                    </div>
                </div>
                <div style="margin-top: 1rem;">
                    <div style="color: {COLORS['danger']}; font-weight: 600;">
                        Quality Issues: {quality_returns}
                    </div>
                    <div style="color: {COLORS['primary']}; margin-top: 0.5rem;">
                        Other Returns: {other_returns}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Quality insights tab
    with tab_list[1]:
        if st.session_state.quality_insights:
            insights = st.session_state.quality_insights
            
            # Risk Assessment with visual indicator
            risk_level = insights['risk_assessment']['overall_risk_level']
            risk_colors = {'HIGH': COLORS['danger'], 'MEDIUM': COLORS['warning'], 'LOW': COLORS['success']}
            risk_color = risk_colors[risk_level]
            
            st.markdown(f"""
            <div class="info-box" style="background: linear-gradient(135deg, rgba(255,0,84,0.1), rgba(255,0,84,0.2)); 
                        border-color: {risk_color};">
                <h3 style="color: {risk_color}; margin: 0; text-align: center;">
                    ⚠️ Quality Risk Level: {risk_level}
                </h3>
                <p style="text-align: center; margin: 0.5rem 0 0 0;">
                    Quality Issue Rate: {insights['risk_assessment']['quality_rate']:.1f}%
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Action Items with cards
            if insights['action_items']:
                st.markdown("### 🎯 Recommended Actions")
                
                for i, action in enumerate(insights['action_items']):
                    severity_color = COLORS['danger'] if action['severity'] == 'HIGH' else COLORS['warning']
                    
                    st.markdown(f"""
                    <div class="info-box fade-in" style="border-left: 4px solid {severity_color}; 
                              animation-delay: {i * 0.1}s;">
                        <h4 style="color: {severity_color}; margin: 0;">
                            {action['severity']} Priority: {action['issue']}
                        </h4>
                        <p style="margin: 0.5rem 0;">
                            <strong>Frequency:</strong> {action['frequency']} cases<br>
                            <strong>Action:</strong> {action['recommendation']}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Products tab
    with tab_list[2]:
        display_product_analysis(df)
    
    # Cost Analysis tab
    with tab_list[3]:
        st.markdown("### 💰 Cost Breakdown")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Cost metrics
            st.markdown(f"""
            <div class="cost-box">
                <h4 style="margin: 0;">Processing Costs</h4>
                <div style="margin-top: 1rem;">
                    <div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">
                        <span>Categorization:</span>
                        <span style="font-weight: 600;">${st.session_state.cost_breakdown.get('categorization', 0):.4f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">
                        <span>Total API Calls:</span>
                        <span style="font-weight: 600;">{st.session_state.api_calls_made}</span>
                    </div>
                    <hr style="border-color: var(--cost); opacity: 0.3;">
                    <div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">
                        <span style="font-weight: 600;">Session Total:</span>
                        <span class="cost-value">${st.session_state.total_cost:.4f}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Cost efficiency metrics
            cost_per_return = st.session_state.total_cost / len(df) if len(df) > 0 else 0
            
            st.markdown(f"""
            <div class="info-box">
                <h4 style="margin: 0;">Efficiency Metrics</h4>
                <div style="margin-top: 1rem;">
                    <div style="text-align: center;">
                        <div style="font-size: 2em; color: var(--cost); font-weight: 700;">
                            ${cost_per_return:.5f}
                        </div>
                        <div style="color: var(--muted);">per return</div>
                    </div>
                    <div style="margin-top: 1rem; text-align: center;">
                        <div style="font-size: 1.2em; color: var(--primary);">
                            {len(df) / (time.time() - st.session_state.get('start_time', time.time())):.1f}
                        </div>
                        <div style="color: var(--muted);">returns/second</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

def export_data(df: pd.DataFrame) -> bytes:
    """Export data with quality insights and cost information"""
    
    output = io.BytesIO()
    
    if EXCEL_AVAILABLE:
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Write main data
            df.to_excel(writer, sheet_name='Categorized Returns', index=False)
            
            # Add summary sheet
            if st.session_state.reason_summary:
                summary_data = []
                total = sum(st.session_state.reason_summary.values())
                
                for reason, count in sorted(st.session_state.reason_summary.items(), 
                                           key=lambda x: x[1], reverse=True):
                    percentage = (count / total) * 100 if total > 0 else 0
                    summary_data.append({
                        'Return Category': reason,
                        'Count': count,
                        'Percentage': f"{percentage:.1f}%",
                        'Is Quality Issue': 'Yes' if reason in QUALITY_CATEGORIES else 'No'
                    })
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Add cost analysis sheet
            cost_data = {
                'Metric': ['Total Returns', 'Total Cost', 'Cost per Return', 'Model Used', 'API Calls'],
                'Value': [
                    len(df),
                    f"${st.session_state.total_cost:.4f}",
                    f"${st.session_state.total_cost / len(df):.5f}" if len(df) > 0 else "$0",
                    st.session_state.model_choice,
                    st.session_state.api_calls_made
                ]
            }
            cost_df = pd.DataFrame(cost_data)
            cost_df.to_excel(writer, sheet_name='Cost Analysis', index=False)
            
            # Add quality insights if available
            if st.session_state.quality_insights:
                insights = st.session_state.quality_insights
                
                # Action items
                if insights['action_items']:
                    action_data = []
                    for action in insights['action_items']:
                        action_data.append({
                            'Priority': action['severity'],
                            'Issue': action['issue'],
                            'Frequency': action['frequency'],
                            'Recommendation': action['recommendation']
                        })
                    
                    action_df = pd.DataFrame(action_data)
                    action_df.to_excel(writer, sheet_name='Quality Actions', index=False)
            
            # Format workbook with custom styling
            workbook = writer.book
            
            # Define formats
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#1A1A2E',
                'font_color': '#00D9FF',
                'border': 1
            })
            
            quality_format = workbook.add_format({
                'bg_color': '#FFE6E6',
                'font_color': '#CC0000'
            })
            
            cost_format = workbook.add_format({
                'bg_color': '#E6F5E6',
                'font_color': '#006600',
                'num_format': '$#,##0.0000'
            })
            
            # Apply formatting
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                worksheet.set_row(0, 20, header_format)  # Header row
                
                # Auto-adjust columns
                for i, col in enumerate(df.columns if sheet_name == 'Categorized Returns' else range(5)):
                    worksheet.set_column(i, i, 15)
    else:
        # CSV fallback
        df.to_csv(output, index=False)
    
    output.seek(0)
    return output.getvalue()

def generate_quality_report(df: pd.DataFrame) -> str:
    """Generate quality analysis report with cost information"""
    
    total_returns = len(df)
    quality_issues = {cat: count for cat, count in st.session_state.reason_summary.items() 
                     if cat in QUALITY_CATEGORIES}
    total_quality = sum(quality_issues.values())
    quality_pct = (total_quality / total_returns * 100) if total_returns > 0 else 0
    
    report = f"""VIVE HEALTH QUALITY ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Version: {APP_CONFIG['version']}

═══════════════════════════════════════════════════════════════

EXECUTIVE SUMMARY
================
Total Returns Analyzed: {total_returns}
Quality-Related Returns: {total_quality} ({quality_pct:.1f}%)
Processing Status: {'Success' if not st.session_state.ai_failed else 'Failed (Manual mode used)'}

COST ANALYSIS
=============
Total Processing Cost: ${st.session_state.total_cost:.4f}
Cost per Return: ${st.session_state.total_cost / total_returns:.5f}
Model Used: {st.session_state.model_choice}
API Calls Made: {st.session_state.api_calls_made}
Average Cost per Call: ${st.session_state.total_cost / st.session_state.api_calls_made:.5f}
"""
    
    if st.session_state.quality_insights:
        insights = st.session_state.quality_insights
        report += f"""
QUALITY RISK ASSESSMENT
======================
Overall Risk Level: {insights['risk_assessment']['overall_risk_level']}
Quality Issue Rate: {insights['risk_assessment']['quality_rate']:.1f}%
"""
        
        if insights['action_items']:
            report += """
RECOMMENDED ACTIONS (PRIORITIZED)
=================================
"""
            for i, action in enumerate(insights['action_items'], 1):
                report += f"\n{i}. [{action['severity']}] {action['issue']} ({action['frequency']} cases)"
                report += f"\n   Action: {action['recommendation']}\n"
    
    report += f"""
RETURN CATEGORIES BREAKDOWN
===========================
"""
    
    for cat, count in sorted(st.session_state.reason_summary.items(), 
                           key=lambda x: x[1], reverse=True):
        pct = (count / total_returns * 100) if total_returns > 0 else 0
        quality_flag = " ⚠️ [QUALITY ISSUE]" if cat in QUALITY_CATEGORIES else ""
        report += f"{cat}{quality_flag}: {count} ({pct:.1f}%)\n"
    
    # Add top products if available
    if st.session_state.product_summary:
        report += f"""
TOP PRODUCTS BY RETURNS
=======================
"""
        product_totals = [(prod, sum(cats.values())) 
                         for prod, cats in st.session_state.product_summary.items()]
        
        for product, total in sorted(product_totals, key=lambda x: x[1], reverse=True)[:10]:
            report += f"{product}: {total} returns\n"
    
    report += f"""
═══════════════════════════════════════════════════════════════

RECOMMENDATIONS
==============
1. Focus quality improvements on top return categories
2. Address high-priority action items immediately  
3. Review products with highest return rates
4. Implement corrective actions for recurring quality issues
5. Monitor improvement metrics after implementing changes
6. Consider cost-benefit of using GPT-4 for critical product lines

COST OPTIMIZATION TIPS
=====================
- Use GPT-3.5 for routine categorization (current session: ${st.session_state.total_cost:.4f})
- Reserve GPT-4 for complex or high-value products
- Batch process returns during off-peak hours
- Current efficiency: ${st.session_state.total_cost / total_returns:.5f} per return

═══════════════════════════════════════════════════════════════
Generated by Vive Health Return Categorizer v{APP_CONFIG['version']}
"""
    
    return report

def main():
    """Main application function with enhanced UI and cost tracking"""
    
    initialize_session_state()
    inject_enhanced_css()
    
    # Track session start time for metrics
    if 'start_time' not in st.session_state:
        st.session_state.start_time = time.time()
    
    # Header with animation
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">VIVE HEALTH RETURN CATEGORIZER</h1>
        <p style="font-size: 1.2em; color: white; margin: 0.5rem 0; position: relative; z-index: 1;">
            AI-Powered Medical Device Quality Management
        </p>
        <p style="font-size: 1em; color: white; opacity: 0.9; position: relative; z-index: 1;">
            🤖 OpenAI Integration | 📄 PDF Support | 💰 Real-time Cost Tracking
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check AI status first
    if not display_ai_status():
        st.stop()
    
    # Sidebar with enhanced styling
    with st.sidebar:
        st.markdown("""
        <div class="info-box">
            <h3 style="color: var(--primary); margin-top: 0;">⚙️ Settings</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Date filtering
        st.markdown("#### 📅 Date Filtering")
        enable_date_filter = st.checkbox("Enable date filter", help="Filter returns by date range")
        st.session_state.date_filter_enabled = enable_date_filter
        
        if enable_date_filter:
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start", 
                                          value=datetime.now() - timedelta(days=30))
                st.session_state.date_range_start = start_date
            
            with col2:
                end_date = st.date_input("End", 
                                        value=datetime.now())
                st.session_state.date_range_end = end_date
        
        st.markdown("---")
        
        # Cost summary in sidebar
        st.markdown("#### 💰 Session Summary")
        st.metric("Total Cost", f"${st.session_state.total_cost:.4f}")
        st.metric("API Calls", st.session_state.api_calls_made)
        
        if st.session_state.api_calls_made > 0:
            avg_cost = st.session_state.total_cost / st.session_state.api_calls_made
            st.metric("Avg Cost/Call", f"${avg_cost:.5f}")
        
        # File support status
        st.markdown("---")
        st.markdown("#### 📁 File Support")
        
        support_items = [
            ("PDF Files", PDF_AVAILABLE, "pdfplumber"),
            ("Excel Files", EXCEL_AVAILABLE, "xlsxwriter"),
            ("CSV Files", True, None),
            ("FBA Reports", True, None)
        ]
        
        for item, available, package in support_items:
            if available:
                st.success(f"✅ {item}")
            else:
                st.error(f"❌ {item}")
                if package:
                    st.caption(f"Install: pip install {package}")
    
    # Main content
    with st.expander("📋 File Formats & Instructions", expanded=False):
        display_required_format()
        
        st.markdown("""
        <div class="info-box" style="margin-top: 1rem;">
            <h4 style="color: var(--primary); margin: 0;">🚀 Quick Start Guide</h4>
            <ol style="margin: 1rem 0 0 0;">
                <li>Upload your return files (PDF, CSV, Excel, or FBA .txt)</li>
                <li>Review the cost estimate</li>
                <li>Click "Categorize" to process with AI</li>
                <li>Export results with quality insights</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    # File upload section with enhanced styling
    st.markdown("---")
    st.markdown("""
    <div class="info-box fade-in">
        <h3 style="color: var(--primary); text-align: center;">📁 Upload Return Files</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.date_filter_enabled:
        st.info(f"📅 Date filter active: {st.session_state.date_range_start} to {st.session_state.date_range_end}")
    
    uploaded_files = st.file_uploader(
        "Drop files here or click to browse",
        type=['xlsx', 'xls', 'csv', 'txt', 'pdf'],
        accept_multiple_files=True,
        help="Upload PDF from Amazon Seller Central, FBA return reports (.txt), or CSV/Excel files"
    )
    
    if uploaded_files:
        all_data = []
        
        # Show PDF warning if needed
        pdf_files = [f for f in uploaded_files if f.name.endswith('.pdf')]
        if pdf_files and not PDF_AVAILABLE:
            st.error("❌ PDF files detected but pdfplumber not installed. Please install: pip install pdfplumber")
        
        # Prepare date filter
        date_filter = None
        if st.session_state.date_filter_enabled:
            date_filter = (st.session_state.date_range_start, st.session_state.date_range_end)
        
        # Process files with enhanced feedback
        with st.spinner("Loading files..."):
            for file in uploaded_files:
                file_content = file.read()
                filename = file.name
                
                df = process_complaints_file(file_content, filename, date_filter)
                
                if df is not None and not df.empty:
                    all_data.append(df)
                    
                    # Show file type with icon
                    if filename.endswith('.pdf'):
                        st.success(f"📄 PDF: {filename} ({len(df)} returns)")
                    elif filename.endswith('.txt') and 'FBA_Reason_Code' in df.columns:
                        st.success(f"📦 FBA Report: {filename} ({len(df)} returns)")
                    else:
                        st.success(f"📊 {filename} ({len(df)} rows)")
        
        if all_data:
            # Combine all data
            combined_df = pd.concat(all_data, ignore_index=True) if len(all_data) > 1 else all_data[0]
            st.session_state.processed_data = combined_df
            
            # Show summary with visual appeal
            st.markdown(f"""
            <div class="info-box fade-in" style="background: linear-gradient(135deg, rgba(0, 245, 160, 0.1), rgba(0, 245, 160, 0.2)); border-color: var(--success);">
                <h3 style="color: var(--success); text-align: center; margin: 0;">
                    ✅ {len(combined_df)} Returns Ready for Processing
                </h3>
                {f'<p style="text-align: center; margin: 0.5rem 0 0 0;">Combined from {len(all_data)} files</p>' if len(all_data) > 1 else ''}
            </div>
            """, unsafe_allow_html=True)
            
            # Display cost estimation
            display_cost_estimation(len(combined_df))
            
            # Preview data
            if st.checkbox("👁️ Preview data", help="Show first 10 rows"):
                preview_cols = ['Complaint', 'Category']
                if 'Product Identifier Tag' in combined_df.columns:
                    preview_cols.append('Product Identifier Tag')
                if 'FBA_Reason_Code' in combined_df.columns:
                    preview_cols.append('FBA_Reason_Code')
                    
                st.dataframe(combined_df[preview_cols].head(10), use_container_width=True)
            
            # Categorize button with cost warning
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                button_label = f"🚀 CATEGORIZE {len(combined_df)} RETURNS (${st.session_state.estimated_cost:.4f})"
                
                if st.button(button_label, type="primary", use_container_width=True):
                    st.session_state.ai_failed = False
                    st.session_state.manual_mode = False
                    
                    st.markdown(f"""
                    <div class="info-box fade-in">
                        <p style="text-align: center; margin: 0;">
                            🤖 Processing with {st.session_state.model_choice}...
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    try:
                        with st.spinner(f"🤖 AI Processing {len(combined_df)} returns..."):
                            categorized_df = categorize_all_data_ai(combined_df)
                            st.session_state.categorized_data = categorized_df
                            st.session_state.processing_complete = True
                        
                        # Show completion with animation
                        elapsed_time = time.time() - st.session_state.start_time
                        st.balloons()
                        
                        st.success(f"""
                        ✅ AI categorization complete!
                        - Time: {elapsed_time:.1f} seconds
                        - Cost: ${st.session_state.cost_breakdown.get('categorization', 0):.4f}
                        - Speed: {len(combined_df)/elapsed_time:.1f} returns/second
                        """)
                        
                    except Exception as e:
                        st.error(f"❌ AI categorization failed: {str(e)}")
                        st.session_state.ai_failed = True
            
            # Show results if processing complete
            if st.session_state.processing_complete and st.session_state.categorized_data is not None:
                
                display_results(st.session_state.categorized_data)
                
                # Export section with enhanced styling
                st.markdown("---")
                
                st.markdown("""
                <div class="info-box fade-in" style="background: linear-gradient(135deg, rgba(80, 200, 120, 0.1), rgba(80, 200, 120, 0.2)); border-color: var(--cost);">
                    <h3 style="color: var(--cost); text-align: center; margin: 0;">
                        💾 Export Your Results
                    </h3>
                    <p style="text-align: center; margin: 0.5rem 0 0 0;">
                        Download categorized data with quality insights and cost analysis
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Export options with icons
                col1, col2, col3 = st.columns(3)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                with col1:
                    excel_data = export_data(st.session_state.categorized_data)
                    
                    st.download_button(
                        label="📥 Excel Report",
                        data=excel_data,
                        file_name=f"categorized_returns_{timestamp}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                        help="Complete analysis with formatting"
                    )
                
                with col2:
                    # CSV export
                    csv_data = st.session_state.categorized_data.to_csv(index=False)
                    
                    st.download_button(
                        label="📄 CSV Data",
                        data=csv_data,
                        file_name=f"categorized_returns_{timestamp}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        help="Raw data for further analysis"
                    )
                
                with col3:
                    # Quality report
                    quality_report = generate_quality_report(st.session_state.categorized_data)
                    st.download_button(
                        label="📊 Quality Report",
                        data=quality_report,
                        file_name=f"quality_analysis_{timestamp}.txt",
                        mime="text/plain",
                        use_container_width=True,
                        help="Detailed analysis with recommendations"
                    )
                
                # Show final summary
                st.markdown(f"""
                <div class="info-box fade-in" style="margin-top: 2rem;">
                    <h4 style="color: var(--primary); margin: 0;">📋 Processing Summary</h4>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 1rem;">
                        <div style="text-align: center;">
                            <div style="font-size: 2em; font-weight: 700; color: var(--primary);">{len(st.session_state.categorized_data)}</div>
                            <div style="color: var(--muted);">Returns Processed</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 2em; font-weight: 700; color: var(--cost);">${st.session_state.total_cost:.4f}</div>
                            <div style="color: var(--muted);">Total Cost</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 2em; font-weight: 700; color: var(--accent);">{st.session_state.model_choice}</div>
                            <div style="color: var(--muted);">Model Used</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 2em; font-weight: 700; color: var(--success);">{'✅' if st.session_state.quality_insights else '❌'}</div>
                            <div style="color: var(--muted);">Quality Analysis</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
