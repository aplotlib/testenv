"""
Vive Health Quality Complaint Categorizer
AI-Powered Return Reason Classification Tool
Version: 6.0 - Multi-Provider AI Support (OpenAI + Claude)

This enhanced version supports:
- PDF files from Amazon Seller Central Manage Returns page
- FBA Return Reports (.txt tab-separated files)
- Product Complaints Ledger (Excel/.xlsx and CSV files)
- Multiple AI providers (OpenAI, Claude, or both)
- Provider comparison mode
- Cost tracking and optimization
- Cross-reference analysis between all data sources
"""

import streamlit as st
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
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for required modules
try:
    from enhanced_ai_analysis import EnhancedAIAnalyzer, APIClient, AIProvider
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
    'version': '6.0',
    'company': 'Vive Health',
    'description': 'Multi-Provider AI Classification with PDF/Excel/CSV Support'
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
    'muted': '#666680',
    'openai': '#74AA9C',
    'claude': '#D4A574'
}

# Medical Device Return Categories (15 total)
RETURN_REASONS = [
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

# FBA reason code mapping to medical device categories
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
    """Inject cyberpunk-themed CSS with multi-provider styling"""
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
        --openai: {COLORS['openai']};
        --claude: {COLORS['claude']};
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
    
    .provider-selector {{
        background: rgba(0, 217, 255, 0.1);
        border: 2px solid var(--primary);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 0 30px rgba(0, 217, 255, 0.3);
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
        'ai_client': None,
        'processing_complete': False,
        'file_types': {},
        'reason_summary': {},
        'product_summary': {},
        'data_sources': set(),
        'pdf_data': None,
        'fba_data': None,
        'ledger_data': None,
        'unified_data': None,
        'selected_provider': 'openai',
        'api_key_configured': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def check_and_set_api_keys():
    """Check for API keys in Streamlit secrets and set them in environment"""
    api_key_found = False
    
    # Check for OpenAI key in secrets
    if hasattr(st, 'secrets'):
        # Try various possible key names for OpenAI
        openai_key_names = ['OPENAI_API_KEY', 'openai_api_key', 'openai', 'api_key']
        for key_name in openai_key_names:
            if key_name in st.secrets:
                os.environ['OPENAI_API_KEY'] = str(st.secrets[key_name])
                st.session_state.api_key_configured = True
                api_key_found = True
                logger.info(f"Found OpenAI API key in Streamlit secrets under '{key_name}'")
                break
        
        # Check for Claude key in secrets
        claude_key_names = ['ANTHROPIC_API_KEY', 'anthropic_api_key', 'claude_api_key', 'claude']
        for key_name in claude_key_names:
            if key_name in st.secrets:
                os.environ['ANTHROPIC_API_KEY'] = str(st.secrets[key_name])
                api_key_found = True
                logger.info(f"Found Claude API key in Streamlit secrets under '{key_name}'")
                break
    
    return api_key_found

def setup_ai_provider():
    """Setup AI provider based on user selection"""
    st.markdown("""
    <div class="provider-selector">
        <h3 style="color: var(--primary); margin-top: 0;">🤖 AI PROVIDER CONFIGURATION</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if API keys are already in Streamlit secrets
    api_keys_in_secrets = check_and_set_api_keys()
    
    if api_keys_in_secrets:
        st.success("✅ API keys loaded from Streamlit secrets")
        st.info("Using OpenAI provider with pre-configured API key")
        st.session_state.selected_provider = 'openai'
    else:
        # Manual configuration
        col1, col2, col3 = st.columns(3)
        
        with col1:
            provider = st.selectbox(
                "Select AI Provider",
                ["openai", "claude", "both"],
                key="provider_select"
            )
            st.session_state.selected_provider = provider
        
        with col2:
            if provider in ["openai", "both"]:
                openai_key = st.text_input(
                    "OpenAI API Key",
                    type="password",
                    placeholder="sk-...",
                    key="openai_key_input"
                )
                if openai_key:
                    os.environ['OPENAI_API_KEY'] = openai_key
                    st.session_state.api_key_configured = True
        
        with col3:
            if provider in ["claude", "both"]:
                claude_key = st.text_input(
                    "Claude API Key",
                    type="password",
                    placeholder="sk-ant-...",
                    key="claude_key_input"
                )
                if claude_key:
                    os.environ['ANTHROPIC_API_KEY'] = claude_key
                    st.session_state.api_key_configured = True

def get_ai_client():
    """Get or create AI client based on selected provider"""
    if st.session_state.ai_client is None and AI_AVAILABLE:
        try:
            # Create appropriate client based on selection
            provider = st.session_state.selected_provider
            
            if provider == "openai":
                st.session_state.ai_client = APIClient(AIProvider.OPENAI)
            elif provider == "claude":
                st.session_state.ai_client = APIClient(AIProvider.CLAUDE)
            elif provider == "both":
                st.session_state.ai_client = APIClient(AIProvider.BOTH)
        except Exception as e:
            logger.error(f"Error creating AI client: {e}")
            st.error(f"Error initializing AI: {str(e)}")
            
    return st.session_state.ai_client

def clean_dataframe_columns(df):
    """Clean dataframe to ensure no duplicate columns"""
    if df is None:
        return None
    
    # First, remove any duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Also strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    return df

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
                        'Date': date_match.group(1) if date_match else '',
                        'Product Identifier Tag': product_match.group(1).strip() if product_match else '',
                        'Imported SKU': sku_match.group(1) if sku_match else '',
                        'UDI': '',  # Not in PDF
                        'CS Ticket #': '',  # Not in PDF
                        'Order #': order_id,
                        'Source': 'PDF',
                        'Categorizing / Investigating Agent': '',  # Not in PDF
                        'Complaint': comment_match.group(1).strip() if comment_match else '',
                        'Category': '',  # Will be filled later
                        'Return_Reason_Raw': reason_match.group(1).strip() if reason_match else '',
                        'ASIN': asin_match.group(1) if asin_match else '',
                        'Quantity': int(quantity_match.group(1)) if quantity_match else 1,
                        'data_source': 'PDF'
                    }
                    
                    # Clean up extracted text
                    for key in ['Product Identifier Tag', 'Complaint', 'Return_Reason_Raw']:
                        if return_data[key]:
                            return_data[key] = ' '.join(return_data[key].split())
                    
                    returns_data.append(return_data)
        
        if returns_data:
            df = pd.DataFrame(returns_data)
            return clean_dataframe_columns(df)
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
        
        # Map to standard columns
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
        
        # Create standardized dataframe
        std_df = pd.DataFrame()
        
        # Map available columns
        for fba_col, std_col in column_mapping.items():
            if fba_col in df.columns:
                std_df[std_col] = df[fba_col]
            else:
                std_df[std_col] = ''
        
        # Add missing standard columns
        std_df['UDI'] = ''
        std_df['CS Ticket #'] = ''
        std_df['Source'] = 'FBA'
        std_df['Categorizing / Investigating Agent'] = ''
        std_df['Category'] = ''
        std_df['data_source'] = 'FBA'
        
        # Format date if present
        if 'Date' in std_df.columns and len(std_df) > 0:
            try:
                std_df['Date'] = pd.to_datetime(std_df['Date']).dt.strftime('%m/%d/%Y')
            except:
                pass
        
        return clean_dataframe_columns(std_df)
        
    except Exception as e:
        logger.error(f"Error processing FBA returns: {e}")
        st.error(f"Error processing FBA returns: {str(e)}")
        return None

def process_complaints_file(file_content, filename: str) -> pd.DataFrame:
    """Process complaints ledger Excel or CSV file"""
    try:
        # Determine file type and read accordingly
        if filename.endswith('.csv'):
            # Try different encodings for CSV
            try:
                df = pd.read_csv(io.BytesIO(file_content), encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(io.BytesIO(file_content), encoding='latin-1')
                except:
                    df = pd.read_csv(io.BytesIO(file_content), encoding='iso-8859-1')
        else:  # Excel
            df = pd.read_excel(io.BytesIO(file_content))
        
        # Clean column names and remove duplicates
        df = clean_dataframe_columns(df)
        
        # Check for complaint column (case-insensitive)
        complaint_found = False
        complaint_columns = []
        
        for col in df.columns:
            if 'complaint' in col.lower():
                complaint_columns.append(col)
        
        # If we have multiple complaint columns, keep only the first one
        if len(complaint_columns) > 1:
            # Keep the first complaint column and rename it
            df = df.rename(columns={complaint_columns[0]: 'Complaint'})
            # Drop other complaint columns
            for col in complaint_columns[1:]:
                if col in df.columns and col != 'Complaint':
                    df = df.drop(columns=[col])
            complaint_found = True
        elif len(complaint_columns) == 1:
            # Rename the single complaint column
            df = df.rename(columns={complaint_columns[0]: 'Complaint'})
            complaint_found = True
        
        # Ensure standard columns exist
        standard_columns = [
            'Date', 'Product Identifier Tag', 'Imported SKU', 'UDI',
            'CS Ticket #', 'Order #', 'Source', 'Categorizing / Investigating Agent',
            'Complaint', 'Category'
        ]
        
        for col in standard_columns:
            if col not in df.columns:
                df[col] = ''
        
        df['data_source'] = 'Ledger'
        
        # Final cleanup
        df = clean_dataframe_columns(df)
        
        return df
        
    except Exception as e:
        logger.error(f"Error processing complaints file: {e}")
        st.error(f"Error processing file: {str(e)}")
        return None

def categorize_with_ai(complaint: str, fba_reason: str = None, ai_client=None) -> str:
    """Use AI to categorize a complaint into return reasons"""
    
    if not ai_client or not ai_client.is_available():
        return fallback_categorization(complaint, fba_reason)
    
    # Create prompt with medical device categories
    reasons_list = "\n".join([f"- {reason}" for reason in RETURN_REASONS])
    
    # Add FBA reason context if available
    fba_context = ""
    if fba_reason and fba_reason in FBA_REASON_MAP:
        suggested_reason = FBA_REASON_MAP[fba_reason]
        fba_context = f"\n\nNote: Amazon's FBA system categorized this as '{fba_reason}' which typically maps to '{suggested_reason}'."
    
    prompt = f"""You are a quality management expert for medical devices. Analyze this customer complaint and select the SINGLE MOST APPROPRIATE return category.

Customer Complaint: {complaint}{fba_context}

Available Medical Device Return Categories:
{reasons_list}

Instructions:
1. Read the complaint carefully
2. Choose the ONE category that best matches the primary issue
3. Consider medical device quality and safety implications
4. If multiple categories could apply, choose the most specific one
5. Only use "Other/Miscellaneous" if no other category fits

Respond with ONLY the exact category text from the list, nothing else."""

    try:
        response = ai_client.call_api(
            messages=[
                {"role": "system", "content": "You are a quality management expert categorizing medical device returns. Always respond with the exact text of one return category from the provided list."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=50
        )
        
        if response['success']:
            reason = response['result'].strip()
            
            # Validate the reason - check for exact match
            if reason in RETURN_REASONS:
                return reason
            
            # Try case-insensitive match
            for r in RETURN_REASONS:
                if r.lower() == reason.lower():
                    return r
            
            # Fallback
            return fallback_categorization(complaint, fba_reason)
        else:
            return fallback_categorization(complaint, fba_reason)
            
    except Exception as e:
        logger.error(f"Error in AI categorization: {e}")
        return fallback_categorization(complaint, fba_reason)

def fallback_categorization(complaint: str, fba_reason: str = None) -> str:
    """Fallback keyword-based categorization"""
    
    # First check FBA reason mapping
    if fba_reason and fba_reason in FBA_REASON_MAP:
        return FBA_REASON_MAP[fba_reason]
    
    complaint_lower = complaint.lower() if complaint else ""
    
    # Define keyword mappings for medical device categories
    keyword_map = {
        'Size/Fit Issues': ['small', 'large', 'size', 'fit', 'tight', 'loose', 'narrow', 'wide', 'big', 'tiny'],
        'Comfort Issues': ['uncomfortable', 'comfort', 'hurts', 'painful', 'pressure', 'sore', 'pain', 'ache'],
        'Product Defects/Quality': ['defective', 'broken', 'damaged', 'quality', 'malfunction', 'faulty', 'poor quality', 'defect', 'crack', 'tear', 'rip'],
        'Performance/Effectiveness': ['not work', 'ineffective', 'useless', 'performance', 'doesn\'t work', 'does not work', 'not effective'],
        'Stability/Positioning Issues': ['unstable', 'slides', 'moves', 'position', 'falls', 'tips', 'wobbly', 'shift'],
        'Equipment Compatibility': ['compatible', 'fit toilet', 'fit wheelchair', 'walker', 'doesn\'t fit', 'not compatible', 'incompatible'],
        'Design/Material Issues': ['heavy', 'bulky', 'material', 'design', 'flimsy', 'thin', 'cheap material'],
        'Wrong Product/Misunderstanding': ['wrong', 'different', 'not as described', 'expected', 'not what', 'incorrect', 'mistake'],
        'Missing Components': ['missing', 'incomplete', 'no instructions', 'parts missing', 'not included'],
        'Customer Error/Changed Mind': ['mistake', 'changed mind', 'no longer', 'patient died', 'don\'t need', 'ordered wrong'],
        'Shipping/Fulfillment Issues': ['shipping', 'damaged arrival', 'late', 'package', 'delivery', 'arrived damaged'],
        'Assembly/Usage Difficulty': ['difficult', 'hard to', 'confusing', 'complicated', 'instructions', 'assembly', 'setup'],
        'Medical/Health Concerns': ['doctor', 'medical', 'health', 'allergic', 'reaction', 'injury', 'condition'],
        'Price/Value': ['price', 'expensive', 'value', 'cheaper', 'cost', 'overpriced']
    }
    
    # Score each category
    scores = {}
    for category, keywords in keyword_map.items():
        score = sum(1 for keyword in keywords if keyword in complaint_lower)
        if score > 0:
            scores[category] = score
    
    # Return highest scoring category
    if scores:
        return max(scores.items(), key=lambda x: x[1])[0]
    
    return 'Other/Miscellaneous'

def categorize_all_data(df: pd.DataFrame) -> pd.DataFrame:
    """Categorize all complaints and prepare export format"""
    
    ai_client = get_ai_client()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    df_copy = df.copy()
    
    # Initialize the return reason column
    df_copy['Return_Reason'] = ''
    
    total_rows = len(df_copy)
    category_counts = Counter()
    product_issues = defaultdict(lambda: defaultdict(int))
    
    for idx, row in df_copy.iterrows():
        # Get relevant fields
        complaint = str(row.get('Complaint', '')) if pd.notna(row.get('Complaint')) else ""
        fba_reason = str(row.get('FBA_Reason_Code', '')) if pd.notna(row.get('FBA_Reason_Code')) else ""
        
        # Categorize
        if complaint or fba_reason:
            reason = categorize_with_ai(complaint, fba_reason, ai_client)
            df_copy.at[idx, 'Return_Reason'] = reason
            category_counts[reason] += 1
            
            # Track by product
            product = row.get('Product Identifier Tag', 'Unknown')
            if product and str(product).strip() and product != 'Unknown':
                product_issues[product][reason] += 1
        else:
            df_copy.at[idx, 'Return_Reason'] = 'Other/Miscellaneous'
            category_counts['Other/Miscellaneous'] += 1
        
        # Update progress
        progress = (idx + 1) / total_rows
        progress_bar.progress(progress)
        status_text.text(f"Processing: {idx + 1}/{total_rows} complaints categorized...")
        
        # Small delay every 10 items to avoid rate limiting
        if (idx + 1) % 10 == 0:
            time.sleep(0.1)
    
    status_text.text("✅ Categorization complete!")
    
    # Store summaries
    st.session_state.reason_summary = dict(category_counts)
    st.session_state.product_summary = dict(product_issues)
    
    return df_copy

def prepare_export_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for export with Column K containing return reason"""
    
    # Create export dataframe matching the exact format shown in the example
    export_df = pd.DataFrame()
    
    # Column A - Date
    export_df['Date'] = df['Date'] if 'Date' in df.columns else ''
    
    # Column B - Complaint with Product info 
    if 'Complaint' in df.columns and 'Product Identifier Tag' in df.columns:
        export_df['Complaint_Full'] = df['Complaint'].astype(str) + ' - ' + df['Product Identifier Tag'].astype(str)
    elif 'Complaint' in df.columns:
        export_df['Complaint_Full'] = df['Complaint']
    else:
        export_df['Complaint_Full'] = ''
    
    # Column C - Product Identifier
    export_df['Product Identifier'] = df['Product Identifier Tag'] if 'Product Identifier Tag' in df.columns else ''
    
    # Column D - Imported SKU
    export_df['Imported SKU'] = df['Imported SKU'] if 'Imported SKU' in df.columns else ''
    
    # Column E - ASIN
    export_df['ASIN'] = df['ASIN'] if 'ASIN' in df.columns else ''
    
    # Column F - UDI
    export_df['UDI'] = df['UDI'] if 'UDI' in df.columns else ''
    
    # Column G - CS Ticket # - mapped to Order # if available
    export_df['CS Ticket # - mapped to Order # if available'] = df['Order #'] if 'Order #' in df.columns else df.get('CS Ticket #', '')
    
    # Column H - Source - dropdown
    export_df['Source - dropdown'] = df['Source'] if 'Source' in df.columns else df.get('data_source', 'Amazon Return')
    
    # Column I - Categorizing / Investigating Agent
    export_df['Categorizing / Investigating Agent'] = df.get('Categorizing / Investigating Agent', 'Carolina')
    
    # Column J - Complaint (raw complaint text)
    export_df['Complaint'] = df['Complaint'] if 'Complaint' in df.columns else ''
    
    # Column K - The output: AI categorized return category
    export_df['Category'] = df['Return_Reason']
    
    # Additional columns for investigation
    export_df['Investigation'] = ''
    
    return export_df

def display_results_summary(df: pd.DataFrame):
    """Display summary of categorization results"""
    
    st.markdown("""
    <div class="neon-box">
        <h2 style="color: var(--primary); text-align: center;">📊 CATEGORIZATION RESULTS</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: var(--primary);">{len(df)}</h3>
            <p>Total Returns</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        categorized = len(df[df['Return_Reason'] != ''])
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: var(--success);">{categorized}</h3>
            <p>Categorized</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        unique_reasons = df['Return_Reason'].nunique()
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: var(--accent);">{unique_reasons}</h3>
            <p>Unique Reasons</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        # Find most common reason
        if st.session_state.reason_summary:
            top_reason = max(st.session_state.reason_summary.items(), key=lambda x: x[1])[0]
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: var(--secondary); font-size: 1em;">{top_reason}</h3>
                <p>Top Reason</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Reason breakdown
    st.markdown("---")
    st.markdown("### 📈 Return Reason Distribution")
    
    # Sort reasons by count
    sorted_reasons = sorted(st.session_state.reason_summary.items(), key=lambda x: x[1], reverse=True)
    
    # Create two columns for the breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Top 10 Return Reasons")
        for i, (reason, count) in enumerate(sorted_reasons[:10]):
            percentage = (count / len(df)) * 100
            
            # Determine color based on reason type
            if reason in ['Product Defects/Quality', 'Performance/Effectiveness', 'Missing Components', 
                         'Design/Material Issues', 'Stability/Positioning Issues']:
                color = COLORS['danger']
            elif reason in ['Size/Fit Issues', 'Wrong Product/Misunderstanding', 'Equipment Compatibility']:
                color = COLORS['warning']
            elif reason in ['Customer Error/Changed Mind', 'Price/Value']:
                color = COLORS['success']
            else:
                color = COLORS['primary']
            
            st.markdown(f"""
            <div style="margin: 0.5rem 0;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span class="category-badge" style="background: {color}40; border-color: {color}; color: {color};">
                        {reason}
                    </span>
                    <span>{count} ({percentage:.1f}%)</span>
                </div>
                <div style="background: rgba(255,255,255,0.1); border-radius: 10px; height: 10px; margin-top: 5px;">
                    <div style="background: {color}; width: {percentage}%; height: 100%; border-radius: 10px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Quality insights
        st.markdown("#### 🎯 Quality Insights")
        
        # Count quality-related reasons
        quality_keywords = ['Product Defects/Quality', 'Performance/Effectiveness', 'Missing Components', 'Design/Material Issues', 'Stability/Positioning Issues']
        quality_count = 0
        quality_reasons = []
        
        for reason, count in st.session_state.reason_summary.items():
            if reason in quality_keywords:
                quality_count += count
                quality_reasons.append((reason, count))
        
        quality_percentage = (quality_count / len(df)) * 100 if len(df) > 0 else 0
        
        st.markdown(f"""
        <div class="neon-box" style="background: rgba(255, 0, 84, 0.1);">
            <h4 style="color: var(--danger); margin: 0;">Quality-Related Returns</h4>
            <h2 style="color: var(--danger); margin: 0.5rem 0;">{quality_percentage:.1f}%</h2>
            <p style="margin: 0;">({quality_count} complaints)</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Top quality issues
        if quality_reasons:
            st.markdown("**Top Quality Issues:**")
            quality_reasons.sort(key=lambda x: x[1], reverse=True)
            for reason, count in quality_reasons[:5]:
                pct = (count / quality_count) * 100
                st.markdown(f"- {reason}: {count} ({pct:.1f}% of quality issues)")

def export_categorized_data(df: pd.DataFrame) -> bytes:
    """Export categorized data to Excel with Column K populated"""
    output = io.BytesIO()
    
    # Prepare export data
    export_df = prepare_export_data(df)
    
    # Rename 'Complaint_Full' back to 'Complaint' for column B in the export
    export_columns = export_df.columns.tolist()
    if 'Complaint_Full' in export_columns:
        # Find the index and rename
        idx = export_columns.index('Complaint_Full')
        export_columns[idx] = 'Complaint (Full)'  # Use a different name to avoid confusion
        export_df.columns = export_columns
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Write main data
        export_df.to_excel(writer, sheet_name='Categorized Complaints', index=False)
        
        # Add summary sheet
        summary_data = []
        for reason, count in sorted(st.session_state.reason_summary.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(df)) * 100
            summary_data.append({
                'Return Reason': reason,
                'Count': count,
                'Percentage': f"{percentage:.1f}%"
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Format the Excel file
        workbook = writer.book
        
        # Format main sheet
        worksheet1 = writer.sheets['Categorized Complaints']
        
        # Set column widths
        column_widths = {
            'A': 12,   # Date
            'B': 50,   # Complaint (Full)
            'C': 25,   # Product Identifier
            'D': 20,   # Imported SKU  
            'E': 15,   # ASIN
            'F': 15,   # UDI
            'G': 25,   # CS Ticket #
            'H': 20,   # Source
            'I': 25,   # Categorizing Agent
            'J': 50,   # Complaint
            'K': 30,   # Category (KEY COLUMN)
        }
        
        for col, width in column_widths.items():
            col_idx = ord(col) - ord('A')
            worksheet1.set_column(col_idx, col_idx, width)
        
        # Highlight Column K
        highlight_format = workbook.add_format({
            'bg_color': '#FFF2CC',
            'border': 1
        })
        
        # Apply highlight to Column K
        worksheet1.set_column('K:K', 25, highlight_format)
        
        # Format summary sheet
        worksheet2 = writer.sheets['Summary']
        worksheet2.set_column('A:A', 30)
        worksheet2.set_column('B:B', 10)
        worksheet2.set_column('C:C', 12)
    
    output.seek(0)
    return output.getvalue()

def main():
    """Main application function"""
    
    # MUST be the first Streamlit command - NO OTHER STREAMLIT CALLS BEFORE THIS
    st.set_page_config(
        page_title=APP_CONFIG['title'],
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # NOW we can check AI availability and show errors
    if not AI_AVAILABLE:
        st.error("❌ Critical Error: AI module (enhanced_ai_analysis.py) not found!")
        st.info("Please ensure the enhanced_ai_analysis.py file is in the same directory as this app.")
        st.stop()
    
    initialize_session_state()
    inject_cyberpunk_css()
    
    # Header
    st.markdown(f"""
    <h1>{APP_CONFIG['title']}</h1>
    <p style="text-align: center; color: var(--primary); font-size: 1.2em; margin-bottom: 2rem;">
        {APP_CONFIG['description']}
    </p>
    """, unsafe_allow_html=True)
    
    # AI Provider Setup
    setup_ai_provider()
    
    # Check if API key is configured
    if not st.session_state.api_key_configured:
        st.warning("⚠️ Please configure API key to continue")
        st.stop()
    
    # Instructions
    with st.expander("📖 How to Use This Tool", expanded=False):
        st.markdown("""
        ### Quick Start Guide
        1. **AI is already configured** from Streamlit secrets
        2. **Upload your files**:
           - Product Complaints Ledger Excel file (.xlsx)
           - Product Complaints Ledger CSV file (.csv)
           - FBA Return Report (.txt tab-separated file)
           - PDF from Amazon Seller Central (requires pdfplumber)
        3. **AI will categorize** each complaint/comment into standard return reasons
        4. **Sort results** by Product Name, SKU, Return Reason, or Date
        5. **Download the results** with Column K populated with the return reason
        
        ### Supported File Types:
        - **Excel**: .xlsx and .xls files with standard columns
        - **CSV**: .csv files with comma-separated values (handles various encodings)
        - **FBA Return Reports**: Tab-separated .txt files from Amazon Seller Central
        - **PDF Returns**: PDF files from Manage Returns page (if pdfplumber installed)
        
        ### Output Format:
        - Columns A-J: Your original data preserved
        - **Column K: AI-categorized return reason** (e.g., "Size/Fit Issues", "Product Defects/Quality")
        - Available in both Excel (.xlsx) and CSV (.csv) formats
        
        The tool maintains your exact data structure and adds the categorized return reason in Column K.
        """)
    
    # File upload section
    st.markdown("""
    <div class="neon-box">
        <h3 style="color: var(--accent);">📁 UPLOAD FILES</h3>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Choose your files",
        type=['xlsx', 'xls', 'csv', 'txt', 'pdf'],
        accept_multiple_files=True,
        help="Upload complaints files (.xlsx), FBA Return Reports (.txt), or PDFs"
    )
    
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        
        all_data = []
        
        with st.spinner("📖 Processing files..."):
            for uploaded_file in uploaded_files:
                file_content = uploaded_file.read()
                filename = uploaded_file.name
                
                df = None
                
                # Process based on file type
                if filename.endswith('.pdf'):
                    if PDFPLUMBER_AVAILABLE:
                        st.info(f"📄 Processing PDF: {filename}")
                        uploaded_file.seek(0)  # Reset file pointer
                        df = parse_pdf_returns(uploaded_file)
                        if df is not None:
                            st.session_state.data_sources.add('PDF')
                    else:
                        st.warning("PDF processing requires pdfplumber. Install with: pip install pdfplumber")
                
                elif filename.endswith('.txt'):
                    st.info(f"📊 Processing FBA Return Report: {filename}")
                    df = process_fba_returns(file_content, filename)
                    if df is not None:
                        st.session_state.data_sources.add('FBA')
                
                elif filename.endswith(('.xlsx', '.xls', '.csv')):
                    st.info(f"📋 Processing Complaints File: {filename}")
                    df = process_complaints_file(file_content, filename)
                    if df is not None:
                        st.session_state.data_sources.add('Ledger')
                
                if df is not None:
                    # Clean the dataframe before adding to list
                    df = clean_dataframe_columns(df)
                    all_data.append(df)
        
        # Combine all data
        if all_data:
            # Combine dataframes
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Final cleanup of combined dataframe
            combined_df = clean_dataframe_columns(combined_df)
            
            st.session_state.processed_data = combined_df
            
            # Show file info
            st.markdown("### 📋 Data Summary")
            st.info(f"Found {len(combined_df)} total records from {len(all_data)} file(s)")
            
            # Show sample data
            st.markdown("#### Sample Data")
            display_cols = ['Date', 'Product Identifier Tag', 'Order #', 'Complaint']
            
            # Get available columns that exist in the dataframe
            available_cols = []
            for col in display_cols:
                if col in combined_df.columns:
                    available_cols.append(col)
            
            if available_cols:
                try:
                    # Create a copy of the data for display
                    display_df = combined_df[available_cols].head(5).copy()
                    st.dataframe(display_df, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not display sample data: {str(e)}")
                    st.info(f"Available columns: {', '.join(combined_df.columns.tolist())}")
            else:
                st.warning("None of the expected columns found in the data")
                st.info(f"Available columns: {', '.join(combined_df.columns.tolist())}")
            
            # Categorize button
            if st.button("🚀 CATEGORIZE ALL RETURNS", type="primary", use_container_width=True):
                with st.spinner("🤖 AI is analyzing and categorizing returns..."):
                    categorized_df = categorize_all_data(combined_df)
                    st.session_state.categorized_data = categorized_df
                    st.session_state.processing_complete = True
                
                st.balloons()
                st.success("✅ Categorization complete!")
            
            # Show results if processing is complete
            if st.session_state.processing_complete and st.session_state.categorized_data is not None:
                # Add sorting options
                st.markdown("---")
                st.markdown("### 🔄 Sort & Filter Options")
                col1, col2, col3 = st.columns([2, 2, 2])
                
                with col1:
                    sort_by = st.selectbox(
                        "Sort by",
                        ["None", "Product Identifier Tag", "Imported SKU", "Return Reason", "Date"],
                        key="sort_field"
                    )
                
                with col2:
                    sort_order = st.radio(
                        "Order",
                        ["Ascending", "Descending"],
                        horizontal=True,
                        key="sort_order"
                    )
                
                with col3:
                    # Add filter for return reasons
                    unique_reasons = sorted(st.session_state.categorized_data['Return_Reason'].unique())
                    filter_reason = st.selectbox(
                        "Filter by Return Reason",
                        ["All"] + unique_reasons,
                        key="filter_reason"
                    )
                
                # Apply sorting if selected
                display_data = st.session_state.categorized_data.copy()
                
                # Apply filter first
                if filter_reason != "All":
                    display_data = display_data[display_data['Return_Reason'] == filter_reason]
                
                # Then apply sorting
                if sort_by != "None":
                    ascending = (sort_order == "Ascending")
                    try:
                        # Handle date sorting specially
                        if sort_by == "Date":
                            display_data['Date_parsed'] = pd.to_datetime(display_data['Date'], errors='coerce')
                            display_data = display_data.sort_values('Date_parsed', ascending=ascending)
                            display_data = display_data.drop('Date_parsed', axis=1)
                        else:
                            display_data = display_data.sort_values(sort_by, ascending=ascending)
                    except Exception as e:
                        st.warning(f"Could not sort by {sort_by}: {str(e)}")
                
                # Show filter info
                if filter_reason != "All":
                    st.info(f"Showing {len(display_data)} of {len(st.session_state.categorized_data)} returns with reason: {filter_reason}")
                
                display_results_summary(display_data)
                
                # Export section
                st.markdown("---")
                st.markdown("""
                <div class="success-box">
                    <h3 style="color: var(--success);">📥 EXPORT RESULTS</h3>
                    <p>Your categorized data is ready for download!</p>
                    <p><strong>Column K contains the AI-categorized return reasons.</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Generate export file
                excel_data = export_categorized_data(st.session_state.categorized_data)
                
                # Download button
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.download_button(
                        label="📥 DOWNLOAD EXCEL (.xlsx)",
                        data=excel_data,
                        file_name=f"categorized_returns_{timestamp}.xlsx",
                        use_container_width=True
                    )
                
                with col2:
                    # Also provide CSV option
                    csv_export_df = prepare_export_data(display_data)
                    # Rename 'Complaint_Full' for clarity in CSV
                    if 'Complaint_Full' in csv_export_df.columns:
                        csv_export_df = csv_export_df.rename(columns={'Complaint_Full': 'Complaint (Full)'})
                    csv_data = csv_export_df.to_csv(index=False)
                    st.download_button(
                        label="📥 DOWNLOAD CSV (.csv)",
                        data=csv_data,
                        file_name=f"categorized_returns_{timestamp}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                # Show preview
                st.markdown("### 🔍 Categorized Data Preview")
                
                # Show key columns including Column K
                preview_df = prepare_export_data(st.session_state.categorized_data).head(10)
                # Rename 'Complaint_Full' for display clarity
                if 'Complaint_Full' in preview_df.columns:
                    preview_df = preview_df.rename(columns={'Complaint_Full': 'Complaint (Full)'})
                st.dataframe(preview_df, use_container_width=True)
                
                # Quality team action items
                st.markdown("---")
                st.markdown("""
                <div class="neon-box">
                    <h3 style="color: var(--primary);">💡 QUALITY TEAM ACTION ITEMS</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Add product-specific insights if sorted by product
                if sort_by in ["Product Identifier Tag", "Imported SKU"]:
                    st.markdown("### 📦 Product-Specific Analysis")
                    
                    # Group by the sort field
                    grouped = display_data.groupby(sort_by)
                    
                    # Show top 5 products by return count
                    product_counts = grouped.size().sort_values(ascending=False).head(5)
                    
                    for product, count in product_counts.items():
                        if product and str(product).strip():
                            product_data = grouped.get_group(product)
                            
                            # Get top return reasons for this product
                            reason_counts = product_data['Return_Reason'].value_counts().head(3)
                            
                            with st.expander(f"📦 {product} - {count} returns"):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("**Top Return Reasons:**")
                                    for reason, reason_count in reason_counts.items():
                                        pct = (reason_count / count) * 100
                                        st.markdown(f"- {reason}: {reason_count} ({pct:.1f}%)")
                                
                                with col2:
                                    # Check if this is a quality concern
                                    quality_categories = ['Product Defects/Quality', 'Performance/Effectiveness', 
                                                        'Missing Components', 'Design/Material Issues', 
                                                        'Stability/Positioning Issues']
                                    quality_count = sum(
                                        reason_counts.get(reason, 0) 
                                        for reason in reason_counts.index
                                        if reason in quality_categories
                                    )
                                    quality_pct = (quality_count / count) * 100
                                    
                                    if quality_pct > 30:
                                        st.error(f"⚠️ High Quality Risk: {quality_pct:.1f}% quality issues")
                                    elif quality_pct > 15:
                                        st.warning(f"⚠️ Medium Risk: {quality_pct:.1f}% quality issues")
                                    else:
                                        st.success(f"✅ Low Risk: {quality_pct:.1f}% quality issues")
                
                # Generate actionable insights
                quality_categories = ['Product Defects/Quality', 'Performance/Effectiveness', 'Missing Components', 
                                    'Design/Material Issues', 'Stability/Positioning Issues']
                quality_reasons = [(reason, count) for reason, count in st.session_state.reason_summary.items() 
                                 if reason in quality_categories]
                quality_count = sum(count for _, count in quality_reasons)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    **Key Findings:**
                    - 🎯 {quality_count} quality-related complaints
                    - 📊 Top reason: {max(st.session_state.reason_summary.items(), key=lambda x: x[1])[0] if st.session_state.reason_summary else 'None'}
                    - 🔍 {len(quality_reasons)} different quality issues identified
                    - 📂 Data sources: {', '.join(st.session_state.data_sources)}
                    """)
                
                with col2:
                    st.markdown("""
                    **Recommended Actions:**
                    1. Review all quality-related complaints in Column K
                    2. Create CAPA for top 3 return reasons
                    3. Cross-reference with product reviews
                    4. Update inspection criteria
                    5. Share findings with suppliers
                    """)

if __name__ == "__main__":
    main()
