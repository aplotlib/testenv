"""
Vive Health Quality Complaint Categorizer
AI-Powered Return Reason Classification Tool
Version: 7.0 - Multi-Provider Support with Correct Export Format

Key Features:
- Choose between OpenAI, Claude, or both AI providers
- Preserves exact input format, only fills Column K
- Supports PDF, Excel, CSV, and FBA return files
- Automatic API key loading from Streamlit secrets
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
import sys

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
    logger.warning("xlsxwriter not available")

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
    logger.warning("pdfplumber not available")

# App Configuration
APP_CONFIG = {
    'title': 'Vive Health Medical Device Return Categorizer',
    'version': '7.0',
    'company': 'Vive Health',
    'description': 'Multi-Provider AI Classification - Column K Categorization'
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

# Medical Device Return Categories
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

# FBA reason code mapping
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
    
    .provider-button {{
        background: rgba(116, 170, 156, 0.2) !important;
        border: 2px solid var(--openai) !important;
    }}
    
    .provider-button-claude {{
        background: rgba(212, 165, 116, 0.2) !important;
        border: 2px solid var(--claude) !important;
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
        'selected_provider': 'openai',
        'api_key_openai': '',
        'api_key_claude': '',
        'api_keys_configured': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def check_streamlit_secrets():
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
            
            # Check for Claude key
            claude_keys = ['ANTHROPIC_API_KEY', 'anthropic_api_key', 'claude_api_key', 'claude']
            for key in claude_keys:
                if key in st.secrets:
                    keys_found['claude'] = str(st.secrets[key]).strip()
                    break
    except Exception as e:
        logger.warning(f"Error checking secrets: {e}")
    
    return keys_found

def setup_ai_provider():
    """Setup AI provider with selection options"""
    st.markdown("""
    <div class="provider-selector">
        <h3 style="color: var(--primary); margin-top: 0;">🤖 AI PROVIDER CONFIGURATION</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Check for keys in secrets
    secret_keys = check_streamlit_secrets()
    
    # Provider selection
    col1, col2, col3 = st.columns(3)
    
    with col1:
        provider_options = ["openai", "claude", "both"]
        provider = st.selectbox(
            "Select AI Provider",
            provider_options,
            key="provider_select",
            help="Choose which AI provider to use for categorization"
        )
        st.session_state.selected_provider = provider
    
    # API key configuration
    with col2:
        if provider in ["openai", "both"]:
            if 'openai' in secret_keys:
                st.success("✅ OpenAI key found in secrets")
                st.session_state.api_key_openai = secret_keys['openai']
                os.environ['OPENAI_API_KEY'] = secret_keys['openai']
            else:
                openai_key = st.text_input(
                    "OpenAI API Key",
                    type="password",
                    placeholder="sk-...",
                    key="openai_key_input",
                    value=st.session_state.api_key_openai
                )
                if openai_key:
                    st.session_state.api_key_openai = openai_key
                    os.environ['OPENAI_API_KEY'] = openai_key
    
    with col3:
        if provider in ["claude", "both"]:
            if 'claude' in secret_keys:
                st.success("✅ Claude key found in secrets")
                st.session_state.api_key_claude = secret_keys['claude']
                os.environ['ANTHROPIC_API_KEY'] = secret_keys['claude']
            else:
                claude_key = st.text_input(
                    "Claude API Key",
                    type="password",
                    placeholder="sk-ant-...",
                    key="claude_key_input",
                    value=st.session_state.api_key_claude
                )
                if claude_key:
                    st.session_state.api_key_claude = claude_key
                    os.environ['ANTHROPIC_API_KEY'] = claude_key
    
    # Check if keys are configured
    keys_configured = False
    if provider == "openai" and (st.session_state.api_key_openai or 'openai' in secret_keys):
        keys_configured = True
    elif provider == "claude" and (st.session_state.api_key_claude or 'claude' in secret_keys):
        keys_configured = True
    elif provider == "both" and ((st.session_state.api_key_openai or 'openai' in secret_keys) and 
                                  (st.session_state.api_key_claude or 'claude' in secret_keys)):
        keys_configured = True
    
    st.session_state.api_keys_configured = keys_configured
    
    # Show status
    if keys_configured:
        st.success(f"✅ {provider.upper()} provider configured and ready")
    else:
        st.warning(f"⚠️ Please configure API key(s) for {provider}")

def get_ai_client():
    """Get or create AI client based on selected provider"""
    if st.session_state.ai_client is None and AI_AVAILABLE:
        try:
            provider = st.session_state.selected_provider
            
            if provider == "openai":
                st.session_state.ai_client = APIClient(AIProvider.OPENAI)
            elif provider == "claude":
                st.session_state.ai_client = APIClient(AIProvider.CLAUDE)
            elif provider == "both":
                st.session_state.ai_client = APIClient(AIProvider.BOTH)
                
            logger.info(f"Created AI client with provider: {provider}")
        except Exception as e:
            logger.error(f"Error creating AI client: {e}")
            st.error(f"Error initializing AI: {str(e)}")
    
    return st.session_state.ai_client

def clean_dataframe(df):
    """Clean dataframe - remove duplicates and standardize"""
    if df is None or df.empty:
        return df
    
    # Remove duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    return df

def parse_pdf_returns(pdf_file) -> pd.DataFrame:
    """Parse Amazon Seller Central returns PDF - flexible format"""
    if not PDFPLUMBER_AVAILABLE:
        st.error("PDF parsing requires pdfplumber. Install with: pip install pdfplumber")
        return None
        
    try:
        import pdfplumber
        
        returns_data = []
        
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if not text:
                    continue
                
                # Extract return entries
                order_pattern = r'Order ID:\s*(\d{3}-\d{7}-\d{7})'
                order_matches = list(re.finditer(order_pattern, text))
                
                for i, match in enumerate(order_matches):
                    start = match.start()
                    end = order_matches[i+1].start() if i+1 < len(order_matches) else len(text)
                    
                    return_block = text[start:end]
                    order_id = match.group(1)
                    
                    # Extract fields
                    asin_match = re.search(r'ASIN:\s*([A-Z0-9]{10})', return_block)
                    sku_match = re.search(r'SKU:\s*([A-Z0-9-]+)', return_block)
                    product_match = re.search(r'(Vive[^\\n]+?)(?:Return Quantity|Return Reason)', return_block, re.DOTALL)
                    reason_match = re.search(r'Return Reason:\s*(.+?)(?:Buyer Comment|Request Date|$)', return_block, re.DOTALL)
                    comment_match = re.search(r'Buyer Comment:\s*(.+?)(?:Request Date|Order Date|$)', return_block, re.DOTALL)
                    date_match = re.search(r'Request Date:\s*(\d{2}/\d{2}/\d{4})', return_block)
                    
                    # Build row data flexibly
                    row = {}
                    
                    # Add whatever columns we can extract
                    if date_match:
                        row['Date'] = date_match.group(1)
                    
                    # Main complaint text
                    complaint_text = comment_match.group(1).strip() if comment_match else ''
                    if complaint_text:
                        row['Complaint'] = complaint_text
                    
                    if product_match:
                        row['Product Identifier Tag'] = product_match.group(1).strip()
                    
                    if sku_match:
                        row['Imported SKU'] = sku_match.group(1)
                    
                    if asin_match:
                        row['ASIN'] = asin_match.group(1)
                    
                    if order_id:
                        row['Order #'] = order_id
                    
                    # Add source
                    row['Source'] = 'PDF'
                    
                    # Add empty Category for AI to fill
                    row['Category'] = ''
                    
                    if row:  # Only add if we extracted something
                        returns_data.append(row)
        
        if returns_data:
            df = pd.DataFrame(returns_data)
            # Ensure we have at least a complaint column
            if 'Complaint' not in df.columns:
                st.warning("No buyer comments found in PDF")
                return None
            return df
        else:
            st.warning("No return data found in PDF")
            return None
            
    except Exception as e:
        logger.error(f"Error parsing PDF: {e}")
        st.error(f"Error parsing PDF: {str(e)}")
        return None

def process_fba_returns(file_content, filename: str) -> pd.DataFrame:
    """Process FBA return report - flexible format"""
    try:
        # Read tab-separated file
        df = pd.read_csv(io.BytesIO(file_content), sep='\t')
        
        # Build result dataframe with available columns
        result_data = {}
        
        # Map columns flexibly
        if 'return-date' in df.columns:
            try:
                result_data['Date'] = pd.to_datetime(df['return-date']).dt.strftime('%m/%d/%Y')
            except:
                result_data['Date'] = df['return-date']
        
        # Look for customer comments as complaint
        if 'customer-comments' in df.columns:
            result_data['Complaint'] = df['customer-comments']
        elif 'comments' in df.columns:
            result_data['Complaint'] = df['comments']
        else:
            # No complaint column - can't process
            st.error("No customer comments column found in FBA file")
            return None
        
        # Add other columns if available
        if 'product-name' in df.columns:
            result_data['Product Identifier Tag'] = df['product-name']
        
        if 'sku' in df.columns:
            result_data['Imported SKU'] = df['sku']
        
        if 'asin' in df.columns:
            result_data['ASIN'] = df['asin']
        
        if 'order-id' in df.columns:
            result_data['Order #'] = df['order-id']
        
        # Add source
        result_data['Source'] = 'FBA'
        
        # Add FBA reason code if available
        if 'reason' in df.columns:
            result_data['FBA_Reason_Code'] = df['reason']
        
        # Create dataframe
        processed_df = pd.DataFrame(result_data)
        
        # Add Category column for AI to fill
        processed_df['Category'] = ''
        
        return processed_df
        
    except Exception as e:
        logger.error(f"Error processing FBA returns: {e}")
        st.error(f"Error processing FBA returns: {str(e)}")
        return None

def process_complaints_file(file_content, filename: str) -> pd.DataFrame:
    """Process complaints file - preserve exact column structure"""
    try:
        # Read file
        if filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_content))
        else:
            df = pd.read_excel(io.BytesIO(file_content))
        
        # Clean columns
        df = clean_dataframe(df)
        
        # Log columns found
        logger.info(f"Columns in uploaded file: {df.columns.tolist()}")
        
        # Look for Complaint column (case-sensitive check first, then case-insensitive)
        complaint_found = False
        
        # First check for exact "Complaint" (capital C)
        if 'Complaint' in df.columns:
            complaint_found = True
            logger.info("Found 'Complaint' column (exact match)")
        else:
            # If not found, check case-insensitive
            for col in df.columns:
                if col.lower() == 'complaint':
                    df = df.rename(columns={col: 'Complaint'})
                    complaint_found = True
                    logger.info(f"Found complaint column '{col}', renamed to 'Complaint'")
                    break
        
        if not complaint_found:
            st.error("No 'Complaint' column found in the file. Please ensure your file has a 'Complaint' column.")
            return None
        
        # Ensure Category column exists (this is what we'll fill)
        if 'Category' not in df.columns:
            # Find where to insert it - typically Column K (11th column)
            if len(df.columns) >= 10:
                # Insert at position 10 (11th column, 0-indexed)
                cols = df.columns.tolist()
                cols.insert(10, 'Category')
                df['Category'] = ''
                df = df[cols]
            else:
                # Just add at the end if file has fewer columns
                df['Category'] = ''
        else:
            # Clear existing categories for AI to refill
            df['Category'] = ''
        
        # Add FBA_Reason_Code if not present (for FBA compatibility)
        if 'FBA_Reason_Code' not in df.columns:
            df['FBA_Reason_Code'] = ''
        
        # Log final structure
        logger.info(f"Final columns: {df.columns.tolist()}")
        logger.info(f"Category column index: {df.columns.tolist().index('Category')}")
        
        return df
            
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        st.error(f"Error processing file: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def categorize_with_ai(complaint: str, fba_reason: str = None, ai_client=None) -> str:
    """Use AI to categorize a complaint"""
    
    if not ai_client or not ai_client.is_available():
        return fallback_categorization(complaint, fba_reason)
    
    # Create prompt
    reasons_list = "\n".join([f"- {reason}" for reason in RETURN_REASONS])
    
    fba_context = ""
    if fba_reason and fba_reason in FBA_REASON_MAP:
        suggested_reason = FBA_REASON_MAP[fba_reason]
        fba_context = f"\n\nFBA reason code: '{fba_reason}' typically indicates '{suggested_reason}'."
    
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
            
            # Validate the reason
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
    
    # Keyword mappings
    keyword_map = {
        'Size/Fit Issues': ['small', 'large', 'size', 'fit', 'tight', 'loose', 'narrow', 'wide', 'big', 'tiny'],
        'Comfort Issues': ['uncomfortable', 'comfort', 'hurts', 'painful', 'pressure', 'sore', 'pain', 'ache'],
        'Product Defects/Quality': ['defective', 'broken', 'damaged', 'quality', 'malfunction', 'faulty', 'poor quality', 'defect', 'crack', 'tear', 'rip'],
        'Performance/Effectiveness': ['not work', 'ineffective', 'useless', 'performance', "doesn't work", 'does not work', 'not effective'],
        'Stability/Positioning Issues': ['unstable', 'slides', 'moves', 'position', 'falls', 'tips', 'wobbly', 'shift'],
        'Equipment Compatibility': ['compatible', 'fit toilet', 'fit wheelchair', 'walker', "doesn't fit", 'not compatible', 'incompatible'],
        'Design/Material Issues': ['heavy', 'bulky', 'material', 'design', 'flimsy', 'thin', 'cheap material'],
        'Wrong Product/Misunderstanding': ['wrong', 'different', 'not as described', 'expected', 'not what', 'incorrect', 'mistake'],
        'Missing Components': ['missing', 'incomplete', 'no instructions', 'parts missing', 'not included'],
        'Customer Error/Changed Mind': ['mistake', 'changed mind', 'no longer', 'patient died', "don't need", 'ordered wrong'],
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
    """Categorize all complaints - only modify Category column"""
    
    ai_client = get_ai_client()
    
    if not ai_client:
        st.error("AI client not initialized")
        return df
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Work on a copy
    df_copy = df.copy()
    
    # Ensure Category column exists
    if 'Category' not in df_copy.columns:
        st.error("Category column not found in dataframe")
        return df
    
    total_rows = len(df_copy)
    category_counts = Counter()
    product_issues = defaultdict(lambda: defaultdict(int))
    
    # Look specifically for "Complaint" column (capital C)
    complaint_column = None
    if 'Complaint' in df_copy.columns:
        complaint_column = 'Complaint'
        logger.info(f"Found 'Complaint' column for categorization")
    else:
        # If not found, look case-insensitive as fallback
        for col in df_copy.columns:
            if col.lower() == 'complaint':
                complaint_column = col
                logger.info(f"Found complaint column '{col}' for categorization")
                break
    
    if not complaint_column:
        st.error("No 'Complaint' column found! Please ensure your file has a 'Complaint' column.")
        return df
    
    # Process each row
    categorized_count = 0
    for idx, row in df_copy.iterrows():
        # Get complaint text from the Complaint column
        complaint = ""
        if pd.notna(row.get(complaint_column)) and str(row.get(complaint_column)).strip():
            complaint = str(row.get(complaint_column))
        
        # Get FBA reason if available
        fba_reason = str(row.get('FBA_Reason_Code', '')) if pd.notna(row.get('FBA_Reason_Code')) else ""
        
        # Categorize
        if complaint or fba_reason:
            reason = categorize_with_ai(complaint, fba_reason, ai_client)
            df_copy.at[idx, 'Category'] = reason
            category_counts[reason] += 1
            categorized_count += 1
            
            # Track by product if column exists
            product_col = None
            for col in ['Product Identifier Tag', 'Product Identifier', 'Product']:
                if col in df_copy.columns:
                    product_col = col
                    break
            
            if product_col:
                product = row.get(product_col, 'Unknown')
                if product and str(product).strip() and product != 'Unknown' and pd.notna(product):
                    product_issues[str(product)][reason] += 1
        else:
            df_copy.at[idx, 'Category'] = 'Other/Miscellaneous'
            category_counts['Other/Miscellaneous'] += 1
        
        # Update progress
        progress = (idx + 1) / total_rows
        progress_bar.progress(progress)
        status_text.text(f"Processing: {idx + 1}/{total_rows} rows | {categorized_count} categorized...")
        
        # Small delay to avoid rate limiting
        if (idx + 1) % 10 == 0:
            time.sleep(0.1)
    
    status_text.text(f"✅ Categorization complete! {categorized_count} complaints categorized.")
    
    # Store summaries
    st.session_state.reason_summary = dict(category_counts)
    st.session_state.product_summary = dict(product_issues)
    
    return df_copy

def export_categorized_data(df: pd.DataFrame) -> bytes:
    """Export data preserving exact input format with Category column filled"""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Write main data - preserve exact column order
        df.to_excel(writer, sheet_name='Categorized Complaints', index=False)
        
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
        
        # Auto-adjust column widths based on content
        for i, col in enumerate(df.columns):
            # Calculate max length for this column
            max_len = len(str(col)) + 2  # Column header
            
            # Check data in column
            for val in df[col].astype(str):
                max_len = max(max_len, len(val))
            
            # Set reasonable limits
            max_len = min(max_len, 50)  # Cap at 50 characters
            max_len = max(max_len, 10)  # Minimum 10 characters
            
            worksheet1.set_column(i, i, max_len)
        
        # Highlight Category column - find its position
        category_col_idx = None
        for idx, col in enumerate(df.columns):
            if col == 'Category':
                category_col_idx = idx
                break
        
        if category_col_idx is not None:
            # Create highlight format
            highlight_format = workbook.add_format({
                'bg_color': '#FFF2CC',
                'border': 1,
                'bold': True
            })
            
            # Apply format to Category column
            worksheet1.set_column(category_col_idx, category_col_idx, 30, highlight_format)
            
            # Also highlight the header
            header_format = workbook.add_format({
                'bg_color': '#FFD966',
                'border': 1,
                'bold': True,
                'align': 'center'
            })
            worksheet1.write(0, category_col_idx, 'Category', header_format)
        
        # Format summary sheet
        worksheet2 = writer.sheets['Summary']
        worksheet2.set_column('A:A', 30)
        worksheet2.set_column('B:B', 10)
        worksheet2.set_column('C:C', 12)
        
        # Add chart to summary
        chart = workbook.add_chart({'type': 'column'})
        chart.add_series({
            'categories': ['Summary', 1, 0, len(summary_data), 0],
            'values': ['Summary', 1, 1, len(summary_data), 1],
            'name': 'Return Count by Category'
        })
        chart.set_title({'name': 'Return Categories Distribution'})
        chart.set_x_axis({'name': 'Category'})
        chart.set_y_axis({'name': 'Count'})
        worksheet2.insert_chart('E2', chart)
    
    output.seek(0)
    return output.getvalue()

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
        categorized = len(df[df['Category'] != ''])
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: var(--success);">{categorized}</h3>
            <p>Categorized</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        unique_reasons = df['Category'].nunique()
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: var(--accent);">{unique_reasons}</h3>
            <p>Unique Reasons</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        if st.session_state.reason_summary:
            top_reason = max(st.session_state.reason_summary.items(), key=lambda x: x[1])[0]
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: var(--secondary); font-size: 1em;">{top_reason}</h3>
                <p>Top Reason</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Show distribution
    st.markdown("---")
    st.markdown("### 📈 Return Reason Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Top Return Reasons")
        sorted_reasons = sorted(st.session_state.reason_summary.items(), key=lambda x: x[1], reverse=True)
        for reason, count in sorted_reasons[:10]:
            percentage = (count / len(df)) * 100
            
            # Color coding
            if reason in ['Product Defects/Quality', 'Performance/Effectiveness', 'Missing Components']:
                color = COLORS['danger']
            elif reason in ['Size/Fit Issues', 'Wrong Product/Misunderstanding']:
                color = COLORS['warning']
            else:
                color = COLORS['primary']
            
            st.markdown(f"""
            <div style="margin: 0.5rem 0;">
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: {color};">{reason}</span>
                    <span>{count} ({percentage:.1f}%)</span>
                </div>
                <div style="background: rgba(255,255,255,0.1); height: 10px; border-radius: 5px;">
                    <div style="background: {color}; width: {percentage}%; height: 100%; border-radius: 5px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Quality insights
        st.markdown("#### 🎯 Quality Insights")
        
        quality_categories = ['Product Defects/Quality', 'Performance/Effectiveness', 'Missing Components', 
                            'Design/Material Issues', 'Stability/Positioning Issues']
        quality_count = sum(st.session_state.reason_summary.get(cat, 0) for cat in quality_categories)
        quality_percentage = (quality_count / len(df)) * 100 if len(df) > 0 else 0
        
        st.markdown(f"""
        <div class="neon-box" style="background: rgba(255, 0, 84, 0.1);">
            <h4 style="color: var(--danger); margin: 0;">Quality-Related Returns</h4>
            <h2 style="color: var(--danger); margin: 0.5rem 0;">{quality_percentage:.1f}%</h2>
            <p style="margin: 0;">({quality_count} complaints)</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main application function"""
    
    st.set_page_config(
        page_title=APP_CONFIG['title'],
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    if not AI_AVAILABLE:
        st.error("❌ AI module (enhanced_ai_analysis.py) not found!")
        st.stop()
    
    initialize_session_state()
    inject_cyberpunk_css()
    
    # Header
    st.markdown(f"""
    <h1>{APP_CONFIG['title']}</h1>
    <p style="text-align: center; color: var(--primary); font-size: 1.2em;">
        {APP_CONFIG['description']}
    </p>
    <p style="text-align: center; color: var(--accent); font-size: 1.1em;">
        📋 Upload your file → 🤖 AI fills the Category column → 📥 Download with categories
    </p>
    """, unsafe_allow_html=True)
    
    # Show version info
    with st.expander("ℹ️ Version Info", expanded=False):
        st.text(f"App Version: {APP_CONFIG['version']}")
        st.text(f"Python: {sys.version.split()[0]}")
        st.text(f"Pandas: {pd.__version__}")
        st.text(f"Streamlit: {st.__version__}")
    
    # AI Provider Setup
    setup_ai_provider()
    
    if not st.session_state.api_keys_configured:
        st.warning("⚠️ Please configure API keys to continue")
        st.stop()
    
    # Instructions
    with st.expander("📖 How to Use", expanded=False):
        st.markdown("""
        ### Quick Guide:
        1. **Select AI Provider**: Choose between OpenAI, Claude, or both
        2. **Upload Files**: Excel, CSV, PDF, or FBA return reports
        3. **Click Categorize**: AI will fill Column K (Category) with return categories
        4. **Download Results**: Get your file back with Column K completed
        
        ### File Requirements:
        - Must have a column with "Complaint" in the name (case-insensitive)
        - The tool will add/update the "Category" column (typically Column K)
        - All other columns are preserved exactly as uploaded
        
        ### Return Categories:
        The AI will categorize returns into one of these categories:
        - Size/Fit Issues
        - Comfort Issues
        - Product Defects/Quality
        - Performance/Effectiveness
        - Equipment Compatibility
        - Wrong Product/Misunderstanding
        - Customer Error/Changed Mind
        - And others...
        """)
    
    # File upload
    st.markdown("""
    <div class="neon-box">
        <h3 style="color: var(--accent);">📁 UPLOAD FILES</h3>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Choose files to categorize",
        type=['xlsx', 'xls', 'csv', 'txt', 'pdf'],
        accept_multiple_files=True,
        help="Upload your complaints file with any column structure - must have a 'Complaint' column"
    )
    
    if uploaded_files:
        all_data = []
        
        with st.spinner("Processing files..."):
            for file in uploaded_files:
                file_content = file.read()
                filename = file.name
                
                df = None
                
                if filename.endswith('.pdf'):
                    if PDFPLUMBER_AVAILABLE:
                        file.seek(0)
                        df = parse_pdf_returns(file)
                    else:
                        st.warning("PDF support requires pdfplumber")
                
                elif filename.endswith('.txt'):
                    df = process_fba_returns(file_content, filename)
                
                elif filename.endswith(('.xlsx', '.xls', '.csv')):
                    df = process_complaints_file(file_content, filename)
                
                if df is not None:
                    all_data.append(df)
                    st.success(f"✅ Loaded: {filename}")
                    
                    # Show what was detected
                    complaint_cols = [col for col in df.columns if 'complaint' in col.lower()]
                    category_col_exists = 'Category' in df.columns
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.info(f"📄 Rows: {len(df)}")
                    with col2:
                        st.info(f"💬 Complaint columns: {len(complaint_cols)}")
                    with col3:
                        st.info(f"📝 Category column: {'Found' if category_col_exists else 'Will be added'}")
                else:
                    st.error(f"❌ Could not process: {filename}")
        
        if all_data:
            # Combine all data
            if len(all_data) == 1:
                combined_df = all_data[0]
            else:
                combined_df = pd.concat(all_data, ignore_index=True)
            
            combined_df = clean_dataframe(combined_df)
            st.session_state.processed_data = combined_df
            
            # Show summary
            st.info(f"📊 Total records: {len(combined_df)}")
            
            # Show column info
            st.markdown("#### 📋 File Structure Detected:")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Columns", len(combined_df.columns))
            
            with col2:
                complaint_cols = [col for col in combined_df.columns if 'complaint' in col.lower()]
                st.metric("Complaint Columns", len(complaint_cols))
            
            with col3:
                category_exists = 'Category' in combined_df.columns
                st.metric("Category Column", "✅ Ready" if category_exists else "📝 Will Add")
            
            # Show column names
            with st.expander("View all column names"):
                st.write("Columns in your file:")
                for i, col in enumerate(combined_df.columns):
                    st.text(f"{i+1}. {col}")
            
            # Show sample (safely)
            if st.checkbox("Show data preview"):
                try:
                    st.dataframe(combined_df.head(10))
                except Exception as e:
                    st.warning("Preview display issue - your data is loaded correctly")
                    # Show first few columns as text
                    st.text("First 5 rows:")
                    st.text(combined_df.head(5).to_string())
            
            # Categorize button
            if st.button("🚀 CATEGORIZE ALL RETURNS", type="primary", use_container_width=True):
                with st.spinner(f"🤖 Categorizing with {st.session_state.selected_provider.upper()}..."):
                    categorized_df = categorize_all_data(combined_df)
                    st.session_state.categorized_data = categorized_df
                    st.session_state.processing_complete = True
                
                st.balloons()
                st.success("✅ Categorization complete!")
            
            # Show results
            if st.session_state.processing_complete and st.session_state.categorized_data is not None:
                
                display_results_summary(st.session_state.categorized_data)
                
                # Export section
                st.markdown("---")
                st.markdown("""
                <div class="success-box">
                    <h3 style="color: var(--success);">📥 EXPORT RESULTS</h3>
                    <p><strong>✅ Category column has been filled with AI-categorized return reasons!</strong></p>
                    <p>Your original file structure is preserved - only the Category column was modified.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show what was done
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Files Processed", len(uploaded_files))
                
                with col2:
                    st.metric("Rows Categorized", len(st.session_state.categorized_data))
                
                with col3:
                    category_col_idx = list(st.session_state.categorized_data.columns).index('Category')
                    st.metric("Category Column", f"Column {chr(65 + category_col_idx)}")
                
                # Generate export
                excel_data = export_categorized_data(st.session_state.categorized_data)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.download_button(
                        label="📥 DOWNLOAD EXCEL",
                        data=excel_data,
                        file_name=f"categorized_returns_{timestamp}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                
                with col2:
                    csv_data = st.session_state.categorized_data.to_csv(index=False)
                    st.download_button(
                        label="📥 DOWNLOAD CSV",
                        data=csv_data,
                        file_name=f"categorized_returns_{timestamp}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                # Preview
                if st.checkbox("Show categorized data preview"):
                    # Show which column is the Category column
                    category_idx = list(st.session_state.categorized_data.columns).index('Category')
                    st.info(f"✨ Category column is at position {category_idx + 1} (Column {chr(65 + category_idx)})")
                    
                    # Display the dataframe
                    st.dataframe(st.session_state.categorized_data.head(20))
                    
                    # Also show a sample of categorizations
                    st.markdown("#### Sample Categorizations:")
                    
                    # Find the complaint column
                    complaint_col = None
                    for col in st.session_state.categorized_data.columns:
                        if 'complaint' in col.lower():
                            complaint_col = col
                            break
                    
                    if complaint_col and 'Category' in st.session_state.categorized_data.columns:
                        sample_df = st.session_state.categorized_data[[complaint_col, 'Category']].head(5)
                        for _, row in sample_df.iterrows():
                            if pd.notna(row[complaint_col]) and row[complaint_col]:
                                st.markdown(f"**Complaint:** {str(row[complaint_col])[:100]}...")
                                st.markdown(f"**→ Category:** `{row['Category']}`")
                                st.markdown("---")

if __name__ == "__main__":
    main()
