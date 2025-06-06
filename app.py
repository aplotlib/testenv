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
    """Parse Amazon Seller Central returns PDF"""
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
                    
                    # Create row matching standard format
                    row = {
                        'Date': date_match.group(1) if date_match else '',
                        'Complaint': (comment_match.group(1).strip() if comment_match else '') + ' - ' + 
                                    (product_match.group(1).strip() if product_match else ''),
                        'Product Identifier Tag': product_match.group(1).strip() if product_match else '',
                        'Imported SKU': sku_match.group(1) if sku_match else '',
                        'ASIN': asin_match.group(1) if asin_match else '',
                        'UDI': '',
                        'CS Ticket # - mapped to Order # if available': order_id,
                        'Source - dropdown': 'PDF',
                        'Categorizing / Investigating Agent': '',
                        'Complaint': comment_match.group(1).strip() if comment_match else '',
                        'Category': '',  # This will be filled by AI
                        'Investigation': '',
                        'FBA_Reason_Code': ''
                    }
                    
                    returns_data.append(row)
        
        if returns_data:
            return pd.DataFrame(returns_data)
        else:
            st.warning("No return data found in PDF")
            return None
            
    except Exception as e:
        logger.error(f"Error parsing PDF: {e}")
        st.error(f"Error parsing PDF: {str(e)}")
        return None

def process_fba_returns(file_content, filename: str) -> pd.DataFrame:
    """Process FBA return report - preserve exact format"""
    try:
        # Read tab-separated file
        df = pd.read_csv(io.BytesIO(file_content), sep='\t')
        
        # Map to standard format
        processed_df = pd.DataFrame({
            'Date': pd.to_datetime(df.get('return-date', '')).dt.strftime('%m/%d/%Y') if 'return-date' in df.columns else '',
            'Complaint': df.get('customer-comments', '') + ' - ' + df.get('product-name', ''),
            'Product Identifier Tag': df.get('product-name', ''),
            'Imported SKU': df.get('sku', ''),
            'ASIN': df.get('asin', ''),
            'UDI': '',
            'CS Ticket # - mapped to Order # if available': df.get('order-id', ''),
            'Source - dropdown': 'FBA',
            'Categorizing / Investigating Agent': '',
            'Complaint': df.get('customer-comments', ''),
            'Category': '',  # This will be filled by AI
            'Investigation': '',
            'FBA_Reason_Code': df.get('reason', '')
        })
        
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
        
        # Check if it already has the expected format
        expected_columns = [
            'Date', 'Complaint', 'Product Identifier Tag', 'Imported SKU', 
            'ASIN', 'UDI', 'CS Ticket # - mapped to Order # if available',
            'Source - dropdown', 'Categorizing / Investigating Agent',
            'Complaint', 'Category', 'Investigation'
        ]
        
        # If it has most expected columns, assume it's already in the right format
        matching_cols = [col for col in expected_columns if col in df.columns]
        if len(matching_cols) >= 8:
            # Ensure Category column exists but is empty for AI to fill
            if 'Category' not in df.columns:
                df['Category'] = ''
            else:
                df['Category'] = ''  # Clear existing categories for AI to refill
            
            # Add FBA_Reason_Code if not present
            if 'FBA_Reason_Code' not in df.columns:
                df['FBA_Reason_Code'] = ''
                
            return df
        else:
            # Try to map columns if they're named differently
            st.warning("File format doesn't match expected structure. Please ensure your file matches the example format.")
            return None
            
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        st.error(f"Error processing file: {str(e)}")
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
        df_copy['Category'] = ''
    
    total_rows = len(df_copy)
    category_counts = Counter()
    product_issues = defaultdict(lambda: defaultdict(int))
    
    # Process each row
    for idx, row in df_copy.iterrows():
        # Get complaint text - handle both single and double complaint columns
        complaint_cols = [col for col in df_copy.columns if 'Complaint' in col]
        complaint = ""
        for col in complaint_cols:
            if pd.notna(row.get(col)) and str(row.get(col)).strip():
                complaint = str(row.get(col))
                break
        
        # Get FBA reason if available
        fba_reason = str(row.get('FBA_Reason_Code', '')) if pd.notna(row.get('FBA_Reason_Code')) else ""
        
        # Categorize
        if complaint or fba_reason:
            reason = categorize_with_ai(complaint, fba_reason, ai_client)
            df_copy.at[idx, 'Category'] = reason
            category_counts[reason] += 1
            
            # Track by product
            product = row.get('Product Identifier Tag', 'Unknown')
            if product and str(product).strip() and product != 'Unknown':
                product_issues[product][reason] += 1
        else:
            df_copy.at[idx, 'Category'] = 'Other/Miscellaneous'
            category_counts['Other/Miscellaneous'] += 1
        
        # Update progress
        progress = (idx + 1) / total_rows
        progress_bar.progress(progress)
        status_text.text(f"Processing: {idx + 1}/{total_rows} complaints categorized...")
        
        # Small delay to avoid rate limiting
        if (idx + 1) % 10 == 0:
            time.sleep(0.1)
    
    status_text.text("✅ Categorization complete!")
    
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
        
        # Set column widths based on standard format
        column_widths = {
            0: 12,   # A - Date
            1: 50,   # B - Complaint (full)
            2: 30,   # C - Product Identifier Tag
            3: 20,   # D - Imported SKU
            4: 15,   # E - ASIN
            5: 15,   # F - UDI
            6: 25,   # G - CS Ticket #
            7: 20,   # H - Source
            8: 25,   # I - Categorizing Agent
            9: 50,   # J - Complaint
            10: 30,  # K - Category (HIGHLIGHTED)
            11: 30   # L - Investigation
        }
        
        for col_num, width in column_widths.items():
            if col_num < len(df.columns):
                worksheet1.set_column(col_num, col_num, width)
        
        # Highlight Column K (Category) - this is the 11th column (index 10)
        highlight_format = workbook.add_format({
            'bg_color': '#FFF2CC',
            'border': 1,
            'bold': True
        })
        
        # Find the Category column index
        category_col_idx = None
        for idx, col in enumerate(df.columns):
            if col == 'Category':
                category_col_idx = idx
                break
        
        if category_col_idx is not None:
            worksheet1.set_column(category_col_idx, category_col_idx, 30, highlight_format)
        
        # Format summary sheet
        worksheet2 = writer.sheets['Summary']
        worksheet2.set_column('A:A', 30)
        worksheet2.set_column('B:B', 10)
        worksheet2.set_column('C:C', 12)
    
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
    """, unsafe_allow_html=True)
    
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
        3. **Click Categorize**: AI will fill Column K with return categories
        4. **Download Results**: Get your file back with Column K completed
        
        ### Important:
        - Your file format is preserved exactly as uploaded
        - Only Column K (Category) is modified by the AI
        - All other data remains unchanged
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
        accept_multiple_files=True
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
            
            # Show sample (safely)
            if st.checkbox("Show data preview"):
                st.dataframe(combined_df.head(10))
            
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
                    <p><strong>Column K has been filled with AI-categorized return reasons!</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
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
                    st.dataframe(st.session_state.categorized_data.head(20))

if __name__ == "__main__":
    main()
