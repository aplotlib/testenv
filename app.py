"""
Vive Health Quality Complaint Categorizer
AI-Powered Return Reason Classification Tool
Version: 8.0 - Enhanced with Product Tracking & Power Modes

Key Features:
- Dual AI analysis for maximum accuracy
- Adjustable token limits for complex tasks
- Product Identifier Tag tracking
- Empty row handling
- Support for multiple file formats
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

# Set page config first
st.set_page_config(
    page_title="Vive Health Return Categorizer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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
    'version': '8.0',
    'company': 'Vive Health',
    'description': 'AI-Powered Quality Management Tool'
}

# Enhanced color scheme
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
    'claude': '#D4A574',
    'quality': '#FF4444'
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

# Quality-related categories for highlighting
QUALITY_CATEGORIES = [
    'Product Defects/Quality',
    'Performance/Effectiveness',
    'Missing Components',
    'Design/Material Issues',
    'Stability/Positioning Issues',
    'Medical/Health Concerns'
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

def inject_simple_css():
    """Inject simple, clean CSS styling"""
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
        --openai: {COLORS['openai']};
        --claude: {COLORS['claude']};
        --quality: {COLORS['quality']};
    }}
    
    .main-header {{
        background: linear-gradient(135deg, rgba(0, 217, 255, 0.1) 0%, rgba(255, 0, 110, 0.1) 100%);
        border: 2px solid var(--primary);
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 2rem;
    }}
    
    .main-title {{
        font-size: 2.5em;
        font-weight: 700;
        background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }}
    
    .format-box {{
        background: rgba(26, 26, 46, 0.8);
        border: 1px solid var(--accent);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }}
    
    .ai-config-box {{
        background: rgba(0, 217, 255, 0.05);
        border: 2px solid var(--primary);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        text-align: center;
    }}
    
    .results-header {{
        background: rgba(0, 245, 160, 0.1);
        border: 2px solid var(--success);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 2rem 0;
    }}
    
    .stMetric {{
        background: rgba(26, 26, 46, 0.6);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(0, 217, 255, 0.3);
    }}
    
    .stButton > button {{
        font-weight: 600;
        transition: all 0.3s ease;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 217, 255, 0.3);
    }}
    
    /* Simple styling for file uploader */
    .uploadedFile {{
        border: 1px solid var(--primary);
        border-radius: 5px;
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
        'api_keys_configured': False,
        'power_mode': 'standard',  # standard or high
        'use_dual_ai': False,
        'max_tokens': 100  # Default for standard mode
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

def display_required_format():
    """Display the required file format"""
    st.markdown("""
    <div class="format-box">
        <h4 style="color: var(--primary);">📋 Required File Format</h4>
        <p>Your file must contain these columns (names are flexible):</p>
        <ul>
            <li><strong>Complaint</strong> - The return reason/comment text (Required)</li>
            <li><strong>Category</strong> - Where AI will place the categorization (Will be added if missing)</li>
            <li><strong>Product Identifier</strong> - Product name/SKU (Optional but recommended)</li>
            <li><strong>Order #</strong> - Order number (Optional)</li>
            <li><strong>Source</strong> - Where the complaint came from (Optional)</li>
        </ul>
        <p style="color: var(--accent); margin-top: 1rem;">
            💡 The tool will automatically detect your columns and only fill the Category column
        </p>
    </div>
    """, unsafe_allow_html=True)

def setup_simplified_ai():
    """Simplified AI setup with power levels"""
    st.markdown("""
    <div class="ai-config-box">
        <h3 style="color: var(--primary);">⚡ AI Configuration</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Check for keys in secrets
    secret_keys = check_streamlit_secrets()
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("### 🤖 AI Provider")
        provider_options = {
            "openai": "🟢 OpenAI (GPT)",
            "claude": "🟠 Claude (Anthropic)",
            "both": "🔷 Both (Maximum Accuracy)"
        }
        
        provider = st.radio(
            "Select Provider",
            options=list(provider_options.keys()),
            format_func=lambda x: provider_options[x],
            key="provider_radio",
            help="Using both providers compares results for maximum accuracy"
        )
        st.session_state.selected_provider = provider
        
        # Show dual AI option only when "both" is selected
        if provider == "both":
            st.session_state.use_dual_ai = st.checkbox(
                "🔄 Compare both AI results",
                value=True,
                help="Get categorization from both AIs for comparison"
            )
    
    with col2:
        st.markdown("### ⚡ Power Level")
        
        # Detect file size to recommend settings
        file_size_info = ""
        if st.session_state.processed_data is not None:
            num_rows = len(st.session_state.processed_data)
            if num_rows < 500:
                file_size_info = "✅ Small file detected (< 500 rows)"
                recommended = "standard"
            elif num_rows < 2000:
                file_size_info = "📊 Medium file detected (500-2000 rows)"
                recommended = "standard"
            else:
                file_size_info = "📈 Large file detected (> 2000 rows)"
                recommended = "high"
            
            st.info(file_size_info)
        
        power_mode = st.radio(
            "Processing Power",
            options=["standard", "high"],
            format_func=lambda x: {
                "standard": "⚡ Standard (Fast, ~50 tokens)",
                "high": "⚡⚡ High Power (Accurate, ~200 tokens)"
            }[x],
            key="power_radio",
            help="High power uses more tokens for complex categorization"
        )
        
        st.session_state.power_mode = power_mode
        st.session_state.max_tokens = 50 if power_mode == "standard" else 200
        
        # Show token info
        if power_mode == "standard":
            st.caption("Best for: Files < 2000 rows, clear complaints")
        else:
            st.caption("Best for: Large files, ambiguous complaints")
    
    with col3:
        st.markdown("### 🔑 API Keys")
        
        # Show key status
        if provider in ["openai", "both"]:
            if 'openai' in secret_keys:
                st.success("✅ OpenAI key configured")
                st.session_state.api_key_openai = secret_keys['openai']
                os.environ['OPENAI_API_KEY'] = secret_keys['openai']
            else:
                openai_key = st.text_input(
                    "OpenAI API Key",
                    type="password",
                    placeholder="sk-...",
                    key="openai_key_input"
                )
                if openai_key:
                    st.session_state.api_key_openai = openai_key
                    os.environ['OPENAI_API_KEY'] = openai_key
        
        if provider in ["claude", "both"]:
            if 'claude' in secret_keys:
                st.success("✅ Claude key configured")
                st.session_state.api_key_claude = secret_keys['claude']
                os.environ['ANTHROPIC_API_KEY'] = secret_keys['claude']
            else:
                claude_key = st.text_input(
                    "Claude API Key",
                    type="password",
                    placeholder="sk-ant-...",
                    key="claude_key_input"
                )
                if claude_key:
                    st.session_state.api_key_claude = claude_key
                    os.environ['ANTHROPIC_API_KEY'] = claude_key
    
    # Check configuration
    keys_configured = False
    if provider == "openai" and (st.session_state.api_key_openai or 'openai' in secret_keys):
        keys_configured = True
    elif provider == "claude" and (st.session_state.api_key_claude or 'claude' in secret_keys):
        keys_configured = True
    elif provider == "both" and ((st.session_state.api_key_openai or 'openai' in secret_keys) and 
                                  (st.session_state.api_key_claude or 'claude' in secret_keys)):
        keys_configured = True
    
    st.session_state.api_keys_configured = keys_configured
    
    # Show configuration summary
    if keys_configured:
        config_summary = f"""
        ✅ **Configuration Ready**
        - Provider: {provider_options[provider]}
        - Power: {"⚡⚡ High" if power_mode == "high" else "⚡ Standard"}
        - Tokens: ~{st.session_state.max_tokens} per categorization
        """
        st.success(config_summary)
    else:
        st.warning(f"⚠️ Please configure API key(s) for {provider}")
    
    return keys_configured

def process_complaints_file(file_content, filename: str) -> pd.DataFrame:
    """Process complaints file - properly capture all columns including Product Identifier Tag"""
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
        logger.info(f"Number of columns: {len(df.columns)}")
        
        # Look for Complaint column (required)
        complaint_column = None
        for col in df.columns:
            if 'complaint' in col.lower():
                complaint_column = col
                logger.info(f"Found complaint column: '{col}'")
                break
        
        if not complaint_column:
            st.error("❌ No 'Complaint' column found. Please ensure your file has a 'Complaint' column.")
            return None
        
        # Look for Category column
        category_column = None
        for col in df.columns:
            if 'category' in col.lower():
                category_column = col
                logger.info(f"Found category column: '{col}'")
                break
        
        if not category_column:
            # Add Category column if it doesn't exist
            df['Category'] = ''
            category_column = 'Category'
            logger.info("Added Category column to dataframe")
        else:
            # Clear existing categories for AI to refill
            df[category_column] = ''
        
        # Look for Product Identifier Tag column (Column B in your file)
        product_column = None
        product_search_terms = ['product identifier tag', 'product identifier', 'product', 'item']
        
        # First try exact matches
        for term in product_search_terms:
            for col in df.columns:
                if col.lower() == term:
                    product_column = col
                    logger.info(f"Found product column: '{col}'")
                    break
            if product_column:
                break
        
        # If not found, try partial matches
        if not product_column:
            for term in product_search_terms:
                for col in df.columns:
                    if term in col.lower():
                        product_column = col
                        logger.info(f"Found product column by partial match: '{col}'")
                        break
                if product_column:
                    break
        
        # If still not found, check if Column B exists (by position)
        if not product_column and len(df.columns) > 1:
            # Column B is index 1 (0-based)
            product_column = df.columns[1]
            logger.info(f"Using column at position B as product column: '{product_column}'")
        
        # Remove rows that have no complaint data (empty or just whitespace)
        initial_row_count = len(df)
        
        # Filter out rows where complaint is NaN, None, empty string, or just whitespace
        df_filtered = df[df[complaint_column].notna()]  # Remove NaN
        df_filtered = df_filtered[df_filtered[complaint_column] != '']  # Remove empty strings
        df_filtered = df_filtered[df_filtered[complaint_column].str.strip() != '']  # Remove whitespace-only
        
        # Also filter out rows that might have "FALSE" or other non-complaint data
        if product_column:
            # Remove rows where product column is "FALSE" and complaint is empty
            mask = ~((df_filtered[product_column] == 'FALSE') & 
                    (df_filtered[complaint_column].str.strip() == ''))
            df_filtered = df_filtered[mask]
        
        rows_removed = initial_row_count - len(df_filtered)
        if rows_removed > 0:
            logger.info(f"Removed {rows_removed} empty rows")
            st.info(f"📋 Filtered out {rows_removed} empty rows")
        
        # Create a standardized dataframe for processing
        df_standardized = df_filtered.copy()
        
        # Ensure we have the key columns in the standardized version
        if 'Complaint' not in df_standardized.columns:
            df_standardized['Complaint'] = df_filtered[complaint_column]
        
        if 'Category' not in df_standardized.columns:
            df_standardized['Category'] = df_filtered[category_column]
        
        if product_column and 'Product Identifier Tag' not in df_standardized.columns:
            df_standardized['Product Identifier Tag'] = df_filtered[product_column]
        
        # Add FBA_Reason_Code if not present (for compatibility)
        if 'FBA_Reason_Code' not in df_standardized.columns:
            df_standardized['FBA_Reason_Code'] = ''
        
        # Log final structure
        logger.info(f"Final columns: {df_standardized.columns.tolist()}")
        logger.info(f"Rows after filtering: {len(df_standardized)}")
        
        return df_standardized
            
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        st.error(f"Error processing file: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

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
                    
                    # Build row data
                    row = {}
                    
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
                    
                    row['Source'] = 'PDF'
                    row['Category'] = ''
                    
                    if row and 'Complaint' in row:  # Only add if we have a complaint
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
    """Process FBA return report"""
    try:
        # Read tab-separated file
        df = pd.read_csv(io.BytesIO(file_content), sep='\t')
        
        # Build result dataframe
        result_data = {}
        
        # Map columns
        if 'return-date' in df.columns:
            try:
                result_data['Date'] = pd.to_datetime(df['return-date']).dt.strftime('%m/%d/%Y')
            except:
                result_data['Date'] = df['return-date']
        
        # Customer comments as complaint
        if 'customer-comments' in df.columns:
            result_data['Complaint'] = df['customer-comments']
        elif 'comments' in df.columns:
            result_data['Complaint'] = df['comments']
        else:
            st.error("No customer comments column found in FBA file")
            return None
        
        # Product info
        if 'product-name' in df.columns:
            result_data['Product Identifier Tag'] = df['product-name']
        
        if 'sku' in df.columns:
            result_data['Imported SKU'] = df['sku']
        
        if 'asin' in df.columns:
            result_data['ASIN'] = df['asin']
        
        if 'order-id' in df.columns:
            result_data['Order #'] = df['order-id']
        
        result_data['Source'] = 'FBA'
        
        if 'reason' in df.columns:
            result_data['FBA_Reason_Code'] = df['reason']
        
        # Create dataframe
        processed_df = pd.DataFrame(result_data)
        
        # Filter out empty complaints
        processed_df = processed_df[processed_df['Complaint'].notna()]
        processed_df = processed_df[processed_df['Complaint'].str.strip() != '']
        
        # Add Category column
        processed_df['Category'] = ''
        
        return processed_df
        
    except Exception as e:
        logger.error(f"Error processing FBA returns: {e}")
        st.error(f"Error processing FBA returns: {str(e)}")
        return None

def categorize_with_ai(complaint: str, fba_reason: str = None, ai_client=None) -> str:
    """Use AI to categorize a complaint with configurable power level"""
    
    if not ai_client or not ai_client.is_available():
        return fallback_categorization(complaint, fba_reason)
    
    # Get token limit from session state
    max_tokens = st.session_state.get('max_tokens', 50)
    
    # Create prompt (same as before but with power-aware context)
    reasons_list = "\n".join([f"- {reason}" for reason in RETURN_REASONS])
    
    fba_context = ""
    if fba_reason and fba_reason in FBA_REASON_MAP:
        suggested_reason = FBA_REASON_MAP[fba_reason]
        fba_context = f"\n\nFBA reason code: '{fba_reason}' typically indicates '{suggested_reason}'."
    
    # Adjust prompt based on power mode
    if st.session_state.get('power_mode') == 'high':
        # More detailed prompt for high power mode
        prompt = f"""You are a quality management expert for medical devices. Carefully analyze this customer complaint and select the SINGLE MOST APPROPRIATE return category.

Customer Complaint: {complaint}{fba_context}

Available Medical Device Return Categories:
{reasons_list}

Instructions:
1. Read the complaint carefully and consider all nuances
2. Look for keywords but also understand context
3. Choose the ONE category that best matches the primary issue
4. Consider medical device quality and safety implications
5. If multiple categories could apply, choose the most specific one
6. Only use "Other/Miscellaneous" if no other category fits

Think step by step about which category best fits, then respond with ONLY the exact category name."""
    else:
        # Simpler prompt for standard mode
        prompt = f"""Categorize this medical device return complaint.

Complaint: {complaint}{fba_context}

Categories:
{reasons_list}

Reply with ONLY the exact category name that best fits."""

    try:
        # Check if we should use dual AI
        if st.session_state.get('use_dual_ai') and st.session_state.get('selected_provider') == 'both':
            # Get results from both providers
            results = {}
            
            # Try OpenAI
            openai_result = ai_client.call_api(
                messages=[
                    {"role": "system", "content": "You are a quality expert. Always respond with exactly one category from the list."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=max_tokens,
                use_specific_provider='openai'
            )
            
            if openai_result['success']:
                openai_category = openai_result['result'].strip()
                # Validate category
                for valid_cat in RETURN_REASONS:
                    if valid_cat.lower() in openai_category.lower() or openai_category.lower() in valid_cat.lower():
                        results['openai'] = valid_cat
                        break
                if 'openai' not in results:
                    results['openai'] = 'Other/Miscellaneous'
            
            # Try Claude
            claude_result = ai_client.call_api(
                messages=[
                    {"role": "system", "content": "You are a quality expert. Always respond with exactly one category from the list."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=max_tokens,
                use_specific_provider='claude'
            )
            
            if claude_result['success']:
                claude_category = claude_result['result'].strip()
                # Validate category
                for valid_cat in RETURN_REASONS:
                    if valid_cat.lower() in claude_category.lower() or claude_category.lower() in valid_cat.lower():
                        results['claude'] = valid_cat
                        break
                if 'claude' not in results:
                    results['claude'] = 'Other/Miscellaneous'
            
            # If both agree, use that category
            if len(results) == 2 and results['openai'] == results['claude']:
                return results['openai']
            # If they disagree, prefer the one that's not "Other/Miscellaneous"
            elif len(results) == 2:
                if results['openai'] != 'Other/Miscellaneous':
                    return results['openai']
                else:
                    return results['claude']
            # If only one worked, use that
            elif len(results) == 1:
                return list(results.values())[0]
            else:
                return fallback_categorization(complaint, fba_reason)
        
        else:
            # Single provider mode
            response = ai_client.call_api(
                messages=[
                    {"role": "system", "content": "You are a quality management expert categorizing medical device returns. Always respond with the exact text of one return category from the provided list."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=max_tokens
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
                
                # Try partial match
                for r in RETURN_REASONS:
                    if r.lower() in reason.lower() or reason.lower() in r.lower():
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
    """Categorize all complaints - only process rows with actual complaint data"""
    
    ai_client = get_ai_client()
    
    if not ai_client:
        st.error("AI client not initialized")
        return df
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Work on a copy to preserve original structure
    df_copy = df.copy()
    
    # Find the actual complaint and category columns
    complaint_column = None
    category_column = None
    product_column = None
    
    # Find Complaint column
    for col in df_copy.columns:
        if 'complaint' in col.lower():
            complaint_column = col
            break
    
    # Find Category column
    for col in df_copy.columns:
        if 'category' in col.lower():
            category_column = col
            break
    
    # Find Product column
    product_search = ['product identifier tag', 'product identifier', 'product']
    for term in product_search:
        for col in df_copy.columns:
            if term in col.lower():
                product_column = col
                break
        if product_column:
            break
    
    if not complaint_column:
        st.error("No 'Complaint' column found in dataframe")
        return df
    
    if not category_column:
        st.error("No 'Category' column found in dataframe")
        return df
    
    logger.info(f"Using complaint column: '{complaint_column}'")
    logger.info(f"Using category column: '{category_column}'")
    logger.info(f"Using product column: '{product_column}' (if found)")
    
    total_rows = len(df_copy)
    category_counts = Counter()
    product_issues = defaultdict(lambda: defaultdict(int))
    
    # Process each row
    categorized_count = 0
    skipped_count = 0
    
    for idx, row in df_copy.iterrows():
        # Get complaint text
        complaint = ""
        if pd.notna(row[complaint_column]):
            complaint = str(row[complaint_column]).strip()
        
        # Skip rows with no complaint data
        if not complaint or complaint.lower() in ['false', 'true', 'none', 'null', '']:
            skipped_count += 1
            # Leave category empty for empty complaints
            df_copy.at[idx, category_column] = ''
            logger.debug(f"Skipped row {idx}: no valid complaint data")
        else:
            # Get FBA reason if available (for compatibility with FBA files)
            fba_reason = ""
            if 'FBA_Reason_Code' in df_copy.columns:
                fba_reason = str(row.get('FBA_Reason_Code', '')) if pd.notna(row.get('FBA_Reason_Code')) else ""
            
            # Categorize using AI
            reason = categorize_with_ai(complaint, fba_reason, ai_client)
            df_copy.at[idx, category_column] = reason
            category_counts[reason] += 1
            categorized_count += 1
            
            # Track by product if we have product column
            if product_column and pd.notna(row.get(product_column)):
                product = str(row[product_column]).strip()
                # Only track if product is meaningful (not FALSE, empty, etc.)
                if product and product.lower() not in ['false', 'true', 'none', 'null', '']:
                    product_issues[product][reason] += 1
                    logger.debug(f"Tracked issue for product: {product}")
        
        # Update progress
        progress = (idx + 1) / total_rows
        progress_bar.progress(progress)
        status_text.text(f"Processing: {idx + 1}/{total_rows} rows | {categorized_count} categorized | {skipped_count} skipped...")
        
        # Small delay to avoid rate limiting
        if (idx + 1) % 10 == 0:
            time.sleep(0.1)
    
    status_text.text(f"✅ Complete! {categorized_count} complaints categorized, {skipped_count} empty rows skipped.")
    
    # Store summaries
    st.session_state.reason_summary = dict(category_counts)
    st.session_state.product_summary = dict(product_issues)
    
    # Log summary
    logger.info(f"Categorization complete: {categorized_count} categorized, {skipped_count} skipped")
    logger.info(f"Products tracked: {len(product_issues)}")
    
    # Return the dataframe with filled categories
    return df_copy

def display_results_with_products(df: pd.DataFrame):
    """Display results with product breakdown"""
    
    st.markdown("""
    <div class="results-header">
        <h2 style="color: var(--primary); text-align: center;">📊 CATEGORIZATION RESULTS</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Returns", len(df))
    
    with col2:
        categorized = len(df[df['Category'].notna() & (df['Category'] != '')])
        st.metric("Categorized", categorized)
    
    with col3:
        if st.session_state.product_summary:
            st.metric("Products Tracked", len(st.session_state.product_summary))
        else:
            st.metric("Products Tracked", 0)
    
    with col4:
        # Calculate quality percentage
        quality_count = sum(
            count for cat, count in st.session_state.reason_summary.items()
            if cat in QUALITY_CATEGORIES
        )
        quality_pct = (quality_count / categorized * 100) if categorized > 0 else 0
        st.metric("Quality Issues", f"{quality_pct:.1f}%")
    
    # Category distribution
    st.markdown("### 📈 Return Categories")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Top Return Reasons")
        sorted_reasons = sorted(st.session_state.reason_summary.items(), 
                              key=lambda x: x[1], reverse=True)
        
        for reason, count in sorted_reasons[:10]:
            percentage = (count / categorized) * 100 if categorized > 0 else 0
            
            # Color coding for quality issues
            if reason in QUALITY_CATEGORIES:
                color = COLORS['danger']
                icon = "🔴"
            elif reason in ['Size/Fit Issues', 'Wrong Product/Misunderstanding']:
                color = COLORS['warning']
                icon = "🟡"
            else:
                color = COLORS['primary']
                icon = "🔵"
            
            st.markdown(f"""
            <div style="margin: 0.5rem 0;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span>{icon} <span style="color: {color};">{reason}</span></span>
                    <span style="font-weight: bold;">{count} ({percentage:.1f}%)</span>
                </div>
                <div style="background: rgba(255,255,255,0.1); height: 8px; border-radius: 4px; overflow: hidden;">
                    <div style="background: {color}; width: {percentage}%; height: 100%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if st.session_state.product_summary:
            st.markdown("#### Top Products by Returns")
            
            # Calculate total returns per product
            product_totals = [
                (prod, sum(cats.values())) 
                for prod, cats in st.session_state.product_summary.items()
            ]
            top_products = sorted(product_totals, key=lambda x: x[1], reverse=True)[:10]
            
            for product, total in top_products:
                # Get top issue for this product
                top_issue = max(st.session_state.product_summary[product].items(), 
                              key=lambda x: x[1])
                
                # Truncate long product names
                display_name = product[:50] + "..." if len(product) > 50 else product
                
                st.markdown(f"""
                <div style="margin: 0.5rem 0; padding: 0.5rem; background: rgba(26,26,46,0.5); 
                          border-radius: 5px; border-left: 3px solid var(--accent);">
                    <div style="font-weight: bold;">{display_name}</div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.9em;">
                        <span style="color: var(--muted);">Total: {total} returns</span>
                        <span style="color: var(--warning);">Top: {top_issue[0]} ({top_issue[1]})</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

def export_categorized_data(df: pd.DataFrame) -> bytes:
    """Export data preserving exact input format with Category column filled"""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Write main data - preserve exact column order and names
        df.to_excel(writer, sheet_name='Categorized Complaints', index=False)
        
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
                    'Percentage': f"{percentage:.1f}%"
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Category Summary', index=False)
        
        # Add quality sheet
        quality_categories = [
            'Product Defects/Quality',
            'Performance/Effectiveness', 
            'Missing Components',
            'Design/Material Issues',
            'Stability/Positioning Issues',
            'Medical/Health Concerns'
        ]
        
        quality_data = []
        total_quality_issues = 0
        
        for cat in quality_categories:
            count = st.session_state.reason_summary.get(cat, 0)
            total_quality_issues += count
            if count > 0:
                percentage = (count / sum(st.session_state.reason_summary.values())) * 100
                quality_data.append({
                    'Quality Category': cat,
                    'Count': count,
                    'Percentage': f"{percentage:.1f}%"
                })
        
        if quality_data:
            quality_df = pd.DataFrame(quality_data)
            quality_df.to_excel(writer, sheet_name='Quality Issues', index=False)
        
        # Add product analysis if available
        if st.session_state.product_summary:
            product_data = []
            for product, issues in st.session_state.product_summary.items():
                total_returns = sum(issues.values())
                top_issue = max(issues.items(), key=lambda x: x[1]) if issues else ('Unknown', 0)
                
                product_data.append({
                    'Product': product[:100],  # Truncate long names
                    'Total Returns': total_returns,
                    'Top Issue': top_issue[0],
                    'Top Issue Count': top_issue[1]
                })
            
            product_df = pd.DataFrame(sorted(product_data, 
                                           key=lambda x: x['Total Returns'], 
                                           reverse=True)[:50])  # Top 50 products
            product_df.to_excel(writer, sheet_name='Product Analysis', index=False)
        
        # Format workbook
        workbook = writer.book
        
        # Format main sheet
        worksheet1 = writer.sheets['Categorized Complaints']
        
        # Find Category column
        category_col_idx = None
        for idx, col in enumerate(df.columns):
            if 'category' in col.lower():
                category_col_idx = idx
                break
        
        # Auto-adjust columns
        for i, col in enumerate(df.columns):
            max_len = len(str(col)) + 2
            
            # Sample data for width
            sample_data = df[col].astype(str).head(100)
            if len(sample_data) > 0:
                max_len = max(max_len, sample_data.str.len().max())
            
            # Set limits
            if 'complaint' in col.lower():
                max_len = min(max_len, 50)
            else:
                max_len = min(max_len, 40)
            max_len = max(max_len, 10)
            
            worksheet1.set_column(i, i, max_len)
        
        # Highlight Category column
        if category_col_idx is not None:
            category_format = workbook.add_format({
                'bg_color': '#FFF2CC',
                'border': 1
            })
            worksheet1.set_column(category_col_idx, category_col_idx, 30, category_format)
    
    output.seek(0)
    return output.getvalue()

def generate_quality_report():
    """Generate a quality-focused text report"""
    
    total_returns = sum(st.session_state.reason_summary.values())
    quality_issues = {cat: count for cat, count in st.session_state.reason_summary.items() 
                     if cat in QUALITY_CATEGORIES}
    total_quality = sum(quality_issues.values())
    quality_pct = (total_quality / total_returns * 100) if total_returns > 0 else 0
    
    report = f"""VIVE HEALTH QUALITY ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY
================
Total Returns Analyzed: {total_returns}
Quality-Related Returns: {total_quality} ({quality_pct:.1f}%)
Categories Identified: {len(st.session_state.reason_summary)}

QUALITY ISSUES BREAKDOWN
=======================
"""
    
    for cat, count in sorted(quality_issues.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total_returns * 100) if total_returns > 0 else 0
        report += f"{cat}: {count} ({pct:.1f}%)\n"
    
    report += f"""
TOP RETURN CATEGORIES (ALL)
==========================
"""
    
    for cat, count in sorted(st.session_state.reason_summary.items(), 
                            key=lambda x: x[1], reverse=True)[:10]:
        pct = (count / total_returns * 100) if total_returns > 0 else 0
        report += f"{cat}: {count} ({pct:.1f}%)\n"
    
    if st.session_state.product_summary:
        report += f"""
PRODUCT-SPECIFIC ISSUES
======================
"""
        product_totals = [(prod, sum(cats.values())) 
                         for prod, cats in st.session_state.product_summary.items()]
        
        for product, total in sorted(product_totals, key=lambda x: x[1], reverse=True)[:10]:
            top_issue = max(st.session_state.product_summary[product].items(), 
                          key=lambda x: x[1])
            report += f"\n{product}\n"
            report += f"  Total Returns: {total}\n"
            report += f"  Top Issue: {top_issue[0]} ({top_issue[1]} returns)\n"
    
    report += f"""
RECOMMENDATIONS
==============
1. Focus on products with highest quality-related returns
2. Investigate root causes for top categories
3. Implement corrective actions for recurring issues
4. Monitor improvement after interventions
5. Consider design modifications for persistent problems
"""
    
    return report

def main():
    """Main application function with simplified UI"""
    
    if not AI_AVAILABLE:
        st.error("❌ AI module (enhanced_ai_analysis.py) not found!")
        st.stop()
    
    initialize_session_state()
    inject_simple_css()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">VIVE HEALTH RETURN CATEGORIZER</h1>
        <p style="font-size: 1.2em; color: var(--text); margin: 0.5rem 0;">
            Medical Device Quality Management Tool
        </p>
        <p style="color: var(--accent);">
            Upload → AI Categorizes → Download Results
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show required format
    with st.expander("📋 Required File Format & Instructions", expanded=True):
        display_required_format()
        
        st.markdown("### 📊 Supported File Types:")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.info("✅ Excel (.xlsx, .xls)")
        with col2:
            st.info("✅ CSV (.csv)")
        with col3:
            st.info("✅ FBA Returns (.txt)")
        with col4:
            st.info("✅ PDF (Seller Central)")
    
    # AI Configuration Section
    config_ready = setup_simplified_ai()
    
    if not config_ready:
        st.warning("⚠️ Please configure API keys to continue")
        st.stop()
    
    # File upload section
    st.markdown("---")
    st.markdown("### 📁 Upload Files")
    
    uploaded_files = st.file_uploader(
        "Choose file(s) to categorize",
        type=['xlsx', 'xls', 'csv', 'txt', 'pdf'],
        accept_multiple_files=True,
        help="Upload your complaints file - must have a 'Complaint' column"
    )
    
    if uploaded_files:
        all_data = []
        
        with st.spinner("Loading files..."):
            for file in uploaded_files:
                file_content = file.read()
                filename = file.name
                
                df = None
                
                if filename.endswith('.pdf'):
                    if PDFPLUMBER_AVAILABLE:
                        file.seek(0)
                        df = parse_pdf_returns(file)
                    else:
                        st.warning(f"⚠️ PDF support requires pdfplumber: `pip install pdfplumber`")
                
                elif filename.endswith('.txt'):
                    df = process_fba_returns(file_content, filename)
                
                elif filename.endswith(('.xlsx', '.xls', '.csv')):
                    df = process_complaints_file(file_content, filename)
                
                if df is not None and not df.empty:
                    all_data.append(df)
                    st.success(f"✅ Loaded: {filename} ({len(df)} rows with complaints)")
                    
                    # Show what was detected
                    cols_info = st.columns(4)
                    
                    # Check for Product Identifier Tag
                    product_cols = [col for col in df.columns if 'product' in col.lower()]
                    if product_cols:
                        with cols_info[0]:
                            st.info(f"📦 Product column: ✅ Found")
                    
                    # Complaint column
                    complaint_cols = [col for col in df.columns if 'complaint' in col.lower()]
                    with cols_info[1]:
                        st.info(f"💬 Complaints: {len(complaint_cols)} column(s)")
                    
                    # Category column
                    category_cols = [col for col in df.columns if 'category' in col.lower()]
                    with cols_info[2]:
                        st.info(f"📝 Category: {'Found' if category_cols else 'Will add'}")
                    
                    # Rows with data
                    with cols_info[3]:
                        st.info(f"📊 Valid rows: {len(df)}")
                elif df is not None:
                    st.warning(f"⚠️ {filename} - No valid complaint data found")
                else:
                    st.error(f"❌ Could not process: {filename}")
        
        if all_data:
            # Combine all data
            if len(all_data) == 1:
                combined_df = all_data[0]
            else:
                combined_df = pd.concat(all_data, ignore_index=True)
            
            st.session_state.processed_data = combined_df
            
            # Show file summary
            st.success(f"📊 **Total records ready for categorization: {len(combined_df)}**")
            
            # Show power recommendation
            if len(combined_df) > 2000:
                st.info("💡 **Recommendation**: Use High Power mode for this large file")
            elif len(combined_df) > 500:
                st.info("💡 **Recommendation**: Standard power should work well for this file")
            else:
                st.info("💡 **Recommendation**: Standard power is perfect for this file size")
            
            # Preview data
            if st.checkbox("Preview data"):
                # Show first 10 rows, focusing on key columns
                preview_cols = []
                
                # Find key columns to show
                for col in combined_df.columns:
                    if any(term in col.lower() for term in ['complaint', 'product', 'order', 'category']):
                        preview_cols.append(col)
                
                # If we found key columns, show those; otherwise show all
                if preview_cols:
                    st.dataframe(combined_df[preview_cols].head(10))
                else:
                    st.dataframe(combined_df.head(10))
            
            # Categorize button
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                if st.button(
                    "🚀 CATEGORIZE ALL RETURNS", 
                    type="primary", 
                    use_container_width=True,
                    help=f"Process {len(combined_df)} returns using {st.session_state.selected_provider.upper()}"
                ):
                    start_time = time.time()
                    
                    # Show AI configuration being used
                    st.info(f"""
                    **Processing Configuration:**
                    - Provider: {st.session_state.selected_provider.upper()}
                    - Power: {"⚡⚡ High" if st.session_state.power_mode == "high" else "⚡ Standard"}
                    - Max Tokens: {st.session_state.max_tokens}
                    {"- Mode: Dual AI Comparison" if st.session_state.get('use_dual_ai') else ""}
                    """)
                    
                    with st.spinner(f"🤖 Categorizing with {st.session_state.selected_provider.upper()}..."):
                        categorized_df = categorize_all_data(combined_df)
                        st.session_state.categorized_data = categorized_df
                        st.session_state.processing_complete = True
                    
                    # Show completion time
                    elapsed_time = time.time() - start_time
                    st.success(f"✅ Categorization complete in {elapsed_time:.1f} seconds!")
                    
                    # Show quick stats
                    categorized_count = len(categorized_df[categorized_df['Category'].notna() & (categorized_df['Category'] != '')])
                    st.info(f"Categorized {categorized_count} complaints into {len(st.session_state.reason_summary)} categories")
            
            # Show results
            if st.session_state.processing_complete and st.session_state.categorized_data is not None:
                
                display_results_with_products(st.session_state.categorized_data)
                
                # Export section
                st.markdown("---")
                st.markdown("""
                <div style="background: rgba(0, 245, 160, 0.1); border: 2px solid var(--success); 
                          border-radius: 15px; padding: 2rem; text-align: center;">
                    <h3 style="color: var(--success);">✅ CATEGORIZATION COMPLETE!</h3>
                    <p>Your data has been categorized and is ready for download.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Export options
                col1, col2, col3 = st.columns([1, 1, 1])
                
                # Generate exports
                excel_data = export_categorized_data(st.session_state.categorized_data)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                with col1:
                    st.download_button(
                        label="📥 DOWNLOAD EXCEL",
                        data=excel_data,
                        file_name=f"categorized_returns_{timestamp}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                        help="Excel file with categorized data and summary sheets"
                    )
                
                with col2:
                    # Find the original columns to preserve order
                    original_df = st.session_state.categorized_data
                    csv_data = original_df.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 DOWNLOAD CSV",
                        data=csv_data,
                        file_name=f"categorized_returns_{timestamp}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        help="CSV file with categorized data"
                    )
                
                with col3:
                    # Generate quality report
                    quality_report = generate_quality_report()
                    st.download_button(
                        label="📥 QUALITY REPORT",
                        data=quality_report,
                        file_name=f"quality_analysis_{timestamp}.txt",
                        mime="text/plain",
                        use_container_width=True,
                        help="Detailed quality analysis report"
                    )
                
                # Show export info
                st.info("""
                **📋 What's in your export:**
                - ✅ All original columns preserved
                - ✅ Category column filled with AI classifications
                - ✅ Summary sheet with category breakdown
                - ✅ Quality issues analysis
                - ✅ Product-specific insights (if product data available)
                """)

if __name__ == "__main__":
    main()
