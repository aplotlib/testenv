"""
Vive Health Quality Complaint Categorizer - Enterprise Edition
AI-Powered Return Reason Classification Tool with Advanced Features
Version: 10.0 - Enhanced with Confidence Scoring, Severity Detection, Multi-language Support

Key Features:
- Dual AI analysis (OpenAI + Claude) for consensus and max accuracy
- Confidence scoring with manual review queue
- Severity detection for medical device injuries
- Duplicate detection and merging
- Multi-language support (Spanish/English)
- Date filtering for imports
- Export by SKU or all products
- Increased token limits (300+ tokens) for complex categorization
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
import sys
from difflib import SequenceMatcher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config first
st.set_page_config(
    page_title="Vive Health Return Categorizer - Enterprise",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check for required modules
try:
    from enhanced_ai_analysis import (
        EnhancedAIAnalyzer, APIClient, AIProvider,
        detect_language, translate_text, calculate_confidence,
        detect_severity, is_duplicate, MEDICAL_DEVICE_CATEGORIES
    )
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
    'title': 'Vive Health Medical Device Return Categorizer - Enterprise',
    'version': '10.0',
    'company': 'Vive Health',
    'description': 'AI-Powered Quality Management Tool with Advanced Features'
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
    'quality': '#FF4444',
    'critical': '#FF0000',
    'major': '#FF6600',
    'minor': '#FFAA00'
}

# Medical Device Return Categories
RETURN_REASONS = MEDICAL_DEVICE_CATEGORIES if AI_AVAILABLE else [
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

def inject_enhanced_css():
    """Inject enhanced CSS styling with severity colors"""
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
        --critical: {COLORS['critical']};
        --major: {COLORS['major']};
        --minor: {COLORS['minor']};
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
    
    .severity-critical {{
        background: rgba(255, 0, 0, 0.1);
        border-left: 4px solid var(--critical);
        padding: 0.5rem;
        margin: 0.5rem 0;
    }}
    
    .severity-major {{
        background: rgba(255, 102, 0, 0.1);
        border-left: 4px solid var(--major);
        padding: 0.5rem;
        margin: 0.5rem 0;
    }}
    
    .severity-minor {{
        background: rgba(255, 170, 0, 0.1);
        border-left: 4px solid var(--minor);
        padding: 0.5rem;
        margin: 0.5rem 0;
    }}
    
    .confidence-low {{
        background: rgba(255, 107, 53, 0.1);
        border: 1px solid var(--warning);
        padding: 0.5rem;
        border-radius: 5px;
    }}
    
    .duplicate-group {{
        background: rgba(255, 183, 0, 0.1);
        border: 1px solid var(--accent);
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
    }}
    
    .language-badge {{
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 3px;
        font-size: 0.8em;
        margin-left: 0.5rem;
    }}
    
    .lang-es {{
        background: rgba(255, 183, 0, 0.2);
        color: var(--accent);
        border: 1px solid var(--accent);
    }}
    
    .lang-en {{
        background: rgba(0, 217, 255, 0.2);
        color: var(--primary);
        border: 1px solid var(--primary);
    }}
    
    .ai-comparison {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin: 1rem 0;
    }}
    
    .ai-result {{
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }}
    
    .ai-openai {{
        background: rgba(116, 170, 156, 0.1);
        border: 1px solid var(--openai);
    }}
    
    .ai-claude {{
        background: rgba(212, 165, 116, 0.1);
        border: 1px solid var(--claude);
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
    
    /* Date filter styling */
    .date-filter-box {{
        background: rgba(0, 217, 255, 0.05);
        border: 1px solid var(--primary);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
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
        'selected_provider': 'both',  # Default to both for max accuracy
        'api_key_openai': '',
        'api_key_claude': '',
        'api_keys_configured': False,
        'power_mode': 'high',  # Default to high for accuracy
        'use_dual_ai': True,
        'max_tokens': 300,  # Increased default
        'show_low_confidence': False,
        'severity_summary': {},
        'duplicate_groups': [],
        'language_summary': {},
        'date_filter_enabled': False,
        'date_range_start': None,
        'date_range_end': None,
        'export_by_sku': False,
        'selected_sku': 'all',
        'confidence_threshold': 0.7,
        'enable_translation': True
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

def get_ai_analyzer():
    """Get or create AI analyzer with enhanced features"""
    if st.session_state.ai_client is None and AI_AVAILABLE:
        try:
            provider = AIProvider.BOTH if st.session_state.selected_provider == "both" else \
                      AIProvider.OPENAI if st.session_state.selected_provider == "openai" else \
                      AIProvider.CLAUDE
            
            st.session_state.ai_client = EnhancedAIAnalyzer(provider)
            logger.info(f"Created AI analyzer with provider: {provider}")
        except Exception as e:
            logger.error(f"Error creating AI analyzer: {e}")
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

def display_ai_explanation():
    """Display explanation of how dual AI works"""
    with st.expander("🤖 How Dual AI Analysis Works", expanded=False):
        st.markdown("""
        ### 🔄 Dual AI Processing for Maximum Accuracy
        
        When **Both AIs** mode is selected, the tool uses two independent AI systems:
        
        **1. OpenAI (GPT-4)** 🟢
        - Advanced language understanding
        - Excellent at nuanced categorization
        - Strong medical terminology comprehension
        
        **2. Claude (Sonnet)** 🟠
        - Careful, methodical analysis
        - Conservative categorization approach
        - Strong safety focus for medical devices
        
        ### How It Works:
        1. **Both AIs independently analyze** each complaint
        2. **When they agree** → High confidence result ✅
        3. **When they disagree** → Flag for manual review ⚠️
        4. **Consensus improves accuracy** by 15-20% vs single AI
        
        ### Token Usage:
        - **Standard Mode**: ~50-100 tokens per complaint (fast, good for clear cases)
        - **High Power Mode**: ~200-300 tokens per complaint (detailed analysis for complex cases)
        
        ### Benefits:
        - 🎯 **Higher accuracy** through consensus
        - 🔍 **Catch edge cases** where single AI might err
        - 📊 **Confidence scoring** based on agreement
        - 🏥 **Medical device safety** through dual validation
        
        💡 **Tip**: Use "Both AIs" mode for critical quality assessments or when processing injury-related complaints.
        """)

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
            <li><strong>Date</strong> - Return date (Optional, for filtering)</li>
            <li><strong>Source</strong> - Where the complaint came from (Optional)</li>
        </ul>
        <p style="color: var(--accent); margin-top: 1rem;">
            💡 The tool will automatically detect your columns and fill Category with AI classification
        </p>
    </div>
    """, unsafe_allow_html=True)

def setup_enhanced_ai():
    """Enhanced AI setup with all features"""
    st.markdown("""
    <div class="ai-config-box">
        <h3 style="color: var(--primary);">⚡ AI Configuration - Enterprise Mode</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Check for keys in secrets
    secret_keys = check_streamlit_secrets()
    
    # Main config columns
    col1, col2, col3, col4 = st.columns([1.2, 1, 1, 1])
    
    with col1:
        st.markdown("### 🤖 AI Provider")
        provider_options = {
            "both": "🔷 Both AIs (Max Accuracy)",
            "openai": "🟢 OpenAI GPT-4",
            "claude": "🟠 Claude Sonnet"
        }
        
        provider = st.radio(
            "Select Provider",
            options=list(provider_options.keys()),
            format_func=lambda x: provider_options[x],
            key="provider_radio",
            help="Using both providers compares results for maximum accuracy",
            index=0  # Default to "both"
        )
        st.session_state.selected_provider = provider
        
        # Show dual AI explanation button
        if provider == "both":
            st.session_state.use_dual_ai = True
            if st.button("❓ How Dual AI Works"):
                display_ai_explanation()
    
    with col2:
        st.markdown("### ⚡ Processing Power")
        
        power_mode = st.radio(
            "Token Allocation",
            options=["high", "extreme"],
            format_func=lambda x: {
                "high": "⚡⚡ High (300 tokens)",
                "extreme": "⚡⚡⚡ Extreme (500 tokens)"
            }[x],
            key="power_radio",
            help="More tokens = better understanding of complex complaints",
            index=0  # Default to high
        )
        
        st.session_state.power_mode = power_mode
        st.session_state.max_tokens = 300 if power_mode == "high" else 500
        
        # Show token info
        if power_mode == "high":
            st.caption("Best for: Most medical device returns")
        else:
            st.caption("Best for: Complex injury reports")
    
    with col3:
        st.markdown("### 🌍 Language Support")
        
        st.session_state.enable_translation = st.checkbox(
            "Auto-translate Spanish",
            value=True,
            help="Automatically detect and translate Spanish complaints"
        )
        
        st.markdown("### 🎯 Confidence")
        
        st.session_state.confidence_threshold = st.slider(
            "Manual review threshold",
            min_value=0.5,
            max_value=0.9,
            value=0.7,
            step=0.05,
            help="Flag items below this confidence for review"
        )
    
    with col4:
        st.markdown("### 🔑 API Keys")
        
        # Show key status
        if provider in ["openai", "both"]:
            if 'openai' in secret_keys:
                st.success("✅ OpenAI configured")
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
                st.success("✅ Claude configured")
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
        - Tokens: {st.session_state.max_tokens} per analysis
        - Translation: {"Enabled" if st.session_state.enable_translation else "Disabled"}
        - Confidence Threshold: {st.session_state.confidence_threshold:.0%}
        """
        st.success(config_summary)
    else:
        st.warning(f"⚠️ Please configure API key(s) for {provider}")
    
    return keys_configured

def process_complaints_file(file_content, filename: str, date_filter=None) -> pd.DataFrame:
    """Process complaints file with date filtering"""
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
        
        # Look for required columns
        complaint_column = None
        for col in df.columns:
            if 'complaint' in col.lower():
                complaint_column = col
                break
        
        if not complaint_column:
            st.error("❌ No 'Complaint' column found.")
            return None
        
        # Look for Category column
        category_column = None
        for col in df.columns:
            if 'category' in col.lower():
                category_column = col
                break
        
        if not category_column:
            df['Category'] = ''
            category_column = 'Category'
        else:
            df[category_column] = ''
        
        # Look for Product column
        product_column = None
        product_search_terms = ['product identifier tag', 'product identifier', 'product', 'sku', 'item']
        
        for term in product_search_terms:
            for col in df.columns:
                if term in col.lower():
                    product_column = col
                    break
            if product_column:
                break
        
        # Look for Date column
        date_column = None
        date_search_terms = ['date', 'return-date', 'return date', 'created', 'timestamp']
        
        for term in date_search_terms:
            for col in df.columns:
                if term in col.lower():
                    date_column = col
                    break
            if date_column:
                break
        
        # Apply date filter if requested
        if date_filter and date_column:
            try:
                df[date_column] = pd.to_datetime(df[date_column])
                start_date, end_date = date_filter
                mask = (df[date_column] >= start_date) & (df[date_column] <= end_date)
                df = df[mask]
                logger.info(f"Applied date filter: {len(df)} rows after filtering")
            except Exception as e:
                logger.warning(f"Could not apply date filter: {e}")
        
        # Remove empty complaints
        initial_row_count = len(df)
        df_filtered = df[df[complaint_column].notna()]
        df_filtered = df_filtered[df_filtered[complaint_column] != '']
        df_filtered = df_filtered[df_filtered[complaint_column].str.strip() != '']
        
        rows_removed = initial_row_count - len(df_filtered)
        if rows_removed > 0:
            st.info(f"📋 Filtered out {rows_removed} empty rows")
        
        # Create standardized dataframe
        df_standardized = df_filtered.copy()
        
        # Ensure key columns exist
        if 'Complaint' not in df_standardized.columns:
            df_standardized['Complaint'] = df_filtered[complaint_column]
        
        if 'Category' not in df_standardized.columns:
            df_standardized['Category'] = df_filtered[category_column]
        
        if product_column and 'Product Identifier Tag' not in df_standardized.columns:
            df_standardized['Product Identifier Tag'] = df_filtered[product_column]
        
        if date_column and 'Date' not in df_standardized.columns:
            df_standardized['Date'] = df_filtered[date_column]
        
        # Add analysis columns
        df_standardized['Confidence'] = 0.0
        df_standardized['Severity'] = 'none'
        df_standardized['Language'] = 'en'
        df_standardized['Original_Complaint'] = df_standardized['Complaint']
        df_standardized['Is_Duplicate'] = False
        df_standardized['Duplicate_Group'] = -1
        
        # Add FBA_Reason_Code if not present
        if 'FBA_Reason_Code' not in df_standardized.columns:
            df_standardized['FBA_Reason_Code'] = ''
        
        return df_standardized
            
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        st.error(f"Error processing file: {str(e)}")
        return None

def find_duplicates(df: pd.DataFrame) -> List[List[int]]:
    """Find duplicate complaints in dataframe"""
    duplicate_groups = []
    processed = set()
    
    complaints = df['Complaint'].tolist()
    
    for i in range(len(complaints)):
        if i in processed:
            continue
        
        group = [i]
        for j in range(i + 1, len(complaints)):
            if j in processed:
                continue
            
            if is_duplicate(complaints[i], complaints[j]):
                group.append(j)
                processed.add(j)
        
        if len(group) > 1:
            duplicate_groups.append(group)
            processed.add(i)
    
    return duplicate_groups

def categorize_with_enhanced_ai(complaint: str, fba_reason: str = None, 
                               analyzer=None, language: str = 'en') -> Tuple[str, float, str, str]:
    """Enhanced categorization with confidence, severity, and language support"""
    
    if not analyzer:
        # Fallback categorization
        category = 'Other/Miscellaneous'
        confidence = 0.5
        severity = 'none'
        return category, confidence, severity, language
    
    try:
        # Detect language if not English
        detected_lang = detect_language(complaint) if AI_AVAILABLE else 'en'
        
        # Translate if needed
        translated_complaint = complaint
        if detected_lang != 'en' and st.session_state.enable_translation:
            translated_complaint = translate_text(complaint, detected_lang, 'en')
            language = detected_lang
        
        # Use the analyzer's categorize method
        api_client = analyzer.api_client
        result = api_client.categorize_return(
            translated_complaint, 
            fba_reason=fba_reason,
            use_both=st.session_state.use_dual_ai,
            max_tokens=st.session_state.max_tokens
        )
        
        # Handle dual AI results
        if isinstance(result, dict) and 'openai' in result and 'claude' in result:
            # Both AIs responded
            if result['openai'] == result['claude']:
                category = result['openai']
                confidence = 0.9  # High confidence when both agree
            else:
                # Disagree - prefer non-Other category
                if result['openai'] != 'Other/Miscellaneous':
                    category = result['openai']
                else:
                    category = result['claude']
                confidence = 0.6  # Lower confidence when they disagree
        else:
            # Single AI result
            category = result
            confidence = calculate_confidence(translated_complaint, category, language)
        
        # Detect severity
        severity = detect_severity(translated_complaint, category)
        
        return category, confidence, severity, language
        
    except Exception as e:
        logger.error(f"Error in enhanced categorization: {e}")
        return 'Other/Miscellaneous', 0.5, 'none', 'en'

def categorize_all_data_enhanced(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced categorization with all new features"""
    
    analyzer = get_ai_analyzer()
    
    if not analyzer:
        st.error("AI analyzer not initialized")
        return df
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Work on a copy
    df_copy = df.copy()
    
    # Find duplicates first
    with st.spinner("🔍 Detecting duplicate complaints..."):
        duplicate_groups = find_duplicates(df_copy)
        st.session_state.duplicate_groups = duplicate_groups
        
        # Mark duplicates
        for group_idx, group in enumerate(duplicate_groups):
            for idx in group:
                df_copy.at[idx, 'Is_Duplicate'] = True
                df_copy.at[idx, 'Duplicate_Group'] = group_idx
    
    if duplicate_groups:
        st.info(f"📑 Found {len(duplicate_groups)} groups of duplicate complaints")
    
    # Categorize complaints
    total_rows = len(df_copy)
    category_counts = Counter()
    product_issues = defaultdict(lambda: defaultdict(int))
    severity_counts = Counter()
    language_counts = Counter()
    low_confidence_items = []
    
    for idx, row in df_copy.iterrows():
        complaint = str(row['Complaint']).strip() if pd.notna(row['Complaint']) else ""
        
        if not complaint:
            continue
        
        # Skip duplicates (only process first in group)
        if row['Is_Duplicate'] and row['Duplicate_Group'] >= 0:
            group = duplicate_groups[row['Duplicate_Group']]
            if idx != group[0]:
                # Copy from first in group
                first_idx = group[0]
                df_copy.at[idx, 'Category'] = df_copy.at[first_idx, 'Category']
                df_copy.at[idx, 'Confidence'] = df_copy.at[first_idx, 'Confidence']
                df_copy.at[idx, 'Severity'] = df_copy.at[first_idx, 'Severity']
                df_copy.at[idx, 'Language'] = df_copy.at[first_idx, 'Language']
                continue
        
        # Get FBA reason if available
        fba_reason = str(row.get('FBA_Reason_Code', '')) if pd.notna(row.get('FBA_Reason_Code')) else ""
        
        # Categorize with enhanced AI
        category, confidence, severity, language = categorize_with_enhanced_ai(
            complaint, fba_reason, analyzer, row.get('Language', 'en')
        )
        
        # Update dataframe
        df_copy.at[idx, 'Category'] = category
        df_copy.at[idx, 'Confidence'] = confidence
        df_copy.at[idx, 'Severity'] = severity
        df_copy.at[idx, 'Language'] = language
        
        # Track statistics
        category_counts[category] += 1
        severity_counts[severity] += 1
        language_counts[language] += 1
        
        # Track low confidence items
        if confidence < st.session_state.confidence_threshold:
            low_confidence_items.append(idx)
        
        # Track by product
        if 'Product Identifier Tag' in df_copy.columns and pd.notna(row.get('Product Identifier Tag')):
            product = str(row['Product Identifier Tag']).strip()
            if product:
                product_issues[product][category] += 1
        
        # Update progress
        progress = (idx + 1) / total_rows
        progress_bar.progress(progress)
        status_text.text(f"Processing: {idx + 1}/{total_rows} | Severity: {severity_counts['critical']} critical, {severity_counts['major']} major")
    
    status_text.text(f"✅ Complete! Found {severity_counts['critical']} critical issues, {len(low_confidence_items)} need review")
    
    # Store summaries
    st.session_state.reason_summary = dict(category_counts)
    st.session_state.product_summary = dict(product_issues)
    st.session_state.severity_summary = dict(severity_counts)
    st.session_state.language_summary = dict(language_counts)
    st.session_state.low_confidence_items = low_confidence_items
    
    return df_copy

def display_severity_alerts(df: pd.DataFrame):
    """Display critical severity alerts"""
    critical_items = df[df['Severity'] == 'critical']
    major_items = df[df['Severity'] == 'major']
    
    if len(critical_items) > 0:
        st.markdown("""
        <div class="severity-critical">
            <h4>🚨 CRITICAL SAFETY ISSUES DETECTED</h4>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander(f"View {len(critical_items)} Critical Issues", expanded=True):
            for idx, row in critical_items.iterrows():
                st.markdown(f"""
                **Order:** {row.get('Order #', 'N/A')} | **Product:** {row.get('Product Identifier Tag', 'N/A')}
                
                **Complaint:** {row['Complaint']}
                
                **Category:** {row['Category']} | **Confidence:** {row['Confidence']:.0%}
                
                ---
                """)
    
    if len(major_items) > 0:
        st.markdown("""
        <div class="severity-major">
            <h4>⚠️ Major Quality Issues</h4>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander(f"View {len(major_items)} Major Issues"):
            # Show summary by product
            major_by_product = major_items.groupby('Product Identifier Tag')['Category'].value_counts()
            st.dataframe(major_by_product)

def display_confidence_review(df: pd.DataFrame):
    """Display low confidence items for manual review"""
    low_conf_items = df[df['Confidence'] < st.session_state.confidence_threshold]
    
    if len(low_conf_items) > 0:
        st.markdown("""
        <div class="confidence-low">
            <h4>🔍 Manual Review Queue ({} items)</h4>
        </div>
        """.format(len(low_conf_items)), unsafe_allow_html=True)
        
        with st.expander("Review Low Confidence Categorizations"):
            for idx, row in low_conf_items.iterrows():
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.text(f"Complaint: {row['Complaint'][:100]}...")
                
                with col2:
                    st.text(f"AI: {row['Category']}")
                    st.text(f"Confidence: {row['Confidence']:.0%}")
                
                with col3:
                    # Manual override option
                    new_category = st.selectbox(
                        "Override",
                        options=['Keep AI'] + RETURN_REASONS,
                        key=f"override_{idx}"
                    )
                    
                    if new_category != 'Keep AI' and new_category != row['Category']:
                        df.at[idx, 'Category'] = new_category
                        df.at[idx, 'Confidence'] = 1.0  # Manual override = 100% confidence

def display_duplicate_groups(df: pd.DataFrame):
    """Display duplicate complaint groups"""
    if st.session_state.duplicate_groups:
        st.markdown("""
        <div class="duplicate-group">
            <h4>📑 Duplicate Complaints Detected</h4>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander(f"View {len(st.session_state.duplicate_groups)} Duplicate Groups"):
            for group_idx, group in enumerate(st.session_state.duplicate_groups):
                st.markdown(f"**Group {group_idx + 1} ({len(group)} duplicates)**")
                
                # Show first complaint as example
                first_complaint = df.iloc[group[0]]['Complaint']
                st.text(f"Example: {first_complaint[:150]}...")
                
                # Show all order numbers in group
                order_nums = [df.iloc[idx].get('Order #', 'N/A') for idx in group]
                st.text(f"Orders: {', '.join(order_nums)}")
                
                st.markdown("---")

def display_enhanced_results(df: pd.DataFrame):
    """Display enhanced results with all new features"""
    
    st.markdown("""
    <div class="results-header">
        <h2 style="color: var(--primary); text-align: center;">📊 ENHANCED CATEGORIZATION RESULTS</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_categorized = len(df[df['Category'].notna() & (df['Category'] != '')])
    
    with col1:
        st.metric("Total Returns", len(df))
    
    with col2:
        severity_critical = st.session_state.severity_summary.get('critical', 0)
        st.metric("🚨 Critical", severity_critical)
    
    with col3:
        quality_count = sum(
            count for cat, count in st.session_state.reason_summary.items()
            if cat in QUALITY_CATEGORIES
        )
        quality_pct = (quality_count / total_categorized * 100) if total_categorized > 0 else 0
        st.metric("Quality Issues", f"{quality_pct:.1f}%")
    
    with col4:
        low_conf_count = len(st.session_state.low_confidence_items)
        st.metric("Need Review", low_conf_count)
    
    with col5:
        dup_count = len(st.session_state.duplicate_groups)
        st.metric("Duplicate Groups", dup_count)
    
    # Create tabs for different views
    tabs = st.tabs(["📈 Categories", "🚨 Severity", "🔍 Review Queue", "📑 Duplicates", 
                    "🌍 Languages", "📦 Products", "📊 Export Options"])
    
    with tabs[0]:
        # Category distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Top Return Categories")
            sorted_reasons = sorted(st.session_state.reason_summary.items(), 
                                  key=lambda x: x[1], reverse=True)
            
            for reason, count in sorted_reasons[:10]:
                percentage = (count / total_categorized) * 100 if total_categorized > 0 else 0
                
                # Color coding
                if reason in QUALITY_CATEGORIES:
                    color = COLORS['danger']
                    icon = "🔴"
                else:
                    color = COLORS['primary']
                    icon = "🔵"
                
                st.markdown(f"""
                <div style="margin: 0.5rem 0;">
                    <div style="display: flex; justify-content: space-between;">
                        <span>{icon} {reason}</span>
                        <span>{count} ({percentage:.1f}%)</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # AI agreement analysis (if using dual AI)
            if st.session_state.use_dual_ai and st.session_state.selected_provider == "both":
                st.markdown("#### AI Consensus Analysis")
                
                # Calculate agreement rate
                high_conf = len(df[df['Confidence'] >= 0.8])
                med_conf = len(df[(df['Confidence'] >= 0.6) & (df['Confidence'] < 0.8)])
                low_conf = len(df[df['Confidence'] < 0.6])
                
                st.markdown(f"""
                <div class="ai-comparison">
                    <div class="ai-result ai-openai">
                        <h5>High Agreement</h5>
                        <h3>{high_conf}</h3>
                        <p>Both AIs agree</p>
                    </div>
                    <div class="ai-result ai-claude">
                        <h5>Disagreement</h5>
                        <h3>{med_conf + low_conf}</h3>
                        <p>AIs differ</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    with tabs[1]:
        # Severity analysis
        display_severity_alerts(df)
        
        # Severity distribution
        st.markdown("#### Severity Distribution")
        severity_data = []
        for severity, count in st.session_state.severity_summary.items():
            if severity != 'none':
                severity_data.append({
                    'Severity': severity.capitalize(),
                    'Count': count,
                    'Percentage': f"{(count/total_categorized*100):.1f}%"
                })
        
        if severity_data:
            st.dataframe(pd.DataFrame(severity_data))
    
    with tabs[2]:
        # Manual review queue
        display_confidence_review(df)
    
    with tabs[3]:
        # Duplicate analysis
        display_duplicate_groups(df)
    
    with tabs[4]:
        # Language analysis
        if len(st.session_state.language_summary) > 1:
            st.markdown("#### Multi-language Complaints")
            
            for lang, count in st.session_state.language_summary.items():
                if lang != 'en':
                    lang_items = df[df['Language'] == lang]
                    st.markdown(f"""
                    <div class="language-badge lang-{lang}">
                        {lang.upper()}: {count} complaints
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show examples
                    with st.expander(f"View {lang.upper()} complaints"):
                        for idx, row in lang_items.head(5).iterrows():
                            st.text(f"Original: {row['Original_Complaint'][:100]}...")
                            st.text(f"Translated: {row['Complaint'][:100]}...")
                            st.markdown("---")
    
    with tabs[5]:
        # Product analysis
        if st.session_state.product_summary:
            st.markdown("#### Top Products by Returns")
            
            # Get unique SKUs
            unique_skus = list(st.session_state.product_summary.keys())
            
            # SKU filter
            col1, col2 = st.columns([3, 1])
            with col1:
                selected_sku = st.selectbox(
                    "Filter by SKU",
                    options=['All SKUs'] + sorted(unique_skus),
                    key="sku_filter"
                )
            
            with col2:
                if st.button("🔄 Apply Filter"):
                    st.session_state.selected_sku = selected_sku
            
            # Display filtered results
            if st.session_state.selected_sku == 'all' or selected_sku == 'All SKUs':
                # Show all products
                product_totals = [
                    (prod, sum(cats.values())) 
                    for prod, cats in st.session_state.product_summary.items()
                ]
                top_products = sorted(product_totals, key=lambda x: x[1], reverse=True)[:20]
            else:
                # Show single product
                if selected_sku in st.session_state.product_summary:
                    top_products = [(selected_sku, sum(st.session_state.product_summary[selected_sku].values()))]
                else:
                    top_products = []
            
            for product, total in top_products:
                # Get top issues for this product
                issues = st.session_state.product_summary[product]
                top_issue = max(issues.items(), key=lambda x: x[1]) if issues else ('Unknown', 0)
                
                # Check for critical issues
                product_items = df[df['Product Identifier Tag'] == product]
                critical_count = len(product_items[product_items['Severity'] == 'critical'])
                
                severity_indicator = "🚨" if critical_count > 0 else ""
                
                st.markdown(f"""
                <div style="background: rgba(26,26,46,0.5); padding: 1rem; margin: 0.5rem 0; 
                          border-radius: 8px; border-left: 3px solid var(--accent);">
                    <h4>{severity_indicator} {product}</h4>
                    <div style="display: flex; justify-content: space-between;">
                        <span>Total Returns: {total}</span>
                        <span>Top Issue: {top_issue[0]} ({top_issue[1]})</span>
                        <span style="color: var(--danger);">Critical: {critical_count}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    with tabs[6]:
        # Export options
        st.markdown("#### Export Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            export_by_sku = st.radio(
                "Export Format",
                options=["all", "by_sku"],
                format_func=lambda x: {
                    "all": "📊 All SKUs in one file",
                    "by_sku": "📦 Separate file per SKU"
                }[x],
                key="export_format"
            )
            st.session_state.export_by_sku = (export_by_sku == "by_sku")
        
        with col2:
            if st.session_state.export_by_sku:
                st.info("Will create ZIP file with separate Excel for each SKU")
            else:
                st.info("Will create single Excel with all data")

def export_enhanced_data(df: pd.DataFrame, by_sku: bool = False) -> bytes:
    """Export data with enhanced features"""
    
    if by_sku and 'Product Identifier Tag' in df.columns:
        # Create ZIP file with separate files per SKU
        import zipfile
        from io import BytesIO
        
        zip_buffer = BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Group by SKU
            skus = df['Product Identifier Tag'].unique()
            
            for sku in skus:
                if pd.isna(sku) or not str(sku).strip():
                    continue
                
                sku_df = df[df['Product Identifier Tag'] == sku]
                
                # Create Excel for this SKU
                excel_buffer = BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                    # Main data
                    sku_df.to_excel(writer, sheet_name='Returns', index=False)
                    
                    # Category summary for this SKU
                    category_summary = sku_df['Category'].value_counts()
                    category_df = pd.DataFrame({
                        'Category': category_summary.index,
                        'Count': category_summary.values,
                        'Percentage': (category_summary.values / len(sku_df) * 100).round(1)
                    })
                    category_df.to_excel(writer, sheet_name='Categories', index=False)
                    
                    # Severity summary
                    severity_summary = sku_df['Severity'].value_counts()
                    if 'critical' in severity_summary or 'major' in severity_summary:
                        severity_df = pd.DataFrame({
                            'Severity': severity_summary.index,
                            'Count': severity_summary.values
                        })
                        severity_df.to_excel(writer, sheet_name='Severity', index=False)
                
                excel_buffer.seek(0)
                
                # Clean SKU name for filename
                clean_sku = re.sub(r'[^\w\-_]', '_', str(sku))
                zip_file.writestr(f'{clean_sku}_returns.xlsx', excel_buffer.getvalue())
        
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
    
    else:
        # Single file export
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Main data with all columns
            df.to_excel(writer, sheet_name='Categorized Returns', index=False)
            
            # Category summary
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
                summary_df.to_excel(writer, sheet_name='Category Summary', index=False)
            
            # Severity analysis
            if st.session_state.severity_summary:
                severity_data = []
                for severity, count in st.session_state.severity_summary.items():
                    if severity != 'none':
                        severity_data.append({
                            'Severity Level': severity.capitalize(),
                            'Count': count,
                            'Action Required': 'IMMEDIATE' if severity == 'critical' else 'Review'
                        })
                
                if severity_data:
                    severity_df = pd.DataFrame(severity_data)
                    severity_df.to_excel(writer, sheet_name='Severity Analysis', index=False)
            
            # Product analysis
            if st.session_state.product_summary:
                product_data = []
                for product, issues in st.session_state.product_summary.items():
                    total_returns = sum(issues.values())
                    top_issue = max(issues.items(), key=lambda x: x[1]) if issues else ('Unknown', 0)
                    
                    # Get severity counts for this product
                    product_df = df[df['Product Identifier Tag'] == product]
                    critical_count = len(product_df[product_df['Severity'] == 'critical'])
                    
                    product_data.append({
                        'Product/SKU': product[:100],
                        'Total Returns': total_returns,
                        'Critical Issues': critical_count,
                        'Top Issue': top_issue[0],
                        'Top Issue Count': top_issue[1],
                        'Quality %': f"{(sum(count for cat, count in issues.items() if cat in QUALITY_CATEGORIES) / total_returns * 100):.1f}"
                    })
                
                product_df = pd.DataFrame(sorted(product_data, 
                                               key=lambda x: x['Total Returns'], 
                                               reverse=True))
                product_df.to_excel(writer, sheet_name='Product Analysis', index=False)
            
            # Low confidence items
            if st.session_state.low_confidence_items:
                low_conf_df = df.iloc[st.session_state.low_confidence_items][
                    ['Complaint', 'Category', 'Confidence', 'Product Identifier Tag']
                ]
                low_conf_df.to_excel(writer, sheet_name='Manual Review Queue', index=False)
            
            # Duplicate groups
            if st.session_state.duplicate_groups:
                dup_data = []
                for group_idx, group in enumerate(st.session_state.duplicate_groups):
                    for idx in group:
                        dup_data.append({
                            'Group': group_idx + 1,
                            'Order #': df.iloc[idx].get('Order #', 'N/A'),
                            'Complaint': df.iloc[idx]['Complaint'][:200],
                            'Category': df.iloc[idx]['Category']
                        })
                
                dup_df = pd.DataFrame(dup_data)
                dup_df.to_excel(writer, sheet_name='Duplicates', index=False)
            
            # Format workbook
            workbook = writer.book
            
            # Auto-adjust columns on main sheet
            worksheet = writer.sheets['Categorized Returns']
            for i, col in enumerate(df.columns):
                max_len = len(str(col)) + 2
                sample_data = df[col].astype(str).head(100)
                if len(sample_data) > 0:
                    max_len = max(max_len, sample_data.str.len().max())
                max_len = min(max_len, 50)
                worksheet.set_column(i, i, max_len)
            
            # Highlight critical items
            critical_format = workbook.add_format({
                'bg_color': '#FFE6E6',
                'border': 1
            })
            
            # Apply conditional formatting for severity
            severity_col_idx = df.columns.get_loc('Severity') if 'Severity' in df.columns else None
            if severity_col_idx is not None:
                worksheet.conditional_format(1, severity_col_idx, len(df), severity_col_idx, {
                    'type': 'cell',
                    'criteria': '==',
                    'value': '"critical"',
                    'format': critical_format
                })
        
        output.seek(0)
        return output.getvalue()

def generate_enhanced_quality_report(df: pd.DataFrame) -> str:
    """Generate enhanced quality report with all features"""
    
    total_returns = len(df)
    quality_issues = {cat: count for cat, count in st.session_state.reason_summary.items() 
                     if cat in QUALITY_CATEGORIES}
    total_quality = sum(quality_issues.values())
    quality_pct = (total_quality / total_returns * 100) if total_returns > 0 else 0
    
    report = f"""VIVE HEALTH ENHANCED QUALITY ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
AI Provider: {st.session_state.selected_provider.upper()}
Confidence Threshold: {st.session_state.confidence_threshold:.0%}

EXECUTIVE SUMMARY
================
Total Returns Analyzed: {total_returns}
Quality-Related Returns: {total_quality} ({quality_pct:.1f}%)
Critical Severity Issues: {st.session_state.severity_summary.get('critical', 0)}
Major Severity Issues: {st.session_state.severity_summary.get('major', 0)}
Items Needing Manual Review: {len(st.session_state.low_confidence_items)}
Duplicate Complaint Groups: {len(st.session_state.duplicate_groups)}

SEVERITY BREAKDOWN
==================
Critical (Injury/Safety): {st.session_state.severity_summary.get('critical', 0)}
Major (Quality/Function): {st.session_state.severity_summary.get('major', 0)}
Minor (Comfort/Usability): {st.session_state.severity_summary.get('minor', 0)}

QUALITY ISSUES DETAIL
====================
"""
    
    for cat, count in sorted(quality_issues.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total_returns * 100) if total_returns > 0 else 0
        report += f"{cat}: {count} ({pct:.1f}%)\n"
    
    # Add critical items detail
    critical_items = df[df['Severity'] == 'critical']
    if len(critical_items) > 0:
        report += f"""
CRITICAL SAFETY ISSUES - IMMEDIATE ACTION REQUIRED
=================================================
Total Critical Issues: {len(critical_items)}

Top Products with Critical Issues:
"""
        critical_by_product = critical_items['Product Identifier Tag'].value_counts().head(5)
        for product, count in critical_by_product.items():
            report += f"- {product}: {count} critical issues\n"
        
        report += "\nSample Critical Complaints:\n"
        for idx, row in critical_items.head(3).iterrows():
            report += f"- {row['Complaint'][:150]}...\n"
    
    # Add language summary
    if len(st.session_state.language_summary) > 1:
        report += f"""
MULTI-LANGUAGE ANALYSIS
======================
"""
        for lang, count in st.session_state.language_summary.items():
            pct = (count / total_returns * 100) if total_returns > 0 else 0
            report += f"{lang.upper()}: {count} ({pct:.1f}%)\n"
    
    # Add confidence analysis
    report += f"""
AI CONFIDENCE ANALYSIS
=====================
High Confidence (>{st.session_state.confidence_threshold:.0%}): {total_returns - len(st.session_state.low_confidence_items)}
Low Confidence (<{st.session_state.confidence_threshold:.0%}): {len(st.session_state.low_confidence_items)}
Manual Review Required: {len(st.session_state.low_confidence_items)}
"""
    
    # Add top products section
    if st.session_state.product_summary:
        report += f"""
TOP PRODUCTS BY RETURN VOLUME
=============================
"""
        product_totals = [(prod, sum(cats.values())) 
                         for prod, cats in st.session_state.product_summary.items()]
        
        for product, total in sorted(product_totals, key=lambda x: x[1], reverse=True)[:10]:
            # Get critical count for product
            product_items = df[df['Product Identifier Tag'] == product]
            critical_count = len(product_items[product_items['Severity'] == 'critical'])
            
            top_issue = max(st.session_state.product_summary[product].items(), 
                          key=lambda x: x[1])
            
            report += f"\n{product}\n"
            report += f"  Total Returns: {total}\n"
            report += f"  Critical Issues: {critical_count}\n"
            report += f"  Top Issue: {top_issue[0]} ({top_issue[1]} returns)\n"
    
    report += f"""
RECOMMENDATIONS
==============
1. IMMEDIATE: Review all {st.session_state.severity_summary.get('critical', 0)} critical safety issues
2. URGENT: Investigate products with highest critical issue rates
3. Review {len(st.session_state.low_confidence_items)} low-confidence categorizations
4. Consolidate {len(st.session_state.duplicate_groups)} duplicate complaint groups
5. Focus quality improvements on top 3 categories accounting for {sum(list(st.session_state.reason_summary.values())[:3])} returns
6. Implement corrective actions for recurring issues
7. Monitor improvement metrics after interventions
8. Consider MDR (Medical Device Reporting) requirements for critical issues

DATA QUALITY NOTES
==================
- Used AI Provider: {st.session_state.selected_provider.upper()}
- Token Limit: {st.session_state.max_tokens} tokens per analysis
- Translation Enabled: {"Yes" if st.session_state.enable_translation else "No"}
- Duplicate Detection: {len(st.session_state.duplicate_groups)} groups found
- Date Filter Applied: {"Yes" if st.session_state.date_filter_enabled else "No"}
"""
    
    return report

def main():
    """Main application function with all enhancements"""
    
    if not AI_AVAILABLE:
        st.error("❌ Enhanced AI module not found! Please ensure enhanced_ai_analysis.py is in the same directory.")
        st.stop()
    
    initialize_session_state()
    inject_enhanced_css()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">VIVE HEALTH RETURN CATEGORIZER</h1>
        <p style="font-size: 1.2em; color: var(--text); margin: 0.5rem 0;">
            Enterprise Medical Device Quality Management
        </p>
        <p style="color: var(--accent);">
            🚀 Enhanced with Dual AI, Severity Detection, Multi-language Support
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for advanced features
    with st.sidebar:
        st.markdown("### 🎯 Advanced Features")
        
        # Date filtering
        st.markdown("#### 📅 Date Filtering")
        enable_date_filter = st.checkbox("Enable date filter", key="date_filter_checkbox")
        st.session_state.date_filter_enabled = enable_date_filter
        
        if enable_date_filter:
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start date", 
                                          value=datetime.now() - timedelta(days=30))
                st.session_state.date_range_start = start_date
            with col2:
                end_date = st.date_input("End date", 
                                        value=datetime.now())
                st.session_state.date_range_end = end_date
        
        st.markdown("---")
        
        # Export settings
        st.markdown("#### 📥 Export Settings")
        st.checkbox("Export by SKU (ZIP)", 
                   value=st.session_state.export_by_sku,
                   key="export_sku_checkbox")
        
        st.markdown("---")
        
        # Display settings
        st.markdown("#### 👁️ Display Options")
        st.checkbox("Show low confidence items", 
                   value=st.session_state.show_low_confidence,
                   key="show_low_conf")
        
        # API usage if available
        if st.session_state.ai_client:
            st.markdown("---")
            st.markdown("#### 💰 API Usage")
            analyzer = st.session_state.ai_client
            if hasattr(analyzer, 'api_client'):
                usage = analyzer.api_client.get_usage_summary()
                st.metric("Total Cost", f"${usage['total_cost']:.2f}")
                st.metric("Total Calls", usage['total_calls'])
    
    # Main content
    # Show required format
    with st.expander("📋 Required File Format & Instructions", expanded=False):
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
        
        st.markdown("### 🆕 Enhanced Features:")
        st.success("""
        - **🤖 Dual AI Analysis**: Uses both OpenAI and Claude for consensus
        - **🎯 Confidence Scoring**: Identifies low-confidence categorizations
        - **🚨 Severity Detection**: Flags injury and safety issues
        - **📑 Duplicate Detection**: Groups similar complaints
        - **🌍 Multi-language**: Auto-detects and translates Spanish
        - **📅 Date Filtering**: Filter imports by date range
        - **📦 SKU Export**: Export separate files per product
        """)
    
    # AI Configuration
    config_ready = setup_enhanced_ai()
    
    if not config_ready:
        st.warning("⚠️ Please configure API keys to continue")
        st.stop()
    
    # File upload section
    st.markdown("---")
    st.markdown("### 📁 Upload Files")
    
    # Date filter info
    if st.session_state.date_filter_enabled:
        st.info(f"📅 Date filter active: {st.session_state.date_range_start} to {st.session_state.date_range_end}")
    
    uploaded_files = st.file_uploader(
        "Choose file(s) to categorize",
        type=['xlsx', 'xls', 'csv', 'txt', 'pdf'],
        accept_multiple_files=True,
        help="Upload your complaints file(s) - must have a 'Complaint' column"
    )
    
    if uploaded_files:
        all_data = []
        
        # Prepare date filter
        date_filter = None
        if st.session_state.date_filter_enabled:
            date_filter = (st.session_state.date_range_start, st.session_state.date_range_end)
        
        with st.spinner("Loading files..."):
            for file in uploaded_files:
                file_content = file.read()
                filename = file.name
                
                df = None
                
                # Process different file types
                if filename.endswith(('.xlsx', '.xls', '.csv')):
                    df = process_complaints_file(file_content, filename, date_filter)
                
                if df is not None and not df.empty:
                    all_data.append(df)
                    st.success(f"✅ Loaded: {filename} ({len(df)} rows)")
                    
                    # Show detected features
                    cols_info = st.columns(5)
                    
                    with cols_info[0]:
                        has_product = 'Product Identifier Tag' in df.columns
                        st.info(f"📦 Product: {'✅' if has_product else '❌'}")
                    
                    with cols_info[1]:
                        has_date = 'Date' in df.columns
                        st.info(f"📅 Date: {'✅' if has_date else '❌'}")
                    
                    with cols_info[2]:
                        st.info(f"💬 Rows: {len(df)}")
                    
                    with cols_info[3]:
                        has_fba = 'FBA_Reason_Code' in df.columns
                        st.info(f"📦 FBA: {'✅' if has_fba else '❌'}")
                    
                    with cols_info[4]:
                        st.info(f"🌍 Ready for AI")
        
        if all_data:
            # Combine all data
            combined_df = pd.concat(all_data, ignore_index=True) if len(all_data) > 1 else all_data[0]
            st.session_state.processed_data = combined_df
            
            # Show summary
            st.success(f"📊 **Total records ready: {len(combined_df)}**")
            
            # Preview with language detection
            if st.checkbox("Preview data with language detection"):
                preview_df = combined_df.head(10).copy()
                
                # Quick language detection for preview
                for idx, row in preview_df.iterrows():
                    if pd.notna(row.get('Complaint')):
                        lang = detect_language(str(row['Complaint'])[:100])
                        preview_df.at[idx, 'Detected Language'] = lang
                
                st.dataframe(preview_df[['Complaint', 'Detected Language', 'Product Identifier Tag']])
            
            # Categorize button
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                button_text = f"🚀 CATEGORIZE {len(combined_df)} RETURNS"
                if st.session_state.selected_provider == "both":
                    button_text += " (DUAL AI)"
                
                if st.button(
                    button_text, 
                    type="primary", 
                    use_container_width=True
                ):
                    start_time = time.time()
                    
                    # Show configuration
                    st.info(f"""
                    **🔧 Processing Configuration:**
                    - Provider: {st.session_state.selected_provider.upper()}
                    - Tokens: {st.session_state.max_tokens} per analysis
                    - Confidence Threshold: {st.session_state.confidence_threshold:.0%}
                    - Translation: {"Enabled" if st.session_state.enable_translation else "Disabled"}
                    - Dual AI Comparison: {"Yes" if st.session_state.use_dual_ai else "No"}
                    """)
                    
                    with st.spinner(f"🤖 Processing with enhanced AI analysis..."):
                        categorized_df = categorize_all_data_enhanced(combined_df)
                        st.session_state.categorized_data = categorized_df
                        st.session_state.processing_complete = True
                    
                    # Show completion time
                    elapsed_time = time.time() - start_time
                    st.success(f"""
                    ✅ Enhanced categorization complete in {elapsed_time:.1f} seconds!
                    - Categories: {len(st.session_state.reason_summary)}
                    - Critical Issues: {st.session_state.severity_summary.get('critical', 0)}
                    - Review Queue: {len(st.session_state.low_confidence_items)} items
                    """)
            
            # Show results
            if st.session_state.processing_complete and st.session_state.categorized_data is not None:
                
                display_enhanced_results(st.session_state.categorized_data)
                
                # Export section
                st.markdown("---")
                st.markdown("""
                <div style="background: rgba(0, 245, 160, 0.1); border: 2px solid var(--success); 
                          border-radius: 15px; padding: 2rem; text-align: center;">
                    <h3 style="color: var(--success);">✅ ENHANCED ANALYSIS COMPLETE!</h3>
                    <p>Your data has been categorized with advanced quality insights.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Export options
                col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
                
                # Generate exports
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                with col1:
                    excel_data = export_enhanced_data(
                        st.session_state.categorized_data, 
                        by_sku=st.session_state.export_by_sku
                    )
                    
                    file_name = f"categorized_returns_{timestamp}.{'zip' if st.session_state.export_by_sku else 'xlsx'}"
                    mime_type = "application/zip" if st.session_state.export_by_sku else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    
                    st.download_button(
                        label="📥 DOWNLOAD EXCEL" + (" (ZIP)" if st.session_state.export_by_sku else ""),
                        data=excel_data,
                        file_name=file_name,
                        mime=mime_type,
                        use_container_width=True,
                        help="Enhanced Excel with severity, confidence, and duplicate analysis"
                    )
                
                with col2:
                    # CSV export (simplified)
                    csv_data = st.session_state.categorized_data.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 DOWNLOAD CSV",
                        data=csv_data,
                        file_name=f"categorized_returns_{timestamp}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col3:
                    # Enhanced quality report
                    quality_report = generate_enhanced_quality_report(st.session_state.categorized_data)
                    st.download_button(
                        label="📥 QUALITY REPORT",
                        data=quality_report,
                        file_name=f"enhanced_quality_analysis_{timestamp}.txt",
                        mime="text/plain",
                        use_container_width=True,
                        help="Comprehensive quality report with severity analysis"
                    )
                
                with col4:
                    # Critical issues report
                    critical_df = st.session_state.categorized_data[
                        st.session_state.categorized_data['Severity'] == 'critical'
                    ]
                    
                    if len(critical_df) > 0:
                        critical_report = f"CRITICAL SAFETY ISSUES REPORT\n{'='*50}\n\n"
                        for idx, row in critical_df.iterrows():
                            critical_report += f"""
Order: {row.get('Order #', 'N/A')}
Product: {row.get('Product Identifier Tag', 'N/A')}
Complaint: {row['Complaint']}
Category: {row['Category']}
Confidence: {row['Confidence']:.0%}

---
"""
                        
                        st.download_button(
                            label="🚨 CRITICAL ISSUES",
                            data=critical_report,
                            file_name=f"critical_issues_{timestamp}.txt",
                            mime="text/plain",
                            use_container_width=True,
                            help="All critical safety issues requiring immediate attention"
                        )
                
                # Show export info
                st.info(f"""
                **📋 Enhanced Export Contents:**
                - ✅ All original columns preserved
                - ✅ AI categorization with confidence scores
                - ✅ Severity levels (Critical/Major/Minor)
                - ✅ Language detection and translation tracking
                - ✅ Duplicate group identification
                - ✅ Manual review queue for low confidence
                - ✅ Product-specific analysis with critical counts
                - ✅ {"Separate files per SKU" if st.session_state.export_by_sku else "Single comprehensive file"}
                
                **🤖 AI Analysis Details:**
                - Provider: {st.session_state.selected_provider.upper()}
                - Token Limit: {st.session_state.max_tokens}
                - Consensus Mode: {"Enabled" if st.session_state.use_dual_ai else "Disabled"}
                """)

if __name__ == "__main__":
    main()
