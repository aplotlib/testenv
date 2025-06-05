"""
Vive Health Quality Complaint Categorizer
AI-Powered Return Reason Classification Tool
Version 3.0 - Enhanced for FBA Returns & Complaints Ledger

Requirements:
- streamlit
- pandas
- numpy
- openpyxl or xlsxwriter (for Excel export)
- enhanced_ai_analysis module (must be in same directory)

This tool processes:
1. Product Complaints Ledger (Excel files with complaint text)
2. FBA Return Reports (.txt tab-separated files from Amazon Seller Central)

Output format:
- Columns A-J: Original data from your file
- Column K: Categorized return reason (e.g., "too small", "defective seat")
- Columns L-M: Blank (as requested)
- Columns N-R: Return category mapping data (if CSV file provided)

The AI analyzes complaint text and FBA reason codes to assign the most appropriate
return reason from a standardized list, making it easy for quality analysts to:
- Identify quality issues vs other return types
- Track patterns and trends
- Create targeted corrective actions
- Connect insights across reviews, complaints, and returns
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
from collections import Counter
import re

# Import AI module (using your existing pattern)
try:
    from enhanced_ai_analysis import APIClient
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    logger.warning("AI module not available")

# Check for xlsxwriter
try:
    import xlsxwriter
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False
    logger.warning("xlsxwriter not available - Excel export will use basic format")

# Check for openpyxl as fallback
try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False
    logger.warning("openpyxl not available - Excel export may fail")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# App Configuration
APP_CONFIG = {
    'title': 'Vive Health Quality Complaint Categorizer',
    'version': '3.0',
    'company': 'Vive Health',
    'description': 'AI-Powered Return Reason Classification for Complaints & FBA Returns'
}

# Cyberpunk color scheme (matching your existing apps)
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

# Standard return reasons (from your dropdown)
RETURN_REASONS = [
    'too small',
    'too large',
    'received used/damaged',
    'wrong item',
    'too heavy',
    'bad brakes',
    'bad wheels',
    'uncomfortable',
    'difficult to use',
    'missing parts',
    'defective seat',
    'no issue',
    'not as advertised',
    'defective handles',
    'defective frame',
    'defective/does not work properly',
    'missing or broken parts',
    'performance or quality not adequate',
    'incompatible or not useful',
    'no longer needed',
    'bought by mistake',
    'wrong size',
    'style not as expected',
    'different from website description',
    'damaged during shipping',
    'item never arrived',
    'unauthorized purchase',
    'better price available',
    'ordered wrong item',
    'changed mind',
    'arrived too late',
    'poor quality',
    'not compatible',
    'missing accessories',
    'installation issues',
    'customer damaged',
    'other'
]

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
    
    .neon-box {{
        background: rgba(10, 10, 15, 0.9);
        border: 1px solid var(--primary);
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 0 20px rgba(0, 217, 255, 0.4), inset 0 0 20px rgba(0, 217, 255, 0.1);
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
    
    .reason-badge {{
        display: inline-block;
        padding: 0.4rem 0.8rem;
        border-radius: 5px;
        font-weight: 600;
        background: rgba(0, 217, 255, 0.2);
        border: 1px solid var(--primary);
        color: var(--primary);
        margin: 0.25rem;
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
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'categorized_data' not in st.session_state:
        st.session_state.categorized_data = None
    if 'api_client' not in st.session_state:
        st.session_state.api_client = None
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'category_mapping' not in st.session_state:
        st.session_state.category_mapping = None
    if 'file_type' not in st.session_state:
        st.session_state.file_type = None
    if 'reason_summary' not in st.session_state:
        st.session_state.reason_summary = {}

def get_api_client():
    """Get or create API client"""
    if st.session_state.api_client is None and AI_AVAILABLE:
        st.session_state.api_client = APIClient()
    return st.session_state.api_client

def load_category_mapping():
    """Load the Return Category to Assign mapping file"""
    try:
        # Try to load the mapping file if it exists
        mapping_df = pd.read_csv('Return Category to Assign  return categories.csv')
        st.session_state.category_mapping = mapping_df
        return mapping_df
    except:
        st.warning("Return Category mapping file not found. Please upload it.")
        return None

def categorize_complaint_with_ai(complaint: str, api_client, fba_reason: str = None) -> str:
    """Use AI to categorize a complaint into return reasons"""
    
    # Create prompt with all available return reasons
    reasons_list = "\n".join([f"- {reason}" for reason in RETURN_REASONS])
    
    # Add FBA reason context if available
    fba_context = ""
    if fba_reason:
        # Map common FBA reason codes
        fba_mapping = {
            'NOT_COMPATIBLE': 'incompatible or not useful',
            'DAMAGED_BY_FC': 'received used/damaged',
            'DAMAGED_BY_CARRIER': 'damaged during shipping',
            'DEFECTIVE': 'defective/does not work properly',
            'NOT_AS_DESCRIBED': 'not as advertised',
            'WRONG_ITEM': 'wrong item',
            'MISSING_PARTS': 'missing parts',
            'QUALITY_NOT_ADEQUATE': 'performance or quality not adequate',
            'UNWANTED_ITEM': 'no longer needed',
            'UNAUTHORIZED_PURCHASE': 'unauthorized purchase',
            'CUSTOMER_DAMAGED': 'customer damaged',
            'SWITCHEROO': 'wrong item',
            'EXPIRED_ITEM': 'poor quality',
            'DAMAGED_GLASS_VIAL': 'received used/damaged',
            'DIFFERENT_PRODUCT': 'wrong item',
            'MISSING_ITEM': 'missing parts',
            'NOT_DELIVERED': 'item never arrived',
            'ORDERED_WRONG_ITEM': 'bought by mistake',
            'UNNEEDED_ITEM': 'no longer needed',
            'BAD_GIFT': 'no longer needed',
            'INACCURATE_WEBSITE_DESCRIPTION': 'not as advertised',
            'BETTER_PRICE_AVAILABLE': 'better price available',
            'DOES_NOT_FIT': 'wrong size',
            'NOT_COMPATIBLE_WITH_DEVICE': 'incompatible or not useful',
            'UNSATISFACTORY_PRODUCT': 'performance or quality not adequate',
            'ARRIVED_LATE': 'arrived too late'
        }
        suggested_reason = fba_mapping.get(fba_reason, None)
        if suggested_reason:
            fba_context = f"\nNote: Amazon's system categorized this as '{fba_reason}' which typically maps to '{suggested_reason}'."
    
    prompt = f"""Analyze this customer complaint and select the SINGLE MOST APPROPRIATE return reason.

Customer Complaint: {complaint}{fba_context}

Available Return Reasons:
{reasons_list}

Instructions:
1. Read the complaint carefully
2. Choose the ONE reason that best matches the primary issue
3. If multiple reasons could apply, choose the most specific one
4. Consider the root cause of the complaint
5. If provided, consider Amazon's categorization as additional context
6. Only use "other" if no other reason fits at all

Respond with ONLY the exact return reason text from the list, nothing else."""

    try:
        response = api_client.call_api(
            messages=[
                {"role": "system", "content": "You are a quality management expert categorizing product returns. Always respond with the exact text of one return reason from the provided list."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature for consistency
            max_tokens=50
        )
        
        if response['success']:
            reason = response['result'].strip().lower()
            
            # Validate the reason
            if reason in [r.lower() for r in RETURN_REASONS]:
                # Return the properly cased version
                for r in RETURN_REASONS:
                    if r.lower() == reason:
                        return r
            
            # Try fuzzy matching if exact match fails
            for r in RETURN_REASONS:
                if reason in r.lower() or r.lower() in reason:
                    return r
            
            # Fallback to keyword-based categorization
            return fallback_categorization(complaint, fba_reason)
        else:
            logger.error(f"AI categorization failed: {response.get('error')}")
            return fallback_categorization(complaint, fba_reason)
            
    except Exception as e:
        logger.error(f"Error in AI categorization: {e}")
        return fallback_categorization(complaint, fba_reason)

def fallback_categorization(complaint: str, fba_reason: str = None) -> str:
    """Fallback keyword-based categorization"""
    complaint_lower = complaint.lower()
    
    # If we have an FBA reason code, check mapping first
    if fba_reason:
        fba_mapping = {
            'NOT_COMPATIBLE': 'incompatible or not useful',
            'DAMAGED_BY_FC': 'received used/damaged',
            'DAMAGED_BY_CARRIER': 'damaged during shipping',
            'DEFECTIVE': 'defective/does not work properly',
            'NOT_AS_DESCRIBED': 'not as advertised',
            'WRONG_ITEM': 'wrong item',
            'MISSING_PARTS': 'missing parts',
            'QUALITY_NOT_ADEQUATE': 'performance or quality not adequate',
            'UNWANTED_ITEM': 'no longer needed',
            'UNAUTHORIZED_PURCHASE': 'unauthorized purchase',
            'CUSTOMER_DAMAGED': 'customer damaged',
            'SWITCHEROO': 'wrong item',
            'EXPIRED_ITEM': 'poor quality',
            'DAMAGED_GLASS_VIAL': 'received used/damaged',
            'DIFFERENT_PRODUCT': 'wrong item',
            'MISSING_ITEM': 'missing parts',
            'NOT_DELIVERED': 'item never arrived',
            'ORDERED_WRONG_ITEM': 'bought by mistake',
            'UNNEEDED_ITEM': 'no longer needed',
            'BAD_GIFT': 'no longer needed',
            'INACCURATE_WEBSITE_DESCRIPTION': 'not as advertised',
            'BETTER_PRICE_AVAILABLE': 'better price available',
            'DOES_NOT_FIT': 'wrong size',
            'NOT_COMPATIBLE_WITH_DEVICE': 'incompatible or not useful',
            'UNSATISFACTORY_PRODUCT': 'performance or quality not adequate',
            'ARRIVED_LATE': 'arrived too late'
        }
        if fba_reason in fba_mapping:
            return fba_mapping[fba_reason]
    
    # Define keyword mappings
    keyword_map = {
        'too small': ['small', 'tight', 'narrow', 'short'],
        'too large': ['large', 'big', 'loose', 'long', 'oversized'],
        'received used/damaged': ['used', 'damaged', 'scratched', 'dented', 'torn'],
        'wrong item': ['wrong', 'incorrect', 'different item', 'not what i ordered'],
        'too heavy': ['heavy', 'weight'],
        'bad brakes': ['brake', 'braking'],
        'bad wheels': ['wheel', 'wheels', 'caster'],
        'uncomfortable': ['uncomfortable', 'comfort', 'hurts', 'painful'],
        'difficult to use': ['difficult', 'hard to use', 'complicated', 'confusing'],
        'missing parts': ['missing', 'incomplete', 'not included'],
        'defective seat': ['seat', 'cushion', 'padding'],
        'not as advertised': ['not as described', 'misleading', 'false advertising'],
        'defective handles': ['handle', 'grip', 'handlebar'],
        'defective frame': ['frame', 'structure', 'bent'],
        'defective/does not work properly': ['defective', 'broken', 'doesn\'t work', 'malfunction', 'faulty', 'not working'],
        'no longer needed': ['don\'t need', 'no longer', 'changed mind', 'patient died'],
        'bought by mistake': ['mistake', 'accident', 'wrong order'],
        'damaged during shipping': ['shipping damage', 'arrived damaged', 'package damaged']
    }
    
    # Check each keyword mapping
    for reason, keywords in keyword_map.items():
        for keyword in keywords:
            if keyword in complaint_lower:
                return reason
    
    # General quality issues
    if any(word in complaint_lower for word in ['quality', 'cheap', 'poor', 'flimsy']):
        return 'performance or quality not adequate'
    
    # Size issues
    if any(word in complaint_lower for word in ['size', 'fit']):
        return 'wrong size'
    
    # Compatibility
    if any(word in complaint_lower for word in ['compatible', 'doesn\'t fit', 'won\'t work with']):
        return 'incompatible or not useful'
    
    return 'other'

def process_complaints_file(file_content, filename: str) -> pd.DataFrame:
    """Process the uploaded complaints file or FBA return report"""
    try:
        # Detect file type
        if filename.endswith(('.xlsx', '.xls')):
            # Excel file - likely complaints ledger
            df = pd.read_excel(io.BytesIO(file_content))
            file_type = 'complaints_ledger'
        elif filename.endswith('.txt'):
            # Text file - likely FBA return report (tab-separated)
            try:
                df = pd.read_csv(io.BytesIO(file_content), sep='\t')
                file_type = 'fba_returns'
            except:
                # Try comma-separated if tab fails
                df = pd.read_csv(io.BytesIO(file_content))
                file_type = 'fba_returns'
        else:
            # CSV file
            df = pd.read_csv(io.BytesIO(file_content))
            file_type = 'unknown'
        
        # Log the columns found
        logger.info(f"Columns found in file: {df.columns.tolist()}")
        
        # Detect file type by columns if not already determined
        if 'customer-comments' in df.columns and 'reason' in df.columns:
            file_type = 'fba_returns'
        elif 'Complaint' in df.columns:
            file_type = 'complaints_ledger'
        
        # Process based on file type
        if file_type == 'fba_returns':
            # FBA Return Report format
            st.info("📦 Detected FBA Return Report format")
            
            # Rename columns to match our expected format
            df = df.rename(columns={
                'customer-comments': 'Complaint',
                'return-date': 'Date',
                'order-id': 'Order #',
                'product-name': 'Product Identifier Tag',
                'sku': 'Imported SKU'
            })
            
            # Add the FBA reason code as additional context
            if 'reason' in df.columns:
                df['FBA_Reason_Code'] = df['reason']
            
            # Store file type for later processing
            st.session_state.file_type = 'fba_returns'
            
        else:
            # Complaints ledger format
            st.info("📋 Detected Complaints Ledger format")
            
            # Check if we have the Complaint column
            if 'Complaint' not in df.columns:
                # Try to find complaint column with different names
                complaint_cols = [col for col in df.columns if 'complaint' in col.lower() or 'comment' in col.lower()]
                if complaint_cols:
                    df = df.rename(columns={complaint_cols[0]: 'Complaint'})
                else:
                    st.error("Could not find 'Complaint' or 'customer-comments' column in the file")
                    st.info(f"Available columns: {', '.join(df.columns)}")
                    return None
            
            st.session_state.file_type = 'complaints_ledger'
        
        return df
        
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        st.error(f"Error reading file: {str(e)}")
        return None

def categorize_all_complaints(df: pd.DataFrame) -> pd.DataFrame:
    """Categorize all complaints and prepare export format"""
    
    api_client = get_api_client()
    if not api_client or not api_client.is_available():
        st.warning("AI not available. Using keyword-based categorization.")
        use_ai = False
    else:
        use_ai = True
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Create a copy for processing
    df_copy = df.copy()
    
    # Add the return reason column (Column K)
    df_copy['Return_Reason'] = ''
    
    total_rows = len(df_copy)
    categorized = 0
    
    # Category counter for summary
    reason_counts = Counter()
    
    # Process each row
    for idx, row in df_copy.iterrows():
        complaint = str(row['Complaint']) if pd.notna(row.get('Complaint')) else ""
        
        # Get FBA reason if available
        fba_reason = None
        if 'FBA_Reason_Code' in row and pd.notna(row.get('FBA_Reason_Code')):
            fba_reason = str(row['FBA_Reason_Code'])
        
        if complaint.strip():
            if use_ai:
                reason = categorize_complaint_with_ai(complaint, api_client, fba_reason)
            else:
                reason = fallback_categorization(complaint, fba_reason)
            
            df_copy.at[idx, 'Return_Reason'] = reason
            reason_counts[reason] += 1
        else:
            # No complaint text, but might have FBA reason
            if fba_reason:
                # Map FBA reason to return reason
                fba_mapping = {
                    'NOT_COMPATIBLE': 'incompatible or not useful',
                    'DAMAGED_BY_FC': 'received used/damaged',
                    'DAMAGED_BY_CARRIER': 'damaged during shipping',
                    'DEFECTIVE': 'defective/does not work properly',
                    'NOT_AS_DESCRIBED': 'not as advertised',
                    'WRONG_ITEM': 'wrong item',
                    'MISSING_PARTS': 'missing parts',
                    'QUALITY_NOT_ADEQUATE': 'performance or quality not adequate',
                    'UNWANTED_ITEM': 'no longer needed',
                    'UNAUTHORIZED_PURCHASE': 'unauthorized purchase',
                    'CUSTOMER_DAMAGED': 'customer damaged',
                    'SWITCHEROO': 'wrong item',
                    'EXPIRED_ITEM': 'poor quality',
                    'DAMAGED_GLASS_VIAL': 'received used/damaged',
                    'DIFFERENT_PRODUCT': 'wrong item',
                    'MISSING_ITEM': 'missing parts',
                    'NOT_DELIVERED': 'item never arrived',
                    'ORDERED_WRONG_ITEM': 'bought by mistake',
                    'UNNEEDED_ITEM': 'no longer needed',
                    'BAD_GIFT': 'no longer needed',
                    'INACCURATE_WEBSITE_DESCRIPTION': 'not as advertised',
                    'BETTER_PRICE_AVAILABLE': 'better price available',
                    'DOES_NOT_FIT': 'wrong size',
                    'NOT_COMPATIBLE_WITH_DEVICE': 'incompatible or not useful',
                    'UNSATISFACTORY_PRODUCT': 'performance or quality not adequate',
                    'ARRIVED_LATE': 'arrived too late'
                }
                reason = fba_mapping.get(fba_reason, 'other')
                df_copy.at[idx, 'Return_Reason'] = reason
                reason_counts[reason] += 1
            else:
                df_copy.at[idx, 'Return_Reason'] = 'no issue'
                reason_counts['no issue'] += 1
        
        categorized += 1
        progress = categorized / total_rows
        progress_bar.progress(progress)
        status_text.text(f"Processing: {categorized}/{total_rows} complaints categorized...")
        
        # Add small delay to avoid rate limiting
        if use_ai and categorized % 10 == 0:
            time.sleep(0.5)
    
    status_text.text("✅ Categorization complete!")
    
    # Store summary
    st.session_state.reason_summary = dict(reason_counts)
    
    return df_copy

def prepare_export_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for export with specific column structure"""
    
    # Check file type
    file_type = st.session_state.get('file_type', 'complaints_ledger')
    
    if file_type == 'fba_returns':
        # FBA Return Report format
        # Original columns from FBA report
        original_columns = ['return-date', 'order-id', 'sku', 'asin', 'fnsku', 
                           'product-name', 'quantity', 'fulfillment-center-id', 
                           'detailed-disposition', 'reason', 'status', 
                           'license-plate-number', 'customer-comments']
        
        # Map back to original column names
        column_mapping = {
            'Date': 'return-date',
            'Order #': 'order-id',
            'Imported SKU': 'sku',
            'Product Identifier Tag': 'product-name',
            'Complaint': 'customer-comments',
            'FBA_Reason_Code': 'reason'
        }
        
        # Create export dataframe with original columns
        export_df = pd.DataFrame()
        
        # Add original columns in order (A-J equivalent)
        for i, col in enumerate(original_columns[:10]):  # First 10 columns (A-J)
            if col in df.columns:
                export_df[col] = df[col]
            else:
                # Try to find mapped column
                reverse_mapping = {v: k for k, v in column_mapping.items()}
                if col in reverse_mapping and reverse_mapping[col] in df.columns:
                    export_df[col] = df[reverse_mapping[col]]
                else:
                    # Use empty column if not found
                    export_df[col] = ''
    
    else:
        # Complaints Ledger format
        # Create export dataframe with original columns (A-J)
        original_columns = ['Date', 'Product Identifier Tag', 'Imported SKU', 'UDI', 
                           'CS Ticket #', 'Order #', 'Source', 'Categorizing / Investigating Agent',
                           'Complaint', 'Review stars']
        
        # Get available original columns
        export_columns = [col for col in original_columns if col in df.columns]
        export_df = df[export_columns].copy()
    
    # Add Column K - Return Reason
    export_df['Return Reason'] = df['Return_Reason']
    
    # Add blank columns L and M
    export_df['Blank1'] = ''
    export_df['Blank2'] = ''
    
    # Add mapping data if available (columns N onwards)
    if st.session_state.category_mapping is not None:
        # For each row, we could match the return reason to the mapping
        # For now, add the column headers
        mapping_columns = ['on report', 'Amazon Reason Code', 'Meaning', 'Odoo Return Category?', 'Vive Reason Code']
        
        for col in mapping_columns:
            export_df[col] = ''
        
        # You could implement matching logic here based on return reasons
        # For example, matching "too small" to CS3, etc.
    
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
            <p>Total Complaints</p>
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
    
    # Sort reasons by count
    sorted_reasons = sorted(st.session_state.reason_summary.items(), key=lambda x: x[1], reverse=True)
    
    # Create two columns for the breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Top 10 Return Reasons")
        if sorted_reasons:
            for i, (reason, count) in enumerate(sorted_reasons[:10]):
                percentage = (count / len(df)) * 100
                
                # Determine color based on reason type
                if 'defective' in reason or 'bad' in reason or 'broken' in reason:
                    color = COLORS['danger']
                elif 'too' in reason or 'wrong' in reason:
                    color = COLORS['warning']
                elif 'no issue' in reason or 'no longer needed' in reason:
                    color = COLORS['success']
                else:
                    color = COLORS['primary']
                
                st.markdown(f"""
                <div style="margin: 0.5rem 0;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span class="reason-badge" style="background: {color}40; border-color: {color}; color: {color};">
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
        quality_keywords = ['defective', 'bad', 'broken', 'missing', 'damaged', 'quality', 'not work']
        quality_count = 0
        quality_reasons = []
        
        for reason, count in st.session_state.reason_summary.items():
            if any(keyword in reason.lower() for keyword in quality_keywords):
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
    """Export categorized data to Excel with specified format"""
    output = io.BytesIO()
    
    # Prepare export data
    export_df = prepare_export_data(df)
    
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
            'A': 15,  # Date
            'B': 30,  # Product Identifier Tag
            'C': 20,  # Imported SKU
            'D': 15,  # UDI
            'E': 15,  # CS Ticket #
            'F': 20,  # Order #
            'G': 20,  # Source
            'H': 25,  # Categorizing Agent
            'I': 50,  # Complaint
            'J': 12,  # Review stars
            'K': 25,  # Return Reason
            'L': 5,   # Blank
            'M': 5,   # Blank
            'N': 20,  # on report
            'O': 25,  # Amazon Reason Code
            'P': 40,  # Meaning
            'Q': 20,  # Odoo Return Category?
            'R': 15   # Vive Reason Code
        }
        
        for col, width in column_widths.items():
            col_idx = ord(col) - ord('A')
            worksheet1.set_column(col_idx, col_idx, width)
        
        # Format summary sheet
        worksheet2 = writer.sheets['Summary']
        worksheet2.set_column('A:A', 30)
        worksheet2.set_column('B:B', 10)
        worksheet2.set_column('C:C', 12)
        
        # Add chart only if we have data
        if len(summary_df) > 0:
            try:
                chart = workbook.add_chart({'type': 'pie'})
                chart.add_series({
                    'categories': ['Summary', 1, 0, min(10, len(summary_df)), 0],
                    'values': ['Summary', 1, 1, min(10, len(summary_df)), 1],
                    'name': 'Top 10 Return Reasons'
                })
                chart.set_title({'name': 'Return Reason Distribution'})
                chart.set_size({'width': 600, 'height': 400})
                worksheet2.insert_chart('E2', chart)
            except Exception as e:
                logger.warning(f"Could not add chart to Excel: {e}")
    
    output.seek(0)
    return output.getvalue()

def main():
    st.set_page_config(
        page_title=APP_CONFIG['title'],
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Check AI availability after page config
    if not AI_AVAILABLE:
        st.error("❌ Critical Error: AI module (enhanced_ai_analysis.py) not found!")
        st.info("Please ensure the enhanced_ai_analysis.py file is in the same directory as this app.")
        st.stop()
    
    initialize_session_state()
    inject_cyberpunk_css()
    
    # Try to load category mapping
    load_category_mapping()
    
    # Header
    st.markdown(f"""
    <h1>{APP_CONFIG['title']}</h1>
    <p style="text-align: center; color: var(--primary); font-size: 1.2em; margin-bottom: 2rem;">
        {APP_CONFIG['description']}
    </p>
    """, unsafe_allow_html=True)
    
    # Instructions
    with st.expander("📖 How to Use This Tool", expanded=False):
        st.markdown("""
        ### Quick Start Guide
        1. **Upload your file**:
           - Product Complaints Ledger Excel file (.xlsx)
           - OR FBA Return Report (.txt tab-separated file)
        2. **AI will categorize** each complaint/comment into standard return reasons
        3. **Download the results** with:
           - Column K: Return Reason (e.g., "too small", "defective seat")
           - Columns L-M: Blank
           - Columns N-R: Mapping data (if available)
        4. **Copy and paste** the categorized data into your final report
        
        ### Supported File Types:
        - **Complaints Ledger**: Excel files (.xlsx) with a "Complaint" column
        - **FBA Return Reports**: Tab-separated .txt files exported from Amazon Seller Central with "customer-comments" and "reason" columns
        - **Coming Soon**: PDF support for Amazon Seller Central Manage Returns page
        
        ### Return Reasons Include:
        - Size issues: too small, too large, wrong size
        - Quality issues: defective parts, bad wheels, bad brakes, etc.
        - Shipping issues: damaged during shipping, wrong item
        - Customer issues: bought by mistake, no longer needed
        - And many more...
        
        ### Output Format
        The tool maintains your original data structure and adds the categorized return reason in column K.
        For FBA returns, it uses both the Amazon reason code and customer comments for better categorization.
        """)
    
    # Check AI availability
    api_client = get_api_client()
    if api_client and api_client.is_available():
        st.success("✅ AI Service Connected - Ready for intelligent categorization")
    else:
        st.warning("⚠️ AI Service Not Available - Will use keyword-based categorization")
    
    # Optional: Upload mapping file if not found
    if st.session_state.category_mapping is None:
        with st.expander("📂 Upload Return Category Mapping File (Optional)", expanded=False):
            mapping_file = st.file_uploader(
                "Upload 'Return Category to Assign return categories.csv'",
                type=['csv'],
                help="This file contains the mapping between return reasons and Vive codes"
            )
            if mapping_file:
                try:
                    st.session_state.category_mapping = pd.read_csv(mapping_file)
                    st.success("✅ Mapping file loaded successfully")
                except Exception as e:
                    st.error(f"Error loading mapping file: {e}")
    
    # File upload section
    st.markdown("""
    <div class="neon-box">
        <h3 style="color: var(--accent);">📁 UPLOAD COMPLAINTS FILE</h3>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose your Product Complaints Ledger Excel file or FBA Return Report",
        type=['xlsx', 'xls', 'csv', 'txt'],
        help="Upload complaints file (.xlsx) or FBA Return Report (.txt)"
    )
    
    if uploaded_file is not None:
        # Process the file
        st.session_state.uploaded_file = uploaded_file
        
        with st.spinner("📖 Reading file..."):
            file_content = uploaded_file.read()
            df = process_complaints_file(file_content, uploaded_file.name)
        
        if df is not None:
            st.session_state.processed_data = df
            
            # Show file info
            st.markdown("### 📋 File Information")
            st.info(f"Found {len(df)} rows with {len(df.columns)} columns")
            
            # Show sample data
            st.markdown("#### Sample Data")
            
            # Show different columns based on file type
            if st.session_state.get('file_type') == 'fba_returns':
                display_cols = ['return-date', 'order-id', 'product-name', 'reason', 'customer-comments']
                available_cols = [col for col in display_cols if col in df.columns]
                # If renamed columns exist, show those
                if 'Date' in df.columns and 'Order #' in df.columns:
                    display_cols = ['Date', 'Order #', 'Product Identifier Tag', 'FBA_Reason_Code', 'Complaint']
                    available_cols = [col for col in display_cols if col in df.columns]
            else:
                display_cols = ['Date', 'Product Identifier Tag', 'Order #', 'Complaint']
                available_cols = [col for col in display_cols if col in df.columns]
            
            if available_cols:
                st.dataframe(df[available_cols].head(5), use_container_width=True)
            
            # Categorize button
            if st.button("🚀 CATEGORIZE COMPLAINTS", type="primary", use_container_width=True):
                with st.spinner("🤖 AI is analyzing and categorizing complaints..."):
                    categorized_df = categorize_all_complaints(df)
                    st.session_state.categorized_data = categorized_df
                    st.session_state.processing_complete = True
                
                st.balloons()
                st.success("✅ Categorization complete!")
            
            # Show results if processing is complete
            if st.session_state.processing_complete and st.session_state.categorized_data is not None:
                display_results_summary(st.session_state.categorized_data)
                
                # Export section
                st.markdown("---")
                st.markdown("""
                <div class="success-box">
                    <h3 style="color: var(--success);">📥 EXPORT RESULTS</h3>
                    <p>Your categorized data is ready for download!</p>
                    <p>Column K contains the return reasons. Columns L-M are blank as requested.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Generate export file
                excel_data = export_categorized_data(st.session_state.categorized_data)
                
                # Download button
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Determine file type
                if EXCEL_AVAILABLE or OPENPYXL_AVAILABLE:
                    file_extension = "xlsx"
                    mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                else:
                    file_extension = "csv"
                    mime_type = "text/csv"
                
                st.download_button(
                    label=f"📥 DOWNLOAD CATEGORIZED DATA",
                    data=excel_data,
                    file_name=f"categorized_complaints_{timestamp}.{file_extension}",
                    mime=mime_type,
                    use_container_width=True
                )
                
                # Show sample of categorized data
                st.markdown("### 🔍 Categorized Data Preview")
                
                # Show key columns
                preview_cols = ['Product Identifier Tag', 'Complaint', 'Return_Reason']
                available_preview = [col for col in preview_cols if col in st.session_state.categorized_data.columns]
                
                if available_preview:
                    preview_df = st.session_state.categorized_data[available_preview].head(10)
                    
                    # Rename for display
                    preview_df = preview_df.rename(columns={'Return_Reason': 'Return Reason (Column K)'})
                    st.dataframe(preview_df, use_container_width=True)
                
                # Quality team action items
                st.markdown("---")
                st.markdown("""
                <div class="neon-box">
                    <h3 style="color: var(--primary);">💡 QUALITY TEAM ACTION ITEMS</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Generate actionable insights
                quality_keywords = ['defective', 'bad', 'broken', 'missing', 'damaged', 'quality']
                quality_reasons = [(reason, count) for reason, count in st.session_state.reason_summary.items() 
                                 if any(keyword in reason.lower() for keyword in quality_keywords)]
                quality_count = sum(count for _, count in quality_reasons)
                
                # Get top reason
                if st.session_state.reason_summary:
                    top_reason = max(st.session_state.reason_summary.items(), key=lambda x: x[1])[0]
                else:
                    top_reason = 'None'
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    **Key Findings:**
                    - 🎯 {quality_count} quality-related complaints
                    - 📊 Top reason: {top_reason}
                    - 🔍 {len(quality_reasons)} different quality issues identified
                    - 📂 File type: {st.session_state.get('file_type', 'Unknown').replace('_', ' ').title()}
                    """)
                
                with col2:
                    st.markdown("""
                    **Recommended Actions:**
                    1. Review all quality-related complaints
                    2. Create CAPA for top 3 issues
                    3. Update inspection criteria
                    4. Share findings with suppliers
                    5. Monitor trends after corrections
                    
                    **💡 Tip:** Cross-reference FBA returns with product reviews and complaint ledger data for comprehensive quality insights.
                    """)

if __name__ == "__main__":
    main()
