"""
Vive Health Quality Complaint Categorizer - Restored Working Version
AI-Powered Return Reason Classification Tool
Version: 11.2 - Back to Working State with Dual AI

Key Features:
- Requires AI to function (OpenAI + Claude from Streamlit secrets)
- Manual categorization only offered as user choice when AI fails
- Restored to working state before recent changes
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
    from enhanced_ai_analysis import EnhancedAIAnalyzer, AIProvider
    AI_AVAILABLE = True
except ImportError as e:
    AI_AVAILABLE = False
    logger.error(f"AI module not available: {str(e)}")

try:
    import xlsxwriter
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False

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
    'version': '11.2',
    'company': 'Vive Health'
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
    'muted': '#666680'
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
    
    .results-header {{
        background: rgba(0, 245, 160, 0.1);
        border: 2px solid var(--success);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 2rem 0;
    }}
    
    .ai-status-box {{
        background: rgba(0, 217, 255, 0.1);
        border: 2px solid var(--primary);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
    }}
    
    .error-box {{
        background: rgba(255, 0, 84, 0.1);
        border: 2px solid var(--danger);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }}
    
    .manual-override {{
        background: rgba(255, 183, 0, 0.1);
        border: 2px solid var(--accent);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }}
    
    .stMetric {{
        background: rgba(26, 26, 46, 0.6);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(0, 217, 255, 0.3);
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
        'reason_summary': {},
        'product_summary': {},
        'date_filter_enabled': False,
        'date_range_start': None,
        'date_range_end': None,
        'severity_counts': {'critical': 0, 'major': 0, 'minor': 0},
        'quality_insights': None,
        'ai_failed': False,
        'manual_mode': False,
        'ai_provider': 'both'  # Default to both APIs
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
    """Get or create AI analyzer with dual API support"""
    if st.session_state.ai_analyzer is None and AI_AVAILABLE:
        try:
            # Set up environment variables from secrets
            keys = check_api_keys()
            
            if 'openai' in keys:
                os.environ['OPENAI_API_KEY'] = keys['openai']
            if 'claude' in keys:
                os.environ['ANTHROPIC_API_KEY'] = keys['claude']
            
            # Create analyzer with preferred provider
            if st.session_state.ai_provider == 'both':
                provider = AIProvider.BOTH
            elif st.session_state.ai_provider == 'openai':
                provider = AIProvider.OPENAI
            else:
                provider = AIProvider.CLAUDE
            
            st.session_state.ai_analyzer = EnhancedAIAnalyzer(provider)
            logger.info(f"Created AI analyzer with provider: {provider}")
        except Exception as e:
            logger.error(f"Error creating AI analyzer: {e}")
            st.error(f"Error initializing AI: {str(e)}")
    
    return st.session_state.ai_analyzer

def display_required_format():
    """Display the required file format"""
    st.markdown("""
    <div class="format-box">
        <h4 style="color: var(--primary);">📋 Required File Format</h4>
        <p>Your file must contain these columns:</p>
        <ul>
            <li><strong>Complaint</strong> - The return reason/comment text (Required)</li>
            <li><strong>Product Identifier Tag</strong> - Product name/SKU (Recommended)</li>
            <li><strong>Order #</strong> - Order number (Optional)</li>
            <li><strong>Date</strong> - Return date (Optional)</li>
            <li><strong>Source</strong> - Where the complaint came from (Optional)</li>
        </ul>
        <p style="color: var(--accent); margin-top: 1rem;">
            💡 The tool will add the Category column with AI classification
        </p>
    </div>
    """, unsafe_allow_html=True)

def display_ai_status():
    """Display AI status and configuration"""
    if not AI_AVAILABLE:
        st.markdown("""
        <div class="error-box">
            <h3 style="color: var(--danger); margin: 0;">❌ AI Module Not Available</h3>
            <p>The enhanced_ai_analysis.py module is required for this tool to function.</p>
            <p>Please ensure the AI module is in the same directory as this app.</p>
        </div>
        """, unsafe_allow_html=True)
        return False
    
    # Check API keys
    keys = check_api_keys()
    
    st.markdown("""
    <div class="ai-status-box">
        <h3 style="color: var(--primary); margin-top: 0;">🤖 AI Configuration</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Display API status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'openai' in keys:
            st.success("✅ OpenAI API")
            st.caption("GPT-4 Ready")
        else:
            st.error("❌ OpenAI API")
            st.caption("Key not found")
    
    with col2:
        if 'claude' in keys:
            st.success("✅ Claude API")
            st.caption("Sonnet Ready")
        else:
            st.error("❌ Claude API")
            st.caption("Key not found")
    
    with col3:
        # AI Provider selection
        provider_options = {
            'both': '🔷 Both (Best)',
            'openai': '🟢 OpenAI Only',
            'claude': '🟠 Claude Only'
        }
        
        st.session_state.ai_provider = st.selectbox(
            "AI Provider",
            options=list(provider_options.keys()),
            format_func=lambda x: provider_options[x],
            index=0
        )
    
    # Check if we have at least one API key
    if not keys:
        st.markdown("""
        <div class="error-box">
            <h4 style="color: var(--danger); margin-top: 0;">⚠️ No API Keys Found</h4>
            <p>Please add your API keys to Streamlit secrets:</p>
            <ul>
                <li><strong>openai_api_key</strong>: Your OpenAI API key</li>
                <li><strong>anthropic_api_key</strong>: Your Claude API key</li>
            </ul>
            <p>You need at least one API key for the tool to work.</p>
        </div>
        """, unsafe_allow_html=True)
        return False
    
    # Check if selected provider is available
    if st.session_state.ai_provider == 'both' and len(keys) < 2:
        st.warning("⚠️ Both APIs selected but only one key available. Will use available API.")
    elif st.session_state.ai_provider == 'openai' and 'openai' not in keys:
        st.error("❌ OpenAI selected but API key not found.")
        return False
    elif st.session_state.ai_provider == 'claude' and 'claude' not in keys:
        st.error("❌ Claude selected but API key not found.")
        return False
    
    return True

def process_complaints_file(file_content, filename: str, date_filter=None) -> pd.DataFrame:
    """Process complaints file with date filtering"""
    try:
        # Read file
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
    """Categorize all complaints using AI"""
    
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
    
    total_rows = len(df)
    category_counts = Counter()
    product_issues = defaultdict(lambda: defaultdict(int))
    severity_counts = Counter()
    successful_categorizations = 0
    failed_categorizations = 0
    
    for idx, row in df.iterrows():
        complaint = str(row['Complaint']).strip() if pd.notna(row['Complaint']) else ""
        
        if not complaint:
            continue
        
        # Get FBA reason if available
        fba_reason = str(row.get('reason', '')) if pd.notna(row.get('reason')) else ""
        
        try:
            # Categorize using AI
            if hasattr(analyzer, 'categorize_return'):
                category, confidence, severity, language = analyzer.categorize_return(complaint, fba_reason)
            else:
                # Fallback to basic API call
                result = analyzer.api_client.categorize_return(complaint, fba_reason=fba_reason)
                if isinstance(result, dict) and 'openai' in result:
                    category = result['openai']
                else:
                    category = result
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
                    
        except Exception as e:
            logger.error(f"AI categorization failed for row {idx}: {e}")
            df.at[idx, 'Category'] = 'Other/Miscellaneous'
            failed_categorizations += 1
        
        # Update progress
        progress = (idx + 1) / total_rows
        progress_bar.progress(progress)
        status_text.text(f"🤖 AI Processing: {idx + 1}/{total_rows} | Success: {successful_categorizations} | Failed: {failed_categorizations}")
    
    # Calculate success rate
    success_rate = (successful_categorizations / total_rows * 100) if total_rows > 0 else 0
    
    if failed_categorizations > total_rows * 0.5:  # More than 50% failed
        st.session_state.ai_failed = True
        raise Exception(f"Too many AI failures: {failed_categorizations}/{total_rows}")
    
    status_text.text(f"✅ AI Complete! Categorized: {successful_categorizations}/{total_rows} ({success_rate:.1f}% success)")
    
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
    except:
        st.session_state.quality_insights = None
    
    return df

def categorize_manually(df: pd.DataFrame) -> pd.DataFrame:
    """Manual categorization option when AI fails"""
    
    st.warning("🔧 **Manual Categorization Mode**")
    st.info("AI categorization failed. You can manually categorize each complaint or assign default categories.")
    
    # Option 1: Assign all to "Other/Miscellaneous"
    if st.button("📝 Set All to 'Other/Miscellaneous' (Quick Option)", use_container_width=True):
        df['Category'] = 'Other/Miscellaneous'
        
        # Basic summaries
        st.session_state.reason_summary = {'Other/Miscellaneous': len(df)}
        st.session_state.product_summary = {}
        st.session_state.severity_counts = {'none': len(df)}
        st.session_state.quality_insights = None
        
        st.success("✅ All complaints categorized as 'Other/Miscellaneous'")
        return df
    
    # Option 2: Manual categorization interface
    st.markdown("---")
    st.markdown("### 🔧 Manual Categorization Interface")
    
    # Show first few complaints for manual categorization
    st.info("Categorize the first 10 complaints manually, then apply patterns to the rest:")
    
    manual_categories = {}
    
    for idx in range(min(10, len(df))):
        row = df.iloc[idx]
        complaint = str(row['Complaint'])[:100] + "..."
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.text(complaint)
        with col2:
            category = st.selectbox(
                "Category",
                options=MEDICAL_DEVICE_CATEGORIES,
                key=f"manual_cat_{idx}",
                index=14  # Default to "Other/Miscellaneous"
            )
            manual_categories[idx] = category
    
    if st.button("📊 Apply Manual Categories", use_container_width=True):
        # Apply manual categories to first 10
        for idx, category in manual_categories.items():
            df.at[idx, 'Category'] = category
        
        # Set remaining to "Other/Miscellaneous"
        for idx in range(10, len(df)):
            df.at[idx, 'Category'] = 'Other/Miscellaneous'
        
        # Generate basic summaries
        category_counts = df['Category'].value_counts().to_dict()
        st.session_state.reason_summary = category_counts
        st.session_state.product_summary = {}
        st.session_state.severity_counts = {'none': len(df)}
        st.session_state.quality_insights = None
        
        st.success(f"✅ Manual categorization complete! {len(manual_categories)} manual, {len(df)-10} auto-assigned.")
        return df
    
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
    """Display product-specific analysis"""
    
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
        
        # Display as dataframe
        if product_data:
            product_df = pd.DataFrame(product_data[:20])
            st.dataframe(product_df, use_container_width=True, hide_index=True)
            
            if len(st.session_state.product_summary) > 20:
                st.caption(f"Showing top 20 of {len(st.session_state.product_summary)} products.")
    else:
        st.info("No product information available. Ensure your file has a 'Product Identifier Tag' column.")

def display_results(df: pd.DataFrame):
    """Display categorization results"""
    
    st.markdown("""
    <div class="results-header">
        <h2 style="color: var(--primary); text-align: center;">📊 CATEGORIZATION RESULTS</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_categorized = len(df[df['Category'].notna() & (df['Category'] != '')])
    
    with col1:
        st.metric("Total Returns", len(df))
    
    with col2:
        quality_count = sum(
            count for cat, count in st.session_state.reason_summary.items()
            if cat in QUALITY_CATEGORIES
        )
        quality_pct = (quality_count / total_categorized * 100) if total_categorized > 0 else 0
        st.metric("Quality Issues", f"{quality_pct:.1f}%")
    
    with col3:
        severity_critical = st.session_state.severity_counts.get('critical', 0)
        st.metric("🚨 Critical", severity_critical)
    
    with col4:
        unique_products = df['Product Identifier Tag'].nunique() if 'Product Identifier Tag' in df.columns else 0
        st.metric("Products", unique_products)
    
    # Create tabs
    if st.session_state.quality_insights:
        tab_list = st.tabs(["📈 Categories", "🔍 Quality Insights", "📦 Products"])
    else:
        tab_list = st.tabs(["📈 Categories", "📦 Products"])
    
    # Category distribution tab
    with tab_list[0]:
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
            # Quality vs Non-Quality breakdown
            st.markdown("#### Quality vs Other Returns")
            quality_returns = sum(count for cat, count in st.session_state.reason_summary.items() 
                                if cat in QUALITY_CATEGORIES)
            other_returns = total_categorized - quality_returns
            
            st.markdown(f"""
            <div style="background: rgba(255,0,84,0.1); padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
                <h4 style="color: var(--danger); margin: 0;">Quality Issues: {quality_returns}</h4>
                <p style="margin: 0.5rem 0 0 0;">{quality_returns/total_categorized*100:.1f}% of all returns</p>
            </div>
            <div style="background: rgba(0,217,255,0.1); padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
                <h4 style="color: var(--primary); margin: 0;">Other Returns: {other_returns}</h4>
                <p style="margin: 0.5rem 0 0 0;">{other_returns/total_categorized*100:.1f}% of all returns</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Quality insights tab (if available)
    if st.session_state.quality_insights and len(tab_list) > 2:
        with tab_list[1]:
            insights = st.session_state.quality_insights
            
            # Risk Assessment
            risk_level = insights['risk_assessment']['overall_risk_level']
            risk_color = {'HIGH': 'danger', 'MEDIUM': 'warning', 'LOW': 'success'}[risk_level]
            
            st.markdown(f"""
            <div style="background: rgba(255,0,84,0.1); border: 2px solid var(--{risk_color}); 
                      border-radius: 10px; padding: 1.5rem; margin-bottom: 1rem;">
                <h3 style="color: var(--{risk_color}); margin: 0;">⚠️ Quality Risk Level: {risk_level}</h3>
                <p style="margin: 0.5rem 0;">Quality Issue Rate: {insights['risk_assessment']['quality_rate']:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Action Items
            if insights['action_items']:
                st.markdown("### 🎯 Recommended Quality Actions")
                
                for action in insights['action_items']:
                    severity_color = 'danger' if action['severity'] == 'HIGH' else 'warning'
                    
                    st.markdown(f"""
                    <div style="background: rgba(255,107,53,0.1); border-left: 4px solid var(--{severity_color}); 
                              padding: 1rem; margin: 0.5rem 0; border-radius: 5px;">
                        <h4 style="color: var(--{severity_color}); margin: 0;">
                            {action['severity']} Priority: {action['issue']} ({action['frequency']} cases)
                        </h4>
                        <p style="margin: 0.5rem 0;"><strong>Action:</strong> {action['recommendation']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # High Risk Products
            if insights['risk_assessment']['top_risk_products']:
                st.markdown("### 🚨 High Risk Products")
                
                risk_data = []
                for prod in insights['risk_assessment']['top_risk_products'][:10]:
                    risk_data.append({
                        'Product': prod['product'][:50] + "..." if len(prod['product']) > 50 else prod['product'],
                        'Total Issues': prod['total_issues'],
                        'Quality Issues': prod['quality_issues'],
                        'Primary Issue': prod['primary_root_cause']
                    })
                
                if risk_data:
                    risk_df = pd.DataFrame(risk_data)
                    st.dataframe(risk_df, use_container_width=True, hide_index=True)
    
    # Products tab
    products_tab_index = 2 if len(tab_list) > 2 else 1
    with tab_list[products_tab_index]:
        display_product_analysis(df)
    
    # Show uncategorized items
    other_items = df[df['Category'] == 'Other/Miscellaneous']
    if len(other_items) > 0:
        st.markdown("---")
        st.markdown(f"### ⚠️ Review {len(other_items)} Uncategorized Items")
        
        with st.expander("View uncategorized complaints"):
            available_columns = ['Complaint']
            if 'Product Identifier Tag' in df.columns:
                available_columns.append('Product Identifier Tag')
            if 'Order #' in df.columns:
                available_columns.append('Order #')
            
            st.dataframe(
                other_items[available_columns].head(20),
                use_container_width=True
            )
            st.caption("Showing first 20 items. Download full export to see all.")

def export_data(df: pd.DataFrame) -> bytes:
    """Export data with quality insights"""
    
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
            
            # Format workbook
            workbook = writer.book
            worksheet = writer.sheets['Categorized Returns']
            
            # Auto-adjust columns
            for i, col in enumerate(df.columns):
                max_len = max(
                    len(str(col)) + 2,
                    df[col].astype(str).str.len().max() + 2 if len(df) > 0 else 10
                )
                max_len = min(max_len, 50)
                worksheet.set_column(i, i, max_len)
            
            # Highlight quality categories
            if 'Category' in df.columns:
                cat_col_idx = df.columns.get_loc('Category')
                quality_format = workbook.add_format({
                    'bg_color': '#FFE6E6',
                    'font_color': '#CC0000'
                })
                
                for quality_cat in QUALITY_CATEGORIES:
                    worksheet.conditional_format(1, cat_col_idx, len(df), cat_col_idx, {
                        'type': 'cell',
                        'criteria': '==',
                        'value': f'"{quality_cat}"',
                        'format': quality_format
                    })
    else:
        # CSV fallback
        df.to_csv(output, index=False)
    
    output.seek(0)
    return output.getvalue()

def generate_quality_report(df: pd.DataFrame) -> str:
    """Generate quality analysis report"""
    
    total_returns = len(df)
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
AI Processing: {'Success' if not st.session_state.ai_failed else 'Failed (Manual mode used)'}
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
RETURN CATEGORIES
=================
"""
    
    for cat, count in sorted(st.session_state.reason_summary.items(), 
                           key=lambda x: x[1], reverse=True):
        pct = (count / total_returns * 100) if total_returns > 0 else 0
        quality_flag = " [QUALITY]" if cat in QUALITY_CATEGORIES else ""
        report += f"{cat}{quality_flag}: {count} ({pct:.1f}%)\n"
    
    report += f"""
RECOMMENDATIONS
==============
1. Focus quality improvements on top return categories
2. Address high-priority action items immediately  
3. Review products with highest return rates
4. Implement corrective actions for recurring quality issues
5. Monitor improvement metrics after implementing changes
"""
    
    return report

def main():
    """Main application function"""
    
    initialize_session_state()
    inject_custom_css()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">VIVE HEALTH RETURN CATEGORIZER</h1>
        <p style="font-size: 1.2em; color: var(--text); margin: 0.5rem 0;">
            AI-Powered Medical Device Quality Management Tool
        </p>
        <p style="font-size: 1em; color: var(--accent);">
            🤖 Dual AI Support: OpenAI + Claude from Streamlit Secrets
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check AI status first
    if not display_ai_status():
        st.stop()
    
    # Sidebar for settings
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        # Date filtering
        st.markdown("#### 📅 Date Filtering")
        enable_date_filter = st.checkbox("Enable date filter")
        st.session_state.date_filter_enabled = enable_date_filter
        
        if enable_date_filter:
            start_date = st.date_input("Start date", 
                                      value=datetime.now() - timedelta(days=30))
            st.session_state.date_range_start = start_date
            
            end_date = st.date_input("End date", 
                                    value=datetime.now())
            st.session_state.date_range_end = end_date
        
        # AI Status summary
        st.markdown("---")
        st.markdown("#### 🤖 Current Status")
        
        keys = check_api_keys()
        if 'openai' in keys:
            st.success("✅ OpenAI Ready")
        if 'claude' in keys:
            st.success("✅ Claude Ready")
        
        st.info(f"Provider: {st.session_state.ai_provider.title()}")
    
    # Main content
    with st.expander("📋 Required File Format & Instructions", expanded=False):
        display_required_format()
        
        st.markdown("### 🤖 AI-Powered Categorization:")
        st.info("""
        This tool uses advanced AI (OpenAI GPT-4 and/or Claude) to analyze each return complaint:
        - Understands context and nuance
        - Medical device specific terminology
        - Handles complex multi-issue returns
        - Dual AI consensus for maximum accuracy
        """)
        
        st.markdown("### 🔍 Quality Insights:")
        st.success("""
        **Automatic Quality Analysis:**
        - Risk assessment by category and product
        - Prioritized action recommendations  
        - Product-specific quality metrics
        - Export with quality highlighting
        """)
    
    # File upload section
    st.markdown("---")
    st.markdown("### 📁 Upload Files")
    
    if st.session_state.date_filter_enabled:
        st.info(f"📅 Date filter: {st.session_state.date_range_start} to {st.session_state.date_range_end}")
    
    uploaded_files = st.file_uploader(
        "Choose file(s) to categorize",
        type=['xlsx', 'xls', 'csv'],
        accept_multiple_files=True,
        help="Upload files with a 'Complaint' column"
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
                
                df = process_complaints_file(file_content, filename, date_filter)
                
                if df is not None and not df.empty:
                    all_data.append(df)
                    st.success(f"✅ Loaded: {filename} ({len(df)} rows)")
        
        if all_data:
            # Combine all data
            combined_df = pd.concat(all_data, ignore_index=True) if len(all_data) > 1 else all_data[0]
            st.session_state.processed_data = combined_df
            
            # Show summary
            st.success(f"📊 **Total records ready: {len(combined_df)}**")
            
            # Preview
            if st.checkbox("Preview data"):
                st.dataframe(combined_df.head(10))
            
            # Categorize button
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                if st.button(
                    f"🚀 CATEGORIZE {len(combined_df)} RETURNS WITH AI", 
                    type="primary", 
                    use_container_width=True
                ):
                    start_time = time.time()
                    st.session_state.ai_failed = False
                    st.session_state.manual_mode = False
                    
                    st.info(f"🤖 Using {st.session_state.ai_provider.upper()} AI for categorization...")
                    
                    try:
                        with st.spinner(f"🤖 AI Processing {len(combined_df)} returns..."):
                            categorized_df = categorize_all_data_ai(combined_df)
                            st.session_state.categorized_data = categorized_df
                            st.session_state.processing_complete = True
                        
                        # Show completion time
                        elapsed_time = time.time() - start_time
                        st.success(f"✅ AI categorization complete in {elapsed_time:.1f} seconds!")
                        
                    except Exception as e:
                        st.error(f"❌ AI categorization failed: {str(e)}")
                        st.session_state.ai_failed = True
                        
                        # Offer manual categorization
                        st.markdown("""
                        <div class="manual-override">
                            <h3 style="color: var(--accent); margin-top: 0;">🔧 Manual Override Available</h3>
                            <p>AI categorization failed. Would you like to categorize manually?</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if st.button("🔧 Switch to Manual Categorization", type="secondary"):
                            st.session_state.manual_mode = True
                            st.rerun()
            
            # Manual categorization if AI failed and user chose manual mode
            if st.session_state.ai_failed and st.session_state.manual_mode:
                try:
                    categorized_df = categorize_manually(combined_df)
                    st.session_state.categorized_data = categorized_df
                    st.session_state.processing_complete = True
                except Exception as e:
                    st.error(f"Manual categorization failed: {str(e)}")
            
            # Show results
            if st.session_state.processing_complete and st.session_state.categorized_data is not None:
                
                display_results(st.session_state.categorized_data)
                
                # Export section
                st.markdown("---")
                completion_msg = "✅ AI ANALYSIS COMPLETE!" if not st.session_state.ai_failed else "✅ MANUAL CATEGORIZATION COMPLETE!"
                
                st.markdown(f"""
                <div style="background: rgba(0, 245, 160, 0.1); border: 2px solid var(--success); 
                          border-radius: 15px; padding: 2rem; text-align: center;">
                    <h3 style="color: var(--success);">{completion_msg}</h3>
                    <p>Your data has been categorized and analyzed for quality insights.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Export options
                col1, col2, col3 = st.columns(3)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                with col1:
                    excel_data = export_data(st.session_state.categorized_data)
                    
                    st.download_button(
                        label="📥 DOWNLOAD EXCEL",
                        data=excel_data,
                        file_name=f"categorized_returns_{timestamp}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                        help="Complete analysis with quality insights"
                    )
                
                with col2:
                    # CSV export
                    csv_data = st.session_state.categorized_data.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 DOWNLOAD CSV",
                        data=csv_data,
                        file_name=f"categorized_returns_{timestamp}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col3:
                    # Quality report
                    quality_report = generate_quality_report(st.session_state.categorized_data)
                    st.download_button(
                        label="📥 QUALITY REPORT",
                        data=quality_report,
                        file_name=f"quality_analysis_{timestamp}.txt",
                        mime="text/plain",
                        use_container_width=True,
                        help="Detailed quality analysis with recommendations"
                    )
                
                # Show export info
                method_used = "AI-powered" if not st.session_state.ai_failed else "Manual"
                
                st.info(f"""
                **📋 Export Contents ({method_used} categorization):**
                - ✅ Original data with categorization
                - ✅ Category summary with quality flagging
                - ✅ Quality insights and action items
                - ✅ Product-specific analysis
                - ✅ Quality categories highlighted in Excel
                
                **🤖 Processing Details:**
                - Method: {method_used}
                - Provider: {st.session_state.ai_provider.upper() if not st.session_state.ai_failed else 'Manual'}
                - Quality analysis: {'Enabled' if st.session_state.quality_insights else 'Basic'}
                """)

if __name__ == "__main__":
    main()
