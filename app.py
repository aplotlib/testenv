"""
Vive Health Quality Complaint Categorizer - Fixed Version
AI-Powered Return Reason Classification Tool
Version: 11.0 - Simplified and Fixed

Key Fixes:
- Removed duplicate detection
- Fixed export format (Category in column K only)
- Fixed API client access issue
- Simplified categorization flow
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
    from enhanced_ai_analysis import (
        EnhancedAIAnalyzer, AIProvider,
        MEDICAL_DEVICE_CATEGORIES, FBA_REASON_MAP,
        generate_quality_insights
    )
    AI_AVAILABLE = True
    api_error_message = None
except ImportError as e:
    AI_AVAILABLE = False
    api_error_message = f"AI module not available: {str(e)}"
    logger.error(api_error_message)
    # Fallback categories
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

try:
    import xlsxwriter
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False

# App Configuration
APP_CONFIG = {
    'title': 'Vive Health Medical Device Return Categorizer',
    'version': '11.0',
    'company': 'Vive Health',
    'description': 'AI-Powered Quality Management Tool'
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

# Quality-related categories for highlighting
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
    
    .quality-action {{
        background: rgba(255, 107, 53, 0.1);
        border-left: 4px solid var(--warning);
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
        transition: all 0.3s ease;
    }}
    
    .quality-action:hover {{
        transform: translateX(5px);
        box-shadow: 0 4px 15px rgba(255, 107, 53, 0.3);
    }}
    
    .risk-assessment {{
        background: rgba(255, 0, 84, 0.1);
        border: 2px solid var(--danger);
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }}
    
    .pattern-box {{
        background: rgba(0, 217, 255, 0.05);
        border: 1px solid var(--primary);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }}
    
    .stMetric {{
        background: rgba(26, 26, 46, 0.6);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(0, 217, 255, 0.3);
    }}
    
    .stTabs [data-baseweb="tab-list"] {{
        background-color: rgba(26, 26, 46, 0.6);
        border-radius: 10px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        color: var(--text);
        font-weight: 600;
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: var(--primary) !important;
        color: var(--dark) !important;
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
        'quality_insights': None  # New: Store quality pattern insights
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def get_ai_analyzer():
    """Get or create AI analyzer"""
    if st.session_state.ai_analyzer is None and AI_AVAILABLE:
        try:
            st.session_state.ai_analyzer = EnhancedAIAnalyzer()
            logger.info("Created AI analyzer")
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
            💡 The tool will add the Category column (Column K) with AI classification
        </p>
    </div>
    """, unsafe_allow_html=True)

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
        
        # Add Category column if not present (Column K position)
        if 'Category' not in df.columns:
            # Insert at position K (10th column, 0-indexed)
            cols = df.columns.tolist()
            if len(cols) >= 10:
                cols.insert(10, 'Category')
                df = df.reindex(columns=cols)
                df['Category'] = ''
            else:
                df['Category'] = ''
        
        return df
            
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        st.error(f"Error processing file: {str(e)}")
        return None

def categorize_all_data(df: pd.DataFrame) -> pd.DataFrame:
    """Categorize all complaints using AI"""
    
    analyzer = get_ai_analyzer()
    
    if not analyzer:
        st.error("AI analyzer not initialized. Please check API key configuration.")
        return df
    
    # Check API status
    api_status = analyzer.get_api_status()
    if not api_status['available']:
        st.error("❌ AI not available. Please configure your OpenAI API key.")
        st.info("Add your OpenAI API key to Streamlit secrets or environment variables.")
        return df
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_rows = len(df)
    category_counts = Counter()
    product_issues = defaultdict(lambda: defaultdict(int))
    severity_counts = Counter()
    successful_categorizations = 0
    
    for idx, row in df.iterrows():
        complaint = str(row['Complaint']).strip() if pd.notna(row['Complaint']) else ""
        
        if not complaint:
            continue
        
        # Get FBA reason if available
        fba_reason = str(row.get('reason', '')) if pd.notna(row.get('reason')) else ""
        
        # Categorize using AI
        category, confidence, severity, language = analyzer.categorize_return(complaint, fba_reason)
        
        # Update dataframe - only set Category
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
        
        # Update progress
        progress = (idx + 1) / total_rows
        progress_bar.progress(progress)
        status_text.text(f"🤖 AI Processing: {idx + 1}/{total_rows} | Categorized: {successful_categorizations}")
    
    # Calculate success rate
    success_rate = (successful_categorizations / total_rows * 100) if total_rows > 0 else 0
    status_text.text(f"✅ Complete! AI categorized {successful_categorizations}/{total_rows} returns ({success_rate:.1f}% specific categories)")
    
    # Show warning if many defaults
    other_count = category_counts.get('Other/Miscellaneous', 0)
    if other_count > total_rows * 0.2:  # More than 20% uncategorized
        st.warning(f"""
        ⚠️ {other_count} returns ({other_count/total_rows*100:.1f}%) were categorized as 'Other/Miscellaneous'.
        
        This might indicate:
        - Very ambiguous complaint text
        - Non-standard return reasons
        - API issues (check logs)
        
        Consider reviewing these manually.
        """)
    
    # Store summaries
    st.session_state.reason_summary = dict(category_counts)
    st.session_state.product_summary = dict(product_issues)
    st.session_state.severity_counts = dict(severity_counts)
    
    # Generate quality insights
    if AI_AVAILABLE and 'generate_quality_insights' in globals():
        try:
            st.session_state.quality_insights = generate_quality_insights(
                df, 
                st.session_state.reason_summary,
                st.session_state.product_summary
            )
            logger.info("Generated quality insights")
        except Exception as e:
            logger.error(f"Error generating quality insights: {e}")
            st.session_state.quality_insights = None
    
    return df

def display_results(df: pd.DataFrame):
    """Display categorization results with quality insights"""
    
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
    
    # Create tabs including quality insights
    tabs = ["📈 Categories", "🔍 Quality Patterns", "📦 Products"]
    if st.session_state.quality_insights:
        tab_list = st.tabs(tabs)
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
            
            # Show items needing review if many "Other/Miscellaneous"
            other_items = df[df['Category'] == 'Other/Miscellaneous']
            if len(other_items) > 0:
                with st.expander(f"⚠️ Review {len(other_items)} uncategorized items"):
                    columns_to_show = ['Complaint']
                    if 'Product Identifier Tag' in df.columns:
                        columns_to_show.append('Product Identifier Tag')
                    if 'Order #' in df.columns:
                        columns_to_show.append('Order #')
                    
                    st.dataframe(
                        other_items[columns_to_show].head(20),
                        use_container_width=True
                    )
                    st.caption("Showing first 20 items. Download full export to see all.")
    
    # Quality Insights tab (if available)
    if st.session_state.quality_insights:
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
                <p style="margin: 0;">Safety Critical Issues: {insights['risk_assessment']['safety_critical_count']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Root Cause Analysis
            st.markdown("### 🔍 Root Cause Patterns Detected")
            
            if insights['root_cause_distribution']:
                for root_cause, data in sorted(insights['root_cause_distribution'].items(), 
                                             key=lambda x: x[1]['count'], reverse=True):
                    with st.expander(f"{root_cause} - {data['count']} occurrences"):
                        products_list = data['products']  # Already converted to list in generate_quality_insights
                        st.markdown(f"**Affected Products:** {', '.join(products_list[:5])}")
                        if len(products_list) > 5:
                            st.caption(f"...and {len(products_list) - 5} more")
                        
                        st.markdown("**Example Complaints:**")
                        for example in data['examples']:
                            st.markdown(f"- {example}...")
            
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
                        <p style="margin: 0; font-size: 0.9em;"><strong>Products:</strong> {', '.join(action['affected_products'][:3])}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # High Risk Products
            if insights['risk_assessment']['top_risk_products']:
                st.markdown("### 🚨 High Risk Products")
                
                try:
                    risk_df = pd.DataFrame(insights['risk_assessment']['top_risk_products'])
                    if not risk_df.empty:
                        # Display as a formatted table
                        display_columns = []
                        rename_dict = {}
                        
                        if 'product' in risk_df.columns:
                            display_columns.append('product')
                            rename_dict['product'] = 'Product'
                        if 'total_issues' in risk_df.columns:
                            display_columns.append('total_issues')
                            rename_dict['total_issues'] = 'Total Quality Issues'
                        if 'safety_issues' in risk_df.columns:
                            display_columns.append('safety_issues')
                            rename_dict['safety_issues'] = 'Safety Critical'
                        if 'primary_root_cause' in risk_df.columns:
                            display_columns.append('primary_root_cause')
                            rename_dict['primary_root_cause'] = 'Primary Root Cause'
                        
                        if display_columns:
                            st.dataframe(
                                risk_df[display_columns].rename(columns=rename_dict),
                                use_container_width=True,
                                hide_index=True
                            )
                except Exception as e:
                    logger.error(f"Error displaying risk products: {e}")
                    st.warning("Unable to display risk products table")
        
        # Products tab
        with tab_list[2]:
            display_product_analysis(df)
    else:
        # Products tab (when no quality insights)
        with tab_list[1]:
            display_product_analysis(df)
    
    # Show items needing review if many "Other/Miscellaneous"
    other_items = df[df['Category'] == 'Other/Miscellaneous']
    if len(other_items) > 0:
        with st.expander(f"⚠️ Review {len(other_items)} uncategorized items"):
            columns_to_show = ['Complaint']
            if 'Product Identifier Tag' in df.columns:
                columns_to_show.append('Product Identifier Tag')
            if 'Order #' in df.columns:
                columns_to_show.append('Order #')
            
            st.dataframe(
                other_items[columns_to_show].head(20),
                use_container_width=True
            )
            st.caption("Showing first 20 items. Download full export to see all.")
    
    # Show items needing review if many "Other/Miscellaneous"
    other_items = df[df['Category'] == 'Other/Miscellaneous']
    if len(other_items) > 0:
        with st.expander(f"⚠️ Review {len(other_items)} uncategorized items"):
            st.dataframe(
                other_items[['Complaint', 'Product Identifier Tag', 'Order #']].head(20),
                use_container_width=True
            )
                        # Show export info
            st.info(f"""
            **📋 Enhanced Export Contents:**
            - ✅ All original columns with Category in column K
            - ✅ Summary sheet with category breakdown
            - 🆕 **Root Cause Analysis** sheet (quality patterns identified)
            - 🆕 **Quality Actions** sheet (prioritized recommendations)
            - 🆕 **High Risk Products** sheet (products needing immediate attention)
            - ✅ Quality categories highlighted in red for easy identification
            
            **🔍 New Quality Insights Include:**
            - Pattern recognition (material vs component vs design issues)
            - Safety-critical issue flagging
            - Actionable recommendations by root cause
            - Risk assessment by product
            """)

def export_data(df: pd.DataFrame) -> bytes:
    """Export data maintaining original format with Category in column K"""
    
    output = io.BytesIO()
    
    if EXCEL_AVAILABLE:
        # Create Excel file
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Write main data - preserving original column order
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
            
            # Add quality insights sheet if available
            if st.session_state.quality_insights:
                insights = st.session_state.quality_insights
                
                # Root Cause Analysis
                if insights['root_cause_distribution']:
                    root_cause_data = []
                    for root_cause, data in sorted(insights['root_cause_distribution'].items(), 
                                                 key=lambda x: x[1]['count'], reverse=True):
                        root_cause_data.append({
                            'Root Cause': root_cause,
                            'Occurrences': data['count'],
                            'Affected Products': ', '.join(data['products'][:5]),  # Already a list
                            'Example': data['examples'][0] if data['examples'] else ''
                        })
                    
                    root_cause_df = pd.DataFrame(root_cause_data)
                    root_cause_df.to_excel(writer, sheet_name='Root Cause Analysis', index=False)
                
                # Action Items
                if insights['action_items']:
                    action_data = []
                    for action in insights['action_items']:
                        action_data.append({
                            'Priority': action['severity'],
                            'Issue': action['issue'],
                            'Frequency': action['frequency'],
                            'Recommendation': action['recommendation'],
                            'Affected Products': ', '.join(action['affected_products'][:3])
                        })
                    
                    action_df = pd.DataFrame(action_data)
                    action_df.to_excel(writer, sheet_name='Quality Actions', index=False)
                
                # High Risk Products
                if insights['risk_assessment']['top_risk_products']:
                    risk_df = pd.DataFrame(insights['risk_assessment']['top_risk_products'])
                    risk_df = risk_df[['product', 'total_issues', 'safety_issues', 'primary_root_cause']]
                    risk_df.columns = ['Product', 'Total Quality Issues', 'Safety Critical', 'Primary Root Cause']
                    risk_df.to_excel(writer, sheet_name='High Risk Products', index=False)
            
            # Format workbook
            workbook = writer.book
            
            # Auto-adjust columns on main sheet
            worksheet = writer.sheets['Categorized Returns']
            for i, col in enumerate(df.columns):
                max_len = max(
                    len(str(col)) + 2,
                    df[col].astype(str).str.len().max() + 2
                )
                max_len = min(max_len, 50)  # Cap at 50 characters
                worksheet.set_column(i, i, max_len)
            
            # Add conditional formatting for quality categories
            if 'Category' in df.columns:
                cat_col_idx = df.columns.get_loc('Category')
                
                # Create format for quality issues
                quality_format = workbook.add_format({
                    'bg_color': '#FFE6E6',
                    'font_color': '#CC0000'
                })
                
                # Apply to quality categories
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
    """Generate enhanced quality analysis report with pattern insights"""
    
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
Critical Issues: {st.session_state.severity_counts.get('critical', 0)}
"""
    
    # Add quality insights if available
    if st.session_state.quality_insights:
        insights = st.session_state.quality_insights
        report += f"""
QUALITY RISK ASSESSMENT
======================
Overall Risk Level: {insights['risk_assessment']['overall_risk_level']}
Quality Issue Rate: {insights['risk_assessment']['quality_rate']:.1f}%
Safety Critical Issues: {insights['risk_assessment']['safety_critical_count']}

ROOT CAUSE ANALYSIS
==================
"""
        for root_cause, data in sorted(insights['root_cause_distribution'].items(), 
                                     key=lambda x: x[1]['count'], reverse=True):
            report += f"\n{root_cause}: {data['count']} occurrences"
            report += f"\n  Affected Products: {', '.join(data['products'][:3])}"  # Already a list
            if len(data['products']) > 3:
                report += f" (+{len(data['products']) - 3} more)"
            report += "\n"
        
        if insights['action_items']:
            report += """
RECOMMENDED ACTIONS (PRIORITIZED)
=================================
"""
            for i, action in enumerate(insights['action_items'], 1):
                report += f"\n{i}. [{action['severity']}] {action['issue']} ({action['frequency']} cases)"
                report += f"\n   Action: {action['recommendation']}"
                report += f"\n   Products: {', '.join(action['affected_products'][:3])}\n"
        
        if insights['risk_assessment']['top_risk_products']:
            report += """
HIGH RISK PRODUCTS
==================
"""
            for prod in insights['risk_assessment']['top_risk_products'][:5]:
                report += f"\n{prod['product']}"
                report += f"\n  Total Issues: {prod['total_issues']}"
                report += f"\n  Safety Issues: {prod['safety_issues']}"
                report += f"\n  Primary Cause: {prod['primary_root_cause']}\n"
    
    report += f"""
RETURN CATEGORIES
=================
"""
    
    for cat, count in sorted(st.session_state.reason_summary.items(), 
                           key=lambda x: x[1], reverse=True):
        pct = (count / total_returns * 100) if total_returns > 0 else 0
        quality_flag = " [QUALITY]" if cat in QUALITY_CATEGORIES else ""
        report += f"{cat}{quality_flag}: {count} ({pct:.1f}%)\n"
    
    # Top products section
    if st.session_state.product_summary:
        report += f"""
TOP PRODUCTS BY RETURN VOLUME
=============================
"""
        product_totals = [(prod, sum(cats.values())) 
                         for prod, cats in st.session_state.product_summary.items()]
        
        for product, total in sorted(product_totals, key=lambda x: x[1], reverse=True)[:10]:
            top_issue = max(st.session_state.product_summary[product].items(), 
                          key=lambda x: x[1])
            report += f"\n{product}"
            report += f"\n  Total Returns: {total}"
            report += f"\n  Top Issue: {top_issue[0]} ({top_issue[1]} returns)\n"
    
    report += f"""
RECOMMENDATIONS
==============
1. Focus quality improvements on identified root causes
2. Prioritize HIGH severity action items immediately  
3. Review high-risk products for potential design changes
4. Implement incoming inspection for components with high failure rates
5. Consider MDR (Medical Device Reporting) for safety-critical issues
6. Track improvement metrics after implementing corrective actions

QUALITY IMPROVEMENT FOCUS AREAS
==============================
Based on pattern analysis, prioritize:
"""
    
    if st.session_state.quality_insights:
        # Add top 3 root causes as focus areas
        for root_cause, data in list(sorted(
            insights['root_cause_distribution'].items(), 
            key=lambda x: x[1]['count'], 
            reverse=True
        ))[:3]:
            report += f"- {root_cause}: Address in {len(data['products'])} products\n"
    else:
        report += "- Run analysis to identify quality patterns\n"
    
    return report

def main():
    """Main application function"""
    
    if not AI_AVAILABLE:
        st.error("""
        ❌ **AI module not found!** 
        
        Please ensure `enhanced_ai_analysis.py` is in the same directory as this app.
        """)
        st.stop()
    
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
            🆕 Now with Quality Pattern Recognition & Root Cause Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for settings
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        # Date filtering
        st.markdown("#### 📅 Date Filtering")
        enable_date_filter = st.checkbox("Enable date filter")
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
        
        # API Status
        if AI_AVAILABLE:
            st.markdown("---")
            st.markdown("#### 🤖 AI Status")
            analyzer = get_ai_analyzer()
            if analyzer:
                status = analyzer.get_api_status()
                if status['available']:
                    st.success("✅ AI Ready")
                    st.caption("Using GPT-4 (AI-first categorization)")
                else:
                    st.error("❌ API key not configured")
                    st.info("""
                    To use AI categorization:
                    1. Get an OpenAI API key
                    2. Add to Streamlit secrets:
                       - Key: `openai_api_key`
                       - Value: Your API key
                    """)
        else:
            st.error("❌ AI module missing")
    
    # Main content
    # Show required format
    with st.expander("📋 Required File Format & Instructions", expanded=False):
        display_required_format()
        
        st.markdown("### 🤖 AI-First Categorization:")
        st.info("""
        This tool uses OpenAI GPT-4 to analyze each return complaint and categorize it accurately.
        The AI considers:
        - The full context of the complaint
        - Medical device specific terminology
        - Complex multi-issue returns
        - Nuanced language and implications
        """)
        
        st.markdown("### 🔍 NEW: Quality Pattern Recognition")
        st.success("""
        **Automatic Root Cause Analysis** identifies:
        - Material failures (velcro, straps, fabric)
        - Component failures (pumps, valves, motors)
        - Design flaws vs manufacturing defects
        - Safety-critical issues requiring immediate attention
        - Specific failure patterns by product
        
        Get **actionable quality insights** with prioritized recommendations!
        """)
        
        st.markdown("### 📊 Supported File Types:")
        col1, col2 = st.columns(2)
        with col1:
            st.info("✅ Excel (.xlsx, .xls)")
        with col2:
            st.info("✅ CSV (.csv)")
    
    # File upload section
    st.markdown("---")
    st.markdown("### 📁 Upload Files")
    
    # Date filter info
    if st.session_state.date_filter_enabled:
        st.info(f"📅 Date filter active: {st.session_state.date_range_start} to {st.session_state.date_range_end}")
    
    uploaded_files = st.file_uploader(
        "Choose file(s) to categorize",
        type=['xlsx', 'xls', 'csv'],
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
                    
                    # Check AI status first
                    analyzer = get_ai_analyzer()
                    if not analyzer or not analyzer.get_api_status()['available']:
                        st.error("❌ AI not available. Please configure your OpenAI API key.")
                        st.stop()
                    
                    st.info("🤖 Using GPT-4 AI for accurate medical device return categorization...")
                    
                    with st.spinner(f"🤖 AI Processing {len(combined_df)} returns..."):
                        categorized_df = categorize_all_data(combined_df)
                        st.session_state.categorized_data = categorized_df
                        st.session_state.processing_complete = True
                    
                    # Show completion time
                    elapsed_time = time.time() - start_time
                    st.success(f"✅ AI categorization complete in {elapsed_time:.1f} seconds!")
            
            # Show results
            if st.session_state.processing_complete and st.session_state.categorized_data is not None:
                
                display_results(st.session_state.categorized_data)
                
                # Export section
                st.markdown("---")
                st.markdown("""
                <div style="background: rgba(0, 245, 160, 0.1); border: 2px solid var(--success); 
                          border-radius: 15px; padding: 2rem; text-align: center;">
                    <h3 style="color: var(--success);">✅ ANALYSIS COMPLETE!</h3>
                    <p>Your data has been categorized and is ready for export.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Export options
                col1, col2, col3 = st.columns(3)
                
                # Generate exports
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                with col1:
                    excel_data = export_data(st.session_state.categorized_data)
                    
                    st.download_button(
                        label="📥 DOWNLOAD EXCEL",
                        data=excel_data,
                        file_name=f"categorized_returns_{timestamp}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                        help="Excel file with Category in column K"
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
                        help="Comprehensive quality report with severity analysis"
                    )
                
                # Show export info
                st.info(f"""
                **📋 Enhanced Export Contents:**
                - ✅ All original columns with Category in column K
                - ✅ Summary sheet with category breakdown
                - 🆕 **Root Cause Analysis** sheet (quality patterns identified)
                - 🆕 **Quality Actions** sheet (prioritized recommendations)  
                - 🆕 **High Risk Products** sheet (products needing immediate attention)
                - ✅ Quality categories highlighted in red for easy identification
                
                **🔍 New Quality Insights Include:**
                - Pattern recognition (material vs component vs design issues)
                - Safety-critical issue flagging
                - Actionable recommendations by root cause
                - Risk assessment by product
                """)

if __name__ == "__main__":
    main()
