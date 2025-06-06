"""
Vive Health Quality Complaint Categorizer
AI-Powered Return Reason Classification Tool
Version: 4.0 - Unified API Support (Claude + OpenAI)

This enhanced version supports:
- Claude API (Haiku, Sonnet, Opus)
- OpenAI API (GPT-3.5, GPT-4)
- Hybrid mode comparing both providers
- Cost tracking and optimization
- Batch processing for 2000+ rows
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

# Import the unified AI module
try:
    from enhanced_ai_analysis import EnhancedAIAnalyzer, APIProvider, UnifiedAPIClient
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    st.error("❌ Critical: enhanced_ai_analysis.py module not found!")

# Excel handling
try:
    import xlsxwriter
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# App Configuration
APP_CONFIG = {
    'title': 'Vive Health Medical Device Return Categorizer',
    'version': '4.0',
    'company': 'Vive Health',
    'description': 'AI-Powered Medical Device Return Classification for Quality Management'
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

# Old categories for backward compatibility (will map to new ones)
OLD_TO_NEW_CATEGORY_MAP = {
    'too small': 'Size/Fit Issues',
    'too large': 'Size/Fit Issues',
    'wrong size': 'Size/Fit Issues',
    'received used/damaged': 'Product Defects/Quality',
    'wrong item': 'Wrong Product/Misunderstanding',
    'too heavy': 'Design/Material Issues',
    'bad brakes': 'Product Defects/Quality',
    'bad wheels': 'Product Defects/Quality',
    'uncomfortable': 'Comfort Issues',
    'difficult to use': 'Assembly/Usage Difficulty',
    'missing parts': 'Missing Components',
    'defective seat': 'Product Defects/Quality',
    'no issue': 'Customer Error/Changed Mind',
    'not as advertised': 'Wrong Product/Misunderstanding',
    'defective handles': 'Product Defects/Quality',
    'defective frame': 'Product Defects/Quality',
    'defective/does not work properly': 'Product Defects/Quality',
    'missing or broken parts': 'Missing Components',
    'performance or quality not adequate': 'Performance/Effectiveness',
    'incompatible or not useful': 'Equipment Compatibility',
    'no longer needed': 'Customer Error/Changed Mind',
    'bought by mistake': 'Customer Error/Changed Mind',
    'style not as expected': 'Wrong Product/Misunderstanding',
    'different from website description': 'Wrong Product/Misunderstanding',
    'damaged during shipping': 'Shipping/Fulfillment Issues',
    'item never arrived': 'Shipping/Fulfillment Issues',
    'unauthorized purchase': 'Customer Error/Changed Mind',
    'better price available': 'Price/Value',
    'ordered wrong item': 'Customer Error/Changed Mind',
    'changed mind': 'Customer Error/Changed Mind',
    'arrived too late': 'Shipping/Fulfillment Issues',
    'poor quality': 'Product Defects/Quality',
    'not compatible': 'Equipment Compatibility',
    'missing accessories': 'Missing Components',
    'installation issues': 'Assembly/Usage Difficulty',
    'customer damaged': 'Customer Error/Changed Mind',
    'other': 'Other/Miscellaneous'
}

def inject_cyberpunk_css():
    """Inject cyberpunk-themed CSS with API provider styling"""
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
    
    .api-selector {{
        background: rgba(0, 217, 255, 0.1);
        border: 2px solid var(--primary);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 0 30px rgba(0, 217, 255, 0.3);
    }}
    
    .api-option {{
        display: inline-block;
        margin: 0.5rem;
        padding: 1rem 2rem;
        background: rgba(26, 26, 46, 0.9);
        border: 2px solid var(--muted);
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.3s ease;
    }}
    
    .api-option:hover {{
        border-color: var(--primary);
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(0, 217, 255, 0.4);
    }}
    
    .api-option.selected {{
        border-color: var(--accent);
        background: rgba(255, 183, 0, 0.2);
    }}
    
    .cost-tracker {{
        position: fixed;
        top: 10px;
        right: 10px;
        background: rgba(10, 10, 15, 0.95);
        border: 1px solid var(--primary);
        border-radius: 10px;
        padding: 1rem;
        min-width: 200px;
        z-index: 1000;
    }}
    
    .model-badge {{
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: 600;
        margin: 0.2rem;
    }}
    
    .badge-openai {{
        background: rgba(116, 185, 255, 0.2);
        border: 1px solid #74b9ff;
        color: #74b9ff;
    }}
    
    .badge-claude {{
        background: rgba(162, 155, 254, 0.2);
        border: 1px solid #a29bfe;
        color: #a29bfe;
    }}
    
    .comparison-result {{
        background: rgba(26, 26, 46, 0.8);
        border: 1px solid var(--primary);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
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
    
    #MainMenu, footer, header {{
        visibility: hidden;
    }}
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        'uploaded_file': None,
        'processed_data': None,
        'categorized_data': None,
        'ai_analyzer': None,
        'processing_complete': False,
        'category_mapping': None,
        'file_type': None,
        'reason_summary': {},
        'api_provider': 'both',  # Default to both
        'model_settings': {
            'openai_categorization': 'gpt-3.5-turbo',
            'openai_analysis': 'gpt-4o-mini',
            'claude_categorization': 'claude-3-haiku',
            'claude_analysis': 'claude-3-sonnet'
        },
        'comparison_mode': False,
        'cost_tracking': {'session_cost': 0.0}
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def display_api_selector():
    """Display API provider selection UI"""
    st.markdown("""
    <div class="api-selector">
        <h3 style="color: var(--primary); margin-top: 0;">🤖 SELECT AI PROVIDER</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col1:
        if st.button("🟦 OpenAI", use_container_width=True, 
                     type="primary" if st.session_state.api_provider == 'openai' else "secondary"):
            st.session_state.api_provider = 'openai'
            st.rerun()
    
    with col2:
        if st.button("🟣 Claude", use_container_width=True,
                     type="primary" if st.session_state.api_provider == 'claude' else "secondary"):
            st.session_state.api_provider = 'claude'
            st.rerun()
    
    with col3:
        if st.button("🔀 Both (Smart)", use_container_width=True,
                     type="primary" if st.session_state.api_provider == 'both' else "secondary"):
            st.session_state.api_provider = 'both'
            st.rerun()
    
    with col4:
        st.checkbox("📊 Compare Mode", 
                   value=st.session_state.comparison_mode,
                   help="Run on both providers and compare results",
                   key="comparison_checkbox")
        st.session_state.comparison_mode = st.session_state.comparison_checkbox
    
    # Display current selection
    provider_info = {
        'openai': ('🟦 OpenAI', 'Using GPT models for categorization'),
        'claude': ('🟣 Claude', 'Using Claude models for categorization'),
        'both': ('🔀 Smart Mode', 'Automatically selects the best provider for each task')
    }
    
    icon, desc = provider_info[st.session_state.api_provider]
    st.info(f"{icon} **Active**: {desc}")
    
    # Advanced settings expander
    with st.expander("⚙️ Advanced Model Settings", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### OpenAI Models")
            openai_cat = st.selectbox(
                "Categorization Model",
                ['gpt-3.5-turbo', 'gpt-4o-mini', 'gpt-4-turbo'],
                index=['gpt-3.5-turbo', 'gpt-4o-mini', 'gpt-4-turbo'].index(
                    st.session_state.model_settings['openai_categorization']
                )
            )
            st.session_state.model_settings['openai_categorization'] = openai_cat
            
            openai_analysis = st.selectbox(
                "Analysis Model",
                ['gpt-4o-mini', 'gpt-4-turbo'],
                index=['gpt-4o-mini', 'gpt-4-turbo'].index(
                    st.session_state.model_settings['openai_analysis']
                )
            )
            st.session_state.model_settings['openai_analysis'] = openai_analysis
        
        with col2:
            st.markdown("### Claude Models")
            claude_cat = st.selectbox(
                "Categorization Model",
                ['claude-3-haiku', 'claude-3-sonnet', 'claude-3-opus'],
                index=['claude-3-haiku', 'claude-3-sonnet', 'claude-3-opus'].index(
                    st.session_state.model_settings['claude_categorization']
                )
            )
            st.session_state.model_settings['claude_categorization'] = claude_cat
            
            claude_analysis = st.selectbox(
                "Analysis Model",
                ['claude-3-sonnet', 'claude-3-opus'],
                index=['claude-3-sonnet', 'claude-3-opus'].index(
                    st.session_state.model_settings['claude_analysis']
                )
            )
            st.session_state.model_settings['claude_analysis'] = claude_analysis

def get_ai_analyzer():
    """Get or create AI analyzer with selected provider"""
    if st.session_state.ai_analyzer is None and AI_AVAILABLE:
        provider_map = {
            'openai': APIProvider.OPENAI,
            'claude': APIProvider.CLAUDE,
            'both': APIProvider.BOTH
        }
        provider = provider_map[st.session_state.api_provider]
        st.session_state.ai_analyzer = EnhancedAIAnalyzer(provider)
        
        # Set model preferences
        analyzer = st.session_state.ai_analyzer
        settings = st.session_state.model_settings
        
        # Map UI names to actual model names
        model_map = {
            'gpt-3.5-turbo': 'gpt-3.5-turbo',
            'gpt-4o-mini': 'gpt-4o-mini',
            'gpt-4-turbo': 'gpt-4-turbo-preview',
            'claude-3-haiku': 'claude-3-haiku-20240307',
            'claude-3-sonnet': 'claude-3-sonnet-20240229',
            'claude-3-opus': 'claude-3-opus-20240229'
        }
        
        # Set preferences
        analyzer.api_client.set_model_preference(
            'categorization', 'openai', model_map[settings['openai_categorization']]
        )
        analyzer.api_client.set_model_preference(
            'analysis', 'openai', model_map[settings['openai_analysis']]
        )
        analyzer.api_client.set_model_preference(
            'categorization', 'claude', model_map[settings['claude_categorization']]
        )
        analyzer.api_client.set_model_preference(
            'analysis', 'claude', model_map[settings['claude_analysis']]
        )
    
    return st.session_state.ai_analyzer

def display_api_status():
    """Display comprehensive API status"""
    analyzer = get_ai_analyzer()
    if not analyzer:
        st.error("AI analyzer not available")
        return
    
    status = analyzer.get_api_status()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if status['openai']['available']:
            st.success("✅ OpenAI Connected")
            if 'tested' in status['openai']:
                st.caption("API test: " + ("✅ Passed" if status['openai']['tested'] else "❌ Failed"))
        else:
            st.error("❌ OpenAI Not Configured")
    
    with col2:
        if status['claude']['available']:
            st.success("✅ Claude Connected")
            if 'tested' in status['claude']:
                st.caption("API test: " + ("✅ Passed" if status['claude']['tested'] else "❌ Failed"))
        else:
            st.error("❌ Claude Not Configured")
    
    with col3:
        if 'usage' in status and status['usage']['combined_total'] > 0:
            st.metric("Session Cost", f"${status['usage']['combined_total']:.3f}")
            with st.expander("Cost Details"):
                usage = status['usage']
                if usage['openai']['total_cost'] > 0:
                    st.write(f"**OpenAI**: ${usage['openai']['total_cost']:.3f}")
                    st.caption(f"Tokens: {usage['openai']['input_tokens']:,} in, {usage['openai']['output_tokens']:,} out")
                if usage['claude']['total_cost'] > 0:
                    st.write(f"**Claude**: ${usage['claude']['total_cost']:.3f}")
                    st.caption(f"Tokens: {usage['claude']['input_tokens']:,} in, {usage['claude']['output_tokens']:,} out")

def categorize_with_comparison(complaint: str, fba_reason: str = None) -> Dict:
    """Categorize using comparison mode"""
    analyzer = get_ai_analyzer()
    
    if st.session_state.comparison_mode and st.session_state.api_provider == 'both':
        # Get results from both providers
        comparison = analyzer.categorize_return(complaint, fba_reason, use_both=True)
        
        # Find consensus or return both results
        openai_result = comparison.get('openai', {}).get('result', 'ERROR')
        claude_result = comparison.get('claude', {}).get('result', 'ERROR')
        
        if openai_result == claude_result:
            return {
                'category': openai_result,
                'consensus': True,
                'openai': openai_result,
                'claude': claude_result,
                'openai_time': comparison.get('openai', {}).get('time', 0),
                'claude_time': comparison.get('claude', {}).get('time', 0)
            }
        else:
            # Disagreement - could implement voting or preference logic
            return {
                'category': claude_result,  # Default to Claude for categorization
                'consensus': False,
                'openai': openai_result,
                'claude': claude_result,
                'openai_time': comparison.get('openai', {}).get('time', 0),
                'claude_time': comparison.get('claude', {}).get('time', 0)
            }
    else:
        # Single provider mode
        category = analyzer.categorize_return(complaint, fba_reason)
        return {'category': category, 'consensus': None}

def categorize_all_complaints(df: pd.DataFrame) -> pd.DataFrame:
    """Categorize all complaints with medical device categories"""
    
    analyzer = get_ai_analyzer()
    if not analyzer or not analyzer.api_client.is_available():
        st.error("No AI provider available. Please configure API keys.")
        return df
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    cost_text = st.empty()
    
    # Prepare data for batch processing with all relevant columns
    returns_data = []
    for idx, row in df.iterrows():
        # Get data from specific columns:
        # A: Date
        # B: Product Identifier Tag (product name)
        # C: Imported SKU
        # F: Order #
        # I: Complaint
        
        complaint = str(row.get('Complaint', '')) if pd.notna(row.get('Complaint')) else ""
        
        # For FBA returns, also get the reason code
        fba_reason = ""
        if st.session_state.file_type == 'fba_returns':
            fba_reason = str(row.get('FBA_Reason_Code', '')) if pd.notna(row.get('FBA_Reason_Code')) else ""
        
        # Collect product information from the specified columns
        product_info = {
            'date': str(row.get('Date', '')) if pd.notna(row.get('Date')) else "",
            'product_name': str(row.get('Product Identifier Tag', '')) if pd.notna(row.get('Product Identifier Tag')) else "",
            'sku': str(row.get('Imported SKU', '')) if pd.notna(row.get('Imported SKU')) else "",
            'order_id': str(row.get('Order #', '')) if pd.notna(row.get('Order #')) else ""
        }
        
        returns_data.append({
            'index': idx,
            'complaint': complaint,
            'reason': fba_reason,
            'sku': product_info['sku'],
            'product_name': product_info['product_name'],
            'order_id': product_info['order_id'],
            'date': product_info['date']
        })
    
    # Process in batches
    batch_results = analyzer.batch_categorize(
        returns_data,
        batch_size=25,
        progress_callback=lambda p: progress_bar.progress(p)
    )
    
    # Apply results to dataframe
    df_copy = df.copy()
    df_copy['Return_Reason'] = ''
    df_copy['AI_Provider'] = ''
    
    # If comparison mode, add extra columns
    if st.session_state.comparison_mode:
        df_copy['OpenAI_Category'] = ''
        df_copy['Claude_Category'] = ''
        df_copy['Consensus'] = ''
    
    # Map results
    result_map = {r['index']: r for r in batch_results}
    
    reason_counts = Counter()
    provider_counts = Counter()
    
    for idx in df_copy.index:
        if idx in result_map:
            result = result_map[idx]
            category = result.get('category', 'Other/Miscellaneous')
            provider = result.get('provider', 'unknown')
            
            # Ensure we're using valid medical device categories
            if category not in MEDICAL_DEVICE_CATEGORIES:
                # Try to map from old categories
                category = OLD_TO_NEW_CATEGORY_MAP.get(category.lower(), 'Other/Miscellaneous')
            
            df_copy.at[idx, 'Return_Reason'] = category
            df_copy.at[idx, 'AI_Provider'] = provider
            
            reason_counts[category] += 1
            provider_counts[provider] += 1
        else:
            # Fallback for any missed items
            df_copy.at[idx, 'Return_Reason'] = 'Other/Miscellaneous'
            reason_counts['Other/Miscellaneous'] += 1
    
    # Update cost display
    usage = analyzer.api_client.get_usage_summary()
    cost_text.text(f"Total cost: ${usage['combined_total']:.3f}")
    
    status_text.text("✅ Categorization complete!")
    
    # Store summary
    st.session_state.reason_summary = dict(reason_counts)
    
    # Add provider breakdown to summary
    st.info(f"Processed with: {', '.join([f'{k}: {v}' for k, v in provider_counts.items()])}")
    
    return df_copy

def display_results_with_comparison(df: pd.DataFrame):
    """Display results including comparison data if available"""
    
    # Standard results display
    display_results_summary(df)
    
    # Additional comparison insights if in comparison mode
    if st.session_state.comparison_mode and 'OpenAI_Category' in df.columns:
        st.markdown("---")
        st.markdown("### 🔀 Provider Comparison Analysis")
        
        # Calculate agreement rate
        consensus_mask = df['OpenAI_Category'] == df['Claude_Category']
        agreement_rate = consensus_mask.sum() / len(df) * 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Agreement Rate", f"{agreement_rate:.1f}%")
        
        with col2:
            disagreements = len(df) - consensus_mask.sum()
            st.metric("Disagreements", disagreements)
        
        with col3:
            # Show which provider was used more
            provider_counts = df['AI_Provider'].value_counts()
            st.metric("Primary Provider", provider_counts.index[0] if len(provider_counts) > 0 else "N/A")
        
        # Show disagreement examples
        if disagreements > 0:
            with st.expander(f"View {min(disagreements, 10)} Disagreement Examples"):
                disagreement_df = df[~consensus_mask].head(10)
                for idx, row in disagreement_df.iterrows():
                    st.markdown(f"""
                    <div class="comparison-result">
                        <strong>Complaint:</strong> {row.get('Complaint', '')[:200]}...<br>
                        <span class="badge-openai">OpenAI: {row['OpenAI_Category']}</span>
                        <span class="badge-claude">Claude: {row['Claude_Category']}</span>
                        <br><strong>Final:</strong> {row['Return_Reason']}
                    </div>
                    """, unsafe_allow_html=True)

def display_results_summary(df: pd.DataFrame):
    """Display summary of categorization results for medical devices"""
    
    st.markdown("""
    <div class="neon-box">
        <h2 style="color: var(--primary); text-align: center;">📊 QUALITY CATEGORIZATION RESULTS</h2>
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
        # Calculate quality-related returns
        quality_categories = [
            'Product Defects/Quality', 'Performance/Effectiveness', 
            'Missing Components', 'Design/Material Issues'
        ]
        quality_count = sum(
            st.session_state.reason_summary.get(cat, 0) 
            for cat in quality_categories
        )
        quality_pct = (quality_count / len(df) * 100) if len(df) > 0 else 0
        
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: var(--danger);">{quality_pct:.1f}%</h3>
            <p>Quality Issues</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        # Show cost if available
        analyzer = get_ai_analyzer()
        if analyzer:
            usage = analyzer.api_client.get_usage_summary()
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: var(--warning);">${usage['combined_total']:.2f}</h3>
                <p>Processing Cost</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Continue with rest of display...
    st.markdown("---")
    st.markdown("### 📈 Return Category Distribution")
    
    # Sort reasons by count
    sorted_reasons = sorted(st.session_state.reason_summary.items(), key=lambda x: x[1], reverse=True)
    
    # Create two columns for the breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Top Return Categories")
        for i, (category, count) in enumerate(sorted_reasons[:10]):
            percentage = (count / len(df)) * 100
            
            # Color coding for medical device categories
            if category in ['Product Defects/Quality', 'Performance/Effectiveness']:
                color = COLORS['danger']
                icon = "🚨"
            elif category in ['Size/Fit Issues', 'Equipment Compatibility']:
                color = COLORS['warning']
                icon = "⚠️"
            elif category in ['Customer Error/Changed Mind', 'Price/Value']:
                color = COLORS['success']
                icon = "✅"
            else:
                color = COLORS['primary']
                icon = "ℹ️"
            
            st.markdown(f"""
            <div style="margin: 0.5rem 0;">
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
        # Quality Management Insights
        st.markdown("#### 🎯 Quality Management Insights")
        
        # Group categories by quality relevance
        quality_breakdown = {
            'Product Quality': [
                'Product Defects/Quality', 'Performance/Effectiveness',
                'Missing Components', 'Design/Material Issues'
            ],
            'User Experience': [
                'Size/Fit Issues', 'Comfort Issues', 'Equipment Compatibility',
                'Stability/Positioning Issues', 'Assembly/Usage Difficulty'
            ],
            'Fulfillment': [
                'Wrong Product/Misunderstanding', 'Shipping/Fulfillment Issues'
            ],
            'Customer': [
                'Customer Error/Changed Mind', 'Medical/Health Concerns', 'Price/Value'
            ]
        }
        
        for group_name, categories in quality_breakdown.items():
            group_count = sum(
                st.session_state.reason_summary.get(cat, 0) 
                for cat in categories
            )
            if group_count > 0:
                group_pct = (group_count / len(df)) * 100
                st.markdown(f"""
                <div style="background: rgba(26, 26, 46, 0.8); border-radius: 10px; 
                            padding: 0.75rem; margin: 0.5rem 0;">
                    <strong>{group_name}:</strong> {group_count} returns ({group_pct:.1f}%)
                </div>
                """, unsafe_allow_html=True)
        
        # FDA/Quality Action Priority
        st.markdown("---")
        st.markdown("#### 🏥 FDA/Quality Action Priority")
        
        # Identify critical quality issues
        critical_categories = ['Product Defects/Quality', 'Medical/Health Concerns']
        critical_count = sum(
            st.session_state.reason_summary.get(cat, 0) 
            for cat in critical_categories
        )
        
        if critical_count > 0:
            st.error(f"⚠️ {critical_count} returns require immediate quality review")
        else:
            st.success("✅ No critical quality issues detected")
        
        # Top products with issues (if available)
        if 'Product Identifier Tag' in df.columns:
            st.markdown("---")
            st.markdown("#### 📦 Products with Most Returns")
            
            # Get top 5 products by return count
            product_returns = df.groupby('Product Identifier Tag').size().sort_values(ascending=False).head(5)
            
            for product, count in product_returns.items():
                if product and str(product).strip():
                    product_display = str(product)[:50] + "..." if len(str(product)) > 50 else str(product)
                    st.markdown(f"- **{product_display}**: {count} returns")


def process_complaints_file(file_content, filename: str) -> pd.DataFrame:
    """Process uploaded file (unchanged from original)"""
    try:
        # Detect file type
        if filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(file_content))
            file_type = 'complaints_ledger'
        elif filename.endswith('.txt'):
            try:
                df = pd.read_csv(io.BytesIO(file_content), sep='\t')
                file_type = 'fba_returns'
            except:
                df = pd.read_csv(io.BytesIO(file_content))
                file_type = 'fba_returns'
        else:
            df = pd.read_csv(io.BytesIO(file_content))
            file_type = 'unknown'
        
        # Log columns
        logger.info(f"Columns found: {df.columns.tolist()}")
        
        # Process based on file type
        if 'customer-comments' in df.columns and 'reason' in df.columns:
            file_type = 'fba_returns'
            st.info("📦 Detected FBA Return Report format")
            
            df = df.rename(columns={
                'customer-comments': 'Complaint',
                'return-date': 'Date',
                'order-id': 'Order #',
                'product-name': 'Product Identifier Tag',
                'sku': 'Imported SKU'
            })
            
            if 'reason' in df.columns:
                df['FBA_Reason_Code'] = df['reason']
            
            st.session_state.file_type = 'fba_returns'
        else:
            st.info("📋 Detected Complaints Ledger format")
            
            if 'Complaint' not in df.columns:
                complaint_cols = [col for col in df.columns if 'complaint' in col.lower() or 'comment' in col.lower()]
                if complaint_cols:
                    df = df.rename(columns={complaint_cols[0]: 'Complaint'})
                else:
                    st.error("Could not find complaint column")
                    return None
            
            st.session_state.file_type = 'complaints_ledger'
        
        return df
        
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        st.error(f"Error reading file: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title=APP_CONFIG['title'],
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    if not AI_AVAILABLE:
        st.error("❌ Critical Error: AI module not found!")
        st.stop()
    
    initialize_session_state()
    inject_cyberpunk_css()
    
    # Header
    st.markdown(f"""
    <h1>{APP_CONFIG['title']}</h1>
    <p style="text-align: center; color: var(--primary); font-size: 1.2em; margin-bottom: 2rem;">
        {APP_CONFIG['description']} - Now with Claude & OpenAI Support!
    </p>
    """, unsafe_allow_html=True)
    
    # API Provider Selection
    display_api_selector()
    
    # API Status
    display_api_status()
    
    # Instructions
    with st.expander("📖 How to Use This Tool", expanded=False):
        st.markdown("""
        ### Quick Start Guide
        1. **Select AI Provider**: Choose OpenAI, Claude, or Both (smart mode)
        2. **Upload your file**: Complaints Ledger (.xlsx) or FBA Return Report (.txt)
        3. **AI categorizes** each return into medical device categories
        4. **Download results** with categorized return reasons in Column K
        
        ### Medical Device Return Categories:
        1. **Size/Fit Issues** - Too large/small, wrong size, doesn't fit
        2. **Comfort Issues** - Uncomfortable, hurts, too firm/soft
        3. **Product Defects/Quality** - Defective, broken, poor quality
        4. **Performance/Effectiveness** - Not as expected, ineffective
        5. **Stability/Positioning Issues** - Doesn't stay in place, slides
        6. **Equipment Compatibility** - Doesn't fit walker/wheelchair/toilet
        7. **Design/Material Issues** - Too bulky/heavy/thin, flimsy
        8. **Wrong Product/Misunderstanding** - Wrong item, not as advertised
        9. **Missing Components** - Missing parts, no instructions
        10. **Customer Error/Changed Mind** - Ordered wrong, no longer needed
        11. **Shipping/Fulfillment Issues** - Arrived late, damaged in shipping
        12. **Assembly/Usage Difficulty** - Difficult to use/adjust
        13. **Medical/Health Concerns** - Doctor didn't approve, allergic
        14. **Price/Value** - Better price found
        15. **Other/Miscellaneous** - Unique situations
        
        ### Key Columns Used:
        - **Column A**: Date
        - **Column B**: Product Identifier Tag
        - **Column C**: Imported SKU
        - **Column F**: Order #
        - **Column I**: Complaint/Return Reason
        - **Column K**: AI-Categorized Return Category (OUTPUT)
        
        ### Provider Options:
        - **🟦 OpenAI**: Fast, reliable categorization with GPT models
        - **🟣 Claude**: Ultra-fast with Haiku, high quality with Opus
        - **🔀 Both**: Automatically selects the best provider for each task
        - **📊 Compare Mode**: Run both and see how they compare!
        
        ### Cost Optimization:
        - Claude Haiku: ~$0.23 per 2000 returns (fastest)
        - GPT-3.5: ~$0.35 per 2000 returns
        - Claude Opus: ~$13.50 per 2000 returns (highest quality)
        
        ### Quality Management Focus:
        The tool automatically identifies:
        - High-priority quality issues requiring immediate action
        - Products with the most returns
        - Patterns indicating design or manufacturing problems
        - Customer error vs actual product issues
        """)
    
    # File upload section
    st.markdown("""
    <div class="neon-box">
        <h3 style="color: var(--accent);">📁 UPLOAD RETURN DATA FILE</h3>
        <p style="color: var(--text);">Upload your complaints ledger or FBA return report for AI categorization</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose your file",
        type=['xlsx', 'xls', 'csv', 'txt'],
        help="Upload complaints file (.xlsx) or FBA Return Report (.txt)"
    )
    
    if uploaded_file is not None:
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
            if st.session_state.get('file_type') == 'fba_returns':
                display_cols = ['Date', 'Order #', 'Product Identifier Tag', 'FBA_Reason_Code', 'Complaint']
            else:
                display_cols = ['Date', 'Product Identifier Tag', 'Order #', 'Complaint']
            
            available_cols = [col for col in display_cols if col in df.columns]
            if available_cols:
                st.dataframe(df[available_cols].head(5), use_container_width=True)
            
            # Categorize button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 CATEGORIZE COMPLAINTS", type="primary", use_container_width=True):
                    with st.spinner(f"🤖 AI is analyzing using {st.session_state.api_provider.upper()}..."):
                        categorized_df = categorize_all_complaints(df)
                        st.session_state.categorized_data = categorized_df
                        st.session_state.processing_complete = True
                    
                    st.balloons()
                    st.success("✅ Categorization complete!")
            
            # Show results if processing is complete
            if st.session_state.processing_complete and st.session_state.categorized_data is not None:
                display_results_with_comparison(st.session_state.categorized_data)
                
                # Export section
                st.markdown("---")
                st.markdown("""
                <div class="success-box">
                    <h3 style="color: var(--success);">📥 EXPORT RESULTS</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Generate export file
                excel_data = export_categorized_data(st.session_state.categorized_data)
                
                # Download button
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                st.download_button(
                    label=f"📥 DOWNLOAD CATEGORIZED DATA",
                    data=excel_data,
                    file_name=f"categorized_complaints_{timestamp}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

# Helper functions
def prepare_export_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for export with medical device categories"""
    file_type = st.session_state.get('file_type', 'complaints_ledger')
    
    if file_type == 'fba_returns':
        # FBA Return Report format
        original_columns = ['return-date', 'order-id', 'sku', 'asin', 'fnsku', 
                           'product-name', 'quantity', 'fulfillment-center-id', 
                           'detailed-disposition', 'reason', 'status', 
                           'license-plate-number', 'customer-comments']
        
        column_mapping = {
            'Date': 'return-date',
            'Order #': 'order-id',
            'Imported SKU': 'sku',
            'Product Identifier Tag': 'product-name',
            'Complaint': 'customer-comments',
            'FBA_Reason_Code': 'reason'
        }
        
        export_df = pd.DataFrame()
        
        for i, col in enumerate(original_columns[:10]):
            if col in df.columns:
                export_df[col] = df[col]
            else:
                reverse_mapping = {v: k for k, v in column_mapping.items()}
                if col in reverse_mapping and reverse_mapping[col] in df.columns:
                    export_df[col] = df[reverse_mapping[col]]
                else:
                    export_df[col] = ''
    else:
        # Complaints Ledger format
        original_columns = ['Date', 'Product Identifier Tag', 'Imported SKU', 'UDI', 
                           'CS Ticket #', 'Order #', 'Source', 'Categorizing / Investigating Agent',
                           'Complaint', 'Review stars']
        
        export_columns = [col for col in original_columns if col in df.columns]
        export_df = df[export_columns].copy()
    
    # Add Column K - Medical Device Return Category
    export_df['Return Reason'] = df['Return_Reason']
    
    # Add blank columns L and M
    export_df['Blank1'] = ''
    export_df['Blank2'] = ''
    
    # Add AI provider info if available
    if 'AI_Provider' in df.columns:
        export_df['AI_Provider'] = df['AI_Provider']
    
    return export_df

def export_categorized_data(df: pd.DataFrame) -> bytes:
    """Export categorized data to Excel with medical device categories"""
    output = io.BytesIO()
    
    export_df = prepare_export_data(df)
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Main data sheet
        export_df.to_excel(writer, sheet_name='Categorized Returns', index=False)
        
        # Add summary sheet
        summary_data = []
        for category, count in sorted(st.session_state.reason_summary.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(df)) * 100
            
            # Determine quality impact
            quality_impact = "High" if category in [
                'Product Defects/Quality', 'Medical/Health Concerns'
            ] else "Medium" if category in [
                'Performance/Effectiveness', 'Missing Components', 'Design/Material Issues'
            ] else "Low"
            
            summary_data.append({
                'Return Category': category,
                'Count': count,
                'Percentage': f"{percentage:.1f}%",
                'Quality Impact': quality_impact
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Category Summary', index=False)
        
        # Add quality analysis sheet
        quality_categories = [
            'Product Defects/Quality', 'Performance/Effectiveness',
            'Missing Components', 'Design/Material Issues'
        ]
        quality_returns = df[df['Return_Reason'].isin(quality_categories)]
        
        if len(quality_returns) > 0:
            # Quality summary by product
            quality_by_product = quality_returns.groupby(['Product Identifier Tag', 'Return_Reason']).size().reset_index(name='Count')
            quality_by_product.to_excel(writer, sheet_name='Quality Issues by Product', index=False)
        
        # Add cost tracking sheet if available
        analyzer = get_ai_analyzer()
        if analyzer:
            usage = analyzer.api_client.get_usage_summary()
            cost_data = []
            
            if usage['openai']['total_cost'] > 0:
                cost_data.append({
                    'Provider': 'OpenAI',
                    'Input Tokens': f"{usage['openai']['input_tokens']:,}",
                    'Output Tokens': f"{usage['openai']['output_tokens']:,}",
                    'Total Calls': usage['openai']['total_calls'],
                    'Total Cost': f"${usage['openai']['total_cost']:.3f}"
                })
            
            if usage['claude']['total_cost'] > 0:
                cost_data.append({
                    'Provider': 'Claude',
                    'Input Tokens': f"{usage['claude']['input_tokens']:,}",
                    'Output Tokens': f"{usage['claude']['output_tokens']:,}",
                    'Total Calls': usage['claude']['total_calls'],
                    'Total Cost': f"${usage['claude']['total_cost']:.3f}"
                })
            
            if cost_data:
                cost_data.append({
                    'Provider': 'TOTAL',
                    'Input Tokens': '',
                    'Output Tokens': '',
                    'Total Calls': '',
                    'Total Cost': f"${usage['combined_total']:.3f}"
                })
                cost_df = pd.DataFrame(cost_data)
                cost_df.to_excel(writer, sheet_name='Processing Costs', index=False)
        
        # Format the Excel file
        workbook = writer.book
        
        # Add header format
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#1A1A2E',
            'font_color': '#00D9FF',
            'border': 1
        })
        
        # Format each sheet
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            
            # Set column widths based on sheet
            if sheet_name == 'Categorized Returns':
                column_widths = {
                    0: 15,   # Date
                    1: 40,   # Product Identifier Tag
                    2: 20,   # Imported SKU
                    3: 15,   # UDI
                    4: 15,   # CS Ticket #
                    5: 20,   # Order #
                    6: 20,   # Source
                    7: 25,   # Categorizing Agent
                    8: 50,   # Complaint
                    9: 12,   # Review stars
                    10: 30,  # Return Reason (Column K)
                }
                
                for col, width in column_widths.items():
                    if col < len(export_df.columns):
                        worksheet.set_column(col, col, width)
    
    output.seek(0)
    return output.getvalue()

if __name__ == "__main__":
    main()
