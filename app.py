"""
Amazon Review Analyzer - Advanced Listing Optimization Engine
Vive Health | Cyberpunk Edition v9.0 - Marketplace Data Integration
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import io
from typing import Dict, List, Any, Optional, Tuple
import re
from collections import Counter, defaultdict
from io import BytesIO
import requests
from bs4 import BeautifulSoup
import time
import json

# Import handling with fallbacks
try:
    import xlsxwriter
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import enhanced_ai_analysis
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    logger.warning("AI module not available")

try:
    from amazon_file_detector import AmazonFileDetector
    DETECTOR_AVAILABLE = True
except ImportError:
    DETECTOR_AVAILABLE = False
    logger.warning("Amazon file detector module not available")

# Configuration
APP_CONFIG = {
    'title': 'Vive Health Review Intelligence',
    'version': '9.0',
    'company': 'Vive Health',
    'support_email': 'alexander.popoff@vivehealth.com'
}

COLORS = {
    'primary': '#00D9FF', 'secondary': '#FF006E', 'accent': '#FFB700',
    'success': '#00F5A0', 'warning': '#FF6B35', 'danger': '#FF0054',
    'dark': '#0A0A0F', 'light': '#1A1A2E', 'text': '#E0E0E0', 'muted': '#666680'
}

# Amazon scraping headers to avoid blocks
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
}

def initialize_session_state():
    """Initialize session state"""
    defaults = {
        'uploaded_data': None, 'analysis_results': None, 'current_view': 'upload',
        'processing': False, 'ai_analyzer': None, 'chat_messages': [],
        'show_ai_chat': False, 'selected_timeframe': 'all', 'filter_rating': 'all',
        'analysis_depth': 'comprehensive', 'use_listing_details': False,
        'listing_details': {
            'title': '', 'bullet_points': ['', '', '', '', ''], 'description': '',
            'backend_keywords': '', 'brand': '', 'category': '', 'asin': '', 'url': ''
        },
        'scraping_status': None, 'auto_populated': False, 'analyze_all_reviews': True,
        'marketplace_files': {
            'reimbursements': None,
            'fba_returns': None,
            'fbm_returns': None
        },
        'marketplace_data': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def extract_asin_from_url(url: str) -> Optional[str]:
    """Extract ASIN from Amazon URL"""
    try:
        # Common ASIN patterns in Amazon URLs
        patterns = [
            r'/dp/([A-Z0-9]{10})',
            r'/product/([A-Z0-9]{10})',
            r'asin=([A-Z0-9]{10})',
            r'/([A-Z0-9]{10})/',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    except Exception as e:
        logger.error(f"ASIN extraction error: {e}")
        return None

def clean_text(text: str) -> str:
    """Clean scraped text"""
    if not text:
        return ""
    
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text.strip())
    # Remove common Amazon artifacts
    text = re.sub(r'See more product details', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Read more', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Show more', '', text, flags=re.IGNORECASE)
    
    return text.strip()

def scrape_amazon_product(url: str) -> Dict[str, Any]:
    """Scrape Amazon product information"""
    try:
        # Validate URL
        if not url or 'amazon.' not in url.lower():
            return {'success': False, 'error': 'Invalid Amazon URL'}
        
        # Extract ASIN
        asin = extract_asin_from_url(url)
        if not asin:
            return {'success': False, 'error': 'Could not extract ASIN from URL'}
        
        # Make request with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=HEADERS, timeout=10)
                if response.status_code == 200:
                    break
                elif response.status_code == 503:
                    time.sleep(2 * (attempt + 1))  # Exponential backoff
                    continue
                else:
                    return {'success': False, 'error': f'HTTP {response.status_code}'}
            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    return {'success': False, 'error': f'Request failed: {str(e)}'}
                time.sleep(1)
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract product information
        product_data = {
            'success': True,
            'asin': asin,
            'url': url,
            'title': '',
            'bullet_points': [],
            'description': '',
            'brand': '',
            'category': ''
        }
        
        # Title extraction (multiple selectors for different layouts)
        title_selectors = [
            '#productTitle',
            '.product-title',
            'h1[data-automation-id="product-title"]',
            '.it-ttl'
        ]
        
        for selector in title_selectors:
            title_element = soup.select_one(selector)
            if title_element:
                product_data['title'] = clean_text(title_element.get_text())
                break
        
        # Bullet points extraction
        bullet_selectors = [
            '#feature-bullets ul li span',
            '.a-unordered-list.a-vertical.a-spacing-mini li span',
            '.feature-bullets-list li',
            '#feature-bullets li'
        ]
        
        for selector in bullet_selectors:
            bullets = soup.select(selector)
            if bullets:
                bullet_texts = []
                for bullet in bullets:
                    text = clean_text(bullet.get_text())
                    if text and len(text) > 10 and not any(skip in text.lower() for skip in ['make sure', 'see more', 'important information']):
                        bullet_texts.append(text)
                
                if bullet_texts:
                    # Take up to 5 bullets
                    product_data['bullet_points'] = bullet_texts[:5]
                    break
        
        # Product description
        desc_selectors = [
            '#productDescription p',
            '#aplus_feature_div',
            '.a-expander-content p',
            '#feature-bullets + div'
        ]
        
        for selector in desc_selectors:
            desc_elements = soup.select(selector)
            if desc_elements:
                desc_texts = []
                for elem in desc_elements:
                    text = clean_text(elem.get_text())
                    if text and len(text) > 20:
                        desc_texts.append(text)
                
                if desc_texts:
                    product_data['description'] = ' '.join(desc_texts)[:2000]  # Limit length
                    break
        
        # Brand extraction
        brand_selectors = [
            '#bylineInfo',
            '.author a',
            '#brand',
            'a[href*="/brand/"]',
            '.po-brand .po-break-word'
        ]
        
        for selector in brand_selectors:
            brand_element = soup.select_one(selector)
            if brand_element:
                brand_text = clean_text(brand_element.get_text())
                # Clean up common prefixes
                brand_text = re.sub(r'^(by |brand:|visit the |store:)', '', brand_text, flags=re.IGNORECASE)
                if brand_text and len(brand_text) < 50:  # Reasonable brand name length
                    product_data['brand'] = brand_text
                    break
        
        # Category/Department
        category_selectors = [
            '#wayfinding-breadcrumbs_feature_div a',
            '.nav-breadcrumb a',
            '#SalesRank .zg_hrsr_ladder a'
        ]
        
        for selector in category_selectors:
            category_elements = soup.select(selector)
            if category_elements and len(category_elements) > 1:
                # Get the most specific category (usually the last one)
                category_text = clean_text(category_elements[-1].get_text())
                if category_text and len(category_text) < 100:
                    product_data['category'] = category_text
                    break
        
        # Ensure we have at least some data
        if not product_data['title'] and not product_data['bullet_points']:
            return {'success': False, 'error': 'Could not extract product information. Page may be blocked or have unusual structure.'}
        
        return product_data
        
    except Exception as e:
        logger.error(f"Scraping error: {e}")
        return {'success': False, 'error': f'Scraping failed: {str(e)}'}

def inject_cyberpunk_css():
    """Inject minimal cyberpunk CSS"""
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&display=swap');
    
    :root {{
        --primary: {COLORS['primary']}; --secondary: {COLORS['secondary']};
        --accent: {COLORS['accent']}; --success: {COLORS['success']};
        --warning: {COLORS['warning']}; --danger: {COLORS['danger']};
        --dark: {COLORS['dark']}; --light: {COLORS['light']};
        --text: {COLORS['text']}; --muted: {COLORS['muted']};
    }}
    
    html, body, .stApp {{
        background: linear-gradient(135deg, var(--dark) 0%, var(--light) 100%);
        color: var(--text); font-family: 'Rajdhani', sans-serif;
    }}
    
    h1, h2, h3 {{ font-family: 'Orbitron', sans-serif; text-transform: uppercase; letter-spacing: 0.1em; }}
    
    h1 {{
        background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(0, 217, 255, 0.4);
    }}
    
    .neon-box {{
        background: rgba(10, 10, 15, 0.9); border: 1px solid var(--primary);
        border-radius: 10px; padding: 1.5rem;
        box-shadow: 0 0 20px rgba(0, 217, 255, 0.4), inset 0 0 20px rgba(0, 217, 255, 0.1);
    }}
    
    .help-box {{
        background: rgba(0, 217, 255, 0.1); border: 1px solid var(--primary);
        border-radius: 10px; padding: 1rem; margin: 1rem 0;
        box-shadow: 0 0 10px rgba(0, 217, 255, 0.2);
    }}
    
    .url-input-box {{
        background: rgba(26, 26, 46, 0.9); border: 2px solid var(--accent);
        border-radius: 15px; padding: 2rem; margin: 1rem 0;
        box-shadow: 0 0 25px rgba(255, 183, 0, 0.3);
    }}
    
    .marketplace-box {{
        background: rgba(255, 0, 110, 0.1); border: 2px solid var(--secondary);
        border-radius: 15px; padding: 2rem; margin: 1rem 0;
        box-shadow: 0 0 25px rgba(255, 0, 110, 0.3);
    }}
    
    .file-upload-status {{
        background: rgba(0, 245, 160, 0.1); border: 1px solid var(--success);
        border-radius: 8px; padding: 0.75rem; margin: 0.5rem 0;
        display: flex; align-items: center; justify-content: space-between;
    }}
    
    .success-box {{
        background: rgba(0, 245, 160, 0.1); border: 1px solid var(--success);
        border-radius: 10px; padding: 1rem;
        box-shadow: 0 0 15px rgba(0, 245, 160, 0.2);
    }}
    
    .error-box {{
        background: rgba(255, 0, 84, 0.1); border: 1px solid var(--danger);
        border-radius: 10px; padding: 1rem;
        box-shadow: 0 0 15px rgba(255, 0, 84, 0.2);
    }}
    
    .stButton > button {{
        font-family: 'Rajdhani', sans-serif; font-weight: 600;
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        color: var(--dark); border: none; padding: 0.75rem 2rem;
        border-radius: 5px; transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 217, 255, 0.4);
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px); box-shadow: 0 6px 25px rgba(0, 217, 255, 0.6);
    }}
    
    .metric-card {{
        background: rgba(26, 26, 46, 0.8); border: 1px solid rgba(0, 217, 255, 0.4);
        border-radius: 10px; padding: 1.5rem; text-align: center;
        transition: all 0.3s ease; cursor: pointer;
    }}
    
    .metric-card:hover {{ transform: translateY(-5px) scale(1.02); }}
    
    .cyber-header {{
        background: linear-gradient(135deg, rgba(0, 217, 255, 0.2) 0%, rgba(255, 0, 110, 0.2) 100%);
        border: 1px solid rgba(0, 217, 255, 0.5); border-radius: 15px;
        padding: 2rem; text-align: center; position: relative; overflow: hidden;
    }}
    
    .chat-message {{ margin: 0.5rem 0; padding: 1rem; border-radius: 10px; }}
    .user-message {{ background: rgba(255, 0, 110, 0.1); border-left: 3px solid var(--secondary); }}
    .ai-message {{ background: rgba(0, 217, 255, 0.1); border-left: 3px solid var(--primary); }}
    
    .priority-high {{ border-left: 4px solid var(--danger); }}
    .priority-medium {{ border-left: 4px solid var(--warning); }}
    .priority-low {{ border-left: 4px solid var(--success); }}
    
    .auto-populated {{ border: 1px solid var(--success); background: rgba(0, 245, 160, 0.05); }}
    
    #MainMenu, footer, header {{ visibility: hidden; }}
    </style>
    """, unsafe_allow_html=True)

def check_ai_status():
    """Check AI availability"""
    if not AI_AVAILABLE:
        return False
    try:
        if st.session_state.ai_analyzer is None:
            st.session_state.ai_analyzer = enhanced_ai_analysis.EnhancedAIAnalyzer()
        status = st.session_state.ai_analyzer.get_api_status()
        return status.get('available', False)
    except Exception as e:
        logger.error(f"Error checking AI status: {e}")
        return False

def display_header():
    """Display header with navigation"""
    st.markdown("""
    <div class="cyber-header">
        <h1 style="font-size: 3em; margin: 0;">VIVE HEALTH REVIEW INTELLIGENCE</h1>
        <p style="color: var(--primary); text-transform: uppercase; letter-spacing: 3px;">
            Advanced Amazon Listing Optimization Engine v9.0
        </p>
        <p style="color: var(--accent); font-size: 0.9em;">✨ Now with Marketplace Data Integration & Return Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick actions bar
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        if st.button("💬 AI CHAT", use_container_width=True, type="primary", help="Discuss analysis results with AI"):
            st.session_state.show_ai_chat = not st.session_state.show_ai_chat
            st.rerun()
    
    with col2:
        if st.button("🔄 New Analysis", use_container_width=True):
            for key in ['uploaded_data', 'analysis_results', 'current_view', 'auto_populated', 'scraping_status', 'marketplace_files', 'marketplace_data']:
                if key == 'current_view':
                    st.session_state[key] = 'upload'
                elif key == 'marketplace_files':
                    st.session_state[key] = {'reimbursements': None, 'fba_returns': None, 'fbm_returns': None}
                else:
                    st.session_state[key] = None if key != 'auto_populated' else False
            st.session_state.show_ai_chat = False
            st.session_state.analyze_all_reviews = True
            # Reset listing details
            st.session_state.listing_details = {
                'title': '', 'bullet_points': ['', '', '', '', ''], 'description': '',
                'backend_keywords': '', 'brand': '', 'category': '', 'asin': '', 'url': ''
            }
            st.rerun()
    
    if st.session_state.uploaded_data:
        with col3:
            view_options = {'metrics': '📊 Metrics', 'ai_results': '🤖 AI Analysis', 'comprehensive': '🎯 Full Report'}
            if not st.session_state.analysis_results:
                view_options.pop('ai_results', None)
                view_options.pop('comprehensive', None)
            
            if st.session_state.current_view != 'upload':
                selected_view = st.selectbox("📍 View", options=list(view_options.keys()),
                                           format_func=lambda x: view_options[x], key='view_selector')
                if selected_view != st.session_state.current_view:
                    st.session_state.current_view = selected_view
                    st.rerun()
    
    with col4:
        st.selectbox("⏱️ Timeframe", options=['all', '30d', '90d', '180d', '365d'],
                    key='selected_timeframe', format_func=lambda x: {
                        'all': 'All Time', '30d': 'Last 30 Days', '90d': 'Last 90 Days',
                        '180d': 'Last 6 Months', '365d': 'Last Year'
                    }[x], help="Filter reviews by date (affects metrics display, not AI analysis by default)")
    
    with col5:
        st.selectbox("⭐ Rating Filter", options=['all', '5', '4', '3', '2', '1', 'positive', 'negative'],
                    key='filter_rating', format_func=lambda x: {
                        'all': 'All Ratings', '5': '5 Stars Only', '4': '4 Stars Only',
                        '3': '3 Stars Only', '2': '2 Stars Only', '1': '1 Star Only',
                        'positive': '4-5 Stars', 'negative': '1-2 Stars'
                    }[x], help="Filter reviews by rating (affects metrics display)")
    
    with col6:
        st.selectbox("🎯 Analysis Depth", options=['quick', 'standard', 'comprehensive'],
                    key='analysis_depth', format_func=lambda x: x.title())

def get_ai_chat_response(user_input: str) -> str:
    """Get AI response for chat"""
    if not check_ai_status():
        return "AI service is currently unavailable."
    
    try:
        # Include context about the current analysis if available
        context = ""
        if st.session_state.analysis_results:
            context = "\n\nContext: I've just completed an analysis of Amazon reviews"
            if st.session_state.use_listing_details and st.session_state.listing_details.get('asin'):
                context += f" for ASIN {st.session_state.listing_details['asin']}"
            context += ". The user may be asking about the analysis results."
        
        if st.session_state.marketplace_data:
            context += "\n\nI also have marketplace data including returns and reimbursements analysis."
        
        system_prompt = f"""You are an expert Amazon listing optimization specialist for medical devices.
        Provide specific, actionable advice for improving Amazon listings, focusing on
        conversion rate optimization and reducing negative reviews. Be concise but comprehensive.
        {context}"""
        
        result = st.session_state.ai_analyzer.api_client.call_api([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ], max_tokens=800, temperature=0.7)
        
        return result['result'] if result['success'] else f"Error: {result.get('error', 'Unknown')}"
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return "Error processing request. Please try again."

def display_ai_chat():
    """Display AI chat interface"""
    st.markdown("""
    <div class="neon-box">
        <h3 style="color: var(--primary);">🤖 AI LISTING OPTIMIZATION ASSISTANT</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Help text
    if not st.session_state.chat_messages:
        st.markdown("""
        <div class="help-box">
            <h4 style="color: var(--primary); margin-top: 0;">💡 How to use AI Chat:</h4>
            <ul style="margin: 0.5rem 0;">
                <li>Ask about specific recommendations from your analysis</li>
                <li>Get help implementing the suggested changes</li>
                <li>Discuss competitor strategies and differentiation</li>
                <li>Request alternative title or bullet point variations</li>
                <li>Ask about medical device compliance considerations</li>
                <li>Inquire about return patterns and how to address them</li>
            </ul>
            <p style="margin-top: 0.5rem; color: var(--accent);">
                <strong>Tip:</strong> After running an analysis, ask me to explain specific recommendations or provide implementation strategies!
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Display messages
    for message in st.session_state.chat_messages:
        msg_class = "user-message" if message["role"] == "user" else "ai-message"
        role = "You" if message["role"] == "user" else "AI Assistant"
        st.markdown(f'<div class="chat-message {msg_class}"><strong>{role}:</strong> {message["content"]}</div>',
                   unsafe_allow_html=True)
    
    # Input
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input("💬 Ask about Amazon listings, analysis results, or optimization strategies...", 
                                   key="chat_input", label_visibility="collapsed")
    with col2:
        send_button = st.button("Send", type="primary", use_container_width=True)
    
    if user_input and send_button:
        st.session_state.chat_messages.extend([
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": get_ai_chat_response(user_input)}
        ])
        st.rerun()
    
    if st.session_state.chat_messages:
        if st.button("🔄 Clear Chat"):
            st.session_state.chat_messages = []
            st.rerun()

def display_url_input_section():
    """Display URL input section for auto-population"""
    st.markdown("""
    <div class="url-input-box">
        <h3 style="color: var(--accent); margin-top: 0;">🔗 AUTO-POPULATE FROM AMAZON URL</h3>
        <p style="color: var(--text); margin-bottom: 1rem;">
            Paste your Amazon product URL below to automatically extract title, bullet points, and other details.
            This enables more targeted AI analysis by comparing your current listing with customer feedback.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # URL input
    amazon_url = st.text_input(
        "Amazon Product URL",
        placeholder="https://www.amazon.com/dp/YOUR-ASIN or https://www.amazon.com/product-name/dp/YOUR-ASIN",
        help="Paste the full Amazon product page URL here. Works with amazon.com, amazon.co.uk, etc.",
        key="amazon_url_input"
    )
    
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col2:
        scrape_button = st.button("🚀 EXTRACT DATA", type="primary", use_container_width=True)
    
    if scrape_button and amazon_url:
        st.session_state.scraping_status = 'processing'
        
        with st.spinner("🔍 Extracting product information from Amazon..."):
            result = scrape_amazon_product(amazon_url)
            
            if result['success']:
                # Update session state with scraped data
                st.session_state.listing_details.update({
                    'title': result.get('title', ''),
                    'description': result.get('description', ''),
                    'brand': result.get('brand', ''),
                    'category': result.get('category', ''),
                    'asin': result.get('asin', ''),
                    'url': result.get('url', '')
                })
                
                # Update bullet points
                bullets = result.get('bullet_points', [])
                for i in range(5):
                    if i < len(bullets):
                        st.session_state.listing_details['bullet_points'][i] = bullets[i]
                    else:
                        st.session_state.listing_details['bullet_points'][i] = ''
                
                st.session_state.scraping_status = 'success'
                st.session_state.auto_populated = True
                st.session_state.use_listing_details = True
                
                st.success("✅ Product data extracted successfully!")
                st.rerun()
                
            else:
                st.session_state.scraping_status = 'error'
                error_msg = result.get('error', 'Unknown error occurred')
                st.error(f"❌ Failed to extract data: {error_msg}")
                
                # Show troubleshooting tips
                st.markdown("""
                <div class="error-box">
                    <h4>Troubleshooting Tips:</h4>
                    <ul>
                        <li>Make sure the URL is a valid Amazon product page</li>
                        <li>Try accessing the URL in your browser first</li>
                        <li>Some products may be restricted or have unusual page structures</li>
                        <li>You can still manually enter the details below</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
    
    elif scrape_button and not amazon_url:
        st.warning("⚠️ Please enter an Amazon URL first")
    
    # Show success message if auto-populated
    if st.session_state.auto_populated and st.session_state.scraping_status == 'success':
        st.markdown("""
        <div class="success-box">
            <h4 style="color: var(--success); margin-top: 0;">🎉 Auto-Population Successful!</h4>
            <p>Product details have been extracted and populated below. You can review and edit them before analysis.</p>
            <p style="color: var(--primary); margin-top: 0.5rem;">
                <strong>AI Analysis Enhancement:</strong> The AI will now compare your current listing with customer feedback to identify specific keyword gaps and optimization opportunities.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display extracted info summary
        details = st.session_state.listing_details
        if details['title'] or details['brand']:
            col1, col2 = st.columns(2)
            with col1:
                if details['title']:
                    st.info(f"**Title:** {details['title'][:100]}{'...' if len(details['title']) > 100 else ''}")
                if details['brand']:
                    st.info(f"**Brand:** {details['brand']}")
            with col2:
                if details['asin']:
                    st.info(f"**ASIN:** {details['asin']}")
                bullet_count = sum(1 for b in details['bullet_points'] if b.strip())
                st.info(f"**Bullet Points:** {bullet_count}/5 extracted")

def display_listing_details_form():
    """Display listing details form with auto-population support"""
    
    # URL input section first
    display_url_input_section()
    
    st.markdown('<hr style="border-color: var(--primary); margin: 2rem 0;">', unsafe_allow_html=True)
    
    st.markdown('<p style="color: #FFB700;">💡 Review and edit the listing details below (auto-populated or manual entry)</p>',
               unsafe_allow_html=True)
    
    if st.checkbox("Include listing details in AI analysis", value=st.session_state.use_listing_details, key="use_details_checkbox"):
        st.session_state.use_listing_details = True
        
        # Product Title
        title_class = "auto-populated" if st.session_state.auto_populated and st.session_state.listing_details['title'] else ""
        title_help = "✨ Auto-populated from URL" if st.session_state.auto_populated and st.session_state.listing_details['title'] else "Enter your product title"
        
        st.text_input("Product Title", 
                     value=st.session_state.listing_details['title'],
                     max_chars=200, 
                     key="listing_title",
                     help=title_help)
        st.session_state.listing_details['title'] = st.session_state.listing_title
        
        # Bullet Points
        st.markdown("**Bullet Points**")
        filled_bullets = 0
        for i in range(5):
            bullet_value = st.session_state.listing_details['bullet_points'][i]
            is_auto = st.session_state.auto_populated and bullet_value.strip()
            help_text = "✨ Auto-populated from URL" if is_auto else f"Enter bullet point {i+1}"
            
            bullet = st.text_input(f"Bullet Point {i+1}",
                                 value=bullet_value,
                                 max_chars=500, 
                                 key=f"bullet_{i}",
                                 help=help_text)
            st.session_state.listing_details['bullet_points'][i] = bullet
            if bullet.strip():
                filled_bullets += 1
        
        # Product Description
        desc_help = "✨ Auto-populated from URL" if st.session_state.auto_populated and st.session_state.listing_details['description'] else "Enter your product description"
        st.text_area("Product Description", 
                    value=st.session_state.listing_details['description'],
                    height=150, 
                    max_chars=2000, 
                    key="listing_description",
                    help=desc_help)
        st.session_state.listing_details['description'] = st.session_state.listing_description
        
        # Two column layout for remaining fields
        col1, col2 = st.columns(2)
        
        with col1:
            backend = st.text_area("Backend Search Terms",
                                 value=st.session_state.listing_details['backend_keywords'],
                                 height=100, 
                                 max_chars=250, 
                                 key="backend_keywords",
                                 help="Enter your backend search terms (manually)")
            st.session_state.listing_details['backend_keywords'] = backend
            
            brand_help = "✨ Auto-populated from URL" if st.session_state.auto_populated and st.session_state.listing_details['brand'] else "Enter your brand name"
            brand = st.text_input("Brand Name", 
                                value=st.session_state.listing_details['brand'],
                                key="brand_name",
                                help=brand_help)
            st.session_state.listing_details['brand'] = brand
        
        with col2:
            category_help = "✨ Auto-populated from URL" if st.session_state.auto_populated and st.session_state.listing_details['category'] else "Enter your product category"
            category = st.text_input("Product Category",
                                   value=st.session_state.listing_details['category'],
                                   key="product_category",
                                   help=category_help)
            st.session_state.listing_details['category'] = category
            
            # ASIN (read-only if auto-populated)
            if st.session_state.listing_details['asin']:
                st.text_input("ASIN", 
                            value=st.session_state.listing_details['asin'],
                            disabled=True,
                            help="✨ Extracted from URL")
            
            # Status display
            if st.session_state.auto_populated:
                st.success(f"✅ Auto-populated: {filled_bullets}/5 bullets, Title: {'✓' if st.session_state.listing_details['title'] else '✗'}")
            else:
                st.info(f"📝 Manual entry: {filled_bullets}/5 bullet points provided")
        
        # Show URL if auto-populated
        if st.session_state.listing_details['url']:
            st.text_input("Source URL", 
                        value=st.session_state.listing_details['url'],
                        disabled=True,
                        help="Amazon URL used for auto-population")
    else:
        st.session_state.use_listing_details = False

def display_marketplace_file_upload():
    """Display marketplace file upload section"""
    st.markdown("""
    <div class="marketplace-box">
        <h3 style="color: var(--secondary); margin-top: 0;">📂 MARKETPLACE DATA FILES (OPTIONAL)</h3>
        <p style="color: var(--text); margin-bottom: 1rem;">
            Upload Amazon marketplace data files to analyze return patterns, reimbursements, and correlate issues with your product.
            Files are automatically detected by their structure.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if not DETECTOR_AVAILABLE:
        st.warning("⚠️ Amazon file detector module not available. Marketplace file analysis will be limited.")
        return
    
    # File upload columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 💰 Reimbursements")
        st.markdown('<small style="color: #666;">18 columns, CSV format</small>', unsafe_allow_html=True)
        reimb_file = st.file_uploader(
            "Upload Reimbursements",
            type=['csv'],
            key="reimb_upload",
            label_visibility="collapsed"
        )
        
        if reimb_file:
            process_marketplace_file(reimb_file, 'reimbursements')
    
    with col2:
        st.markdown("#### 📦 FBA Returns")
        st.markdown('<small style="color: #666;">13 columns, CSV format</small>', unsafe_allow_html=True)
        fba_file = st.file_uploader(
            "Upload FBA Returns",
            type=['csv'],
            key="fba_upload",
            label_visibility="collapsed"
        )
        
        if fba_file:
            process_marketplace_file(fba_file, 'fba_returns')
    
    with col3:
        st.markdown("#### 🚚 FBM Returns")
        st.markdown('<small style="color: #666;">34 columns, TSV/CSV format</small>', unsafe_allow_html=True)
        fbm_file = st.file_uploader(
            "Upload FBM Returns",
            type=['csv', 'tsv', 'txt'],
            key="fbm_upload",
            label_visibility="collapsed"
        )
        
        if fbm_file:
            process_marketplace_file(fbm_file, 'fbm_returns')
    
    # Display status of uploaded files
    if any(st.session_state.marketplace_files.values()):
        st.markdown("---")
        st.markdown("#### 📊 Uploaded Marketplace Files:")
        
        for file_type, data in st.session_state.marketplace_files.items():
            if data:
                file_name = file_type.replace('_', ' ').title()
                st.markdown(f"""
                <div class="file-upload-status">
                    <span>✅ {file_name}: {data['summary']['row_count']} rows</span>
                    <span style="color: var(--primary);">Ready for analysis</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Show key metrics
                if file_type == 'reimbursements' and 'total_cash_reimbursed' in data['summary']:
                    st.info(f"Total reimbursed: ${data['summary']['total_cash_reimbursed']:.2f}")
                elif file_type in ['fba_returns', 'fbm_returns'] and 'total_returns' in data['summary']:
                    st.info(f"Total returns: {data['summary']['total_returns']}")

def process_marketplace_file(uploaded_file, expected_type: str):
    """Process a marketplace file upload"""
    try:
        # Read file content
        file_content = uploaded_file.read()
        
        # Process with detector
        result = AmazonFileDetector.process_file(file_content, uploaded_file.name)
        
        if result['success']:
            detected_type = result['file_type']
            
            # Verify file type matches expected
            if detected_type != expected_type:
                st.warning(f"⚠️ File appears to be {detected_type.replace('_', ' ').title()}, not {expected_type.replace('_', ' ').title()}")
                if not st.checkbox(f"Upload as {expected_type.replace('_', ' ').title()} anyway?", key=f"override_{expected_type}"):
                    return
            
            # Store the processed data
            st.session_state.marketplace_files[expected_type] = {
                'dataframe': result['dataframe'],
                'summary': result['summary'],
                'filename': uploaded_file.name
            }
            
            st.success(f"✅ {expected_type.replace('_', ' ').title()} file processed successfully!")
            
            # Process correlations if we have an ASIN
            if st.session_state.listing_details.get('asin'):
                process_marketplace_correlations()
                
        else:
            st.error(f"❌ Error: {result['error']}")
            
    except Exception as e:
        st.error(f"❌ Failed to process file: {str(e)}")
        logger.error(f"Marketplace file processing error: {e}")

def process_marketplace_correlations():
    """Process correlations between marketplace data and current ASIN"""
    if not st.session_state.listing_details.get('asin'):
        return
    
    target_asin = st.session_state.listing_details['asin']
    
    # Prepare dataframes
    marketplace_dfs = {}
    for file_type, data in st.session_state.marketplace_files.items():
        if data:
            marketplace_dfs[file_type] = data['dataframe']
    
    if marketplace_dfs:
        # Get correlations
        correlations = AmazonFileDetector.correlate_with_asin(marketplace_dfs, target_asin)
        st.session_state.marketplace_data = correlations
        
        # Show immediate insights if data found
        if correlations.get('return_patterns') or correlations.get('financial_impact'):
            st.info(f"🔍 Found marketplace data for ASIN {target_asin}")

# ... [Continue with the rest of the existing functions from PA9.py - they remain the same]
# Including: parse_amazon_date, calculate_basic_stats, analyze_sentiment_patterns, etc.

def parse_amazon_date(date_string):
    """Parse Amazon review dates"""
    try:
        if pd.isna(date_string) or not date_string:
            return None
        date_part = str(date_string).split("on ")[-1] if "on " in str(date_string) else str(date_string)
        for fmt in ['%B %d, %Y', '%b %d, %Y', '%m/%d/%Y', '%Y-%m-%d']:
            try:
                return datetime.strptime(date_part.strip(), fmt).date()
            except:
                continue
        return pd.to_datetime(date_part, errors='coerce').date()
    except:
        return None

def calculate_basic_stats(df):
    """Calculate basic statistics"""
    try:
        ratings = df['Rating'].dropna()
        rating_counts = ratings.value_counts().sort_index().to_dict()
        
        stats = {
            'total_reviews': len(df),
            'average_rating': round(ratings.mean(), 2),
            'rating_distribution': rating_counts,
            'verified_count': sum(df['Verified'] == 'yes') if 'Verified' in df.columns else 0,
            '1_2_star_percentage': round((sum(ratings <= 2) / len(ratings)) * 100, 1) if len(ratings) > 0 else 0,
            '4_5_star_percentage': round((sum(ratings >= 4) / len(ratings)) * 100, 1) if len(ratings) > 0 else 0,
            'median_rating': ratings.median(),
            'rating_std': round(ratings.std(), 2)
        }
        
        if 'Date' in df.columns:
            df['parsed_date'] = df['Date'].apply(parse_amazon_date)
            valid_dates = df['parsed_date'].dropna()
            if len(valid_dates) > 0:
                stats['date_range'] = {
                    'earliest': valid_dates.min(),
                    'latest': valid_dates.max(),
                    'days_covered': (valid_dates.max() - valid_dates.min()).days
                }
        
        return stats
    except Exception as e:
        logger.error(f"Stats calculation error: {e}")
        return None

def analyze_sentiment_patterns(df):
    """Analyze sentiment in reviews"""
    sentiments = {
        'positive_keywords': ['love', 'great', 'excellent', 'perfect', 'amazing', 'best', 'wonderful', 'quality'],
        'negative_keywords': ['hate', 'terrible', 'awful', 'worst', 'horrible', 'poor', 'cheap', 'broken']
    }
    
    results = {'positive': 0, 'negative': 0, 'neutral': 0, 'mixed': 0}
    
    for _, row in df.iterrows():
        if pd.isna(row.get('Body')):
            continue
        text = str(row['Body']).lower()
        pos_count = sum(1 for word in sentiments['positive_keywords'] if word in text)
        neg_count = sum(1 for word in sentiments['negative_keywords'] if word in text)
        
        if pos_count > neg_count:
            results['positive'] += 1
        elif neg_count > pos_count:
            results['negative'] += 1
        elif pos_count == neg_count and pos_count > 0:
            results['mixed'] += 1
        else:
            results['neutral'] += 1
    
    return results

def extract_keywords(df, top_n=20):
    """Extract keywords from reviews"""
    positive_reviews = df[df['Rating'] >= 4]['Body'].dropna()
    negative_reviews = df[df['Rating'] <= 2]['Body'].dropna()
    
    def get_keywords(texts):
        all_words = []
        for text in texts:
            words = re.findall(r'\b[a-z]+\b', str(text).lower())
            all_words.extend(words)
        
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'was', 'it', 'this', 'that', 'have', 'has', 'i', 'my', 'your'}
        filtered_words = [w for w in all_words if w not in stopwords and len(w) > 3]
        return Counter(filtered_words).most_common(top_n)
    
    return {
        'positive_keywords': get_keywords(positive_reviews),
        'negative_keywords': get_keywords(negative_reviews)
    }

def analyze_temporal_trends(df):
    """Analyze rating trends over time"""
    if 'Date' not in df.columns:
        return {}
    
    df['parsed_date'] = df['Date'].apply(parse_amazon_date)
    df_with_dates = df.dropna(subset=['parsed_date'])
    
    if len(df_with_dates) == 0:
        return {}
    
    df_with_dates['month'] = pd.to_datetime(df_with_dates['parsed_date']).dt.to_period('M')
    monthly_avg = df_with_dates.groupby('month')['Rating'].agg(['mean', 'count'])
    monthly_avg.index = monthly_avg.index.astype(str)
    
    if len(monthly_avg) > 1:
        ratings = monthly_avg['mean'].values
        trend = 'improving' if ratings[-1] > ratings[0] else 'declining' if ratings[-1] < ratings[0] else 'stable'
    else:
        trend = 'insufficient_data'
    
    return {
        'trend': trend,
        'monthly_averages': monthly_avg.to_dict(),
        'recent_performance': monthly_avg.tail(3)['mean'].mean() if len(monthly_avg) >= 3 else None
    }

def categorize_issues(df):
    """Categorize issues from negative reviews"""
    categories = {
        'quality': ['quality', 'cheap', 'flimsy', 'broken', 'defect', 'poor'],
        'size_fit': ['size', 'fit', 'small', 'large', 'tight', 'loose'],
        'shipping': ['shipping', 'package', 'delivery', 'damaged', 'late'],
        'functionality': ['work', 'function', 'feature', 'button', 'operate'],
        'value': ['price', 'expensive', 'value', 'worth', 'money'],
        'durability': ['last', 'durable', 'broke', 'wear', 'tear'],
        'instructions': ['instructions', 'manual', 'setup', 'confusing'],
        'customer_service': ['service', 'support', 'response', 'help']
    }
    
    issue_counts = {cat: 0 for cat in categories}
    negative_reviews = df[df['Rating'] <= 3]['Body'].dropna()
    
    for review in negative_reviews:
        review_lower = str(review).lower()
        for category, keywords in categories.items():
            if any(keyword in review_lower for keyword in keywords):
                issue_counts[category] += 1
    
    return issue_counts

def calculate_review_quality(df):
    """Calculate review quality scores"""
    quality_scores = []
    
    for _, row in df.iterrows():
        if pd.isna(row.get('Body')):
            continue
        body = str(row['Body'])
        score = 0
        
        word_count = len(body.split())
        if word_count > 50: score += 3
        elif word_count > 20: score += 2
        elif word_count > 10: score += 1
        
        detail_keywords = ['size', 'color', 'material', 'quality', 'feature']
        score += sum(1 for keyword in detail_keywords if keyword in body.lower())
        
        if any(phrase in body.lower() for phrase in ['pros:', 'cons:', 'update:']):
            score += 2
        
        quality_scores.append(score)
    
    return {
        'avg_quality_score': np.mean(quality_scores) if quality_scores else 0,
        'high_quality_count': sum(1 for s in quality_scores if s >= 5),
        'low_quality_count': sum(1 for s in quality_scores if s <= 2)
    }

def analyze_verification_impact(df):
    """Compare verified vs unverified reviews"""
    if 'Verified' not in df.columns:
        return {}
    
    verified = df[df['Verified'] == 'yes']
    unverified = df[df['Verified'] != 'yes']
    
    return {
        'verified_avg_rating': verified['Rating'].mean() if len(verified) > 0 else None,
        'unverified_avg_rating': unverified['Rating'].mean() if len(unverified) > 0 else None,
        'verified_count': len(verified),
        'unverified_count': len(unverified)
    }

def find_competitor_mentions(df):
    """Find competitor mentions in reviews"""
    patterns = [r'better than\s+\w+', r'compared to\s+\w+', r'switch from\s+\w+']
    mentions = []
    
    for _, row in df.iterrows():
        if pd.isna(row.get('Body')):
            continue
        text = str(row['Body'])
        for pattern in patterns:
            mentions.extend(re.findall(pattern, text, re.IGNORECASE))
    
    return Counter(mentions).most_common(10)

def calculate_listing_health_score(metrics):
    """Calculate overall health score"""
    components = {
        'rating_score': (metrics['basic_stats']['average_rating'] / 5) * 25,
        'review_volume_score': min((metrics['basic_stats']['total_reviews'] / 100) * 15, 15),
        'sentiment_score': (metrics['sentiment_breakdown']['positive'] / sum(metrics['sentiment_breakdown'].values()) * 20) if sum(metrics['sentiment_breakdown'].values()) > 0 else 0,
        'trend_score': 15 if metrics['temporal_trends'].get('trend') == 'improving' else 10 if metrics['temporal_trends'].get('trend') == 'stable' else 5,
        'quality_score': min((metrics['review_quality_scores'].get('avg_quality_score', 0) / 8) * 15, 15),
        'issue_score': max(10 - (sum(metrics['issue_categories'].values()) / metrics['basic_stats']['total_reviews'] * 50), 0)
    }
    
    total_score = sum(components.values())
    
    return {
        'total_score': round(total_score, 1),
        'components': components,
        'grade': 'A' if total_score >= 85 else 'B' if total_score >= 70 else 'C' if total_score >= 55 else 'D' if total_score >= 40 else 'F',
        'status': 'Excellent' if total_score >= 85 else 'Good' if total_score >= 70 else 'Needs Improvement' if total_score >= 55 else 'Poor' if total_score >= 40 else 'Critical'
    }

def calculate_advanced_metrics(df):
    """Calculate all advanced metrics"""
    try:
        metrics = {
            'basic_stats': calculate_basic_stats(df),
            'sentiment_breakdown': analyze_sentiment_patterns(df),
            'keyword_analysis': extract_keywords(df),
            'temporal_trends': analyze_temporal_trends(df),
            'verified_vs_unverified': analyze_verification_impact(df),
            'review_quality_scores': calculate_review_quality(df),
            'issue_categories': categorize_issues(df),
            'competitor_mentions': find_competitor_mentions(df)
        }
        metrics['listing_health_score'] = calculate_listing_health_score(metrics)
        return metrics
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return None

def apply_filters(df):
    """Apply user-selected filters"""
    df_filtered = df.copy()
    
    if st.session_state.selected_timeframe != 'all' and 'Date' in df.columns:
        df_filtered['parsed_date'] = pd.to_datetime(df_filtered['Date'].apply(parse_amazon_date))
        days_map = {'30d': 30, '90d': 90, '180d': 180, '365d': 365}
        if st.session_state.selected_timeframe in days_map:
            cutoff = datetime.now() - timedelta(days=days_map[st.session_state.selected_timeframe])
            df_filtered = df_filtered[df_filtered['parsed_date'] >= cutoff]
    
    if st.session_state.filter_rating != 'all':
        if st.session_state.filter_rating in ['1', '2', '3', '4', '5']:
            df_filtered = df_filtered[df_filtered['Rating'] == int(st.session_state.filter_rating)]
        elif st.session_state.filter_rating == 'positive':
            df_filtered = df_filtered[df_filtered['Rating'] >= 4]
        elif st.session_state.filter_rating == 'negative':
            df_filtered = df_filtered[df_filtered['Rating'] <= 2]
    
    return df_filtered

def prepare_reviews_for_ai(df):
    """Prepare reviews for AI analysis"""
    reviews = []
    
    for idx, row in df.iterrows():
        body = row.get('Body')
        if pd.isna(body) or not str(body).strip():
            continue
        
        rating = int(float(row.get('Rating', 3)))
        rating = max(1, min(5, rating))
        
        review = {
            'id': idx + 1,
            'rating': rating,
            'title': str(row.get('Title', '')).strip()[:200],
            'body': str(body).strip()[:1000],
            'verified': row.get('Verified', '') == 'yes',
            'date': str(row.get('Date', ''))
        }
        reviews.append(review)
    
    reviews.sort(key=lambda x: (x['rating'], x['date']))
    
    # Include more reviews for comprehensive analysis
    if len(reviews) > 100:
        # Balanced sampling across ratings
        low = [r for r in reviews if r['rating'] <= 2][:25]
        mid = [r for r in reviews if r['rating'] == 3][:15]
        high = [r for r in reviews if r['rating'] >= 4][:35]
        reviews = low + mid + high
    
    return reviews

def run_comprehensive_ai_analysis(df, metrics, product_info):
    """Run AI analysis on reviews"""
    if not check_ai_status():
        st.error("AI service is not available.")
        return None
    
    try:
        # Determine which dataframe to use based on user preference
        if st.session_state.analyze_all_reviews and 'df' in st.session_state.uploaded_data:
            # Use ALL reviews for AI analysis
            analysis_df = st.session_state.uploaded_data['df']
            analysis_note = f"Analyzing ALL {len(analysis_df)} reviews"
        else:
            # Use filtered reviews
            analysis_df = df
            analysis_note = f"Analyzing {len(analysis_df)} filtered reviews"
        
        reviews = prepare_reviews_for_ai(analysis_df)
        if not reviews:
            st.error("No reviews to analyze")
            return None
        
        # Add marketplace data note if available
        if st.session_state.marketplace_data:
            analysis_note += " + marketplace data"
        
        st.info(f"🤖 {analysis_note} with AI...")
        
        listing_context = ""
        if st.session_state.use_listing_details:
            details = st.session_state.listing_details
            listing_context = f"""
            CURRENT LISTING (Auto-populated: {'Yes' if st.session_state.auto_populated else 'No'}):
            Title: {details['title'] or 'Not provided'}
            ASIN: {details['asin'] or 'Not provided'}
            Brand: {details['brand'] or 'Not provided'}
            Category: {details['category'] or 'Not provided'}
            Bullet Points: {chr(10).join([f'• {b}' for b in details['bullet_points'] if b.strip()])}
            Description: {details['description'][:500] if details['description'] else 'Not provided'}
            Backend Keywords: {details['backend_keywords'] or 'Not provided'}
            """
        
        # Pass listing details, metrics, and marketplace data to the enhanced analyzer
        result = st.session_state.ai_analyzer.analyze_reviews_for_listing_optimization(
            reviews=reviews,
            product_info=product_info,
            listing_details=st.session_state.listing_details if st.session_state.use_listing_details else None,
            metrics=metrics,
            marketplace_data=st.session_state.marketplace_data
        )
        
        if result:
            return {
                'success': True,
                'analysis': result,
                'timestamp': datetime.now(),
                'reviews_analyzed': len(reviews),
                'total_reviews': len(analysis_df),
                'analysis_scope': 'all_reviews' if st.session_state.analyze_all_reviews else 'filtered_reviews',
                'marketplace_data_included': bool(st.session_state.marketplace_data)
            }
        else:
            st.error("AI analysis failed")
            return None
            
    except Exception as e:
        logger.error(f"AI analysis error: {e}")
        st.error(f"Error: {str(e)}")
        return None

def handle_file_upload():
    """Handle file upload and processing"""
    st.markdown("""
    <div class="neon-box">
        <h2 style="color: var(--primary);">📊 HELIUM 10 DATA IMPORT</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Add AI analysis scope toggle
    col1, col2 = st.columns([3, 1])
    with col1:
        st.checkbox("🎯 Analyze ALL reviews with AI (ignore time/rating filters)", 
                   value=st.session_state.analyze_all_reviews,
                   key="analyze_all_checkbox",
                   help="When checked, AI will analyze all reviews regardless of filters. Unchecked = AI only analyzes filtered reviews.")
    st.session_state.analyze_all_reviews = st.session_state.analyze_all_checkbox
    
    # Tab layout for better organization
    tab1, tab2, tab3 = st.tabs(["📝 Listing Details", "📂 Marketplace Data", "📊 Review Data"])
    
    with tab1:
        display_listing_details_form()
    
    with tab2:
        display_marketplace_file_upload()
    
    with tab3:
        uploaded_file = st.file_uploader("Drop your review file here", type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file:
            try:
                # Read file
                with st.spinner("🔄 Processing data..."):
                    if uploaded_file.name.endswith('.csv'):
                        try:
                            df = pd.read_csv(uploaded_file, encoding='utf-8')
                        except UnicodeDecodeError:
                            uploaded_file.seek(0)
                            df = pd.read_csv(uploaded_file, encoding='latin-1')
                    else:
                        df = pd.read_excel(uploaded_file, engine='openpyxl')
                
                # Validate columns
                required_cols = ['Title', 'Body', 'Rating']
                missing = [col for col in required_cols if col not in df.columns]
                
                if missing:
                    st.error(f"❌ Missing required columns: {', '.join(missing)}")
                    return
                
                # Clean data
                df = df.dropna(subset=['Rating', 'Body'])
                df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
                df = df[df['Rating'].between(1, 5)]
                
                # Apply filters
                df_filtered = apply_filters(df)
                
                if len(df_filtered) == 0:
                    st.warning("⚠️ No reviews match the current filters.")
                    return
                
                # Calculate metrics
                metrics = calculate_advanced_metrics(df_filtered)
                
                if not metrics:
                    st.error("❌ Failed to calculate metrics.")
                    return
                
                # Store data
                st.session_state.uploaded_data = {
                    'df': df,
                    'df_filtered': df_filtered,
                    'product_info': {
                        'asin': st.session_state.listing_details.get('asin') or df['Variation'].iloc[0] if 'Variation' in df.columns else 'Unknown',
                        'total_reviews': len(df),
                        'filtered_reviews': len(df_filtered)
                    },
                    'metrics': metrics
                }
                
                # Display preview metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    score = metrics['listing_health_score']['total_score']
                    color = 'var(--success)' if score >= 70 else 'var(--warning)' if score >= 50 else 'var(--danger)'
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3 style="color: {color};">{score:.0f}</h3>
                        <p>Health Score</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3 style="color: var(--primary);">{metrics['basic_stats']['average_rating']}/5</h3>
                        <p>Avg Rating</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    positive_pct = (metrics['sentiment_breakdown']['positive'] / sum(metrics['sentiment_breakdown'].values()) * 100) if sum(metrics['sentiment_breakdown'].values()) > 0 else 0
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3 style="color: var(--success);">{positive_pct:.0f}%</h3>
                        <p>Positive</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    trend = metrics['temporal_trends'].get('trend', 'stable')
                    trend_icon = '📈' if trend == 'improving' else '📉' if trend == 'declining' else '➡️'
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{trend_icon}</h3>
                        <p>{trend}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show analysis scope info
                if st.session_state.analyze_all_reviews:
                    st.info(f"📊 Metrics shown: {len(df_filtered)} filtered reviews | 🤖 AI will analyze: ALL {len(df)} reviews")
                else:
                    st.info(f"📊 Metrics & AI will analyze: {len(df_filtered)} filtered reviews")
                
                # Show status messages
                status_messages = []
                
                if st.session_state.auto_populated:
                    status_messages.append("✨ Listing details auto-populated from URL")
                
                if st.session_state.marketplace_data:
                    status_messages.append("📂 Marketplace data loaded and correlated")
                
                if status_messages:
                    st.markdown(f"""
                    <div class="success-box">
                        <h4 style="color: var(--success); margin-top: 0;">Ready for Enhanced Analysis</h4>
                        {"<br>".join(status_messages)}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Action buttons
                st.markdown("### 🚀 Choose Your Analysis Path")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📊 VIEW DETAILED METRICS", use_container_width=True):
                        st.session_state.current_view = 'metrics'
                        st.rerun()
                
                with col2:
                    if check_ai_status():
                        button_text = "🚀 RUN AI ANALYSIS"
                        if st.session_state.use_listing_details:
                            button_text += " (Enhanced)"
                        if st.session_state.marketplace_data:
                            button_text += " +"
                            
                        if st.button(button_text, type="primary", use_container_width=True):
                            with st.spinner("🤖 AI analyzing..."):
                                ai_results = run_comprehensive_ai_analysis(df_filtered, metrics, st.session_state.uploaded_data['product_info'])
                                if ai_results:
                                    st.session_state.analysis_results = ai_results
                                    st.session_state.current_view = 'ai_results'
                                    st.rerun()
                    else:
                        st.button("🚀 AI UNAVAILABLE", disabled=True, use_container_width=True)
                
                with col3:
                    if st.session_state.analysis_results:
                        if st.button("🎯 FULL REPORT", use_container_width=True):
                            st.session_state.current_view = 'comprehensive'
                            st.rerun()
                    else:
                        st.info("Run AI Analysis first")
                        
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                logger.error(f"Upload error: {e}", exc_info=True)

def create_visualization_data(df, metrics):
    """Prepare data for visualizations"""
    viz_data = {}
    
    # Rating distribution
    rating_dist = metrics['basic_stats']['rating_distribution']
    viz_data['rating_distribution'] = pd.DataFrame({
        'Stars': [5, 4, 3, 2, 1],
        'Count': [rating_dist.get(i, 0) for i in range(5, 0, -1)]
    })
    
    # Sentiment
    sentiment = metrics['sentiment_breakdown']
    viz_data['sentiment'] = pd.DataFrame(list(sentiment.items()), columns=['Type', 'Count'])
    
    # Issues
    viz_data['issues'] = pd.DataFrame(
        list(metrics['issue_categories'].items()), 
        columns=['Category', 'Count']
    ).sort_values('Count', ascending=False)
    
    # Temporal trend
    if 'monthly_averages' in metrics['temporal_trends']:
        monthly = metrics['temporal_trends']['monthly_averages']
        if monthly and 'mean' in monthly:
            viz_data['trend'] = pd.DataFrame({
                'Month': list(monthly['mean'].keys()),
                'Average Rating': list(monthly['mean'].values()),
                'Review Count': list(monthly['count'].values())
            })
    
    return viz_data

def display_metrics_dashboard(metrics):
    """Display metrics dashboard"""
    viz_data = create_visualization_data(st.session_state.uploaded_data['df_filtered'], metrics)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ⭐ Rating Distribution")
        if 'rating_distribution' in viz_data:
            st.bar_chart(viz_data['rating_distribution'].set_index('Stars'), color='#00D9FF')
        
        st.markdown("#### ⚠️ Issue Categories")
        if 'issues' in viz_data:
            for _, row in viz_data['issues'].head(5).iterrows():
                if row['Count'] > 0:
                    severity_color = '#FF0054' if row['Count'] > 20 else '#FF6B35' if row['Count'] > 10 else '#00F5A0'
                    st.markdown(f"""
                    <div style="background: rgba(10, 10, 15, 0.8); border-left: 3px solid {severity_color}; 
                                padding: 0.5rem; margin: 0.5rem 0;">
                        <strong style="color: {severity_color};">{row['Category'].replace('_', ' ').title()}</strong>: {row['Count']} mentions
                    </div>
                    """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 💭 Sentiment Analysis")
        if 'sentiment' in viz_data:
            for _, row in viz_data['sentiment'].iterrows():
                color = {'Positive': '#00F5A0', 'Negative': '#FF0054', 'Neutral': '#666680', 'Mixed': '#FF6B35'}.get(row['Type'], '#00D9FF')
                percentage = (row['Count'] / viz_data['sentiment']['Count'].sum() * 100)
                st.markdown(f"""
                <div style="margin: 0.5rem 0;">
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: {color};">{row['Type']}</span>
                        <span>{row['Count']} ({percentage:.1f}%)</span>
                    </div>
                    <div style="background: var(--dark); border-radius: 10px; height: 20px;">
                        <div style="background: {color}; width: {percentage}%; height: 100%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        if 'trend' in viz_data and len(viz_data['trend']) > 0:
            st.markdown("#### 📈 Rating Trend")
            st.line_chart(viz_data['trend'].set_index('Month')['Average Rating'], color='#00D9FF')
    
    # Add marketplace data insights if available
    if st.session_state.marketplace_data:
        st.markdown("---")
        st.markdown("### 📂 Marketplace Data Insights")
        
        col1, col2, col3 = st.columns(3)
        
        marketplace = st.session_state.marketplace_data
        
        with col1:
            if 'return_patterns' in marketplace:
                total_returns = sum(
                    data.get('count', 0) 
                    for data in marketplace['return_patterns'].values()
                )
                st.metric("Total Returns", total_returns)
        
        with col2:
            if 'financial_impact' in marketplace and 'reimbursements' in marketplace['financial_impact']:
                total_reimb = marketplace['financial_impact']['reimbursements'].get('total_amount', 0)
                st.metric("Total Reimbursed", f"${total_reimb:.2f}")
        
        with col3:
            if 'related_products' in marketplace:
                related_count = sum(len(asins) for asins in marketplace['related_products'].values())
                st.metric("Related Products", related_count)

def display_ai_insights(analysis):
    """Display AI insights"""
    sections = {
        'TITLE OPTIMIZATION': '🎯',
        'BULLET POINT REWRITE': '📝',
        'A9 ALGORITHM OPTIMIZATION': '🔍',
        'IMMEDIATE QUICK WINS': '⚡',
        'QUALITY & SAFETY PRIORITIES': '🏥',
        'RETURN REDUCTION STRATEGY': '📦'
    }
    
    for section, icon in sections.items():
        if section.upper() in analysis.upper():
            st.markdown(f"""
            <div class="neon-box priority-high">
                <h4 style="color: var(--primary);">{icon} {section}</h4>
            </div>
            """, unsafe_allow_html=True)
            
            start = analysis.upper().find(section.upper())
            end = len(analysis)
            for next_section in sections:
                next_pos = analysis.upper().find(next_section.upper(), start + len(section))
                if next_pos > 0 and next_pos < end:
                    end = next_pos
            
            content = analysis[start + len(section):end].strip()
            st.warning(content)

def display_metrics_view():
    """Display metrics view"""
    if not st.session_state.uploaded_data:
        st.error("No data available")
        return
    
    metrics = st.session_state.uploaded_data['metrics']
    
    st.markdown('<div class="cyber-header"><h1>DETAILED METRICS ANALYSIS</h1></div>', unsafe_allow_html=True)
    
    if not st.session_state.analysis_results:
        if st.button("🚀 RUN AI ANALYSIS NOW", type="primary", use_container_width=True):
            with st.spinner("🤖 AI analyzing..."):
                ai_results = run_comprehensive_ai_analysis(
                    st.session_state.uploaded_data['df_filtered'],
                    metrics,
                    st.session_state.uploaded_data['product_info']
                )
                if ai_results:
                    st.session_state.analysis_results = ai_results
                    st.session_state.current_view = 'ai_results'
                    st.rerun()
    
    display_metrics_dashboard(metrics)
    
    # Keyword analysis
    st.markdown("### 🔍 Keyword Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ✅ Positive Keywords")
        for keyword, count in metrics['keyword_analysis']['positive_keywords'][:10]:
            st.markdown(f'<div style="display: flex; justify-content: space-between;"><span style="color: #00F5A0;">{keyword}</span><span>{count}</span></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### ❌ Negative Keywords")
        for keyword, count in metrics['keyword_analysis']['negative_keywords'][:10]:
            st.markdown(f'<div style="display: flex; justify-content: space-between;"><span style="color: #FF0054;">{keyword}</span><span>{count}</span></div>', unsafe_allow_html=True)

def display_ai_results():
    """Display AI results"""
    if not st.session_state.analysis_results or not st.session_state.uploaded_data:
        st.error("No AI analysis results available")
        return
    
    results = st.session_state.analysis_results
    metrics = st.session_state.uploaded_data['metrics']
    
    enhancement_note = " (Enhanced with Listing Details)" if st.session_state.use_listing_details else ""
    if results.get('marketplace_data_included'):
        enhancement_note += " + Marketplace Data"
    scope_note = f" - Analyzed {'ALL' if results.get('analysis_scope') == 'all_reviews' else 'FILTERED'} reviews"
    
    st.markdown(f"""
    <div class="neon-box">
        <h2 style="color: var(--success);">✅ AI ANALYSIS COMPLETE{enhancement_note}</h2>
        <p>Analyzed {results['reviews_analyzed']} of {results['total_reviews']} total reviews{scope_note}</p>
        <p>{results['timestamp'].strftime('%B %d, %Y at %I:%M %p')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Help box for AI Chat
    st.markdown("""
    <div class="help-box">
        <h4 style="color: var(--primary); margin-top: 0;">💡 Pro Tip: Use AI Chat for Deeper Insights!</h4>
        <p>Click the <strong>💬 AI CHAT</strong> button in the top menu to:</p>
        <ul style="margin: 0.5rem 0;">
            <li>Ask follow-up questions about these recommendations</li>
            <li>Get alternative title or bullet point variations</li>
            <li>Discuss implementation strategies for your specific situation</li>
            <li>Request help with medical device compliance considerations</li>
            <li>Explore return reduction strategies based on the data</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    tabs = st.tabs(["🎯 AI Insights", "📊 Key Metrics", "📂 Marketplace Analysis", "📥 Export"])
    
    with tabs[0]:
        if st.session_state.use_listing_details:
            st.markdown("""
            <div class="success-box">
                <h4 style="color: var(--success); margin-top: 0;">✨ Enhanced Analysis</h4>
                <p>This analysis compared your current listing with customer feedback to identify specific optimization opportunities.</p>
            </div>
            """, unsafe_allow_html=True)
        
        display_ai_insights(results['analysis'])
    
    with tabs[1]:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Average Rating", f"{metrics['basic_stats']['average_rating']}/5")
        with col2:
            st.metric("Health Score", f"{metrics['listing_health_score']['total_score']:.0f}/100")
        with col3:
            positive_pct = (metrics['sentiment_breakdown']['positive'] / sum(metrics['sentiment_breakdown'].values()) * 100)
            st.metric("Positive Sentiment", f"{positive_pct:.0f}%")
        with col4:
            st.metric("Trend", metrics['temporal_trends'].get('trend', 'stable').title())
    
    with tabs[2]:
        if st.session_state.marketplace_data:
            marketplace = st.session_state.marketplace_data
            
            st.markdown("### 📦 Return Analysis")
            if 'return_patterns' in marketplace:
                for return_type, data in marketplace['return_patterns'].items():
                    if data:
                        st.markdown(f"#### {return_type.upper()} Returns")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Count", data.get('count', 0))
                            st.metric("Quantity", data.get('quantity', 0))
                        with col2:
                            if 'reasons' in data and data['reasons']:
                                st.markdown("**Top Reasons:**")
                                for reason, count in list(data['reasons'].items())[:3]:
                                    st.markdown(f"- {reason}: {count}")
            
            st.markdown("### 💰 Financial Impact")
            if 'financial_impact' in marketplace and 'reimbursements' in marketplace['financial_impact']:
                reimb = marketplace['financial_impact']['reimbursements']
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Reimbursement Count", reimb.get('count', 0))
                with col2:
                    st.metric("Total Amount", f"${reimb.get('total_amount', 0):.2f}")
        else:
            st.info("No marketplace data uploaded. Upload files in the Marketplace Data tab for additional insights.")
    
    with tabs[3]:
        # Export options
        enhancement_info = f"\nListing Details Used: {'Yes' if st.session_state.use_listing_details else 'No'}"
        if st.session_state.use_listing_details:
            details = st.session_state.listing_details
            enhancement_info += f"\nASIN: {details.get('asin', 'N/A')}"
            enhancement_info += f"\nBrand: {details.get('brand', 'N/A')}"
            enhancement_info += f"\nAuto-populated: {'Yes' if st.session_state.auto_populated else 'No'}"
        
        if st.session_state.marketplace_data:
            enhancement_info += "\nMarketplace Data: Included"
        
        text_report = f"""
AMAZON LISTING ANALYSIS REPORT
Generated: {datetime.now().strftime('%B %d, %Y')}
{enhancement_info}
Analysis Scope: {'All Reviews' if results.get('analysis_scope') == 'all_reviews' else 'Filtered Reviews'}

EXECUTIVE SUMMARY
Health Score: {metrics['listing_health_score']['total_score']:.0f}/100
Average Rating: {metrics['basic_stats']['average_rating']}/5
Total Reviews: {metrics['basic_stats']['total_reviews']}
Reviews Analyzed by AI: {results['reviews_analyzed']}

AI ANALYSIS
{results['analysis']}
"""
        
        st.download_button(
            "📄 Download Report",
            data=text_report,
            file_name=f"amazon_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )

def display_comprehensive_view():
    """Display comprehensive view"""
    if not st.session_state.uploaded_data or not st.session_state.analysis_results:
        st.error("Both metrics and AI analysis required")
        return
    
    st.markdown('<div class="cyber-header"><h1>COMPREHENSIVE ANALYSIS REPORT</h1></div>', unsafe_allow_html=True)
    
    tabs = st.tabs(["📊 Overview", "📈 Metrics", "🤖 AI Insights", "📂 Marketplace", "🎯 Action Plan"])
    
    with tabs[0]:
        metrics = st.session_state.uploaded_data['metrics']
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="neon-box"><h3>Performance Snapshot</h3></div>', unsafe_allow_html=True)
            st.metric("Health Score", f"{metrics['listing_health_score']['total_score']:.0f}/100")
            st.metric("Average Rating", f"{metrics['basic_stats']['average_rating']}/5")
            st.metric("Total Reviews", metrics['basic_stats']['total_reviews'])
            
            if st.session_state.use_listing_details and st.session_state.listing_details.get('asin'):
                st.metric("ASIN", st.session_state.listing_details['asin'])
        
        with col2:
            st.markdown('<div class="neon-box"><h3>Key Issues</h3></div>', unsafe_allow_html=True)
            for issue, count in sorted(metrics['issue_categories'].items(), key=lambda x: x[1], reverse=True)[:3]:
                if count > 0:
                    st.markdown(f"- {issue.replace('_', ' ').title()}: {count} mentions")
            
            if st.session_state.marketplace_data:
                st.markdown("---")
                st.markdown("**Marketplace Insights:**")
                marketplace = st.session_state.marketplace_data
                if 'return_patterns' in marketplace:
                    total_returns = sum(
                        data.get('count', 0) 
                        for data in marketplace['return_patterns'].values()
                    )
                    st.markdown(f"- Total Returns: {total_returns}")
                if 'financial_impact' in marketplace and 'reimbursements' in marketplace['financial_impact']:
                    reimb_amount = marketplace['financial_impact']['reimbursements'].get('total_amount', 0)
                    st.markdown(f"- Reimbursements: ${reimb_amount:.2f}")
    
    with tabs[1]:
        display_metrics_dashboard(metrics)
    
    with tabs[2]:
        display_ai_insights(st.session_state.analysis_results['analysis'])
    
    with tabs[3]:
        if st.session_state.marketplace_data:
            st.markdown("### 📂 Marketplace Data Analysis")
            
            marketplace = st.session_state.marketplace_data
            
            # Return patterns
            if 'return_patterns' in marketplace:
                st.markdown("#### 📦 Return Patterns")
                for return_type, data in marketplace['return_patterns'].items():
                    if data:
                        with st.expander(f"{return_type.upper()} Returns ({data.get('count', 0)} total)"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Quantity Returned", data.get('quantity', 0))
                                if 'refund_amount' in data:
                                    st.metric("Refund Amount", f"${data['refund_amount']:.2f}")
                            with col2:
                                if 'reasons' in data and data['reasons']:
                                    st.markdown("**Top Return Reasons:**")
                                    for reason, count in list(data['reasons'].items())[:5]:
                                        pct = (count / data.get('count', 1)) * 100
                                        st.markdown(f"- {reason}: {count} ({pct:.1f}%)")
            
            # Financial impact
            if 'financial_impact' in marketplace:
                st.markdown("#### 💰 Financial Impact")
                for impact_type, data in marketplace['financial_impact'].items():
                    if data:
                        st.markdown(f"**{impact_type.title()}:**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Count", data.get('count', 0))
                        with col2:
                            st.metric("Total Amount", f"${data.get('total_amount', 0):.2f}")
        else:
            st.info("No marketplace data uploaded. Consider uploading return and reimbursement files for deeper insights.")
    
    with tabs[4]:
        st.markdown('<div class="neon-box"><h3>🎯 IMPLEMENTATION TIMELINE</h3></div>', unsafe_allow_html=True)
        
        st.markdown("#### Today")
        for task in ["Update title with keywords", "Revise first bullet point", "Respond to negative reviews"]:
            st.checkbox(task)
        
        st.markdown("#### This Week")
        for task in ["Complete bullet revisions", "Update images", "Backend keyword changes", "Address top return reasons"]:
            st.checkbox(task)
        
        st.markdown("#### This Month")
        for task in ["Monitor rating trends", "Implement quality improvements", "Update documentation", "Review competitor changes"]:
            st.checkbox(task)

def generate_comprehensive_excel_report(metrics, ai_results):
    """Generate Excel report"""
    buffer = BytesIO()
    
    if EXCEL_AVAILABLE:
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            # Executive Summary
            summary_data = {
                'Metric': ['Health Score', 'Average Rating', 'Total Reviews', 'Positive Sentiment %'],
                'Value': [
                    f"{metrics['listing_health_score']['total_score']:.0f}/100",
                    f"{metrics['basic_stats']['average_rating']}/5",
                    metrics['basic_stats']['total_reviews'],
                    f"{(metrics['sentiment_breakdown']['positive'] / sum(metrics['sentiment_breakdown'].values()) * 100):.0f}%"
                ]
            }
            
            if st.session_state.use_listing_details:
                summary_data['Metric'].extend(['ASIN', 'Brand', 'Auto-Populated'])
                details = st.session_state.listing_details
                summary_data['Value'].extend([
                    details.get('asin', 'N/A'),
                    details.get('brand', 'N/A'),
                    'Yes' if st.session_state.auto_populated else 'No'
                ])
            
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            # Issues
            issues_data = [(k.replace('_', ' ').title(), v) for k, v in metrics['issue_categories'].items() if v > 0]
            if issues_data:
                pd.DataFrame(issues_data, columns=['Issue', 'Count']).to_excel(writer, sheet_name='Issues', index=False)
            
            # AI Insights
            pd.DataFrame({'AI Analysis': [ai_results['analysis']]}).to_excel(writer, sheet_name='AI Insights', index=False)
            
            # Marketplace data if available
            if st.session_state.marketplace_data:
                marketplace_summary = []
                marketplace = st.session_state.marketplace_data
                
                if 'return_patterns' in marketplace:
                    for return_type, data in marketplace['return_patterns'].items():
                        if data:
                            marketplace_summary.append({
                                'Type': f"{return_type} Returns",
                                'Count': data.get('count', 0),
                                'Quantity': data.get('quantity', 0),
                                'Top Reason': list(data.get('reasons', {}).keys())[0] if data.get('reasons') else 'N/A'
                            })
                
                if marketplace_summary:
                    pd.DataFrame(marketplace_summary).to_excel(writer, sheet_name='Marketplace Data', index=False)
    else:
        # CSV fallback
        csv_data = f"Health Score,{metrics['listing_health_score']['total_score']:.0f}/100\n"
        csv_data += f"Average Rating,{metrics['basic_stats']['average_rating']}/5\n"
        if st.session_state.use_listing_details:
            csv_data += f"ASIN,{st.session_state.listing_details.get('asin', 'N/A')}\n"
        buffer.write(csv_data.encode('utf-8'))
    
    buffer.seek(0)
    return buffer

# Main execution
def main():
    st.set_page_config(
        page_title=APP_CONFIG['title'],
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    initialize_session_state()
    inject_cyberpunk_css()
    display_header()
    
    # Show AI chat if toggled
    if st.session_state.show_ai_chat:
        with st.container():
            display_ai_chat()
            st.markdown("<hr>", unsafe_allow_html=True)
    
    # Main content based on current view
    if st.session_state.current_view == 'upload':
        handle_file_upload()
    elif st.session_state.current_view == 'metrics':
        display_metrics_view()
    elif st.session_state.current_view == 'ai_results':
        display_ai_results()
    elif st.session_state.current_view == 'comprehensive':
        display_comprehensive_view()

if __name__ == "__main__":
    main()
