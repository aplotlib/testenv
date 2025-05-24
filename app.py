"""
Amazon Review Analyzer - Advanced Listing Optimization Engine
Vive Health | Cyberpunk Edition v8.1 - URL Auto-Population
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

# Configuration
APP_CONFIG = {
    'title': 'Vive Health Review Intelligence',
    'version': '8.1',
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
        'scraping_status': None, 'auto_populated': False
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
    
    .url-input-box {{
        background: rgba(26, 26, 46, 0.9); border: 2px solid var(--accent);
        border-radius: 15px; padding: 2rem; margin: 1rem 0;
        box-shadow: 0 0 25px rgba(255, 183, 0, 0.3);
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
            Advanced Amazon Listing Optimization Engine v8.1
        </p>
        <p style="color: var(--accent); font-size: 0.9em;">✨ Now with URL Auto-Population</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick actions bar
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        if st.button("💬 AI CHAT", use_container_width=True, type="primary"):
            st.session_state.show_ai_chat = not st.session_state.show_ai_chat
            st.rerun()
    
    with col2:
        if st.button("🔄 New Analysis", use_container_width=True):
            for key in ['uploaded_data', 'analysis_results', 'current_view', 'auto_populated', 'scraping_status']:
                if key == 'current_view':
                    st.session_state[key] = 'upload'
                else:
                    st.session_state[key] = None if key != 'auto_populated' else False
            st.session_state.show_ai_chat = False
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
                    }[x])
    
    with col5:
        st.selectbox("⭐ Rating Filter", options=['all', '5', '4', '3', '2', '1', 'positive', 'negative'],
                    key='filter_rating', format_func=lambda x: {
                        'all': 'All Ratings', '5': '5 Stars Only', '4': '4 Stars Only',
                        '3': '3 Stars Only', '2': '2 Stars Only', '1': '1 Star Only',
                        'positive': '4-5 Stars', 'negative': '1-2 Stars'
                    }[x])
    
    with col6:
        st.selectbox("🎯 Analysis Depth", options=['quick', 'standard', 'comprehensive'],
                    key='analysis_depth', format_func=lambda x: x.title())

def get_ai_chat_response(user_input: str) -> str:
    """Get AI response for chat"""
    if not check_ai_status():
        return "AI service is currently unavailable."
    
    try:
        system_prompt = """You are an expert Amazon listing optimization specialist.
        Provide specific, actionable advice for improving Amazon listings, focusing on
        conversion rate optimization and reducing negative reviews. Be concise but comprehensive."""
        
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
    
    # Display messages
    for message in st.session_state.chat_messages:
        msg_class = "user-message" if message["role"] == "user" else "ai-message"
        role = "You" if message["role"] == "user" else "AI Assistant"
        st.markdown(f'<div class="chat-message {msg_class}"><strong>{role}:</strong> {message["content"]}</div>',
                   unsafe_allow_html=True)
    
    # Input
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input("💬 Ask about Amazon listings...", key="chat_input", label_visibility="collapsed")
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

# ... [Rest of the existing functions remain the same: parse_amazon_date, calculate_basic_stats, etc.] ...

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
    
    if len(reviews) > 50:
        low = [r for r in reviews if r['rating'] <= 2][:15]
        mid = [r for r in reviews if r['rating'] == 3][:10]
        high = [r for r in reviews if r['rating'] >= 4][:25]
        reviews = low + mid + high
    
    return reviews

def run_comprehensive_ai_analysis(df, metrics, product_info):
    """Run AI analysis on reviews"""
    if not check_ai_status():
        st.error("AI service is not available.")
        return None
    
    try:
        reviews = prepare_reviews_for_ai(df)
        if not reviews:
            st.error("No reviews to analyze")
            return None
        
        st.info(f"🤖 Analyzing {len(reviews)} reviews with AI...")
        
        listing_context = ""
        if st.session_state.use_listing_details:
            details = st.session_state.listing_details
            listing_context = f"""
            CURRENT LISTING:
            Title: {details['title'] or 'Not provided'}
            Bullet Points: {chr(10).join([f'• {b}' for b in details['bullet_points'] if b.strip()])}
            Description: {details['description'][:500] if details['description'] else 'Not provided'}
            Backend Keywords: {details['backend_keywords'] or 'Not provided'}
            Brand: {details['brand'] or 'Not provided'}
            ASIN: {details['asin'] or 'Not provided'}
            """
        
        prompt = f"""
        Analyze these Amazon reviews for LISTING OPTIMIZATION.
        
        Product: {product_info.get('asin', 'Unknown')}
        Total Reviews: {len(reviews)}
        Average Rating: {metrics['basic_stats']['average_rating']}/5
        {listing_context}
        
        METRICS:
        - Positive: {metrics['sentiment_breakdown']['positive']} vs Negative: {metrics['sentiment_breakdown']['negative']}
        - Top Issues: {', '.join([k for k, v in metrics['issue_categories'].items() if v > 5])}
        
        PROVIDE:
        1. **TITLE OPTIMIZATION** - New title (200 chars max) with customer keywords
        2. **BULLET POINT REWRITE** - 5 bullets addressing concerns
        3. **A9 ALGORITHM OPTIMIZATION** - Backend keywords from reviews
        4. **IMMEDIATE QUICK WINS** - Top 3 changes to implement today
        
        Be specific with exact copy to use.
        """
        
        reviews_text = "\n".join([f"[{r['rating']}/5]: {r['body'][:200]}" for r in reviews[:30]])
        prompt += f"\n\nREVIEWS:\n{reviews_text}"
        
        result = st.session_state.ai_analyzer.api_client.call_api([
            {"role": "system", "content": "You are an Amazon listing optimization expert. Provide specific, actionable recommendations."},
            {"role": "user", "content": prompt}
        ], max_tokens=2000, temperature=0.3)
        
        if result['success']:
            return {
                'success': True,
                'analysis': result['result'],
                'timestamp': datetime.now(),
                'reviews_analyzed': len(reviews)
            }
        else:
            st.error(f"AI Error: {result.get('error', 'Unknown')}")
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
    
    with st.expander("🔗 Add Current Listing Details (URL Auto-Population Available)", expanded=not st.session_state.auto_populated):
        display_listing_details_form()
    
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
            
            # Show auto-population status
            if st.session_state.auto_populated:
                st.markdown("""
                <div class="success-box">
                    <h4 style="color: var(--success); margin-top: 0;">✨ Listing Details Auto-Populated</h4>
                    <p>AI analysis will use the extracted listing details for more targeted recommendations.</p>
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
                    button_text = "🚀 RUN AI ANALYSIS" + (" (Enhanced)" if st.session_state.use_listing_details else "")
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

def display_ai_insights(analysis):
    """Display AI insights"""
    sections = {
        'TITLE OPTIMIZATION': '🎯',
        'BULLET POINT REWRITE': '📝',
        'A9 ALGORITHM OPTIMIZATION': '🔍',
        'IMMEDIATE QUICK WINS': '⚡'
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
    
    st.markdown(f"""
    <div class="neon-box">
        <h2 style="color: var(--success);">✅ AI ANALYSIS COMPLETE{enhancement_note}</h2>
        <p>Analyzed {results['reviews_analyzed']} reviews • {results['timestamp'].strftime('%B %d, %Y at %I:%M %p')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    tabs = st.tabs(["🎯 AI Insights", "📊 Key Metrics", "📥 Export"])
    
    with tabs[0]:
        if st.session_state.use_listing_details:
            st.markdown("""
            <div class="success-box">
                <h4 style="color: var(--success); margin-top: 0;">✨ Enhanced Analysis</h4>
                <p>This analysis used your current listing details for more targeted recommendations.</p>
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
        # Export options
        enhancement_info = f"\nListing Details Used: {'Yes' if st.session_state.use_listing_details else 'No'}"
        if st.session_state.use_listing_details:
            details = st.session_state.listing_details
            enhancement_info += f"\nASIN: {details.get('asin', 'N/A')}"
            enhancement_info += f"\nBrand: {details.get('brand', 'N/A')}"
        
        text_report = f"""
AMAZON LISTING ANALYSIS REPORT
Generated: {datetime.now().strftime('%B %d, %Y')}
{enhancement_info}

EXECUTIVE SUMMARY
Health Score: {metrics['listing_health_score']['total_score']:.0f}/100
Average Rating: {metrics['basic_stats']['average_rating']}/5
Total Reviews: {metrics['basic_stats']['total_reviews']}

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
    
    tabs = st.tabs(["📊 Overview", "📈 Metrics", "🤖 AI Insights", "🎯 Action Plan"])
    
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
    
    with tabs[1]:
        display_metrics_dashboard(metrics)
    
    with tabs[2]:
        display_ai_insights(st.session_state.analysis_results['analysis'])
    
    with tabs[3]:
        st.markdown('<div class="neon-box"><h3>🎯 IMPLEMENTATION TIMELINE</h3></div>', unsafe_allow_html=True)
        
        st.markdown("#### Today")
        for task in ["Update title with keywords", "Revise first bullet point", "Respond to negative reviews"]:
            st.checkbox(task)
        
        st.markdown("#### This Week")
        for task in ["Complete bullet revisions", "Update images", "Backend keyword changes"]:
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
