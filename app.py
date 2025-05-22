"""
Amazon Review Analyzer - Advanced Listing Optimization Engine
Vive Health | Cyberpunk Edition v8.0 with AI Chat
AI-powered deep review analysis for Amazon listing managers
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
import io
from typing import Dict, List, Any, Optional, Tuple
import re
from collections import Counter, defaultdict
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import AI modules
try:
    import enhanced_ai_analysis
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    logger.warning("AI module not available - running in limited mode")

# Application configuration
APP_CONFIG = {
    'title': 'Vive Health Review Intelligence',
    'version': '8.0 Cyberpunk AI',
    'description': 'Advanced AI-powered Amazon review analysis with interactive chat',
    'company': 'Vive Health',
    'support_email': 'alexander.popoff@vivehealth.com'
}

# Enhanced Cyberpunk color scheme
COLORS = {
    'primary': '#00D9FF',      # Cyan
    'secondary': '#FF006E',    # Hot pink
    'accent': '#FFB700',       # Gold
    'success': '#00F5A0',      # Neon green
    'warning': '#FF6B35',      # Orange
    'danger': '#FF0054',       # Red
    'dark': '#0A0A0F',         # Deep black
    'light': '#1A1A2E',        # Dark blue
    'text': '#E0E0E0',         # Light gray
    'muted': '#666680',        # Muted purple
    'chat_user': '#2A2A3E',    # User message bg
    'chat_ai': '#1E1E2E',      # AI message bg
    'glass': 'rgba(255,255,255,0.1)'  # Glassmorphism
}

# Chat message types
class MessageRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

@dataclass
class ChatMessage:
    role: MessageRole
    content: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

def initialize_session_state():
    """Initialize session state with advanced features including chat"""
    defaults = {
        'uploaded_data': None,
        'analysis_results': None,
        'current_view': 'upload',
        'processing': False,
        'ai_analyzer': None,
        'competitor_insights': None,
        'keyword_opportunities': None,
        'sentiment_analysis': None,
        'review_clusters': None,
        'quality_issues': None,
        'listing_score': None,
        'improvement_priority': None,
        'selected_timeframe': 'all',
        'filter_rating': 'all',
        'analysis_depth': 'comprehensive',
        'chat_messages': [],
        'chat_context': None,
        'show_chat': False,
        'theme_mode': 'dark',
        'ui_animations': True,
        'last_activity': datetime.now()
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def inject_enhanced_cyberpunk_css():
    """Inject enhanced cyberpunk-themed CSS with animations"""
    st.markdown(f"""
    <style>
    /* Enhanced Cyberpunk Theme with Animations */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&display=swap');
    
    /* Global styles with smooth transitions */
    * {{
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }}
    
    .stApp {{
        background: linear-gradient(135deg, {COLORS['dark']} 0%, {COLORS['light']} 100%);
        color: {COLORS['text']};
        animation: backgroundPulse 20s ease-in-out infinite;
    }}
    
    @keyframes backgroundPulse {{
        0%, 100% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
    }}
    
    /* Headers with enhanced glow */
    h1, h2, h3 {{
        font-family: 'Orbitron', monospace;
        text-transform: uppercase;
        letter-spacing: 2px;
        animation: textGlow 2s ease-in-out infinite alternate;
    }}
    
    @keyframes textGlow {{
        from {{ text-shadow: 0 0 10px {COLORS['primary']}40; }}
        to {{ text-shadow: 0 0 20px {COLORS['primary']}80, 0 0 30px {COLORS['primary']}40; }}
    }}
    
    h1 {{
        background: linear-gradient(90deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-size: 200% auto;
        animation: gradientShift 3s ease-in-out infinite;
    }}
    
    @keyframes gradientShift {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    
    /* Enhanced neon box with glassmorphism */
    .neon-box {{
        background: {COLORS['glass']};
        border: 1px solid {COLORS['primary']};
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 
            0 0 20px {COLORS['primary']}40,
            inset 0 0 20px {COLORS['primary']}10;
        backdrop-filter: blur(20px) saturate(150%);
        -webkit-backdrop-filter: blur(20px) saturate(150%);
        position: relative;
        overflow: hidden;
    }}
    
    .neon-box::before {{
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, {COLORS['primary']}, {COLORS['secondary']}, {COLORS['primary']});
        border-radius: 15px;
        opacity: 0;
        z-index: -1;
        transition: opacity 0.3s ease;
    }}
    
    .neon-box:hover::before {{
        opacity: 1;
        animation: borderRotate 3s linear infinite;
    }}
    
    @keyframes borderRotate {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
    }}
    
    /* Enhanced buttons with ripple effect */
    .stButton > button {{
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
        color: {COLORS['dark']};
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 15px {COLORS['primary']}40;
        transform: translateZ(0);
    }}
    
    .stButton > button::after {{
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.5);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }}
    
    .stButton > button:active::after {{
        width: 300px;
        height: 300px;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px) translateZ(0);
        box-shadow: 
            0 6px 25px {COLORS['primary']}60,
            0 0 30px {COLORS['primary']}40;
    }}
    
    /* Chat interface styles */
    .chat-container {{
        background: {COLORS['glass']};
        border: 1px solid {COLORS['primary']}50;
        border-radius: 20px;
        padding: 1.5rem;
        height: 600px;
        display: flex;
        flex-direction: column;
        backdrop-filter: blur(20px);
        box-shadow: 0 0 40px {COLORS['primary']}20;
    }}
    
    .chat-messages {{
        flex: 1;
        overflow-y: auto;
        padding: 1rem;
        display: flex;
        flex-direction: column;
        gap: 1rem;
        scrollbar-width: thin;
        scrollbar-color: {COLORS['primary']} {COLORS['dark']};
    }}
    
    .chat-message {{
        padding: 1rem 1.5rem;
        border-radius: 15px;
        max-width: 80%;
        animation: messageSlide 0.3s ease-out;
        word-wrap: break-word;
    }}
    
    @keyframes messageSlide {{
        from {{
            opacity: 0;
            transform: translateY(20px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}
    
    .chat-message.user {{
        background: linear-gradient(135deg, {COLORS['chat_user']} 0%, {COLORS['primary']}20 100%);
        align-self: flex-end;
        border: 1px solid {COLORS['primary']}30;
        box-shadow: 0 0 20px {COLORS['primary']}20;
    }}
    
    .chat-message.assistant {{
        background: linear-gradient(135deg, {COLORS['chat_ai']} 0%, {COLORS['secondary']}20 100%);
        align-self: flex-start;
        border: 1px solid {COLORS['secondary']}30;
        box-shadow: 0 0 20px {COLORS['secondary']}20;
    }}
    
    .chat-input-container {{
        display: flex;
        gap: 1rem;
        padding: 1rem;
        border-top: 1px solid {COLORS['primary']}30;
        align-items: center;
    }}
    
    /* Floating action button for chat */
    .chat-fab {{
        position: fixed;
        bottom: 2rem;
        right: 2rem;
        width: 60px;
        height: 60px;
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        box-shadow: 0 4px 20px {COLORS['primary']}60;
        z-index: 1000;
        transition: all 0.3s ease;
        animation: fabPulse 2s ease-in-out infinite;
    }}
    
    @keyframes fabPulse {{
        0%, 100% {{ transform: scale(1); }}
        50% {{ transform: scale(1.05); }}
    }}
    
    .chat-fab:hover {{
        transform: scale(1.1);
        box-shadow: 0 6px 30px {COLORS['primary']}80;
    }}
    
    /* Enhanced metrics cards */
    .metric-card {{
        background: linear-gradient(135deg, {COLORS['glass']} 0%, {COLORS['light']}40 100%);
        border: 1px solid {COLORS['primary']}40;
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 0 20px {COLORS['primary']}20;
        position: relative;
        overflow: hidden;
    }}
    
    .metric-card::before {{
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, {COLORS['primary']}20 0%, transparent 70%);
        opacity: 0;
        transition: opacity 0.3s ease;
    }}
    
    .metric-card:hover {{
        transform: translateY(-5px) scale(1.02);
        box-shadow: 
            0 10px 40px {COLORS['primary']}40,
            0 0 60px {COLORS['primary']}30;
        border-color: {COLORS['secondary']};
    }}
    
    .metric-card:hover::before {{
        opacity: 1;
        animation: ripple 1s ease-out;
    }}
    
    @keyframes ripple {{
        0% {{ transform: scale(0); }}
        100% {{ transform: scale(1); }}
    }}
    
    /* Loading animations */
    .loading-spinner {{
        width: 50px;
        height: 50px;
        border: 3px solid {COLORS['dark']};
        border-top: 3px solid {COLORS['primary']};
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin: 2rem auto;
    }}
    
    @keyframes spin {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
    }}
    
    /* Enhanced file uploader */
    [data-testid="stFileUploader"] {{
        background: {COLORS['glass']};
        padding: 3rem;
        border-radius: 20px;
        border: 2px dashed {COLORS['primary']};
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }}
    
    [data-testid="stFileUploader"]::before {{
        content: '⚡';
        position: absolute;
        font-size: 100px;
        opacity: 0.1;
        right: 20px;
        top: 50%;
        transform: translateY(-50%);
        animation: float 3s ease-in-out infinite;
    }}
    
    @keyframes float {{
        0%, 100% {{ transform: translateY(-50%); }}
        50% {{ transform: translateY(-60%); }}
    }}
    
    [data-testid="stFileUploader"]:hover {{
        border-color: {COLORS['secondary']};
        box-shadow: 0 0 40px {COLORS['secondary']}40;
        background: {COLORS['glass']};
        transform: scale(1.01);
    }}
    
    /* Tooltips */
    .tooltip {{
        position: relative;
        display: inline-block;
    }}
    
    .tooltip .tooltiptext {{
        visibility: hidden;
        width: 200px;
        background-color: {COLORS['dark']}F0;
        color: {COLORS['text']};
        text-align: center;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
        border: 1px solid {COLORS['primary']}50;
        box-shadow: 0 0 20px {COLORS['primary']}40;
        font-size: 0.9rem;
    }}
    
    .tooltip:hover .tooltiptext {{
        visibility: visible;
        opacity: 1;
    }}
    
    /* Progress indicators */
    .stProgress > div > div > div {{
        background: linear-gradient(90deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
        box-shadow: 0 0 20px {COLORS['primary']}60;
        animation: progressPulse 1.5s ease-in-out infinite;
    }}
    
    @keyframes progressPulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.8; }}
    }}
    
    /* Responsive improvements */
    @media (max-width: 768px) {{
        .chat-container {{
            height: 400px;
        }}
        
        .chat-fab {{
            width: 50px;
            height: 50px;
            bottom: 1rem;
            right: 1rem;
        }}
        
        .metric-card {{
            padding: 1rem;
        }}
        
        h1 {{
            font-size: 1.5em !important;
        }}
    }}
    
    /* Smooth scrollbar */
    ::-webkit-scrollbar {{
        width: 12px;
        background: {COLORS['dark']};
    }}
    
    ::-webkit-scrollbar-track {{
        background: {COLORS['dark']};
        border-radius: 10px;
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: linear-gradient(180deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
        border-radius: 10px;
        border: 2px solid {COLORS['dark']};
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: linear-gradient(180deg, {COLORS['secondary']} 0%, {COLORS['primary']} 100%);
    }}
    
    /* Keyboard shortcuts hint */
    .keyboard-hint {{
        position: fixed;
        bottom: 1rem;
        left: 1rem;
        background: {COLORS['dark']}E0;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        border: 1px solid {COLORS['primary']}30;
        font-size: 0.8rem;
        opacity: 0.7;
        transition: opacity 0.3s ease;
    }}
    
    .keyboard-hint:hover {{
        opacity: 1;
    }}
    
    /* Error messages with style */
    .stAlert {{
        background: {COLORS['glass']};
        border-left: 4px solid {COLORS['primary']};
        border-radius: 10px;
        animation: alertSlide 0.3s ease-out;
    }}
    
    @keyframes alertSlide {{
        from {{
            opacity: 0;
            transform: translateX(-20px);
        }}
        to {{
            opacity: 1;
            transform: translateX(0);
        }}
    }}
    
    /* Hide Streamlit branding */
    #MainMenu, footer, header {{
        visibility: hidden;
    }}
    
    /* Custom loader */
    .custom-loader {{
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100px;
    }}
    
    .custom-loader div {{
        width: 20px;
        height: 20px;
        background: {COLORS['primary']};
        border-radius: 50%;
        margin: 0 5px;
        animation: loaderBounce 1.4s infinite ease-in-out both;
    }}
    
    .custom-loader div:nth-child(1) {{ animation-delay: -0.32s; }}
    .custom-loader div:nth-child(2) {{ animation-delay: -0.16s; }}
    
    @keyframes loaderBounce {{
        0%, 80%, 100% {{
            transform: scale(0);
        }}
        40% {{
            transform: scale(1);
        }}
    }}
    </style>
    """, unsafe_allow_html=True)

def display_loading_animation(text="Processing..."):
    """Display custom loading animation"""
    st.markdown(f"""
    <div class="custom-loader">
        <div></div>
        <div></div>
        <div></div>
    </div>
    <p style="text-align: center; color: {COLORS['primary']}; margin-top: 1rem;">{text}</p>
    """, unsafe_allow_html=True)

def display_chat_interface():
    """Display the AI chat interface"""
    st.markdown(f"""
    <div class="neon-box">
        <h3 style="color: {COLORS['primary']}; margin-top: 0;">
            💬 AI Review Assistant
        </h3>
        <p style="opacity: 0.8;">Ask questions about your review data and get instant insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat messages
        st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
        
        for message in st.session_state.chat_messages:
            role_class = "user" if message.role == MessageRole.USER else "assistant"
            st.markdown(f"""
            <div class="chat-message {role_class}">
                <strong>{'You' if message.role == MessageRole.USER else '🤖 AI Assistant'}:</strong><br>
                {message.content}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input
    with st.container():
        col1, col2 = st.columns([5, 1])
        
        with col1:
            user_input = st.text_input(
                "Ask about your reviews...",
                key="chat_input",
                placeholder="e.g., What are the main complaints? How can I improve my listing?",
                label_visibility="collapsed"
            )
        
        with col2:
            send_button = st.button("Send 💬", use_container_width=True, type="primary")
    
    # Process chat input
    if send_button and user_input:
        process_chat_message(user_input)
        st.rerun()
    
    # Suggested questions
    if len(st.session_state.chat_messages) == 0:
        st.markdown("### 💡 Suggested Questions:")
        
        suggestions = [
            "What are the top 3 things customers complain about?",
            "How can I improve my product listing based on reviews?",
            "What keywords should I add to my title?",
            "Compare positive vs negative review themes",
            "What features do customers love most?",
            "Generate a response template for negative reviews"
        ]
        
        cols = st.columns(2)
        for i, suggestion in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(suggestion, key=f"suggest_{i}", use_container_width=True):
                    process_chat_message(suggestion)
                    st.rerun()

def process_chat_message(user_input: str):
    """Process user chat message and generate AI response"""
    # Add user message
    user_message = ChatMessage(role=MessageRole.USER, content=user_input)
    st.session_state.chat_messages.append(user_message)
    
    # Generate AI response
    with st.spinner("🤖 AI is thinking..."):
        response = generate_ai_chat_response(user_input)
    
    # Add AI response
    ai_message = ChatMessage(role=MessageRole.ASSISTANT, content=response)
    st.session_state.chat_messages.append(ai_message)

def generate_ai_chat_response(user_input: str) -> str:
    """Generate AI response based on user input and review data"""
    if not st.session_state.uploaded_data:
        return "Please upload review data first. I'll be happy to help analyze your reviews once you've uploaded them!"
    
    try:
        metrics = st.session_state.uploaded_data['metrics']
        df = st.session_state.uploaded_data['df_filtered']
        
        # Create context for AI
        context = f"""
        You are an expert Amazon listing optimization assistant analyzing review data.
        
        Current Data Summary:
        - Total Reviews: {len(df)}
        - Average Rating: {metrics['basic_stats']['average_rating']}/5
        - Sentiment: {metrics['sentiment_breakdown']['positive']} positive, {metrics['sentiment_breakdown']['negative']} negative
        - Top Issues: {', '.join([k for k, v in metrics['issue_categories'].items() if v > 5][:3])}
        - Trend: {metrics['temporal_trends'].get('trend', 'stable')}
        - Health Score: {metrics['listing_health_score']['total_score']}/100
        
        Previous Analysis Available: {st.session_state.analysis_results is not None}
        
        User Question: {user_input}
        
        Provide a helpful, specific answer based on the review data. Be conversational but professional.
        Include specific examples from reviews when relevant.
        If suggesting improvements, be very specific and actionable.
        """
        
        # For demonstration, create a contextual response
        # In production, this would call the actual AI API
        if AI_AVAILABLE and st.session_state.ai_analyzer:
            result = st.session_state.ai_analyzer.api_client.call_api([
                {"role": "system", "content": "You are a helpful Amazon listing optimization expert. Provide specific, actionable insights based on review data."},
                {"role": "user", "content": context}
            ], max_tokens=500, temperature=0.7)
            
            if result['success']:
                return result['result']
        
        # Fallback responses based on common queries
        return generate_fallback_response(user_input, metrics, df)
        
    except Exception as e:
        logger.error(f"Chat response error: {e}")
        return "I encountered an error analyzing your request. Please try rephrasing your question or check if your data is properly loaded."

def generate_fallback_response(query: str, metrics: Dict, df: pd.DataFrame) -> str:
    """Generate contextual fallback responses"""
    query_lower = query.lower()
    
    # Complaint analysis
    if any(word in query_lower for word in ['complaint', 'problem', 'issue', 'negative']):
        top_issues = sorted(metrics['issue_categories'].items(), key=lambda x: x[1], reverse=True)[:3]
        response = f"Based on {len(df)} reviews, the top customer complaints are:\n\n"
        for i, (issue, count) in enumerate(top_issues, 1):
            percentage = (count / len(df) * 100)
            response += f"{i}. **{issue.replace('_', ' ').title()}** - mentioned in {count} reviews ({percentage:.1f}%)\n"
        response += f"\n💡 **Recommendation**: Focus on addressing {top_issues[0][0].replace('_', ' ')} issues first, as they have the highest impact on customer satisfaction."
        return response
    
    # Keyword suggestions
    elif any(word in query_lower for word in ['keyword', 'title', 'seo', 'search']):
        positive_keywords = metrics['keyword_analysis']['positive_keywords'][:5]
        response = "Here are the top keywords customers use in positive reviews:\n\n"
        for keyword, count in positive_keywords:
            response += f"• **{keyword}** (used {count} times)\n"
        response += "\n💡 **Recommendation**: Add these keywords to your title and bullet points to better match customer language."
        return response
    
    # Improvement suggestions
    elif any(word in query_lower for word in ['improve', 'better', 'enhance', 'optimize']):
        score = metrics['listing_health_score']['total_score']
        grade = metrics['listing_health_score']['grade']
        response = f"Your listing health score is {score}/100 (Grade: {grade}).\n\n"
        response += "Top improvement priorities:\n"
        response += "1. **Address Quality Issues** - Focus on the most mentioned problems\n"
        response += "2. **Update Product Images** - Show features customers ask about\n"
        response += "3. **Revise Bullet Points** - Emphasize benefits customers praise\n"
        response += "4. **Add FAQ Section** - Answer common questions from reviews\n"
        return response
    
    # Positive feedback
    elif any(word in query_lower for word in ['positive', 'love', 'like', 'good']):
        sentiment = metrics['sentiment_breakdown']
        positive_pct = (sentiment['positive'] / sum(sentiment.values()) * 100)
        response = f"{positive_pct:.1f}% of reviews have positive sentiment!\n\n"
        response += "Customers particularly love:\n"
        for keyword, count in metrics['keyword_analysis']['positive_keywords'][:5]:
            response += f"• {keyword}\n"
        response += "\n💡 **Tip**: Highlight these strengths prominently in your listing!"
        return response
    
    # Default response
    else:
        return f"""I can help you analyze your {len(df)} reviews! Try asking:
        
• "What are the main complaints?"
• "What keywords should I use?"
• "How can I improve my listing?"
• "What do customers love?"
• "Compare positive vs negative themes"
• "Generate listing improvement ideas"

What would you like to know?"""

def display_floating_chat_button():
    """Display floating action button for chat"""
    if not st.session_state.show_chat:
        st.markdown(f"""
        <div class="chat-fab" onclick="window.parent.postMessage('toggle_chat', '*')">
            💬
        </div>
        """, unsafe_allow_html=True)

def display_keyboard_shortcuts():
    """Display keyboard shortcuts hint"""
    st.markdown(f"""
    <div class="keyboard-hint">
        <strong>Shortcuts:</strong> 
        <span style="opacity: 0.8;">
        Ctrl+K: Quick Search | 
        Ctrl+/: Help | 
        Ctrl+Enter: Send Chat
        </span>
    </div>
    """, unsafe_allow_html=True)

def display_header():
    """Display enhanced cyberpunk-themed header"""
    st.markdown("""
    <div class="cyber-header">
        <h1 style="font-size: 3em; margin: 0; z-index: 2; position: relative;">
            VIVE HEALTH REVIEW INTELLIGENCE
        </h1>
        <p style="font-family: 'Rajdhani', sans-serif; font-size: 1.2em; margin: 0.5rem 0 0 0; 
                  color: {primary}; text-transform: uppercase; letter-spacing: 3px; z-index: 2; position: relative;">
            Advanced Amazon Listing Optimization Engine with AI Chat
        </p>
    </div>
    """.format(primary=COLORS['primary']), unsafe_allow_html=True)
    
    # Enhanced quick actions bar
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
    
    with col1:
        if st.button("🔄 New Analysis", use_container_width=True, help="Start fresh analysis"):
            for key in st.session_state.keys():
                if key not in ['ai_analyzer']:
                    st.session_state[key] = None
            st.session_state.current_view = 'upload'
            st.rerun()
    
    with col2:
        st.selectbox(
            "⏱️ Timeframe",
            options=['all', '30d', '90d', '180d', '365d'],
            key='selected_timeframe',
            format_func=lambda x: {
                'all': 'All Time',
                '30d': 'Last 30 Days',
                '90d': 'Last 90 Days',
                '180d': 'Last 6 Months',
                '365d': 'Last Year'
            }[x],
            help="Filter reviews by time period"
        )
    
    with col3:
        st.selectbox(
            "⭐ Rating Filter",
            options=['all', '5', '4', '3', '2', '1', 'positive', 'negative'],
            key='filter_rating',
            format_func=lambda x: {
                'all': 'All Ratings',
                '5': '5 Stars Only',
                '4': '4 Stars Only',
                '3': '3 Stars Only',
                '2': '2 Stars Only',
                '1': '1 Star Only',
                'positive': '4-5 Stars',
                'negative': '1-2 Stars'
            }[x],
            help="Filter reviews by rating"
        )
    
    with col4:
        st.selectbox(
            "🎯 Analysis Depth",
            options=['quick', 'standard', 'comprehensive'],
            key='analysis_depth',
            format_func=lambda x: x.title(),
            help="Choose analysis thoroughness"
        )
    
    with col5:
        if st.button("💬 AI Chat", use_container_width=True, type="primary", help="Chat with AI about your reviews"):
            st.session_state.show_chat = not st.session_state.show_chat
            st.rerun()

def handle_file_upload():
    """Enhanced file upload interface with better UX"""
    st.markdown("""
    <div class="neon-box" style="margin-top: 2rem;">
        <h2 style="color: {primary}; margin-top: 0;">📊 HELIUM 10 DATA IMPORT</h2>
        <p style="color: {text}; opacity: 0.8;">Upload your Amazon review export for deep AI analysis</p>
        <div class="tooltip">
            <span style="color: {accent};">ℹ️</span>
            <span class="tooltiptext">Supported formats: CSV, XLSX from Helium 10. Required columns: Title, Body, Rating</span>
        </div>
    </div>
    """.format(primary=COLORS['primary'], text=COLORS['text'], accent=COLORS['accent']), unsafe_allow_html=True)
    
    # Add sample data option
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if st.button("📥 Load Sample Data", help="Try with sample review data"):
            load_sample_data()
            st.rerun()
    
    with col1:
        uploaded_file = st.file_uploader(
            "Drop your review file here or click to browse",
            type=['csv', 'xlsx', 'xls'],
            help="Supported: Helium 10 review exports (CSV/Excel)"
        )
    
    if uploaded_file:
        try:
            # Show upload progress
            progress_container = st.container()
            
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Reading file
                status_text.text("🔄 Reading file...")
                progress_bar.progress(10)
                time.sleep(0.5)  # Smooth animation
                
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                # Step 2: Validating data
                status_text.text("✅ Validating data structure...")
                progress_bar.progress(30)
                
                # Validate columns
                required_cols = ['Title', 'Body', 'Rating']
                missing = [col for col in required_cols if col not in df.columns]
                
                if missing:
                    st.error(f"❌ Missing required columns: {', '.join(missing)}")
                    st.info("💡 Required columns: Title, Body, Rating. Optional: Date, Verified")
                    # Show sample format
                    st.markdown("### Expected Format:")
                    sample_df = pd.DataFrame({
                        'Title': ['Great product!', 'Not satisfied'],
                        'Body': ['Love this item...', 'Quality issues...'],
                        'Rating': [5, 2],
                        'Date': ['2024-01-15', '2024-01-10'],
                        'Verified': ['yes', 'no']
                    })
                    st.dataframe(sample_df, use_container_width=True)
                    return
                
                # Step 3: Apply filters
                status_text.text("🔍 Applying filters...")
                progress_bar.progress(50)
                df_filtered = apply_filters(df)
                
                # Step 4: Calculate metrics
                status_text.text("📊 Computing advanced metrics...")
                progress_bar.progress(70)
                metrics = calculate_advanced_metrics(df_filtered)
                
                if not metrics:
                    st.error("❌ Failed to calculate metrics. Please check your data format.")
                    return
                
                # Step 5: Store data
                status_text.text("💾 Preparing analysis...")
                progress_bar.progress(90)
                
                product_info = {
                    'asin': df['Variation'].iloc[0] if 'Variation' in df.columns else 'Unknown',
                    'total_reviews': len(df),
                    'filtered_reviews': len(df_filtered),
                    'upload_time': datetime.now()
                }
                
                st.session_state.uploaded_data = {
                    'df': df,
                    'df_filtered': df_filtered,
                    'product_info': product_info,
                    'metrics': metrics
                }
                
                # Complete
                status_text.text("✅ Analysis ready!")
                progress_bar.progress(100)
                time.sleep(0.5)
                
                # Clear progress indicators
                progress_container.empty()
            
            # Display key metrics with enhanced styling
            display_upload_summary(metrics, df_filtered)
            
            # Action buttons with better layout
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                if st.button("🚀 RUN AI DEEP ANALYSIS", type="primary", use_container_width=True):
                    with st.spinner(""):
                        display_loading_animation("AI is analyzing your reviews... This may take 1-2 minutes")
                        time.sleep(1)  # Smooth transition
                        ai_results = run_comprehensive_ai_analysis(df_filtered, metrics, product_info)
                        
                        if ai_results:
                            st.session_state.analysis_results = ai_results
                            st.session_state.current_view = 'results'
                            st.success("✅ Analysis complete! Redirecting to results...")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("❌ AI analysis failed. Please try again or contact support.")
            
            with col2:
                if st.button("📊 VIEW DETAILED METRICS", use_container_width=True):
                    st.session_state.current_view = 'metrics'
                    st.rerun()
            
            with col3:
                if st.button("💬 CHAT", use_container_width=True):
                    st.session_state.show_chat = True
                    st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            with st.expander("🔍 Error Details"):
                st.code(str(e))
            st.info(f"💡 Need help? Contact {APP_CONFIG['support_email']}")

def display_upload_summary(metrics: Dict[str, Any], df_filtered: pd.DataFrame):
    """Display enhanced upload summary with animations"""
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Main metrics in cyberpunk cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        score = metrics['listing_health_score']['total_score']
        color = COLORS['success'] if score >= 70 else COLORS['warning'] if score >= 50 else COLORS['danger']
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: {color}; font-size: 2.5em; margin: 0;">{score:.0f}</h3>
            <p style="margin: 0; text-transform: uppercase;">Health Score</p>
            <small style="color: {color};">{metrics['listing_health_score']['status']}</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_rating = metrics['basic_stats']['average_rating']
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: {COLORS['primary']}; font-size: 2.5em; margin: 0;">
                {avg_rating}/5
            </h3>
            <p style="margin: 0; text-transform: uppercase;">Avg Rating</p>
            <small>⭐ x {metrics['basic_stats']['total_reviews']}</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        sentiment = metrics['sentiment_breakdown']
        positive_pct = (sentiment['positive'] / sum(sentiment.values()) * 100) if sum(sentiment.values()) > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: {COLORS['success']}; font-size: 2.5em; margin: 0;">
                {positive_pct:.0f}%
            </h3>
            <p style="margin: 0; text-transform: uppercase;">Positive</p>
            <small>😊 Sentiment</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        trend = metrics['temporal_trends'].get('trend', 'stable')
        trend_icon = '📈' if trend == 'improving' else '📉' if trend == 'declining' else '➡️'
        trend_color = COLORS['success'] if trend == 'improving' else COLORS['danger'] if trend == 'declining' else COLORS['warning']
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="font-size: 2.5em; margin: 0; color: {trend_color};">{trend_icon}</h3>
            <p style="margin: 0; text-transform: uppercase;">{trend}</p>
            <small>Trend</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick insights with icons
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Create insight cards
    insights = []
    
    # Top issue
    top_issue = max(metrics['issue_categories'].items(), key=lambda x: x[1])[0] if metrics['issue_categories'] else 'none'
    insights.append(f"🎯 **Top Issue**: {top_issue.replace('_', ' ').title()}")
    
    # Verified percentage
    verified_pct = (metrics['basic_stats']['verified_count'] / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
    insights.append(f"✅ **Verified**: {verified_pct:.0f}%")
    
    # Review age
    if 'date_range' in metrics['basic_stats']:
        days = metrics['basic_stats']['date_range'].get('days_covered', 0)
        insights.append(f"📅 **Period**: {days} days")
    
    # Keywords found
    keyword_count = len(metrics['keyword_analysis']['positive_keywords'])
    insights.append(f"🔍 **Keywords**: {keyword_count} found")
    
    # Display insights in a styled box
    insight_html = " • ".join(insights)
    st.info(f"💡 **Quick Insights**: {insight_html}")

def load_sample_data():
    """Load sample review data for testing"""
    sample_data = {
        'Title': [
            'Great product, highly recommend!',
            'Not what I expected',
            'Perfect for my needs',
            'Quality issues',
            'Amazing value for money',
            'Disappointed with durability',
            'Excellent customer service',
            'Size runs small',
            'Love the design',
            'Broke after a week'
        ],
        'Body': [
            'This product exceeded my expectations. The quality is outstanding and it works exactly as described. Would definitely buy again!',
            'The product looked different from the pictures. Material feels cheap and flimsy. Not worth the price.',
            'Exactly what I was looking for. Great size, perfect functionality, and arrived quickly. Very satisfied with this purchase.',
            'Started having issues after just a few days. The mechanism is poorly made and broke easily. Very disappointed.',
            'For the price, this is an incredible deal. Works just as well as more expensive alternatives. Highly recommend!',
            'Seemed good at first but didnt last long. After two weeks of normal use, it completely fell apart. Save your money.',
            'Had an issue with my order but customer service resolved it immediately. Great product and even better support!',
            'Be aware that this runs smaller than expected. Had to return and get a larger size. Product itself is good quality though.',
            'Beautiful design and well-made. Gets compliments every time I use it. Worth every penny!',
            'Complete waste of money. Broke within a week of light use. Do not recommend this product to anyone.'
        ],
        'Rating': [5, 2, 5, 1, 5, 2, 4, 3, 5, 1],
        'Date': [
            'January 15, 2024',
            'January 12, 2024',
            'January 10, 2024',
            'January 8, 2024',
            'January 5, 2024',
            'January 3, 2024',
            'December 28, 2023',
            'December 25, 2023',
            'December 20, 2023',
            'December 15, 2023'
        ],
        'Verified': ['yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'no', 'yes'],
        'Variation': ['B08XYZ123'] * 10
    }
    
    df = pd.DataFrame(sample_data)
    
    # Process sample data
    metrics = calculate_advanced_metrics(df)
    
    product_info = {
        'asin': 'B08XYZ123 (Sample)',
        'total_reviews': len(df),
        'filtered_reviews': len(df),
        'upload_time': datetime.now()
    }
    
    st.session_state.uploaded_data = {
        'df': df,
        'df_filtered': df,
        'product_info': product_info,
        'metrics': metrics
    }
    
    st.success("✅ Sample data loaded successfully!")

def display_results():
    """Display comprehensive analysis results with enhanced UI"""
    if not st.session_state.analysis_results or not st.session_state.uploaded_data:
        st.error("No results available. Please upload data and run analysis first.")
        if st.button("Go to Upload"):
            st.session_state.current_view = 'upload'
            st.rerun()
        return
    
    results = st.session_state.analysis_results
    metrics = st.session_state.uploaded_data['metrics']
    
    # Enhanced results header
    st.markdown(f"""
    <div class="neon-box" style="background: linear-gradient(135deg, {COLORS['success']}20 0%, {COLORS['primary']}20 100%);">
        <h2 style="color: {COLORS['success']}; margin: 0;">✅ ANALYSIS COMPLETE</h2>
        <p style="margin: 0.5rem 0 0 0;">
            Analyzed {results['reviews_analyzed']} reviews • 
            Generated {results['timestamp'].strftime('%B %d, %Y at %I:%M %p')}
        </p>
        <div style="margin-top: 1rem;">
            <span style="background: {COLORS['primary']}20; padding: 0.25rem 0.75rem; border-radius: 20px; margin-right: 0.5rem;">
                🧠 AI Powered
            </span>
            <span style="background: {COLORS['secondary']}20; padding: 0.25rem 0.75rem; border-radius: 20px;">
                📊 Data-Driven
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create enhanced tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 AI Insights", 
        "📊 Metrics Dashboard", 
        "💡 Quick Wins", 
        "🏭 Quality Report",
        "💬 AI Chat",
        "📥 Export"
    ])
    
    with tab1:
        display_ai_insights(results['analysis'])
    
    with tab2:
        display_metrics_dashboard(metrics)
    
    with tab3:
        display_quick_wins(results['analysis'])
    
    with tab4:
        display_quality_report(results['analysis'], metrics)
    
    with tab5:
        display_chat_interface()
    
    with tab6:
        display_export_options(results, metrics)

# [Rest of the functions remain the same as in the original code, including:]
# - calculate_advanced_metrics
# - analyze_sentiment_patterns
# - extract_keywords
# - analyze_temporal_trends
# - analyze_verification_impact
# - calculate_review_quality
# - categorize_issues
# - find_competitor_mentions
# - calculate_listing_health_score
# - parse_amazon_date
# - calculate_basic_stats
# - create_visualizations
# - run_comprehensive_ai_analysis
# - prepare_reviews_for_ai
# - check_ai_status
# - display_ai_insights
# - display_metrics_dashboard
# - display_quick_wins
# - display_quality_report
# - display_export_options
# - generate_executive_summary
# - generate_full_report
# - generate_quality_report
# - generate_issue_csv
# - display_metrics_view
# - apply_filters

# Keep all the original analysis functions from the previous code
# I'll just include the main function with the enhanced features

def main():
    """Main application with enhanced cyberpunk theme and AI chat"""
    st.set_page_config(
        page_title=APP_CONFIG['title'],
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Inject enhanced cyberpunk CSS
    inject_enhanced_cyberpunk_css()
    
    # Initialize session state
    initialize_session_state()
    
    # Check AI status
    ai_status = check_ai_status()
    if not ai_status and AI_AVAILABLE:
        st.warning("⚠️ AI Service is initializing. Some features may be limited.")
    
    # Display header
    display_header()
    
    # Display keyboard shortcuts
    display_keyboard_shortcuts()
    
    # Main content based on view
    if st.session_state.show_chat and st.session_state.uploaded_data:
        # Show chat in a modal-like view
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            display_chat_interface()
            if st.button("← Back to Analysis", use_container_width=True):
                st.session_state.show_chat = False
                st.rerun()
    else:
        # Regular views
        if st.session_state.current_view == 'upload':
            handle_file_upload()
        elif st.session_state.current_view == 'results':
            display_results()
        elif st.session_state.current_view == 'metrics':
            display_metrics_view()
    
    # Display floating chat button if data is loaded
    if st.session_state.uploaded_data and not st.session_state.show_chat:
        display_floating_chat_button()
    
    # Enhanced footer
    st.markdown(f"""
    <div style="text-align: center; color: {COLORS['muted']}; padding: 2rem; margin-top: 4rem; 
                border-top: 1px solid {COLORS['muted']}40; background: {COLORS['glass']};">
        <p style="font-family: 'Rajdhani', sans-serif; letter-spacing: 2px; text-transform: uppercase;">
            {APP_CONFIG['title']} • v{APP_CONFIG['version']}
        </p>
        <p style="font-size: 0.9em; opacity: 0.8;">
            🤖 Powered by Advanced AI • 💬 Interactive Chat • 📊 Real-time Analytics
        </p>
        <p style="font-size: 0.8em; opacity: 0.6; margin-top: 0.5rem;">
            Support: {APP_CONFIG['support_email']} • © 2024 {APP_CONFIG['company']}
        </p>
    </div>
    """, unsafe_allow_html=True)

# Copy all the original analysis functions here (calculate_advanced_metrics, etc.)
# They remain exactly the same as in your original code

if __name__ == "__main__":
    main()
