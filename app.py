"""
Amazon Review Analyzer - Advanced Listing Optimization Engine
Vive Health | Cyberpunk Edition v7.0
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
# Using Streamlit's built-in visualization capabilities instead of external libraries

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import AI modules
try:
    import enhanced_ai_analysis
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

# Application configuration
APP_CONFIG = {
    'title': 'Vive Health Review Intelligence',
    'version': '7.0 Cyberpunk',
    'description': 'Advanced AI-powered Amazon review analysis',
    'company': 'Vive Health',
    'support_email': 'alexander.popoff@vivehealth.com'
}

# Cyberpunk color scheme
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
    'muted': '#666680'         # Muted purple
}

def initialize_session_state():
    """Initialize session state with advanced features"""
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
        'analysis_depth': 'comprehensive'
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def inject_cyberpunk_css():
    """Inject cyberpunk-themed CSS"""
    st.markdown(f"""
    <style>
    /* Cyberpunk Theme */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&display=swap');
    
    /* Global styles */
    .stApp {{
        background: linear-gradient(135deg, {COLORS['dark']} 0%, {COLORS['light']} 100%);
        color: {COLORS['text']};
    }}
    
    /* Headers */
    h1, h2, h3 {{
        font-family: 'Orbitron', monospace;
        text-transform: uppercase;
        letter-spacing: 2px;
    }}
    
    h1 {{
        background: linear-gradient(90deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px {COLORS['primary']}40;
    }}
    
    /* Neon glow effect */
    .neon-box {{
        background: {COLORS['dark']}90;
        border: 1px solid {COLORS['primary']};
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 
            0 0 20px {COLORS['primary']}40,
            inset 0 0 20px {COLORS['primary']}10;
        backdrop-filter: blur(10px);
    }}
    
    /* Buttons */
    .stButton > button {{
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
        color: {COLORS['dark']};
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px {COLORS['primary']}40;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 
            0 6px 25px {COLORS['primary']}60,
            0 0 30px {COLORS['primary']}40;
    }}
    
    /* Primary button */
    .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, {COLORS['success']} 0%, {COLORS['primary']} 100%);
        box-shadow: 0 4px 15px {COLORS['success']}40;
    }}
    
    /* Metrics */
    [data-testid="metric-container"] {{
        background: {COLORS['light']}90;
        border: 1px solid {COLORS['primary']}50;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 
            0 0 15px {COLORS['primary']}20,
            inset 0 0 10px {COLORS['primary']}10;
        backdrop-filter: blur(5px);
    }}
    
    [data-testid="metric-container"] [data-testid="metric-value"] {{
        font-family: 'Orbitron', monospace;
        color: {COLORS['primary']};
        text-shadow: 0 0 10px {COLORS['primary']}60;
    }}
    
    /* File uploader */
    [data-testid="stFileUploader"] {{
        background: {COLORS['dark']}80;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed {COLORS['primary']};
        transition: all 0.3s ease;
    }}
    
    [data-testid="stFileUploader"]:hover {{
        border-color: {COLORS['secondary']};
        box-shadow: 0 0 30px {COLORS['secondary']}40;
        background: {COLORS['dark']}95;
    }}
    
    /* Selectbox & inputs */
    .stSelectbox > div > div,
    .stTextInput > div > div > input {{
        background: {COLORS['dark']}80 !important;
        border: 1px solid {COLORS['primary']}50 !important;
        color: {COLORS['text']} !important;
        border-radius: 5px;
    }}
    
    /* Progress bars */
    .stProgress > div > div > div {{
        background: linear-gradient(90deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
        box-shadow: 0 0 20px {COLORS['primary']}60;
    }}
    
    /* Expanders */
    .streamlit-expanderHeader {{
        background: {COLORS['light']}80;
        border: 1px solid {COLORS['primary']}30;
        border-radius: 8px;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
        letter-spacing: 1px;
    }}
    
    /* Info/Warning/Error boxes */
    .stAlert {{
        background: {COLORS['dark']}90;
        border-left: 4px solid {COLORS['primary']};
        border-radius: 5px;
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        background: {COLORS['dark']}80;
        border-radius: 10px;
        padding: 5px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        color: {COLORS['text']};
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
        letter-spacing: 1px;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
        color: {COLORS['dark']};
    }}
    
    /* Custom classes */
    .cyber-header {{
        background: linear-gradient(135deg, {COLORS['primary']}20 0%, {COLORS['secondary']}20 100%);
        border: 1px solid {COLORS['primary']}50;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        box-shadow: 
            0 0 40px {COLORS['primary']}30,
            inset 0 0 40px {COLORS['primary']}10;
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
    }}
    
    .cyber-header::before {{
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: repeating-linear-gradient(
            45deg,
            transparent,
            transparent 10px,
            {COLORS['primary']}10 10px,
            {COLORS['primary']}10 20px
        );
        animation: scan 10s linear infinite;
    }}
    
    @keyframes scan {{
        0% {{ transform: translate(0, 0); }}
        100% {{ transform: translate(50px, 50px); }}
    }}
    
    .metric-card {{
        background: {COLORS['light']}80;
        border: 1px solid {COLORS['primary']}40;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 0 20px {COLORS['primary']}20;
    }}
    
    .metric-card:hover {{
        transform: translateY(-5px);
        box-shadow: 
            0 5px 30px {COLORS['primary']}40,
            0 0 40px {COLORS['primary']}30;
        border-color: {COLORS['secondary']};
    }}
    
    .priority-high {{
        border-left: 4px solid {COLORS['danger']};
        box-shadow: 0 0 20px {COLORS['danger']}40;
    }}
    
    .priority-medium {{
        border-left: 4px solid {COLORS['warning']};
        box-shadow: 0 0 20px {COLORS['warning']}40;
    }}
    
    .priority-low {{
        border-left: 4px solid {COLORS['success']};
        box-shadow: 0 0 20px {COLORS['success']}40;
    }}
    
    /* Scrollbar */
    ::-webkit-scrollbar {{
        width: 10px;
        background: {COLORS['dark']};
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: linear-gradient(180deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
        border-radius: 5px;
    }}
    
    /* Hide Streamlit branding */
    #MainMenu, footer, header {{
        visibility: hidden;
    }}
    </style>
    """, unsafe_allow_html=True)

def calculate_advanced_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate advanced metrics from review data"""
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
        
        # Calculate overall listing health score
        metrics['listing_health_score'] = calculate_listing_health_score(metrics)
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating advanced metrics: {e}")
        return None

def analyze_sentiment_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze sentiment patterns in reviews"""
    sentiments = {
        'positive_keywords': ['love', 'great', 'excellent', 'perfect', 'amazing', 'best', 'wonderful', 'fantastic', 'quality', 'recommend'],
        'negative_keywords': ['hate', 'terrible', 'awful', 'worst', 'horrible', 'poor', 'cheap', 'broken', 'disappointed', 'waste'],
        'neutral_keywords': ['okay', 'fine', 'average', 'decent', 'alright', 'satisfactory']
    }
    
    results = {
        'positive': 0,
        'negative': 0,
        'neutral': 0,
        'mixed': 0
    }
    
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

def extract_keywords(df: pd.DataFrame, top_n: int = 20) -> Dict[str, List[Tuple[str, int]]]:
    """Extract top keywords and phrases from reviews"""
    positive_reviews = df[df['Rating'] >= 4]['Body'].dropna()
    negative_reviews = df[df['Rating'] <= 2]['Body'].dropna()
    
    def get_keywords(texts):
        all_words = []
        for text in texts:
            # Simple keyword extraction - can be enhanced with NLP
            words = re.findall(r'\b[a-z]+\b', str(text).lower())
            all_words.extend(words)
        
        # Filter common words
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'was', 'are', 'were', 'it', 'this', 'that', 'have', 'has', 'had', 'be', 'been', 'being', 'i', 'me', 'my', 'we', 'our', 'you', 'your'}
        filtered_words = [w for w in all_words if w not in stopwords and len(w) > 3]
        
        return Counter(filtered_words).most_common(top_n)
    
    return {
        'positive_keywords': get_keywords(positive_reviews),
        'negative_keywords': get_keywords(negative_reviews)
    }

def analyze_temporal_trends(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze rating trends over time"""
    if 'Date' not in df.columns:
        return {}
    
    df['parsed_date'] = df['Date'].apply(parse_amazon_date)
    df_with_dates = df.dropna(subset=['parsed_date'])
    
    if len(df_with_dates) == 0:
        return {}
    
    # Group by month
    df_with_dates['month'] = pd.to_datetime(df_with_dates['parsed_date']).dt.to_period('M')
    monthly_avg = df_with_dates.groupby('month')['Rating'].agg(['mean', 'count'])
    
    # Detect trend
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

def analyze_verification_impact(df: pd.DataFrame) -> Dict[str, Any]:
    """Compare verified vs unverified reviews"""
    if 'Verified' not in df.columns:
        return {}
    
    verified = df[df['Verified'] == 'yes']
    unverified = df[df['Verified'] != 'yes']
    
    return {
        'verified_avg_rating': verified['Rating'].mean() if len(verified) > 0 else None,
        'unverified_avg_rating': unverified['Rating'].mean() if len(unverified) > 0 else None,
        'verified_count': len(verified),
        'unverified_count': len(unverified),
        'verification_impact': 'positive' if len(verified) > 0 and len(unverified) > 0 and verified['Rating'].mean() > unverified['Rating'].mean() else 'negative'
    }

def calculate_review_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate quality scores for reviews"""
    quality_scores = []
    
    for _, row in df.iterrows():
        if pd.isna(row.get('Body')):
            continue
            
        body = str(row['Body'])
        score = 0
        
        # Length score
        word_count = len(body.split())
        if word_count > 50:
            score += 3
        elif word_count > 20:
            score += 2
        elif word_count > 10:
            score += 1
        
        # Detail score (mentions specific features)
        detail_keywords = ['size', 'color', 'material', 'quality', 'feature', 'function', 'use', 'compare']
        score += sum(1 for keyword in detail_keywords if keyword in body.lower())
        
        # Helpfulness indicators
        if any(phrase in body.lower() for phrase in ['pros:', 'cons:', 'update:', 'edit:']):
            score += 2
        
        quality_scores.append(score)
    
    return {
        'avg_quality_score': np.mean(quality_scores) if quality_scores else 0,
        'high_quality_count': sum(1 for s in quality_scores if s >= 5),
        'low_quality_count': sum(1 for s in quality_scores if s <= 2)
    }

def categorize_issues(df: pd.DataFrame) -> Dict[str, int]:
    """Categorize common issues mentioned in reviews"""
    categories = {
        'quality': ['quality', 'cheap', 'flimsy', 'broken', 'defect', 'poor', 'material'],
        'size_fit': ['size', 'fit', 'small', 'large', 'tight', 'loose', 'measurement'],
        'shipping': ['shipping', 'package', 'delivery', 'damaged', 'late', 'box'],
        'functionality': ['work', 'function', 'feature', 'button', 'mechanism', 'operate'],
        'value': ['price', 'expensive', 'value', 'worth', 'money', 'cost'],
        'durability': ['last', 'durable', 'broke', 'wear', 'tear', 'months', 'weeks'],
        'instructions': ['instructions', 'manual', 'setup', 'install', 'confusing', 'unclear'],
        'customer_service': ['service', 'support', 'response', 'help', 'contact']
    }
    
    issue_counts = {cat: 0 for cat in categories}
    
    negative_reviews = df[df['Rating'] <= 3]['Body'].dropna()
    
    for review in negative_reviews:
        review_lower = str(review).lower()
        for category, keywords in categories.items():
            if any(keyword in review_lower for keyword in keywords):
                issue_counts[category] += 1
    
    return issue_counts

def find_competitor_mentions(df: pd.DataFrame) -> Dict[str, int]:
    """Find mentions of competitors or competitor products"""
    # Common competitor indicators
    competitor_patterns = [
        r'better than\s+\w+',
        r'worse than\s+\w+',
        r'compared to\s+\w+',
        r'unlike\s+\w+',
        r'switch from\s+\w+',
        r'instead of\s+\w+',
        r'[A-Z]\w+\s+brand',
        r'[A-Z]\w+\s+version'
    ]
    
    mentions = []
    
    for _, row in df.iterrows():
        if pd.isna(row.get('Body')):
            continue
            
        text = str(row['Body'])
        for pattern in competitor_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            mentions.extend(matches)
    
    return Counter(mentions).most_common(10)

def calculate_listing_health_score(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate overall listing health score"""
    score_components = {
        'rating_score': 0,
        'review_volume_score': 0,
        'sentiment_score': 0,
        'trend_score': 0,
        'quality_score': 0,
        'issue_score': 0
    }
    
    # Rating score (0-25)
    avg_rating = metrics['basic_stats']['average_rating']
    score_components['rating_score'] = (avg_rating / 5) * 25
    
    # Review volume score (0-15)
    total_reviews = metrics['basic_stats']['total_reviews']
    if total_reviews >= 1000:
        score_components['review_volume_score'] = 15
    elif total_reviews >= 500:
        score_components['review_volume_score'] = 12
    elif total_reviews >= 100:
        score_components['review_volume_score'] = 8
    else:
        score_components['review_volume_score'] = 5
    
    # Sentiment score (0-20)
    sentiment = metrics['sentiment_breakdown']
    total_sentiment = sum(sentiment.values())
    if total_sentiment > 0:
        positive_ratio = sentiment['positive'] / total_sentiment
        score_components['sentiment_score'] = positive_ratio * 20
    
    # Trend score (0-15)
    trend = metrics['temporal_trends'].get('trend', 'stable')
    if trend == 'improving':
        score_components['trend_score'] = 15
    elif trend == 'stable':
        score_components['trend_score'] = 10
    else:
        score_components['trend_score'] = 5
    
    # Quality score (0-15)
    review_quality = metrics['review_quality_scores'].get('avg_quality_score', 0)
    score_components['quality_score'] = min((review_quality / 8) * 15, 15)
    
    # Issue score (0-10) - inverse scoring
    issues = metrics['issue_categories']
    total_issues = sum(issues.values())
    issue_ratio = total_issues / max(metrics['basic_stats']['total_reviews'], 1)
    score_components['issue_score'] = max(10 - (issue_ratio * 50), 0)
    
    total_score = sum(score_components.values())
    
    return {
        'total_score': round(total_score, 1),
        'components': score_components,
        'grade': 'A' if total_score >= 85 else 'B' if total_score >= 70 else 'C' if total_score >= 55 else 'D' if total_score >= 40 else 'F',
        'status': 'Excellent' if total_score >= 85 else 'Good' if total_score >= 70 else 'Needs Improvement' if total_score >= 55 else 'Poor' if total_score >= 40 else 'Critical'
    }

def parse_amazon_date(date_string):
    """Parse Amazon review date formats"""
    try:
        if pd.isna(date_string) or not date_string:
            return None
            
        if "on " in str(date_string):
            date_part = str(date_string).split("on ")[-1]
        else:
            date_part = str(date_string)
        
        # Try common formats
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
        
        # Rating distribution
        rating_counts = ratings.value_counts().sort_index().to_dict()
        
        # Basic metrics
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
        
        # Date range if available
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

def create_visualization_data(df: pd.DataFrame, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare data for Streamlit native visualizations"""
    viz_data = {}
    
    # 1. Rating Distribution Data
    rating_dist = metrics['basic_stats']['rating_distribution']
    viz_data['rating_distribution'] = pd.DataFrame({
        'Stars': [5, 4, 3, 2, 1],
        'Count': [rating_dist.get(5, 0), rating_dist.get(4, 0), rating_dist.get(3, 0), 
                  rating_dist.get(2, 0), rating_dist.get(1, 0)]
    })
    
    # 2. Sentiment Data
    sentiment = metrics['sentiment_breakdown']
    viz_data['sentiment'] = pd.DataFrame({
        'Type': ['Positive', 'Negative', 'Neutral', 'Mixed'],
        'Count': [sentiment['positive'], sentiment['negative'], sentiment['neutral'], sentiment['mixed']]
    })
    
    # 3. Issue Categories Data
    issues = metrics['issue_categories']
    viz_data['issues'] = pd.DataFrame(
        list(issues.items()), 
        columns=['Category', 'Count']
    ).sort_values('Count', ascending=False)
    
    # 4. Temporal Trend Data (if available)
    if 'monthly_averages' in metrics['temporal_trends'] and metrics['temporal_trends']['monthly_averages']:
        monthly_data = metrics['temporal_trends']['monthly_averages']
        months = list(monthly_data['mean'].keys())
        ratings = list(monthly_data['mean'].values())
        counts = list(monthly_data['count'].values())
        
        viz_data['trend'] = pd.DataFrame({
            'Month': [str(m) for m in months],
            'Average Rating': ratings,
            'Review Count': counts
        })
    
    return viz_data

def run_comprehensive_ai_analysis(df: pd.DataFrame, metrics: Dict[str, Any], product_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Run comprehensive AI analysis on reviews"""
    if not check_ai_status():
        return None
    
    try:
        # Prepare review samples
        reviews = prepare_reviews_for_ai(df)
        
        # Create comprehensive analysis prompt
        prompt = f"""
        Analyze these Amazon reviews for ADVANCED LISTING OPTIMIZATION INSIGHTS.
        
        Product: {product_info.get('asin', 'Unknown')}
        Total Reviews: {len(reviews)}
        Average Rating: {metrics['basic_stats']['average_rating']}/5
        
        METRICS SUMMARY:
        - Positive Sentiment: {metrics['sentiment_breakdown']['positive']}
        - Negative Sentiment: {metrics['sentiment_breakdown']['negative']}
        - Top Positive Keywords: {', '.join([k[0] for k in metrics['keyword_analysis']['positive_keywords'][:5]])}
        - Top Negative Keywords: {', '.join([k[0] for k in metrics['keyword_analysis']['negative_keywords'][:5]])}
        - Main Issues: {', '.join([k for k, v in metrics['issue_categories'].items() if v > 5])}
        - Trend: {metrics['temporal_trends'].get('trend', 'unknown')}
        
        PROVIDE THE FOLLOWING ANALYSIS:
        
        1. **LISTING OPTIMIZATION PRIORITIES** (Top 5)
           - Specific changes to title, bullets, description
           - Keywords to add/emphasize
           - Features to highlight
        
        2. **COMPETITIVE POSITIONING**
           - How to differentiate from competitors mentioned
           - Unique value propositions to emphasize
           - Price positioning insights
        
        3. **IMAGE/VIDEO RECOMMENDATIONS**
           - What visual content customers want to see
           - Common misconceptions to address visually
           - A+ content priorities
        
        4. **CUSTOMER OBJECTION HANDLING**
           - Main purchase barriers
           - How to address in listing copy
           - FAQ recommendations
        
        5. **QUALITY IMPROVEMENT PRIORITIES** (For Quality Team)
           - Top 3 product improvements needed
           - Packaging/instruction improvements
           - Quality control focus areas
        
        6. **QUICK WINS** (Can implement immediately)
           - Simple listing tweaks
           - Response templates for reviews
           - Keyword additions
        
        Be extremely specific and actionable. Include exact copy suggestions where relevant.
        Focus on changes that will directly impact conversion rate and reduce returns.
        """
        
        # Add review samples
        review_samples = reviews[:50]  # Limit to prevent token overflow
        reviews_text = "\n".join([
            f"[{r['rating']}/5]: {r['title']} - {r['body'][:200]}"
            for r in review_samples
        ])
        
        prompt += f"\n\nREVIEW SAMPLES:\n{reviews_text}"
        
        # Call AI
        result = st.session_state.ai_analyzer.api_client.call_api([
            {"role": "system", "content": """You are an expert Amazon listing optimization specialist. 
            Provide ultra-specific, actionable recommendations that will immediately improve conversion rates and reduce negative reviews.
            Format your response with clear sections and bullet points."""},
            {"role": "user", "content": prompt}
        ], max_tokens=3000, temperature=0.3)
        
        if result['success']:
            return {
                'success': True,
                'analysis': result['result'],
                'timestamp': datetime.now(),
                'reviews_analyzed': len(reviews)
            }
        
        return None
        
    except Exception as e:
        logger.error(f"AI analysis error: {e}")
        return None

def prepare_reviews_for_ai(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Prepare reviews for AI analysis"""
    reviews = []
    
    for idx, row in df.iterrows():
        if pd.notna(row.get('Body')) and len(str(row['Body']).strip()) > 10:
            review = {
                'id': idx + 1,
                'rating': row.get('Rating', 3),
                'title': str(row.get('Title', '')),
                'body': str(row.get('Body', '')),
                'verified': row.get('Verified', '') == 'yes',
                'date': str(row.get('Date', ''))
            }
            reviews.append(review)
    
    # Sort by most recent and mix of ratings
    reviews.sort(key=lambda x: (x['date'], x['rating']), reverse=True)
    
    return reviews

def check_ai_status():
    """Check AI availability"""
    if not AI_AVAILABLE:
        return False
    
    try:
        if st.session_state.ai_analyzer is None:
            st.session_state.ai_analyzer = enhanced_ai_analysis.EnhancedAIAnalyzer()
        
        status = st.session_state.ai_analyzer.get_api_status()
        return status.get('available', False)
    except:
        return False

def display_header():
    """Display cyberpunk-themed header"""
    st.markdown("""
    <div class="cyber-header">
        <h1 style="font-size: 3em; margin: 0; z-index: 2; position: relative;">
            VIVE HEALTH REVIEW INTELLIGENCE
        </h1>
        <p style="font-family: 'Rajdhani', sans-serif; font-size: 1.2em; margin: 0.5rem 0 0 0; 
                  color: {primary}; text-transform: uppercase; letter-spacing: 3px; z-index: 2; position: relative;">
            Advanced Amazon Listing Optimization Engine
        </p>
    </div>
    """.format(primary=COLORS['primary']), unsafe_allow_html=True)
    
    # Quick actions bar
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col1:
        if st.button("🔄 New Analysis", use_container_width=True):
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
            }[x]
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
            }[x]
        )
    
    with col4:
        st.selectbox(
            "🎯 Analysis Depth",
            options=['quick', 'standard', 'comprehensive'],
            key='analysis_depth',
            format_func=lambda x: x.title()
        )

def handle_file_upload():
    """Cyberpunk-themed file upload interface"""
    st.markdown("""
    <div class="neon-box" style="margin-top: 2rem;">
        <h2 style="color: {primary}; margin-top: 0;">📊 HELIUM 10 DATA IMPORT</h2>
        <p style="color: {text}; opacity: 0.8;">Upload your Amazon review export for deep analysis</p>
    </div>
    """.format(primary=COLORS['primary'], text=COLORS['text']), unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Drop your review file here",
        type=['csv', 'xlsx', 'xls'],
        help="Supported: Helium 10 review exports (CSV/Excel)"
    )
    
    if uploaded_file:
        try:
            # Read file with progress
            with st.spinner("🔄 Initializing data matrix..."):
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
            
            # Validate columns
            required_cols = ['Title', 'Body', 'Rating']
            missing = [col for col in required_cols if col not in df.columns]
            
            if missing:
                st.error(f"❌ Missing required columns: {', '.join(missing)}")
                st.info("Required: Title, Body, Rating, Date (optional), Verified (optional)")
                return
            
            # Apply filters
            df_filtered = apply_filters(df)
            
            # Process with progress indicator
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Basic processing
            status_text.text("🔍 Analyzing review patterns...")
            progress_bar.progress(20)
            
            # Calculate metrics
            status_text.text("📊 Computing advanced metrics...")
            metrics = calculate_advanced_metrics(df_filtered)
            progress_bar.progress(40)
            
            if not metrics:
                st.error("❌ Failed to calculate metrics")
                return
            
            # Store data
            product_info = {
                'asin': df['Variation'].iloc[0] if 'Variation' in df.columns else 'Unknown',
                'total_reviews': len(df),
                'filtered_reviews': len(df_filtered)
            }
            
            st.session_state.uploaded_data = {
                'df': df,
                'df_filtered': df_filtered,
                'product_info': product_info,
                'metrics': metrics
            }
            
            # Show preview metrics
            status_text.text("✅ Analysis ready!")
            progress_bar.progress(100)
            
            # Display key metrics in cyberpunk style
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                score = metrics['listing_health_score']['total_score']
                color = COLORS['success'] if score >= 70 else COLORS['warning'] if score >= 50 else COLORS['danger']
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: {color}; font-size: 2.5em; margin: 0;">{score:.0f}</h3>
                    <p style="margin: 0; text-transform: uppercase;">Health Score</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: {COLORS['primary']}; font-size: 2.5em; margin: 0;">
                        {metrics['basic_stats']['average_rating']}/5
                    </h3>
                    <p style="margin: 0; text-transform: uppercase;">Avg Rating</p>
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
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                trend = metrics['temporal_trends'].get('trend', 'stable')
                trend_icon = '📈' if trend == 'improving' else '📉' if trend == 'declining' else '➡️'
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="font-size: 2.5em; margin: 0;">{trend_icon}</h3>
                    <p style="margin: 0; text-transform: uppercase;">{trend}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Quick insights
            st.markdown("<br>", unsafe_allow_html=True)
            st.info(f"""
            🎯 **Quick Insights**: Analyzed {len(df_filtered)} reviews 
            • Top issue: {max(metrics['issue_categories'].items(), key=lambda x: x[1])[0].replace('_', ' ').title()} 
            • Verified reviews: {metrics['basic_stats']['verified_count']} 
            • Date range: {metrics['basic_stats'].get('date_range', {}).get('days_covered', 'N/A')} days
            """)
            
            # Action buttons
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🚀 RUN AI DEEP ANALYSIS", type="primary", use_container_width=True):
                    with st.spinner("🤖 AI analyzing reviews... This may take 1-2 minutes"):
                        ai_results = run_comprehensive_ai_analysis(df_filtered, metrics, product_info)
                        
                        if ai_results:
                            st.session_state.analysis_results = ai_results
                            st.session_state.current_view = 'results'
                            st.rerun()
                        else:
                            st.error("❌ AI analysis failed. Please try again.")
            
            with col2:
                if st.button("📊 VIEW DETAILED METRICS", use_container_width=True):
                    st.session_state.current_view = 'metrics'
                    st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info(f"Need help? Contact {APP_CONFIG['support_email']}")

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply selected filters to dataframe"""
    df_filtered = df.copy()
    
    # Time filter
    if st.session_state.selected_timeframe != 'all' and 'Date' in df.columns:
        df_filtered['parsed_date'] = df_filtered['Date'].apply(parse_amazon_date)
        df_filtered['parsed_date'] = pd.to_datetime(df_filtered['parsed_date'])
        
        days_map = {'30d': 30, '90d': 90, '180d': 180, '365d': 365}
        if st.session_state.selected_timeframe in days_map:
            cutoff_date = datetime.now() - timedelta(days=days_map[st.session_state.selected_timeframe])
            df_filtered = df_filtered[df_filtered['parsed_date'] >= cutoff_date]
    
    # Rating filter
    if st.session_state.filter_rating != 'all':
        if st.session_state.filter_rating in ['1', '2', '3', '4', '5']:
            df_filtered = df_filtered[df_filtered['Rating'] == int(st.session_state.filter_rating)]
        elif st.session_state.filter_rating == 'positive':
            df_filtered = df_filtered[df_filtered['Rating'] >= 4]
        elif st.session_state.filter_rating == 'negative':
            df_filtered = df_filtered[df_filtered['Rating'] <= 2]
    
    return df_filtered

def display_results():
    """Display comprehensive analysis results"""
    if not st.session_state.analysis_results or not st.session_state.uploaded_data:
        st.error("No results available")
        return
    
    results = st.session_state.analysis_results
    metrics = st.session_state.uploaded_data['metrics']
    
    # Results header
    st.markdown(f"""
    <div class="neon-box" style="background: linear-gradient(135deg, {COLORS['success']}20 0%, {COLORS['primary']}20 100%);">
        <h2 style="color: {COLORS['success']}; margin: 0;">✅ ANALYSIS COMPLETE</h2>
        <p style="margin: 0.5rem 0 0 0;">
            Analyzed {results['reviews_analyzed']} reviews • 
            Generated {results['timestamp'].strftime('%B %d, %Y at %I:%M %p')}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different insights
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 AI Insights", 
        "📊 Metrics Dashboard", 
        "💡 Quick Wins", 
        "🏭 Quality Report",
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
        display_export_options(results, metrics)

def display_ai_insights(analysis: str):
    """Display AI insights in structured format"""
    st.markdown(f"""
    <div class="neon-box">
        <h3 style="color: {COLORS['primary']};">🤖 AI LISTING OPTIMIZATION ANALYSIS</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Parse and display sections
    sections = {
        'LISTING OPTIMIZATION PRIORITIES': {'icon': '🎯', 'color': COLORS['primary'], 'priority': 'high'},
        'COMPETITIVE POSITIONING': {'icon': '🏆', 'color': COLORS['accent'], 'priority': 'high'},
        'IMAGE/VIDEO RECOMMENDATIONS': {'icon': '📸', 'color': COLORS['secondary'], 'priority': 'medium'},
        'CUSTOMER OBJECTION HANDLING': {'icon': '🛡️', 'color': COLORS['warning'], 'priority': 'high'},
        'QUALITY IMPROVEMENT PRIORITIES': {'icon': '🏭', 'color': COLORS['danger'], 'priority': 'high'},
        'QUICK WINS': {'icon': '⚡', 'color': COLORS['success'], 'priority': 'medium'}
    }
    
    for section, config in sections.items():
        if section.upper() in analysis.upper():
            priority_class = f"priority-{config['priority']}"
            
            st.markdown(f"""
            <div class="neon-box {priority_class}" style="margin-top: 1rem;">
                <h4 style="color: {config['color']}; margin-top: 0;">
                    {config['icon']} {section}
                </h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Extract section content
            start = analysis.upper().find(section.upper())
            if start != -1:
                end = len(analysis)
                for next_section in sections:
                    next_pos = analysis.upper().find(next_section.upper(), start + len(section))
                    if next_pos > 0 and next_pos < end:
                        end = next_pos
                
                content = analysis[start + len(section):end].strip()
                
                # Format content based on priority
                if config['priority'] == 'high':
                    st.warning(content)
                else:
                    st.info(content)

def display_metrics_dashboard(metrics: Dict[str, Any]):
    """Display interactive metrics dashboard with native Streamlit charts"""
    st.markdown(f"""
    <div class="neon-box">
        <h3 style="color: {COLORS['primary']};">📊 PERFORMANCE METRICS DASHBOARD</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Get visualization data
    viz_data = create_visualization_data(st.session_state.uploaded_data['df_filtered'], metrics)
    
    # Display charts in grid
    col1, col2 = st.columns(2)
    
    with col1:
        # Rating Distribution Bar Chart
        st.markdown("#### ⭐ Rating Distribution")
        if 'rating_distribution' in viz_data:
            st.bar_chart(
                viz_data['rating_distribution'].set_index('Stars'),
                color=COLORS['primary']
            )
        
        # Issue Categories
        st.markdown("#### ⚠️ Issue Categories")
        if 'issues' in viz_data and not viz_data['issues'].empty:
            # Display as styled metrics instead of chart
            for _, row in viz_data['issues'].head(5).iterrows():
                if row['Count'] > 0:
                    severity_color = COLORS['danger'] if row['Count'] > 20 else COLORS['warning'] if row['Count'] > 10 else COLORS['success']
                    st.markdown(f"""
                    <div style="background: {COLORS['dark']}80; border-left: 3px solid {severity_color}; 
                                padding: 0.5rem; margin: 0.5rem 0; border-radius: 5px;">
                        <strong style="color: {severity_color};">{row['Category'].replace('_', ' ').title()}</strong>: {row['Count']} mentions
                    </div>
                    """, unsafe_allow_html=True)
    
    with col2:
        # Sentiment Analysis
        st.markdown("#### 💭 Sentiment Analysis")
        if 'sentiment' in viz_data:
            # Create custom colored bar display
            sentiment_data = viz_data['sentiment']
            total = sentiment_data['Count'].sum()
            
            for _, row in sentiment_data.iterrows():
                percentage = (row['Count'] / total * 100) if total > 0 else 0
                color = {
                    'Positive': COLORS['success'],
                    'Negative': COLORS['danger'],
                    'Neutral': COLORS['muted'],
                    'Mixed': COLORS['warning']
                }.get(row['Type'], COLORS['primary'])
                
                st.markdown(f"""
                <div style="margin: 0.5rem 0;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                        <span style="color: {color};">{row['Type']}</span>
                        <span style="color: {COLORS['text']};">{row['Count']} ({percentage:.1f}%)</span>
                    </div>
                    <div style="background: {COLORS['dark']}; border-radius: 10px; height: 20px; overflow: hidden;">
                        <div style="background: {color}; width: {percentage}%; height: 100%; 
                                    box-shadow: 0 0 10px {color}60; transition: width 0.5s ease;">
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Temporal Trend (if available)
        if 'trend' in viz_data:
            st.markdown("#### 📈 Rating Trend")
            trend_data = viz_data['trend']
            
            # Display mini metrics for trend
            if len(trend_data) >= 2:
                recent_avg = trend_data['Average Rating'].iloc[-1]
                previous_avg = trend_data['Average Rating'].iloc[-2]
                trend_direction = "↑" if recent_avg > previous_avg else "↓" if recent_avg < previous_avg else "→"
                trend_color = COLORS['success'] if recent_avg > previous_avg else COLORS['danger'] if recent_avg < previous_avg else COLORS['warning']
                
                st.markdown(f"""
                <div style="background: {COLORS['dark']}90; border: 1px solid {trend_color}50; 
                            padding: 1rem; border-radius: 10px; text-align: center;">
                    <h2 style="color: {trend_color}; margin: 0;">
                        {recent_avg:.2f} {trend_direction}
                    </h2>
                    <p style="margin: 0; opacity: 0.8;">Latest Month Average</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Simple line chart
                st.line_chart(
                    trend_data.set_index('Month')['Average Rating'],
                    color=COLORS['primary']
                )
    
    # Key metrics cards
    st.markdown("### 🎯 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    health_score = metrics['listing_health_score']
    
    with col1:
        color = COLORS['success'] if health_score['grade'] in ['A', 'B'] else COLORS['warning'] if health_score['grade'] == 'C' else COLORS['danger']
        st.markdown(f"""
        <div class="metric-card" style="border-color: {color};">
            <h2 style="color: {color}; margin: 0;">{health_score['grade']}</h2>
            <p style="margin: 0;">Overall Grade</p>
            <small>{health_score['status']}</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        keyword_count = len(metrics['keyword_analysis']['positive_keywords'])
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="color: {COLORS['primary']}; margin: 0;">{keyword_count}</h2>
            <p style="margin: 0;">Keywords Found</p>
            <small>Opportunity keywords</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        issue_count = sum(1 for v in metrics['issue_categories'].values() if v > 5)
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="color: {COLORS['warning']}; margin: 0;">{issue_count}</h2>
            <p style="margin: 0;">Major Issues</p>
            <small>Need attention</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        competitor_count = len(metrics['competitor_mentions'])
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="color: {COLORS['secondary']}; margin: 0;">{competitor_count}</h2>
            <p style="margin: 0;">Competitor Mentions</p>
            <small>In reviews</small>
        </div>
        """, unsafe_allow_html=True)

def display_quick_wins(analysis: str):
    """Extract and display quick win recommendations"""
    st.markdown(f"""
    <div class="neon-box">
        <h3 style="color: {COLORS['success']};">⚡ QUICK WINS - IMPLEMENT TODAY</h3>
        <p style="opacity: 0.8;">High-impact changes you can make immediately</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Extract quick wins section
    if "QUICK WINS" in analysis.upper():
        start = analysis.upper().find("QUICK WINS")
        end = len(analysis)
        
        content = analysis[start:end]
        
        # Create actionable cards
        st.markdown("""
        <div class="neon-box priority-high" style="margin-top: 1rem;">
            <h4 style="color: {}; margin-top: 0;">🚀 Immediate Actions</h4>
        </div>
        """.format(COLORS['success']), unsafe_allow_html=True)
        
        st.success(content)
    
    # Add implementation checklist
    st.markdown("### ✅ Implementation Checklist")
    
    checklist_items = [
        "Update product title with top keywords",
        "Revise first 3 bullet points",
        "Add FAQ section to listing",
        "Update main product image",
        "Respond to recent negative reviews",
        "Adjust pricing if recommended",
        "Update A+ content",
        "Add comparison chart"
    ]
    
    for item in checklist_items:
        st.checkbox(item)

def display_quality_report(analysis: str, metrics: Dict[str, Any]):
    """Display quality-focused report for internal team"""
    st.markdown(f"""
    <div class="neon-box">
        <h3 style="color: {COLORS['danger']};">🏭 QUALITY TEAM REPORT</h3>
        <p style="opacity: 0.8;">Product improvement priorities based on customer feedback</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Issue severity breakdown
    issues = metrics['issue_categories']
    
    # Sort by severity
    sorted_issues = sorted(issues.items(), key=lambda x: x[1], reverse=True)
    
    # Display top issues
    st.markdown("### 🔴 Critical Issues Requiring Attention")
    
    for issue, count in sorted_issues[:5]:
        if count > 0:
            severity = 'HIGH' if count > 20 else 'MEDIUM' if count > 10 else 'LOW'
            color = COLORS['danger'] if severity == 'HIGH' else COLORS['warning'] if severity == 'MEDIUM' else COLORS['success']
            
            st.markdown(f"""
            <div class="neon-box" style="border-left: 4px solid {color}; margin-top: 0.5rem;">
                <h4 style="color: {color}; margin: 0;">
                    {issue.replace('_', ' ').title()} - {count} mentions
                </h4>
                <p style="margin: 0.5rem 0 0 0; opacity: 0.8;">Severity: {severity}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Extract quality improvements from AI analysis
    if "QUALITY IMPROVEMENT PRIORITIES" in analysis.upper():
        st.markdown("### 🔧 AI-Recommended Product Improvements")
        
        start = analysis.upper().find("QUALITY IMPROVEMENT PRIORITIES")
        end = analysis.find("\n\n", start + 50)
        if end == -1:
            end = len(analysis)
        
        quality_content = analysis[start:end]
        st.warning(quality_content)
    
    # Add quality metrics
    st.markdown("### 📊 Quality Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        defect_rate = (sum(1 for v in issues.values() if v > 0) / len(issues)) * 100
        st.metric("Issue Coverage", f"{defect_rate:.1f}%", help="% of issue categories with complaints")
    
    with col2:
        return_indicator = metrics['basic_stats']['1_2_star_percentage']
        st.metric("Low Rating %", f"{return_indicator}%", help="Potential return indicator")
    
    with col3:
        quality_score = metrics['review_quality_scores']['avg_quality_score']
        st.metric("Review Quality", f"{quality_score:.1f}/10", help="Average review detail level")

def display_export_options(results: Dict[str, Any], metrics: Dict[str, Any]):
    """Display export options with multiple formats"""
    st.markdown(f"""
    <div class="neon-box">
        <h3 style="color: {COLORS['primary']};">📥 EXPORT ANALYSIS RESULTS</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Executive summary
        executive_summary = generate_executive_summary(results, metrics)
        st.download_button(
            "📄 Executive Summary",
            data=executive_summary,
            file_name=f"executive_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True,
            help="High-level overview for management"
        )
    
    with col2:
        # Full analysis report
        full_report = generate_full_report(results, metrics)
        st.download_button(
            "📊 Full Analysis Report",
            data=full_report,
            file_name=f"listing_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True,
            help="Complete analysis with all details"
        )
    
    with col3:
        # Raw data export
        export_data = {
            'timestamp': results['timestamp'].isoformat(),
            'metrics': metrics,
            'ai_analysis': results['analysis'],
            'product_info': st.session_state.uploaded_data['product_info']
        }
        
        st.download_button(
            "💾 Raw Data (JSON)",
            data=json.dumps(export_data, indent=2, default=str),
            file_name=f"raw_analysis_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
            help="Complete data for further processing"
        )
    
    # Quality team export
    st.markdown("### 🏭 Quality Team Exports")
    
    col1, col2 = st.columns(2)
    
    with col1:
        quality_report = generate_quality_report(metrics, results['analysis'])
        st.download_button(
            "🔧 Quality Improvement Report",
            data=quality_report,
            file_name=f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True
        )
    
    with col2:
        issue_csv = generate_issue_csv(metrics)
        st.download_button(
            "📋 Issue Tracking CSV",
            data=issue_csv,
            file_name=f"issue_tracking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

def generate_executive_summary(results: Dict[str, Any], metrics: Dict[str, Any]) -> str:
    """Generate executive summary"""
    health_score = metrics['listing_health_score']
    
    summary = f"""# Executive Summary - Amazon Listing Analysis
Generated: {results['timestamp'].strftime('%B %d, %Y')}
Product: {st.session_state.uploaded_data['product_info']['asin']}

## Overall Performance
- **Health Score**: {health_score['total_score']}/100 (Grade: {health_score['grade']})
- **Status**: {health_score['status']}
- **Reviews Analyzed**: {results['reviews_analyzed']}
- **Average Rating**: {metrics['basic_stats']['average_rating']}/5

## Key Findings
1. **Sentiment**: {metrics['sentiment_breakdown']['positive']} positive vs {metrics['sentiment_breakdown']['negative']} negative reviews
2. **Trend**: {metrics['temporal_trends'].get('trend', 'Unknown').title()}
3. **Main Issues**: {', '.join([k.replace('_', ' ').title() for k, v in metrics['issue_categories'].items() if v > 5][:3])}

## Top 3 Recommendations
{results['analysis'][:500]}...

## Next Steps
1. Implement quick wins immediately
2. Review full report for detailed insights
3. Forward quality issues to product team
"""
    
    return summary

def generate_full_report(results: Dict[str, Any], metrics: Dict[str, Any]) -> str:
    """Generate comprehensive report"""
    report = f"""# Comprehensive Amazon Listing Analysis Report
Generated: {results['timestamp'].strftime('%B %d, %Y at %I:%M %p')}
Analyzed by: Vive Health Review Intelligence v{APP_CONFIG['version']}

## Product Information
- ASIN: {st.session_state.uploaded_data['product_info']['asin']}
- Total Reviews: {st.session_state.uploaded_data['product_info']['total_reviews']}
- Analyzed Reviews: {results['reviews_analyzed']}

## Performance Metrics

### Overall Health Score: {metrics['listing_health_score']['total_score']}/100 (Grade: {metrics['listing_health_score']['grade']})

Component Scores:
"""
    
    for component, score in metrics['listing_health_score']['components'].items():
        report += f"- {component.replace('_', ' ').title()}: {score:.1f}\n"
    
    report += f"""

### Rating Analysis
- Average Rating: {metrics['basic_stats']['average_rating']}/5
- Median Rating: {metrics['basic_stats']['median_rating']}/5
- Standard Deviation: {metrics['basic_stats']['rating_std']}
- 4-5 Star Percentage: {metrics['basic_stats']['4_5_star_percentage']}%
- 1-2 Star Percentage: {metrics['basic_stats']['1_2_star_percentage']}%

### Sentiment Analysis
- Positive: {metrics['sentiment_breakdown']['positive']}
- Negative: {metrics['sentiment_breakdown']['negative']}
- Neutral: {metrics['sentiment_breakdown']['neutral']}
- Mixed: {metrics['sentiment_breakdown']['mixed']}

### Temporal Trends
- Trend Direction: {metrics['temporal_trends'].get('trend', 'Unknown').title()}
- Recent Performance: {metrics['temporal_trends'].get('recent_performance', 'N/A')}

### Issue Categories
"""
    
    for issue, count in sorted(metrics['issue_categories'].items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            report += f"- {issue.replace('_', ' ').title()}: {count} mentions\n"
    
    report += f"""

### Top Keywords
**Positive Keywords:**
{', '.join([f"{k[0]} ({k[1]})" for k in metrics['keyword_analysis']['positive_keywords'][:10]])}

**Negative Keywords:**
{', '.join([f"{k[0]} ({k[1]})" for k in metrics['keyword_analysis']['negative_keywords'][:10]])}

## AI Analysis

{results['analysis']}

## Implementation Checklist
- [ ] Update product title with recommended keywords
- [ ] Revise bullet points based on customer feedback
- [ ] Add missing information to description
- [ ] Update A+ content to address concerns
- [ ] Implement image/video recommendations
- [ ] Set up FAQ section
- [ ] Address quality issues with product team
- [ ] Monitor performance after changes

---
Report generated by Vive Health Review Intelligence
For support: {APP_CONFIG['support_email']}
"""
    
    return report

def generate_quality_report(metrics: Dict[str, Any], ai_analysis: str) -> str:
    """Generate quality-focused report"""
    report = f"""# Quality Improvement Report
Generated: {datetime.now().strftime('%B %d, %Y')}

## Issue Summary
Total Issues Identified: {sum(metrics['issue_categories'].values())}

### Issue Breakdown by Category
"""
    
    for issue, count in sorted(metrics['issue_categories'].items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            severity = 'CRITICAL' if count > 20 else 'HIGH' if count > 10 else 'MEDIUM' if count > 5 else 'LOW'
            report += f"\n#### {issue.replace('_', ' ').title()}\n"
            report += f"- Mentions: {count}\n"
            report += f"- Severity: {severity}\n"
            report += f"- Impact: {(count / metrics['basic_stats']['total_reviews'] * 100):.1f}% of reviews\n"
    
    # Extract quality section from AI analysis
    if "QUALITY IMPROVEMENT PRIORITIES" in ai_analysis.upper():
        start = ai_analysis.upper().find("QUALITY IMPROVEMENT PRIORITIES")
        end = ai_analysis.find("\n\n", start + 50)
        if end == -1:
            end = len(ai_analysis)
        
        report += f"\n## AI-Recommended Improvements\n{ai_analysis[start:end]}\n"
    
    report += """
## Action Items
1. Review and prioritize critical issues
2. Conduct root cause analysis
3. Implement corrective actions
4. Update quality control procedures
5. Monitor customer feedback post-implementation

## Quality Metrics
"""
    
    report += f"- Average Review Quality Score: {metrics['review_quality_scores']['avg_quality_score']:.1f}/10\n"
    report += f"- High Quality Reviews: {metrics['review_quality_scores']['high_quality_count']}\n"
    report += f"- Low Quality Reviews: {metrics['review_quality_scores']['low_quality_count']}\n"
    
    return report

def generate_issue_csv(metrics: Dict[str, Any]) -> str:
    """Generate CSV for issue tracking"""
    csv_data = "Category,Count,Severity,Percentage\n"
    
    total_reviews = metrics['basic_stats']['total_reviews']
    
    for issue, count in sorted(metrics['issue_categories'].items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            severity = 'CRITICAL' if count > 20 else 'HIGH' if count > 10 else 'MEDIUM' if count > 5 else 'LOW'
            percentage = (count / total_reviews * 100)
            csv_data += f"{issue.replace('_', ' ').title()},{count},{severity},{percentage:.1f}%\n"
    
    return csv_data

def display_metrics_view():
    """Display detailed metrics view with native Streamlit visualizations"""
    if not st.session_state.uploaded_data:
        st.error("No data available")
        return
    
    metrics = st.session_state.uploaded_data['metrics']
    
    st.markdown(f"""
    <div class="cyber-header" style="margin-bottom: 2rem;">
        <h1 style="font-size: 2.5em;">DETAILED METRICS ANALYSIS</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Get visualization data
    viz_data = create_visualization_data(st.session_state.uploaded_data['df_filtered'], metrics)
    
    # Display visualizations
    st.markdown("### 📊 Rating Distribution Analysis")
    if 'rating_distribution' in viz_data:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.bar_chart(
                viz_data['rating_distribution'].set_index('Stars')['Count'],
                color=COLORS['primary']
            )
        with col2:
            st.markdown("**Distribution Stats:**")
            for _, row in viz_data['rating_distribution'].iterrows():
                total = viz_data['rating_distribution']['Count'].sum()
                pct = (row['Count'] / total * 100) if total > 0 else 0
                st.markdown(f"- {row['Stars']} stars: {row['Count']} ({pct:.1f}%)")
    
    # Sentiment breakdown
    st.markdown("### 💭 Sentiment Analysis Breakdown")
    if 'sentiment' in viz_data:
        # Create columns for sentiment cards
        cols = st.columns(4)
        for i, row in enumerate(viz_data['sentiment'].itertuples()):
            with cols[i]:
                color = {
                    'Positive': COLORS['success'],
                    'Negative': COLORS['danger'],
                    'Neutral': COLORS['muted'],
                    'Mixed': COLORS['warning']
                }.get(row.Type, COLORS['primary'])
                
                st.markdown(f"""
                <div class="metric-card" style="border-color: {color};">
                    <h3 style="color: {color}; margin: 0;">{row.Count}</h3>
                    <p style="margin: 0;">{row.Type}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Issue categories detailed view
    st.markdown("### ⚠️ Issue Category Deep Dive")
    if 'issues' in viz_data and not viz_data['issues'].empty:
        # Create expandable sections for each issue
        for _, row in viz_data['issues'].iterrows():
            if row['Count'] > 0:
                severity = 'CRITICAL' if row['Count'] > 20 else 'HIGH' if row['Count'] > 10 else 'MEDIUM' if row['Count'] > 5 else 'LOW'
                color = COLORS['danger'] if severity == 'CRITICAL' else COLORS['warning'] if severity == 'HIGH' else COLORS['accent'] if severity == 'MEDIUM' else COLORS['success']
                
                with st.expander(f"{row['Category'].replace('_', ' ').title()} - {row['Count']} mentions ({severity})"):
                    st.markdown(f"""
                    <div style="border-left: 4px solid {color}; padding-left: 1rem;">
                        <p><strong>Impact:</strong> {(row['Count'] / metrics['basic_stats']['total_reviews'] * 100):.1f}% of all reviews</p>
                        <p><strong>Severity Level:</strong> <span style="color: {color};">{severity}</span></p>
                        <p><strong>Recommended Action:</strong> {"Immediate attention required" if severity in ['CRITICAL', 'HIGH'] else "Monitor and address in next update"}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Detailed statistics table
    st.markdown("### 📊 Comprehensive Statistics")
    
    # Convert metrics to displayable format
    stats_data = []
    basic = metrics['basic_stats']
    
    stats_data.extend([
        {'Category': 'Reviews', 'Metric': 'Total Reviews', 'Value': str(basic['total_reviews'])},
        {'Category': 'Reviews', 'Metric': 'Verified Reviews', 'Value': str(basic['verified_count'])},
        {'Category': 'Ratings', 'Metric': 'Average Rating', 'Value': f"{basic['average_rating']}/5"},
        {'Category': 'Ratings', 'Metric': 'Median Rating', 'Value': f"{basic['median_rating']}/5"},
        {'Category': 'Ratings', 'Metric': 'Standard Deviation', 'Value': str(basic['rating_std'])},
        {'Category': 'Ratings', 'Metric': '4-5 Star %', 'Value': f"{basic['4_5_star_percentage']}%"},
        {'Category': 'Ratings', 'Metric': '1-2 Star %', 'Value': f"{basic['1_2_star_percentage']}%"},
        {'Category': 'Quality', 'Metric': 'Avg Review Quality', 'Value': f"{metrics['review_quality_scores']['avg_quality_score']:.1f}/10"},
        {'Category': 'Quality', 'Metric': 'High Quality Reviews', 'Value': str(metrics['review_quality_scores']['high_quality_count'])},
        {'Category': 'Trend', 'Metric': 'Direction', 'Value': metrics['temporal_trends'].get('trend', 'Unknown').title()},
        {'Category': 'Health', 'Metric': 'Listing Score', 'Value': f"{metrics['listing_health_score']['total_score']:.1f}/100"},
        {'Category': 'Health', 'Metric': 'Grade', 'Value': metrics['listing_health_score']['grade']}
    ])
    
    stats_df = pd.DataFrame(stats_data)
    
    # Style the dataframe
    st.dataframe(
        stats_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Category": st.column_config.TextColumn("Category", width="small"),
            "Metric": st.column_config.TextColumn("Metric", width="medium"),
            "Value": st.column_config.TextColumn("Value", width="small")
        }
    )
    
    # Keyword analysis
    st.markdown("### 🔍 Keyword Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ✅ Positive Keywords")
        positive_keywords = metrics['keyword_analysis']['positive_keywords'][:10]
        for keyword, count in positive_keywords:
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; padding: 0.25rem 0;">
                <span style="color: {COLORS['success']};">{keyword}</span>
                <span style="color: {COLORS['text']}; opacity: 0.7;">{count}</span>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### ❌ Negative Keywords")
        negative_keywords = metrics['keyword_analysis']['negative_keywords'][:10]
        for keyword, count in negative_keywords:
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; padding: 0.25rem 0;">
                <span style="color: {COLORS['danger']};">{keyword}</span>
                <span style="color: {COLORS['text']}; opacity: 0.7;">{count}</span>
            </div>
            """, unsafe_allow_html=True)

def main():
    """Main application with cyberpunk theme"""
    st.set_page_config(
        page_title=APP_CONFIG['title'],
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Inject cyberpunk CSS
    inject_cyberpunk_css()
    
    # Initialize session state
    initialize_session_state()
    
    # Check AI status
    if not check_ai_status():
        st.error(f"❌ AI Service Unavailable. Contact {APP_CONFIG['support_email']}")
        st.stop()
    
    # Display header
    display_header()
    
    # Main content based on view
    if st.session_state.current_view == 'upload':
        handle_file_upload()
    elif st.session_state.current_view == 'results':
        display_results()
    elif st.session_state.current_view == 'metrics':
        display_metrics_view()
    
    # Footer
    st.markdown(f"""
    <div style="text-align: center; color: {COLORS['muted']}; padding: 2rem; margin-top: 4rem; 
                border-top: 1px solid {COLORS['muted']}40;">
        <p style="font-family: 'Rajdhani', sans-serif; letter-spacing: 2px; text-transform: uppercase;">
            {APP_CONFIG['title']} • v{APP_CONFIG['version']}
        </p>
        <p style="font-size: 0.9em; opacity: 0.8;">
            Powered by Advanced AI • Support: {APP_CONFIG['support_email']}
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
