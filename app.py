"""
Amazon Review Analyzer - Medical Device & Listing Optimization
AI-powered analysis for post-market surveillance and listing improvements

Version: 6.0 - Streamlined AI-First Architecture
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
import io
from typing import Dict, List, Any, Optional
import re

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
    'title': 'Medical Device Review Analyzer',
    'version': '6.0',
    'description': 'AI-powered Amazon review analysis',
    'support_email': 'alexander.popoff@vivehealth.com'
}

def initialize_session_state():
    """Initialize session state"""
    defaults = {
        'uploaded_data': None,
        'analysis_results': None,
        'current_view': 'upload',  # 'upload', 'results'
        'processing': False,
        'ai_analyzer': None,
        'show_medical_features': False,
        'basic_stats': None,
        'analysis_mode': 'listing',  # 'listing' or 'quality'
        'input_method': 'file'  # 'file' or 'manual'
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def calculate_performance_score(metrics):
    """Calculate performance score based on manual metrics"""
    # Base scores for each metric (0-100)
    scores = {}
    
    # Rating score (40% weight)
    rating = metrics['avg_rating']
    if rating >= 4.5:
        scores['rating'] = 100
    elif rating >= 4.0:
        scores['rating'] = 80
    elif rating >= 3.5:
        scores['rating'] = 60
    elif rating >= 3.0:
        scores['rating'] = 40
    else:
        scores['rating'] = 20
    
    # Review volume score (20% weight)
    reviews = metrics['total_reviews']
    if reviews >= 1000:
        scores['volume'] = 100
    elif reviews >= 500:
        scores['volume'] = 80
    elif reviews >= 100:
        scores['volume'] = 60
    elif reviews >= 50:
        scores['volume'] = 40
    else:
        scores['volume'] = 20
    
    # Return rate score (20% weight) - lower is better
    returns = metrics['return_rate']
    if returns <= 2:
        scores['returns'] = 100
    elif returns <= 5:
        scores['returns'] = 80
    elif returns <= 10:
        scores['returns'] = 60
    elif returns <= 15:
        scores['returns'] = 40
    else:
        scores['returns'] = 20
    
    # Margin score (20% weight)
    margin = metrics['profit_margin']
    if margin >= 40:
        scores['margin'] = 100
    elif margin >= 30:
        scores['margin'] = 80
    elif margin >= 20:
        scores['margin'] = 60
    elif margin >= 10:
        scores['margin'] = 40
    else:
        scores['margin'] = 20
    
    # Calculate weighted total
    total_score = (
        scores['rating'] * 0.4 +
        scores['volume'] * 0.2 +
        scores['returns'] * 0.2 +
        scores['margin'] * 0.2
    )
    
    return {
        'total_score': round(total_score),
        'component_scores': scores,
        'grade': 'A' if total_score >= 90 else 'B' if total_score >= 80 else 'C' if total_score >= 70 else 'D' if total_score >= 60 else 'F'
    }

def handle_manual_entry():
    """Handle manual data entry for quick scoring"""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #FF6B6B 0%, #FFE66D 100%); 
                padding: 2rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem;">
        <h2 style="margin: 0;">📝 Manual Performance Entry</h2>
        <p style="margin: 0.5rem 0 0 0;">Quick scoring based on your current metrics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Listing information
    with st.expander("📋 Listing Information", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            asin = st.text_input("Product ASIN", placeholder="B08XXXXXX")
            title = st.text_area("Product Title", height=100, placeholder="Your current Amazon listing title...")
        
        with col2:
            category = st.text_input("Product Category", placeholder="e.g., Health & Household")
            price = st.number_input("Current Price ($)", min_value=0.0, value=0.0, step=0.01)
    
    # Performance metrics
    st.markdown("### 📊 Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_rating = st.number_input(
            "Average Rating",
            min_value=1.0,
            max_value=5.0,
            value=4.0,
            step=0.1,
            help="Your product's average star rating"
        )
    
    with col2:
        total_reviews = st.number_input(
            "Total Reviews",
            min_value=0,
            value=100,
            step=1,
            help="Total number of customer reviews"
        )
    
    with col3:
        monthly_sales = st.number_input(
            "Monthly Sales (units)",
            min_value=0,
            value=100,
            step=1,
            help="Average units sold per month"
        )
    
    with col4:
        return_rate = st.number_input(
            "Return Rate (%)",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step=0.1,
            help="Percentage of orders returned"
        )
    
    # Financial metrics
    st.markdown("### 💰 Financial Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        unit_cost = st.number_input(
            "Unit Cost ($)",
            min_value=0.0,
            value=10.0,
            step=0.01,
            help="Your cost per unit including shipping"
        )
    
    with col2:
        amazon_fees = st.number_input(
            "Amazon Fees (%)",
            min_value=0.0,
            max_value=100.0,
            value=15.0,
            step=0.1,
            help="Total Amazon fees percentage"
        )
    
    with col3:
        # Calculate profit margin
        if price > 0:
            profit = price - unit_cost - (price * amazon_fees / 100)
            margin = (profit / price) * 100
        else:
            margin = 0
        
        st.metric("Profit Margin", f"{margin:.1f}%", help="Calculated automatically")
    
    # Optional: Common issues
    with st.expander("🔍 Common Customer Issues (Optional)", expanded=False):
        st.markdown("Check any issues frequently mentioned in reviews:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            size_issues = st.checkbox("Size/Fit Issues")
            quality_issues = st.checkbox("Quality Concerns")
            shipping_issues = st.checkbox("Shipping/Packaging Problems")
        
        with col2:
            description_issues = st.checkbox("Inaccurate Description")
            durability_issues = st.checkbox("Durability Problems")
            value_issues = st.checkbox("Value/Price Concerns")
    
    # Analyze button
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🎯 Calculate Performance Score", type="primary", use_container_width=True):
        if asin and title:
            # Prepare metrics
            metrics = {
                'asin': asin,
                'title': title,
                'category': category,
                'price': price,
                'avg_rating': avg_rating,
                'total_reviews': total_reviews,
                'monthly_sales': monthly_sales,
                'return_rate': return_rate,
                'unit_cost': unit_cost,
                'amazon_fees': amazon_fees,
                'profit_margin': margin,
                'issues': {
                    'size': size_issues,
                    'quality': quality_issues,
                    'shipping': shipping_issues,
                    'description': description_issues,
                    'durability': durability_issues,
                    'value': value_issues
                }
            }
            
            # Calculate score
            score_data = calculate_performance_score(metrics)
            
            # Run AI analysis if available
            if check_ai_status():
                with st.spinner("🤖 Generating AI insights..."):
                    ai_insights = generate_manual_entry_insights(metrics, score_data)
                    if ai_insights:
                        metrics['ai_insights'] = ai_insights
            
            # Store results
            st.session_state.analysis_results = {
                'success': True,
                'manual_entry': True,
                'metrics': metrics,
                'score_data': score_data,
                'timestamp': datetime.now()
            }
            
            st.session_state.current_view = 'results'
            st.rerun()
        else:
            st.error("❌ Please enter at least ASIN and Title")

def generate_manual_entry_insights(metrics, score_data):
    """Generate AI insights for manual entry data"""
    try:
        prompt = f"""
        Analyze this Amazon product performance data and provide actionable insights:
        
        Product: {metrics['asin']} - {metrics['title']}
        Category: {metrics['category']}
        
        PERFORMANCE METRICS:
        - Average Rating: {metrics['avg_rating']}/5
        - Total Reviews: {metrics['total_reviews']}
        - Monthly Sales: {metrics['monthly_sales']} units
        - Return Rate: {metrics['return_rate']}%
        - Price: ${metrics['price']}
        - Profit Margin: {metrics['profit_margin']:.1f}%
        
        PERFORMANCE SCORE: {score_data['total_score']}/100 (Grade: {score_data['grade']})
        
        IDENTIFIED ISSUES:
        {', '.join([k for k, v in metrics['issues'].items() if v]) or 'None specified'}
        
        Provide:
        1. TOP 3 PRIORITIES to improve this listing
        2. SPECIFIC LISTING OPTIMIZATIONS (title, bullets, images)
        3. PRICING STRATEGY recommendation
        4. REVIEW MANAGEMENT tactics
        5. PROFIT IMPROVEMENT opportunities
        
        Focus on actionable, specific recommendations based on the metrics.
        """
        
        if st.session_state.analysis_mode == 'quality':
            prompt += "\n\nQUALITY FOCUS: Emphasize reducing returns, improving product quality, and addressing customer issues."
        
        result = st.session_state.ai_analyzer.api_client.call_api([
            {"role": "system", "content": "You are an Amazon listing optimization expert. Provide specific, actionable advice based on performance metrics."},
            {"role": "user", "content": prompt}
        ], max_tokens=1500, temperature=0.3)
        
        if result['success']:
            return result['result']
        else:
            return None
            
    except Exception as e:
        logger.error(f"Manual entry AI insights error: {e}")
        return None
    """Generate formatted AI report for export"""
    mode_title = "Quality & Regulatory Analysis" if results['mode'] == 'quality' else "Amazon Listing Optimization Analysis"
    
    report = f"""# {mode_title} Report
Generated: {results['timestamp'].strftime('%B %d, %Y at %I:%M %p')}

## Product Information
- **ASIN**: {product_info['asin']}
- **Total Reviews Analyzed**: {results['reviews_analyzed']}
- **Average Rating**: {stats['average_rating']}/5
- **Verified Reviews**: {stats['verified_count']}
- **Low Ratings (1-2 stars)**: {stats['1_2_star_percentage']}%

"""
    
    if stats.get('date_range'):
        report += f"- **Review Period**: {stats['date_range']['earliest']} to {stats['date_range']['latest']}\n"
    
    report += f"\n## AI Analysis\n\n{results['analysis']}\n\n"
    
    # Add rating distribution
    report += "## Rating Distribution\n"
    for rating in range(5, 0, -1):
        count = stats['rating_distribution'].get(rating, 0)
        percentage = (count / stats['total_reviews']) * 100 if stats['total_reviews'] > 0 else 0
        report += f"- **{rating} stars**: {count} reviews ({percentage:.1f}%)\n"
    
    # Add mode-specific sections
    if results['mode'] == 'quality':
        report += """
## Quality Management Actions
- [ ] Review all critical safety issues
- [ ] Document corrective actions
- [ ] Update quality procedures
- [ ] File regulatory reports if required
- [ ] Schedule follow-up review

## Regulatory Considerations
- Review findings against 21 CFR 820 requirements
- Check for MDR reportable events
- Update risk management file
- Consider post-market clinical follow-up
"""
    else:
        report += """
## Listing Optimization Checklist
- [ ] Update product title with key benefits
- [ ] Revise bullet points based on customer feedback
- [ ] Add missing information to description
- [ ] Update A+ content to address concerns
- [ ] Optimize keywords based on customer language
- [ ] Add comparison chart if needed
"""
    
    return report

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

def parse_amazon_date(date_string):
    """Parse Amazon review date"""
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
    """Calculate basic statistics only"""
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
            '1_2_star_percentage': round((sum(ratings <= 2) / len(ratings)) * 100, 1) if len(ratings) > 0 else 0
        }
        
        # Date range if available
        if 'Date' in df.columns:
            df['parsed_date'] = df['Date'].apply(parse_amazon_date)
            valid_dates = df['parsed_date'].dropna()
            if len(valid_dates) > 0:
                stats['date_range'] = {
                    'earliest': valid_dates.min(),
                    'latest': valid_dates.max()
                }
        
        return stats
        
    except Exception as e:
        logger.error(f"Stats calculation error: {e}")
        return None

def prepare_reviews_for_ai(df):
    """Prepare all reviews for AI analysis"""
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
    
    return reviews

def run_ai_analysis(df, product_info):
    """Send all reviews to AI for comprehensive analysis"""
    if not check_ai_status():
        st.error(f"❌ AI Analysis unavailable. Please contact {APP_CONFIG['support_email']}")
        return None
    
    try:
        reviews = prepare_reviews_for_ai(df)
        
        # Different prompts based on mode
        if st.session_state.analysis_mode == 'quality':
            # Quality/Regulatory focused prompt
            context = f"""
            Analyze these {len(reviews)} Amazon reviews for QUALITY and REGULATORY insights.
            Product ASIN: {product_info.get('asin', 'Unknown')}
            
            Focus on:
            1. SAFETY ISSUES & ADVERSE EVENTS (critical for FDA/regulatory)
            2. PRODUCT DEFECTS & FAILURE MODES
            3. QUALITY CONTROL PROBLEMS
            4. NON-CONFORMANCE PATTERNS
            5. REGULATORY COMPLIANCE CONCERNS
            6. POST-MARKET SURVEILLANCE FINDINGS
            7. CORRECTIVE ACTION RECOMMENDATIONS
            
            Categorize by severity: CRITICAL / MAJOR / MINOR
            """
            
            if st.session_state.show_medical_features:
                context += "\n\nMEDICAL DEVICE: Apply 21 CFR 820, ISO 13485, and MDR requirements. Flag any potential reportable events."
        
        else:  # listing mode
            context = f"""
            Analyze these {len(reviews)} Amazon reviews for LISTING OPTIMIZATION.
            Product ASIN: {product_info.get('asin', 'Unknown')}
            
            Focus on:
            1. KEY CUSTOMER COMPLAINTS (what's hurting sales)
            2. MISSING FEATURES customers expected
            3. POSITIVE FEATURES to highlight
            4. COMPETITOR COMPARISONS mentioned
            5. SPECIFIC LISTING IMPROVEMENTS (title, bullets, A+ content)
            6. KEYWORD OPPORTUNITIES from customer language
            7. MAIN OBJECTIONS to address
            """
            
            if st.session_state.show_medical_features:
                context += "\n\nMedical device listing - ensure claims are FDA compliant."
        
        # Create comprehensive prompt
        reviews_text = []
        for r in reviews:
            reviews_text.append(f"[Review {r['id']} - {r['rating']}/5 stars{' - VERIFIED' if r['verified'] else ''}]\n{r['title']}\n{r['body']}\n")
        
        prompt = f"""{context}

REVIEWS:
{''.join(reviews_text[:100])}  # Limit to prevent token overflow

Provide a structured analysis with clear sections and actionable insights.
Be specific with percentages, counts, and examples.
Format for easy reading with clear headers and bullet points."""

        # Call AI
        result = st.session_state.ai_analyzer.api_client.call_api([
            {"role": "system", "content": "You are an expert at analyzing Amazon reviews. Provide specific, actionable insights formatted clearly."},
            {"role": "user", "content": prompt}
        ], max_tokens=2500, temperature=0.3)
        
        if result['success']:
            return {
                'success': True,
                'analysis': result['result'],
                'reviews_analyzed': len(reviews),
                'timestamp': datetime.now(),
                'mode': st.session_state.analysis_mode,
                'raw_reviews': reviews  # Keep for export
            }
        else:
            return None
            
    except Exception as e:
        logger.error(f"AI analysis error: {e}")
        return None

def display_header():
    """Display application header"""
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        if st.button("🔄 New Analysis", use_container_width=True):
            reset_analysis()
            st.rerun()
    
    with col2:
        st.markdown("""
        <div style="text-align: center;">
            <h1 style="color: #667eea; margin: 0;">Medical Device Review Analyzer</h1>
            <p style="color: #666; margin: 0;">AI-Powered Amazon Review Analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.toggle("🏥 Medical Mode", key="show_medical_features")
    
    # Analysis mode selector
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        mode = st.radio(
            "Analysis Focus:",
            options=['listing', 'quality'],
            format_func=lambda x: "🛒 Amazon Listing Optimization" if x == 'listing' else "📋 Quality & Regulatory Focus",
            horizontal=True,
            key='analysis_mode',
            help="Listing: Optimize for sales | Quality: Regulatory compliance & surveillance"
        )

def handle_file_upload():
    """Streamlined file upload with manual entry option"""
    # Input method selector
    tab1, tab2 = st.tabs(["📁 File Upload", "📝 Manual Entry"])
    
    with tab1:
        mode_emoji = "📋" if st.session_state.analysis_mode == 'quality' else "🛒"
        mode_text = "Quality & Regulatory Analysis" if st.session_state.analysis_mode == 'quality' else "Amazon Listing Optimization"
        mode_color = "#f44336" if st.session_state.analysis_mode == 'quality' else "#2196f3"
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {mode_color} 0%, {mode_color}80 100%); 
                    padding: 2rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem;">
            <h2 style="margin: 0;">{mode_emoji} Upload Your Helium 10 Review Export</h2>
            <p style="margin: 0.5rem 0 0 0;">Mode: {mode_text}</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose your review file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload your Helium 10 Amazon review export"
        )
        
        if uploaded_file:
            try:
                # Read file
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                # Validate
                required_cols = ['Title', 'Body', 'Rating']
                missing = [col for col in required_cols if col not in df.columns]
                
                if missing:
                    st.error(f"❌ Missing columns: {', '.join(missing)}")
                    st.info("Required: Title, Body, Rating, Date (optional)")
                    return
                
                # Process
                with st.spinner("Processing reviews..."):
                    # Calculate basic stats
                    stats = calculate_basic_stats(df)
                    st.session_state.basic_stats = stats
                    
                    # Prepare data
                    product_info = {
                        'asin': df['Variation'].iloc[0] if 'Variation' in df.columns else 'Unknown',
                        'total_reviews': len(df)
                    }
                    
                    st.session_state.uploaded_data = {
                        'df': df,
                        'product_info': product_info,
                        'stats': stats
                    }
                    
                    # Show preview
                    if stats:
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Total Reviews", stats['total_reviews'])
                        col2.metric("Avg Rating", f"{stats['average_rating']}/5")
                        col3.metric("Verified", stats['verified_count'])
                        
                        # Mode-specific metric
                        if st.session_state.analysis_mode == 'quality':
                            col4.metric("⚠️ Low Ratings", f"{stats['1_2_star_percentage']}%", 
                                       help="Critical for quality surveillance")
                        else:
                            col4.metric("💰 Opportunity", f"{100 - stats['average_rating']*20:.0f}%",
                                       help="Potential rating improvement")
                    
                    # Mode-specific messaging
                    if st.session_state.analysis_mode == 'quality':
                        st.info("📋 **Quality Mode**: AI will analyze for safety issues, defects, and regulatory compliance")
                    else:
                        st.info("🛒 **Listing Mode**: AI will identify optimization opportunities and customer insights")
                    
                    # Analyze button
                    st.markdown("<br>", unsafe_allow_html=True)
                    button_text = "🚀 Run Quality Analysis" if st.session_state.analysis_mode == 'quality' else "🚀 Run Listing Analysis"
                    
                    if st.button(button_text, type="primary", use_container_width=True):
                        with st.spinner(f"🤖 AI performing {mode_text.lower()}... (this may take 1-2 minutes)"):
                            result = run_ai_analysis(df, product_info)
                            
                            if result:
                                st.session_state.analysis_results = result
                                st.session_state.current_view = 'results'
                                st.rerun()
                            else:
                                st.error(f"❌ Analysis failed. Contact {APP_CONFIG['support_email']}")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info(f"Need help? Contact {APP_CONFIG['support_email']}")
    
    with tab2:
        handle_manual_entry()

def display_results():
    """Display AI analysis results"""
    if not st.session_state.analysis_results:
        st.error("No results available")
        return
    
    results = st.session_state.analysis_results
    
    # Check if this is manual entry or file upload
    if results.get('manual_entry'):
        display_manual_entry_results(results)
    else:
        display_file_analysis_results(results)

def display_manual_entry_results(results):
    """Display results for manual entry"""
    metrics = results['metrics']
    score_data = results['score_data']
    
    # Score header with color based on grade
    grade_colors = {
        'A': '#4CAF50',
        'B': '#8BC34A', 
        'C': '#FF9800',
        'D': '#FF5722',
        'F': '#F44336'
    }
    
    grade_color = grade_colors.get(score_data['grade'], '#666')
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {grade_color} 0%, {grade_color}80 100%);
                padding: 2rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem;">
        <h1 style="margin: 0; font-size: 4em;">{score_data['grade']}</h1>
        <h2 style="margin: 0.5rem 0;">Performance Score: {score_data['total_score']}/100</h2>
        <p style="margin: 0;">{metrics['asin']} • Analyzed {results['timestamp'].strftime('%B %d, %Y at %I:%M %p')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Product info
    st.markdown(f"### 📦 {metrics['title']}")
    if metrics['category']:
        st.caption(f"Category: {metrics['category']} • Price: ${metrics['price']}")
    
    # Score breakdown
    st.markdown("### 📊 Score Breakdown")
    
    col1, col2, col3, col4 = st.columns(4)
    
    scores = score_data['component_scores']
    
    with col1:
        rating_color = '#4CAF50' if scores['rating'] >= 80 else '#FF9800' if scores['rating'] >= 60 else '#F44336'
        st.markdown(f"""
        <div style="background: {rating_color}20; padding: 1rem; border-radius: 10px; 
                    border: 2px solid {rating_color}; text-align: center;">
            <h3 style="color: {rating_color}; margin: 0;">{scores['rating']}/100</h3>
            <p style="margin: 0;">Rating Score</p>
            <small>{metrics['avg_rating']}/5 stars</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        volume_color = '#4CAF50' if scores['volume'] >= 80 else '#FF9800' if scores['volume'] >= 60 else '#F44336'
        st.markdown(f"""
        <div style="background: {volume_color}20; padding: 1rem; border-radius: 10px;
                    border: 2px solid {volume_color}; text-align: center;">
            <h3 style="color: {volume_color}; margin: 0;">{scores['volume']}/100</h3>
            <p style="margin: 0;">Review Volume</p>
            <small>{metrics['total_reviews']} reviews</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        return_color = '#4CAF50' if scores['returns'] >= 80 else '#FF9800' if scores['returns'] >= 60 else '#F44336'
        st.markdown(f"""
        <div style="background: {return_color}20; padding: 1rem; border-radius: 10px;
                    border: 2px solid {return_color}; text-align: center;">
            <h3 style="color: {return_color}; margin: 0;">{scores['returns']}/100</h3>
            <p style="margin: 0;">Return Score</p>
            <small>{metrics['return_rate']}% returns</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        margin_color = '#4CAF50' if scores['margin'] >= 80 else '#FF9800' if scores['margin'] >= 60 else '#F44336'
        st.markdown(f"""
        <div style="background: {margin_color}20; padding: 1rem; border-radius: 10px;
                    border: 2px solid {margin_color}; text-align: center;">
            <h3 style="color: {margin_color}; margin: 0;">{scores['margin']}/100</h3>
            <p style="margin: 0;">Margin Score</p>
            <small>{metrics['profit_margin']:.1f}% margin</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Performance metrics
    st.markdown("### 📈 Performance Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Monthly Sales", f"{metrics['monthly_sales']} units")
        st.metric("Monthly Revenue", f"${metrics['monthly_sales'] * metrics['price']:,.2f}")
    
    with col2:
        monthly_profit = metrics['monthly_sales'] * (metrics['price'] * metrics['profit_margin'] / 100)
        st.metric("Monthly Profit", f"${monthly_profit:,.2f}")
        st.metric("Annual Profit Projection", f"${monthly_profit * 12:,.2f}")
    
    # Issues identified
    if any(metrics['issues'].values()):
        st.markdown("### ⚠️ Identified Issues")
        issue_names = {
            'size': '📏 Size/Fit Issues',
            'quality': '🔧 Quality Concerns',
            'shipping': '📦 Shipping/Packaging',
            'description': '📝 Inaccurate Description',
            'durability': '💪 Durability Problems',
            'value': '💰 Value/Price Concerns'
        }
        
        for key, value in metrics['issues'].items():
            if value:
                st.warning(issue_names.get(key, key))
    
    # AI insights
    if metrics.get('ai_insights'):
        st.markdown("### 🤖 AI Optimization Insights")
        st.info(metrics['ai_insights'])
    
    # Recommendations based on score
    st.markdown("### 🎯 Performance Recommendations")
    
    if score_data['total_score'] >= 90:
        st.success("""
        **Excellent Performance!** Your listing is performing at top tier. Focus on:
        - Maintaining quality standards
        - Expanding product variations
        - Increasing advertising spend for growth
        """)
    elif score_data['total_score'] >= 80:
        st.info("""
        **Good Performance!** Room for optimization:
        - Address any identified customer issues
        - Optimize listing content for better conversion
        - Consider pricing adjustments
        """)
    elif score_data['total_score'] >= 70:
        st.warning("""
        **Average Performance** - Significant improvement needed:
        - Urgently address customer complaints
        - Improve product quality or description accuracy
        - Review pricing strategy
        """)
    else:
        st.error("""
        **Poor Performance** - Major changes required:
        - Conduct thorough product quality review
        - Completely revise listing content
        - Consider product improvements or discontinuation
        """)
    
    # Export options
    st.markdown("### 📥 Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export score report
        score_report = f"""# Performance Score Report
Generated: {results['timestamp'].strftime('%B %d, %Y at %I:%M %p')}

## Product Information
- ASIN: {metrics['asin']}
- Title: {metrics['title']}
- Category: {metrics['category']}
- Price: ${metrics['price']}

## Performance Score: {score_data['total_score']}/100 (Grade: {score_data['grade']})

### Score Breakdown
- Rating Score: {scores['rating']}/100 ({metrics['avg_rating']}/5 stars)
- Review Volume: {scores['volume']}/100 ({metrics['total_reviews']} reviews)
- Return Score: {scores['returns']}/100 ({metrics['return_rate']}% return rate)
- Margin Score: {scores['margin']}/100 ({metrics['profit_margin']:.1f}% margin)

### Financial Performance
- Monthly Sales: {metrics['monthly_sales']} units
- Monthly Revenue: ${metrics['monthly_sales'] * metrics['price']:,.2f}
- Monthly Profit: ${metrics['monthly_sales'] * (metrics['price'] * metrics['profit_margin'] / 100):,.2f}

### Identified Issues
{chr(10).join(['- ' + k for k, v in metrics['issues'].items() if v]) or 'None'}

### AI Insights
{metrics.get('ai_insights', 'Not available')}
"""
        
        st.download_button(
            "📄 Download Score Report",
            data=score_report,
            file_name=f"performance_score_{metrics['asin']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True
        )
    
    with col2:
        # Export raw data
        export_data = {
            'timestamp': results['timestamp'].isoformat(),
            'metrics': metrics,
            'score_data': score_data
        }
        
        st.download_button(
            "💾 Download Raw Data",
            data=json.dumps(export_data, indent=2, default=str),
            file_name=f"performance_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

def display_file_analysis_results(results):
    """Display results from file upload analysis"""
    stats = st.session_state.basic_stats
    product_info = st.session_state.uploaded_data['product_info']
    
    # Results header with mode indication
    mode_color = "#f44336" if results['mode'] == 'quality' else "#2196f3"
    mode_text = "Quality & Regulatory Analysis" if results['mode'] == 'quality' else "Listing Optimization Analysis"
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {mode_color} 0%, {mode_color}80 100%);
                padding: 1.5rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem;">
        <h2 style="margin: 0;">✅ {mode_text} Complete</h2>
        <p style="margin: 0.5rem 0 0 0;">AI analyzed {{count}} reviews • {{timestamp}}</p>
    </div>
    """.format(
        count=results['reviews_analyzed'],
        timestamp=results['timestamp'].strftime('%B %d, %Y at %I:%M %p')
    ), unsafe_allow_html=True)
    
    # Continue with existing file analysis display...
    # [Rest of the existing display_results code continues here]
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Reviews Analyzed", results['reviews_analyzed'])
    col2.metric("Average Rating", f"{stats['average_rating']}/5")
    col3.metric("Product ASIN", product_info['asin'])
    col4.metric("Analysis Type", "Quality" if results['mode'] == 'quality' else "Listing")
    
    # Main AI analysis
    st.markdown("### 🤖 AI Analysis")
    
    # Display analysis in a clean format
    analysis_text = results['analysis']
    
    # Different section mappings based on mode
    if results['mode'] == 'quality':
        sections = {
            'SAFETY ISSUES': '🚨',
            'ADVERSE EVENTS': '⚠️',
            'PRODUCT DEFECTS': '🔧',
            'QUALITY CONTROL': '📋',
            'NON-CONFORMANCE': '❌',
            'REGULATORY': '📜',
            'CORRECTIVE ACTION': '✅',
            'CRITICAL': '🔴',
            'MAJOR': '🟠',
            'MINOR': '🟡'
        }
    else:
        sections = {
            'KEY CUSTOMER COMPLAINTS': '😞',
            'MISSING FEATURES': '❓',
            'POSITIVE FEATURES': '✨',
            'COMPETITOR': '🏆',
            'LISTING IMPROVEMENTS': '📝',
            'KEYWORD': '🔍',
            'OBJECTIONS': '🚫'
        }
    
    # Try to display sections if found
    displayed_sections = False
    for section, icon in sections.items():
        if section in analysis_text.upper():
            displayed_sections = True
            st.markdown(f"### {icon} {section.title()}")
            # Extract section content
            start = analysis_text.upper().find(section) + len(section)
            end = len(analysis_text)
            for next_section in sections:
                next_pos = analysis_text.upper().find(next_section, start)
                if next_pos > 0 and next_pos < end:
                    end = next_pos
            
            content = analysis_text[start:end].strip()
            if content:
                if results['mode'] == 'quality' and section in ['CRITICAL', 'MAJOR']:
                    st.error(content)
                elif results['mode'] == 'quality' and section == 'MINOR':
                    st.warning(content)
                else:
                    st.info(content)
    
    # If no sections found, display full analysis
    if not displayed_sections:
        st.info(analysis_text)
    
    # Additional details
    with st.expander("📊 Additional Statistics"):
        # Rating distribution
        st.markdown("**Rating Distribution:**")
        rating_dist = stats['rating_distribution']
        for rating in range(5, 0, -1):
            count = rating_dist.get(rating, 0)
            percentage = (count / stats['total_reviews']) * 100 if stats['total_reviews'] > 0 else 0
            st.progress(percentage / 100, text=f"{rating} stars: {count} reviews ({percentage:.1f}%)")
        
        # Date range
        if stats.get('date_range'):
            st.markdown(f"**Review Period:** {stats['date_range']['earliest']} to {stats['date_range']['latest']}")
    
    # Medical device alert (if enabled and quality mode)
    if st.session_state.show_medical_features and results['mode'] == 'quality':
        if any(word in analysis_text.upper() for word in ['SAFETY', 'ADVERSE', 'CRITICAL', 'FDA', 'REGULATORY']):
            st.markdown("""
            <div style="background: #ffebee; padding: 1rem; border-radius: 10px; border-left: 4px solid #f44336; margin-top: 1rem;">
                <h4 style="color: #c62828; margin: 0;">🏥 Medical Device Alert</h4>
                <p style="margin: 0.5rem 0 0 0;">Critical findings detected - review for regulatory reporting requirements</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Export options
    st.markdown("### 📥 Export Options")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export AI report
        report_content = generate_ai_report(results, stats, product_info)
        
        st.download_button(
            "📄 Download AI Report",
            data=report_content,
            file_name=f"ai_review_report_{results['mode']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True,
            help="Formatted report with AI insights and action items"
        )
    
    with col2:
        # Export raw data
        export_data = {
            'analysis_date': results['timestamp'].isoformat(),
            'analysis_mode': results['mode'],
            'reviews_analyzed': results['reviews_analyzed'],
            'basic_stats': stats,
            'ai_analysis': results['analysis'],
            'product_info': product_info
        }
        
        st.download_button(
            "💾 Download Raw Data",
            data=json.dumps(export_data, indent=2, default=str),
            file_name=f"review_analysis_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
            help="Complete analysis data in JSON format"
        )
    
    with col3:
        # Quality-specific export
        if results['mode'] == 'quality':
            # Create CSV for tracking
            tracking_data = f"Date,ASIN,Total Reviews,Avg Rating,Low Rating %,Critical Issues\n"
            tracking_data += f"{results['timestamp'].strftime('%Y-%m-%d')},{product_info['asin']},{stats['total_reviews']},{stats['average_rating']},{stats['1_2_star_percentage']},"
            
            st.download_button(
                "📊 Quality Tracking CSV",
                data=tracking_data,
                file_name=f"quality_tracking_{product_info['asin']}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True,
                help="CSV format for quality tracking systems"
            )
    
    # Main AI analysis
    st.markdown("### 🤖 AI Analysis")
    
    # Display analysis in a clean format
    analysis_text = results['analysis']
    
    # Different section mappings based on mode
    if results['mode'] == 'quality':
        sections = {
            'SAFETY ISSUES': '🚨',
            'ADVERSE EVENTS': '⚠️',
            'PRODUCT DEFECTS': '🔧',
            'QUALITY CONTROL': '📋',
            'NON-CONFORMANCE': '❌',
            'REGULATORY': '📜',
            'CORRECTIVE ACTION': '✅',
            'CRITICAL': '🔴',
            'MAJOR': '🟠',
            'MINOR': '🟡'
        }
    else:
        sections = {
            'KEY CUSTOMER COMPLAINTS': '😞',
            'MISSING FEATURES': '❓',
            'POSITIVE FEATURES': '✨',
            'COMPETITOR': '🏆',
            'LISTING IMPROVEMENTS': '📝',
            'KEYWORD': '🔍',
            'OBJECTIONS': '🚫'
        }
    
    # Try to display sections if found
    displayed_sections = False
    for section, icon in sections.items():
        if section in analysis_text.upper():
            displayed_sections = True
            st.markdown(f"### {icon} {section.title()}")
            # Extract section content
            start = analysis_text.upper().find(section) + len(section)
            end = len(analysis_text)
            for next_section in sections:
                next_pos = analysis_text.upper().find(next_section, start)
                if next_pos > 0 and next_pos < end:
                    end = next_pos
            
            content = analysis_text[start:end].strip()
            if content:
                if results['mode'] == 'quality' and section in ['CRITICAL', 'MAJOR']:
                    st.error(content)
                elif results['mode'] == 'quality' and section == 'MINOR':
                    st.warning(content)
                else:
                    st.info(content)
    
    # If no sections found, display full analysis
    if not displayed_sections:
        st.info(analysis_text)
    
    # Additional details
    with st.expander("📊 Additional Statistics"):
        # Rating distribution
        st.markdown("**Rating Distribution:**")
        rating_dist = stats['rating_distribution']
        for rating in range(5, 0, -1):
            count = rating_dist.get(rating, 0)
            percentage = (count / stats['total_reviews']) * 100 if stats['total_reviews'] > 0 else 0
            st.progress(percentage / 100, text=f"{rating} stars: {count} reviews ({percentage:.1f}%)")
        
        # Date range
        if stats.get('date_range'):
            st.markdown(f"**Review Period:** {stats['date_range']['earliest']} to {stats['date_range']['latest']}")
    
    # Medical device alert (if enabled and quality mode)
    if st.session_state.show_medical_features and results['mode'] == 'quality':
        if any(word in analysis_text.upper() for word in ['SAFETY', 'ADVERSE', 'CRITICAL', 'FDA', 'REGULATORY']):
            st.markdown("""
            <div style="background: #ffebee; padding: 1rem; border-radius: 10px; border-left: 4px solid #f44336; margin-top: 1rem;">
                <h4 style="color: #c62828; margin: 0;">🏥 Medical Device Alert</h4>
                <p style="margin: 0.5rem 0 0 0;">Critical findings detected - review for regulatory reporting requirements</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Export options
    st.markdown("### 📥 Export Options")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export AI report
        report_content = generate_ai_report(results, stats, product_info)
        
        st.download_button(
            "📄 Download AI Report",
            data=report_content,
            file_name=f"ai_review_report_{results['mode']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True,
            help="Formatted report with AI insights and action items"
        )
    
    with col2:
        # Export raw data
        export_data = {
            'analysis_date': results['timestamp'].isoformat(),
            'analysis_mode': results['mode'],
            'reviews_analyzed': results['reviews_analyzed'],
            'basic_stats': stats,
            'ai_analysis': results['analysis'],
            'product_info': product_info
        }
        
        st.download_button(
            "💾 Download Raw Data",
            data=json.dumps(export_data, indent=2, default=str),
            file_name=f"review_analysis_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
            help="Complete analysis data in JSON format"
        )
    
    with col3:
        # Quality-specific export
        if results['mode'] == 'quality':
            # Create CSV for tracking
            tracking_data = f"Date,ASIN,Total Reviews,Avg Rating,Low Rating %,Critical Issues\n"
            tracking_data += f"{results['timestamp'].strftime('%Y-%m-%d')},{product_info['asin']},{stats['total_reviews']},{stats['average_rating']},{stats['1_2_star_percentage']},"
            
            st.download_button(
                "📊 Quality Tracking CSV",
                data=tracking_data,
                file_name=f"quality_tracking_{product_info['asin']}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True,
                help="CSV format for quality tracking systems"
            )

def main():
    """Main application"""
    st.set_page_config(
        page_title=APP_CONFIG['title'],
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS for modern UI
    st.markdown("""
    <style>
    /* Modern, clean styling */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* Primary button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
    }
    
    /* Metrics */
    [data-testid="metric-container"] {
        background-color: white;
        border: 1px solid #e0e0e0;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #667eea;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #764ba2;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.1);
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: white;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
    
    /* Hide default sidebar */
    section[data-testid="stSidebar"] {
        display: none;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize
    initialize_session_state()
    
    # Check AI status on startup
    if not check_ai_status():
        st.error(f"❌ AI Service Unavailable. Please contact {APP_CONFIG['support_email']} for assistance.")
        st.stop()
    
    # Display header with reset button
    display_header()
    
    # Main content
    if st.session_state.current_view == 'upload':
        handle_file_upload()
    elif st.session_state.current_view == 'results':
        display_results()
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>Medical Device Review Analyzer v{APP_CONFIG['version']} • Powered by AI</p>
        <p style="font-size: 0.9em;">Support: {APP_CONFIG['support_email']}</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
