"""
AI-Assisted Quality Case Screening Wizard Module

Provides a step-by-step SOP-guided workflow for screening quality cases with AI recommendations.
Populates Smartsheet-compatible case data with priority assessment.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


# =====================================================
# PRODUCT CATEGORY HIERARCHY WITH RETURN RATE THRESHOLDS
# Based on Trailing 12 Month Amazon Returns Analysis
# =====================================================

PRODUCT_CATEGORIES = {
    'MOB': {
        'name': 'Mobility',
        'description': 'Mobility aids and equipment',
        'default_threshold': 0.096,  # 9.6% category average
        'subcategories': {
            'Scooter': {'name': 'Knee Scooters', 'threshold': 0.10, 'description': 'Knee scooters and rolling mobility aids'},
            'Knee Walker': {'name': 'Knee Walkers', 'threshold': 0.11, 'description': 'Knee walkers and rollators'},
            'Electric Wheelchair': {'name': 'Electric Wheelchairs', 'threshold': 0.15, 'description': 'Powered wheelchairs and mobility chairs'},
            'Canes': {'name': 'Canes & Walking Sticks', 'threshold': 0.075, 'description': 'Walking canes, quad canes, folding canes'},
            'Walker Accessories': {'name': 'Walker Accessories', 'threshold': 0.055, 'description': 'Walker ski glides, cups, bags'},
            'Cupholder': {'name': 'Cupholders', 'threshold': 0.11, 'description': 'Wheelchair and walker cupholders'},
            'Other MOB': {'name': 'Other Mobility', 'threshold': 0.10, 'description': 'Other mobility products'}
        }
    },
    'LVA': {
        'name': 'Living Aids',
        'description': 'Daily living assistance products',
        'default_threshold': 0.138,  # 13.8% category average
        'subcategories': {
            'Stand Assist': {'name': 'Stand Assist Devices', 'threshold': 0.185, 'description': 'Stand assist rails, poles, handles'},
            'Commode': {'name': 'Commodes', 'threshold': 0.16, 'description': 'Bedside commodes, portable toilets'},
            'Commode Rail': {'name': 'Commode Rails', 'threshold': 0.17, 'description': 'Toilet safety rails and frames'},
            'Commode Riser': {'name': 'Toilet Risers', 'threshold': 0.168, 'description': 'Raised toilet seats and risers'},
            'Commode Cushion': {'name': 'Commode Cushions', 'threshold': 0.16, 'description': 'Toilet seat cushions and pads'},
            'Shower Mat': {'name': 'Shower Safety', 'threshold': 0.09, 'description': 'Shower mats, bath safety'},
            'Walker Bag': {'name': 'Walker Bags', 'threshold': 0.07, 'description': 'Walker and rollator bags'},
            'Sling': {'name': 'Transfer Slings', 'threshold': 0.13, 'description': 'Patient transfer slings'},
            'APM': {'name': 'Alternating Pressure', 'threshold': 0.18, 'description': 'Alternating pressure mattresses'},
            'Other LVA': {'name': 'Other Living Aids', 'threshold': 0.14, 'description': 'Other daily living aids'}
        }
    },
    'SUP': {
        'name': 'Support Products',
        'description': 'Braces, wraps, and support devices',
        'default_threshold': 0.11,  # 11% category average
        'subcategories': {
            'Splint': {'name': 'Splints', 'threshold': 0.14, 'description': 'Wrist, finger, thumb splints'},
            'Shoulder Brace': {'name': 'Shoulder Braces', 'threshold': 0.14, 'description': 'Shoulder immobilizers and supports'},
            'Groin': {'name': 'Groin Supports', 'threshold': 0.20, 'description': 'Groin wraps, thigh compression'},
            'Thigh': {'name': 'Thigh Supports', 'threshold': 0.13, 'description': 'Thigh sleeves and compression'},
            'Shin Support': {'name': 'Shin Guards', 'threshold': 0.12, 'description': 'Shin splint sleeves and guards'},
            'Ankle Wrap': {'name': 'Ankle Supports', 'threshold': 0.11, 'description': 'Ankle wraps and braces'},
            'Wrist': {'name': 'Wrist Supports', 'threshold': 0.11, 'description': 'Wrist braces and wraps'},
            'Wraps': {'name': 'General Wraps', 'threshold': 0.11, 'description': 'Elastic bandages and compression wraps'},
            'Strap': {'name': 'Support Straps', 'threshold': 0.11, 'description': 'Knee straps, patella bands'},
            'Sling': {'name': 'Arm Slings', 'threshold': 0.13, 'description': 'Arm and shoulder slings'},
            'Gloves': {'name': 'Support Gloves', 'threshold': 0.075, 'description': 'Compression and arthritis gloves'},
            'Post Op Shoes': {'name': 'Post-Op Shoes', 'threshold': 0.24, 'description': 'Post-surgical walking shoes'},
            'Other SUP': {'name': 'Other Support', 'threshold': 0.11, 'description': 'Other support products'}
        }
    },
    'RHB': {
        'name': 'Rehabilitation',
        'description': 'Rehabilitation and therapy products',
        'default_threshold': 0.086,  # 8.6% category average
        'subcategories': {
            'Post Op Shoes': {'name': 'Post-Op Shoes', 'threshold': 0.24, 'description': 'Post-surgical walking boots'},
            'Transfer Belts': {'name': 'Transfer Belts', 'threshold': 0.105, 'description': 'Gait belts and transfer aids'},
            'Shoulder Pulley': {'name': 'Shoulder Pulleys', 'threshold': 0.04, 'description': 'Shoulder exercise pulleys'},
            'Ice/Bracing': {'name': 'Ice & Bracing', 'threshold': 0.06, 'description': 'Ice packs, hot/cold therapy'},
            'Massage Ball': {'name': 'Massage Therapy', 'threshold': 0.05, 'description': 'Massage balls, foam rollers'},
            'Gauze': {'name': 'Wound Care', 'threshold': 0.025, 'description': 'Gauze, bandages, wound care'},
            'Wrist': {'name': 'Wrist Rehab', 'threshold': 0.072, 'description': 'Wrist rehab and braces'},
            'Splint': {'name': 'Rehab Splints', 'threshold': 0.14, 'description': 'Rehabilitation splints'},
            'Other RHB': {'name': 'Other Rehab', 'threshold': 0.09, 'description': 'Other rehabilitation products'}
        }
    },
    'CSH': {
        'name': 'Cushions',
        'description': 'Seat cushions and padding',
        'default_threshold': 0.122,  # 12.2% category average
        'subcategories': {
            'Chair Cushion': {'name': 'Chair Cushions', 'threshold': 0.085, 'description': 'Seat and chair cushions'},
            'Wheelchair Cushion': {'name': 'Wheelchair Cushions', 'threshold': 0.085, 'description': 'Wheelchair seat cushions'},
            'Commode Cushion': {'name': 'Commode Cushions', 'threshold': 0.16, 'description': 'Toilet seat cushions'},
            'Crutch Pads': {'name': 'Crutch Pads', 'threshold': 0.13, 'description': 'Crutch padding and grips'},
            'Other CSH': {'name': 'Other Cushions', 'threshold': 0.12, 'description': 'Other cushion products'}
        }
    },
    'INS': {
        'name': 'Insoles & Foot Care',
        'description': 'Foot care, insoles, and orthotics',
        'default_threshold': 0.106,  # 10.6% category average
        'subcategories': {
            'Toe Separators': {'name': 'Toe Separators', 'threshold': 0.05, 'description': 'Toe spacers and separators'},
            'Bunion': {'name': 'Bunion Care', 'threshold': 0.14, 'description': 'Bunion pads and correctors'},
            'Splint': {'name': 'Foot Splints', 'threshold': 0.14, 'description': 'Toe and foot splints'},
            'Wraps': {'name': 'Foot Wraps', 'threshold': 0.11, 'description': 'Foot and ankle wraps'},
            'Other INS': {'name': 'Other Foot Care', 'threshold': 0.11, 'description': 'Other insoles and foot care'}
        }
    },
    'CAN': {
        'name': 'Canes',
        'description': 'Walking canes and accessories',
        'default_threshold': 0.075,  # 7.5% category average
        'subcategories': {
            'Standard Canes': {'name': 'Standard Canes', 'threshold': 0.075, 'description': 'Single-point canes'},
            'Quad Canes': {'name': 'Quad Canes', 'threshold': 0.075, 'description': 'Four-point base canes'},
            'Folding Canes': {'name': 'Folding Canes', 'threshold': 0.075, 'description': 'Collapsible travel canes'},
            'Other CAN': {'name': 'Other Canes', 'threshold': 0.075, 'description': 'Other cane types'}
        }
    },
    'B2B': {
        'name': 'B2B Products',
        'description': 'Business-to-business and wholesale products',
        'default_threshold': 0.025,  # 2.5% B2B threshold
        'subcategories': {
            'B2B General': {'name': 'B2B General', 'threshold': 0.025, 'description': 'General B2B products'}
        }
    },
    'Other': {
        'name': 'Other Products',
        'description': 'Uncategorized products',
        'default_threshold': 0.10,  # 10% default
        'subcategories': {
            'Uncategorized': {'name': 'Uncategorized', 'threshold': 0.10, 'description': 'Products pending categorization'}
        }
    }
}


# Screening thresholds
SCREENING_THRESHOLDS = {
    'return_rate': {
        'B2B': 0.025,
        'INS': 0.106,
        'RHB': 0.086,
        'LVA': 0.138,
        'MOB': 0.096,
        'CSH': 0.122,
        'SUP': 0.11,
        'CAN': 0.075,
        'Other': 0.10,
    },
    'ncx_rate': 0.02,  # 2% NCX rate threshold
    'star_rating': 3.8,  # Below this triggers review
    'cost_threshold': 10000,  # Annual cost threshold for priority
    'ncx_orders_min': 10,  # Minimum NCX orders to consider
    'safety_keywords': ['brake', 'fall', 'collapse', 'unstable', 'shock', 'burn', 'injury', 'hazard', 'dangerous', 'unsafe', 'cut', 'pinch', 'trap']
}


# Priority scoring weights
PRIORITY_WEIGHTS = {
    'safety_risk': 40,
    'financial_impact': 25,
    'return_rate_severity': 20,
    'customer_volume': 10,
    'brand_risk': 5
}


def get_category_options():
    """Get list of category options for UI dropdowns"""
    options = []
    for cat_code, cat_data in PRODUCT_CATEGORIES.items():
        options.append(f"{cat_code} - {cat_data['name']}")
    return options


def get_subcategory_options(category_code: str):
    """Get list of subcategory options for a given category"""
    if category_code not in PRODUCT_CATEGORIES:
        return ["Other"]

    subcats = PRODUCT_CATEGORIES[category_code].get('subcategories', {})
    return [f"{subcat_data['name']}" for subcat_data in subcats.values()]


def get_threshold_for_product(category_code: str, subcategory_name: str = None) -> float:
    """
    Get the return rate threshold for a product based on category and subcategory.

    Args:
        category_code: Main category code (MOB, LVA, SUP, etc.)
        subcategory_name: Optional subcategory name

    Returns:
        Threshold as decimal (e.g., 0.10 for 10%)
    """
    if category_code not in PRODUCT_CATEGORIES:
        return 0.10  # Default 10%

    cat_data = PRODUCT_CATEGORIES[category_code]

    # If no subcategory specified, return category default
    if not subcategory_name:
        return cat_data['default_threshold']

    # Look for matching subcategory
    for subcat_key, subcat_data in cat_data.get('subcategories', {}).items():
        if subcat_data['name'] == subcategory_name or subcat_key.lower() == subcategory_name.lower():
            return subcat_data['threshold']

    # Fall back to category default
    return cat_data['default_threshold']


def get_all_thresholds_flat():
    """Get a flat dictionary of all thresholds for quick lookup"""
    thresholds = {}
    for cat_code, cat_data in PRODUCT_CATEGORIES.items():
        thresholds[cat_code] = cat_data['default_threshold']
        for subcat_key, subcat_data in cat_data.get('subcategories', {}).items():
            # Key by both the key and the name
            thresholds[f"{cat_code}_{subcat_key}"] = subcat_data['threshold']
            thresholds[subcat_data['name']] = subcat_data['threshold']
    return thresholds


def initialize_wizard_state():
    """Initialize the wizard state in session state if not present"""
    if 'wizard_state' not in st.session_state:
        st.session_state.wizard_state = {
            'step': 0,
            'case_data': {},
            'ai_recommendation': None,
            'priority_score': 0,
            'override_requested': False,
            'thresholds': SCREENING_THRESHOLDS.copy()
        }


def render_ai_screening_wizard(tracker, QualityTrackerCase):
    """
    AI-Assisted Quality Case Screening Wizard

    Guides user through SOP-based screening with AI recommendations.
    Populates Smartsheet-compatible case data with priority assessment.

    Args:
        tracker: QualityTrackerManager instance
        QualityTrackerCase: The case class for creating new cases
    """

    initialize_wizard_state()
    wizard = st.session_state.wizard_state

    # Wizard Header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #8e44ad 0%, #3498db 100%);
                padding: 1.5rem; border-radius: 10px; margin: 1rem 0;
                box-shadow: 0 4px 8px rgba(0,0,0,0.15);">
        <h3 style="color: white; font-family: 'Poppins', sans-serif; margin-bottom: 0.5rem; font-weight: 600;">
            🧙‍♂️ AI-Assisted Quality Case Screening Wizard
        </h3>
        <p style="color: rgba(255,255,255,0.9); font-family: 'Poppins', sans-serif; font-size: 0.95em; margin: 0;">
            Step-by-step SOP-guided case creation with AI priority recommendations
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Expandable wizard content - start expanded if wizard is in progress
    with st.expander("🚀 Start New Case Screening", expanded=wizard['step'] > 0):

        # Progress indicator
        steps = ["📋 Flag Source", "📊 Product & Metrics", "🔍 Issue Analysis", "📝 Action Planning", "🎯 Priority Review"]
        current_step = wizard['step']

        # Progress bar
        progress_cols = st.columns(len(steps))
        for i, (col, step_name) in enumerate(zip(progress_cols, steps)):
            with col:
                if i < current_step:
                    st.markdown(f"<div style='text-align:center; color:#27ae60; font-size:0.8em;'>✅ {step_name}</div>", unsafe_allow_html=True)
                elif i == current_step:
                    st.markdown(f"<div style='text-align:center; color:#3498db; font-weight:bold; font-size:0.85em;'>➡️ {step_name}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='text-align:center; color:#95a5a6; font-size:0.8em;'>⏳ {step_name}</div>", unsafe_allow_html=True)

        st.markdown("---")

        # Render the appropriate step
        if current_step == 0:
            render_step_flag_source(wizard, tracker)
        elif current_step == 1:
            render_step_product_metrics(wizard, tracker)
        elif current_step == 2:
            render_step_issue_analysis(wizard, tracker)
        elif current_step == 3:
            render_step_action_planning(wizard, tracker)
        elif current_step == 4:
            render_step_priority_review(wizard, tracker, QualityTrackerCase)


def render_step_flag_source(wizard, tracker):
    """Step 1: Flag Source & Initial Assessment"""
    st.markdown("### 📋 Step 1: Flag Source & Initial Assessment")
    st.markdown("""
    <div style="background: rgba(52,152,219,0.1); border-left: 4px solid #3498db; padding: 1rem; margin: 1rem 0; border-radius: 4px;">
        <strong>SOP Question:</strong> What triggered this quality flag? Understanding the source helps determine the investigation approach.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        flag_source = st.selectbox(
            "Primary Flag Source*",
            ["Returns Analysis", "B2B Sales Feedback", "Reviews Analysis", "Customer Service Escalation", "Internal QA Audit", "Regulatory Alert", "Other"],
            key="wiz_flag_source",
            help="Which report or source identified this potential quality issue?"
        )

        flag_source_detail = st.text_input(
            "Flag Source Detail (Internal)",
            placeholder="e.g., High Return Rate, Badge Warning, Customer Complaint",
            key="wiz_flag_detail",
            help="Specific trigger within the flag source"
        )

    with col2:
        flag_date = st.date_input(
            "Date Flag Identified",
            value=datetime.now().date(),
            key="wiz_flag_date"
        )

        urgency = st.selectbox(
            "Initial Urgency Assessment",
            ["🔴 Critical - Safety/Regulatory", "🟠 High - Financial Impact", "🟡 Medium - Quality Concern", "🟢 Low - Monitoring"],
            key="wiz_urgency"
        )

    st.markdown("#### 💡 AI Context Helper")
    if st.button("🤖 Get AI Guidance for This Flag Source", key="ai_flag_help"):
        if tracker.ai_analyzer:
            with st.spinner("AI analyzing flag source..."):
                prompt = f"""Based on this quality flag source, provide brief guidance:

Flag Source: {flag_source}
Detail: {flag_source_detail or 'Not specified'}
Urgency: {urgency}

Provide:
1. What data should be gathered next (2-3 bullet points)
2. Key questions to investigate
3. Typical root causes for this flag type

Keep response under 150 words."""

                guidance = tracker.ai_analyzer.generate_text(
                    prompt,
                    "You are a medical device quality expert following FDA 21 CFR 820 and ISO 13485 guidelines.",
                    mode='chat'
                )
                st.info(guidance)
        else:
            st.warning("AI analyzer not available. Continue with manual entry.")

    col_nav1, col_nav2 = st.columns([3, 1])
    with col_nav2:
        if st.button("Next →", key="step0_next", type="primary", use_container_width=True):
            wizard['case_data']['flag_source'] = flag_source
            wizard['case_data']['flag_source_1'] = flag_source_detail
            wizard['case_data']['flag_date'] = flag_date
            wizard['case_data']['urgency'] = urgency
            wizard['step'] = 1
            st.rerun()


def render_step_product_metrics(wizard, tracker):
    """Step 2: Product Information & Quality Metrics"""
    st.markdown("### 📊 Step 2: Product Information & Quality Metrics")
    st.markdown("""
    <div style="background: rgba(52,152,219,0.1); border-left: 4px solid #3498db; padding: 1rem; margin: 1rem 0; border-radius: 4px;">
        <strong>SOP Question:</strong> What product is affected and what are the current quality metrics?
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##### Product Identification")
    col1, col2, col3 = st.columns(3)
    with col1:
        product_name = st.text_input("Product Name*", placeholder="e.g., Vive Knee Scooter", key="wiz_product")
    with col2:
        sku = st.text_input("SKU*", placeholder="e.g., MOB-1234", key="wiz_sku")
    with col3:
        asin = st.text_input("ASIN", placeholder="e.g., B07XXXXXXX", key="wiz_asin")

    st.markdown("##### Product Classification")
    st.caption("Select category and product type - thresholds are automatically applied based on your selection")

    col4, col5 = st.columns(2)
    with col4:
        # Main category selection
        category_options = [f"{code} - {data['name']}" for code, data in PRODUCT_CATEGORIES.items()]
        category = st.selectbox(
            "Main Category*",
            category_options,
            key="wiz_category",
            help="Select the main product category (MOB, LVA, SUP, etc.)"
        )
        cat_code = category.split(" - ")[0] if " - " in category else category

    with col5:
        # Subcategory/Product Type based on main category
        if cat_code in PRODUCT_CATEGORIES:
            subcat_options = [data['name'] for data in PRODUCT_CATEGORIES[cat_code]['subcategories'].values()]
            subcategory = st.selectbox(
                "Product Type*",
                subcat_options,
                key="wiz_subcategory",
                help="Select the specific product type for accurate threshold"
            )
        else:
            subcategory = st.selectbox("Product Type", ["Other"], key="wiz_subcategory")

    # Get threshold for selected category/subcategory
    selected_threshold = get_threshold_for_product(cat_code, subcategory)

    # Show threshold info
    st.markdown(f"""
    <div style="background: rgba(35,178,190,0.1); border: 2px solid #23b2be; padding: 0.8rem;
                border-radius: 8px; margin: 0.5rem 0;">
        <span style="font-weight: 600; color: #004366;">📊 Return Rate Threshold for {subcategory}:</span>
        <span style="font-size: 1.2em; font-weight: 700; color: #23b2be;"> {selected_threshold*100:.1f}%</span>
    </div>
    """, unsafe_allow_html=True)

    col6, col7 = st.columns(2)
    with col6:
        sales_channel = st.selectbox("Main Sales Channel", ["Amazon", "B2B", "Direct", "Multi-Channel"], key="wiz_channel")
    with col7:
        fulfilled_by = st.selectbox("Fulfilled By", ["FBA", "FBM", "Direct Ship", "Hybrid"], key="wiz_fulfilled")

    st.markdown("##### Quality Metrics")
    st.caption("Enter current metrics - these will be compared against the threshold above")

    col8, col9, col10, col11 = st.columns(4)
    with col8:
        return_rate = st.number_input(
            "Return Rate Amazon (%)",
            min_value=0.0, max_value=100.0, value=0.0, step=0.1,
            key="wiz_return_rate",
            help="Current Amazon return rate as percentage"
        )
    with col9:
        return_rate_b2b = st.number_input(
            "Return Rate B2B (%)",
            min_value=0.0, max_value=100.0, value=0.0, step=0.1,
            key="wiz_return_rate_b2b"
        )
    with col10:
        ncx_rate = st.number_input(
            "NCX Rate (%)",
            min_value=0.0, max_value=100.0, value=0.0, step=0.1,
            key="wiz_ncx_rate",
            help="Negative Customer Experience rate"
        )
    with col11:
        star_rating = st.number_input(
            "Star Rating",
            min_value=1.0, max_value=5.0, value=4.5, step=0.1,
            key="wiz_star_rating"
        )

    col12, col13, col14 = st.columns(3)
    with col12:
        ncx_orders = st.number_input("NCX Orders (count)", min_value=0, value=0, key="wiz_ncx_orders")
    with col13:
        total_orders = st.number_input("Total Orders (t30)", min_value=0, value=0, key="wiz_total_orders")
    with col14:
        badge_displayed = st.selectbox("Return Badge Displayed?", ["No", "Yes", "Unknown"], key="wiz_badge")

    # Real-time threshold check
    st.markdown("##### 🎯 Real-Time Threshold Analysis")

    col_thresh1, col_thresh2, col_thresh3 = st.columns(3)
    with col_thresh1:
        if return_rate > 0:
            if return_rate / 100 > selected_threshold:
                st.error(f"⚠️ Return Rate {return_rate}% EXCEEDS threshold ({selected_threshold*100}%)")
            else:
                st.success(f"✅ Return Rate {return_rate}% within threshold ({selected_threshold*100}%)")
    with col_thresh2:
        if ncx_rate > 0:
            if ncx_rate / 100 > wizard['thresholds']['ncx_rate']:
                st.error(f"⚠️ NCX Rate {ncx_rate}% EXCEEDS threshold ({wizard['thresholds']['ncx_rate']*100}%)")
            else:
                st.success(f"✅ NCX Rate within threshold")
    with col_thresh3:
        if star_rating < wizard['thresholds']['star_rating']:
            st.warning(f"⚠️ Star Rating {star_rating} below {wizard['thresholds']['star_rating']}")
        else:
            st.success(f"✅ Star Rating acceptable")

    col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
    with col_nav1:
        if st.button("← Back", key="step1_back", use_container_width=True):
            wizard['step'] = 0
            st.rerun()
    with col_nav3:
        if st.button("Next →", key="step1_next", type="primary", use_container_width=True):
            if not product_name or not sku:
                st.error("Product Name and SKU are required")
            else:
                wizard['case_data']['product_name'] = product_name
                wizard['case_data']['sku'] = sku
                wizard['case_data']['asin'] = asin
                wizard['case_data']['category'] = category
                wizard['case_data']['category_code'] = cat_code
                wizard['case_data']['subcategory'] = subcategory
                wizard['case_data']['threshold'] = selected_threshold
                wizard['case_data']['sales_channel'] = sales_channel
                wizard['case_data']['fulfilled_by'] = fulfilled_by
                wizard['case_data']['return_rate'] = return_rate / 100
                wizard['case_data']['return_rate_b2b'] = return_rate_b2b / 100
                wizard['case_data']['ncx_rate'] = ncx_rate / 100
                wizard['case_data']['star_rating'] = star_rating
                wizard['case_data']['ncx_orders'] = ncx_orders
                wizard['case_data']['total_orders'] = total_orders
                wizard['case_data']['badge_displayed'] = badge_displayed
                wizard['step'] = 2
                st.rerun()


def render_step_issue_analysis(wizard, tracker):
    """Step 3: Issue Analysis & Root Cause Investigation"""
    st.markdown("### 🔍 Step 3: Issue Analysis & Root Cause Investigation")
    st.markdown("""
    <div style="background: rgba(52,152,219,0.1); border-left: 4px solid #3498db; padding: 1rem; margin: 1rem 0; border-radius: 4px;">
        <strong>SOP Questions:</strong>
        <ul style="margin: 0.5rem 0 0 0; padding-left: 1.5rem;">
            <li>Are there actionable return reasons/complaints/negative reviews?</li>
            <li>What are the top 3 issues identified?</li>
            <li>Is this a safety concern?</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##### Top Issues Identified")
    st.caption("List the primary quality issues from returns/reviews/complaints")

    top_issues = st.text_area(
        "Top Issue(s)*",
        placeholder="Enter the main quality issues identified, e.g.:\n1. Wheel locking mechanism fails after 2-3 months\n2. Paint peeling on handlebars\n3. Assembly instructions unclear",
        height=120,
        key="wiz_top_issues",
        help="Be specific - these will be used for AI analysis and tracking"
    )

    st.markdown("##### Issue Classification")
    col1, col2 = st.columns(2)
    with col1:
        actionable = st.radio(
            "Are these actionable issues?",
            ["Yes - Clear corrective action possible", "Partially - Some aspects actionable", "No - Cosmetic/preference issues", "Needs Investigation"],
            key="wiz_actionable"
        )

        issue_trend = st.selectbox(
            "Issue Trend",
            ["New - First occurrence", "Recurring - Seen before", "Worsening - Getting worse", "Improving - Getting better", "Stable - No change"],
            key="wiz_trend"
        )

    with col2:
        # Safety keyword detection
        safety_detected = False
        if top_issues:
            for keyword in wizard['thresholds']['safety_keywords']:
                if keyword.lower() in top_issues.lower():
                    safety_detected = True
                    break

        if safety_detected:
            st.error("⚠️ **SAFETY KEYWORDS DETECTED** - This case requires immediate attention")
            safety_concern = st.selectbox(
                "Confirm Safety Classification",
                ["🔴 CONFIRMED SAFETY ISSUE", "🟠 Potential Safety Concern", "🟢 False Positive - Not Safety Related"],
                key="wiz_safety"
            )
        else:
            safety_concern = st.selectbox(
                "Safety Classification",
                ["🟢 No Safety Concern", "🟠 Potential Safety Concern", "🔴 CONFIRMED SAFETY ISSUE"],
                key="wiz_safety"
            )

        issue_source = st.multiselect(
            "Issue Evidence Sources",
            ["Amazon Returns Data", "Customer Reviews", "B2B Feedback", "Customer Service Tickets", "QA Inspection", "Regulatory Report"],
            key="wiz_evidence"
        )

    st.markdown("##### 🤖 AI Issue Analysis")
    if st.button("Analyze Issues with AI", key="ai_analyze_issues"):
        if tracker.ai_analyzer and top_issues:
            with st.spinner("AI analyzing issues..."):
                prompt = f"""Analyze these quality issues for a medical device product:

Product: {wizard['case_data'].get('product_name', 'Unknown')} ({wizard['case_data'].get('sku', 'Unknown')})
Category: {wizard['case_data'].get('category', 'Unknown')}
Return Rate: {wizard['case_data'].get('return_rate', 0)*100:.1f}%

Top Issues Reported:
{top_issues}

Provide:
1. **Root Cause Hypothesis** (most likely 2-3 causes)
2. **Severity Assessment** (Critical/High/Medium/Low)
3. **Recommended Investigation Steps** (3-4 specific actions)
4. **Similar Historical Patterns** (common in this product category?)

Keep response concise and actionable."""

                analysis = tracker.ai_analyzer.generate_text(
                    prompt,
                    "You are a medical device quality engineer with expertise in root cause analysis and CAPA processes.",
                    mode='chat'
                )

                st.markdown("""
                <div style="background: #f8f9fa; border: 2px solid #8e44ad; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                """, unsafe_allow_html=True)
                st.markdown(analysis)
                st.markdown("</div>", unsafe_allow_html=True)

                wizard['case_data']['ai_analysis'] = analysis
        elif not top_issues:
            st.warning("Please enter top issues first")
        else:
            st.warning("AI analyzer not available")

    col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
    with col_nav1:
        if st.button("← Back", key="step2_back", use_container_width=True):
            wizard['step'] = 1
            st.rerun()
    with col_nav3:
        if st.button("Next →", key="step2_next", type="primary", use_container_width=True):
            if not top_issues:
                st.error("Please enter the top issues")
            else:
                wizard['case_data']['top_issues'] = top_issues
                wizard['case_data']['actionable'] = actionable
                wizard['case_data']['issue_trend'] = issue_trend
                wizard['case_data']['safety_concern'] = safety_concern
                wizard['case_data']['issue_sources'] = issue_source
                wizard['step'] = 3
                st.rerun()


def render_step_action_planning(wizard, tracker):
    """Step 4: Action Planning & Notifications"""
    st.markdown("### 📝 Step 4: Action Planning & Notifications")
    st.markdown("""
    <div style="background: rgba(52,152,219,0.1); border-left: 4px solid #3498db; padding: 1rem; margin: 1rem 0; border-radius: 4px;">
        <strong>SOP Questions:</strong>
        <ul style="margin: 0.5rem 0 0 0; padding-left: 1.5rem;">
            <li>What is our plan to address these issues?</li>
            <li>Who needs to be notified?</li>
            <li>What is the follow-up timeline?</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##### Planned Actions")
    action_taken = st.text_area(
        "Action Plan / Actions Taken",
        placeholder="Describe the corrective actions planned or already taken:\n• Contacted supplier about wheel mechanism\n• Initiated quality hold on current inventory\n• Scheduled engineering review",
        height=100,
        key="wiz_action"
    )

    col1, col2 = st.columns(2)
    with col1:
        action_date = st.date_input("Date Action Taken/Planned", value=datetime.now().date(), key="wiz_action_date")
        follow_up_date = st.date_input(
            "Follow-Up Date",
            value=datetime.now().date() + timedelta(days=30),
            key="wiz_followup"
        )

    with col2:
        case_status = st.selectbox(
            "Case Status",
            ["Open - New", "Open - In Progress", "Open - Awaiting Response", "Monitoring", "Closed - Resolved", "Closed - No Action"],
            key="wiz_status"
        )

    st.markdown("##### Notifications")
    col3, col4 = st.columns(2)
    with col3:
        listing_notified = st.selectbox("Listing Manager Notified?", ["No", "Yes", "N/A"], key="wiz_listing")
    with col4:
        product_dev_notified = st.selectbox("Product Dev Notified?", ["No", "Yes", "N/A"], key="wiz_proddev")

    notification_notes = st.text_area(
        "Notification Notes / Additional Comments",
        placeholder="Any additional context, escalation notes, or communication details",
        height=80,
        key="wiz_notes"
    )

    st.markdown("##### Financial Impact (Leadership Only)")
    show_financial = st.checkbox("Include Financial Data", value=st.session_state.get('show_leadership_fields', False), key="wiz_show_financial")

    if show_financial:
        col5, col6 = st.columns(2)
        with col5:
            cost_of_refunds = st.number_input(
                "Cost of Refunds (Annualized $)",
                min_value=0.0, value=0.0, step=100.0,
                key="wiz_cost"
            )
        with col6:
            savings_captured = st.number_input(
                "12m Savings Captured ($)",
                min_value=0.0, value=0.0, step=100.0,
                key="wiz_savings"
            )

        col7, col8 = st.columns(2)
        with col7:
            financial_impact = st.text_input(
                "Financial Impact Description",
                placeholder="e.g., High impact - lost sales, refund costs",
                key="wiz_financial_impact",
                help="Text description of the overall financial impact"
            )
        with col8:
            defective_inventory = st.number_input(
                "Total Defective Inventory (units)",
                min_value=0, value=0, step=1,
                key="wiz_defective_inventory",
                help="Count of defective units in inventory"
            )
    else:
        cost_of_refunds = 0.0
        savings_captured = 0.0
        financial_impact = ""
        defective_inventory = 0

    st.markdown("##### Completion Timeline")
    col_dates1, col_dates2 = st.columns(2)
    with col_dates1:
        estimated_completion = st.date_input(
            "Estimated Completion Date",
            value=None,
            key="wiz_est_completion",
            help="Target date for resolving this case"
        )
    with col_dates2:
        actual_completion = st.date_input(
            "Actual Completion Date",
            value=None,
            key="wiz_actual_completion",
            help="Leave blank if case is not yet completed"
        )

    # AI Action Recommendation
    if st.button("🤖 Get AI Action Recommendations", key="ai_action_rec"):
        if tracker.ai_analyzer:
            with st.spinner("Generating recommendations..."):
                prompt = f"""Based on this quality case, recommend specific actions:

Product: {wizard['case_data'].get('product_name')} ({wizard['case_data'].get('sku')})
Issues: {wizard['case_data'].get('top_issues')}
Safety Level: {wizard['case_data'].get('safety_concern')}
Return Rate: {wizard['case_data'].get('return_rate', 0)*100:.1f}%
Current Plan: {action_taken or 'None specified'}

Provide:
1. **Immediate Actions** (within 24-48 hours)
2. **Short-term Actions** (within 1-2 weeks)
3. **Long-term Prevention** (process/design changes)
4. **Stakeholders to Notify** (specific roles)

Be specific and actionable."""

                recommendations = tracker.ai_analyzer.generate_text(
                    prompt,
                    "You are a quality management expert specializing in CAPA and corrective actions for medical devices.",
                    mode='chat'
                )
                st.info(recommendations)

    col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
    with col_nav1:
        if st.button("← Back", key="step3_back", use_container_width=True):
            wizard['step'] = 2
            st.rerun()
    with col_nav3:
        if st.button("Next →", key="step3_next", type="primary", use_container_width=True):
            wizard['case_data']['action_taken'] = action_taken
            wizard['case_data']['action_date'] = action_date
            wizard['case_data']['follow_up_date'] = follow_up_date
            wizard['case_data']['case_status'] = case_status
            wizard['case_data']['listing_notified'] = listing_notified
            wizard['case_data']['product_dev_notified'] = product_dev_notified
            wizard['case_data']['notification_notes'] = notification_notes
            wizard['case_data']['cost_of_refunds'] = cost_of_refunds
            wizard['case_data']['savings_captured'] = savings_captured
            # New fields
            wizard['case_data']['financial_impact'] = financial_impact
            wizard['case_data']['defective_inventory'] = defective_inventory
            wizard['case_data']['estimated_completion'] = estimated_completion
            wizard['case_data']['actual_completion'] = actual_completion
            wizard['step'] = 4
            st.rerun()


def render_step_priority_review(wizard, tracker, QualityTrackerCase):
    """Step 5: AI Priority Assessment & Case Submission"""
    st.markdown("### 🎯 Step 5: AI Priority Assessment & Case Submission")

    # Calculate priority score
    data = wizard['case_data']
    priority_score = 0
    priority_factors = []

    # Safety risk (40 points max)
    if '🔴' in data.get('safety_concern', ''):
        priority_score += 40
        priority_factors.append(("🔴 Safety Issue Confirmed", 40))
    elif '🟠' in data.get('safety_concern', ''):
        priority_score += 25
        priority_factors.append(("🟠 Potential Safety Concern", 25))

    # Return rate severity (20 points max)
    threshold = data.get('threshold', 0.10)
    subcategory = data.get('subcategory', 'Unknown')
    return_rate = data.get('return_rate', 0)
    if return_rate > 0:
        exceedance = (return_rate - threshold) / threshold if threshold > 0 else 0
        if exceedance > 0.5:  # 50%+ over threshold
            priority_score += 20
            priority_factors.append((f"Return Rate {exceedance*100:.0f}% over {subcategory} threshold ({threshold*100:.1f}%)", 20))
        elif exceedance > 0.25:
            priority_score += 15
            priority_factors.append((f"Return Rate {exceedance*100:.0f}% over {subcategory} threshold", 15))
        elif exceedance > 0:
            priority_score += 10
            priority_factors.append((f"Return Rate above {subcategory} threshold", 10))

    # Financial impact (25 points max)
    cost = data.get('cost_of_refunds', 0)
    if cost >= 50000:
        priority_score += 25
        priority_factors.append((f"High Financial Impact (${cost:,.0f})", 25))
    elif cost >= 25000:
        priority_score += 18
        priority_factors.append((f"Moderate Financial Impact (${cost:,.0f})", 18))
    elif cost >= 10000:
        priority_score += 12
        priority_factors.append((f"Financial Impact (${cost:,.0f})", 12))

    # Customer volume (10 points max)
    ncx_orders = data.get('ncx_orders', 0)
    if ncx_orders >= 100:
        priority_score += 10
        priority_factors.append((f"High NCX Volume ({ncx_orders} orders)", 10))
    elif ncx_orders >= 50:
        priority_score += 7
        priority_factors.append((f"Moderate NCX Volume ({ncx_orders} orders)", 7))
    elif ncx_orders >= 20:
        priority_score += 4
        priority_factors.append((f"NCX Volume ({ncx_orders} orders)", 4))

    # Star rating impact (5 points max)
    star = data.get('star_rating', 5.0)
    if star < 3.5:
        priority_score += 5
        priority_factors.append((f"Low Star Rating ({star})", 5))
    elif star < 3.8:
        priority_score += 3
        priority_factors.append((f"Below Target Star Rating ({star})", 3))

    # Determine priority level
    if priority_score >= 70:
        priority_level = "🔴 CRITICAL"
        priority_num = 1
        should_add = True
    elif priority_score >= 50:
        priority_level = "🟠 HIGH"
        priority_num = 2
        should_add = True
    elif priority_score >= 30:
        priority_level = "🟡 MEDIUM"
        priority_num = 3
        should_add = True
    else:
        priority_level = "🟢 LOW"
        priority_num = 4
        should_add = False

    wizard['priority_score'] = priority_score

    # Display priority assessment
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {'#e74c3c' if priority_score >= 70 else '#f39c12' if priority_score >= 50 else '#f1c40f' if priority_score >= 30 else '#27ae60'} 0%,
                {'#c0392b' if priority_score >= 70 else '#d68910' if priority_score >= 50 else '#d4ac0d' if priority_score >= 30 else '#1e8449'} 100%);
                padding: 1.5rem; border-radius: 10px; margin: 1rem 0; text-align: center;">
        <h2 style="color: white; margin: 0;">AI Priority Assessment: {priority_level}</h2>
        <h3 style="color: rgba(255,255,255,0.9); margin: 0.5rem 0;">Score: {priority_score}/100</h3>
    </div>
    """, unsafe_allow_html=True)

    # Show scoring breakdown
    with st.expander("📊 Priority Score Breakdown", expanded=True):
        for factor, points in priority_factors:
            st.markdown(f"• **{factor}**: +{points} points")
        if not priority_factors:
            st.info("No significant risk factors identified")

    # Case summary
    st.markdown("##### 📋 Case Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        **Product:** {data.get('product_name')} ({data.get('sku')})
        **Category:** {data.get('category_code', 'N/A')} → {data.get('subcategory', 'N/A')}
        **Threshold:** {data.get('threshold', 0)*100:.1f}%
        **Flag Source:** {data.get('flag_source')}
        **Return Rate:** {data.get('return_rate', 0)*100:.1f}%
        **Star Rating:** {data.get('star_rating', 'N/A')}
        """)
    with col2:
        st.markdown(f"""
        **Safety Level:** {data.get('safety_concern')}
        **Status:** {data.get('case_status')}
        **Follow-up:** {data.get('follow_up_date')}
        **Cost Impact:** ${data.get('cost_of_refunds', 0):,.0f}
        """)

    st.markdown(f"**Top Issues:** {data.get('top_issues', 'Not specified')[:200]}...")

    # AI recommendation
    if not should_add:
        st.markdown("""
        <div style="background: rgba(46,204,113,0.1); border: 2px solid #27ae60; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
            <h4 style="color: #27ae60; margin: 0;">🤖 AI Recommendation: MONITOR ONLY</h4>
            <p style="margin: 0.5rem 0 0 0;">Based on current metrics, this case does not meet the threshold for priority tracking.
            Consider monitoring and re-evaluating if metrics worsen.</p>
        </div>
        """, unsafe_allow_html=True)

        override = st.checkbox("⚠️ Override AI recommendation and add to tracker anyway", key="wiz_override")
        if override:
            override_reason = st.text_input("Reason for override*", placeholder="e.g., Executive request, Customer escalation", key="wiz_override_reason")
            should_add = True
            data['override_reason'] = override_reason
    else:
        st.markdown(f"""
        <div style="background: rgba(231,76,60,0.1); border: 2px solid #e74c3c; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
            <h4 style="color: #e74c3c; margin: 0;">🤖 AI Recommendation: ADD TO PRIORITY TRACKER</h4>
            <p style="margin: 0.5rem 0 0 0;">This case meets criteria for priority tracking based on the factors above.</p>
        </div>
        """, unsafe_allow_html=True)

    # Navigation and submission
    st.markdown("---")
    col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 2])
    with col_nav1:
        if st.button("← Back", key="step4_back", use_container_width=True):
            wizard['step'] = 3
            st.rerun()
    with col_nav2:
        if st.button("🗑️ Cancel", key="step4_cancel", use_container_width=True):
            wizard['step'] = 0
            wizard['case_data'] = {}
            st.rerun()
    with col_nav3:
        submit_label = "✅ Add to Priority Tracker" if should_add else "📋 Add to Tracker (Override)"
        if st.button(submit_label, key="step4_submit", type="primary", use_container_width=True, disabled=not should_add):
            # Create the case
            new_case = QualityTrackerCase()
            new_case.priority = priority_num
            new_case.product_name = data.get('product_name', '')
            new_case.main_sales_channel = data.get('sales_channel', '')
            new_case.asin = data.get('asin', '')
            new_case.sku = data.get('sku', '')
            new_case.fulfilled_by = data.get('fulfilled_by', '')
            new_case.ncx_rate = data.get('ncx_rate')
            new_case.ncx_orders = data.get('ncx_orders')
            new_case.total_orders_t30 = data.get('total_orders')
            new_case.star_rating_amazon = data.get('star_rating')
            new_case.return_rate_amazon = data.get('return_rate')
            new_case.return_rate_b2b = data.get('return_rate_b2b')
            new_case.flag_source_1 = data.get('flag_source_1', '')
            new_case.return_badge_displayed = data.get('badge_displayed', '')
            new_case.notification_notes = data.get('notification_notes', '')
            new_case.top_issues = data.get('top_issues', '')
            new_case.cost_of_refunds_annualized = data.get('cost_of_refunds')
            new_case.savings_captured_12m = data.get('savings_captured')
            new_case.action_taken = data.get('action_taken', '')
            new_case.date_action_taken = data.get('action_date')
            new_case.listing_manager_notified = data.get('listing_notified', '')
            new_case.product_dev_notified = data.get('product_dev_notified', '')
            new_case.flag_source = data.get('flag_source', '')
            new_case.follow_up_date = data.get('follow_up_date')
            new_case.case_status = data.get('case_status', 'Open - New')
            # New fields (columns 32-35)
            new_case.financial_impact = data.get('financial_impact', '')
            new_case.total_defective_inventory = data.get('defective_inventory')
            new_case.estimated_completion_date = data.get('estimated_completion')
            new_case.completion_date_actual = data.get('actual_completion')

            # Add to tracker
            tracker.add_case(new_case)
            st.session_state.tracker_cases = tracker.cases

            # Reset wizard
            wizard['step'] = 0
            wizard['case_data'] = {}

            st.success(f"✅ Case added to tracker: {new_case.product_name} ({new_case.sku}) - Priority: {priority_level}")
            st.balloons()
            st.rerun()
