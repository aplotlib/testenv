"""
Vive Health Quality Suite - Version 24.0
Enterprise-Grade Quality Management System

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TASK-BASED WORKFLOW (7 Tools):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Return Categorizer      - AI categorizes customer complaints
📑 B2B Report Generator    - Odoo export → B2B compliant report
📋 Quality Case Tracker    - Track cases, dual exports (Leadership/Company)
🧪 Quality Screening       - AI screening with TQM methodology
📦 Inventory Integration   - DOI & reorder point analysis
📚 Resources               - Regulatory links & quality guides
🌐 Global Recall Survey    - Worldwide regulatory intelligence with multi-language support

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPLIANCE: ISO 13485 | FDA 21 CFR 820 | EU MDR | UK MDR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Features:
- Task-based landing page with intuitive tool selection
- 🆕 v24.0: VoC Analysis integration with period-over-period sales trends (L30D)
- 🆕 v24.0: Amazon return rate fee threshold monitoring (2026 policy)
- 🆕 v24.0: Intuitive VoC-style import workflow for Quality Screening
- 🆕 v24.0: Dated sheet support for multi-period comparison
- 🆕 v24.0: Sales trend analysis (Increasing/Decreasing/Stable/New)
- 🆕 v24.0: Return rate change tracking with fee risk calculations
- 🆕 v24.0: Amazon badge visibility impact tracking
- v23.0: Multi-language search (ES, PT, DE, FR, JA, ZH, KO)
- v23.0: Auto-translation of international results to English
- v23.0: Enhanced FDA search with product codes and wildcards
- v23.0: 20+ international regulatory feeds (FDA, EMA, MHRA, TGA, PMDA, BfArM, ANSM, etc.)
- v23.0: EU Safety Gate (RAPEX), WHO alerts, IMDRF news
- Global Recall Surveillance: FDA, EU EMA, UK MHRA, Health Canada, ANVISA, CPSC
- FDA MAUDE adverse event search integration
- Google News RSS media monitoring for safety signals
- OFAC sanctions and watchlist checking
- Quick Case Evaluation Mode: 1-3 product SOP comparison with AI qualification
- ANOVA/MANOVA statistical analysis with p-values and post-hoc testing
- SPC Control Charting (CUSUM, Shewhart)
- Weighted Risk Scoring with FDA/ISO compliance
- AI-powered cross-case correlation and deep dive analysis
- Fuzzy threshold matching with custom profiles
- Bulk vendor email and investigation plan generation
- Smartsheet CAPA/Investigation/Rework exporters
- Inventory + Quality integration with DOI calculations
- 35-column Leadership Export with sensitive data protection
- Company-Wide Export (28 columns - excludes sensitive fields)
- State persistence and audit trail
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import io
from typing import Dict, List, Any, Optional, Tuple
import time
from collections import Counter, defaultdict
import re
import os
import gc
import json

# Visualization
try:
    import altair as alt
    ALTAIR_AVAILABLE = True
except ImportError:
    ALTAIR_AVAILABLE = False

# --- Custom Modules ---
try:
    from enhanced_ai_analysis import (
        EnhancedAIAnalyzer, AIProvider, FBA_REASON_MAP,
        MEDICAL_DEVICE_CATEGORIES, DeepDiveAnalyzer, BulkOperationsManager
    )
    from quality_analytics import (
        QualityAnalytics, QualityStatistics, SPCAnalysis, TrendAnalysis,
        RiskScoring, ActionDetermination, VendorEmailGenerator,
        InvestigationPlanGenerator, DataValidation,
        SOP_THRESHOLDS, parse_numeric, parse_percentage,
        fuzzy_match_category, generate_methodology_markdown
    )
    from inventory_integration import (
        OdooInventoryParser, PivotReturnReportParser,
        InventoryConfiguration, InventoryCalculator, IntegratedAnalyzer
    )
    from multilingual_vendor_comms import (
        MultilingualVendorCommunicator, EnglishLevel, TargetLanguage, LANGUAGE_INFO
    )
    from product_matching import ProductMatcher
    from regulatory_compliance import RegulatoryComplianceAnalyzer, REGULATORY_MARKETS
    from quality_cases_dashboard import (
        QualityCase, QualityCasesDashboard, REPORT_CRITERIA,
        generate_demo_cases
    )
    from quality_tracker_manager import (
        QualityTrackerManager, QualityTrackerCase,
        ALL_COLUMNS_LEADERSHIP, ALL_COLUMNS_COMPANY_WIDE,
        LEADERSHIP_ONLY_COLUMNS, generate_demo_cases as generate_demo_tracker_cases
    )
    from quality_resources import QUALITY_RESOURCES, get_total_link_count
    # Import new modular components
    from advanced_analytics import (
        render_root_cause_analysis as rca_render,
        render_capa_management as capa_render,
        render_risk_analysis_fmea as fmea_render,
        render_predictive_analytics as predictive_render
    )
    from ai_screening_wizard import (
        render_ai_screening_wizard as wizard_render,
        PRODUCT_CATEGORIES, SCREENING_THRESHOLDS, PRIORITY_WEIGHTS,
        get_category_options, get_subcategory_options,
        get_threshold_for_product, get_all_thresholds_flat
    )
    from src.services.voc_analysis_integration import (
        VoCAnalysisService, ProductTrendAnalysis, AMAZON_RETURN_RATE_THRESHOLDS
    )
    AI_AVAILABLE = True
    MODULAR_IMPORTS = True
except ImportError as e:
    AI_AVAILABLE = False
    MODULAR_IMPORTS = False
    print(f"Module Missing: {e}")

# Check optional imports
try:
    import xlsxwriter
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- App Configuration ---
st.set_page_config(
    page_title="Vive Health Quality Suite",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

APP_CONFIG = {
    'title': 'Vive Health Quality Suite',
    'version': '24.0',
    'chunk_sizes': [100, 250, 500, 1000],
    'default_chunk': 500,
}

# Colors - Vive Health Brand Guide
COLORS = {
    'primary': '#23b2be',      # Vive Turquoise (Pantone P 121-6 C)
    'secondary': '#004366',    # Navy Blue (Pantone P 111-16 C)
    'accent': '#EB3300',       # Red/Orange (Pantone P 2028 C)
    'success': '#23b2be',      # Using Vive Turquoise for success
    'warning': '#F0B323',      # Yellow/Gold (Pantone 7409 C)
    'danger': '#EB3300',       # Red/Orange for alerts
    'dark': '#004366',         # Navy for dark elements
    'light': '#1A1A2E',        # Keep original for backgrounds
    'text': '#E0E0E0',         # Keep original for readability
    'muted': '#777473',        # Gray (Pantone P 172-9 C)
    'cost': '#23b2be'          # Vive Turquoise
}

# Quality categories (For Tab 1 Analysis)
QUALITY_CATEGORIES = [
    'Product Defects/Quality',
    'Performance/Effectiveness',
    'Missing Components',
    'Design/Material Issues',
    'Stability/Positioning Issues',
    'Medical/Health Concerns'
]

# AI Provider options - OpenAI default for Tab 3
AI_PROVIDER_OPTIONS = {
    'OpenAI GPT-3.5 (Default)': AIProvider.OPENAI,
    'Claude Haiku (Fast)': AIProvider.FASTEST,
    'Claude Sonnet': AIProvider.CLAUDE,
    'Both (Consensus)': AIProvider.BOTH
}

# Source of Flag options for tracking how issues came to attention
SOURCE_OF_FLAG_OPTIONS = [
    "Support Ticket",
    "Amazon Review", 
    "B2B Sales Rep",
    "Internal Meeting",
    "Internal Request",
    "QA Inspection",
    "Warehouse Report",
    "Customer Complaint",
    "Routine Screening",
    "Management Request",
    "Other (specify)"
]

# Statistical Analysis Options with clear explanations
STATISTICAL_ANALYSIS_OPTIONS = {
    "Auto (AI Recommended)": {
        "description": "AI analyzes your data and recommends the best statistical test",
        "when_to_use": "Best default choice - let the system decide based on your data characteristics",
        "example": "AI might suggest MANOVA if you have multiple metrics, or Kruskal-Wallis if data is non-normal",
        "tooltip": "🤖 AI picks the right test for your data automatically - perfect if you're unsure which statistical method to use"
    },
    "ANOVA (Analysis of Variance)": {
        "description": "Compares average return rates across different categories to determine if any category is statistically different from others",
        "when_to_use": "When comparing ONE metric (like return rate) across multiple product categories",
        "example": "Is MOB's 12% return rate significantly higher than SUP's 8%? ANOVA gives you a p-value to answer this definitively.",
        "tooltip": "📊 Tests if categories have truly different return rates or if differences are just random chance. F-score measures how different the groups are; p-value tells you if it's statistically significant (p<0.05 = real difference, not luck)"
    },
    "MANOVA (Multivariate ANOVA)": {
        "description": "Compares MULTIPLE metrics simultaneously across categories - more powerful than running separate ANOVAs",
        "when_to_use": "When you have return rate AND landed cost (or other metrics) and want to test differences considering all metrics together",
        "example": "Do categories differ when considering both return rate AND financial impact together? MANOVA answers this.",
        "tooltip": "📈 Like ANOVA but tests multiple metrics at once (return rate + cost + sales). Wilks' Lambda shows overall difference; p<0.05 means categories differ significantly across all metrics combined"
    },
    "Kruskal-Wallis (Non-parametric)": {
        "description": "Like ANOVA but doesn't assume your data follows a normal bell curve - more robust for real-world messy data",
        "when_to_use": "When you have outliers, skewed distributions, or small sample sizes where normality can't be assumed",
        "example": "If one product has 50% returns while others are 5-10%, Kruskal-Wallis handles these outliers better than ANOVA",
        "tooltip": "🎯 Robust version of ANOVA that works with messy real-world data. H-statistic measures group differences; p<0.05 means significant difference. Use when you have extreme outliers or small samples"
    },
    "Descriptive Only": {
        "description": "Just calculates summary statistics (means, medians, ranges) without formal hypothesis testing",
        "when_to_use": "Quick overview, very small datasets (<5 products), or when you just need numbers not statistical significance",
        "example": "Simple summary: MOB avg 10.2%, SUP avg 8.5%, LVA avg 9.1% - no p-values, just the facts",
        "tooltip": "📋 Simple averages and summaries without statistical testing - fastest option, good for quick overviews or when you have very few products"
    }
}

# Statistical Terms Plain Language Explanations
STATS_EXPLAINER = {
    "p_value": "The probability that your results happened by random chance. p<0.05 = only 5% chance it's random, so we trust the result is real. Lower is better!",
    "f_score": "Measures how different your groups are compared to variation within groups. Higher F-score = bigger differences between categories. F>3 usually means something interesting is happening.",
    "h_statistic": "Like F-score but for non-normal data. Higher H = bigger differences between groups. Compare to critical value to determine significance.",
    "wilks_lambda": "Tests multiple metrics at once (0 to 1 scale). Lower = bigger overall differences. Think of it as 'how much do groups differ across ALL metrics combined'",
    "confidence_interval": "Range where the true value likely falls. '95% CI: 8-12%' means we're 95% confident the real return rate is between 8-12%.",
    "effect_size": "How BIG is the difference (not just 'is there a difference'). Small=0.2, Medium=0.5, Large=0.8. Helps you know if a difference actually matters in practice.",
    "post_hoc": "After finding differences exist (via ANOVA), post-hoc tests identify WHICH specific groups differ. Example: 'MOB differs from SUP but not from LVA'",
    "degrees_of_freedom": "Number of independent data points used in calculation. More = more reliable results. Technical term you'll see in outputs but don't worry about the details.",
}

# Investigation Methods Dictionary
INVESTIGATION_METHODS = {
    '5_whys': {
        'name': '5 Whys Analysis',
        'best_for': 'Simple, linear problems with clear cause-effect relationships',
        'use_when': 'Problem has a clear starting point and you need to dig deep into root cause',
        'example': 'Product defect → Why? → Manufacturing issue → Why? → Machine calibration → Why? (repeat 5x)'
    },
    'fishbone': {
        'name': 'Fishbone Diagram (Ishikawa)',
        'best_for': 'Complex problems with multiple potential contributing factors',
        'use_when': 'Many possible causes from different categories (people, process, materials, equipment)',
        'example': 'Analyzing all potential causes of product quality issues across manufacturing, design, materials, etc.'
    },
    'rca': {
        'name': 'Root Cause Analysis (Formal RCA)',
        'best_for': 'Critical/high-impact issues requiring comprehensive investigation',
        'use_when': 'Safety concerns, regulatory issues, or high-value/high-volume problems',
        'example': 'Medical device failure with potential patient impact - requires full documentation'
    },
    'fmea': {
        'name': 'FMEA (Failure Mode Effects Analysis)',
        'best_for': 'Proactive risk assessment of potential failures',
        'use_when': 'New product launches, design changes, or preventing future issues',
        'example': 'Analyzing all ways a product could fail and prioritizing prevention efforts'
    },
    '8d': {
        'name': '8D Problem Solving',
        'best_for': 'Team-based problem solving with customer impact',
        'use_when': 'Customer complaints requiring cross-functional investigation and containment',
        'example': 'Batch quality issue affecting multiple customers - requires immediate containment + long-term fix'
    },
    'pareto': {
        'name': 'Pareto Analysis (80/20 Rule)',
        'best_for': 'Prioritizing which issues to tackle first',
        'use_when': 'Multiple issues and you need to focus resources on the biggest impact areas',
        'example': 'Identifying that 20% of defect types cause 80% of returns'
    }
}

# TQM & Kaizen Terminology (Official + Layman's Terms)
TQM_TERMINOLOGY = {
    'genchi_genbutsu': {
        'official': 'Genchi Genbutsu (現地現物)',
        'layman': 'Go & See for Yourself',
        'definition': 'Go to the source to find facts and make correct decisions. See the actual products, processes, and data.',
        'in_practice': 'Instead of just reading reports, physically inspect returned products and talk to warehouse staff who handle them.'
    },
    'pdca': {
        'official': 'PDCA Cycle (Plan-Do-Check-Act)',
        'layman': 'Plan → Try It → Check Results → Make it Standard',
        'definition': 'Continuous improvement cycle: Plan improvements, Do (implement), Check (measure results), Act (standardize or adjust).',
        'in_practice': 'Screen products (Plan) → Investigate issues (Do) → Verify results (Check) → Update processes (Act)'
    },
    'hoshin_kanri': {
        'official': 'Hoshin Kanri (Policy Deployment)',
        'layman': 'Strategic Goal Alignment',
        'definition': 'Align daily work with strategic goals. Everyone works on what matters most for the company.',
        'in_practice': 'Your threshold profiles align screening with company quality goals (e.g., "Q1 Strict Review" for peak season prep)'
    },
    'muda': {
        'official': 'Muda (無駄) - Waste Elimination',
        'layman': 'Cut Out Wasted Effort',
        'definition': 'Eliminate activities that consume resources but create no value (overproduction, waiting, excess inventory, defects, etc.).',
        'in_practice': 'Bulk operations save hours vs screening products one-by-one. AI categorization eliminates manual complaint reading.'
    },
    'jidoka': {
        'official': 'Jidoka (自働化) - Automation with Human Touch',
        'layman': 'Smart Automation that Stops for Problems',
        'definition': 'Build quality into processes with automation that stops when problems occur, alerting humans to fix root causes.',
        'in_practice': 'Statistical screening auto-flags problems (automation) but requires your judgment for escalation decisions (human touch)'
    },
    'yokoten': {
        'official': 'Yokoten (横展) - Horizontal Deployment',
        'layman': 'Share Lessons Across Teams',
        'definition': 'When you solve a problem, share the solution across the organization so others can benefit.',
        'in_practice': 'Export screening results to shared tracker so all teams see flagged products and learn from investigations'
    },
    'gemba': {
        'official': 'Gemba (現場) - The Real Place',
        'layman': 'Where the Work Happens',
        'definition': 'The actual location where value is created (factory floor, warehouse, customer location).',
        'in_practice': 'Use "Deep Dive Analysis" with product manuals/specs to understand gemba (how products actually fail in customer hands)'
    },
    'hansei': {
        'official': 'Hansei (反省) - Critical Self-Reflection',
        'layman': 'Learn from Mistakes',
        'definition': 'Reflect honestly on what went wrong, not to blame, but to learn and improve processes.',
        'in_practice': 'Investigation plans include "Lessons Learned" sections - document what caused issues and how to prevent them'
    },
    'poka_yoke': {
        'official': 'Poka-Yoke (ポカヨケ) - Error-Proofing',
        'layman': 'Make it Hard to Mess Up',
        'definition': 'Design processes/products so mistakes are impossible or immediately obvious.',
        'in_practice': 'Tool validates data on upload, auto-flags statistical outliers, prevents proceeding without required fields'
    },
    'kaizen': {
        'official': 'Kaizen (改善) - Continuous Improvement',
        'layman': 'Always Get a Little Better',
        'definition': 'Philosophy of continuous, incremental improvement by everyone, every day.',
        'in_practice': 'Each screening cycle improves: adjust thresholds based on results, refine AI prompts, update threshold profiles'
    }
}

# Default category thresholds (from SOPs)
DEFAULT_CATEGORY_THRESHOLDS = {
    'B2B': 0.025,
    'INS': 0.07,
    'RHB': 0.075,
    'LVA': 0.095,
    'MOB': 0.10,
    'CSH': 0.105,
    'SUP': 0.11,
    'All Others': 0.10
}

# --- Initialization & Styling ---

def inject_custom_css():
    """Inject custom CSS for modern UI - Vive Health Brand"""
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&family=Montserrat:wght@400;600;700&display=swap');

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
    }}

    html, body, .stApp {{
        font-family: 'Poppins', 'Montserrat', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
    }}

    /* Main header with Vive brand colors and diagonal accent */
    .main-header {{
        background: linear-gradient(135deg, #23b2be 0%, #1a8f98 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 16px rgba(35, 178, 190, 0.3);
        position: relative;
        overflow: hidden;
    }}

    .main-header::before {{
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 30%;
        height: 100%;
        background: #004366;
        clip-path: polygon(30% 0, 100% 0, 100% 100%, 0% 100%);
        opacity: 0.8;
        z-index: 0;
    }}

    .main-title {{
        font-size: 2.2em;
        font-weight: 700;
        font-family: 'Poppins', sans-serif;
        color: #ffffff;
        margin: 0;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
        position: relative;
        z-index: 1;
    }}

    /* Info boxes - Vive brand colors, dark mode compatible */
    .info-box {{
        background: rgba(35, 178, 190, 0.15);
        border: 2px solid #23b2be;
        border-left: 4px solid #004366;
        border-radius: 8px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        color: #f0f0f0;
    }}

    /* Light mode info boxes */
    @media (prefers-color-scheme: light) {{
        .info-box {{
            background: rgba(35, 178, 190, 0.08);
            border: 2px solid #23b2be;
            border-left: 4px solid #004366;
            color: #1a1a1a;
        }}
        .processing-log {{
            background: #f5f5f5;
            border: 1px solid #23b2be;
            color: #1a1a1a;
        }}
        .methodology-box {{
            background: #ffffff;
            border-left: 4px solid #23b2be;
            color: #1a1a1a;
        }}
    }}

    /* Buttons - Vive brand colors */
    .stButton > button {{
        background: linear-gradient(135deg, #23b2be 0%, #1a8f98 100%);
        color: white;
        border: none;
        padding: 0.7rem 1.5rem;
        border-radius: 6px;
        font-weight: 600;
        font-family: 'Poppins', sans-serif;
        box-shadow: 0 2px 8px rgba(35, 178, 190, 0.3);
        transition: all 0.3s ease;
    }}

    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(35, 178, 190, 0.5);
        background: linear-gradient(135deg, #1a8f98 0%, #23b2be 100%);
    }}

    /* Risk indicators - Vive brand colors with high contrast */
    .risk-critical {{
        background-color: #EB3300 !important;
        color: white !important;
        font-weight: 700 !important;
        font-family: 'Poppins', sans-serif !important;
        padding: 0.5rem 1rem !important;
        border-radius: 6px !important;
        border-left: 4px solid #004366 !important;
    }}

    .risk-warning {{
        background-color: #F0B323 !important;
        color: #1a1a1a !important;
        font-weight: 700 !important;
        font-family: 'Poppins', sans-serif !important;
        padding: 0.5rem 1rem !important;
        border-radius: 6px !important;
        border-left: 4px solid #004366 !important;
    }}

    .risk-monitor {{
        background-color: #23b2be !important;
        color: white !important;
        font-weight: 700 !important;
        font-family: 'Poppins', sans-serif !important;
        padding: 0.5rem 1rem !important;
        border-radius: 6px !important;
        border-left: 4px solid #004366 !important;
    }}

    .risk-ok {{
        background-color: #004366 !important;
        color: white !important;
        font-weight: 700 !important;
        font-family: 'Poppins', sans-serif !important;
        padding: 0.5rem 1rem !important;
        border-radius: 6px !important;
        border-left: 4px solid #23b2be !important;
    }}

    /* Processing log - works in both modes */
    .processing-log {{
        background: #1e1e2e;
        border: 1px solid #404050;
        border-radius: 5px;
        padding: 10px;
        max-height: 200px;
        overflow-y: auto;
        font-family: 'Courier New', monospace;
        font-size: 13px;
        color: #e0e0e0;
        line-height: 1.5;
    }}

    /* Methodology boxes - Vive brand style */
    .methodology-box {{
        background: rgba(35, 178, 190, 0.12);
        border-left: 4px solid #23b2be;
        padding: 15px;
        margin: 10px 0;
        border-radius: 4px;
        color: #f0f0f0;
        font-family: 'Poppins', sans-serif;
    }}

    /* Ensure text is readable in all contexts */
    .stMarkdown, .stText {{
        color: inherit;
    }}

    /* Headings - Poppins Bold */
    h1, h2, h3, h4, h5, h6 {{
        font-family: 'Poppins', sans-serif !important;
        font-weight: 700 !important;
        color: #23b2be !important;
    }}

    /* Light mode heading colors */
    @media (prefers-color-scheme: light) {{
        h1, h2, h3, h4, h5, h6 {{
            color: #004366 !important;
        }}
    }}

    /* Improve dataframe visibility */
    .dataframe {{
        font-size: 14px !important;
        font-family: 'Poppins', sans-serif !important;
    }}

    /* Better metric card styling - Vive brand */
    [data-testid="stMetricValue"] {{
        font-size: 2rem !important;
        font-weight: 700 !important;
        font-family: 'Poppins', sans-serif !important;
        color: #23b2be !important;
    }}

    [data-testid="stMetricLabel"] {{
        font-size: 1rem !important;
        font-weight: 600 !important;
        font-family: 'Poppins', sans-serif !important;
    }}

    /* Light mode metrics */
    @media (prefers-color-scheme: light) {{
        [data-testid="stMetricValue"] {{
            color: #004366 !important;
        }}
    }}
    
    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    </style>
    """, unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables for all tabs"""
    defaults = {
        # General AI
        'ai_analyzer': None,
        'ai_provider': AIProvider.OPENAI,  # Default to OpenAI for Tab 3
        
        # Tab 1: Categorizer
        'categorized_data': None,
        'processing_complete': False,
        'reason_summary': {},
        'product_summary': {},
        'total_cost': 0.0,
        'batch_size': 20,
        'chunk_size': APP_CONFIG['default_chunk'],
        'processing_errors': [],
        'column_mapping': {},
        'show_product_analysis': False,
        'processing_speed': 0.0,
        'export_data': None,
        'export_filename': None,
        
        # Tab 2: B2B Reports
        'b2b_processed_data': None,
        'b2b_processing_complete': False,
        'b2b_export_data': None,
        'b2b_export_filename': None,
        'b2b_perf_mode': 'Small (< 500 rows)',
        
        # Tab 3: Quality Screening - NEW
        'qc_mode': 'Lite',
        'qc_results': None,
        'qc_results_df': None,
        'processing_log': [],
        'anova_result': None,
        'manova_result': None,
        'statistical_suggestion': None,

        # Historical data and product matching
        'historical_data': None,
        'product_matcher': None,
        'multilingual_communicator': None,

        # Threshold profiles
        'threshold_profiles': {
            'Standard Review': DEFAULT_CATEGORY_THRESHOLDS.copy(),
            'Aggressive Q4 Review': {k: v * 0.8 for k, v in DEFAULT_CATEGORY_THRESHOLDS.items()}
        },
        'active_profile': 'Standard Review',
        'custom_thresholds': None,
        
        # User-uploaded threshold data
        'user_threshold_data': None,
        
        # AI Chat state
        'ai_chat_history': [],
        'ai_needs_clarification': False,
        'ai_clarification_question': '',
        'ai_guidance_chat': [],  # For sidebar guidance chat
        
        # Session persistence
        'saved_sessions': {},
        'current_session_id': None,
        
        # Manual entry data (Lite mode)
        'lite_entries': [],
        
        # Screening metadata for tracker export
        'screened_by': '',
        'screening_date': datetime.now().strftime('%Y-%m-%d'),
        'source_of_flag': 'Routine Screening',
        'source_of_flag_other': '',

        # Task-based navigation (Landing Page)
        'selected_task': None,  # None = show landing, string = show specific tool
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def check_api_keys():
    """Check for API keys and set environment variables"""
    keys_found = {}
    try:
        if hasattr(st, 'secrets'):
            # Check OpenAI
            for key in ['OPENAI_API_KEY', 'openai_api_key', 'openai']:
                if key in st.secrets:
                    val = str(st.secrets[key]).strip()
                    if val:
                        keys_found['openai'] = val
                        os.environ['OPENAI_API_KEY'] = val
                        break
            
            # Check Claude
            for key in ['ANTHROPIC_API_KEY', 'anthropic_api_key', 'claude_api_key', 'claude']:
                if key in st.secrets:
                    val = str(st.secrets[key]).strip()
                    if val:
                        keys_found['claude'] = val
                        os.environ['ANTHROPIC_API_KEY'] = val
                        break
    except Exception as e:
        logger.warning(f"Error checking secrets: {e}")
    
    return keys_found


def get_ai_analyzer(provider: AIProvider = None, max_workers: int = 5):
    """Get or create AI analyzer instance"""
    if provider is None:
        provider = st.session_state.ai_provider
    
    if st.session_state.ai_analyzer is None or st.session_state.ai_analyzer.provider != provider:
        try:
            check_api_keys()
            st.session_state.ai_analyzer = EnhancedAIAnalyzer(provider, max_workers=max_workers)
            logger.info(f"Created AI analyzer: {provider.value}, Workers: {max_workers}")
        except Exception as e:
            st.error(f"Error initializing AI: {str(e)}")
    
    return st.session_state.ai_analyzer


def load_historical_data():
    """Load historical return data from trailing 12-month CSV"""
    if st.session_state.historical_data is None:
        try:
            # Try to load the trailing 12-month data file
            hist_file_path = os.path.join(os.path.dirname(__file__), 'Trailing 12 Month Returns on Amazon - Use This.csv')
            if os.path.exists(hist_file_path):
                st.session_state.historical_data = pd.read_csv(hist_file_path)
                logger.info(f"Loaded historical data: {len(st.session_state.historical_data)} products")

                # Initialize product matcher
                if AI_AVAILABLE and st.session_state.ai_analyzer:
                    st.session_state.product_matcher = ProductMatcher(
                        st.session_state.historical_data,
                        st.session_state.ai_analyzer
                    )
                    logger.info("Product matcher initialized with AI support")
                else:
                    st.session_state.product_matcher = ProductMatcher(
                        st.session_state.historical_data,
                        None
                    )
                    logger.info("Product matcher initialized without AI")

                # Initialize multilingual communicator
                if AI_AVAILABLE and st.session_state.ai_analyzer:
                    st.session_state.multilingual_communicator = MultilingualVendorCommunicator(
                        st.session_state.ai_analyzer
                    )
                    logger.info("Multilingual communicator initialized")

        except Exception as e:
            logger.error(f"Failed to load historical data: {e}")
            st.session_state.historical_data = pd.DataFrame()

    return st.session_state.historical_data


def log_process(message: str, msg_type: str = 'info'):
    """Adds message to the Processing Transparency Log"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    entry = f"[{timestamp}] [{msg_type.upper()}] {message}"
    st.session_state.processing_log.append(entry)
    if msg_type == 'error':
        logger.error(message)
    else:
        logger.info(message)


def render_api_health_check():
    """Render API health check status in sidebar"""
    keys = check_api_keys()
    
    st.sidebar.markdown("### 🔌 API Status")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if keys.get('openai'):
            st.success("OpenAI ✓")
        else:
            st.error("OpenAI ✗")
    
    with col2:
        if keys.get('claude'):
            st.success("Claude ✓")
        else:
            st.warning("Claude ✗")
    
    return keys


# -------------------------
# TAB 1 LOGIC: Categorizer (PRESERVED)
# -------------------------

def process_file_preserve_structure(file_content, filename):
    """Process file while preserving original structure"""
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_content), dtype=str)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(file_content), dtype=str)
        elif filename.endswith('.txt'):
            df = pd.read_csv(io.BytesIO(file_content), sep='\t', dtype=str)
        else:
            return None, None
        
        column_mapping = {}
        cols = df.columns.tolist()
        
        if len(cols) >= 11:
            if len(cols) > 8: column_mapping['complaint'] = cols[8]
            if len(cols) > 1: column_mapping['sku'] = cols[1]
            
            while len(df.columns) < 11:
                df[f'Column_{len(df.columns)}'] = ''
            column_mapping['category'] = df.columns[10]
            df[column_mapping['category']] = ''
        else:
            st.error("File structure not recognized. Need at least 11 columns (A-K).")
            return None, None
            
        return df, column_mapping
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None, None


def process_in_chunks(df, analyzer, column_mapping, chunk_size=None):
    """Process large datasets in chunks"""
    if chunk_size is None:
        chunk_size = st.session_state.chunk_size
    
    complaint_col = column_mapping['complaint']
    category_col = column_mapping['category']
    
    # Get rows with complaints
    valid_indices = df[df[complaint_col].notna() & (df[complaint_col].str.strip() != '')].index
    total_valid = len(valid_indices)
    
    if total_valid == 0:
        st.warning("No valid complaints found in Column I to process")
        return df
    
    # Clear messaging about processing
    st.info(f"""
    📊 **Processing Details:**
    - Total complaints to categorize (from Column I): **{total_valid:,}**
    - Processing chunk size: **{chunk_size}** rows at a time
    - API batch size: **{st.session_state.batch_size}** items per call
    """)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    stats_container = st.container()
    
    processed_count = 0
    start_time = time.time()
    
    # Process in chunks
    for chunk_start in range(0, total_valid, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_valid)
        chunk_indices = valid_indices[chunk_start:chunk_end]
        chunk_num = (chunk_start // chunk_size) + 1
        total_chunks = (total_valid + chunk_size - 1) // chunk_size
        
        # Prepare batch data
        batch_data = []
        for idx in chunk_indices:
            complaint = str(df.at[idx, complaint_col]).strip()
            
            # Check for FBA reason code
            fba_reason = None
            if 'reason' in df.columns:
                fba_reason = str(df.at[idx, 'reason'])
            
            batch_data.append({
                'index': idx,
                'complaint': complaint,
                'fba_reason': fba_reason
            })
        
        try:
            # Process batch with smaller sub-batches
            sub_batch_size = st.session_state.batch_size
            
            for i in range(0, len(batch_data), sub_batch_size):
                sub_batch = batch_data[i:i+sub_batch_size]
                
                # Categorize sub-batch
                results = analyzer.categorize_batch(sub_batch, mode='standard')
                
                # Update dataframe
                for result in results:
                    idx = result['index']
                    category = result.get('category', 'Other/Miscellaneous')
                    df.at[idx, category_col] = category
                    
                    # Track stats
                    processed_count += 1
                
                # Update progress
                progress = processed_count / total_valid
                progress_bar.progress(progress)
                
                elapsed = time.time() - start_time
                speed = processed_count / elapsed if elapsed > 0 else 0
                remaining = (total_valid - processed_count) / speed if speed > 0 else 0
                
                # Update status with clear information
                with stats_container:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Progress", f"{processed_count:,}/{total_valid:,}")
                    with col2:
                        st.metric("Speed", f"{speed:.1f}/sec")
                    with col3:
                        st.metric("Chunk", f"{chunk_num}/{total_chunks}")
                    with col4:
                        if remaining > 0:
                            st.metric("ETA", f"{int(remaining)}s")
                        else:
                            st.metric("ETA", "Complete")
                
                # Small delay to prevent overwhelming
                time.sleep(0.05)
                
        except Exception as e:
            logger.error(f"Chunk processing error: {e}")
            st.session_state.processing_errors.append(f"Chunk {chunk_num}: {str(e)}")
            
            # Fill failed items with default category
            for item in batch_data:
                if pd.isna(df.at[item['index'], category_col]):
                    df.at[item['index'], category_col] = 'Other/Miscellaneous'
        
        # Force garbage collection after each chunk
        gc.collect()
    
    # Final update
    progress_bar.progress(1.0)
    elapsed = time.time() - start_time
    st.session_state.processing_speed = processed_count / elapsed if elapsed > 0 else 0
    
    # Clear the stats container and show final message
    stats_container.empty()
    status_text.success(f"✅ Complete! Processed {processed_count:,} returns in {elapsed:.1f} seconds at {st.session_state.processing_speed:.1f} returns/second")
    
    return df


def generate_statistics(df, column_mapping):
    """Generate statistics from categorized data"""
    category_col = column_mapping.get('category')
    sku_col = column_mapping.get('sku')
    
    if not category_col:
        logger.warning("No category column in mapping, cannot generate statistics")
        return
    
    # Category statistics
    categorized_df = df[df[category_col].notna() & (df[category_col] != '')]
    if len(categorized_df) == 0:
        logger.warning("No categorized data found")
        return
    
    category_counts = categorized_df[category_col].value_counts()
    st.session_state.reason_summary = category_counts.to_dict()
    
    # SKU statistics
    if sku_col and sku_col in df.columns:
        product_summary = defaultdict(lambda: defaultdict(int))
        
        for _, row in categorized_df.iterrows():
            if pd.notna(row.get(sku_col)):
                sku = str(row[sku_col]).strip()
                if sku and sku != 'nan':
                    category = row[category_col]
                    product_summary[sku][category] += 1
        
        st.session_state.product_summary = dict(product_summary)
        logger.info(f"Generated product summary for {len(product_summary)} SKUs")
    
    logger.info(f"Statistics generated: {len(st.session_state.reason_summary)} categories")


def export_with_column_k(df):
    """Export to Excel preserving format"""
    output = io.BytesIO()
    if EXCEL_AVAILABLE:
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Returns')
            workbook = writer.book
            worksheet = writer.sheets['Returns']
            fmt = workbook.add_format({'bg_color': '#E6F5E6', 'font_color': '#006600', 'bold': True})
            worksheet.set_column(10, 10, 20, fmt)
    else:
        df.to_csv(output, index=False)
    output.seek(0)
    return output.getvalue()


def display_results_dashboard(df, column_mapping):
    """Display enhanced results dashboard (Tab 1)"""
    st.markdown("### 📊 Analysis Results")
    
    # Validate column mapping
    category_col = column_mapping.get('category')
    sku_col = column_mapping.get('sku')
    
    if not category_col:
        st.error("Category column not detected. Unable to render summary metrics.")
        return
    
    # Calculate metrics
    total_rows = len(df)
    categorized_rows = len(df[df[category_col].notna() & (df[category_col] != '')])
    
    # Key Metrics Row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Rows", f"{total_rows:,}")
    
    with col2:
        st.metric("Categorized", f"{categorized_rows:,}")
    
    with col3:
        success_rate = categorized_rows / total_rows * 100 if total_rows > 0 else 0
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    with col4:
        quality_count = sum(count for cat, count in st.session_state.reason_summary.items() 
                          if cat in QUALITY_CATEGORIES)
        quality_rate = quality_count / categorized_rows * 100 if categorized_rows > 0 else 0
        st.metric("Quality Issues", f"{quality_rate:.1f}%", 
                 help=f"{quality_count:,} quality-related returns")
    
    with col5:
        cost_per_return = st.session_state.total_cost / categorized_rows if categorized_rows > 0 else 0
        st.metric("Cost/Return", f"${cost_per_return:.4f}")
    
    # Category Distribution
    st.markdown("---")
    st.markdown("#### 📈 Category Distribution")
    
    if st.session_state.reason_summary:
        # Create two columns for better layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Show top categories with visual bars
            top_categories = sorted(st.session_state.reason_summary.items(), 
                                  key=lambda x: x[1], reverse=True)[:10]
            
            for i, (cat, count) in enumerate(top_categories):
                pct = count / categorized_rows * 100 if categorized_rows > 0 else 0
                
                # Determine color based on category type
                if cat in QUALITY_CATEGORIES:
                    color = COLORS['danger']
                    icon = "🔴"
                else:
                    color = COLORS['primary']
                    icon = "🔵"
                
                # Create visual bar
                st.markdown(f"""
                <div style="margin: 0.8rem 0;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.3rem;">
                        <span style="font-weight: 500;">{icon} {cat}</span>
                        <span style="color: {COLORS['muted']};">{count:,} ({pct:.1f}%)</span>
                    </div>
                    <div style="background: rgba(255,255,255,0.1); height: 20px; border-radius: 10px; overflow: hidden;">
                        <div style="background: {color}; width: {pct}%; height: 100%; 
                                    border-radius: 10px; transition: width 0.5s ease;">
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Summary stats box
            top_pct = (top_categories[0][1] / categorized_rows * 100) if top_categories and categorized_rows > 0 else 0
            st.markdown(f"""
            <div class="info-box" style="text-align: center;">
                <h4 style="color: var(--primary); margin: 0;">Summary</h4>
                <div style="margin-top: 1rem;">
                    <div style="font-size: 2em; font-weight: 700; color: var(--danger);">
                        {quality_rate:.0f}%
                    </div>
                    <div style="color: var(--muted);">Quality Issues</div>
                </div>
                <hr style="opacity: 0.2; margin: 1rem 0;">
                <div style="font-size: 0.9em;">
                    <div>Categories: {len(st.session_state.reason_summary)}</div>
                    <div style="color: var(--muted); margin-top: 0.5rem;">
                        Top category accounts for {top_pct:.0f}% of returns
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Product Analysis (if enabled)
    if st.session_state.show_product_analysis and st.session_state.product_summary:
        st.markdown("---")
        st.markdown("#### 📦 Product/SKU Analysis")
        
        # Product metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Unique SKUs", f"{len(st.session_state.product_summary):,}")
        
        with col2:
            avg_returns_per_sku = categorized_rows / len(st.session_state.product_summary) if st.session_state.product_summary else 0
            st.metric("Avg Returns/SKU", f"{avg_returns_per_sku:.1f}")
        
        with col3:
            # Find SKUs with high quality issues
            high_quality_skus = 0
            for sku, issues in st.session_state.product_summary.items():
                quality_count_sku = sum(count for cat, count in issues.items() if cat in QUALITY_CATEGORIES)
                total_count = sum(issues.values())
                if total_count > 0 and quality_count_sku / total_count > 0.5:
                    high_quality_skus += 1
            st.metric("High Risk SKUs", f"{high_quality_skus:,}", 
                     help="SKUs with >50% quality issues")
        
        # Top problematic products
        st.markdown("##### 🚨 Top 10 Products by Return Volume (from Column B)")
        
        # Calculate product metrics
        product_data = []
        for sku, issues in st.session_state.product_summary.items():
            total = sum(issues.values())
            quality = sum(count for cat, count in issues.items() if cat in QUALITY_CATEGORIES)
            quality_pct_prod = quality / total * 100 if total > 0 else 0
            top_issue = max(issues.items(), key=lambda x: x[1])[0] if issues else 'N/A'
            
            product_data.append({
                'SKU': sku,
                'Total Returns': total,
                'Quality Issues': quality,
                'Quality %': quality_pct_prod,
                'Top Issue': top_issue,
                'Risk Score': quality * (quality_pct_prod / 100)
            })
        
        # Sort by total returns
        product_data.sort(key=lambda x: x['Total Returns'], reverse=True)
        
        # Display top products
        for i, product in enumerate(product_data[:10]):
            if i < 5:
                if product['Quality %'] > 50:
                    risk_color = COLORS['danger']
                    risk_label = "High Risk"
                elif product['Quality %'] > 25:
                    risk_color = COLORS['warning']
                    risk_label = "Medium Risk"
                else:
                    risk_color = COLORS['success']
                    risk_label = "Low Risk"
                
                st.markdown(f"""
                <div class="info-box" style="border-left: 4px solid {risk_color}; margin: 0.5rem 0;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong>{i+1}. SKU: {product['SKU'][:40]}{'...' if len(product['SKU']) > 40 else ''}</strong>
                            <div style="color: {COLORS['muted']}; font-size: 0.9em; margin-top: 0.2rem;">
                                Top issue: {product['Top Issue']}
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 1.2em; font-weight: 600;">{product['Total Returns']:,} returns</div>
                            <div style="color: {risk_color}; font-size: 0.9em;">
                                {product['Quality %']:.0f}% quality ({risk_label})
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"{i+1}. **{product['SKU'][:30]}...**: {product['Total Returns']} returns ({product['Quality %']:.0f}% quality)")
        
        # Option to export full product analysis
        if st.button("📥 Export Full SKU Analysis"):
            export_data = []
            for sku, issues in st.session_state.product_summary.items():
                for category, count in issues.items():
                    export_data.append({
                        'SKU': sku,
                        'Category': category,
                        'Count': count,
                        'Is_Quality_Issue': 'Yes' if category in QUALITY_CATEGORIES else 'No'
                    })
            
            export_df = pd.DataFrame(export_data)
            csv = export_df.to_csv(index=False)
            
            st.download_button(
                label="Download SKU Analysis CSV",
                data=csv,
                file_name=f"sku_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )


# -------------------------
# TAB 2 LOGIC: B2B Reports (PRESERVED)
# -------------------------

def extract_main_sku(text):
    """
    Extracts the Main SKU (3 Caps + 4 Digits) from text.
    Ignores variants (e.g., matches MOB1027 from MOB1027BLU).
    Strictly follows the pattern: 3 Uppercase Letters + 4 Digits.
    """
    if not isinstance(text, str):
        return None
    
    # Pattern: 3 uppercase letters followed by 4 digits.
    match = re.search(r'\b([A-Z]{3}\d{4})', text)
    
    if match:
        return match.group(1)
    return None


def find_sku_in_row(row):
    """
    Attempts to find the Main SKU in various columns.
    Priority: 
    1. Explicit SKU columns (Product, SKU, Main SKU)
    2. Subject/Display Name
    3. Description/Body
    """
    # 1. Check explicit columns first
    sku_cols = ['Main SKU', 'Main SKU/Display Name', 'SKU', 'Product', 'Internal Reference']
    for col in sku_cols:
        if col in row.index and pd.notna(row[col]):
            sku = extract_main_sku(str(row[col]))
            if sku: return sku
    
    # 2. Check Display Name / Subject
    subject_cols = ['Display Name', 'Subject', 'Name']
    for col in subject_cols:
        if col in row.index and pd.notna(row[col]):
            sku = extract_main_sku(str(row[col]))
            if sku: return sku
            
    # 3. Check Description (Last resort)
    desc_cols = ['Description', 'Body']
    for col in desc_cols:
        if col in row.index and pd.notna(row[col]):
            sku = extract_main_sku(str(row[col]))
            if sku: return sku
            
    return "Unknown"


def strip_html(text):
    """Remove HTML tags from description for cleaner AI processing"""
    if not text or not isinstance(text, str):
        return ""
    clean = re.compile('<.*?>')
    return re.sub(clean, ' ', text).strip()


def process_b2b_file(file_content, filename):
    """Process raw Odoo export for B2B Report"""
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_content), dtype=str)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(file_content), dtype=str)
        else:
            st.error("Unsupported file format")
            return None

        logger.info(f"B2B Processing: {len(df)} rows loaded")
        return df
        
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None


def generate_b2b_report(df, analyzer, batch_size):
    """Generate the B2B report with AI summaries and Cleaned SKUs"""
    
    # Columns to use for display/AI
    display_col = 'Display Name' if 'Display Name' in df.columns else df.columns[0]
    desc_col = 'Description' if 'Description' in df.columns else None
    
    # Pre-process data for AI batching
    items_to_process = []
    
    for idx, row in df.iterrows():
        subject = str(row.get(display_col, ''))
        description = str(row.get(desc_col, '')) if desc_col else ''
        
        # Extract Main SKU using strict logic
        main_sku = find_sku_in_row(row)
        
        items_to_process.append({
            'index': idx,
            'subject': subject,
            'details': strip_html(description)[:1000],
            'full_description': description,
            'sku': main_sku
        })
    
    # Batch Process with AI
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_items = len(items_to_process)
    processed_results = []
    
    for i in range(0, total_items, batch_size):
        batch = items_to_process[i:i+batch_size]
        
        # Use the summarize_batch method
        batch_results = analyzer.summarize_batch(batch)
        processed_results.extend(batch_results)
        
        # Update progress
        progress = min((i + batch_size) / total_items, 1.0)
        progress_bar.progress(progress)
        status_text.text(f"⏳ Generating summaries: {min(i + batch_size, total_items)}/{total_items}")
        
    status_text.success("✅ AI Summarization Complete!")
    time.sleep(1)
    status_text.empty()
    progress_bar.empty()
    
    # Construct Final DataFrame
    final_rows = []
    for item in processed_results:
        final_rows.append({
            'Display Name': item['subject'],
            'Description': item['full_description'],
            'SKU': item['sku'],
            'Reason': item.get('summary', 'Summary Unavailable')
        })
        
    return pd.DataFrame(final_rows)


# -------------------------
# TAB 3 LOGIC: Quality Screening (REBUILT)
# -------------------------

# Advanced Analytics Functions for World-Class Quality Teams

def render_root_cause_analysis(tracker):
    """Root Cause Analysis Tools - 5-Why, Fishbone, Pareto Charts"""
    st.markdown("""
    <div style="background: rgba(142, 68, 173, 0.1); border-left: 4px solid #8e44ad;
                padding: 1rem; margin: 1rem 0; border-radius: 6px;">
        <h4 style="color: #8e44ad; font-family: 'Poppins', sans-serif; margin: 0 0 0.5rem 0;">
            🔍 Root Cause Analysis Tools
        </h4>
        <p style="color: #666; font-family: 'Poppins', sans-serif; font-size: 0.9em; margin: 0;">
            Systematic problem-solving using proven methodologies: 5-Why Analysis, Fishbone Diagrams (Ishikawa),
            and Pareto Charts for identifying the vital few from the trivial many.
        </p>
    </div>
    """, unsafe_allow_html=True)

    rca_method = st.selectbox(
        "Select Analysis Method",
        ["Pareto Analysis (80/20 Rule)", "5-Why Analysis", "Fishbone Diagram (Ishikawa)"],
        key="rca_method"
    )

    if rca_method == "Pareto Analysis (80/20 Rule)":
        st.markdown("### 📊 Pareto Chart - Top Issues by Frequency")

        # Count issues
        issue_counts = {}
        for case in tracker.cases:
            if case.top_issues:
                # Normalize issue text
                issue = case.top_issues[:100].strip()
                issue_counts[issue] = issue_counts.get(issue, 0) + 1

        if issue_counts:
            # Sort by frequency
            sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
            total_count = sum(issue_counts.values())

            # Calculate cumulative percentage
            pareto_data = []
            cumulative_pct = 0
            for issue, count in sorted_issues[:10]:  # Top 10
                pct = (count / total_count) * 100
                cumulative_pct += pct
                pareto_data.append({
                    'Issue': issue[:50] + '...' if len(issue) > 50 else issue,
                    'Count': count,
                    'Percentage': f"{pct:.1f}%",
                    'Cumulative %': f"{cumulative_pct:.1f}%"
                })

            df_pareto = pd.DataFrame(pareto_data)
            st.dataframe(df_pareto, use_container_width=True)

            # Identify vital few (80% rule)
            vital_few = [item for item in pareto_data if float(item['Cumulative %'].rstrip('%')) <= 80]
            st.success(f"🎯 **Vital Few (80% of issues):** {len(vital_few)} issue categories account for 80% of all quality cases")

            st.markdown("""
            <div style="background: #e8f5e9; border-left: 4px solid #4caf50; padding: 1rem; margin: 1rem 0;">
                <strong>💡 Pareto Principle:</strong> Focus corrective actions on the top issues shown above.
                Resolving these will have the greatest impact on overall quality metrics.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No issue data available for Pareto analysis")

    elif rca_method == "5-Why Analysis":
        st.markdown("### 🤔 5-Why Root Cause Analysis")

        selected_case = st.selectbox(
            "Select Case for 5-Why Analysis",
            options=range(len(tracker.cases)),
            format_func=lambda i: f"{tracker.cases[i].product_name} ({tracker.cases[i].sku}) - {tracker.cases[i].top_issues[:50]}..."
        )

        case = tracker.cases[selected_case]

        st.markdown(f"""
        **Problem Statement:** {case.top_issues}
        **Product:** {case.product_name} ({case.sku})
        **Return Rate:** {case.return_rate_amazon*100:.1f}%
        """)

        st.markdown("---")

        # 5-Why Form
        if 'five_whys' not in st.session_state:
            st.session_state.five_whys = {}

        case_key = f"{case.sku}_{selected_case}"
        if case_key not in st.session_state.five_whys:
            st.session_state.five_whys[case_key] = [""] * 5

        whys = st.session_state.five_whys[case_key]

        for i in range(5):
            whys[i] = st.text_input(
                f"Why #{i+1}: Why did this happen?",
                value=whys[i],
                key=f"why_{case_key}_{i}",
                placeholder=f"Enter reason #{i+1}..."
            )

        if st.button("🤖 AI-Assisted 5-Why Analysis", type="primary"):
            if tracker.ai_analyzer:
                with st.spinner("Analyzing root causes with AI..."):
                    prompt = f"""Perform a 5-Why root cause analysis for this quality issue:

Product: {case.product_name}
Issue: {case.top_issues}
Return Rate: {case.return_rate_amazon*100:.1f}%
Additional Context: {case.notification_notes}

Provide 5 progressive "why" questions that drill down to the root cause. Format as:
Why 1: [question and answer]
Why 2: [question and answer]
... etc

End with "Root Cause:" and the fundamental issue."""

                    analysis = tracker.ai_analyzer.generate_text(
                        prompt,
                        "You are a quality engineer expert in root cause analysis using 5-Why methodology.",
                        mode='chat'
                    )

                    st.markdown("""
                    <div style="background: white; border: 2px solid #8e44ad; padding: 1.5rem;
                                border-radius: 8px; margin: 1rem 0;">
                    """, unsafe_allow_html=True)
                    st.markdown(analysis)
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.warning("AI analyzer not available")

        if whys[4]:  # If all 5 whys filled
            st.success("✅ **Root Cause Identified:** " + whys[4])

            st.markdown("#### 🎯 Recommended Corrective Actions")
            st.text_area(
                "Document corrective actions to address root cause:",
                key=f"corrective_actions_{case_key}",
                height=100,
                placeholder="What actions will prevent recurrence of this root cause?"
            )

    elif rca_method == "Fishbone Diagram (Ishikawa)":
        st.markdown("### 🐟 Fishbone (Ishikawa) Diagram Analysis")

        selected_case = st.selectbox(
            "Select Case for Fishbone Analysis",
            options=range(len(tracker.cases)),
            format_func=lambda i: f"{tracker.cases[i].product_name} ({tracker.cases[i].sku}) - {tracker.cases[i].top_issues[:50]}...",
            key="fishbone_case"
        )

        case = tracker.cases[selected_case]

        st.markdown(f"""
        **Problem (Effect):** {case.top_issues}
        **Product:** {case.product_name}
        """)

        st.markdown("---")
        st.markdown("### 📋 Identify Causes by Category (6M Framework)")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🏭 Man (People)")
            man_causes = st.text_area("Training, skill, knowledge issues?", key="fish_man", height=80)

            st.markdown("#### 🔧 Method (Process)")
            method_causes = st.text_area("Process, procedure, workflow issues?", key="fish_method", height=80)

            st.markdown("#### 📦 Material")
            material_causes = st.text_area("Raw material, component quality issues?", key="fish_material", height=80)

        with col2:
            st.markdown("#### ⚙️ Machine (Equipment)")
            machine_causes = st.text_area("Equipment, tooling, maintenance issues?", key="fish_machine", height=80)

            st.markdown("#### 📏 Measurement")
            measurement_causes = st.text_area("Inspection, testing, calibration issues?", key="fish_measurement", height=80)

            st.markdown("#### 🌍 Environment")
            environment_causes = st.text_area("Temperature, humidity, cleanliness issues?", key="fish_environment", height=80)

        if st.button("🤖 AI-Generate Fishbone Categories", type="primary"):
            if tracker.ai_analyzer:
                with st.spinner("Analyzing potential causes with AI..."):
                    prompt = f"""Create a Fishbone (Ishikawa) diagram analysis for this quality issue using the 6M framework:

Problem: {case.top_issues}
Product: {case.product_name}
Context: {case.notification_notes}

For each category, list 2-3 potential causes:
1. Man (People)
2. Method (Process)
3. Material
4. Machine (Equipment)
5. Measurement
6. Environment (Mother Nature)

Format clearly with bullet points under each category."""

                    analysis = tracker.ai_analyzer.generate_text(
                        prompt,
                        "You are a Six Sigma Black Belt expert in fishbone diagram root cause analysis.",
                        mode='chat'
                    )

                    st.markdown("""
                    <div style="background: white; border: 2px solid #8e44ad; padding: 1.5rem;
                                border-radius: 8px; margin: 1rem 0;">
                    """, unsafe_allow_html=True)
                    st.markdown(analysis)
                    st.markdown("</div>", unsafe_allow_html=True)


def render_capa_management(tracker):
    """CAPA (Corrective and Preventive Action) Management System"""
    st.markdown("""
    <div style="background: rgba(52, 152, 219, 0.1); border-left: 4px solid #3498db;
                padding: 1rem; margin: 1rem 0; border-radius: 6px;">
        <h4 style="color: #3498db; font-family: 'Poppins', sans-serif; margin: 0 0 0.5rem 0;">
            📋 CAPA Management System
        </h4>
        <p style="color: #666; font-family: 'Poppins', sans-serif; font-size: 0.9em; margin: 0;">
            Full lifecycle tracking of Corrective and Preventive Actions from identification through
            verification and effectiveness checks. FDA 21 CFR 820.100 and ISO 13485:2016 compliant.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # CAPA Statistics
    total_cases = len(tracker.cases)
    cases_with_actions = sum(1 for c in tracker.cases if c.action_taken and c.action_taken.strip())
    cases_pending = total_cases - cases_with_actions

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total CAPAs", total_cases, help="Total quality cases requiring CAPA")
    with col2:
        st.metric("Actions Taken", cases_with_actions, help="Cases with documented corrective actions")
    with col3:
        st.metric("Pending Action", cases_pending, help="Cases awaiting corrective action", delta=f"-{cases_pending}" if cases_pending > 0 else "0")
    with col4:
        closed_cases = sum(1 for c in tracker.cases if hasattr(c, 'case_status') and c.case_status == 'Closed')
        st.metric("Closed", closed_cases, help="Cases verified and closed")

    st.markdown("---")

    # CAPA Workflow Stages
    st.markdown("### 🔄 CAPA Lifecycle Stages")

    capa_stage = st.selectbox(
        "Select CAPA Stage",
        [
            "1️⃣ Identification & Documentation",
            "2️⃣ Root Cause Analysis",
            "3️⃣ Corrective Action Planning",
            "4️⃣ Implementation & Tracking",
            "5️⃣ Verification & Effectiveness",
            "6️⃣ Closure & Documentation"
        ]
    )

    if "Identification" in capa_stage:
        st.markdown("#### 📝 CAPA Identification")
        st.markdown("""
        **Purpose:** Document the nonconformance or quality issue requiring corrective/preventive action.

        **Inputs:**
        - Quality case details (already captured)
        - Return rate data
        - Customer complaints
        - Internal audit findings
        """)

        # Show cases pending CAPA initiation
        st.markdown("##### Cases Requiring CAPA Initiation:")
        pending_df = tracker.get_cases_dataframe(leadership_version=False)
        if not pending_df.empty:
            display_cols = ['Product name', 'SKU', 'Return rate Amazon', 'Top Issue(s)']
            display_cols = [c for c in display_cols if c in pending_df.columns]
            st.dataframe(pending_df[display_cols], use_container_width=True, height=300)

    elif "Root Cause" in capa_stage:
        st.markdown("#### 🔍 Root Cause Investigation")
        st.info("💡 Use the **Root Cause Analysis** tab to perform 5-Why, Fishbone, or Pareto analysis")

        # Show cases with/without RCA
        cases_with_rca = []
        cases_without_rca = []

        for case in tracker.cases:
            if case.action_taken and case.action_taken.strip():
                cases_with_rca.append(f"{case.product_name} ({case.sku})")
            else:
                cases_without_rca.append(f"{case.product_name} ({case.sku})")

        if cases_without_rca:
            st.warning(f"⚠️ {len(cases_without_rca)} cases pending root cause analysis")
            with st.expander("View Cases Pending RCA"):
                for case in cases_without_rca[:10]:
                    st.write(f"• {case}")

    elif "Planning" in capa_stage:
        st.markdown("#### 📋 Corrective Action Plan")

        selected_case_idx = st.selectbox(
            "Select Case for Action Planning",
            options=range(len(tracker.cases)),
            format_func=lambda i: f"{tracker.cases[i].product_name} ({tracker.cases[i].sku})"
        )

        case = tracker.cases[selected_case_idx]

        st.markdown(f"**Product:** {case.product_name}")
        st.markdown(f"**Issue:** {case.top_issues}")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### ✅ Corrective Actions (Fix Current Issue)")
            corrective = st.text_area(
                "What will be done to correct the existing nonconformance?",
                value=case.action_taken if case.action_taken else "",
                key=f"corrective_{selected_case_idx}",
                height=120
            )

        with col2:
            st.markdown("##### 🛡️ Preventive Actions (Prevent Recurrence)")
            preventive = st.text_area(
                "What will be done to prevent recurrence?",
                key=f"preventive_{selected_case_idx}",
                height=120,
                placeholder="Changes to process, training, specifications, etc."
            )

        st.markdown("##### 👥 Assignments & Timeline")
        col3, col4, col5 = st.columns(3)
        with col3:
            responsible_person = st.text_input("Responsible Person", key=f"resp_{selected_case_idx}")
        with col4:
            target_date = st.date_input("Target Completion", key=f"target_{selected_case_idx}")
        with col5:
            priority = st.selectbox("Priority", ["High", "Medium", "Low"], key=f"priority_{selected_case_idx}")

    elif "Implementation" in capa_stage:
        st.markdown("#### ⚙️ CAPA Implementation & Tracking")

        # Show cases by implementation status
        in_progress = []
        completed = []

        for case in tracker.cases:
            status = getattr(case, 'case_status', 'Open')
            if status in ['Active Investigation', 'Action Taken - Monitoring']:
                in_progress.append(case)
            elif status == 'Closed':
                completed.append(case)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("In Progress", len(in_progress))
        with col2:
            st.metric("Completed", len(completed))

        st.markdown("##### 🔄 Actions In Progress:")
        for case in in_progress[:5]:
            with st.expander(f"{case.product_name} - {case.sku}"):
                st.write(f"**Issue:** {case.top_issues}")
                st.write(f"**Action Taken:** {case.action_taken if case.action_taken else 'Not documented'}")
                st.write(f"**Status:** {getattr(case, 'case_status', 'Open')}")

                # Progress tracker
                progress = st.slider(
                    "Implementation Progress %",
                    0, 100, 50,
                    key=f"progress_{case.sku}",
                    help="Track implementation completion percentage"
                )

    elif "Verification" in capa_stage:
        st.markdown("#### ✔️ CAPA Verification & Effectiveness Check")

        st.markdown("""
        **Verification Criteria:**
        1. Actions implemented as planned
        2. Results documented (Result 1, Result 2 fields)
        3. Return rate improved
        4. No recurrence observed
        """)

        # Show cases with results tracking
        cases_with_results = []
        for case in tracker.cases:
            if case.result_1_rr is not None or case.result_check_date_1 is not None:
                improvement = None
                if case.return_rate_amazon and case.result_1_rr:
                    improvement = ((case.return_rate_amazon - case.result_1_rr) / case.return_rate_amazon) * 100

                cases_with_results.append({
                    'Product': case.product_name,
                    'SKU': case.sku,
                    'Initial RR': f"{case.return_rate_amazon*100:.1f}%" if case.return_rate_amazon else "N/A",
                    'Result 1 RR': f"{case.result_1_rr*100:.1f}%" if case.result_1_rr else "N/A",
                    'Improvement': f"{improvement:.1f}%" if improvement else "N/A",
                    'Effective': '✅ Yes' if improvement and improvement > 10 else '⚠️ Needs Review'
                })

        if cases_with_results:
            st.dataframe(pd.DataFrame(cases_with_results), use_container_width=True)
        else:
            st.info("No verification results documented yet")

    elif "Closure" in capa_stage:
        st.markdown("#### 🎯 CAPA Closure")

        st.markdown("""
        **Closure Checklist:**
        - [ ] Root cause identified and documented
        - [ ] Corrective actions implemented
        - [ ] Effectiveness verified (results show improvement)
        - [ ] Documentation complete
        - [ ] Related procedures updated if needed
        - [ ] Training completed if required
        """)

        # Show cases ready for closure
        ready_for_closure = []
        for case in tracker.cases:
            if (case.action_taken and case.action_taken.strip() and
                case.result_1_rr is not None and
                hasattr(case, 'case_status') and case.case_status != 'Closed'):
                ready_for_closure.append(case)

        if ready_for_closure:
            st.success(f"✅ {len(ready_for_closure)} CAPAs ready for closure")
            for case in ready_for_closure:
                with st.expander(f"{case.product_name} - {case.sku}"):
                    st.write(f"**Action:** {case.action_taken}")
                    if case.result_1_rr:
                        st.write(f"**Result:** RR reduced to {case.result_1_rr*100:.1f}%")

                    if st.button(f"Close CAPA for {case.sku}", key=f"close_{case.sku}"):
                        case.case_status = 'Closed'
                        st.success(f"✅ CAPA closed for {case.product_name}")
                        st.rerun()


def render_risk_analysis_fmea(tracker):
    """Failure Mode and Effects Analysis (FMEA) with Risk Priority Numbers"""
    st.markdown("""
    <div style="background: rgba(231, 76, 60, 0.1); border-left: 4px solid #e74c3c;
                padding: 1rem; margin: 1rem 0; border-radius: 6px;">
        <h4 style="color: #e74c3c; font-family: 'Poppins', sans-serif; margin: 0 0 0.5rem 0;">
            ⚠️ Risk Analysis - FMEA
        </h4>
        <p style="color: #666; font-family: 'Poppins', sans-serif; font-size: 0.9em; margin: 0;">
            Failure Mode and Effects Analysis (FMEA) for proactive risk assessment.
            Calculate Risk Priority Numbers (RPN) based on Severity, Occurrence, and Detection ratings.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Risk Priority Matrix")

    # Calculate RPN for each case
    risk_data = []
    for case in tracker.cases:
        # Severity (1-10): Based on return rate
        if case.return_rate_amazon:
            if case.return_rate_amazon >= 0.15:  # 15%+
                severity = 10
            elif case.return_rate_amazon >= 0.10:
                severity = 8
            elif case.return_rate_amazon >= 0.05:
                severity = 5
            else:
                severity = 3
        else:
            severity = 5  # Default

        # Occurrence (1-10): Based on volume
        if case.ncx_orders:
            if case.ncx_orders >= 100:
                occurrence = 10
            elif case.ncx_orders >= 50:
                occurrence = 7
            elif case.ncx_orders >= 20:
                occurrence = 5
            else:
                occurrence = 3
        else:
            occurrence = 5

        # Detection (1-10): Lower = easier to detect. Check if being monitored
        if case.action_taken and case.action_taken.strip():
            detection = 3  # Already detected and acted upon
        else:
            detection = 7  # Not yet addressed

        rpn = severity * occurrence * detection

        # Risk Level
        if rpn >= 200:
            risk_level = "🔴 Critical"
            risk_color = "#e74c3c"
        elif rpn >= 100:
            risk_level = "🟠 High"
            risk_color = "#f39c12"
        elif rpn >= 50:
            risk_level = "🟡 Medium"
            risk_color = "#f1c40f"
        else:
            risk_level = "🟢 Low"
            risk_color = "#27ae60"

        risk_data.append({
            'Product': case.product_name,
            'SKU': case.sku,
            'Failure Mode': case.top_issues[:50] + '...' if len(case.top_issues) > 50 else case.top_issues,
            'Severity (S)': severity,
            'Occurrence (O)': occurrence,
            'Detection (D)': detection,
            'RPN': rpn,
            'Risk Level': risk_level,
            'risk_color': risk_color
        })

    # Sort by RPN descending
    risk_data.sort(key=lambda x: x['RPN'], reverse=True)

    # Display risk matrix
    st.markdown("#### 🎯 Risk Priority Ranking (High to Low RPN)")

    for item in risk_data[:10]:  # Top 10 risks
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(231,76,60,0.05) 0%, rgba(255,255,255,1) 100%);
                    border-left: 4px solid {item['risk_color']}; padding: 1rem; margin-bottom: 0.8rem;
                    border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h4 style="color: #004366; font-family: 'Poppins', sans-serif; margin: 0 0 0.3rem 0; font-weight: 600;">
                        {item['Product']} ({item['SKU']})
                    </h4>
                    <p style="color: #555; font-family: 'Poppins', sans-serif; font-size: 0.9em; margin: 0.3rem 0;">
                        <strong>Failure Mode:</strong> {item['Failure Mode']}
                    </p>
                    <p style="color: #666; font-family: 'Poppins', sans-serif; font-size: 0.85em; margin: 0.3rem 0;">
                        S: {item['Severity (S)']} × O: {item['Occurrence (O)']} × D: {item['Detection (D)']} = <strong>RPN: {item['RPN']}</strong>
                    </p>
                </div>
                <div style="text-align: center; min-width: 100px;">
                    <div style="font-size: 2em;">{item['Risk Level'].split()[0]}</div>
                    <div style="color: {item['risk_color']}; font-weight: 600; font-size: 0.9em;">{item['Risk Level'].split()[1]}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Risk Heat Map
    st.markdown("#### 🗺️ Risk Heat Map (Severity vs Occurrence)")

    st.markdown("""
    <div style="background: #fff3cd; border: 1px solid #ffc107; padding: 1rem; border-radius: 6px; margin: 1rem 0;">
        <strong>📋 FMEA Rating Scale (1-10):</strong><br/>
        <strong>Severity:</strong> 10=Catastrophic, 7-9=Serious, 4-6=Moderate, 1-3=Minor<br/>
        <strong>Occurrence:</strong> 10=Very High (>100 cases), 7-9=High (50-99), 4-6=Moderate (20-49), 1-3=Low (<20)<br/>
        <strong>Detection:</strong> 10=Cannot detect, 7-9=Poor detection, 4-6=Moderate, 1-3=High detection capability
    </div>
    """, unsafe_allow_html=True)

    # Mitigation Recommendations
    st.markdown("#### 🎯 Risk Mitigation Recommendations")

    critical_risks = [r for r in risk_data if r['RPN'] >= 200]
    if critical_risks:
        st.error(f"🔴 **{len(critical_risks)} Critical Risks (RPN ≥ 200)** - Immediate action required!")

        for risk in critical_risks[:3]:
            with st.expander(f"Critical: {risk['Product']} - RPN {risk['RPN']}"):
                st.markdown(f"**Failure Mode:** {risk['Failure Mode']}")
                st.markdown("**Recommended Actions:**")
                st.markdown("""
                1. **Immediate:** Contain the issue - stop shipments if necessary
                2. **Short-term:** Implement corrective action within 30 days
                3. **Long-term:** Process/design changes to reduce occurrence
                4. **Detection:** Enhance inspection/testing to improve detection rating
                """)

                if tracker.ai_analyzer:
                    if st.button(f"🤖 AI Risk Mitigation Plan for {risk['SKU']}", key=f"risk_ai_{risk['SKU']}"):
                        with st.spinner("Generating mitigation plan..."):
                            # Find the case
                            case = next((c for c in tracker.cases if c.sku == risk['SKU']), None)
                            if case:
                                prompt = f"""Create a risk mitigation plan for this quality issue:

Product: {case.product_name}
Failure Mode: {case.top_issues}
Current RPN: {risk['RPN']} (Severity: {risk['Severity (S)']}, Occurrence: {risk['Occurrence (O)']}, Detection: {risk['Detection (D)']})

Provide specific recommendations to:
1. Reduce Severity (product redesign, safety features)
2. Reduce Occurrence (process improvements, quality controls)
3. Improve Detection (inspection methods, testing protocols)

Include timeline and responsibility assignments."""

                                plan = tracker.ai_analyzer.generate_text(
                                    prompt,
                                    "You are a quality engineer expert in FMEA and risk mitigation strategies.",
                                    mode='chat'
                                )

                                st.markdown("""
                                <div style="background: white; border: 2px solid #e74c3c; padding: 1.5rem;
                                            border-radius: 8px; margin: 1rem 0;">
                                """, unsafe_allow_html=True)
                                st.markdown(plan)
                                st.markdown("</div>", unsafe_allow_html=True)


def render_predictive_analytics(tracker):
    """AI-Powered Predictive Quality Analytics and Forecasting"""
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(46,204,113,0.1) 0%, rgba(52,152,219,0.1) 100%);
                border-left: 4px solid #2ecc71; padding: 1rem; margin: 1rem 0; border-radius: 6px;">
        <h4 style="color: #27ae60; font-family: 'Poppins', sans-serif; margin: 0 0 0.5rem 0;">
            📈 Predictive Analytics & AI Forecasting
        </h4>
        <p style="color: #666; font-family: 'Poppins', sans-serif; font-size: 0.9em; margin: 0;">
            Machine learning-powered forecasting of quality trends, return rates, and cost impacts.
            Predict future quality issues before they occur.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not tracker.ai_analyzer:
        st.warning("⚠️ AI analyzer not available. Predictive analytics requires AI capabilities.")
        return

    # Predictive Analytics Dashboard
    st.markdown("### 🔮 Quality Trend Forecasting")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="background: #e8f5e9; border: 2px solid #4caf50; padding: 1rem; border-radius: 8px; text-align: center;">
            <h3 style="color: #27ae60; margin: 0;">📊</h3>
            <p style="color: #666; margin: 0.5rem 0; font-weight: 600;">Trend Analysis</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background: #fff3e0; border: 2px solid #ff9800; padding: 1rem; border-radius: 8px; text-align: center;">
            <h3 style="color: #f57c00; margin: 0;">🎯</h3>
            <p style="color: #666; margin: 0.5rem 0; font-weight: 600;">Risk Prediction</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="background: #e3f2fd; border: 2px solid #2196f3; padding: 1rem; border-radius: 8px; text-align: center;">
            <h3 style="color: #1976d2; margin: 0;">💰</h3>
            <p style="color: #666; margin: 0.5rem 0; font-weight: 600;">Cost Forecast</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Aggregate quality metrics
    total_return_rate = sum(c.return_rate_amazon or 0 for c in tracker.cases) / len(tracker.cases) if tracker.cases else 0
    total_refund_cost = sum(c.cost_of_refunds_annualized or 0 for c in tracker.cases)
    total_ncx = sum(c.ncx_orders or 0 for c in tracker.cases)

    # Build context for AI
    context = f"""Current Quality Metrics:
- Total Cases: {len(tracker.cases)}
- Average Return Rate: {total_return_rate*100:.2f}%
- Total Annual Refund Cost: ${total_refund_cost:,.0f}
- Total NCX Orders: {total_ncx}

Top Issues:
"""

    issue_counts = {}
    for case in tracker.cases:
        if case.top_issues:
            issue_counts[case.top_issues[:50]] = issue_counts.get(case.top_issues[:50], 0) + 1

    top_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    for issue, count in top_issues:
        context += f"- {issue}: {count} occurrences\n"

    # Prediction Type Selection
    prediction_type = st.selectbox(
        "Select Prediction Type",
        [
            "Quality Trend Forecast (30/60/90 days)",
            "Risk Prediction - Emerging Issues",
            "Cost Impact Forecast",
            "Product-Specific Forecast"
        ]
    )

    if prediction_type == "Quality Trend Forecast (30/60/90 days)":
        st.markdown("#### 📈 Return Rate Trend Forecast")

        if st.button("🤖 Generate Trend Forecast", type="primary"):
            with st.spinner("Analyzing patterns and generating forecast..."):
                prompt = f"""{context}

Based on these current quality metrics and patterns, provide a data-driven forecast for:

1. **30-Day Outlook:**
   - Expected average return rate
   - Confidence level (High/Medium/Low)
   - Key risk factors

2. **60-Day Outlook:**
   - Projected return rate trend
   - Potential new issues to watch
   - Cost implications

3. **90-Day Outlook:**
   - Long-term quality trend
   - Recommended proactive actions
   - Expected ROI from quality improvements

Consider:
- Current issue patterns and recurrence
- Seasonal factors (if applicable)
- Actions already taken
- Industry benchmarks (medical devices: 5-8% acceptable return rate)

Format with clear headers and bullet points."""

                forecast = tracker.ai_analyzer.generate_text(
                    prompt,
                    "You are a predictive analytics expert specializing in quality forecasting and statistical trend analysis.",
                    mode='chat'
                )

                st.markdown("""
                <div style="background: linear-gradient(135deg, rgba(46,204,113,0.1) 0%, rgba(52,152,219,0.1) 100%);
                            border: 2px solid #2ecc71; padding: 1.5rem; border-radius: 8px; margin: 1rem 0;">
                """, unsafe_allow_html=True)
                st.markdown(forecast)
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("""
                <div style="background: #fffde7; border-left: 4px solid #fbc02d; padding: 1rem; margin: 1rem 0;">
                    <strong>⚠️ Note:</strong> This forecast is AI-generated based on current data patterns.
                    Actual outcomes may vary. Use as a planning tool alongside traditional statistical methods.
                </div>
                """, unsafe_allow_html=True)

    elif prediction_type == "Risk Prediction - Emerging Issues":
        st.markdown("#### 🎯 Emerging Risk Prediction")

        if st.button("🤖 Identify Emerging Risks", type="primary"):
            with st.spinner("Analyzing cross-case patterns for emerging risks..."):
                prompt = f"""{context}

Analyze these quality cases for emerging risk patterns:

1. **Pattern Detection:**
   - Are there similar issues across different products?
   - Common root causes appearing?
   - Vendor/supplier patterns?

2. **Early Warning Signals:**
   - Which products are showing early signs of quality degradation?
   - Trends that might escalate if not addressed?

3. **Risk Prioritization:**
   - Rank emerging risks by severity and likelihood
   - Recommend which to investigate first

4. **Preventive Actions:**
   - What can be done now to prevent these from becoming major issues?

Focus on trends not immediately obvious in individual cases."""

                analysis = tracker.ai_analyzer.generate_text(
                    prompt,
                    "You are a quality management AI expert at pattern recognition and predictive risk analysis.",
                    mode='chat'
                )

                st.markdown("""
                <div style="background: linear-gradient(135deg, rgba(231,76,60,0.1) 0%, rgba(255,255,255,1) 100%);
                            border: 2px solid #e74c3c; padding: 1.5rem; border-radius: 8px; margin: 1rem 0;">
                """, unsafe_allow_html=True)
                st.markdown(analysis)
                st.markdown("</div>", unsafe_allow_html=True)

    elif prediction_type == "Cost Impact Forecast":
        st.markdown("#### 💰 Cost of Quality Forecast")

        current_monthly_cost = total_refund_cost / 12 if total_refund_cost > 0 else 0

        st.metric(
            "Current Monthly Refund Cost",
            f"${current_monthly_cost:,.0f}",
            help="Based on annualized refund costs"
        )

        if st.button("🤖 Generate Cost Forecast", type="primary"):
            with st.spinner("Forecasting cost impacts..."):
                prompt = f"""{context}

Current Monthly Refund Cost: ${current_monthly_cost:,.0f}
Annual Projection: ${total_refund_cost:,.0f}

Provide a cost impact forecast:

1. **If No Action Taken:**
   - 3-month cost projection
   - 6-month cost projection
   - Risk of escalation

2. **If Current Actions Succeed:**
   - Expected cost reduction
   - Timeline to see results
   - ROI calculation

3. **Best Case Scenario:**
   - Maximum achievable cost reduction
   - Required actions to achieve
   - Timeline

4. **Recommendations:**
   - Highest ROI opportunities
   - Quick wins (30 days)
   - Long-term investments

Include specific dollar amounts and percentages."""

                forecast = tracker.ai_analyzer.generate_text(
                    prompt,
                    "You are a financial analyst specializing in cost of quality analysis and ROI forecasting.",
                    mode='chat'
                )

                st.markdown("""
                <div style="background: linear-gradient(135deg, rgba(241,196,15,0.1) 0%, rgba(243,156,18,0.1) 100%);
                            border: 2px solid #f39c12; padding: 1.5rem; border-radius: 8px; margin: 1rem 0;">
                """, unsafe_allow_html=True)
                st.markdown(forecast)
                st.markdown("</div>", unsafe_allow_html=True)

    elif prediction_type == "Product-Specific Forecast":
        st.markdown("#### 📦 Product-Specific Quality Forecast")

        selected_case_idx = st.selectbox(
            "Select Product for Forecast",
            options=range(len(tracker.cases)),
            format_func=lambda i: f"{tracker.cases[i].product_name} ({tracker.cases[i].sku}) - RR: {tracker.cases[i].return_rate_amazon*100:.1f}%"
        )

        case = tracker.cases[selected_case_idx]

        st.markdown(f"**Product:** {case.product_name}")
        st.markdown(f"**Current Return Rate:** {case.return_rate_amazon*100:.1f}%")
        st.markdown(f"**Issue:** {case.top_issues}")

        if st.button("🤖 Generate Product Forecast", type="primary"):
            with st.spinner(f"Analyzing {case.product_name}..."):
                prompt = f"""Product Quality Forecast:

Product: {case.product_name} ({case.sku})
Current Return Rate: {case.return_rate_amazon*100:.1f}%
Issue: {case.top_issues}
Action Taken: {case.action_taken if case.action_taken else 'None yet'}
Sales Channel: {case.main_sales_channel}

Provide a detailed forecast:

1. **Quality Trajectory:**
   - If no action: where will this product be in 90 days?
   - If action is taken: expected improvement timeline

2. **Specific Predictions:**
   - Return rate in 30/60/90 days
   - Customer satisfaction trend
   - Warranty cost implications

3. **Decision Recommendations:**
   - Should we continue selling this product?
   - What improvements are critical?
   - Timeline for corrective actions

4. **Success Metrics:**
   - What KPIs should we track?
   - When should we see improvement?
   - Verification checkpoints

Be specific with numbers and timelines."""

                forecast = tracker.ai_analyzer.generate_text(
                    prompt,
                    "You are a product quality analyst expert in forecasting product performance and quality trends.",
                    mode='chat'
                )

                st.markdown("""
                <div style="background: white; border: 2px solid #3498db; padding: 1.5rem;
                            border-radius: 8px; margin: 1rem 0;">
                """, unsafe_allow_html=True)
                st.markdown(forecast)
                st.markdown("</div>", unsafe_allow_html=True)


# =====================================================
# AI-ASSISTED QUALITY CASE SCREENING WIZARD
# =====================================================

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


# Legacy compatibility - flat threshold dict
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
        # Product-type specific thresholds
        'Post Op Shoes': 0.24,
        'Electric Wheelchair': 0.15,
        'Scooter': 0.10,
        'Knee Walker': 0.11,
        'Canes': 0.075,
        'Stand Assist': 0.185,
        'Commode Rail': 0.17,
        'Splint': 0.14,
        'Shoulder Brace': 0.14,
        'Groin': 0.20,
        'Transfer Belts': 0.105,
        'Ice/Bracing': 0.06,
        'Gauze': 0.025,
        'APM': 0.18,
        'Thigh': 0.13,
        'Sling': 0.13,
        'Crutch Pads': 0.13,
        'Shin Support': 0.12,
        'Wraps': 0.11,
        'Ankle Wrap': 0.11,
        'Wrist': 0.11,
        'Strap': 0.11,
        'Gloves': 0.075,
        'Toe Separators': 0.05,
        'Bunion': 0.14,
        'Massage Ball': 0.05,
        'Shoulder Pulley': 0.04,
        'Walker Accessories': 0.055,
        'Shower Mat': 0.09,
        'Chair Cushion': 0.085,
        'Wheelchair Cushion': 0.085,
        'Commode Cushion': 0.16,
        'Commode Riser': 0.168,
        'Walker Bag': 0.07
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


def render_ai_screening_wizard(tracker):
    """
    AI-Assisted Quality Case Screening Wizard

    Guides user through SOP-based screening with AI recommendations.
    Populates Smartsheet-compatible case data with priority assessment.
    """

    # Initialize wizard state
    if 'wizard_state' not in st.session_state:
        st.session_state.wizard_state = {
            'step': 0,
            'case_data': {},
            'ai_recommendation': None,
            'priority_score': 0,
            'override_requested': False,
            'thresholds': SCREENING_THRESHOLDS.copy()
        }

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

    # Expandable wizard content
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

        # ==================== STEP 0: FLAG SOURCE ====================
        if current_step == 0:
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

        # ==================== STEP 1: PRODUCT & METRICS ====================
        elif current_step == 1:
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

            # Real-time threshold check using selected threshold
            st.markdown("##### 🎯 Real-Time Threshold Analysis")
            threshold = selected_threshold

            col_thresh1, col_thresh2, col_thresh3 = st.columns(3)
            with col_thresh1:
                if return_rate > 0:
                    if return_rate / 100 > threshold:
                        st.error(f"⚠️ Return Rate {return_rate}% EXCEEDS threshold ({threshold*100}%)")
                    else:
                        st.success(f"✅ Return Rate {return_rate}% within threshold ({threshold*100}%)")
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

        # ==================== STEP 2: ISSUE ANALYSIS ====================
        elif current_step == 2:
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

        # ==================== STEP 3: ACTION PLANNING ====================
        elif current_step == 3:
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
            else:
                cost_of_refunds = 0.0
                savings_captured = 0.0

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
                    wizard['step'] = 4
                    st.rerun()

        # ==================== STEP 4: PRIORITY REVIEW & SUBMISSION ====================
        elif current_step == 4:
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
            # Use saved threshold from product selection, or fall back to category default
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

            # Threshold configuration
            with st.expander("⚙️ Adjust Screening Thresholds", expanded=False):
                st.caption("Modify thresholds for this screening session")

                thresh_col1, thresh_col2 = st.columns(2)
                with thresh_col1:
                    st.markdown("**Return Rate Thresholds by Category:**")
                    for cat, thresh in wizard['thresholds']['return_rate'].items():
                        new_thresh = st.number_input(
                            f"{cat} (%)",
                            min_value=0.0, max_value=50.0,
                            value=thresh * 100,
                            step=0.5,
                            key=f"thresh_{cat}"
                        )
                        wizard['thresholds']['return_rate'][cat] = new_thresh / 100

                with thresh_col2:
                    st.markdown("**Other Thresholds:**")
                    wizard['thresholds']['ncx_rate'] = st.number_input(
                        "NCX Rate Threshold (%)",
                        min_value=0.0, max_value=20.0,
                        value=wizard['thresholds']['ncx_rate'] * 100,
                        step=0.5,
                        key="thresh_ncx"
                    ) / 100

                    wizard['thresholds']['star_rating'] = st.number_input(
                        "Star Rating Threshold",
                        min_value=1.0, max_value=5.0,
                        value=wizard['thresholds']['star_rating'],
                        step=0.1,
                        key="thresh_star"
                    )

                    wizard['thresholds']['cost_threshold'] = st.number_input(
                        "Cost Threshold ($)",
                        min_value=0, max_value=100000,
                        value=int(wizard['thresholds']['cost_threshold']),
                        step=1000,
                        key="thresh_cost"
                    )

                if st.button("🔄 Recalculate Priority with New Thresholds", key="recalc_priority"):
                    st.rerun()

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

                    # Add to tracker
                    tracker.add_case(new_case)
                    st.session_state.tracker_cases = tracker.cases

                    # Reset wizard
                    wizard['step'] = 0
                    wizard['case_data'] = {}

                    st.success(f"✅ Case added to tracker: {new_case.product_name} ({new_case.sku}) - Priority: {priority_level}")
                    st.balloons()
                    st.rerun()


def render_quality_cases_dashboard():
    """Render the Quality Tracker Dashboard - Manual Entry with Leadership/Company Wide Exports"""

    # Hero header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #23b2be 0%, #004366 100%);
                padding: 2rem; border-radius: 12px; margin-bottom: 2rem;
                box-shadow: 0 6px 12px rgba(0,0,0,0.15);">
        <h1 style="color: white; font-family: 'Poppins', sans-serif; margin-bottom: 0.5rem; font-weight: 700; font-size: 2.2em;">
            📊 Quality Tracker Dashboard
        </h1>
        <p style="color: rgba(255,255,255,0.95); font-family: 'Poppins', sans-serif; font-size: 1.1em; margin-bottom: 0; line-height: 1.6;">
            <strong>🔄 Smartsheet Workflow:</strong> Import cases from Smartsheet → Screen & analyze with AI → Export confirmed cases back to Smartsheet
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("💡 How This Tool Works", expanded=False):
        st.markdown("""
**🎯 Purpose:** This tool is for *screening* quality cases, generating AI summaries, and preparing exports.
Think of it as a smart workspace for case analysis before committing to Smartsheet.

**🔄 4-Step Workflow:**
""")
        # Visual workflow using columns (renders cleanly)
        wf1, wf2, wf3, wf4 = st.columns(4)
        with wf1:
            st.info("**📥 IMPORT**\n\nFrom Smartsheet (Excel/CSV)")
        with wf2:
            st.warning("**🔍 SCREEN**\n\nDuplicates & AI Review")
        with wf3:
            st.success("**➕ ADD**\n\nNew Cases Manually")
        with wf4:
            st.info("**📤 EXPORT**\n\nTo Smartsheet (Permanent DB)")

        st.markdown("""
⚠️ **Important:** This tool has **no memory between sessions**.
Smartsheet is your permanent database. Use this tool to screen cases, take screenshots for emails,
and export confirmed cases back to Smartsheet for tracking.
""")

    # Initialize tracker manager (EMPTY state by default) - MUST be OUTSIDE the expander
    if 'quality_tracker' not in st.session_state:
        st.session_state.quality_tracker = QualityTrackerManager(st.session_state.get('ai_analyzer'))
        st.session_state.tracker_cases = []
        st.session_state.show_leadership_fields = False

    tracker = st.session_state.quality_tracker

    # Action buttons section with enhanced styling
    st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(35,178,190,0.08) 0%, rgba(0,67,102,0.08) 100%);
                    padding: 1.2rem; border-radius: 10px; margin: 1rem 0;">
            <h4 style="color: #004366; font-family: 'Poppins', sans-serif; margin-bottom: 0.8rem; font-weight: 600;">
                ⚡ Quick Actions
            </h4>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 2])

    with col1:
        if st.button(
            "📥 Load Demo",
            key="load_demo_tracker",
            help="Load 3 sample cases for testing",
            type="secondary",
            use_container_width=True
        ):
            demo_cases = generate_demo_tracker_cases()
            st.session_state.tracker_cases = demo_cases
            tracker.cases = demo_cases
            st.success("✅ Loaded 3 demo cases")
            st.rerun()

    with col2:
        # Import from file
        uploaded_file = st.file_uploader(
            "📂 Import",
            type=['xlsx', 'csv'],
            key="import_cases_file",
            help="Import cases from Smartsheet export (Excel or CSV)",
            label_visibility="visible"
        )
        if uploaded_file is not None:
            try:
                file_type = 'excel' if uploaded_file.name.endswith('.xlsx') else 'csv'
                imported_count, duplicates = tracker.import_from_file(uploaded_file, file_type)

                if imported_count > 0:
                    st.success(f"✅ Imported {imported_count} cases")
                    if duplicates:
                        st.warning(f"⚠️ Skipped {len(duplicates)} duplicate SKUs: {', '.join(duplicates[:5])}")
                    st.session_state.tracker_cases = tracker.cases
                    st.rerun()
                else:
                    st.info("No new cases imported (all duplicates or invalid data)")
            except Exception as e:
                st.error(f"❌ Import failed: {str(e)}")

    with col3:
        st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)
        st.session_state.show_leadership_fields = st.checkbox(
            "🔒 Leadership",
            value=st.session_state.show_leadership_fields,
            help="Show Priority, Total orders, Financials, Case Status"
        )

    with col4:
        if tracker.cases:
            if st.button(
                "🗑️ Clear All",
                key="clear_all_cases",
                help="Remove all cases from current session",
                type="secondary",
                use_container_width=True
            ):
                st.session_state.tracker_cases = []
                tracker.cases = []
                st.success("✅ Cleared all cases")
                st.rerun()

    with col5:
        if tracker.cases and tracker.ai_analyzer:
            if st.button(
                "🤖 AI Review",
                key="ai_review_all",
                help="Generate AI analysis of all current cases",
                type="primary",
            use_container_width=True
            ):
                with st.spinner("🤖 Analyzing cases..."):
                    review = tracker.generate_ai_review()

                    # Display AI review with enhanced formatting
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, rgba(35,178,190,0.15) 0%, rgba(0,67,102,0.15) 100%);
                                border-left: 5px solid #23b2be; padding: 1.5rem; margin: 1rem 0;
                                border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                        <h4 style="color: #004366; font-family: 'Poppins', sans-serif; margin-bottom: 1rem; font-weight: 600;">
                            🤖 AI Quality Expert Analysis
                        </h4>
                        <p style="color: #333; font-family: 'Poppins', sans-serif; font-size: 0.9em; margin-bottom: 0.5rem;">
                            <strong>What this analysis provides:</strong> AI has reviewed all loaded cases and identified the top priorities,
                            common patterns, and recommended actions based on severity, return rates, and business impact.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(f"""
                    <div style="background: white; border: 2px solid #23b2be; padding: 1.5rem;
                                border-radius: 8px; font-family: 'Poppins', sans-serif; line-height: 1.7;">
                        {review}
                    </div>
                    """, unsafe_allow_html=True)

                    st.caption("💡 Use this analysis to prioritize cases for corrective action. Export these cases to Smartsheet to track progress.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Session Status Badge
    session_case_count = len(tracker.cases)
    if session_case_count > 0:
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, #27ae60 0%, #229954 100%);
                    color: white; padding: 0.6rem 1.2rem; border-radius: 20px;
                    display: inline-block; font-family: 'Poppins', sans-serif; font-weight: 600;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 1rem;">
            ✓ Session Active: {session_case_count} case{'s' if session_case_count != 1 else ''} loaded
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: linear-gradient(90deg, #95a5a6 0%, #7f8c8d 100%);
                    color: white; padding: 0.6rem 1.2rem; border-radius: 20px;
                    display: inline-block; font-family: 'Poppins', sans-serif; font-weight: 600;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 1rem;">
            ○ No cases loaded - Import or load demo data to begin
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Cases Summary (if cases exist)
    if tracker.cases:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #23b2be 0%, #004366 100%);
                    padding: 1.5rem; border-radius: 10px; margin-bottom: 1.5rem;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h3 style="color: white; font-family: 'Poppins', sans-serif; margin-bottom: 1rem; font-weight: 600;">
                📊 Quality Case Summary
            </h3>
            <p style="color: rgba(255,255,255,0.9); font-family: 'Poppins', sans-serif; font-size: 0.95em; margin-bottom: 0;">
                💡 <strong>About these metrics:</strong> This summary shows live data from cases currently loaded in this session.
                These cases were either imported from Smartsheet or manually entered. Use AI Review to get intelligent analysis,
                then export confirmed cases back to Smartsheet for permanent storage.
                <em>Remember: This tool has no memory between sessions - Smartsheet is your permanent database.</em>
            </p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Total Cases",
                len(tracker.cases),
                help="Total number of quality cases currently loaded in this session (imported or manually added)"
            )
        with col2:
            total_refund_cost = sum(c.cost_of_refunds_annualized or 0 for c in tracker.cases)
            st.metric(
                "Total Refund Cost (Annual)",
                f"${total_refund_cost:,.0f}" if st.session_state.show_leadership_fields else "---",
                help="Sum of annualized refund costs across all cases (Leadership view only). Based on return rate × order volume × average product cost."
            )
        with col3:
            total_savings = sum(c.savings_captured_12m or 0 for c in tracker.cases)
            st.metric(
                "Total Savings (12m)",
                f"${total_savings:,.0f}" if st.session_state.show_leadership_fields else "---",
                help="Total savings captured over last 12 months from corrective actions (Leadership view only). Calculated from return rate reduction × volume."
            )
        with col4:
            avg_return_rate = sum(c.return_rate_amazon or 0 for c in tracker.cases) / len(tracker.cases) if tracker.cases else 0
            st.metric(
                "Avg Return Rate",
                f"{avg_return_rate:.2%}",
                help="Average return rate across all loaded cases. Industry benchmark for medical supplies: 5-8%. Above 10% requires immediate action."
            )

        st.markdown("---")

        # Check for duplicate SKUs
        duplicates = tracker.find_duplicate_skus()
        if duplicates:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(231,76,60,0.15) 0%, rgba(231,76,60,0.05) 100%);
                        border-left: 5px solid #e74c3c; padding: 1.2rem; margin: 1rem 0;
                        border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h4 style="color: #c0392b; font-family: 'Poppins', sans-serif; margin-bottom: 0.5rem; font-weight: 600;">
                    ⚠️ Duplicate SKUs Detected
                </h4>
                <p style="color: #333; font-family: 'Poppins', sans-serif; font-size: 0.95em; margin: 0;">
                    Found <strong>{len(duplicates)} SKUs</strong> appearing multiple times across different sources.
                    Review these carefully - they may be legitimate cases from different channels or data entry errors.
                </p>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("🔍 View Duplicate Details", expanded=True):
                for sku, cases in duplicates.items():
                    st.markdown(f"""
                    <div style="background: rgba(35,178,190,0.05); border-left: 3px solid #23b2be;
                                padding: 0.8rem; margin-bottom: 0.5rem; border-radius: 4px;">
                        <p style="font-family: 'Poppins', sans-serif; color: #004366; font-weight: 600; margin: 0 0 0.5rem 0;">
                            SKU: <strong>{sku}</strong> - appears in {len(cases)} cases:
                        </p>
                    """, unsafe_allow_html=True)
                    for case in cases:
                        st.markdown(f"  → {case.product_name} ({case.flag_source})")
                    st.markdown("</div>", unsafe_allow_html=True)

        # Display cases table with enhanced header
        st.markdown("""
        <div style="background: linear-gradient(90deg, rgba(35,178,190,0.1) 0%, rgba(0,67,102,0.1) 100%);
                    border-left: 4px solid #23b2be; padding: 1rem; margin: 1rem 0;
                    border-radius: 6px;">
            <h4 style="color: #004366; font-family: 'Poppins', sans-serif; margin: 0 0 0.3rem 0; font-weight: 600;">
                📋 Current Cases in Session
            </h4>
            <p style="color: #666; font-family: 'Poppins', sans-serif; font-size: 0.85em; margin: 0;">
                💡 Cases displayed here for screening review. Export confirmed cases to Smartsheet for permanent tracking.
            </p>
        </div>
        """, unsafe_allow_html=True)

        cases_df = tracker.get_cases_dataframe(leadership_version=st.session_state.show_leadership_fields)

        if not cases_df.empty:
            # Show only key columns for display
            display_cols = ['Product name', 'SKU', 'Return rate Amazon', 'Top Issue(s)', 'Case Status'] if st.session_state.show_leadership_fields else ['Product name', 'SKU', 'Return rate Amazon', 'Top Issue(s)']
            display_cols = [c for c in display_cols if c in cases_df.columns]

            if display_cols:
                display_df = cases_df[display_cols].copy()

                # Highlight duplicate SKUs
                if duplicates:
                    duplicate_sku_list = list(duplicates.keys())
                    display_df['SKU'] = display_df['SKU'].apply(
                        lambda x: f"⚠️ {x}" if x in duplicate_sku_list else x
                    )

                # Format return rate
                if 'Return rate Amazon' in display_df.columns:
                    display_df['Return rate Amazon'] = display_df['Return rate Amazon'].apply(
                        lambda x: f"{x:.2%}" if pd.notna(x) and x is not None else "N/A"
                    )

                st.dataframe(display_df, use_container_width=True, height=300)

        st.markdown("---")

        # Export Section with enhanced styling
        st.markdown("""
        <div style="background: linear-gradient(135deg, #23b2be 0%, #004366 100%);
                    padding: 1.5rem; border-radius: 10px; margin: 1.5rem 0;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            <h3 style="color: white; font-family: 'Poppins', sans-serif; margin-bottom: 0.5rem; font-weight: 600;">
                📤 Export to Smartsheet
            </h3>
            <p style="color: rgba(255,255,255,0.9); font-family: 'Poppins', sans-serif; font-size: 0.95em; margin: 0;">
                Export confirmed cases back to Smartsheet for permanent tracking. Choose Leadership (full data) or Company Wide (sanitized).
            </p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(0,67,102,0.1) 0%, rgba(0,67,102,0.02) 100%);
                        border: 2px solid #004366; padding: 1.2rem; border-radius: 8px; margin-bottom: 1rem;">
                <h4 style="color: #004366; font-family: 'Poppins', sans-serif; margin: 0 0 0.5rem 0; font-weight: 600;">
                    🔒 Leadership Export
                </h4>
                <p style="color: #555; font-family: 'Poppins', sans-serif; font-size: 0.85em; margin: 0 0 0.3rem 0;">
                    31 columns with full financial data
                </p>
                <p style="color: #666; font-family: 'Poppins', sans-serif; font-size: 0.8em; margin: 0; line-height: 1.4;">
                    <strong>Includes:</strong> Priority, Total orders, Cost of Refunds, Savings, Case Status
                </p>
            </div>
            """, unsafe_allow_html=True)

            col1a, col1b = st.columns(2)
            with col1a:
                leadership_excel = tracker.export_leadership_excel()
                st.download_button(
                    "📥 Excel",
                    data=leadership_excel,
                    file_name="Tracker_ Priority List (Leadership).xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_leadership_excel",
                    type="primary",
                    use_container_width=True
                )
            with col1b:
                leadership_csv = tracker.export_leadership_csv()
                st.download_button(
                    "📥 CSV",
                    data=leadership_csv,
                    file_name="Tracker_ Priority List (Leadership).csv",
                    mime="text/csv",
                    key="dl_leadership_csv",
                    use_container_width=True
                )

        with col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(35,178,190,0.1) 0%, rgba(35,178,190,0.02) 100%);
                        border: 2px solid #23b2be; padding: 1.2rem; border-radius: 8px; margin-bottom: 1rem;">
                <h4 style="color: #23b2be; font-family: 'Poppins', sans-serif; margin: 0 0 0.5rem 0; font-weight: 600;">
                    🌐 Company Wide Export
                </h4>
                <p style="color: #555; font-family: 'Poppins', sans-serif; font-size: 0.85em; margin: 0 0 0.3rem 0;">
                    25 columns, sanitized (no financials)
                </p>
                <p style="color: #666; font-family: 'Poppins', sans-serif; font-size: 0.8em; margin: 0; line-height: 1.4;">
                    <strong>Excludes:</strong> Priority, Total orders, Flag Source 1, Financials, Case Status
                </p>
            </div>
            """, unsafe_allow_html=True)

            col2a, col2b = st.columns(2)
            with col2a:
                company_excel = tracker.export_company_wide_excel()
                st.download_button(
                    "📥 Excel",
                    data=company_excel,
                    file_name="Company Wide Quality Tracker.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_company_excel",
                    type="primary",
                    use_container_width=True
                )
            with col2b:
                company_csv = tracker.export_company_wide_csv()
                st.download_button(
                    "📥 CSV",
                    data=company_csv,
                    file_name="Company Wide Quality Tracker.csv",
                    mime="text/csv",
                    key="dl_company_csv",
                    use_container_width=True
                )

        st.markdown("---")

    # =====================================================
    # AI-ASSISTED QUALITY CASE SCREENING WIZARD
    # =====================================================
    if MODULAR_IMPORTS:
        wizard_render(tracker, QualityTrackerCase)
    else:
        render_ai_screening_wizard(tracker)

    st.markdown("---")

    # Manual Entry Form
    st.markdown("#### ➕ Add New Quality Case")

    with st.form("add_quality_case_form"):
        st.markdown("##### Product Information")

        col1, col2, col3 = st.columns(3)
        with col1:
            product_name = st.text_input("Product Name*", placeholder="e.g., Vive Mobility Walker")
        with col2:
            sku = st.text_input("SKU*", placeholder="e.g., VMW-001")
        with col3:
            asin = st.text_input("ASIN", placeholder="e.g., B07EXAMPLE")

        col4, col5, col6 = st.columns(3)
        with col4:
            main_sales_channel = st.selectbox("Main Sales Channel", ["Amazon", "B2B", "Direct", "Other"])
        with col5:
            fulfilled_by = st.selectbox("Fulfilled By", ["FBA", "FBM", "Direct Ship", "Other"])
        with col6:
            if st.session_state.show_leadership_fields:
                priority = st.number_input("Priority (1=Highest)", min_value=1, max_value=100, value=1)

        st.markdown("##### Quality Metrics")

        col7, col8, col9, col10 = st.columns(4)
        with col7:
            return_rate_amazon = st.number_input("Return Rate Amazon (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1) / 100
        with col8:
            return_rate_b2b = st.number_input("Return Rate B2B (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1) / 100
        with col9:
            star_rating = st.number_input("Star Rating Amazon", min_value=0.0, max_value=5.0, value=0.0, step=0.1)
        with col10:
            ncx_rate = st.number_input("NCX Rate (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1) / 100

        col11, col12, col13 = st.columns(3)
        with col11:
            ncx_orders = st.number_input("NCX Orders", min_value=0, value=0)
        with col12:
            if st.session_state.show_leadership_fields:
                total_orders_t30 = st.number_input("Total Orders (t30)", min_value=0, value=0)
        with col13:
            return_badge = st.selectbox("Return Badge Displayed?", ["No", "Yes"])

        st.markdown("##### Issue Documentation")

        top_issues = st.text_area("Top Issue(s)*", placeholder="Describe the main quality issues", height=80)
        notification_notes = st.text_area("Notification/Notes", placeholder="Additional context", height=60)

        col14, col15 = st.columns(2)
        with col14:
            flag_source = st.selectbox("Flag Source", ["Returns Analysis", "B2B Sales Feedback", "Reviews Analysis", "Analytics", "Customer Service", "Other"])
        with col15:
            if st.session_state.show_leadership_fields:
                flag_source_1 = st.text_input("Flag Source 1 (Internal)", placeholder="e.g., High Return Rate")

        if st.session_state.show_leadership_fields:
            st.markdown("##### Financial Information (Leadership Only)")
            col16, col17 = st.columns(2)
            with col16:
                cost_of_refunds = st.number_input("Cost of Refunds (Annualized) $", min_value=0.0, value=0.0, step=100.0)
            with col17:
                savings_captured = st.number_input("12m Savings Captured $", min_value=0.0, value=0.0, step=100.0)

        st.markdown("##### Corrective Actions")

        action_taken = st.text_area("Action Taken", placeholder="Describe actions taken to address the issue", height=60)

        col18, col19, col20 = st.columns(3)
        with col18:
            date_action_taken = st.date_input("Date Action Taken", value=None)
        with col19:
            listing_manager_notified = st.selectbox("Listing Manager Notified?", ["No", "Yes"])
        with col20:
            product_dev_notified = st.selectbox("Product Dev Notified?", ["No", "Yes"])

        col21, col22 = st.columns(2)
        with col21:
            follow_up_date = st.date_input("Follow Up Date", value=None)
        with col22:
            if st.session_state.show_leadership_fields:
                case_status = st.selectbox("Case Status", ["Open", "Active Investigation", "Monitoring", "Action Taken - Monitoring", "Closed"])

        st.markdown("##### Results Tracking")

        col23, col24, col25, col26 = st.columns(4)
        with col23:
            result_1_rr = st.number_input("Result 1 (rr%) %", min_value=0.0, max_value=100.0, value=0.0, step=0.1) / 100
        with col24:
            result_check_date_1 = st.date_input("Result Check Date 1", value=None)
        with col25:
            result_2_rr = st.number_input("Result 2 (rr%) %", min_value=0.0, max_value=100.0, value=0.0, step=0.1) / 100
        with col26:
            result_2_date = st.date_input("Result 2 Date", value=None)

        col27, col28 = st.columns(2)
        with col27:
            top_issues_change = st.text_input("Top Issue(s) Change", placeholder="Describe improvements")
        with col28:
            top_issues_change_date = st.date_input("Top Issue(s) Change Date", value=None)

        submitted = st.form_submit_button("➕ Add Case", type="primary")

        if submitted:
            if not product_name or not sku or not top_issues:
                st.error("⚠️ Please fill in required fields: Product Name, SKU, Top Issue(s)")
            else:
                # Create new case
                new_case = QualityTrackerCase()
                new_case.product_name = product_name
                new_case.sku = sku
                new_case.asin = asin
                new_case.main_sales_channel = main_sales_channel
                new_case.fulfilled_by = fulfilled_by
                new_case.return_rate_amazon = return_rate_amazon if return_rate_amazon > 0 else None
                new_case.return_rate_b2b = return_rate_b2b if return_rate_b2b > 0 else None
                new_case.star_rating_amazon = star_rating if star_rating > 0 else None
                new_case.ncx_rate = ncx_rate if ncx_rate > 0 else None
                new_case.ncx_orders = ncx_orders if ncx_orders > 0 else None
                new_case.return_badge_displayed = return_badge
                new_case.top_issues = top_issues
                new_case.notification_notes = notification_notes
                new_case.flag_source = flag_source
                new_case.action_taken = action_taken
                new_case.date_action_taken = date_action_taken
                new_case.listing_manager_notified = listing_manager_notified
                new_case.product_dev_notified = product_dev_notified
                new_case.follow_up_date = follow_up_date
                new_case.result_1_rr = result_1_rr if result_1_rr > 0 else None
                new_case.result_check_date_1 = result_check_date_1
                new_case.result_2_rr = result_2_rr if result_2_rr > 0 else None
                new_case.result_2_date = result_2_date
                new_case.top_issues_change = top_issues_change
                new_case.top_issues_change_date = top_issues_change_date

                # Leadership fields
                if st.session_state.show_leadership_fields:
                    new_case.priority = priority
                    new_case.total_orders_t30 = total_orders_t30 if total_orders_t30 > 0 else None
                    new_case.flag_source_1 = flag_source_1
                    new_case.cost_of_refunds_annualized = cost_of_refunds if cost_of_refunds > 0 else None
                    new_case.savings_captured_12m = savings_captured if savings_captured > 0 else None
                    new_case.case_status = case_status

                # Add to tracker
                tracker.add_case(new_case)
                st.session_state.tracker_cases.append(new_case)

                st.success(f"✅ Added case for {product_name} ({sku})")

                # Generate AI summary if available
                if tracker.ai_analyzer:
                    with st.spinner("🤖 Generating AI summary..."):
                        summary = tracker.generate_ai_summary(new_case)
                        st.info(f"**AI Summary:** {summary}")

                st.rerun()

    st.markdown("---")

    # Advanced Mode Section for World-Class Quality Teams
    # Always show the Advanced Analytics section - available regardless of cases loaded
    st.markdown("""
    <div style="background: linear-gradient(135deg, #8e44ad 0%, #3498db 100%);
                padding: 1.5rem; border-radius: 10px; margin: 1.5rem 0;
                box-shadow: 0 4px 8px rgba(0,0,0,0.15);">
        <h3 style="color: white; font-family: 'Poppins', sans-serif; margin-bottom: 0.5rem; font-weight: 600;">
            🎯 Advanced Quality Analytics (Enterprise)
        </h3>
        <p style="color: rgba(255,255,255,0.9); font-family: 'Poppins', sans-serif; font-size: 0.95em; margin: 0;">
            Enterprise-grade tools for world-class quality management teams - Root Cause Analysis, CAPA, FMEA, and Predictive Analytics
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Show info if no cases loaded
    if not tracker.cases:
        st.info("💡 **Tip:** Load cases (Demo or Import) to unlock full analytics capabilities. Some features work with sample data.")

    with st.expander("🔬 Advanced Quality Analytics Tools", expanded=bool(tracker.cases)):
        adv_tab1, adv_tab2, adv_tab3, adv_tab4 = st.tabs([
            "🔍 Root Cause Analysis",
            "📋 CAPA Management",
            "⚠️ Risk Analysis (FMEA)",
            "📈 Predictive Analytics"
        ])

        with adv_tab1:
            if MODULAR_IMPORTS:
                rca_render(tracker)
            else:
                render_root_cause_analysis(tracker)

        with adv_tab2:
            if MODULAR_IMPORTS:
                capa_render(tracker)
            else:
                render_capa_management(tracker)

        with adv_tab3:
            if MODULAR_IMPORTS:
                fmea_render(tracker)
            else:
                render_risk_analysis_fmea(tracker)

        with adv_tab4:
            if MODULAR_IMPORTS:
                predictive_render(tracker)
            else:
                render_predictive_analytics(tracker)

    st.markdown("---")

    # Report Criteria Section
    st.markdown("#### 📋 Resulting Criteria by Report Type")
    st.caption("Cases are automatically generated when products meet these criteria")

    tab1, tab2, tab3 = st.tabs(["📦 Returns Analysis", "🏢 B2B Sales Feedback", "⭐ Reviews Analysis"])

    with tab1:
        st.markdown(f"**{REPORT_CRITERIA['Returns Analysis']['description']}**")
        st.markdown("**Case Generation Logic:** " + REPORT_CRITERIA['Returns Analysis']['case_trigger'])

        for criterion in REPORT_CRITERIA['Returns Analysis']['criteria']:
            if criterion['name'] == 'Category Return Rate Threshold':
                st.markdown(f"**{criterion['name']}**")
                st.markdown(f"*Logic:* {criterion['logic']}")

                # Show thresholds table
                threshold_data = []
                for category, threshold in criterion['thresholds'].items():
                    threshold_data.append({
                        'Category': category,
                        'Threshold': threshold
                    })
                st.dataframe(pd.DataFrame(threshold_data), use_container_width=True, height=250)
            else:
                st.markdown(f"**{criterion['name']}:** {criterion['logic']}")

    with tab2:
        st.markdown(f"**{REPORT_CRITERIA['B2B Sales Feedback']['description']}**")
        st.markdown("**Case Generation Logic:** " + REPORT_CRITERIA['B2B Sales Feedback']['case_trigger'])

        for criterion in REPORT_CRITERIA['B2B Sales Feedback']['criteria']:
            st.markdown(f"**{criterion['name']}:** {criterion['logic']}")

    with tab3:
        st.markdown(f"**{REPORT_CRITERIA['Reviews Analysis']['description']}**")
        st.markdown("**Case Generation Logic:** " + REPORT_CRITERIA['Reviews Analysis']['case_trigger'])

        for criterion in REPORT_CRITERIA['Reviews Analysis']['criteria']:
            st.markdown(f"**{criterion['name']}:** {criterion['logic']}")



def render_quality_resources():
    """Render Quality Resources tab with categorized regulatory and tool links"""

    # Hero header with gradient
    st.markdown("""
    <div style="background: linear-gradient(135deg, #23b2be 0%, #004366 100%);
                padding: 2rem; border-radius: 12px; margin-bottom: 2rem;
                box-shadow: 0 6px 12px rgba(0,0,0,0.15);">
        <h1 style="color: white; font-family: 'Poppins', sans-serif; margin-bottom: 0.5rem; font-weight: 700; font-size: 2.2em;">
            📚 Quality Resources Hub
        </h1>
        <p style="color: rgba(255,255,255,0.95); font-family: 'Poppins', sans-serif; font-size: 1.1em; margin-bottom: 0; line-height: 1.6;">
            Comprehensive collection of regulatory databases, quality management tools, and global documentation.
            <br>Access Vive internal tools, FDA databases, EU/UK/LATAM regulatory resources, and international standards - all in one place.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Summary metrics with Vive colors
    total_links = get_total_link_count()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(35,178,190,0.2) 0%, rgba(35,178,190,0.05) 100%);
                    padding: 1.5rem; border-radius: 10px; border: 2px solid #23b2be; text-align: center;">
            <h2 style="color: #23b2be; font-family: 'Poppins', sans-serif; font-size: 2.5em; margin: 0; font-weight: 700;">""" + str(total_links) + """</h2>
            <p style="color: #004366; font-family: 'Poppins', sans-serif; font-size: 1em; margin: 0.5rem 0 0 0; font-weight: 600;">Total Resources</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(0,67,102,0.2) 0%, rgba(0,67,102,0.05) 100%);
                    padding: 1.5rem; border-radius: 10px; border: 2px solid #004366; text-align: center;">
            <h2 style="color: #004366; font-family: 'Poppins', sans-serif; font-size: 2.5em; margin: 0; font-weight: 700;">""" + str(len(QUALITY_RESOURCES)) + """</h2>
            <p style="color: #004366; font-family: 'Poppins', sans-serif; font-size: 1em; margin: 0.5rem 0 0 0; font-weight: 600;">Categories</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(35,178,190,0.2) 0%, rgba(35,178,190,0.05) 100%);
                    padding: 1.5rem; border-radius: 10px; border: 2px solid #23b2be; text-align: center;">
            <h2 style="color: #23b2be; font-family: 'Poppins', sans-serif; font-size: 2.5em; margin: 0; font-weight: 700;">15+</h2>
            <p style="color: #004366; font-family: 'Poppins', sans-serif; font-size: 1em; margin: 0.5rem 0 0 0; font-weight: 600;">Countries Covered</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Render each category with improved styling
    for category_name, category_data in QUALITY_RESOURCES.items():
        icon = category_data.get("icon", "📁")
        description = category_data.get("description", "")
        links = category_data.get("links", [])

        with st.expander(f"{icon} **{category_name}** ({len(links)} resources)", expanded=(category_name == "Vive Quality Tools")):
            st.markdown(f"""
            <p style="font-family: 'Poppins', sans-serif; color: #666; font-size: 0.95em; margin-bottom: 1rem;">
                {description}
            </p>
            """, unsafe_allow_html=True)

            # Display links in enhanced card format
            for link in links:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(35,178,190,0.05) 0%, rgba(255,255,255,1) 100%);
                            border-left: 4px solid #23b2be; padding: 1rem; margin-bottom: 0.8rem;
                            border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                    <a href="{link['url']}" target="_blank" style="text-decoration: none;">
                        <h4 style="color: #004366; font-family: 'Poppins', sans-serif; margin: 0 0 0.3rem 0; font-weight: 600;">
                            🔗 {link['name']}
                        </h4>
                    </a>
                    <p style="color: #555; font-family: 'Poppins', sans-serif; font-size: 0.9em; margin: 0; line-height: 1.5;">
                        {link['description']}
                    </p>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Quick reference guide
    with st.expander("📖 **Quick Reference Guide** - When to use which resource"):
        st.markdown("""
        ### 🔍 Use Case Guide

        **Need to check if a device is registered in a country?**
        - US: FDA 510(k) Database or Registration & Listing
        - EU: EUDAMED
        - UK: MHRA Device Registration
        - Canada: Health Canada MDALL
        - Brazil: ANVISA Medical Device Database
        - Mexico: COFEPRIS Medical Device Registry

        **Need to check for recalls or safety alerts?**
        - US: FDA Recall Database, MAUDE Database
        - EU: EU Safety Gate (RAPEX)
        - UK: UK Medical Device Alerts
        - Canada: Canada Recalls & Safety Alerts
        - Global: Autocapa Tool (Vive internal)

        **Need to find a notified body or approved body?**
        - EU: NANDO Database
        - UK: UK Approved Bodies

        **Need to understand regulatory requirements?**
        - US: FDA 21 CFR Part 820 (QSR)
        - EU: EU MDR/IVDR Regulations
        - International: IMDRF Documents, ISO Standards

        **Need to calculate sampling plans?**
        - AQL Calculator (ISO 2859-1 based)

        **Need Vive-specific quality information?**
        - Quality Goals 2026 (annual objectives)
        - Quality Impact Tracker ($) (financial tracking)
        - Quality Intranet Site (central hub)
        - Quality Manual & SOPs (procedures)
        """)


def render_comprehensive_user_guide():
    """Render comprehensive user guide with TQM methodology and actionable examples"""

    with st.expander("📖 **COMPREHENSIVE USER GUIDE** - How to Use Quality Case Screening", expanded=False):

        # Quick Start Guide
        st.markdown("## 🚀 Quick Start (3 Steps)")
        st.markdown("""
        1. **Enter Your Info** → Fill in "Screened By" and "Source of Flag"
        2. **Add Product Data** → Use Lite mode (manual) or Pro mode (upload CSV)
        3. **Click "Run Screening"** → Get instant results with action recommendations

        **💡 First time?** Start with Lite mode and enter 1-2 products manually to learn the tool.
        """)

        st.divider()

        # TQM/Kaizen Philosophy Section
        st.markdown("## 🏭 TQM & Kaizen Methodology (Quality Management Philosophy)")
        st.caption("This tool follows proven Japanese quality management principles")

        col1, col2 = st.columns(2)

        with col1:
            for key in ['kaizen', 'pdca', 'jidoka', 'gemba', 'genchi_genbutsu']:
                term = TQM_TERMINOLOGY[key]
                with st.expander(f"**{term['official']}** = _{term['layman']}_"):
                    st.markdown(f"**Definition:** {term['definition']}")
                    st.info(f"**In This Tool:** {term['in_practice']}")

        with col2:
            for key in ['hoshin_kanri', 'muda', 'yokoten', 'hansei', 'poka_yoke']:
                term = TQM_TERMINOLOGY[key]
                with st.expander(f"**{term['official']}** = _{term['layman']}_"):
                    st.markdown(f"**Definition:** {term['definition']}")
                    st.info(f"**In This Tool:** {term['in_practice']}")

        st.divider()

        # Feature Explanations with Examples
        st.markdown("## 🔍 Feature Guide: What Each Function Does")

        # THRESHOLD PROFILES
        with st.expander("### 1️⃣ Threshold Profiles (Hoshin Kanri = Strategic Goal Alignment)"):
            st.markdown("""
            **What it is:** Pre-set return rate limits for each product category. Products exceeding these get flagged for review.

            **Official Term:** *Hoshin Kanri* (Policy Deployment)
            **Layman's Term:** *Set Your Quality Standards*

            #### How It Works:
            - Each product category has a maximum acceptable return rate (e.g., MOB = 10%, SUP = 11%)
            - Products above their threshold are flagged as potential quality issues
            - You can create multiple profiles for different scenarios

            #### Example Profiles:
            - **"Standard (SOP Defaults)"** → Use everyday for routine screening
            - **"Strict (Pre-Peak Season)"** → Tighten thresholds before Black Friday/holidays
            - **"Post-Launch Monitoring"** → Lower thresholds for new product launches
            - **"Cleanup Mode"** → Higher thresholds when focusing only on critical issues

            #### Actionable Result Example:
            ```
            Product: MOB-2847 (Knee Scooter)
            Return Rate: 12.5%
            Threshold: 10.0%
            Result: ⚠️ FLAGGED - 2.5% above acceptable limit

            ACTION: Investigate immediately. Check if issue is isolated batch or systemic design problem.
            ```

            **💡 Pro Tip:** Start with SOP defaults, then create custom profiles as you learn your data patterns.
            """)

        # STATISTICAL ANALYSIS
        with st.expander("### 2️⃣ Statistical Analysis (Jidoka = Smart Automation)"):
            st.markdown("""
            **What it is:** Mathematical tests that tell you if differences in return rates are real or just random luck.

            **Official Term:** *Jidoka* (Automation with Human Intelligence)
            **Layman's Term:** *Let Math Find the Real Problems*

            #### Available Tests:

            **🤖 Auto (AI Recommended)** - Best for beginners
            AI picks the right statistical test for your data. Use this if you're unsure.

            **📊 ANOVA** - Compare return rates across categories
            *"Are MOB's 12% returns significantly worse than SUP's 8%?"*
            - **F-score**: How different the groups are
            - **p-value < 0.05**: Differences are real (not random chance)
            - **Effect size**: How BIG is the difference (Small/Medium/Large)

            **📈 MANOVA** - Compare multiple metrics at once
            Tests return rate AND cost AND sales volume together

            **🎯 Kruskal-Wallis** - For messy real-world data
            Use when you have outliers or small samples

            #### Actionable Result Example:
            ```
            Test: ANOVA
            F-score: 8.42
            p-value: 0.003
            Result: ✅ SIGNIFICANT - Categories have truly different return rates

            Post-Hoc Test Results:
            - MOB (12%) significantly higher than SUP (8%) → p = 0.002
            - MOB (12%) NOT significantly different from CSH (10.5%) → p = 0.18

            ACTION: Focus investigation on MOB category. SUP is performing well (use as benchmark).
            CSH trends toward MOB levels - add to watch list.
            ```

            **What "Significant" Means:**
            - p < 0.05 = Only 5% chance results are random → TRUST IT, take action
            - p > 0.05 = Could be random variation → MONITOR but don't overreact
            """)

        # RISK SCORING
        with st.expander("### 3️⃣ Risk Score (Weighted Multi-Factor Analysis)"):
            st.markdown("""
            **What it is:** Composite score (0-100) combining return rate, cost, safety, trends, and volume.

            **Official Term:** *Multi-Criteria Decision Analysis (MCDA)*
            **Layman's Term:** *Priority Calculator - What to Fix First*

            #### Risk Score Formula:
            ```
            Risk Score = (25% × Statistical Deviation)
                       + (25% × Financial Impact)
                       + (30% × Safety Severity)
                       + (10% × Trend Direction)
                       + (10% × Complaint Volume)
            ```

            #### Score Interpretation:
            - **0-30 (Low):** 🟢 Normal variation, routine monitoring
            - **31-60 (Medium):** 🟡 Watch closely, investigate if trend continues
            - **61-80 (High):** 🟠 Likely quality issue, investigate this week
            - **81-100 (Critical):** 🔴 Immediate action required, escalate now

            #### Actionable Result Example:
            ```
            Product: MOB-1893 (Electric Wheelchair)
            Risk Score: 87 (CRITICAL)

            Breakdown:
            - Statistical Deviation: 22/25 (return rate 3σ above mean)
            - Financial Impact: 24/25 (landed cost $425 × 13 returns = $5,525 loss)
            - Safety Severity: 28/30 (battery compartment loose → fall risk)
            - Trend: 8/10 (returns increased 40% last 30 days)
            - Volume: 8/10 (13 returns from only 85 sold = 15.3% rate)

            ACTION:
            1. IMMEDIATE: Quarantine remaining inventory (72 units)
            2. SAME DAY: Open critical investigation (safety risk)
            3. NEXT 24HRS: Contact vendor for emergency CAPA
            4. NOTIFY: Regulatory affairs (potential MDR reporting)
            ```

            **💡 Pro Tip:** High risk scores don't always mean bad products. New launches with small sample sizes can score high even if return rate is acceptable. Use judgment!
            """)

        # SPC CONTROL CHARTS
        with st.expander("### 4️⃣ SPC Control Charts (Process Stability Monitoring)"):
            st.markdown("""
            **What it is:** Statistical Process Control - detects when your process goes "out of control"

            **Official Term:** *Shewhart Control Charts / CUSUM*
            **Layman's Term:** *Early Warning System for Trends*

            #### SPC Signals:

            **Normal** 🟢: Within ±1 standard deviation of average
            → Everything operating as expected

            **Watch** 🟡: Between 1-2 standard deviations
            → Keep an eye on it, might be early pattern

            **Warning** 🟠: Between 2-3 standard deviations
            → Investigate within 1 week, likely real issue emerging

            **Critical** 🔴: Beyond 3 standard deviations
            → Immediate investigation, process is out of control

            #### Actionable Result Example:
            ```
            Product: SUP-5621 (Lumbar Cushion)
            Current Return Rate: 14.2%
            Category Average: 10.0%
            Standard Deviation: 1.8%
            Z-Score: 2.33

            SPC Signal: ⚠️ WARNING (2σ above mean)

            Interpretation:
            - This return rate is unusual (only 1% chance it's random)
            - Not quite "critical" yet, but definitely abnormal
            - Could indicate emerging process issue

            ACTION:
            1. Pull recent return data (last 30 days) - is trend worsening?
            2. Check if specific batch/lot affected (manufacturing date codes)
            3. Interview warehouse - any consistent complaint patterns?
            4. If still at warning level next screening: open quality case
            ```
            """)

        # BULK OPERATIONS
        with st.expander("### 5️⃣ Bulk Operations (Muda = Waste Elimination)"):
            st.markdown("""
            **What it is:** Generate vendor emails and investigation plans for multiple products at once

            **Official Term:** *Muda Elimination* (Remove Wasteful Work)
            **Layman's Term:** *Do 20 Tasks in 2 Minutes*

            #### Time Savings:
            - **Manual Way:** Write individual email per product → 5-10 min each × 15 products = 75-150 minutes
            - **Bulk Way:** Select all 15 products, click "Generate" → 30 seconds total
            - **Saved:** ~2 hours per screening session

            #### What You Can Bulk-Generate:
            1. **Vendor CAPA Request Emails** - Formal requests for corrective action
            2. **RCA Request Emails** - Ask vendor for root cause analysis
            3. **Inspection Notice Emails** - Alert vendor of upcoming inspection
            4. **Investigation Plans** - Full project plans with timelines & tasks

            #### Actionable Result Example:
            ```
            Flagged Products: 12 products need vendor follow-up

            Select All 12 → Choose "CAPA Request" → Click Generate → Done!

            Output:
            - 12 professional emails ready to send
            - Each customized with product-specific details:
              * SKU and product name
              * Return rate and units affected
              * Specific defects described
              * Required response timeline
            - Export to CSV for your records
            - Preview in tool before sending

            ACTION: Review first 2-3 emails to ensure tone/details correct,
            then send all 12 to respective vendors. Follow up in 5 business days.
            ```
            """)

        # DEEP DIVE ANALYSIS
        with st.expander("### 6️⃣ Deep Dive Analysis with Document Upload (Genchi Genbutsu)"):
            st.markdown("""
            **What it is:** AI analyzes your product documentation to understand root causes

            **Official Term:** *Genchi Genbutsu* (Go & See for Yourself)
            **Layman's Term:** *Get the Full Story Before Deciding*

            #### Documents You Can Upload:
            - **Product Manual**: Understand intended use vs actual use
            - **Amazon Listing**: Compare marketed features to complaints
            - **IFU (Instructions for Use)**: Check if returns are due to unclear instructions
            - **Technical Specs**: Identify spec deviations causing failures

            #### AI Analysis Provides:
            1. **Risk Level**: Low/Medium/High/Critical assessment
            2. **Recommended Investigation Method**: 5 Whys, Fishbone, Formal RCA, etc.
            3. **Intended Use Questions**: Critical questions about how customers use the product
            4. **Key Focus Areas**: Specific aspects to investigate
            5. **Immediate Actions**: What to do right now

            #### Actionable Result Example:
            ```
            Product: MOB-2847 (Premium Knee Scooter with Basket)
            Documents Uploaded: Product manual, Amazon listing

            AI Deep Dive Results:

            🔴 Risk Level: HIGH

            🎯 Recommended Method: Fishbone Diagram (Ishikawa)
            Rationale: Multiple failure modes reported (wheels, brakes, basket)
            suggests systemic design or manufacturing issue requiring multi-factor analysis.

            ❓ Critical Intended Use Questions:
            1. Are customers using scooter on rough outdoor terrain vs smooth indoor floors?
               → Manual specifies "indoor use only" but Amazon photos show outdoor settings
            2. Is basket being overloaded beyond 10 lb weight limit?
               → Listing doesn't clearly state weight restriction
            3. Are wheels failing due to user weight or terrain conditions?
               → 3 different wheel failure modes reported

            🔍 Key Investigation Areas:
            - Wheel assembly torque specs (may be undertightened at factory)
            - Basket weight limit communication (add to listing and manual)
            - Terrain usage expectations (clarify indoor vs outdoor capability)
            - Component supplier quality (wheels sourced from 2 different vendors)

            ⚡ Immediate Actions:
            1. Add prominent "Indoor Use Only" warning to Amazon listing TODAY
            2. Request vendor to verify wheel assembly torque on next production run
            3. Test scooter on various terrains to establish actual outdoor capability
            4. If outdoor use is acceptable, update specs; if not, add explicit warnings

            ACTION: Start Fishbone diagram mapping all contributing factors.
            Update listing immediately (low-cost high-impact fix). Schedule vendor meeting.
            ```
            """)

        st.divider()

        # Common Workflows
        st.markdown("## 🔄 Common Workflows (PDCA Cycle = Plan-Do-Check-Act)")

        workflow_tab1, workflow_tab2, workflow_tab3 = st.tabs([
            "📅 Weekly Routine Screening",
            "🚨 Emergency Response",
            "📊 Monthly Strategic Review"
        ])

        with workflow_tab1:
            st.markdown("""
            ### Weekly Routine Screening (PLAN-DO-CHECK-ACT)

            **PLAN (Monday Morning - 15 minutes):**
            1. Upload last week's return data (Pro mode)
            2. Use "Standard (SOP Defaults)" threshold profile
            3. Run statistical screening with Auto mode

            **DO (Monday - Wednesday):**
            4. Review flagged products (Risk Score > 60)
            5. For High/Critical items: Run Deep Dive Analysis
            6. Generate bulk vendor emails for all flagged items
            7. Open quality cases in Odoo for Critical items

            **CHECK (Thursday):**
            8. Review SPC signals - any "Warning" or "Critical" flags?
            9. Compare to last week - are same products still flagged?
            10. Check vendor responses (5 days have passed)

            **ACT (Friday):**
            11. Update threshold profile if patterns emerge
            12. Export results to team tracker (Yokoten - share knowledge)
            13. Document lessons learned
            14. Adjust processes based on findings

            **Time Required:** ~2-3 hours total per week (vs 8-10 hours manually)
            """)

        with workflow_tab2:
            st.markdown("""
            ### Emergency Response (Critical Safety Issue)

            **Immediate (Within 1 Hour):**
            1. Enter product in Lite mode
            2. Check "Safety Concern" flag
            3. Run Deep Dive Analysis with all available documents
            4. AI will assess risk level

            **If Risk = Critical:**
            5. Generate Critical Investigation Plan (Smartsheet export)
            6. Notify management immediately
            7. Open critical quality case in Odoo

            **Next 24 Hours:**
            8. Execute Phase 1 of investigation plan (immediate actions)
            9. Quarantine inventory
            10. Assess regulatory reporting requirements
            11. Generate vendor CAPA request (mark URGENT)

            **Ongoing:**
            12. Follow investigation plan timeline
            13. Update team tracker daily
            14. Document all findings in quality case

            **Example:** Battery compartment loose on electric wheelchair
            → Safety risk (fall hazard) → Immediate quarantine → Regulatory notification → Vendor emergency CAPA
            """)

        with workflow_tab3:
            st.markdown("""
            ### Monthly Strategic Review (Hansei = Reflection)

            **Preparation (Last Week of Month):**
            1. Upload entire month's return data
            2. Run ANOVA/MANOVA to identify statistical differences
            3. Generate trend charts (if available)

            **Analysis (First Week of New Month):**
            4. Review which categories consistently exceed thresholds
            5. Identify repeat offender products (flagged 3+ times)
            6. Calculate financial impact of returns by category
            7. Check effectiveness of previous month's actions

            **Strategic Adjustments:**
            8. Update threshold profiles for next quarter if needed
            9. Adjust vendor quality agreements based on performance
            10. Propose process changes to prevent recurring issues
            11. Share findings with leadership (Yokoten)

            **Kaizen Improvements:**
            12. What worked well this month? → Make it standard practice
            13. What didn't work? → Adjust process
            14. What new patterns emerged? → Add to watch list
            15. How can we prevent issues vs just detecting them? (Poka-Yoke)

            **Output:** Executive summary showing:
            - Month's total return rate vs target
            - Category performance trends
            - Cost of quality (return costs)
            - Improvement initiatives status
            - Next month's priorities
            """)

        st.divider()

        # FAQs
        st.markdown("## ❓ Frequently Asked Questions")

        faq_col1, faq_col2 = st.columns(2)

        with faq_col1:
            with st.expander("**Q: What's a good return rate threshold?**"):
                st.markdown("""
                **A:** Depends on your product category:
                - B2B: 2.5% (most stable customers)
                - INS (Insoles): 7% (fit/comfort issues common)
                - RHB (Rehab): 7.5%
                - LVA (Living Aids): 9.5%
                - MOB (Mobility): 10% (complex products)
                - CSH (Cushions): 10.5%
                - SUP (Support): 11%

                **Start with these (SOP Defaults), then adjust based on your data.**
                """)

            with st.expander("**Q: When should I investigate vs just monitor?**"):
                st.markdown("""
                **Investigate NOW if:**
                - Risk Score > 80 (Critical)
                - Safety concern flagged
                - SPC signal = "Critical" (3σ)
                - Return rate > 25% (absolute cap)
                - Multiple complaints citing same failure mode

                **Monitor closely (investigate next week) if:**
                - Risk Score 60-80 (High)
                - SPC signal = "Warning" (2σ)
                - Return rate 20% above category average
                - Trending worse for 2+ screening cycles

                **Just monitor (routine) if:**
                - Risk Score < 60
                - SPC signal = "Normal" or "Watch"
                - Within threshold limits
                - Stable trend
                """)

        with faq_col2:
            with st.expander("**Q: How do I know which statistical test to use?**"):
                st.markdown("""
                **A:** Use "Auto (AI Recommended)" - AI picks the right test.

                **If you want to choose manually:**
                - Comparing return rates across categories? → ANOVA
                - Multiple metrics at once (rate + cost + sales)? → MANOVA
                - Messy data with outliers? → Kruskal-Wallis
                - Just want summary stats? → Descriptive Only

                **Don't overthink it - Auto mode works great!**
                """)

            with st.expander("**Q: What do I do with screening results?**"):
                st.markdown("""
                **A:** Follow the PDCA cycle:

                **1. Prioritize (Plan):**
                - Sort by Risk Score (highest first)
                - Focus on Critical and High items

                **2. Investigate (Do):**
                - Use Deep Dive Analysis for context
                - Generate vendor emails / investigation plans
                - Open quality cases in Odoo

                **3. Verify (Check):**
                - Re-screen in 30 days
                - Check if return rate improved
                - Validate corrective actions worked

                **4. Standardize (Act):**
                - Export to team tracker (share knowledge)
                - Update SOPs if process changes made
                - Adjust threshold profiles for next cycle
                """)

        st.divider()

        # Action Button
        st.success("""
        ✅ **Ready to Start?** Close this guide and begin with Step 1: Enter your name and source of flag above.

        💡 **Tip:** Keep this guide open in a second browser tab for reference as you work!
        """)


def render_quality_screening_tab():
    """Render the Quality Case Screening tab - focused on screening tools"""

    # Enhanced Header with TQM Philosophy
    st.markdown("### 🧪 Quality Case Screening")
    st.markdown("**TQM Methodology:** *Kaizen* (改善 = Continuous Improvement) | *Jidoka* (自働化 = Smart Automation) | *Genchi Genbutsu* (現地現物 = Go & See)")
    st.caption("AI-powered quality screening compliant with ISO 13485, FDA 21 CFR 820, EU MDR, UK MDR")

    # Comprehensive User Guide
    render_comprehensive_user_guide()

    # --- SCREENING SESSION INFO (Who, When, Source) ---
    with st.expander("👤 Screening Session Info", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.session_state.screened_by = st.text_input(
                "Screened By*",
                value=st.session_state.screened_by,
                placeholder="Enter your name",
                help="Your name - included in exports for accountability tracking"
            )
        
        with col2:
            source_idx = SOURCE_OF_FLAG_OPTIONS.index(st.session_state.source_of_flag) \
                if st.session_state.source_of_flag in SOURCE_OF_FLAG_OPTIONS else 8  # Default to Routine Screening
            st.session_state.source_of_flag = st.selectbox(
                "Source of Flag*",
                options=SOURCE_OF_FLAG_OPTIONS,
                index=source_idx,
                help="How did this product/issue come to your attention?"
            )
        
        with col3:
            if st.session_state.source_of_flag == "Other (specify)":
                st.session_state.source_of_flag_other = st.text_input(
                    "Specify Source",
                    value=st.session_state.source_of_flag_other,
                    placeholder="Enter custom source"
                )
            else:
                st.session_state.screening_date = st.date_input(
                    "Screening Date",
                    value=datetime.now()
                ).strftime('%Y-%m-%d')
    
    st.divider()
    
    # --- MODE SELECTION ---
    col1, col2, col3 = st.columns([2, 2, 2])
    
    with col1:
        mode = st.radio(
            "Screening Mode",
            ["🎯 Quick Case Eval", "Lite (1-5 Products)", "Pro (Mass Analysis)"],
            horizontal=False,
            help="Quick Case Eval: 1-3 products with SOP comparison & case qualification. Lite: Manual entry. Pro: Batch analysis."
        )
        if "Quick" in mode:
            st.session_state.qc_mode = "QuickEval"
        elif "Lite" in mode:
            st.session_state.qc_mode = "Lite"
        else:
            st.session_state.qc_mode = "Pro"
    
    with col2:
        # AI Provider selection - OpenAI default
        ai_provider = st.selectbox(
            "AI Provider",
            options=list(AI_PROVIDER_OPTIONS.keys()),
            index=0,  # OpenAI default
            help="Select AI provider. OpenAI is default. Claude available for additional review."
        )
        st.session_state.ai_provider = AI_PROVIDER_OPTIONS[ai_provider]
    
    with col3:
        # Threshold profile selection
        profile_options = list(st.session_state.threshold_profiles.keys())
        profile = st.selectbox(
            "Threshold Profile",
            options=profile_options + ["➕ Create New Profile"],
            index=profile_options.index(st.session_state.active_profile) 
                  if st.session_state.active_profile in profile_options else 0,
            help="Select return rate thresholds to screen against"
        )
        
        if profile != "➕ Create New Profile":
            st.session_state.active_profile = profile
    
    st.divider()
    
    # --- THRESHOLD PROFILE MANAGEMENT ---
    render_threshold_manager(profile)
    
    # --- SIDEBAR: AI CHAT + CONFIG ---
    with st.sidebar:
        st.markdown("---")
        render_ai_chat_panel()

        st.markdown("---")
        st.markdown("### 🧪 Screening Config")
        
        # Custom threshold file upload
        st.markdown("#### Upload Threshold File")
        threshold_file = st.file_uploader(
            "Upload threshold CSV",
            type=['csv', 'xlsx'],
            help="Upload file with Category and Return Rate Threshold columns",
            key="threshold_upload"
        )
        
        if threshold_file:
            try:
                if threshold_file.name.endswith('.csv'):
                    threshold_df = pd.read_csv(threshold_file)
                else:
                    threshold_df = pd.read_excel(threshold_file)
                st.session_state.user_threshold_data = threshold_df
                st.success(f"Loaded {len(threshold_df)} threshold rules")
            except Exception as e:
                st.error(f"Error loading thresholds: {e}")
        
        # Processing log
        with st.expander("📜 Processing Log", expanded=False):
            if st.session_state.processing_log:
                log_text = "\n".join(st.session_state.processing_log[-20:])
                st.code(log_text, language="")
            else:
                st.caption("No logs yet")
            
            if st.button("Clear Log", key="clear_log"):
                st.session_state.processing_log = []
    
    # --- MAIN CONTENT ---

    if st.session_state.qc_mode == "QuickEval":
        render_quick_eval_mode()
    elif st.session_state.qc_mode == "Lite":
        render_lite_mode()
    else:
        render_pro_mode()

    # --- RESULTS DISPLAY ---
    if st.session_state.qc_results_df is not None:
        render_screening_results()


def render_threshold_manager(selected_profile):
    """Render threshold profile viewer/editor"""
    
    with st.expander("📊 Threshold Profile Manager", expanded=(selected_profile == "➕ Create New Profile")):
        
        if selected_profile == "➕ Create New Profile":
            # CREATE NEW PROFILE
            st.markdown("#### Create New Threshold Profile")
            
            new_profile_name = st.text_input(
                "Profile Name",
                placeholder="e.g., Q1 Strict Review, Post-Holiday Cleanup",
                key="new_profile_name"
            )
            
            st.markdown("**Set Return Rate Thresholds by Category**")
            st.caption("Enter the maximum acceptable return rate (%) for each category. Products exceeding these will be flagged.")
            
            # Base profile to start from
            base_profile = st.selectbox(
                "Start from existing profile",
                options=list(st.session_state.threshold_profiles.keys()),
                key="base_profile"
            )
            base_thresholds = st.session_state.threshold_profiles[base_profile].copy()
            categories = list(DEFAULT_CATEGORY_THRESHOLDS.keys())

            # Apply adjustment action if set (before widgets are created)
            if '_adjustment_action' in st.session_state:
                action = st.session_state['_adjustment_action']
                # Delete all widget keys first
                for cat in categories:
                    if f"thresh_{cat}" in st.session_state:
                        del st.session_state[f"thresh_{cat}"]
                # Modify base_thresholds which will be used for widget values
                for cat in categories:
                    current_val = base_thresholds.get(cat, 0.10) * 100
                    if action == 'tighten':
                        base_thresholds[cat] = (current_val * 0.8) / 100
                    elif action == 'loosen':
                        base_thresholds[cat] = (current_val * 1.2) / 100
                    elif action == 'reset':
                        base_thresholds[cat] = DEFAULT_CATEGORY_THRESHOLDS.get(cat, 0.10)
                del st.session_state['_adjustment_action']

            # Threshold inputs in columns
            new_thresholds = {}
            cols = st.columns(4)

            for idx, cat in enumerate(categories):
                with cols[idx % 4]:
                    default_val = base_thresholds.get(cat, 0.10) * 100
                    new_val = st.number_input(
                        f"{cat}",
                        min_value=0.0,
                        max_value=100.0,
                        value=float(default_val),
                        step=0.5,
                        format="%.1f",
                        key=f"thresh_{cat}",
                        help=f"SOP default: {DEFAULT_CATEGORY_THRESHOLDS.get(cat, 0.10)*100:.1f}%"
                    )
                    new_thresholds[cat] = new_val / 100  # Convert back to decimal

            # Quick adjustment buttons
            st.markdown("**Quick Adjustments**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if st.button("Tighten All (-20%)", key="tighten"):
                    # Set flag to apply adjustment on next render
                    st.session_state['_adjustment_action'] = 'tighten'
                    st.rerun()
            with col2:
                if st.button("Loosen All (+20%)", key="loosen"):
                    # Set flag to apply adjustment on next render
                    st.session_state['_adjustment_action'] = 'loosen'
                    st.rerun()
            with col3:
                if st.button("Reset to SOP", key="reset_sop"):
                    # Set flag to apply adjustment on next render
                    st.session_state['_adjustment_action'] = 'reset'
                    st.rerun()
            with col4:
                pass  # spacer
            
            # Save button
            st.markdown("---")
            if st.button("💾 Save New Profile", type="primary", disabled=not new_profile_name):
                if new_profile_name in st.session_state.threshold_profiles:
                    st.error(f"Profile '{new_profile_name}' already exists. Choose a different name.")
                else:
                    st.session_state.threshold_profiles[new_profile_name] = new_thresholds
                    st.session_state.active_profile = new_profile_name
                    st.success(f"✅ Created profile: {new_profile_name}")
                    st.rerun()
        
        else:
            # VIEW/EDIT EXISTING PROFILE
            st.markdown(f"#### Profile: {selected_profile}")
            
            current_thresholds = st.session_state.threshold_profiles.get(
                selected_profile, DEFAULT_CATEGORY_THRESHOLDS
            )
            
            # Show current thresholds in a nice format
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Table view
                threshold_data = []
                for cat, thresh in current_thresholds.items():
                    sop_default = DEFAULT_CATEGORY_THRESHOLDS.get(cat, 0.10)
                    diff = ((thresh - sop_default) / sop_default) * 100 if sop_default > 0 else 0
                    threshold_data.append({
                        'Category': cat,
                        'Threshold': f"{thresh*100:.1f}%",
                        'SOP Default': f"{sop_default*100:.1f}%",
                        'Diff': f"{diff:+.0f}%" if diff != 0 else "—"
                    })
                
                st.dataframe(
                    pd.DataFrame(threshold_data),
                    hide_index=True,
                    use_container_width=True
                )
            
            with col2:
                st.markdown("**Profile Actions**")
                
                if selected_profile not in ['Standard Review']:  # Protect default
                    if st.button("🗑️ Delete Profile", key="delete_profile"):
                        del st.session_state.threshold_profiles[selected_profile]
                        st.session_state.active_profile = 'Standard Review'
                        st.rerun()
                
                if st.button("📋 Duplicate Profile", key="dup_profile"):
                    new_name = f"{selected_profile} (Copy)"
                    st.session_state.threshold_profiles[new_name] = current_thresholds.copy()
                    st.success(f"Created: {new_name}")
                    st.rerun()
            
            # Edit mode
            st.markdown("---")
            edit_mode = st.checkbox("✏️ Edit this profile", key="edit_mode")
            
            if edit_mode and selected_profile != 'Standard Review':
                st.markdown("**Edit Return Rate Thresholds (%)**")
                
                edited_thresholds = {}
                cols = st.columns(4)
                categories = list(current_thresholds.keys())
                
                for idx, cat in enumerate(categories):
                    with cols[idx % 4]:
                        current_val = current_thresholds.get(cat, 0.10) * 100
                        new_val = st.number_input(
                            f"{cat}",
                            min_value=0.0,
                            max_value=100.0,
                            value=float(current_val),
                            step=0.5,
                            format="%.1f",
                            key=f"edit_{cat}"
                        )
                        edited_thresholds[cat] = new_val / 100
                
                if st.button("💾 Save Changes", type="primary"):
                    st.session_state.threshold_profiles[selected_profile] = edited_thresholds
                    st.success("✅ Profile updated")
                    st.rerun()
            
            elif edit_mode and selected_profile == 'Standard Review':
                st.warning("⚠️ Cannot edit Standard Review (SOP defaults). Duplicate it first to customize.")
            
            # Explanation
            st.markdown("---")
            st.markdown("""
            **How Thresholds Work:**
            - Products with return rates **above** their category threshold are flagged
            - Lower thresholds = stricter screening (more products flagged)
            - Higher thresholds = looser screening (fewer products flagged)
            
            **Example:** If MOB threshold is 10%, a product with 12% return rate gets flagged.
            """)


def render_ai_chat_panel():
    """Render AI chat panel in sidebar for guidance and discussion"""
    
    st.markdown("### 💬 AI Assistant")
    
    # Initialize chat history if needed
    if 'ai_guidance_chat' not in st.session_state:
        st.session_state.ai_guidance_chat = []
    
    # Chat container with scrollable history
    chat_container = st.container()
    
    with chat_container:
        # Show chat history (last 10 messages)
        for msg in st.session_state.ai_guidance_chat[-10:]:
            if msg['role'] == 'user':
                st.markdown(f"**You:** {msg['content']}")
            else:
                st.markdown(f"**AI:** {msg['content']}")
            st.markdown("---")
    
    # Quick action buttons
    st.markdown("**Quick Questions:**")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("❓ How to set thresholds?", key="q1", use_container_width=True):
            _add_ai_response("threshold_help")
    
    with col2:
        if st.button("📊 Explain my results", key="q2", use_container_width=True):
            _add_ai_response("results_help")
    
    col3, col4 = st.columns(2)
    
    with col3:
        if st.button("🎯 What should I screen?", key="q3", use_container_width=True):
            _add_ai_response("screening_help")
    
    with col4:
        if st.button("⚠️ Risk score meaning?", key="q4", use_container_width=True):
            _add_ai_response("risk_help")
    
    # Free text input
    st.markdown("**Ask anything:**")
    user_question = st.text_input(
        "Type your question",
        placeholder="e.g., Should I flag products under 5% return rate?",
        key="ai_chat_input",
        label_visibility="collapsed"
    )
    
    if st.button("Send", key="send_chat", use_container_width=True):
        if user_question.strip():
            _process_ai_chat(user_question)


def _add_ai_response(response_type: str):
    """Add predefined AI responses for quick questions"""
    
    responses = {
        "threshold_help": {
            "question": "How should I set thresholds?",
            "answer": """**Setting Return Rate Thresholds:**

1. **Start with SOP defaults** - These are based on historical category performance

2. **Adjust based on your goals:**
   - **Stricter (lower %)**: Use before peak seasons, new product launches, or after quality issues
   - **Looser (higher %)**: For stable products or when focusing resources on high-priority items

3. **Category guidance:**
   - **MOB (Mobility)**: 10% typical - complex products, higher returns expected
   - **SUP (Support)**: 11% typical - sizing issues common
   - **B2B**: 2.5% typical - professional buyers, lower returns

4. **Pro tip**: Create an "Aggressive Q4" profile at 80% of standard thresholds before holiday season."""
        },
        "results_help": {
            "question": "Explain my results",
            "answer": """**Understanding Your Screening Results:**

1. **Risk Score (0-100):**
   - 70-100: Immediate attention needed
   - 40-69: Open quality case for investigation
   - 20-39: Monitor closely
   - 0-19: No action required

2. **Action Column:**
   - "Immediate Escalation" = Safety/high-value issue
   - "Open Quality Case" = Exceeds thresholds
   - "Monitor" = Elevated but within tolerance

3. **SPC Signal:**
   - "Critical (>3σ)" = Statistical outlier
   - "Warning (>2σ)" = Trending concerning
   - "Normal" = Within expected variation

4. **Key tip**: Sort by Risk Score descending to prioritize your investigation queue."""
        },
        "screening_help": {
            "question": "What should I screen?",
            "answer": """**What to Screen:**

1. **Regular screening (monthly):**
   - All products with >$50K monthly sales
   - Products launched in last 90 days
   - Any product with customer complaints

2. **Priority screening:**
   - Products mentioned in safety reports
   - High landed cost items (>$100)
   - Products with sudden return rate changes

3. **Data to gather:**
   - Units sold and returned (required)
   - Top complaint reasons (highly recommended)
   - Landed cost (helps prioritization)
   - Any customer feedback verbatims

4. **Pro tip**: Use the trailing 12-month Amazon data as your baseline, then screen specific products in Lite mode for deeper investigation."""
        },
        "risk_help": {
            "question": "What does Risk Score mean?",
            "answer": """**Risk Score Breakdown (0-100):**

The composite score combines 5 factors:

1. **Statistical Deviation (25%)** - How far above category threshold
   - >50% above = 25 pts
   - >25% above = 20 pts
   - Any amount above = 15 pts

2. **Financial Impact (25%)** - Landed cost consideration
   - >$150 cost = 25 pts
   - >$100 cost = 18 pts
   - >$50 cost = 10 pts

3. **Safety Severity (30%)** - Largest weight
   - Safety risk flagged = 30 pts
   - Critical complaint = 25 pts
   - Major defect = 15 pts

4. **Trend Direction (10%)** - Is it getting worse?
   - Rapidly deteriorating = 10 pts
   - Stable = 3 pts
   - Improving = 0 pts

5. **Complaint Volume (10%)** - Frequency of issues
   - High complaint rate = 10 pts
   - Moderate = 6 pts
   - Low = 0 pts"""
        }
    }
    
    if response_type in responses:
        resp = responses[response_type]
        st.session_state.ai_guidance_chat.append({
            'role': 'user',
            'content': resp['question']
        })
        st.session_state.ai_guidance_chat.append({
            'role': 'assistant',
            'content': resp['answer']
        })
        st.rerun()


def _process_ai_chat(user_question: str):
    """Process free-form user question with AI"""
    
    # Add user message
    st.session_state.ai_guidance_chat.append({
        'role': 'user',
        'content': user_question
    })
    
    # Build context-aware prompt
    context_parts = []
    
    # Add current results context if available
    if st.session_state.qc_results_df is not None:
        df = st.session_state.qc_results_df
        context_parts.append(f"Current screening has {len(df)} products.")
        high_risk = len(df[df['Risk_Score'] >= 70]) if 'Risk_Score' in df.columns else 0
        context_parts.append(f"High risk items: {high_risk}")
    
    # Add active profile context
    context_parts.append(f"Active threshold profile: {st.session_state.active_profile}")
    
    context = " ".join(context_parts)
    
    system_prompt = f"""You are an AI-powered medical device quality management expert assistant integrated into the Vive Health Quality Suite.

**About This Application:**
This app extensively uses AI for:
- AI-powered return categorization (Tab 1) using OpenAI/Claude
- Multilingual vendor email generation with translation
- Product similarity matching and benchmarking
- Deep dive quality analysis and root cause recommendations
- Fuzzy product matching across 231 historical products
- Semantic analysis of customer complaints

**Your Role:**
Help users understand how to use the AI features, interpret results, set thresholds, and make quality decisions.
Be concise but helpful. Use bullet points for clarity.

Current context: {context}

When asked about AI capabilities, ALWAYS explain that this is an AI-powered quality management system with extensive ML/LLM integration, NOT a rules-based system."""
    
    try:
        analyzer = get_ai_analyzer()
        if analyzer:
            response = analyzer.generate_text(
                user_question,
                system_prompt,
                mode='chat'
            )
            
            if response:
                st.session_state.ai_guidance_chat.append({
                    'role': 'assistant',
                    'content': response
                })
            else:
                st.session_state.ai_guidance_chat.append({
                    'role': 'assistant',
                    'content': "I couldn't generate a response. Please check your API connection."
                })
        else:
            st.session_state.ai_guidance_chat.append({
                'role': 'assistant',
                'content': "AI not available. Please check API configuration."
            })
    except Exception as e:
        st.session_state.ai_guidance_chat.append({
            'role': 'assistant',
            'content': f"Error: {str(e)}"
        })
    
    st.rerun()


def render_quick_eval_mode():
    """
    Render Quick Case Evaluation Mode - designed for rapid case qualification with 1-3 products.

    Features:
    - Compare products against SOP thresholds (not against each other)
    - Clear pass/fail indicators
    - Speech-to-text summary input
    - AI-powered case qualification determination
    - Demo-ready UI for presentations
    """

    # Initialize session state for quick eval
    if 'quick_eval_products' not in st.session_state:
        st.session_state.quick_eval_products = []
    if 'quick_eval_summary' not in st.session_state:
        st.session_state.quick_eval_summary = ""
    if 'quick_eval_results' not in st.session_state:
        st.session_state.quick_eval_results = None

    # Header with clear purpose
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; color: white;">
        <h3 style="margin: 0; color: white;">🎯 Quick Case Evaluation</h3>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.95;">
            Rapid SOP compliance check for 1-3 products. Compare against category thresholds,
            not against each other. Add context summary for AI-powered case qualification.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Get active threshold profile
    active_profile_name = st.session_state.active_profile
    active_thresholds = st.session_state.threshold_profiles.get(
        active_profile_name, DEFAULT_CATEGORY_THRESHOLDS
    )

    # Display SOP Requirements prominently
    st.markdown("### 📋 Current SOP Requirements")
    st.caption(f"**Active Profile:** {active_profile_name}")

    # Show thresholds in a clean, scannable format
    threshold_cols = st.columns(4)
    threshold_items = list(active_thresholds.items())

    for idx, (category, threshold) in enumerate(threshold_items):
        with threshold_cols[idx % 4]:
            st.metric(
                label=category,
                value=f"{threshold*100:.1f}%",
                help=f"Maximum acceptable return rate for {category} category"
            )

    st.markdown("---")

    # Date Range Section
    st.markdown("### 📅 Evaluation Period")
    date_col1, date_col2 = st.columns(2)

    with date_col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=90),
            key="qe_start_date",
            help="Beginning of analysis period"
        )

    with date_col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            key="qe_end_date",
            help="End of analysis period"
        )

    if start_date >= end_date:
        st.error("⚠️ Start date must be before end date")
        return

    days_in_period = (end_date - start_date).days
    st.caption(f"Analysis Period: {days_in_period} days ({start_date.strftime('%B %d, %Y')} - {end_date.strftime('%B %d, %Y')})")

    st.markdown("---")

    # Product Entry Section
    st.markdown("### 📦 Product Information (1-3 Products)")

    num_products = st.number_input(
        "Number of products to evaluate",
        min_value=1,
        max_value=3,
        value=min(len(st.session_state.quick_eval_products), 3) or 1,
        help="Enter 1-3 products for SOP compliance evaluation"
    )

    # Initialize products list to match count
    while len(st.session_state.quick_eval_products) < num_products:
        st.session_state.quick_eval_products.append({
            'sku': '',
            'product_name': '',
            'category': 'MOB',
            'units_sold': 0,
            'units_returned': 0,
            'return_rate': 0.0,
            'complaint_summary': ''
        })

    # Trim if too many
    st.session_state.quick_eval_products = st.session_state.quick_eval_products[:num_products]

    # Product input forms
    for i in range(num_products):
        with st.expander(f"📦 Product {i+1}", expanded=True):
            col1, col2, col3 = st.columns([2, 2, 1])

            with col1:
                sku = st.text_input(
                    "SKU / Product ID",
                    value=st.session_state.quick_eval_products[i]['sku'],
                    key=f"qe_sku_{i}",
                    placeholder="e.g., MOB-2847"
                )
                st.session_state.quick_eval_products[i]['sku'] = sku

            with col2:
                product_name = st.text_input(
                    "Product Name",
                    value=st.session_state.quick_eval_products[i]['product_name'],
                    key=f"qe_name_{i}",
                    placeholder="e.g., Premium Knee Scooter"
                )
                st.session_state.quick_eval_products[i]['product_name'] = product_name

            with col3:
                category = st.selectbox(
                    "Category",
                    options=list(active_thresholds.keys()),
                    index=list(active_thresholds.keys()).index(
                        st.session_state.quick_eval_products[i]['category']
                    ) if st.session_state.quick_eval_products[i]['category'] in active_thresholds else 0,
                    key=f"qe_cat_{i}"
                )
                st.session_state.quick_eval_products[i]['category'] = category

            col4, col5, col6 = st.columns(3)

            with col4:
                units_sold = st.number_input(
                    "Units Sold",
                    min_value=0,
                    value=st.session_state.quick_eval_products[i]['units_sold'],
                    key=f"qe_sold_{i}"
                )
                st.session_state.quick_eval_products[i]['units_sold'] = units_sold

            with col5:
                units_returned = st.number_input(
                    "Units Returned",
                    min_value=0,
                    value=st.session_state.quick_eval_products[i]['units_returned'],
                    key=f"qe_returned_{i}"
                )
                st.session_state.quick_eval_products[i]['units_returned'] = units_returned

            with col6:
                # Calculate return rate
                if units_sold > 0:
                    return_rate = (units_returned / units_sold) * 100
                else:
                    return_rate = 0.0

                st.session_state.quick_eval_products[i]['return_rate'] = return_rate

                # Get threshold for this category
                category_threshold = active_thresholds.get(category, 0.10) * 100

                # Show return rate with status indicator
                exceeds_threshold = return_rate > category_threshold

                # Calculate both percentage points and percent change
                percentage_points_diff = return_rate - category_threshold
                if category_threshold > 0:
                    percent_change = ((return_rate - category_threshold) / category_threshold) * 100
                else:
                    percent_change = 0

                st.metric(
                    label="Return Rate",
                    value=f"{return_rate:.2f}%",
                    delta=f"{percentage_points_diff:+.2f} pts",
                    delta_color="inverse"
                )

                # Show both metrics
                if exceeds_threshold:
                    st.markdown(f"🔴 **EXCEEDS SOP**")
                    st.caption(f"+{percentage_points_diff:.2f} pts | +{percent_change:.1f}% change")
                else:
                    st.markdown(f"🟢 **Within SOP**")
                    st.caption(f"{percentage_points_diff:+.2f} pts | {percent_change:+.1f}% change")

            # Complaint summary for this product
            complaint_summary = st.text_area(
                "Complaint / Issue Summary",
                value=st.session_state.quick_eval_products[i]['complaint_summary'],
                key=f"qe_complaint_{i}",
                placeholder="Brief description of the quality issue or customer complaint...",
                height=80
            )
            st.session_state.quick_eval_products[i]['complaint_summary'] = complaint_summary

    st.markdown("---")

    # Overall Situation Summary Section
    st.markdown("### 📝 Overall Situation Summary")
    st.caption("Provide context about these products: Why are they being evaluated? What's the business impact? Any urgency factors?")

    col1, col2 = st.columns([4, 1])

    with col1:
        summary = st.text_area(
            "Summary / Context",
            value=st.session_state.quick_eval_summary,
            key="qe_summary_input",
            placeholder="Example: These products have elevated returns from Q4 batch. Customer complaints mention packaging damage during shipping. Need to determine if this warrants a formal CAPA investigation...",
            height=120,
            help="Provide background, urgency, business context, or any other relevant information"
        )
        st.session_state.quick_eval_summary = summary

    with col2:
        st.markdown("**🎤 Voice Input**")
        st.caption("*Dictate your summary*")

        # Initialize audio transcription session state
        if 'audio_transcription' not in st.session_state:
            st.session_state.audio_transcription = ""

        # Simple voice recording button
        if st.button("🎙️ Start Recording", key="voice_record", use_container_width=True):
            st.info("🎤 **Voice Recording Instructions:**\n\n"
                   "1. Click the button below\n"
                   "2. Speak clearly into your microphone\n"
                   "3. Click 'Stop' when finished\n"
                   "4. Text will appear in the summary box")

        # Alternative: Quick voice input
        voice_input = st.text_input(
            "Or type to speak:",
            key="quick_voice",
            placeholder="Type here, or use browser's voice input...",
            help="Many browsers support voice-to-text when you click the microphone icon in this field"
        )

        if voice_input and st.button("➕ Add to Summary", use_container_width=True):
            current_summary = st.session_state.quick_eval_summary
            if current_summary:
                st.session_state.quick_eval_summary = current_summary + " " + voice_input
            else:
                st.session_state.quick_eval_summary = voice_input
            st.rerun()

    st.markdown("---")

    # Evaluate Button
    if st.button("🚀 Evaluate Case Qualification", type="primary", use_container_width=True):
        # Validate input
        valid_products = [p for p in st.session_state.quick_eval_products
                         if p['sku'] and p['product_name'] and p['units_sold'] > 0]

        if not valid_products:
            st.error("⚠️ Please enter at least one complete product (SKU, name, and units sold required)")
            return

        # Perform evaluation
        with st.spinner("🔍 Analyzing products against SOP requirements..."):
            results = {
                'products': [],
                'overall_assessment': '',
                'meets_case_criteria': False,
                'recommended_actions': [],
                'severity': 'Low'
            }

            total_exceeds = 0
            total_products = len(valid_products)
            highest_excess = 0

            for product in valid_products:
                category_threshold = active_thresholds.get(product['category'], 0.10) * 100
                exceeds = product['return_rate'] > category_threshold
                excess_amount = product['return_rate'] - category_threshold

                if exceeds:
                    total_exceeds += 1
                    highest_excess = max(highest_excess, excess_amount)

                product_result = {
                    'sku': product['sku'],
                    'product_name': product['product_name'],
                    'category': product['category'],
                    'return_rate': product['return_rate'],
                    'threshold': category_threshold,
                    'exceeds_sop': exceeds,
                    'excess_amount': excess_amount,
                    'status': '🔴 EXCEEDS SOP' if exceeds else '🟢 Within SOP',
                    'units_returned': product['units_returned'],
                    'units_sold': product['units_sold']
                }

                results['products'].append(product_result)

            # Determine case qualification
            if total_exceeds == 0:
                results['meets_case_criteria'] = False
                results['severity'] = 'None'
                results['overall_assessment'] = f"""
                **✅ NO CASE REQUIRED**

                All {total_products} product(s) are within SOP thresholds for their respective categories.
                No formal case investigation is warranted at this time.
                """
                results['recommended_actions'] = [
                    "Continue monitoring return rates",
                    "Document findings for record-keeping",
                    "No immediate action required"
                ]

            elif total_exceeds == total_products and highest_excess > 5.0:
                # All products exceed, and significantly
                results['meets_case_criteria'] = True
                results['severity'] = 'High' if highest_excess > 10.0 else 'Medium'
                results['overall_assessment'] = f"""
                **🔴 CASE INVESTIGATION RECOMMENDED**

                All {total_products} product(s) exceed SOP thresholds, with maximum excess of {highest_excess:.1f}%.
                This pattern indicates a systemic quality issue requiring formal investigation.
                """
                results['recommended_actions'] = [
                    "Open formal CAPA investigation immediately",
                    "Quarantine affected inventory pending investigation",
                    "Notify relevant stakeholders (Quality Manager, Production, Suppliers)",
                    "Conduct root cause analysis using 5 Whys or Fishbone",
                    "Review production records for affected batches"
                ]

            elif total_exceeds >= 1:
                # Some exceed
                results['meets_case_criteria'] = True
                results['severity'] = 'Medium' if highest_excess > 5.0 else 'Low'
                results['overall_assessment'] = f"""
                **🟡 CASE INVESTIGATION WARRANTED**

                {total_exceeds} of {total_products} product(s) exceed SOP thresholds (max excess: {highest_excess:.1f}%).
                Recommend opening a quality case to investigate the elevated return rates.
                """
                results['recommended_actions'] = [
                    "Open quality case for affected product(s)",
                    "Review customer complaints and return reasons",
                    "Investigate potential common causes across products",
                    "Consider containment actions if pattern is identified",
                    "Monitor closely for trend development"
                ]

            # AI Enhancement (if available - always run for comprehensive analysis)
            if AI_AVAILABLE:
                try:
                    ai_analyzer = EnhancedAIAnalyzer(st.session_state.ai_provider)

                    # Build comprehensive context for AI including product names, complaints, and date range
                    products_detail = "\n".join([
                        f"- **{p['sku']} - {p['product_name']}** ({p['category']})\n"
                        f"  Return Rate: {p['return_rate']:.2f}% (SOP Threshold: {p['threshold']:.1f}%)\n"
                        f"  Units: {p['units_returned']:,} returned of {p['units_sold']:,} sold\n"
                        f"  Status: {'🔴 EXCEEDS SOP' if p['exceeds_sop'] else '🟢 Within SOP'}\n"
                        f"  Complaints: {valid_products[results['products'].index(p)].get('complaint_summary', 'None provided')}"
                        for p in results['products']
                    ])

                    ai_prompt = f"""
                    You are a medical device quality expert conducting a formal case evaluation.

                    EVALUATION PERIOD: {start_date.strftime('%B %d, %Y')} to {end_date.strftime('%B %d, %Y')} ({days_in_period} days)

                    PRODUCTS EVALUATED:
                    {products_detail}

                    USER CONTEXT:
                    {st.session_state.quick_eval_summary if st.session_state.quick_eval_summary.strip() else 'No additional context provided'}

                    Based on the PRODUCT NAMES (identify failure modes based on product type), COMPLAINT PATTERNS, RETURN DATA, and DATE RANGE, provide:

                    1. **PRODUCT-SPECIFIC RISK ASSESSMENT:**
                       - Analyze each product name to identify likely failure modes (e.g., "Knee Scooter" → mobility/stability issues, "Rollator Walker" → wheel/brake failures)
                       - Cross-reference with complaint summaries to validate hypotheses

                    2. **SEVERITY & URGENCY:**
                       - Overall severity (Low/Medium/High/Critical)
                       - Time sensitivity based on trend (getting worse/stable/improving?)

                    3. **ROOT CAUSE HYPOTHESES:**
                       - Based on product types and complaint patterns
                       - Consider: design flaws, manufacturing defects, supplier issues, user errors

                    4. **REGULATORY IMPLICATIONS:**
                       - FDA reporting requirements (MDR/MAUDE)?
                       - ISO 13485 / EU MDR concerns?
                       - Recall risk assessment

                    5. **SPECIFIC RECOMMENDED ACTIONS:**
                       - Numbered list with PRODUCT-SPECIFIC steps
                       - Investigation methods (5 Whys, Fishbone, FMEA, 8D)
                       - Timeline for each action

                    Keep response detailed but actionable (400 words max). Be specific to the actual products, not generic advice.
                    """

                    ai_response = ai_analyzer.analyze_with_retry(ai_prompt, max_retries=2)

                    if ai_response:
                        results['ai_analysis'] = ai_response
                        # Extract severity from AI if mentioned
                        if 'Critical' in ai_response:
                            results['severity'] = 'Critical'
                        elif 'High' in ai_response and results['severity'] not in ['Critical']:
                            results['severity'] = 'High'

                except Exception as e:
                    logger.error(f"AI analysis failed: {e}")
                    results['ai_analysis'] = None
            else:
                results['ai_analysis'] = None

            st.session_state.quick_eval_results = results

    # Display Results
    if st.session_state.quick_eval_results:
        results = st.session_state.quick_eval_results

        st.markdown("---")
        st.markdown("## 📊 Evaluation Results")

        # Overall Status Card
        severity_colors = {
            'None': '#10b981',
            'Low': '#fbbf24',
            'Medium': '#f59e0b',
            'High': '#ef4444',
            'Critical': '#dc2626'
        }

        severity_color = severity_colors.get(results['severity'], '#6b7280')

        st.markdown(f"""
        <div style="background: {severity_color}; color: white; border-radius: 12px;
                    padding: 1.5rem; margin-bottom: 1.5rem;">
            <h3 style="margin: 0; color: white;">
                {'✅ NO CASE REQUIRED' if not results['meets_case_criteria'] else '⚠️ CASE INVESTIGATION RECOMMENDED'}
            </h3>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">
                <strong>Severity:</strong> {results['severity']}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Assessment
        st.markdown(results['overall_assessment'])

        # Product-by-Product Results
        st.markdown("### 📦 Product-Level Analysis")

        for idx, product in enumerate(results['products']):
            status_icon = "🔴" if product['exceeds_sop'] else "🟢"

            with st.expander(f"{status_icon} {product['sku']} - {product['product_name']}",
                           expanded=product['exceeds_sop']):
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Category", product['category'])

                with col2:
                    st.metric("Return Rate", f"{product['return_rate']:.2f}%")

                with col3:
                    st.metric("SOP Threshold", f"{product['threshold']:.1f}%")

                with col4:
                    st.metric(
                        "vs Threshold",
                        f"{product['excess_amount']:+.2f}%",
                        delta_color="inverse"
                    )

                st.markdown(f"**Status:** {product['status']}")
                st.markdown(f"**Units:** {product['units_returned']:,} returned of {product['units_sold']:,} sold")

        # Recommended Actions
        st.markdown("### 🎯 Recommended Actions")

        for idx, action in enumerate(results['recommended_actions'], 1):
            st.markdown(f"{idx}. {action}")

        # AI Analysis (if available)
        if 'ai_analysis' in results and results['ai_analysis']:
            st.markdown("### 🤖 AI Expert Analysis")

            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        border-radius: 12px; padding: 1.5rem; color: white;">
                {results['ai_analysis'].replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)

        # Export Options
        st.markdown("---")
        st.markdown("### 📤 Export")

        col1, col2, col3 = st.columns(3)

        with col1:
            # Generate comprehensive summary report matching display
            report_lines = [
                "=" * 80,
                "QUICK CASE EVALUATION REPORT",
                "=" * 80,
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Evaluation Period: {start_date.strftime('%B %d, %Y')} - {end_date.strftime('%B %d, %Y')} ({days_in_period} days)",
                f"Threshold Profile: {active_profile_name}",
                f"Screened By: {st.session_state.screened_by}",
                "",
                "=" * 80,
                "OVERALL ASSESSMENT",
                "=" * 80,
                f"Case Investigation Required: {'YES' if results['meets_case_criteria'] else 'NO'}",
                f"Severity Level: {results['severity']}",
                "",
                results['overall_assessment'].strip(),
                "",
                "=" * 80,
                "PRODUCT-BY-PRODUCT ANALYSIS",
                "=" * 80,
            ]

            for idx, p in enumerate(results['products'], 1):
                # Calculate percent change
                if p['threshold'] > 0:
                    percent_change = ((p['return_rate'] - p['threshold']) / p['threshold']) * 100
                else:
                    percent_change = 0

                report_lines.extend([
                    f"",
                    f"Product {idx}: {p['sku']} - {p['product_name']}",
                    f"-" * 80,
                    f"Category: {p['category']}",
                    f"Status: {p['status']}",
                    f"",
                    f"Return Rate: {p['return_rate']:.2f}%",
                    f"SOP Threshold: {p['threshold']:.1f}%",
                    f"Difference: {p['excess_amount']:+.2f} percentage points ({percent_change:+.1f}% change)",
                    f"",
                    f"Units Sold: {p['units_sold']:,}",
                    f"Units Returned: {p['units_returned']:,}",
                ])

                # Add complaint summary if available
                for prod in valid_products:
                    if prod['sku'] == p['sku'] and prod.get('complaint_summary'):
                        report_lines.extend([
                            f"",
                            f"Complaint Summary:",
                            f"{prod['complaint_summary']}"
                        ])
                        break

            report_lines.extend([
                "",
                "=" * 80,
                "RECOMMENDED ACTIONS",
                "=" * 80,
            ])

            for idx, action in enumerate(results['recommended_actions'], 1):
                report_lines.append(f"{idx}. {action}")

            if st.session_state.quick_eval_summary:
                report_lines.extend([
                    "",
                    "=" * 80,
                    "SITUATIONAL CONTEXT",
                    "=" * 80,
                    st.session_state.quick_eval_summary
                ])

            # Add AI analysis if available
            if 'ai_analysis' in results and results['ai_analysis']:
                report_lines.extend([
                    "",
                    "=" * 80,
                    "AI EXPERT ANALYSIS",
                    "=" * 80,
                    results['ai_analysis']
                ])

            report_lines.extend([
                "",
                "=" * 80,
                "END OF REPORT",
                "=" * 80
            ])

            report_text = "\n".join(report_lines)

            st.download_button(
                "📄 Download Report (TXT)",
                report_text,
                file_name=f"case_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )

        with col2:
            # CSV export
            csv_data = []
            for p in results['products']:
                csv_data.append({
                    'SKU': p['sku'],
                    'Product Name': p['product_name'],
                    'Category': p['category'],
                    'Return Rate (%)': f"{p['return_rate']:.2f}",
                    'SOP Threshold (%)': f"{p['threshold']:.1f}",
                    'Exceeds SOP': 'YES' if p['exceeds_sop'] else 'NO',
                    'Excess Amount (%)': f"{p['excess_amount']:+.2f}",
                    'Units Returned': p['units_returned'],
                    'Units Sold': p['units_sold']
                })

            csv_df = pd.DataFrame(csv_data)
            csv_buffer = io.StringIO()
            csv_df.to_csv(csv_buffer, index=False)

            st.download_button(
                "📊 Download Data (CSV)",
                csv_buffer.getvalue(),
                file_name=f"case_eval_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col3:
            if st.button("🔄 New Evaluation", use_container_width=True):
                st.session_state.quick_eval_results = None
                st.session_state.quick_eval_products = []
                st.session_state.quick_eval_summary = ""
                st.rerun()


def render_lite_mode():
    """Render Lite mode - manual entry for 1-5 products with dynamic forms"""
    
    st.info("ℹ️ **Lite Mode**: Enter product details manually for quick screening (1-5 products)")
    
    # Initialize lite entries in session state if not exists
    if 'lite_entries' not in st.session_state or not st.session_state.lite_entries:
        st.session_state.lite_entries = [{'id': 0}]  # Start with one empty entry
    
    # Date range selection (applies to all products)
    st.markdown("#### 📅 Date Range (applies to all products)")
    col1, col2 = st.columns(2)
    with col1:
        date_range = st.selectbox(
            "Data Date Range",
            ["Last 30 days", "Last 60 days", "Last 90 days", "Last 180 days", "Last 365 days", "Custom Range"],
            index=0,
            key="lite_date_range"
        )
    with col2:
        if date_range == "Custom Range":
            date_start = st.date_input("Start Date", datetime.now() - timedelta(days=30), key="lite_date_start")
            date_end = st.date_input("End Date", datetime.now(), key="lite_date_end")
        else:
            days = int(re.search(r'\d+', date_range).group())
            date_start = datetime.now() - timedelta(days=days)
            date_end = datetime.now()
    
    st.markdown("---")
    
    # Product count controls
    col_add, col_remove, col_count = st.columns([1, 1, 2])
    with col_add:
        if st.button("➕ Add Product", disabled=len(st.session_state.lite_entries) >= 5):
            new_id = max([e['id'] for e in st.session_state.lite_entries]) + 1
            st.session_state.lite_entries.append({'id': new_id})
            st.rerun()
    with col_remove:
        if st.button("➖ Remove Last", disabled=len(st.session_state.lite_entries) <= 1):
            st.session_state.lite_entries.pop()
            st.rerun()
    with col_count:
        st.markdown(f"**Products to screen:** {len(st.session_state.lite_entries)} of 5 max")
    
    st.markdown("---")
    
    # Collect all product data
    all_entries = []
    all_valid = True
    
    # Create expandable sections for each product
    for idx, entry in enumerate(st.session_state.lite_entries):
        entry_id = entry['id']
        with st.expander(f"📦 Product {idx + 1}", expanded=(idx == 0 or idx == len(st.session_state.lite_entries) - 1)):
            
            # Required fields
            st.markdown("**Required Fields**")
            col1, col2, col3 = st.columns(3)
            with col1:
                product_name = st.text_input("Product Name*", placeholder="e.g., Knee Walker", key=f"name_{entry_id}")
            with col2:
                product_sku = st.text_input("Product SKU*", placeholder="e.g., MOB1027", key=f"sku_{entry_id}")
            with col3:
                category = st.selectbox(
                    "Category*",
                    options=['Select...'] + list(DEFAULT_CATEGORY_THRESHOLDS.keys()),
                    index=0,
                    key=f"cat_{entry_id}"
                )
            
            col4, col5, col6 = st.columns(3)
            with col4:
                units_sold = st.number_input("Units Sold*", min_value=1, value=100, key=f"sold_{entry_id}")
            with col5:
                units_returned = st.number_input("Units Returned*", min_value=0, value=0, key=f"ret_{entry_id}")
            with col6:
                return_rate_calc = (units_returned / units_sold * 100) if units_sold > 0 else 0
                st.metric("Return Rate", f"{return_rate_calc:.1f}%")
            
            # Complaint reasons
            complaint_reasons = st.text_input(
                "Top Return Reasons (comma-separated)*",
                placeholder="e.g., Uncomfortable, Too small, Defective wheel",
                key=f"complaints_{entry_id}"
            )
            
            # Optional fields in a collapsed section
            with st.container():
                show_optional = st.checkbox("Show optional fields", key=f"show_opt_{entry_id}")
                
                if show_optional:
                    st.markdown("**Optional Fields**")
                    col7, col8, col9 = st.columns(3)
                    with col7:
                        unit_cost = st.number_input("Landed Cost ($)", min_value=0.0, value=0.0, step=0.01, key=f"cost_{entry_id}")
                    with col8:
                        primary_channel = st.selectbox(
                            "Primary Channel",
                            ["Select...", "Amazon", "B2B", "Website", "Other"],
                            key=f"channel_{entry_id}"
                        )
                    with col9:
                        packaging_method = st.selectbox(
                            "Packaging",
                            ["Select...", "Standard Box", "Poly Bag", "Custom", "Other"],
                            key=f"pack_{entry_id}"
                        )
                    
                    col10, col11 = st.columns(2)
                    with col10:
                        b2b_feedback = st.text_area("B2B Feedback", placeholder="Optional B2B feedback...", height=68, key=f"b2b_fb_{entry_id}")
                    with col11:
                        amazon_feedback = st.text_area("Amazon Feedback", placeholder="Optional Amazon feedback...", height=68, key=f"amz_fb_{entry_id}")
                    
                    manual_context = st.text_area(
                        "Additional Context",
                        placeholder="Any relevant background info, manual excerpts, known issues...",
                        height=68,
                        key=f"context_{entry_id}"
                    )
                else:
                    unit_cost = 0.0
                    primary_channel = "Select..."
                    packaging_method = "Select..."
                    b2b_feedback = ""
                    amazon_feedback = ""
                    manual_context = ""
            
            # Safety flags
            col_safe, col_new = st.columns(2)
            with col_safe:
                safety_risk = st.checkbox("⚠️ Safety Risk?", key=f"safety_{entry_id}")
            with col_new:
                is_new_product = st.checkbox("🆕 New Product?", key=f"new_{entry_id}")
            
            # Validate this entry
            entry_valid = product_name and product_sku and category != 'Select...'
            if not entry_valid:
                all_valid = False
            
            # Build entry dict
            all_entries.append({
                'SKU': product_sku,
                'Name': product_name,
                'Category': category if category != 'Select...' else '',
                'Sold': units_sold,
                'Returned': units_returned,
                'Return_Rate': units_returned / units_sold if units_sold > 0 else 0,
                'Landed Cost': unit_cost,
                'Complaint_Text': complaint_reasons,
                'Manual_Context': manual_context,
                'Safety Risk': 'Yes' if safety_risk else 'No',
                'Is_New_Product': is_new_product,
                'Primary_Channel': primary_channel if primary_channel != 'Select...' else '',
                'B2B_Feedback': b2b_feedback,
                'Amazon_Feedback': amazon_feedback,
                'Date_Range': f"{date_start} to {date_end}",
                '_valid': entry_valid
            })
    
    st.markdown("---")
    
    # Summary before processing
    valid_count = sum(1 for e in all_entries if e.get('_valid', False))
    if valid_count < len(all_entries):
        st.warning(f"⚠️ {len(all_entries) - valid_count} product(s) missing required fields (Name, SKU, Category)")
    
    # Process button
    col_btn, col_clear = st.columns([3, 1])
    with col_btn:
        if st.button("🔍 Run AI Screening", type="primary", use_container_width=True, disabled=valid_count == 0):
            # Filter to valid entries only
            valid_entries = [e for e in all_entries if e.get('_valid', False)]
            
            # Remove internal _valid flag
            for e in valid_entries:
                e.pop('_valid', None)
            
            # Create DataFrame and process
            df_input = pd.DataFrame(valid_entries)
            process_screening(df_input)
    
    with col_clear:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.lite_entries = [{'id': 0}]
            st.rerun()


def render_pro_mode():
    """Render Pro mode - mass upload analysis"""
    
    st.info("🚀 **Pro Mode**: Upload CSV/Excel for mass analysis (up to 500+ products)")
    
    # Template download section
    st.markdown("#### 📋 Download Template")
    col_template, col_example = st.columns(2)
    
    with col_template:
        # Create blank template
        template_df = pd.DataFrame(columns=[
            'SKU', 'Name', 'Category', 'Sold', 'Returned', 'Landed Cost',
            'Complaint_Text', 'Safety Risk', 'Primary_Channel',
            'B2B_Feedback', 'Amazon_Feedback', 'Manual_Context'
        ])
        
        # Add one example row with instructions
        template_df.loc[0] = [
            'MOB1027', 'Knee Walker Deluxe', 'MOB', 1000, 120, 85.00,
            'Wheel squeaks, uncomfortable padding, hard to fold',
            'No', 'Amazon',
            '', '', ''
        ]
        
        template_csv = template_df.to_csv(index=False)
        st.download_button(
            "📥 Download Blank Template",
            template_csv,
            file_name="quality_screening_template.csv",
            mime="text/csv",
            help="Download a blank CSV template with all columns"
        )
    
    with col_example:
        # Create example with multiple rows
        example_df = pd.DataFrame([
            {
                'SKU': 'MOB1027', 'Name': 'Knee Walker Deluxe', 'Category': 'MOB',
                'Sold': 1000, 'Returned': 120, 'Landed Cost': 85.00,
                'Complaint_Text': 'Wheel squeaks, uncomfortable padding',
                'Safety Risk': 'No', 'Primary_Channel': 'Amazon',
                'B2B_Feedback': '', 'Amazon_Feedback': '3 star avg', 'Manual_Context': ''
            },
            {
                'SKU': 'SUP1036', 'Name': 'Post Op Shoe', 'Category': 'SUP',
                'Sold': 500, 'Returned': 125, 'Landed Cost': 12.00,
                'Complaint_Text': 'Wrong size, poor fit, runs small',
                'Safety Risk': 'No', 'Primary_Channel': 'Amazon',
                'B2B_Feedback': '', 'Amazon_Feedback': '', 'Manual_Context': ''
            },
            {
                'SKU': 'LVA1004', 'Name': 'Alternating Pressure Mattress', 'Category': 'LVA',
                'Sold': 800, 'Returned': 150, 'Landed Cost': 145.00,
                'Complaint_Text': 'Pump failure, air leak, motor noise',
                'Safety Risk': 'Yes', 'Primary_Channel': 'B2B',
                'B2B_Feedback': 'Multiple facilities reporting pump issues',
                'Amazon_Feedback': '', 'Manual_Context': 'Known batch issue from Q3'
            },
            {
                'SKU': 'CSH1006', 'Name': 'Knee Walker Pad Cover', 'Category': 'CSH',
                'Sold': 2000, 'Returned': 180, 'Landed Cost': 8.50,
                'Complaint_Text': 'Velcro wears out, material pilling',
                'Safety Risk': 'No', 'Primary_Channel': 'Amazon',
                'B2B_Feedback': '', 'Amazon_Feedback': '', 'Manual_Context': ''
            },
            {
                'SKU': 'RHB1022', 'Name': 'Shoulder Pulley', 'Category': 'RHB',
                'Sold': 3000, 'Returned': 90, 'Landed Cost': 6.00,
                'Complaint_Text': 'Rope fraying, door bracket weak',
                'Safety Risk': 'No', 'Primary_Channel': 'Amazon',
                'B2B_Feedback': '', 'Amazon_Feedback': '', 'Manual_Context': ''
            }
        ])
        
        example_csv = example_df.to_csv(index=False)

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                "📥 Download Simple Example (5 products)",
                example_csv,
                file_name="quality_screening_example.csv",
                mime="text/csv",
                help="Basic example with 5 sample products"
            )

        with col_dl2:
            # Load advanced demo data if available
            try:
                demo_path = os.path.join(os.path.dirname(__file__), 'demo_quality_screening_data_advanced.csv')
                if os.path.exists(demo_path):
                    demo_df = pd.read_csv(demo_path)
                    demo_csv = demo_df.to_csv(index=False)
                    st.download_button(
                        "🚀 Download Advanced Demo (70 products)",
                        demo_csv,
                        file_name="demo_quality_screening_advanced.csv",
                        mime="text/csv",
                        help="⭐ Realistic demo with 70 products from actual product catalog - shows AI screening, multilingual emails, fuzzy matching",
                        type="primary"
                    )
                else:
                    st.info("Advanced demo not found")
            except Exception as e:
                logger.error(f"Could not load advanced demo: {e}")
    
    # Column reference
    with st.expander("📖 Column Reference Guide"):
        st.markdown("""
        | Column | Required | Description | Example |
        |--------|----------|-------------|---------|
        | `SKU` | ✅ Yes | Product SKU/identifier | MOB1027 |
        | `Name` | No | Product name | Knee Walker Deluxe |
        | `Category` | ✅ Yes | Product category code | MOB, SUP, LVA, CSH, RHB, INS |
        | `Sold` | ✅ Yes | Units sold in period | 1000 |
        | `Returned` | ✅ Yes | Units returned | 120 |
        | `Landed Cost` | No | Unit cost in USD | 85.00 |
        | `Complaint_Text` | No | Top complaints (comma-separated) | Wheel squeaks, hard to fold |
        | `Safety Risk` | No | Safety concern flag | Yes / No |
        | `Primary_Channel` | No | Main sales channel | Amazon, B2B, Website |
        | `B2B_Feedback` | No | B2B customer feedback | Facilities reporting issues |
        | `Amazon_Feedback` | No | Amazon reviews/feedback | 3 star average |
        | `Manual_Context` | No | Additional context | Known batch issue |
        
        **Category Codes:** B2B, INS, RHB, LVA, MOB, CSH, SUP
        
        **Notes:**
        - Return rate is auto-calculated from Sold/Returned
        - If your file has different column names, the system will attempt to map them
        - Blank optional fields are fine
        """)

    st.markdown("---")

    # VoC Analysis Import Section
    with st.expander("📊 VoC Analysis Import (Period-over-Period Comparison)", expanded=False):
        st.markdown("""
        **Import VoC Analysis data with automatic period-over-period trend analysis**

        This feature analyzes your VoC Analysis workbook to identify:
        - 📈 Sales trends (Increasing/Decreasing/Stable from previous period)
        - 📉 Return rate changes (compared to L30D)
        - 🚨 Amazon return rate fee threshold violations (2026 policy)
        - ⚠️ Return badge visibility impact
        - 💰 Estimated fee risk from excess returns
        """)

        voc_file = st.file_uploader(
            "Upload VoC Analysis.xlsx",
            type=['xlsx', 'xls'],
            help="Upload your VoC Analysis workbook with dated sheets",
            key="voc_upload"
        )

        if voc_file:
            try:
                # Get available periods
                available_periods = VoCAnalysisService.get_available_periods(voc_file)

                if not available_periods:
                    st.error("No dated sheets found in workbook. Expected sheets like 'January_2026_01162026'")
                else:
                    st.success(f"Found {len(available_periods)} dated periods")

                    col_current, col_previous = st.columns(2)

                    with col_current:
                        period_names = [display for _, display in available_periods]
                        current_period_idx = st.selectbox(
                            "Current Period",
                            range(len(available_periods)),
                            format_func=lambda i: available_periods[i][1],
                            index=0,
                            help="Select the most recent period to analyze"
                        )
                        current_sheet = available_periods[current_period_idx][0]

                    with col_previous:
                        previous_options = ["(No comparison)"] + period_names
                        previous_idx = st.selectbox(
                            "Compare to Period",
                            range(len(previous_options)),
                            format_func=lambda i: previous_options[i],
                            index=1 if len(previous_options) > 1 else 0,
                            help="Select previous period for trend analysis"
                        )

                        if previous_idx == 0:
                            previous_sheet = None
                        else:
                            previous_sheet = available_periods[previous_idx - 1][0]

                    if st.button("🔄 Import & Analyze VoC Data", type="primary"):
                        with st.spinner("Analyzing VoC data with period comparison..."):
                            # Parse workbook with period comparison
                            trend_analyses = VoCAnalysisService.parse_voc_workbook(
                                voc_file,
                                current_sheet,
                                previous_sheet
                            )

                            # Generate summary
                            summary = VoCAnalysisService.generate_period_comparison_summary(trend_analyses)

                            # Display summary
                            st.markdown("#### 📊 Period Comparison Summary")
                            col1, col2, col3, col4 = st.columns(4)

                            with col1:
                                st.metric("Products Analyzed", summary['total_products'])
                                st.metric("Sales Increasing",
                                         summary['sales_trends']['increasing'],
                                         delta=f"{summary['sales_trends']['increasing']}")
                                st.metric("Sales Decreasing",
                                         summary['sales_trends']['decreasing'],
                                         delta=f"-{summary['sales_trends']['decreasing']}",
                                         delta_color="inverse")

                            with col2:
                                st.metric("Returns Improving",
                                         summary['return_trends']['improving'],
                                         delta=f"{summary['return_trends']['improving']}")
                                st.metric("Returns Worsening",
                                         summary['return_trends']['worsening'],
                                         delta=f"-{summary['return_trends']['worsening']}",
                                         delta_color="inverse")

                            with col3:
                                st.metric("Above Amazon Threshold",
                                         summary['amazon_thresholds']['above_threshold'])
                                st.metric("Return Badge Displayed",
                                         summary['badges']['with_badge'])

                            with col4:
                                st.metric("Fee Risk Units",
                                         summary['amazon_thresholds']['fee_risk_units'])
                                st.metric("Est. Fee Impact",
                                         f"${summary['amazon_thresholds']['estimated_fees']:.2f}",
                                         delta_color="off")
                                st.metric("Action Required",
                                         summary['actions']['action_required'])

                            # Convert to screening DataFrame
                            df_voc = VoCAnalysisService.convert_to_screening_dataframe(trend_analyses)

                            st.markdown("#### 🔍 VoC Data Preview (Top 10 by Risk)")
                            st.dataframe(
                                df_voc.head(10)[[
                                    'SKU', 'Name', 'Sold', 'Return_Rate',
                                    'Sales_Trend', 'Return_Trend',
                                    'Above_Threshold', 'Risk_Flags'
                                ]],
                                use_container_width=True
                            )

                            # Store in session for screening
                            st.session_state.voc_import_data = df_voc

                            st.success(f"✅ VoC data imported successfully! {len(df_voc)} products ready for screening.")
                            st.info("👇 Click 'Run Screening' below to analyze this data with AI-powered quality screening")

            except Exception as e:
                st.error(f"Error processing VoC file: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

        # Check if VoC data is loaded and ready
        if 'voc_import_data' in st.session_state and st.session_state.voc_import_data is not None:
            st.success(f"✅ VoC data ready: {len(st.session_state.voc_import_data)} products loaded")

            if st.button("📊 Use VoC Data for Screening", type="primary"):
                df_input = st.session_state.voc_import_data.copy()
                process_screening(df_input, 'Pro', include_claude=False)
                st.rerun()

    st.markdown("---")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload Product Data",
        type=['csv', 'xlsx', 'xls'],
        help="Upload file with columns: SKU, Category, Sold, Returned (required). Optional: Name, Landed Cost, Complaint_Text, etc.",
        key="qc_pro_upload"
    )
    
    if uploaded_file:
        try:
            # Load file
            if uploaded_file.name.endswith('.csv'):
                df_input = pd.read_csv(uploaded_file)
            else:
                df_input = pd.read_excel(uploaded_file)
            
            log_process(f"Loaded file: {uploaded_file.name} ({len(df_input)} rows)")
            
            # Validate
            validation = DataValidation.validate_upload(df_input)
            
            # Show validation report
            with st.expander("📋 Data Validation Report", expanded=True):
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Rows", validation['total_rows'])
                col2.metric("Columns Found", len(validation['found_cols']))
                col3.metric("Validation", "✅ Passed" if validation['valid'] else "❌ Issues Found")
                
                if validation['warnings']:
                    for warning in validation['warnings']:
                        st.warning(warning)
                
                if validation['numeric_issues']:
                    st.markdown("**Numeric Issues:**")
                    for issue in validation['numeric_issues']:
                        st.caption(f"- {issue['column']}: {issue['issue']}")
                
                if validation['column_mapping']:
                    st.markdown("**Column Mapping:**")
                    mapping_df = pd.DataFrame([
                        {'Standard': k, 'Your Column': v}
                        for k, v in validation['column_mapping'].items()
                    ])
                    st.dataframe(mapping_df, hide_index=True)
            
            if not validation['valid']:
                st.error("Please fix the validation issues before proceeding.")
                return
            
            # Preview data
            st.markdown("#### Data Preview")
            st.dataframe(df_input.head(10), use_container_width=True)
            
            # Statistical analysis suggestion
            st.markdown("#### 📊 Statistical Analysis Options")
            
            # Prepare numeric columns for suggestion
            numeric_cols = []
            for col in ['Return_Rate', 'Landed Cost', 'Sold', 'Returned']:
                mapped_col = validation['column_mapping'].get(col)
                if mapped_col and mapped_col in df_input.columns:
                    numeric_cols.append(col)
            
            # Get AI suggestion
            suggestion = QualityStatistics.suggest_analysis_type(
                df_input.rename(columns={v: k for k, v in validation['column_mapping'].items()}),
                [c for c in numeric_cols if c not in ['Sold', 'Returned']]
            )
            st.session_state.statistical_suggestion = suggestion
            
            # Display AI recommendation prominently
            rec_col, alt_col = st.columns([2, 1])
            
            with rec_col:
                st.success(f"🤖 **AI Recommends: {suggestion['recommended']}**")
                st.caption(suggestion['reason'])
                
                if suggestion['warnings']:
                    for warning in suggestion['warnings']:
                        st.warning(warning)
            
            with alt_col:
                if suggestion['alternatives']:
                    st.markdown("**Alternatives:**")
                    for alt in suggestion['alternatives'][:2]:
                        st.caption(f"• {alt['test']}")
            
            st.markdown("---")
            
            # Analysis type selection with detailed explanations
            st.markdown("##### Choose Your Analysis Method")
            
            analysis_type = st.selectbox(
                "Statistical Test",
                options=list(STATISTICAL_ANALYSIS_OPTIONS.keys()),
                index=0,
                help="Select the statistical method to use. Auto uses AI recommendation."
            )
            
            # Show detailed explanation for selected analysis
            selected_info = STATISTICAL_ANALYSIS_OPTIONS[analysis_type]
            
            with st.expander(f"ℹ️ About: {analysis_type}", expanded=True):
                st.markdown(f"**What it does:** {selected_info['description']}")
                st.markdown(f"**When to use:** {selected_info['when_to_use']}")
                st.markdown(f"**Example:** _{selected_info['example']}_")
            
            # Additional options
            col1, col2 = st.columns(2)
            with col1:
                include_claude_review = st.checkbox(
                    "🔍 Request Claude AI Review",
                    help="Get additional cross-analysis from Claude (slower but more thorough)"
                )
            with col2:
                run_posthoc = st.checkbox(
                    "📈 Run Post-Hoc Tests",
                    value=True,
                    help="If results are significant, identify exactly which categories differ"
                )
            
            st.markdown("---")
            
            # Run analysis button
            if st.button("🚀 Run Full Screening Analysis", type="primary", use_container_width=True):
                # Rename columns to standard names
                df_renamed = df_input.rename(columns={v: k for k, v in validation['column_mapping'].items()})
                
                # Determine analysis type
                if analysis_type == "Auto (AI Recommended)":
                    actual_analysis = suggestion['recommended']
                else:
                    # Extract just the test name without description
                    actual_analysis = analysis_type.split(" (")[0]
                
                process_screening(df_renamed, analysis_type=actual_analysis, include_claude=include_claude_review)
        
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            log_process(f"File load error: {e}", 'error')


def process_screening(df: pd.DataFrame, analysis_type: str = "ANOVA", include_claude: bool = False):
    """Process screening analysis"""
    
    log_process("Starting Quality Case Screening Analysis")
    progress = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Data Preparation
        status_text.text("Step 1/6: Preparing data...")
        log_process("Preparing data...")
        
        # Parse numeric columns
        if 'Sold' in df.columns:
            df['Sold'] = parse_numeric(df['Sold'])
        else:
            df['Sold'] = 100  # Default
        
        if 'Returned' in df.columns:
            df['Returned'] = parse_numeric(df['Returned'])
        else:
            df['Returned'] = 0
        
        if 'Return_Rate' not in df.columns or df['Return_Rate'].isna().all():
            df['Return_Rate'] = df['Returned'] / df['Sold'].replace(0, 1)
        else:
            df['Return_Rate'] = parse_percentage(df['Return_Rate']).abs()
        
        if 'Landed Cost' in df.columns:
            df['Landed Cost'] = parse_numeric(df['Landed Cost'])
        else:
            df['Landed Cost'] = 0
        
        progress.progress(15)
        
        # Step 2: Get thresholds
        status_text.text("Step 2/6: Applying thresholds...")
        log_process("Applying category thresholds...")
        
        active_thresholds = st.session_state.threshold_profiles.get(
            st.session_state.active_profile, 
            DEFAULT_CATEGORY_THRESHOLDS
        )
        
        # Apply fuzzy matching if user threshold data available
        if st.session_state.user_threshold_data is not None:
            df['Category_Threshold'] = df.apply(
                lambda row: fuzzy_match_category(
                    str(row.get('Name', '')),
                    str(row.get('Category', '')),
                    st.session_state.user_threshold_data
                )[1],
                axis=1
            )
        else:
            df['Category_Threshold'] = df['Category'].apply(
                lambda x: active_thresholds.get(x, active_thresholds.get('All Others', 0.10))
            )
        
        progress.progress(30)
        
        # Step 3: Statistical Analysis
        status_text.text("Step 3/6: Running statistical analysis...")
        log_process(f"Running {analysis_type} analysis...")
        
        if analysis_type == "MANOVA" and len(df) > 10:
            metrics = ['Return_Rate']
            if 'Landed Cost' in df.columns and df['Landed Cost'].sum() > 0:
                metrics.append('Landed Cost')
            
            manova_result = QualityStatistics.perform_manova(df, 'Category', metrics)
            st.session_state.manova_result = manova_result
            log_process(f"MANOVA p-value: {manova_result.p_value:.4f}")
        
        elif analysis_type in ["ANOVA", "Auto"] and len(df) > 5:
            anova_result = QualityStatistics.perform_anova(df, 'Category', 'Return_Rate')
            st.session_state.anova_result = anova_result
            log_process(f"ANOVA F={anova_result.statistic:.2f}, p={anova_result.p_value:.4f}")
        
        elif analysis_type == "Kruskal-Wallis" and len(df) > 5:
            kw_result = QualityStatistics.perform_kruskal_wallis(df, 'Category', 'Return_Rate')
            st.session_state.anova_result = kw_result
            log_process(f"Kruskal-Wallis H={kw_result.statistic:.2f}, p={kw_result.p_value:.4f}")
        
        progress.progress(45)
        
        # Step 4: Calculate Risk Scores and Determine Actions
        status_text.text("Step 4/6: Calculating risk scores...")
        log_process("Calculating weighted risk scores...")
        
        results = []
        for idx, row in df.iterrows():
            # Risk score
            risk_score, risk_components = RiskScoring.calculate_risk_score(
                return_rate=row['Return_Rate'],
                category_threshold=row['Category_Threshold'],
                landed_cost=row.get('Landed Cost', 0),
                safety_risk=str(row.get('Safety Risk', '')).lower() in ['yes', 'true', '1'],
                complaint_count=len(str(row.get('Complaint_Text', '')).split(',')) if row.get('Complaint_Text') else 0,
                units_sold=row['Sold']
            )
            
            # SPC Signal
            cat_std = df[df['Category'] == row['Category']]['Return_Rate'].std()
            cat_mean = df[df['Category'] == row['Category']]['Return_Rate'].mean()
            spc_result = SPCAnalysis.detect_signal(row['Return_Rate'], cat_mean, cat_std if cat_std > 0 else 0.01)
            
            # Determine action
            action, triggers = ActionDetermination.determine_action(
                return_rate=row['Return_Rate'],
                category_threshold=row['Category_Threshold'],
                landed_cost=row.get('Landed Cost', 0),
                safety_risk=str(row.get('Safety Risk', '')).lower() in ['yes', 'true', '1'],
                is_new_product=row.get('Is_New_Product', False),
                complaint_count=len(str(row.get('Complaint_Text', '')).split(',')) if row.get('Complaint_Text') else 0,
                risk_score=risk_score
            )
            
            # Build result row with all data + screening metadata
            result_row = row.to_dict()
            
            # Get source of flag (handle "Other" case)
            source_flag = st.session_state.source_of_flag
            if source_flag == "Other (specify)":
                source_flag = st.session_state.source_of_flag_other or "Other"
            
            result_row.update({
                'Risk_Score': risk_score,
                'Risk_Components': json.dumps(risk_components),
                'SPC_Signal': spc_result.signal_type,
                'SPC_Z_Score': spc_result.z_score,
                'Action': action,
                'Triggers': '; '.join(triggers) if triggers else 'None',
                # Screening metadata for tracker
                'Screened_By': st.session_state.screened_by or 'Not specified',
                'Screening_Date': st.session_state.screening_date,
                'Source_of_Flag': source_flag,
                'Threshold_Profile': st.session_state.active_profile,
                # Blank columns for Google Sheets tracker
                'Current_Status': '',  # User fills in: Open, Investigating, Closed, etc.
                'Notes': ''  # User adds their own notes
            })
            results.append(result_row)
        
        results_df = pd.DataFrame(results)
        progress.progress(60)
        
        # Step 5: AI Analysis
        status_text.text("Step 5/6: Running AI analysis...")
        log_process("Running AI-powered analysis...")
        
        analyzer = get_ai_analyzer()
        
        # AI analysis for high-risk items
        high_risk_items = results_df[results_df['Risk_Score'] >= 50]
        
        if len(high_risk_items) > 0 and analyzer:
            ai_recommendations = []
            for idx, row in high_risk_items.head(10).iterrows():  # Limit to top 10
                prompt = f"""Analyze this medical device quality issue:
Product: {row.get('Name', row.get('SKU', 'Unknown'))} (SKU: {row.get('SKU', 'N/A')})
Category: {row.get('Category', 'Unknown')}
Return Rate: {row['Return_Rate']:.1%} (Category threshold: {row['Category_Threshold']:.1%})
Risk Score: {row['Risk_Score']:.0f}/100
Main Complaints: {row.get('Complaint_Text', 'N/A')}
Safety Concern: {row.get('Safety Risk', 'No')}

Based on ISO 13485 and FDA QSR requirements, provide:
1. Brief assessment (2-3 sentences)
2. Primary investigation area
3. Recommended immediate action"""
                
                system_prompt = "You are a medical device quality expert. Be concise and action-oriented."
                
                try:
                    recommendation = analyzer.generate_text(prompt, system_prompt, mode='chat')
                    ai_recommendations.append({
                        'SKU': row.get('SKU', 'Unknown'),
                        'AI_Recommendation': recommendation or "AI analysis unavailable"
                    })
                except Exception as e:
                    log_process(f"AI analysis error for {row.get('SKU')}: {e}", 'error')
                    ai_recommendations.append({
                        'SKU': row.get('SKU', 'Unknown'),
                        'AI_Recommendation': f"Error: {str(e)}"
                    })
            
            # Merge AI recommendations
            if ai_recommendations:
                ai_df = pd.DataFrame(ai_recommendations)
                results_df = results_df.merge(ai_df, on='SKU', how='left')
        
        progress.progress(80)
        
        # Step 6: Claude Review (if requested)
        if include_claude and st.session_state.ai_provider != AIProvider.CLAUDE:
            status_text.text("Step 6/6: Running Claude review...")
            log_process("Requesting Claude AI additional review...")
            
            try:
                claude_analyzer = EnhancedAIAnalyzer(AIProvider.CLAUDE, max_workers=3)
                
                # Get Claude's overall assessment
                summary_prompt = f"""Review this quality screening batch:
- Total products: {len(results_df)}
- Products requiring action: {len(results_df[results_df['Action'].str.contains('Escalat|Case')])}
- Highest risk score: {results_df['Risk_Score'].max():.0f}
- Categories with issues: {', '.join(results_df[results_df['Risk_Score'] > 50]['Category'].unique()[:5])}

Provide a brief executive summary and any patterns you notice."""
                
                claude_review = claude_analyzer.generate_text(
                    summary_prompt,
                    "You are a senior quality director reviewing screening results.",
                    mode='chat'
                )
                
                st.session_state.ai_chat_history.append({
                    'role': 'claude_review',
                    'content': claude_review
                })
                log_process("Claude review completed")
            except Exception as e:
                log_process(f"Claude review error: {e}", 'error')
        
        progress.progress(100)
        status_text.text("Analysis complete!")
        
        # Store results
        st.session_state.qc_results_df = results_df
        log_process(f"Analysis complete. {len(results_df)} products screened.")
        
        st.rerun()
        
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        log_process(f"Processing error: {e}", 'error')
        raise


def render_screening_results():
    """Render the screening results dashboard"""
    
    df = st.session_state.qc_results_df
    
    st.markdown("---")
    st.markdown("### 📊 Screening Results")
    
    # Summary Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total = len(df)
    escalations = len(df[df['Action'].str.contains('Escalat', na=False)])
    cases = len(df[df['Action'].str.contains('Case', na=False)])
    monitors = len(df[df['Action'].str.contains('Monitor', na=False)])
    
    col1.metric("Total Analyzed", total)
    col2.metric("Immediate Escalations", escalations, delta_color="inverse")
    col3.metric("Quality Cases", cases, delta_color="inverse")
    col4.metric("Monitor", monitors)
    
    # Statistical Results with Enhanced Tooltips
    if st.session_state.anova_result or st.session_state.manova_result:
        with st.expander("📈 Statistical Analysis Results (Click for Plain English Explanations)", expanded=True):
            if st.session_state.manova_result:
                result = st.session_state.manova_result
                st.markdown(f"**MANOVA Results** - {STATISTICAL_ANALYSIS_OPTIONS['MANOVA (Multivariate ANOVA)']['tooltip']}")
                col1, col2, col3 = st.columns(3)
                col1.metric("F-Statistic", f"{result.statistic:.3f}",
                           help=STATS_EXPLAINER['f_score'])
                col2.metric("p-value", f"{result.p_value:.4f}",
                           help=STATS_EXPLAINER['p_value'])
                col3.metric("Significant", "Yes ✓" if result.significant else "No",
                           help="p<0.05 means the differences are statistically significant (not due to random chance)")
                st.info(result.recommendation)

                # Plain English Summary
                with st.expander("🗣️ What does this mean in plain English?"):
                    significance_text = "ARE statistically significant" if result.significant else "are NOT statistically significant"
                    st.markdown(f"""
                    **Bottom Line:** Your product categories {significance_text} when looking at multiple metrics together.

                    - **F-Statistic ({result.statistic:.3f})**: {STATS_EXPLAINER['f_score']}
                    - **p-value ({result.p_value:.4f})**: {STATS_EXPLAINER['p_value']}
                    - **Wilks' Lambda**: {STATS_EXPLAINER['wilks_lambda']}

                    {"✅ **Action:** The differences are real. Focus on the categories with highest return rates." if result.significant else "⚠️ **Action:** Differences might be random variation. Monitor trends but don't overreact."}
                    """)

            elif st.session_state.anova_result:
                result = st.session_state.anova_result
                test_name = result.test_type
                is_anova = 'ANOVA' in test_name
                st.markdown(f"**{test_name} Results** - {STATISTICAL_ANALYSIS_OPTIONS.get(test_name, STATISTICAL_ANALYSIS_OPTIONS['ANOVA (Analysis of Variance)'])['tooltip']}")

                col1, col2, col3, col4 = st.columns(4)
                stat_name = "F-Statistic" if is_anova else "H-Statistic"
                stat_help = STATS_EXPLAINER['f_score'] if is_anova else STATS_EXPLAINER['h_statistic']

                col1.metric(stat_name, f"{result.statistic:.3f}", help=stat_help)
                col2.metric("p-value", f"{result.p_value:.4f}", help=STATS_EXPLAINER['p_value'])
                col3.metric("Effect Size (η²)", f"{result.effect_size:.3f}" if result.effect_size else "N/A",
                           help=STATS_EXPLAINER['effect_size'])
                col4.metric("Significant", "Yes ✓" if result.significant else "No",
                           help="p<0.05 = statistically significant difference between groups")

                st.info(result.recommendation)

                # Plain English Summary
                with st.expander("🗣️ What does this mean in plain English?"):
                    significance_text = "ARE statistically different" if result.significant else "are NOT statistically different"
                    effect_interp = "large" if result.effect_size and result.effect_size > 0.8 else \
                                   "medium" if result.effect_size and result.effect_size > 0.5 else \
                                   "small" if result.effect_size and result.effect_size > 0.2 else "negligible"

                    effect_size_str = f"{result.effect_size:.3f}" if result.effect_size else "N/A"
                    st.markdown(f"""
                    **Bottom Line:** Your product categories {significance_text} in their return rates.

                    - **{stat_name} ({result.statistic:.3f})**: {stat_help}
                    - **p-value ({result.p_value:.4f})**: {STATS_EXPLAINER['p_value']}
                    - **Effect Size ({effect_size_str})**: The practical difference is **{effect_interp}**. {STATS_EXPLAINER['effect_size']}

                    {"✅ **Action:** The differences are real and meaningful. Investigate high-return categories." if result.significant and effect_interp in ['large', 'medium']
                     else "⚠️ **Action:** Differences exist but may not be practically significant. Monitor trends." if result.significant
                     else "⚠️ **Action:** No significant differences detected. Variation is within normal range."}
                    """)

                if result.outlier_categories:
                    st.warning(f"⚠️ Outlier Categories: {', '.join(str(c) for c in result.outlier_categories)}")
    
    # Risk Heatmap
    if ALTAIR_AVAILABLE and 'Landed Cost' in df.columns and df['Landed Cost'].sum() > 0:
        with st.expander("🔥 Risk Heatmap (Return Rate vs Cost)", expanded=True):
            chart_df = df[['SKU', 'Return_Rate', 'Landed Cost', 'Risk_Score', 'Action', 'Category']].copy()
            chart_df['Return_Rate_Pct'] = chart_df['Return_Rate'] * 100
            
            chart = alt.Chart(chart_df).mark_circle(size=100).encode(
                x=alt.X('Landed Cost:Q', title='Landed Cost ($)', scale=alt.Scale(zero=False)),
                y=alt.Y('Return_Rate_Pct:Q', title='Return Rate (%)', scale=alt.Scale(zero=False)),
                color=alt.Color('Risk_Score:Q', 
                               scale=alt.Scale(scheme='redyellowgreen', reverse=True, domain=[0, 100]),
                               legend=alt.Legend(title='Risk Score')),
                size=alt.Size('Risk_Score:Q', scale=alt.Scale(range=[50, 500]), legend=None),
                tooltip=['SKU', 'Category', 'Return_Rate_Pct', 'Landed Cost', 'Risk_Score', 'Action']
            ).interactive().properties(height=400)
            
            st.altair_chart(chart, use_container_width=True)
    
    # Claude Review (if available)
    claude_reviews = [c for c in st.session_state.ai_chat_history if c.get('role') == 'claude_review']
    if claude_reviews:
        with st.expander("🤖 Claude AI Review", expanded=True):
            st.markdown(claude_reviews[-1]['content'])
    
    # Results Table
    st.markdown("#### Detailed Results")

    # Load historical data and product matcher
    load_historical_data()

    # Add product comparison benchmarking if available
    if st.session_state.product_matcher:
        with st.expander("🔍 Product Comparison vs. Historical Data", expanded=False):
            st.markdown("""
            <div style="background: linear-gradient(90deg, rgba(35,178,190,0.15) 0%, rgba(35,178,190,0.03) 100%);
                        border-left: 4px solid #004366; padding: 1.2rem; margin-bottom: 1.2rem; border-radius: 6px;">
                <strong style="color: #23b2be; font-size: 1.1em; font-family: 'Poppins', sans-serif;">✨ AI-Powered Product Benchmarking</strong><br>
                <span style="font-family: 'Poppins', sans-serif;">Compare your products against <strong>231 historical products</strong> from trailing 12-month Amazon data.<br>
                Uses fuzzy matching to find similar products (e.g., "4-wheel scooter" matches "3-wheel scooter").</span>
            </div>
            """, unsafe_allow_html=True)

            # Add benchmark columns to dataframe
            benchmark_data = []
            for _, row in df.iterrows():
                if st.session_state.product_matcher and 'Name' in row and 'Return_Rate' in row:
                    try:
                        comparison = st.session_state.product_matcher.compare_to_similar_products(
                            product_name=row['Name'],
                            product_return_rate=row['Return_Rate'],
                            product_category=row.get('Category')
                        )

                        if comparison.get('comparison_available'):
                            benchmark_data.append({
                                'SKU': row.get('SKU', ''),
                                'Product': row['Name'][:40] + '...' if len(row['Name']) > 40 else row['Name'],
                                'Your Return Rate': f"{row['Return_Rate']:.1%}",
                                'Similar Products Avg': f"{comparison['benchmark_average']:.1%}",
                                'vs. Average': f"{comparison['vs_average_pct']:+.1f}%",
                                'Performance': comparison['performance_category'],
                                'Similar Count': comparison['similar_product_count']
                            })
                    except Exception as e:
                        logger.warning(f"Benchmark comparison failed for {row.get('SKU')}: {e}")
                        continue

            if benchmark_data:
                benchmark_df = pd.DataFrame(benchmark_data)

                # Color code performance with Vive brand colors
                def color_performance(row):
                    perf = row['Performance']
                    colors = []
                    for _ in range(len(row)):
                        if perf == 'Excellent':
                            # Vive Turquoise with Navy accent
                            colors.append('background-color: rgba(35,178,190,0.25); color: #004366; font-weight: 600; font-family: Poppins, sans-serif')
                        elif perf == 'Good':
                            # Light Turquoise
                            colors.append('background-color: rgba(35,178,190,0.12); color: #004366; font-family: Poppins, sans-serif')
                        elif perf == 'Fair':
                            # Yellow/Gold
                            colors.append('background-color: rgba(240,179,35,0.2); color: #004366; font-family: Poppins, sans-serif')
                        else:
                            # Red/Orange for needs improvement
                            colors.append('background-color: rgba(235,51,0,0.2); color: #004366; font-weight: 600; font-family: Poppins, sans-serif')
                    return colors

                st.markdown("##### 📊 Benchmark Comparison Results")
                st.dataframe(
                    benchmark_df.style.apply(color_performance, axis=1),
                    use_container_width=True,
                    height=400
                )

                st.markdown("""
                <div style="background: rgba(35,178,190,0.08); border-left: 3px solid #23b2be; padding: 0.8rem; margin-top: 1rem; border-radius: 4px; font-family: 'Poppins', sans-serif;">
                    <strong style="color: #004366;">Performance Categories:</strong><br>
                    <span style="background: rgba(35,178,190,0.25); padding: 0.2rem 0.5rem; border-radius: 3px; margin-right: 0.5rem; color: #004366; font-weight: 600;">Excellent</span> Top 25% (Best in class)<br>
                    <span style="background: rgba(35,178,190,0.12); padding: 0.2rem 0.5rem; border-radius: 3px; margin-right: 0.5rem; color: #004366;">Good</span> Above Median (Better than average)<br>
                    <span style="background: rgba(240,179,35,0.2); padding: 0.2rem 0.5rem; border-radius: 3px; margin-right: 0.5rem; color: #004366;">Fair</span> Below Median (Room for improvement)<br>
                    <span style="background: rgba(235,51,0,0.2); padding: 0.2rem 0.5rem; border-radius: 3px; margin-right: 0.5rem; color: #004366; font-weight: 600;">Needs Improvement</span> Bottom 25% (Requires immediate attention)
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Product matching in progress... Refresh to see results.")

    # Add color coding based on action - Vive brand colors
    def highlight_action(row):
        if 'Immediate' in str(row.get('Action', '')):
            return ['background-color: rgba(235,51,0,0.25); color: #004366; font-weight: 600; font-family: Poppins, sans-serif'] * len(row)
        elif 'Case' in str(row.get('Action', '')):
            return ['background-color: rgba(240,179,35,0.25); color: #004366; font-weight: 600; font-family: Poppins, sans-serif'] * len(row)
        elif 'Monitor' in str(row.get('Action', '')):
            return ['background-color: rgba(35,178,190,0.15); color: #004366; font-family: Poppins, sans-serif'] * len(row)
        return ['font-family: Poppins, sans-serif'] * len(row)

    # Select display columns
    display_cols = ['SKU', 'Name', 'Category', 'Return_Rate', 'Category_Threshold',
                   'Landed Cost', 'Risk_Score', 'SPC_Signal', 'Action', 'Triggers']
    display_cols = [c for c in display_cols if c in df.columns]

    # Format display
    display_df = df[display_cols].copy()
    if 'Return_Rate' in display_df.columns:
        display_df['Return_Rate'] = display_df['Return_Rate'].apply(lambda x: f"{x:.1%}")
    if 'Category_Threshold' in display_df.columns:
        display_df['Category_Threshold'] = display_df['Category_Threshold'].apply(lambda x: f"{x:.1%}")
    if 'Risk_Score' in display_df.columns:
        display_df['Risk_Score'] = display_df['Risk_Score'].apply(lambda x: f"{x:.0f}")

    st.dataframe(
        display_df.style.apply(highlight_action, axis=1),
        use_container_width=True,
        height=400
    )
    
    # Action Items Section
    st.markdown("---")
    st.markdown("### 🎯 Action Items")
    
    # Filter for items needing action
    action_items = df[df['Action'].str.contains('Escalat|Case', na=False)]
    
    if len(action_items) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📧 Generate AI-Powered Vendor Email")
            st.caption("✨ NEW: Multi-language support with English proficiency adjustment")

            selected_sku = st.selectbox(
                "Select SKU",
                options=action_items['SKU'].unique(),
                key="email_sku_select"
            )

            col_a, col_b = st.columns(2)
            with col_a:
                email_type = st.selectbox(
                    "Email Type",
                    ["CAPA Request", "RCA Request", "Quality Report"]
                )

            with col_b:
                vendor_region = st.selectbox(
                    "Vendor Region",
                    ["China", "India", "LATAM", "EU", "USA"],
                    help="Adjusts communication style for cultural context"
                )

            col_c, col_d = st.columns(2)
            with col_c:
                english_level = st.selectbox(
                    "English Proficiency",
                    ["Native", "Fluent", "Intermediate", "Basic", "Minimal"],
                    index=2,
                    help="Adjusts language complexity for recipient"
                )

            with col_d:
                target_lang = st.selectbox(
                    "Target Language",
                    ["English", "Chinese (Simplified)", "Chinese (Traditional)",
                     "Spanish", "Portuguese", "Hindi", "German", "French", "Italian"],
                    help="Generates translation alongside English"
                )

            # Map string selections to enums
            english_level_map = {
                "Native": EnglishLevel.NATIVE,
                "Fluent": EnglishLevel.FLUENT,
                "Intermediate": EnglishLevel.INTERMEDIATE,
                "Basic": EnglishLevel.BASIC,
                "Minimal": EnglishLevel.MINIMAL
            }

            target_lang_map = {
                "English": TargetLanguage.ENGLISH,
                "Chinese (Simplified)": TargetLanguage.CHINESE_SIMPLIFIED,
                "Chinese (Traditional)": TargetLanguage.CHINESE_TRADITIONAL,
                "Spanish": TargetLanguage.SPANISH,
                "Portuguese": TargetLanguage.PORTUGUESE,
                "Hindi": TargetLanguage.HINDI,
                "German": TargetLanguage.GERMAN,
                "French": TargetLanguage.FRENCH,
                "Italian": TargetLanguage.ITALIAN
            }

            if st.button("🚀 Generate AI Email", key="gen_email", type="primary"):
                # Ensure AI and multilingual communicator are loaded
                get_ai_analyzer()
                load_historical_data()

                if st.session_state.multilingual_communicator is None:
                    st.warning("AI communicator not initialized. Generating basic email...")
                    # Fallback to old generator
                    row = action_items[action_items['SKU'] == selected_sku].iloc[0]
                    if email_type == "CAPA Request":
                        email = VendorEmailGenerator.generate_capa_request(
                            sku=row['SKU'],
                            product_name=row.get('Name', row['SKU']),
                            issue_summary=row.get('Complaint_Text', 'Quality concerns identified'),
                            return_rate=row['Return_Rate'],
                            defect_description=row.get('Triggers', ''),
                            units_affected=int(row.get('Returned', 0))
                        )
                        st.text_area("Generated Email", email, height=400)
                else:
                    with st.spinner("🤖 AI is generating your customized email..."):
                        row = action_items[action_items['SKU'] == selected_sku].iloc[0]

                        try:
                            if email_type == "CAPA Request":
                                result = st.session_state.multilingual_communicator.generate_capa_email(
                                    sku=row['SKU'],
                                    product_name=row.get('Name', row['SKU']),
                                    issue_summary=row.get('Complaint_Text', 'Quality concerns identified'),
                                    return_rate=row['Return_Rate'],
                                    defect_description=row.get('Triggers', ''),
                                    units_affected=int(row.get('Returned', 0)),
                                    severity='major',
                                    english_level=english_level_map[english_level],
                                    target_language=target_lang_map[target_lang],
                                    vendor_region=vendor_region
                                )
                            elif email_type == "RCA Request":
                                result = st.session_state.multilingual_communicator.generate_rca_request(
                                    sku=row['SKU'],
                                    product_name=row.get('Name', row['SKU']),
                                    defect_type=row['Action'],
                                    occurrence_rate=row['Return_Rate'],
                                    sample_complaints=str(row.get('Complaint_Text', '')).split(',')[:5],
                                    english_level=english_level_map[english_level],
                                    target_language=target_lang_map[target_lang],
                                    vendor_region=vendor_region
                                )
                            else:  # Quality Report
                                products_list = [{
                                    'sku': row['SKU'],
                                    'product_name': row.get('Name', row['SKU']),
                                    'return_rate': row['Return_Rate'],
                                    'issue_summary': row.get('Triggers', '')
                                }]
                                result = st.session_state.multilingual_communicator.generate_quality_report(
                                    products=products_list,
                                    report_type="product review",
                                    english_level=english_level_map[english_level],
                                    target_language=target_lang_map[target_lang],
                                    vendor_region=vendor_region
                                )

                            # Display results
                            st.success("✅ Email generated successfully!")

                            st.markdown("### 📧 English Version")
                            st.markdown(f"**Subject:** {result['subject_english']}")
                            st.text_area("Email Body (English)", result['body_english'], height=350, key="email_en")

                            # Show translation if available
                            if result['body_translated']:
                                st.markdown(f"### 🌐 {result['language']} Translation")
                                if result['subject_translated']:
                                    st.markdown(f"**Subject:** {result['subject_translated']}")
                                st.text_area(f"Email Body ({result['language']})", result['body_translated'], height=350, key="email_trans")

                            # Download options
                            col_dl1, col_dl2 = st.columns(2)
                            with col_dl1:
                                st.download_button(
                                    "📥 Download English",
                                    f"Subject: {result['subject_english']}\n\n{result['body_english']}\n\nRef: {result['reference']}",
                                    file_name=f"vendor_email_EN_{selected_sku}_{datetime.now().strftime('%Y%m%d')}.txt"
                                )

                            with col_dl2:
                                if result['body_translated']:
                                    translated_content = f"Subject: {result['subject_translated'] or result['subject_english']}\n\n{result['body_translated']}\n\n---\n\nEnglish Version:\nSubject: {result['subject_english']}\n\n{result['body_english']}\n\nRef: {result['reference']}"
                                    st.download_button(
                                        f"📥 Download {result['language']}",
                                        translated_content,
                                        file_name=f"vendor_email_{target_lang_map[target_lang].value}_{selected_sku}_{datetime.now().strftime('%Y%m%d')}.txt"
                                    )

                        except Exception as e:
                            st.error(f"Error generating email: {str(e)}")
                            logger.error(f"Email generation error: {e}", exc_info=True)
        
        with col2:
            st.markdown("#### 📋 Generate Investigation Plan")
            plan_sku = st.selectbox(
                "Select SKU",
                options=action_items['SKU'].unique(),
                key="plan_sku_select"
            )
            
            if st.button("Generate Plan", key="gen_plan"):
                row = action_items[action_items['SKU'] == plan_sku].iloc[0]
                
                plan = InvestigationPlanGenerator.generate_plan(
                    sku=row['SKU'],
                    product_name=row.get('Name', row['SKU']),
                    category=row.get('Category', 'Unknown'),
                    issue_type=row.get('Action', 'Quality Issue'),
                    complaint_summary=row.get('Complaint_Text', 'See triggers'),
                    return_rate=row['Return_Rate'],
                    risk_score=row['Risk_Score']
                )
                
                plan_md = InvestigationPlanGenerator.format_plan_markdown(plan)
                st.markdown(plan_md)
                
                st.download_button(
                    "📥 Download Plan",
                    plan_md,
                    file_name=f"investigation_plan_{plan_sku}_{datetime.now().strftime('%Y%m%d')}.md"
                )
    else:
        st.success("✅ No immediate action items. All products within acceptable thresholds.")

    # ========== NEW: DEEP DIVE ANALYSIS & BULK OPERATIONS ==========
    if len(action_items) > 0:
        st.markdown("---")
        st.markdown("### 🔬 Advanced Analysis & Bulk Operations")

        tab1, tab2, tab3 = st.tabs(["🔍 Deep Dive Analysis", "📧 Bulk Vendor Emails", "📋 Bulk Investigation Plans"])

        # TAB 1: Deep Dive Analysis
        with tab1:
            st.markdown("#### AI-Powered Deep Dive Analysis")
            st.caption("Upload product documentation for comprehensive AI analysis with investigation method recommendations")

            col1, col2 = st.columns([1, 2])

            with col1:
                deep_dive_sku = st.selectbox(
                    "Select Product for Deep Dive",
                    options=action_items['SKU'].unique(),
                    key="deep_dive_sku",
                    help=STATS_EXPLAINER.get('confidence_interval', '')
                )

                # Investigation method info
                st.markdown("**Investigation Methods:**")
                for method_key, method_info in {
                    '5_whys': '5 Whys - Simple, linear problems',
                    'fishbone': 'Fishbone - Complex, multi-factor issues',
                    'rca': 'Formal RCA - Critical/safety issues',
                    'fmea': 'FMEA - Proactive risk assessment',
                    '8d': '8D - Customer-facing team response',
                    'pareto': 'Pareto - Prioritize multiple issues'
                }.items():
                    with st.expander(method_info):
                        st.caption(f"**Best for:** {INVESTIGATION_METHODS.get(method_key, {}).get('best_for', 'N/A')}")
                        st.caption(f"**Use when:** {INVESTIGATION_METHODS.get(method_key, {}).get('use_when', 'N/A')}")

            with col2:
                # Document uploads
                st.markdown("**Upload Product Documentation (Optional but Recommended)**")

                col_a, col_b = st.columns(2)

                with col_a:
                    manual_file = st.file_uploader(
                        "📖 Product Manual",
                        type=['pdf', 'txt', 'docx'],
                        key="manual_upload",
                        help="Upload product manual for AI to analyze intended use and identify design issues"
                    )

                    amazon_file = st.file_uploader(
                        "🛒 Amazon Listing",
                        type=['pdf', 'txt', 'html'],
                        key="amazon_upload",
                        help="Upload Amazon listing/bullets to compare marketed features vs reported issues"
                    )

                with col_b:
                    ifu_file = st.file_uploader(
                        "📋 IFU (Instructions for Use)",
                        type=['pdf', 'txt', 'docx'],
                        key="ifu_upload",
                        help="Upload IFU to check if customer errors relate to unclear instructions"
                    )

                    specs_file = st.file_uploader(
                        "⚙️ Technical Specs",
                        type=['pdf', 'txt', 'xlsx'],
                        key="specs_upload",
                        help="Upload specs to identify if returns relate to spec deviations"
                    )

                # Run Deep Dive Analysis
                if st.button("🚀 Run Deep Dive Analysis", type="primary", key="run_deep_dive"):
                    with st.spinner("AI is analyzing product details, documentation, and recommending investigation methods..."):
                        try:
                            # Get product data
                            product_row = action_items[action_items['SKU'] == deep_dive_sku].iloc[0]
                            product_data = product_row.to_dict()

                            # Process uploaded docs
                            product_docs = {}
                            if manual_file:
                                product_docs['manual'] = manual_file.read().decode('utf-8', errors='ignore')
                            if amazon_file:
                                product_docs['amazon_listing'] = amazon_file.read().decode('utf-8', errors='ignore')
                            if ifu_file:
                                product_docs['ifu'] = ifu_file.read().decode('utf-8', errors='ignore')
                            if specs_file:
                                product_docs['specs'] = specs_file.read().decode('utf-8', errors='ignore')

                            # Run deep dive (using AI)
                            if AI_AVAILABLE:
                                ai_analyzer = EnhancedAIAnalyzer(provider=st.session_state.ai_provider)
                                deep_dive = DeepDiveAnalyzer(ai_analyzer)
                                analysis = deep_dive.analyze_flagged_product(product_data, product_docs)

                                # Display results
                                st.success("✅ Deep Dive Analysis Complete!")

                                # Risk Level
                                risk_level = analysis.get('risk_level', 'Medium')
                                risk_colors = {'Low': '🟢', 'Medium': '🟡', 'High': '🟠', 'Critical': '🔴'}
                                st.markdown(f"### {risk_colors.get(risk_level, '⚪')} Risk Level: {risk_level}")

                                # Recommended Method
                                st.markdown("### 🎯 Recommended Investigation Method")
                                method_key = analysis.get('recommended_method', 'rca')
                                method_details = deep_dive.get_method_details(method_key)
                                st.info(f"**{method_details['name']}**\n\n{method_details['best_for']}\n\n**Use when:** {method_details['use_when']}")

                                # Full Analysis
                                with st.expander("📊 Full AI Analysis", expanded=True):
                                    if 'raw_analysis' in analysis:
                                        st.markdown(analysis['raw_analysis'])
                                    else:
                                        st.json(analysis)

                                # Store for use in investigation plan
                                st.session_state.last_deep_dive = {
                                    'sku': deep_dive_sku,
                                    'analysis': analysis,
                                    'method': method_key
                                }

                            else:
                                st.error("AI modules not available. Install required packages.")

                        except Exception as e:
                            st.error(f"Deep dive analysis failed: {e}")
                            logger.error(f"Deep dive error: {e}")

        # TAB 2: Bulk Vendor Emails - ENHANCED WITH MULTILINGUAL AI
        with tab2:
            st.markdown("#### 🌐 AI-Powered Bulk Vendor Emails (Multilingual)")
            st.caption("✨ NEW: Generate customized emails for multiple products with language/proficiency options")

            # Select products
            selected_for_email = st.multiselect(
                "Select Products for Vendor Communication",
                options=action_items['SKU'].tolist(),
                default=action_items['SKU'].tolist()[:5],  # Default to first 5
                help="Select which products need vendor follow-up"
            )

            # Email configuration in columns
            col1, col2 = st.columns(2)
            with col1:
                bulk_email_type = st.selectbox(
                    "Email Type (applies to all)",
                    ["CAPA Request", "RCA Request", "Quality Report"],
                    help="Same email type will be used for all selected products"
                )

                bulk_vendor_region = st.selectbox(
                    "Vendor Region",
                    ["China", "India", "LATAM", "EU", "USA"],
                    help="Adjusts communication style for cultural context",
                    key="bulk_vendor_region"
                )

            with col2:
                bulk_english_level = st.selectbox(
                    "English Proficiency",
                    ["Native", "Fluent", "Intermediate", "Basic", "Minimal"],
                    index=2,
                    help="Adjusts language complexity",
                    key="bulk_english"
                )

                bulk_target_lang = st.selectbox(
                    "Target Language",
                    ["English", "Chinese (Simplified)", "Chinese (Traditional)",
                     "Spanish", "Portuguese", "Hindi", "German", "French", "Italian"],
                    help="Generates translation",
                    key="bulk_lang"
                )

            vendor_name = st.text_input(
                "Vendor/Supplier Name (Optional)",
                placeholder="e.g., ABC Manufacturing Ltd.",
                help="For email personalization"
            )

            # Map selections to enums
            english_level_map = {
                "Native": EnglishLevel.NATIVE,
                "Fluent": EnglishLevel.FLUENT,
                "Intermediate": EnglishLevel.INTERMEDIATE,
                "Basic": EnglishLevel.BASIC,
                "Minimal": EnglishLevel.MINIMAL
            }

            target_lang_map = {
                "English": TargetLanguage.ENGLISH,
                "Chinese (Simplified)": TargetLanguage.CHINESE_SIMPLIFIED,
                "Chinese (Traditional)": TargetLanguage.CHINESE_TRADITIONAL,
                "Spanish": TargetLanguage.SPANISH,
                "Portuguese": TargetLanguage.PORTUGUESE,
                "Hindi": TargetLanguage.HINDI,
                "German": TargetLanguage.GERMAN,
                "French": TargetLanguage.FRENCH,
                "Italian": TargetLanguage.ITALIAN
            }

            if st.button("🚀 Generate All AI Emails", type="primary", key="bulk_emails"):
                if not selected_for_email:
                    st.warning("Please select at least one product")
                else:
                    # Ensure AI is initialized
                    get_ai_analyzer()
                    load_historical_data()

                    with st.spinner(f"🤖 AI is generating {len(selected_for_email)} customized emails..."):
                        try:
                            bulk_emails = []
                            progress_bar = st.progress(0)

                            # Use multilingual communicator if available
                            use_ai = st.session_state.multilingual_communicator is not None

                            for idx, sku in enumerate(selected_for_email):
                                row = action_items[action_items['SKU'] == sku].iloc[0]

                                if use_ai:
                                    # AI-powered generation
                                    if bulk_email_type == "CAPA Request":
                                        result = st.session_state.multilingual_communicator.generate_capa_email(
                                            sku=row['SKU'],
                                            product_name=row.get('Name', row['SKU']),
                                            issue_summary=row.get('Complaint_Text', 'Quality concerns identified'),
                                            return_rate=row['Return_Rate'],
                                            defect_description=row.get('Triggers', ''),
                                            units_affected=int(row.get('Returned', 0)),
                                            severity='major',
                                            english_level=english_level_map[bulk_english_level],
                                            target_language=target_lang_map[bulk_target_lang],
                                            vendor_region=bulk_vendor_region
                                        )
                                    elif bulk_email_type == "RCA Request":
                                        result = st.session_state.multilingual_communicator.generate_rca_request(
                                            sku=row['SKU'],
                                            product_name=row.get('Name', row['SKU']),
                                            defect_type=row['Action'],
                                            occurrence_rate=row['Return_Rate'],
                                            sample_complaints=str(row.get('Complaint_Text', '')).split(',')[:5],
                                            english_level=english_level_map[bulk_english_level],
                                            target_language=target_lang_map[bulk_target_lang],
                                            vendor_region=bulk_vendor_region
                                        )
                                    else:  # Quality Report
                                        products_list = [{
                                            'sku': row['SKU'],
                                            'product_name': row.get('Name', row['SKU']),
                                            'return_rate': row['Return_Rate'],
                                            'issue_summary': row.get('Triggers', '')
                                        }]
                                        result = st.session_state.multilingual_communicator.generate_quality_report(
                                            products=products_list,
                                            report_type="product review",
                                            english_level=english_level_map[bulk_english_level],
                                            target_language=target_lang_map[bulk_target_lang],
                                            vendor_region=bulk_vendor_region
                                        )

                                    bulk_emails.append({
                                        'SKU': sku,
                                        'Product': row.get('Name', sku)[:50],
                                        'Email_Type': bulk_email_type,
                                        'Subject_English': result['subject_english'],
                                        'Body_English': result['body_english'],
                                        'Subject_Translated': result.get('subject_translated', ''),
                                        'Body_Translated': result.get('body_translated', ''),
                                        'Language': result['language'],
                                        'Priority': row.get('Action', 'Monitor'),
                                        'Vendor_Region': bulk_vendor_region,
                                        'English_Level': bulk_english_level
                                    })
                                else:
                                    # Fallback to basic generator
                                    if bulk_email_type == "CAPA Request":
                                        email = VendorEmailGenerator.generate_capa_request(
                                            sku=row['SKU'],
                                            product_name=row.get('Name', row['SKU']),
                                            issue_summary=row.get('Complaint_Text', 'Quality concerns identified'),
                                            return_rate=row['Return_Rate'],
                                            defect_description=row.get('Triggers', ''),
                                            units_affected=int(row.get('Returned', 0))
                                        )
                                    elif bulk_email_type == "RCA Request":
                                        email = VendorEmailGenerator.generate_rca_request(
                                            sku=row['SKU'],
                                            product_name=row.get('Name', row['SKU']),
                                            defect_type=row['Action'],
                                            occurrence_rate=row['Return_Rate'],
                                            sample_complaints=str(row.get('Complaint_Text', '')).split(',')[:5]
                                        )

                                    bulk_emails.append({
                                        'SKU': sku,
                                        'Product': row.get('Name', sku),
                                        'Email_Type': bulk_email_type,
                                        'Subject_English': f"Quality Issue - {sku}",
                                        'Body_English': email,
                                        'Priority': row.get('Action', 'Monitor')
                                    })

                                # Update progress
                                progress_bar.progress((idx + 1) / len(selected_for_email))

                            progress_bar.empty()
                            st.success(f"✅ Generated {len(bulk_emails)} {'AI-powered' if use_ai else ''} emails!")

                            # Display preview
                            st.markdown("### 📧 Email Previews (First 3)")
                            for i, email_data in enumerate(bulk_emails[:3]):
                                with st.expander(f"📧 {email_data['SKU']} - {email_data['Product']}", expanded=(i==0)):
                                    st.markdown(f"**Priority:** {email_data['Priority']}")
                                    if use_ai:
                                        st.markdown(f"**Language:** {email_data.get('Language', 'English')} | **Region:** {email_data.get('Vendor_Region', 'N/A')}")

                                    st.markdown("**English Version:**")
                                    st.text_area("Subject", email_data['Subject_English'], height=50, key=f"subj_en_{i}")
                                    st.text_area("Body", email_data['Body_English'], height=200, key=f"body_en_{i}")

                                    if email_data.get('Body_Translated'):
                                        st.markdown(f"**{email_data['Language']} Translation:**")
                                        if email_data.get('Subject_Translated'):
                                            st.text_area("Subject (Translated)", email_data['Subject_Translated'], height=50, key=f"subj_tr_{i}")
                                        st.text_area("Body (Translated)", email_data['Body_Translated'], height=200, key=f"body_tr_{i}")

                            if len(bulk_emails) > 3:
                                st.info(f"+ {len(bulk_emails) - 3} more emails (see exports below)")

                            # Export options
                            col_exp1, col_exp2 = st.columns(2)

                            with col_exp1:
                                email_df = pd.DataFrame(bulk_emails)
                                csv_data = email_df.to_csv(index=False)
                                st.download_button(
                                    "📥 Download All Emails (CSV)",
                                    csv_data,
                                    file_name=f"bulk_vendor_emails_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                    mime="text/csv"
                                )

                            with col_exp2:
                                # Create individual text files combined
                                combined_text = ""
                                for email in bulk_emails:
                                    combined_text += f"\n{'='*80}\n"
                                    combined_text += f"SKU: {email['SKU']} - {email['Product']}\n"
                                    combined_text += f"Priority: {email['Priority']}\n"
                                    combined_text += f"{'='*80}\n\n"
                                    combined_text += f"SUBJECT: {email['Subject_English']}\n\n"
                                    combined_text += email['Body_English']
                                    if email.get('Body_Translated'):
                                        combined_text += f"\n\n--- {email['Language']} TRANSLATION ---\n\n"
                                        combined_text += f"SUBJECT: {email.get('Subject_Translated', email['Subject_English'])}\n\n"
                                        combined_text += email['Body_Translated']
                                    combined_text += "\n\n"

                                st.download_button(
                                    "📥 Download All as Text",
                                    combined_text,
                                    file_name=f"bulk_vendor_emails_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                                    mime="text/plain"
                                )

                        except Exception as e:
                            st.error(f"Bulk email generation failed: {e}")
                            logger.error(f"Bulk email error: {e}", exc_info=True)

        # TAB 3: Bulk Investigation Plans
        with tab3:
            st.markdown("#### Generate Investigation Plans for Multiple Products")
            st.caption("Create comprehensive investigation plans with timelines and team assignments")

            # Select products
            selected_for_plans = st.multiselect(
                "Select Products for Investigation Planning",
                options=action_items['SKU'].tolist(),
                default=action_items['SKU'].tolist()[:5],
                help="Select which products need investigation plans"
            )

            # Allow custom method selection per SKU
            use_custom_methods = st.checkbox(
                "Customize investigation method per product",
                help="Assign different investigation methods to different products"
            )

            method_assignments = {}
            if use_custom_methods and selected_for_plans:
                st.markdown("**Assign Investigation Methods:**")
                cols = st.columns(3)
                for i, sku in enumerate(selected_for_plans):
                    with cols[i % 3]:
                        method_assignments[sku] = st.selectbox(
                            f"{sku}",
                            ["Auto (AI)", "5 Whys", "Fishbone", "Formal RCA", "FMEA", "8D", "Pareto"],
                            key=f"method_{sku}"
                        )

            # Regulatory Compliance Section
            st.markdown("---")
            st.markdown("#### 🌍 Regulatory Compliance Analysis")
            st.caption("Select markets to analyze regulatory requirements with AI-powered screening")

            # Market selection with checkboxes
            st.markdown("**Select Markets for Compliance Analysis:**")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Americas**")
                us_market = st.checkbox("🇺🇸 United States (FDA)", value=True, key="market_us")
                mexico_market = st.checkbox("🇲🇽 Mexico (COFEPRIS)", key="market_mexico")
                colombia_market = st.checkbox("🇨🇴 Colombia (INVIMA)", key="market_colombia")
                brazil_market = st.checkbox("🇧🇷 Brazil (ANVISA)", key="market_brazil")
                chile_market = st.checkbox("🇨🇱 Chile (ISP)", key="market_chile")

            with col2:
                st.markdown("**Europe**")
                uk_market = st.checkbox("🇬🇧 United Kingdom (MHRA)", key="market_uk")
                germany_market = st.checkbox("🇩🇪 Germany (BfArM)", key="market_germany")
                france_market = st.checkbox("🇫🇷 France (ANSM)", key="market_france")
                italy_market = st.checkbox("🇮🇹 Italy", key="market_italy")
                spain_market = st.checkbox("🇪🇸 Spain (AEMPS)", key="market_spain")
                netherlands_market = st.checkbox("🇳🇱 Netherlands (IGJ)", key="market_netherlands")

            with col3:
                st.markdown("**Other Markets**")
                other_markets_text = st.text_area(
                    "Specify other countries:",
                    placeholder="e.g., Canada, Australia, Japan",
                    height=100,
                    key="other_markets"
                )

            # Select all button
            if st.button("✅ Select All Major Markets", key="select_all_markets"):
                st.info("To select all markets, check the boxes above manually")

            # Collect selected markets
            selected_markets = []
            if us_market:
                selected_markets.append('US')
            if mexico_market:
                selected_markets.append('Mexico')
            if colombia_market:
                selected_markets.append('Colombia')
            if brazil_market:
                selected_markets.append('Brazil')
            if chile_market:
                selected_markets.append('Chile')
            if uk_market:
                selected_markets.append('UK')
            if germany_market:
                selected_markets.append('Germany')
            if france_market:
                selected_markets.append('France')
            if italy_market:
                selected_markets.append('Italy')
            if spain_market:
                selected_markets.append('Spain')
            if netherlands_market:
                selected_markets.append('Netherlands')

            # Show regulatory links
            if selected_markets:
                with st.expander("📚 Regulatory Agency Links", expanded=False):
                    for market_code in selected_markets:
                        if market_code in REGULATORY_MARKETS:
                            market_info = REGULATORY_MARKETS[market_code]
                            st.markdown(f"**{market_info['name']}:**")
                            for agency in market_info['agencies']:
                                st.markdown(f"- [{agency['name']}]({agency['url']})")

            st.markdown("---")

            if st.button("📋 Generate All Plans with Regulatory Analysis", type="primary", key="bulk_plans"):
                if not selected_for_plans:
                    st.warning("Please select at least one product")
                else:
                    with st.spinner(f"Generating {len(selected_for_plans)} investigation plans with regulatory compliance analysis..."):
                        try:
                            bulk_plans = []

                            # Initialize regulatory analyzer if markets selected
                            reg_analyzer = None
                            if selected_markets and AI_AVAILABLE:
                                try:
                                    ai_analyzer = st.session_state.get('ai_analyzer')
                                    reg_analyzer = RegulatoryComplianceAnalyzer(ai_analyzer)
                                except Exception as e:
                                    logger.warning(f"Regulatory analyzer initialization failed: {e}")

                            for sku in selected_for_plans:
                                row = action_items[action_items['SKU'] == sku].iloc[0]
                                assigned_method = method_assignments.get(sku, "Auto (AI)")

                                plan = InvestigationPlanGenerator.generate_plan(
                                    sku=row['SKU'],
                                    product_name=row.get('Name', row['SKU']),
                                    category=row.get('Category', 'Unknown'),
                                    issue_type=row.get('Action', 'Quality Issue'),
                                    complaint_summary=row.get('Complaint_Text', 'See triggers'),
                                    return_rate=row['Return_Rate'],
                                    risk_score=row['Risk_Score']
                                )

                                # Regulatory analysis if enabled
                                reg_requirements = ""
                                reg_actions = ""
                                ai_compliance_suggestions = ""

                                if reg_analyzer and selected_markets:
                                    product_data = {
                                        'sku': row['SKU'],
                                        'product_name': row.get('Name', row['SKU']),
                                        'category': row.get('Category', 'Unknown'),
                                        'return_rate': row['Return_Rate'],
                                        'complaint_summary': row.get('Complaint_Text', ''),
                                        'safety_risk': row.get('Safety_Risk', False)
                                    }

                                    reg_analysis = reg_analyzer.analyze_compliance_requirements(
                                        selected_markets,
                                        product_data,
                                        row.get('Action', 'Quality Issue')
                                    )

                                    # Format regulatory requirements
                                    if reg_analysis.get('injury_reporting_required'):
                                        reg_requirements = "; ".join([
                                            f"{r['market']}: {r['timeline']}"
                                            for r in reg_analysis['injury_reporting_required']
                                        ])

                                    # Format AI suggestions
                                    if reg_analysis.get('ai_suggestions'):
                                        ai_compliance_suggestions = "\n".join([
                                            f"[{s.get('confidence', 0)}% confidence] {s.get('requirement', '')}"
                                            for s in reg_analysis['ai_suggestions']
                                        ])

                                bulk_plans.append({
                                    'SKU': sku,
                                    'Product': row.get('Name', sku),
                                    'Category': row.get('Category', 'Unknown'),
                                    'Return_Rate': f"{row['Return_Rate'] * 100:.2f}%",
                                    'Risk_Score': row['Risk_Score'],
                                    'Severity': row.get('Action', 'Monitor'),
                                    'Safety_Risk': row.get('Safety_Risk', False),
                                    'Complaint_Summary': row.get('Complaint_Text', '')[:200],
                                    'Investigation_Method': assigned_method,
                                    'Priority': row.get('Action', 'Monitor'),
                                    'Estimated_Days': plan.get('timeline_days', 14),
                                    'Team_Required': ', '.join(plan.get('team', [])),
                                    'Markets_Analyzed': ', '.join(selected_markets) if selected_markets else 'None',
                                    'Injury_Reporting_Requirements': reg_requirements or 'N/A',
                                    'AI_Compliance_Suggestions': ai_compliance_suggestions or 'N/A',
                                    'Investigation_Plan': InvestigationPlanGenerator.format_plan_markdown(plan),
                                })

                            st.success(f"✅ Generated {len(bulk_plans)} investigation plans with regulatory analysis!")

                            # Display preview
                            for i, plan_data in enumerate(bulk_plans[:2]):  # Show first 2
                                with st.expander(f"📋 {plan_data['SKU']} - {plan_data['Product']}", expanded=(i==0)):
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Return Rate", plan_data['Return_Rate'])
                                        st.metric("Risk Score", plan_data['Risk_Score'])
                                    with col2:
                                        st.write(f"**Method:** {plan_data['Investigation_Method']}")
                                        st.write(f"**Timeline:** {plan_data['Estimated_Days']} days")
                                    with col3:
                                        st.write(f"**Markets:** {plan_data['Markets_Analyzed']}")
                                        st.write(f"**Safety Risk:** {'⚠️ YES' if plan_data['Safety_Risk'] else 'No'}")

                                    if plan_data['AI_Compliance_Suggestions'] != 'N/A':
                                        st.info(f"🤖 **AI Compliance Alerts:**\n{plan_data['AI_Compliance_Suggestions']}")

                                    st.markdown("**Investigation Plan:**")
                                    st.markdown(plan_data['Investigation_Plan'][:500] + "..." if len(plan_data['Investigation_Plan']) > 500 else plan_data['Investigation_Plan'])

                            if len(bulk_plans) > 2:
                                st.info(f"+ {len(bulk_plans) - 2} more plans (see CSV/Excel export)")

                            # Enhanced Export options
                            st.markdown("---")
                            st.markdown("#### 📥 Export Investigation Plans")

                            plan_df = pd.DataFrame(bulk_plans)

                            col1, col2 = st.columns(2)

                            with col1:
                                csv_data = plan_df.to_csv(index=False)
                                st.download_button(
                                    "📥 Download All Plans (CSV)",
                                    csv_data,
                                    file_name=f"bulk_investigation_plans_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                    mime="text/csv",
                                    key="bulk_csv_download"
                                )

                            with col2:
                                # Excel export with multiple sheets
                                excel_buffer = io.BytesIO()
                                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                                    # Main plans
                                    plan_df.to_excel(writer, sheet_name='Investigation Plans', index=False)

                                    # Summary sheet
                                    summary_df = plan_df[['SKU', 'Product', 'Risk_Score', 'Severity', 'Markets_Analyzed', 'Estimated_Days']].copy()
                                    summary_df.to_excel(writer, sheet_name='Summary', index=False)

                                    # Regulatory sheet if applicable
                                    if selected_markets:
                                        reg_df = plan_df[['SKU', 'Product', 'Markets_Analyzed', 'Injury_Reporting_Requirements', 'AI_Compliance_Suggestions']].copy()
                                        reg_df.to_excel(writer, sheet_name='Regulatory Compliance', index=False)

                                st.download_button(
                                    "📥 Download Excel (Multi-Sheet)",
                                    excel_buffer.getvalue(),
                                    file_name=f"investigation_plans_complete_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key="bulk_excel_download"
                                )

                            # Regulatory compliance report if markets selected
                            if reg_analyzer and selected_markets:
                                st.markdown("---")
                                st.markdown("#### 📄 Regulatory Compliance Report")

                                products_data = [
                                    {
                                        'sku': p['SKU'],
                                        'product_name': p['Product'],
                                        'return_rate': float(p['Return_Rate'].rstrip('%')) / 100,
                                        'safety_risk': p['Safety_Risk']
                                    }
                                    for p in bulk_plans
                                ]

                                # Generate overall compliance report
                                compliance_report = reg_analyzer.generate_compliance_report(
                                    selected_markets,
                                    products_data,
                                    {'injury_reporting_required': [], 'ai_suggestions': []}  # Would aggregate all
                                )

                                with st.expander("📋 View Compliance Report", expanded=False):
                                    st.markdown(compliance_report)

                                st.download_button(
                                    "📥 Download Compliance Report (Markdown)",
                                    compliance_report,
                                    file_name=f"regulatory_compliance_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                                    mime="text/markdown",
                                    key="compliance_report_download"
                                )

                        except Exception as e:
                            st.error(f"Bulk plan generation failed: {e}")
                            logger.error(f"Bulk plan error: {e}", exc_info=True)

    # ========== END ADVANCED FEATURES ==========

    # ========== SMARTSHEET PROJECT PLANS ==========
    if len(action_items) > 0:
        st.markdown("---")
        st.markdown("### 📊 Smartsheet Project Plans")
        st.caption("Export ready-to-import project plans for Smartsheet project management")

        smartsheet_tab1, smartsheet_tab2, smartsheet_tab3 = st.tabs(["📋 CAPA Plan", "🚨 Critical Investigation", "🔧 Rework Operation"])

        # Import Smartsheet modules at runtime
        try:
            from smartsheet_plans import CAPAProjectPlan, CriticalInvestigationPlan, ReworkProjectPlan
            SMARTSHEET_AVAILABLE = True
        except ImportError:
            SMARTSHEET_AVAILABLE = False
            st.error("⚠️ Smartsheet plan module not available. Please ensure smartsheet_plans.py is in the directory.")

        if SMARTSHEET_AVAILABLE:
            # TAB 1: CAPA Project Plan
            with smartsheet_tab1:
                st.markdown("#### CAPA (Corrective & Preventive Action) Project Plan")
                st.caption("Based on 8D methodology - FDA/ISO 13485 compliant")

                col1, col2 = st.columns(2)

                with col1:
                    capa_sku = st.selectbox(
                        "Select Product for CAPA",
                        options=action_items['SKU'].unique(),
                        key="capa_sku"
                    )

                    capa_severity = st.selectbox(
                        "Severity Level",
                        ["Critical", "High", "Medium", "Low"],
                        index=1,
                        key="capa_severity",
                        help="Determines timeline urgency. Critical = fastest timeline"
                    )

                with col2:
                    capa_lead = st.text_input(
                        "Team Lead",
                        value="Quality Manager",
                        key="capa_lead"
                    )

                    capa_start = st.date_input(
                        "Start Date",
                        value=datetime.now(),
                        key="capa_start"
                    )

                if st.button("📊 Generate CAPA Project Plan", type="primary", key="gen_capa"):
                    with st.spinner("Generating CAPA project plan..."):
                        try:
                            row = action_items[action_items['SKU'] == capa_sku].iloc[0]

                            capa_plan = CAPAProjectPlan(
                                sku=capa_sku,
                                product_name=row.get('Name', capa_sku),
                                issue_description=row.get('Triggers', 'Quality concern identified'),
                                return_rate=row.get('Return_Rate', 0),
                                units_affected=int(row.get('Returned', 0)),
                                severity=capa_severity,
                                assigned_team_lead=capa_lead,
                                start_date=capa_start
                            )

                            st.success(f"✅ Generated CAPA plan with {len(capa_plan.tasks)} tasks!")

                            # Preview
                            preview_df = capa_plan.to_dataframe()
                            with st.expander("📋 Preview Plan", expanded=True):
                                st.dataframe(preview_df[['Task ID', 'Task Name', 'Assigned To', 'Duration (Days)', 'Status', 'Priority']],
                                           use_container_width=True, height=400)

                            # Download options
                            col1, col2 = st.columns(2)
                            with col1:
                                csv_data = capa_plan.to_csv()
                                st.download_button(
                                    "📥 Download CSV (Smartsheet Import)",
                                    csv_data,
                                    file_name=f"CAPA_{capa_sku}_{datetime.now().strftime('%Y%m%d')}.csv",
                                    mime="text/csv",
                                    help="✅ Import this CSV directly into Smartsheet"
                                )
                                st.success("✅ CSV ready for Smartsheet import!")

                            with col2:
                                excel_data = capa_plan.to_excel()
                                st.download_button(
                                    "📥 Download Excel",
                                    excel_data,
                                    file_name=f"CAPA_{capa_sku}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                                st.success("✅ Excel format available!")

                            # Statistics and Instructions
                            total_days = max([t['Duration (Days)'] for t in capa_plan.tasks])
                            st.info(f"📅 **Estimated Timeline:** {total_days} days | **Total Tasks:** {len(capa_plan.tasks)}")

                            with st.expander("📖 How to Import into Smartsheet"):
                                st.markdown("""
                                **Import Instructions:**
                                1. Download the CSV file above
                                2. Go to Smartsheet.com and create a new sheet
                                3. Click **File → Import → Microsoft Excel**
                                4. Select the downloaded CSV file
                                5. Smartsheet will automatically detect columns and structure
                                6. Click "Import" to complete

                                **Features Included:**
                                - ✅ Task hierarchy with dependencies
                                - ✅ Assigned team members
                                - ✅ Auto-calculated start/end dates
                                - ✅ Progress tracking (0%, 50%, 100%)
                                - ✅ Priority levels
                                - ✅ 8D methodology phases
                                """)

                        except Exception as e:
                            st.error(f"CAPA generation failed: {e}")
                            logger.error(f"CAPA error: {e}")

            # TAB 2: Critical Investigation
            with smartsheet_tab2:
                st.markdown("#### Critical Case Investigation Project Plan")
                st.caption("For safety concerns, regulatory issues, or high-impact quality problems")

                col1, col2 = st.columns(2)

                with col1:
                    crit_sku = st.selectbox(
                        "Select Product for Investigation",
                        options=action_items['SKU'].unique(),
                        key="crit_sku"
                    )

                    crit_severity = st.selectbox(
                        "Severity Level",
                        ["Critical", "High", "Medium"],
                        index=0,
                        key="crit_severity"
                    )

                with col2:
                    crit_safety = st.checkbox(
                        "⚠️ Safety Concern",
                        value=False,
                        key="crit_safety",
                        help="Check if there is potential patient/customer safety risk"
                    )

                    crit_regulatory = st.checkbox(
                        "📋 Regulatory Impact",
                        value=False,
                        key="crit_regulatory",
                        help="Check if FDA/MDR reporting may be required"
                    )

                crit_lead = st.text_input(
                    "Investigation Lead",
                    value="Quality Manager",
                    key="crit_lead"
                )

                if st.button("🚨 Generate Critical Investigation Plan", type="primary", key="gen_crit"):
                    with st.spinner("Generating critical investigation plan..."):
                        try:
                            row = action_items[action_items['SKU'] == crit_sku].iloc[0]

                            crit_plan = CriticalInvestigationPlan(
                                sku=crit_sku,
                                product_name=row.get('Name', crit_sku),
                                issue_description=row.get('Triggers', 'Critical quality issue'),
                                severity_level=crit_severity,
                                regulatory_impact=crit_regulatory,
                                safety_concern=crit_safety,
                                assigned_lead=crit_lead,
                                start_date=datetime.now()
                            )

                            st.success(f"✅ Generated critical investigation plan with {len(crit_plan.tasks)} tasks!")

                            if crit_safety:
                                st.warning("⚠️ **SAFETY CONCERN FLAGGED** - Plan includes immediate safety assessment and customer notification tasks")

                            if crit_regulatory:
                                st.warning("📋 **REGULATORY IMPACT FLAGGED** - Plan includes FDA/MDR notification and reporting tasks")

                            # Preview
                            preview_df = crit_plan.to_dataframe()
                            with st.expander("📋 Preview Plan", expanded=True):
                                st.dataframe(preview_df[['Task ID', 'Task Name', 'Assigned To', 'Duration (Days)', 'Status', 'Priority']],
                                           use_container_width=True, height=400)

                            # Download options
                            col1, col2 = st.columns(2)
                            with col1:
                                csv_data = crit_plan.to_csv()
                                st.download_button(
                                    "📥 Download CSV (Smartsheet Import)",
                                    csv_data,
                                    file_name=f"CRITICAL_INVESTIGATION_{crit_sku}_{datetime.now().strftime('%Y%m%d')}.csv",
                                    mime="text/csv"
                                )
                                st.success("✅ CSV ready for Smartsheet!")

                            with col2:
                                excel_data = crit_plan.to_excel()
                                st.download_button(
                                    "📥 Download Excel",
                                    excel_data,
                                    file_name=f"CRITICAL_INVESTIGATION_{crit_sku}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                                st.success("✅ Excel format available!")

                            # Timeline info
                            total_days = sum([t['Duration (Days)'] for t in crit_plan.tasks if t['Indent Level'] == 0])
                            st.info(f"⏱️ **Critical Timeline:** {total_days} days total | Phases include immediate 24-hour actions")

                        except Exception as e:
                            st.error(f"❌ Investigation plan generation failed: {str(e)}")
                            logger.error(f"Investigation error: {e}", exc_info=True)

            # TAB 3: Rework Operation (AI-Driven with Questions)
            with smartsheet_tab3:
                st.markdown("#### Rework Operation Project Plan")
                st.caption("AI-customized plan based on your specific rework requirements")

                # AI Questions for Rework
                st.markdown("**🤖 Answer questions to customize the rework plan:**")

                rework_sku = st.selectbox(
                    "Select Product for Rework",
                    options=action_items['SKU'].unique(),
                    key="rework_sku"
                )

                row = action_items[action_items['SKU'] == rework_sku].iloc[0]

                # Create form for rework questions
                with st.form("rework_questions_form"):
                    st.markdown("##### 📝 Rework Details Questionnaire")

                    col1, col2 = st.columns(2)

                    with col1:
                        rework_units = st.number_input(
                            "How many units need rework?",
                            min_value=1,
                            value=int(row.get('Returned', 100)),
                            step=1
                        )

                        rework_type = st.selectbox(
                            "What type of rework is required?",
                            [
                                "Component Replacement",
                                "Cosmetic Repair",
                                "Firmware Update",
                                "Label/Packaging Fix",
                                "Assembly Correction",
                                "Cleaning/Refinishing",
                                "Calibration/Adjustment",
                                "Other"
                            ]
                        )

                        complexity = st.selectbox(
                            "Rework complexity level?",
                            ["Low", "Medium", "High"]
                        )

                    with col2:
                        requires_disassembly = st.checkbox("Requires disassembly?", value=False)
                        requires_cleaning = st.checkbox("Requires cleaning?", value=False)
                        requires_testing = st.checkbox("Requires functional testing?", value=True)
                        requires_reassembly = st.checkbox("Requires reassembly?", value=False)

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        inspection_level = st.selectbox(
                            "Inspection level after rework?",
                            ["100%", "Sample", "Skip-lot"]
                        )

                    with col2:
                        batch_size = st.number_input("Batch size?", min_value=1, value=50, step=10)

                    with col3:
                        team_size = st.text_input("Team size?", value="2-3 operators")

                    requires_relabeling = st.checkbox("Requires relabeling?", value=False)
                    requires_repackaging = st.checkbox("Requires repackaging?", value=True)

                    required_materials = st.text_area(
                        "Required materials/parts:",
                        value="Standard rework materials"
                    )

                    rework_steps = st.text_area(
                        "Specific rework steps (optional):",
                        value="Follow approved rework procedure"
                    )

                    submitted = st.form_submit_button("🚀 Generate Rework Plan", type="primary")

                # Process form submission OUTSIDE the form context
                if submitted:
                    with st.spinner("Generating customized rework plan..."):
                        try:
                            rework_details = {
                                'batch_size': batch_size,
                                'complexity': complexity,
                                'inspection_level': inspection_level,
                                'team_size': team_size,
                                'requires_disassembly': requires_disassembly,
                                'requires_cleaning': requires_cleaning,
                                'requires_testing': requires_testing,
                                'requires_reassembly': requires_reassembly,
                                'requires_relabeling': requires_relabeling,
                                'requires_repackaging': requires_repackaging,
                                'required_materials': required_materials,
                                'rework_steps': rework_steps
                            }

                            rework_plan = ReworkProjectPlan(
                                sku=rework_sku,
                                product_name=row.get('Name', rework_sku),
                                units_to_rework=rework_units,
                                rework_type=rework_type,
                                rework_details=rework_details,
                                assigned_lead="Production Manager",
                                start_date=datetime.now()
                            )

                            # Store in session state so download buttons work
                            st.session_state['rework_plan'] = rework_plan
                            st.session_state['rework_generated'] = True

                            st.success(f"✅ Generated rework plan with {len(rework_plan.tasks)} tasks for {rework_units:,} units!")

                            # Summary
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Units to Rework", f"{rework_units:,}")
                            col2.metric("Rework Type", rework_type)
                            col3.metric("Complexity", complexity)

                            # Preview
                            preview_df = rework_plan.to_dataframe()
                            with st.expander("📋 Preview Plan", expanded=True):
                                st.dataframe(preview_df[['Task ID', 'Task Name', 'Assigned To', 'Duration (Days)', 'Status', 'Priority']],
                                           use_container_width=True, height=400)

                            # AI-customized info
                            st.info(f"🤖 **AI-Customized Plan:** Based on {complexity} complexity, {batch_size} units, {team_size} team members")

                        except Exception as e:
                            st.error(f"❌ Rework plan generation failed: {str(e)}")
                            logger.error(f"Rework error: {e}", exc_info=True)

                # Download buttons OUTSIDE form - displayed if plan exists
                if st.session_state.get('rework_generated', False) and 'rework_plan' in st.session_state:
                    st.markdown("---")
                    st.markdown("#### 📥 Download Rework Plan")

                    col1, col2 = st.columns(2)
                    rework_plan = st.session_state['rework_plan']

                    with col1:
                        csv_data = rework_plan.to_csv()
                        st.download_button(
                            "📥 Download CSV (Smartsheet)",
                            csv_data,
                            file_name=f"REWORK_{rework_sku}_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                            key="rework_csv_dl"
                        )

                    with col2:
                        excel_data = rework_plan.to_excel()
                        st.download_button(
                            "📥 Download Excel",
                            excel_data,
                            file_name=f"REWORK_{rework_sku}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="rework_excel_dl"
                        )

    # ========== END SMARTSHEET FEATURES ==========

    # Safety Disclaimer
    st.markdown("---")
    st.warning("""
    ⚠️ **Important Safety Notice**: Any safety concern or potential/confirmed injury requires a Quality Issue
    to be opened immediately in Odoo. This can be opened and closed same day as long as an investigation took place.
    Refer to Quality Incident Response SOP (QMS-SOP-001-9) for full procedures.
    """)
    
    # Methodology
    with st.expander("📐 Methodology & Math", expanded=False):
        st.markdown(generate_methodology_markdown())
    
    # Export Section - Google Sheets Tracker Compatible
    st.markdown("---")
    st.markdown("### 📥 Export for Team Tracker")
    
    st.info("""
    **Google Sheets Compatible Format**: Exports include all screening data plus blank columns for 
    **Current Status** and **Notes**. Copy/paste new rows directly into your team's quality tracker spreadsheet.
    """)
    
    # Define tracker-friendly column order
    tracker_columns = [
        'Screening_Date', 'Screened_By', 'Source_of_Flag', 
        'SKU', 'Name', 'Category',
        'Sold', 'Returned', 'Return_Rate', 'Category_Threshold',
        'Risk_Score', 'SPC_Signal', 'Action', 'Triggers',
        'Complaint_Text', 'Landed Cost', 'Safety Risk',
        'Threshold_Profile', 'Current_Status', 'Notes'
    ]
    
    # Prepare export dataframe
    export_df = df.copy()
    
    # Format percentages for readability
    if 'Return_Rate' in export_df.columns:
        export_df['Return_Rate'] = export_df['Return_Rate'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else '')
    if 'Category_Threshold' in export_df.columns:
        export_df['Category_Threshold'] = export_df['Category_Threshold'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else '')
    if 'Risk_Score' in export_df.columns:
        export_df['Risk_Score'] = export_df['Risk_Score'].apply(lambda x: f"{x:.0f}" if pd.notna(x) else '')
    
    # Ensure all tracker columns exist
    for col in tracker_columns:
        if col not in export_df.columns:
            export_df[col] = ''
    
    # Reorder to tracker format
    final_columns = [c for c in tracker_columns if c in export_df.columns]
    other_columns = [c for c in export_df.columns if c not in tracker_columns and c not in ['Risk_Components', 'SPC_Z_Score']]
    export_df = export_df[final_columns + other_columns]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # CSV - best for Google Sheets paste
        csv_data = export_df.to_csv(index=False)
        st.download_button(
            "📥 CSV (Google Sheets)",
            csv_data,
            file_name=f"quality_screening_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            help="Best for copy/paste into Google Sheets tracker",
            use_container_width=True
        )
    
    with col2:
        # Excel with formatting
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            export_df.to_excel(writer, index=False, sheet_name='Screening Results')
            
            # Add metadata sheet
            metadata = pd.DataFrame([
                ['Screening Date', st.session_state.screening_date],
                ['Screened By', st.session_state.screened_by],
                ['Source of Flag', st.session_state.source_of_flag],
                ['AI Provider', st.session_state.ai_provider.value],
                ['Threshold Profile', st.session_state.active_profile],
                ['Total Products', len(df)],
                ['Immediate Escalations', len(df[df['Action'].str.contains('Escalat', na=False)])],
                ['Quality Cases', len(df[df['Action'].str.contains('Case', na=False)])],
                ['Monitor Items', len(df[df['Action'].str.contains('Monitor', na=False)])],
                ['Export Time', datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
            ], columns=['Parameter', 'Value'])
            metadata.to_excel(writer, index=False, sheet_name='Metadata')
            
            # Format the results sheet
            workbook = writer.book
            worksheet = writer.sheets['Screening Results']
            
            # Header format
            header_fmt = workbook.add_format({
                'bold': True, 
                'bg_color': '#4FACFE', 
                'font_color': 'white',
                'border': 1
            })
            
            # Write formatted headers
            for col_num, value in enumerate(export_df.columns):
                worksheet.write(0, col_num, value, header_fmt)
                worksheet.set_column(col_num, col_num, 15)  # Set column width
        
        output.seek(0)
        st.download_button(
            "📥 Excel (Full Report)",
            output.getvalue(),
            file_name=f"quality_screening_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Full report with metadata sheet",
            use_container_width=True
        )
    
    with col3:
        # Clear results
        if st.button("🗑️ Clear Results", use_container_width=True):
            st.session_state.qc_results_df = None
            st.session_state.anova_result = None
            st.session_state.manova_result = None
            st.session_state.ai_chat_history = []
            st.rerun()


# --- Interactive Help Guide ---
def render_help_guide():
    """Render interactive help guide"""
    with st.expander("📚 Interactive Help Guide", expanded=False):
        st.markdown("""
        ### 🤖 AI-Powered Features

        **This application extensively uses AI/ML for quality management:**

        #### 1️⃣ **AI Return Categorization (Tab 1)**
        - **Technology:** OpenAI GPT-3.5 / Claude Haiku/Sonnet
        - **What it does:** Automatically categorizes customer complaints into standardized quality categories
        - **How it helps:** Eliminates manual categorization, processes thousands of complaints in minutes

        #### 2️⃣ **Multilingual Vendor Communications (Tab 3)**
        - **Technology:** AI-powered content generation + translation
        - **What it does:**
          - Generates professional vendor emails (CAPA requests, RCA requests)
          - Translates to 9 languages (Chinese, Spanish, Portuguese, Hindi, German, French, Italian)
          - Adjusts language complexity based on recipient's English proficiency (5 levels)
          - Adapts cultural communication style (China, India, LATAM, EU, USA)
        - **How it helps:** Communicate effectively with global supply chain partners

        #### 3️⃣ **Fuzzy Product Matching (Tab 3)**
        - **Technology:** AI semantic similarity + keyword matching
        - **What it does:**
          - Compares your products against 231 historical products
          - Finds similar products using fuzzy logic (e.g., "4-wheel scooter" matches "3-wheel scooter")
          - Benchmarks performance against similar historical products
        - **How it helps:** Understand if your return rates are normal compared to similar products

        #### 4️⃣ **Deep Dive Analysis (Tab 3)**
        - **Technology:** Claude Sonnet with document analysis
        - **What it does:**
          - Analyzes product manuals, specs, and listings
          - Recommends investigation methodologies (5 Whys, Fishbone, RCA, FMEA, 8D)
          - Identifies root cause patterns and risk levels
        - **How it helps:** AI reads your documents and suggests the best investigation approach

        #### 5️⃣ **AI Assistant Chat**
        - **Technology:** Conversational AI with quality management expertise
        - **What it does:** Answers questions about thresholds, results, risk scores, and quality processes
        - **How it helps:** On-demand expert guidance without searching through SOPs

        ---

        ### Quality Case Screening - Quick Start Guide

        #### Lite Mode (1-5 Products)
        1. Fill in **required fields**: Product Name, SKU, Category, Sales, Returns
        2. Add **complaint reasons** (comma-separated)
        3. Check **Safety Risk** if any safety concerns exist
        4. Click **Run AI Screening**
        
        #### Pro Mode (Mass Analysis)
        1. **Upload** your CSV/Excel file
        2. Review the **Data Validation Report**
        3. Check the **AI Analysis Suggestion**
        4. Optionally select a different analysis type
        5. Click **Run Full Screening Analysis**
        
        #### Understanding Results
        - **Risk Score**: 0-100 composite score (higher = more urgent)
        - **SPC Signal**: Statistical process control status
        - **Action**: Recommended next step based on SOPs
        
        #### Color Coding
        - 🔴 **Red**: Immediate escalation required
        - 🟠 **Orange**: Open Quality Case
        - 🟡 **Yellow**: Monitor closely
        - ⬜ **White**: No action required
        
        #### Required CSV Columns
        - `SKU` or `Product_SKU`
        - `Category`
        - `Sold` or `Units_Sold`
        - `Returned` or `Units_Returned`
        
        #### Optional Columns
        - `Name` or `Product_Name`
        - `Landed Cost` or `Cost`
        - `Complaint_Text` or `Complaints`
        - `Return_Rate` (calculated if not provided)
        """)
        
        # Example data download
        example_data = pd.DataFrame([
            {'SKU': 'MOB1027', 'Name': 'Knee Walker', 'Category': 'MOB', 'Sold': 1000, 'Returned': 120, 'Landed Cost': 85.00, 'Complaint_Text': 'Wheel squeaks, uncomfortable padding'},
            {'SKU': 'SUP1036', 'Name': 'Post Op Shoe', 'Category': 'SUP', 'Sold': 500, 'Returned': 45, 'Landed Cost': 12.00, 'Complaint_Text': 'Wrong size, poor fit'},
            {'SKU': 'LVA1004', 'Name': 'Pressure Mattress', 'Category': 'LVA', 'Sold': 800, 'Returned': 150, 'Landed Cost': 145.00, 'Complaint_Text': 'Pump failure, air leak'},
        ])

        csv_buffer = io.StringIO()
        example_data.to_csv(csv_buffer, index=False)

        col_ex1, col_ex2 = st.columns(2)

        with col_ex1:
            st.download_button(
                "📥 Download Simple Example (3 products)",
                csv_buffer.getvalue(),
                file_name="example_screening_data.csv",
                mime="text/csv",
                help="Basic example with 3 sample products"
            )

        with col_ex2:
            # Load advanced demo data if available
            try:
                demo_path = os.path.join(os.path.dirname(__file__), 'demo_quality_screening_data_advanced.csv')
                if os.path.exists(demo_path):
                    demo_df = pd.read_csv(demo_path)
                    demo_csv = demo_df.to_csv(index=False)
                    st.download_button(
                        "🚀 Download Advanced Demo Dataset",
                        demo_csv,
                        file_name="demo_quality_screening_advanced.csv",
                        mime="text/csv",
                        help="⭐ 70 real products with realistic quality issues - perfect for testing all AI features",
                        type="primary",
                        use_container_width=True
                    )
                    st.caption("✨ **Advanced Demo includes:** Products from actual catalog, realistic return scenarios, safety risks, multilingual support testing, fuzzy matching against 231 historical products")
            except Exception as e:
                pass  # Silently fail if advanced demo not available


def render_inventory_integration_tab():
    """
    Render the Inventory Integration tab.

    Integrates Odoo inventory data with B2B return reports to provide:
    - DOI calculations (planning and conservative views)
    - Reorder points and lead time windows
    - Corrective action windows
    - At-risk pipeline exposure
    - Integrated quality + inventory recommendations
    """
    st.markdown("### 📦 Inventory + Quality Integration")
    st.markdown("""
    <div style="background: rgba(0, 217, 255, 0.1); border: 1px solid #00D9FF;
                border-radius: 8px; padding: 1rem; margin-bottom: 1.5rem;">
        <strong>🎯 Purpose:</strong> Integrate inventory management with quality screening to determine:
        <ul style="margin: 0.5rem 0 0 1.5rem;">
            <li>Days of Inventory (DOI) - Planning & Conservative views</li>
            <li>Reorder points and timing windows</li>
            <li>Corrective action windows BEFORE reordering</li>
            <li>At-risk pipeline exposure (units & dollars)</li>
            <li>Integrated recommendations (quality + inventory + reorder)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state for inventory
    if 'inventory_data' not in st.session_state:
        st.session_state.inventory_data = None
    if 'inventory_config' not in st.session_state:
        st.session_state.inventory_config = InventoryConfiguration()
    if 'inventory_results' not in st.session_state:
        st.session_state.inventory_results = None

    # Create tabs for different sections
    inv_tab1, inv_tab2, inv_tab3, inv_tab4 = st.tabs([
        "📤 Data Upload",
        "⚙️ Configuration",
        "📊 Dashboard & Results",
        "🚨 Critical Integration View"
    ])

    # --- TAB 1: Data Upload ---
    with inv_tab1:
        st.markdown("#### Upload Inventory & Return Data")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### 📋 Odoo Inventory File")
            st.caption("Upload your Odoo Inventory Forecast export (Excel format)")
            odoo_file = st.file_uploader(
                "Odoo Inventory Forecast",
                type=['xlsx', 'xls'],
                key="odoo_upload",
                help="First data row contains headers: SKU, ASIN, Product Title, On Hand, DOI, etc."
            )

            if odoo_file:
                st.success("✅ Odoo file uploaded")
                with st.expander("ℹ️ Expected Odoo Columns"):
                    st.markdown("""
                    **Required columns:**
                    - SKU, ASIN, Product Title
                    - On Hand, On Order, Shipments in Transit, FBA Inbound, Total Units
                    - Total Daily rate, Unit Cost
                    - DOI, Warehouse DOI
                    - Status, Amazon Status
                    """)

        with col2:
            st.markdown("##### 📊 Pivot Return Report (Optional)")
            st.caption("Upload B2B returns data for enhanced analysis")
            pivot_file = st.file_uploader(
                "Pivot Return Report",
                type=['xlsx', 'xls'],
                key="pivot_upload",
                help="B2B returns with format: [SKU] Product Name"
            )

            if pivot_file:
                st.success("✅ Pivot return report uploaded")
                with st.expander("ℹ️ Pivot Report Format"):
                    st.markdown("""
                    **Expected format:**
                    - Hierarchical pivot structure
                    - SKU rows: `[SKU] Product Name`
                    - Return quantities by month (optional)
                    - **Note:** This is B2B returns only
                    """)

        st.markdown("---")

        # Process button
        if odoo_file:
            if st.button("🚀 Process Inventory Data", type="primary", use_container_width=True):
                with st.spinner("Processing inventory data..."):
                    try:
                        # Parse Odoo file
                        odoo_parser = OdooInventoryParser()
                        odoo_df = odoo_parser.parse_file(odoo_file)

                        st.success(f"✅ Parsed {len(odoo_df)} SKUs from Odoo")

                        # Parse Pivot Return Report if provided
                        returns_df = None
                        if pivot_file:
                            pivot_parser = PivotReturnReportParser()
                            returns_df = pivot_parser.parse_file(pivot_file)
                            st.success(f"✅ Parsed {len(returns_df)} SKUs from B2B returns")

                        # Calculate inventory metrics
                        calculator = InventoryCalculator(st.session_state.inventory_config)
                        results_df = calculator.calculate_inventory_metrics(odoo_df, returns_df)

                        # Store in session state
                        st.session_state.inventory_data = odoo_df
                        st.session_state.inventory_results = results_df

                        st.success("✅ Inventory calculations complete!")
                        st.info("📊 Switch to 'Dashboard & Results' tab to view analysis")

                    except Exception as e:
                        st.error(f"❌ Error processing files: {str(e)}")
                        logger.error(f"Inventory processing error: {str(e)}", exc_info=True)

    # --- TAB 2: Configuration ---
    with inv_tab2:
        st.markdown("#### ⚙️ Configuration Settings")
        st.caption("Set global defaults and per-SKU overrides for lead time and safety stock")

        config_tab1, config_tab2 = st.tabs(["🌐 Global Defaults", "📋 Per-SKU Config"])

        # Global defaults
        with config_tab1:
            st.markdown("##### Global Default Values")
            st.caption("These apply to all SKUs unless overridden")

            col1, col2 = st.columns(2)

            with col1:
                global_lead_time = st.number_input(
                    "Lead Time (days)",
                    min_value=1,
                    max_value=365,
                    value=st.session_state.inventory_config.global_lead_time_days,
                    help="Total time from PO to stock availability"
                )

            with col2:
                global_safety_stock = st.number_input(
                    "Safety Stock (days)",
                    min_value=0,
                    max_value=90,
                    value=st.session_state.inventory_config.global_safety_stock_days,
                    help="Buffer inventory in days"
                )

            if st.button("💾 Update Global Defaults"):
                st.session_state.inventory_config.global_lead_time_days = global_lead_time
                st.session_state.inventory_config.global_safety_stock_days = global_safety_stock
                st.success("✅ Global defaults updated")

                # Recalculate if data exists
                if st.session_state.inventory_data is not None:
                    calculator = InventoryCalculator(st.session_state.inventory_config)
                    st.session_state.inventory_results = calculator.calculate_inventory_metrics(
                        st.session_state.inventory_data
                    )
                    st.info("📊 Results recalculated with new defaults")

        # Per-SKU configuration
        with config_tab2:
            st.markdown("##### Per-SKU Configuration Upload")
            st.caption("Upload CSV with SKU-specific lead times and safety stock")

            sku_config_file = st.file_uploader(
                "Upload SKU Config (CSV)",
                type=['csv'],
                key="sku_config_upload",
                help="Required columns: SKU, LeadTimeDays, SafetyStockDays"
            )

            if sku_config_file:
                try:
                    st.session_state.inventory_config.load_sku_config(sku_config_file)
                    st.success(f"✅ Loaded SKU-specific configurations")

                    # Recalculate if data exists
                    if st.session_state.inventory_data is not None:
                        calculator = InventoryCalculator(st.session_state.inventory_config)
                        st.session_state.inventory_results = calculator.calculate_inventory_metrics(
                            st.session_state.inventory_data
                        )
                        st.info("📊 Results recalculated with SKU configs")

                except Exception as e:
                    st.error(f"❌ Error loading SKU config: {str(e)}")

            with st.expander("📄 Download Template"):
                st.markdown("**Expected CSV format:**")
                template_df = pd.DataFrame({
                    'SKU': ['MOB-2847', 'SUP-5621', 'INS-3421'],
                    'LeadTimeDays': [45, 30, 60],
                    'SafetyStockDays': [14, 7, 21]
                })
                st.dataframe(template_df, use_container_width=True)

                csv_buffer = io.StringIO()
                template_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    "📥 Download Template",
                    csv_buffer.getvalue(),
                    file_name="sku_config_template.csv",
                    mime="text/csv"
                )

    # --- TAB 3: Dashboard & Results ---
    with inv_tab3:
        if st.session_state.inventory_results is None:
            st.info("📤 Upload and process inventory data in the 'Data Upload' tab first")
            return

        results_df = st.session_state.inventory_results

        st.markdown("#### 📊 Inventory Analysis Dashboard")

        # Summary KPIs
        st.markdown("##### 📈 Summary Metrics")
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

        with kpi_col1:
            total_skus = len(results_df)
            st.metric("Total SKUs", f"{total_skus:,}")

        with kpi_col2:
            past_reorder = len(results_df[results_df['DaysToReorder'] < 0])
            st.metric("⚠️ Past Reorder Point", past_reorder,
                     delta=None if past_reorder == 0 else "Action Needed",
                     delta_color="inverse")

        with kpi_col3:
            reorder_soon = len(results_df[(results_df['DaysToReorder'] >= 0) &
                                         (results_df['DaysToReorder'] < 14)])
            st.metric("🟡 Reorder Soon (<14d)", reorder_soon)

        with kpi_col4:
            total_at_risk = results_df['AtRiskDollars'].sum()
            st.metric("💰 At-Risk Pipeline", f"${total_at_risk:,.0f}")

        st.markdown("---")

        # Filters
        st.markdown("##### 🔍 Filters")
        filter_col1, filter_col2, filter_col3 = st.columns(3)

        with filter_col1:
            status_filter = st.multiselect(
                "Status",
                options=['Past Reorder', 'Reorder Soon', 'Healthy'],
                default=['Past Reorder', 'Reorder Soon']
            )

        with filter_col2:
            category_filter = st.multiselect(
                "Category",
                options=['All'] + list(results_df['Product Title'].str[:3].unique()),
                default=['All']
            )

        with filter_col3:
            sort_by = st.selectbox(
                "Sort By",
                options=['DaysToReorder', 'DOI_Conservative', 'AtRiskDollars', 'SKU'],
                index=0
            )

        # Apply filters
        filtered_df = results_df.copy()

        # Status filter
        if status_filter:
            status_conditions = []
            if 'Past Reorder' in status_filter:
                status_conditions.append(filtered_df['DaysToReorder'] < 0)
            if 'Reorder Soon' in status_filter:
                status_conditions.append((filtered_df['DaysToReorder'] >= 0) &
                                        (filtered_df['DaysToReorder'] < 14))
            if 'Healthy' in status_filter:
                status_conditions.append(filtered_df['DaysToReorder'] >= 14)

            if status_conditions:
                combined_condition = status_conditions[0]
                for cond in status_conditions[1:]:
                    combined_condition = combined_condition | cond
                filtered_df = filtered_df[combined_condition]

        # Sort
        filtered_df = filtered_df.sort_values(by=sort_by)

        st.markdown("---")
        st.markdown(f"##### 📋 Results ({len(filtered_df)} SKUs)")

        # Display results table
        display_columns = [
            'SKU', 'Product Title', 'On Hand', 'Total Units',
            'Total Daily rate', 'DOI_Planning', 'DOI_Conservative',
            'DaysToReorder', 'MustOrderBy', 'CA_Window_BeforePO',
            'AtRiskUnits', 'AtRiskDollars'
        ]

        display_df = filtered_df[display_columns].copy()

        # Format numeric columns
        display_df['DOI_Planning'] = display_df['DOI_Planning'].apply(
            lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
        )
        display_df['DOI_Conservative'] = display_df['DOI_Conservative'].apply(
            lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
        )
        display_df['DaysToReorder'] = display_df['DaysToReorder'].apply(
            lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
        )
        display_df['CA_Window_BeforePO'] = display_df['CA_Window_BeforePO'].apply(
            lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
        )
        display_df['AtRiskDollars'] = display_df['AtRiskDollars'].apply(
            lambda x: f"${x:,.0f}"
        )

        st.dataframe(display_df, use_container_width=True, height=600)

        # Export options
        st.markdown("---")
        st.markdown("##### 📤 Export Options")

        export_col1, export_col2 = st.columns(2)

        with export_col1:
            # CSV export
            csv_buffer = io.StringIO()
            filtered_df.to_csv(csv_buffer, index=False)
            st.download_button(
                "📥 Export to CSV",
                csv_buffer.getvalue(),
                file_name=f"inventory_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with export_col2:
            # Excel export (if available)
            if EXCEL_AVAILABLE:
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                    filtered_df.to_excel(writer, sheet_name='Inventory Analysis', index=False)

                st.download_button(
                    "📥 Export to Excel",
                    excel_buffer.getvalue(),
                    file_name=f"inventory_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

    # --- TAB 4: Critical Integration View ---
    with inv_tab4:
        st.markdown("### 🚨 Critical Integration View: Quality + Inventory + Reorder")
        st.caption("Unified view showing SKUs with BOTH quality issues AND inventory urgency")

        # Prompt to use quality screening data
        if st.session_state.qc_results_df is None or st.session_state.qc_results_df.empty:
            st.info("""
            💡 **Tip: Integrate Quality Screening Data**

            This view combines quality issues with inventory urgency to help you answer:
            - "Can we fix quality issues BEFORE reordering?"
            - "Which problems are most urgent financially?"
            - "Do we have time for corrective action?"

            **To enable this view:**
            1. Go to **Tab 3: Quality Screening**
            2. Run screening (Lite, Pro, or Quick Eval modes)
            3. Return here to see integrated analysis

            The tool will automatically merge quality flags with your inventory data!
            """)

        # Check if we have both quality and inventory data
        has_quality_data = st.session_state.qc_results_df is not None and not st.session_state.qc_results_df.empty
        has_inventory_data = st.session_state.inventory_results is not None and not st.session_state.inventory_results.empty

        if not has_quality_data and not has_inventory_data:
            st.info("""
            📊 **No data available yet**

            To use the Critical Integration View:
            1. Run quality screening in **Tab 3: Quality Screening**
            2. Upload inventory data in **Tab 4: Inventory Integration → Data Upload**

            This view will then show products with BOTH quality flags AND inventory urgency.
            """)
            return

        if not has_quality_data:
            st.warning("⚠️ Missing quality screening data. Please run analysis in Tab 3 first.")
            return

        if not has_inventory_data:
            st.warning("⚠️ Missing inventory data. Please upload Odoo data in the 'Data Upload' tab first.")
            return

        # Merge quality and inventory data
        quality_df = st.session_state.qc_results_df.copy()
        inventory_df = st.session_state.inventory_results.copy()

        # Try to merge on SKU column (handle different naming)
        sku_col_quality = None
        for col in ['SKU', 'Main SKU', 'sku', 'Product_ID']:
            if col in quality_df.columns:
                sku_col_quality = col
                break

        sku_col_inventory = 'SKU'  # Standard from Odoo parser

        if sku_col_quality is None:
            st.error("Could not find SKU column in quality data")
            return

        # Merge datasets
        try:
            merged_df = quality_df.merge(
                inventory_df,
                left_on=sku_col_quality,
                right_on=sku_col_inventory,
                how='inner',
                suffixes=('_quality', '_inventory')
            )

            if merged_df.empty:
                st.warning("⚠️ No matching SKUs found between quality screening and inventory data")
                return

            # Calculate integrated priority scoring
            merged_df['IntegratedPriority'] = 0

            # Quality factors
            if 'Return_Rate' in merged_df.columns:
                merged_df['IntegratedPriority'] += merged_df['Return_Rate'] * 100

            if 'Risk_Score' in merged_df.columns:
                merged_df['IntegratedPriority'] += merged_df['Risk_Score'] / 10

            # Inventory urgency factors
            if 'DaysToReorder' in merged_df.columns:
                # More urgent = higher priority
                merged_df['IntegratedPriority'] += merged_df['DaysToReorder'].apply(
                    lambda x: 50 if pd.isna(x) or x < 0 else (30 if x < 7 else (15 if x < 14 else 0))
                )

            # Financial exposure
            if 'AtRiskDollars' in merged_df.columns:
                max_risk = merged_df['AtRiskDollars'].max()
                if max_risk > 0:
                    merged_df['IntegratedPriority'] += (merged_df['AtRiskDollars'] / max_risk) * 20

            # Assign urgency classification
            def classify_urgency(row):
                days = row.get('DaysToReorder', None)
                return_rate = row.get('Return_Rate', 0)
                risk = row.get('Risk_Score', 0)

                # Critical: Past reorder + quality issue
                if pd.notna(days) and days < 0 and (return_rate > 0.10 or risk > 70):
                    return '🔴 CRITICAL'
                # High: <7 days + quality issue
                elif pd.notna(days) and days < 7 and (return_rate > 0.08 or risk > 60):
                    return '🟠 HIGH'
                # Medium: 7-14 days + quality issue
                elif pd.notna(days) and days < 14 and (return_rate > 0.05 or risk > 50):
                    return '🟡 MEDIUM'
                # Low: 14-30 days buffer
                elif pd.notna(days) and days < 30:
                    return '🟢 LOW'
                else:
                    return '⚪ MONITOR'

            merged_df['UrgencyClass'] = merged_df.apply(classify_urgency, axis=1)

            # Generate recommendations
            def generate_recommendation(row):
                days = row.get('DaysToReorder', None)
                ca_before_po = row.get('CA_Window_BeforePO', None)
                urgency = row.get('UrgencyClass', '')

                if '🔴 CRITICAL' in urgency:
                    return "URGENT: Order now + expedite fix OR find substitute - past reorder point"
                elif '🟠 HIGH' in urgency:
                    if pd.notna(ca_before_po) and ca_before_po > 3:
                        return f"Expedite investigation - {ca_before_po:.0f} days to fix before PO"
                    else:
                        return "Critical window - order with known issue + fix in parallel"
                elif '🟡 MEDIUM' in urgency:
                    if pd.notna(ca_before_po) and ca_before_po > 14:
                        return f"Plan correction - {ca_before_po:.0f} days available before reorder"
                    else:
                        return "Monitor closely - investigate and track progress"
                else:
                    return "Continue monitoring - sufficient buffer time"

            merged_df['Recommendation'] = merged_df.apply(generate_recommendation, axis=1)

            # Sort by priority
            merged_df = merged_df.sort_values('IntegratedPriority', ascending=False)

            # Summary metrics
            st.markdown("#### 📊 Integration Summary")
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)

            with kpi1:
                total_integrated = len(merged_df)
                st.metric("Total SKUs", f"{total_integrated:,}", help="Products with both quality and inventory data")

            with kpi2:
                critical_count = len(merged_df[merged_df['UrgencyClass'] == '🔴 CRITICAL'])
                st.metric("🔴 Critical", critical_count,
                         delta="Action in 0-3 days" if critical_count > 0 else None,
                         delta_color="inverse")

            with kpi3:
                high_count = len(merged_df[merged_df['UrgencyClass'] == '🟠 HIGH'])
                st.metric("🟠 High", high_count,
                         delta="Action in 4-7 days" if high_count > 0 else None,
                         delta_color="inverse")

            with kpi4:
                total_at_risk = merged_df['AtRiskDollars'].sum() if 'AtRiskDollars' in merged_df.columns else 0
                st.metric("💰 At Risk", f"${total_at_risk:,.0f}", help="Total pipeline exposure")

            st.markdown("---")

            # Filters
            st.markdown("#### 🔍 Filters")
            filter_col1, filter_col2 = st.columns(2)

            with filter_col1:
                urgency_filter = st.multiselect(
                    "Urgency Level",
                    options=['🔴 CRITICAL', '🟠 HIGH', '🟡 MEDIUM', '🟢 LOW', '⚪ MONITOR'],
                    default=['🔴 CRITICAL', '🟠 HIGH'],
                    help="Filter by urgency classification"
                )

            with filter_col2:
                min_priority = st.slider(
                    "Minimum Priority Score",
                    min_value=0,
                    max_value=int(merged_df['IntegratedPriority'].max()),
                    value=0,
                    help="Show only products above this priority threshold"
                )

            # Apply filters
            filtered_integrated = merged_df.copy()
            if urgency_filter:
                filtered_integrated = filtered_integrated[
                    filtered_integrated['UrgencyClass'].isin(urgency_filter)
                ]
            filtered_integrated = filtered_integrated[
                filtered_integrated['IntegratedPriority'] >= min_priority
            ]

            st.markdown(f"#### 📋 Critical Items ({len(filtered_integrated)} SKUs)")

            if filtered_integrated.empty:
                st.success("✅ No critical items match current filters")
            else:
                # Display critical items
                for idx, row in filtered_integrated.iterrows():
                    urgency_colors = {
                        '🔴 CRITICAL': '#dc2626',
                        '🟠 HIGH': '#f59e0b',
                        '🟡 MEDIUM': '#fbbf24',
                        '🟢 LOW': '#10b981',
                        '⚪ MONITOR': '#6b7280'
                    }

                    urgency_color = urgency_colors.get(row['UrgencyClass'], '#6b7280')

                    with st.expander(
                        f"{row['UrgencyClass']} | {row.get(sku_col_quality, 'Unknown')} - "
                        f"{row.get('Product Name', row.get('Product_Name', row.get('Product Title', 'Unknown')))}",
                        expanded=(row['UrgencyClass'] in ['🔴 CRITICAL', '🟠 HIGH'])
                    ):
                        st.markdown(f"""
                        <div style="background: {urgency_color}; color: white; padding: 0.75rem;
                                    border-radius: 8px; margin-bottom: 1rem;">
                            <strong>Priority Score: {row['IntegratedPriority']:.1f}</strong>
                        </div>
                        """, unsafe_allow_html=True)

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.markdown("**Quality Metrics**")
                            return_rate = row.get('Return_Rate', 0) * 100 if row.get('Return_Rate', 0) < 1 else row.get('Return_Rate', 0)
                            st.metric("Return Rate", f"{return_rate:.2f}%")
                            if 'Risk_Score' in row:
                                st.metric("Risk Score", f"{row['Risk_Score']:.0f}")
                            if 'Action_Required' in row:
                                st.write(f"**Action:** {row['Action_Required']}")

                        with col2:
                            st.markdown("**Inventory Status**")
                            if 'DaysToReorder' in row and pd.notna(row['DaysToReorder']):
                                st.metric("Days to Reorder", f"{row['DaysToReorder']:.1f}")
                            if 'DOI_Conservative' in row and pd.notna(row['DOI_Conservative']):
                                st.metric("DOI (Conservative)", f"{row['DOI_Conservative']:.1f}")
                            if 'MustOrderBy' in row:
                                st.write(f"**Must Order By:** {row['MustOrderBy']}")

                        with col3:
                            st.markdown("**Financial Impact**")
                            if 'AtRiskDollars' in row:
                                st.metric("At Risk", f"${row['AtRiskDollars']:,.0f}")
                            if 'AtRiskUnits' in row:
                                st.metric("Units at Risk", f"{row['AtRiskUnits']:,}")

                        # Recommendation box
                        st.markdown(f"""
                        <div style="background: #f0f9ff; border-left: 4px solid #0284c7;
                                    padding: 1rem; margin-top: 1rem; border-radius: 4px;">
                            <strong>📋 Recommendation:</strong><br>
                            {row['Recommendation']}
                        </div>
                        """, unsafe_allow_html=True)

                        # Action buttons
                        action_col1, action_col2, action_col3 = st.columns(3)

                        with action_col1:
                            if st.button(f"📧 Email Vendor", key=f"email_{idx}"):
                                st.info("Vendor email generator feature - navigate to Tab 3 Quality Screening")

                        with action_col2:
                            if st.button(f"📋 Generate CAPA", key=f"capa_{idx}"):
                                st.info("CAPA generator feature - navigate to Tab 3 Quality Screening")

                        with action_col3:
                            if st.button(f"🔍 Deep Dive", key=f"deep_{idx}"):
                                st.info("Deep dive analysis feature - navigate to Tab 3 Quality Screening")

                # Export integrated view
                st.markdown("---")
                st.markdown("#### 📤 Export Integrated View")

                export_cols = [sku_col_quality, 'Product Name', 'UrgencyClass', 'IntegratedPriority',
                              'Return_Rate', 'Risk_Score', 'DaysToReorder', 'DOI_Conservative',
                              'AtRiskDollars', 'Recommendation']

                export_df = filtered_integrated[[col for col in export_cols if col in filtered_integrated.columns]]

                csv_buffer = io.StringIO()
                export_df.to_csv(csv_buffer, index=False)

                st.download_button(
                    "📥 Export Critical Integration View (CSV)",
                    csv_buffer.getvalue(),
                    file_name=f"critical_integration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"Error merging data: {str(e)}")
            logger.error(f"Integration view error: {str(e)}", exc_info=True)


# --- EXTRACTED TOOL FUNCTIONS (For Task-Based Navigation) ---

def render_categorizer_tool(provider_map=None, provider_selection=None):
    """Render the Return Categorizer tool (Tab 1 content)"""
    # Set AI provider if provided
    if provider_map and provider_selection:
        st.session_state.ai_provider = provider_map[provider_selection]

    st.markdown("### 📁 AI-Powered Return Categorization (Column I → K)")
    st.markdown("""
    <div style="background: rgba(255, 183, 0, 0.1); border: 1px solid var(--accent);
                border-radius: 8px; padding: 0.8rem; margin-bottom: 1rem;">
        <strong>🤖 AI-Powered:</strong> Uses OpenAI/Claude LLMs to automatically categorize customer complaints<br/>
        <strong>📌 Goal:</strong> Convert unstructured complaint text into standardized Quality Categories<br/>
        <strong>⚡ Speed:</strong> Processes thousands of complaints in minutes (vs hours manually)
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Return Data (Excel/CSV)", type=['csv', 'xlsx', 'xls', 'txt'], key="tab1_uploader")

    if uploaded_file:
        with st.spinner(f"Reading {uploaded_file.name}..."):
            file_content = uploaded_file.read()
            df, column_mapping = process_file_preserve_structure(file_content, uploaded_file.name)

        if df is not None and column_mapping:
            st.session_state.column_mapping = column_mapping

            complaint_col = column_mapping.get('complaint')
            if complaint_col:
                valid_complaints = df[df[complaint_col].notna() & (df[complaint_col].str.strip() != '')].shape[0]
                st.info(f"Found {valid_complaints:,} complaints to categorize in Column I.")
            else:
                st.warning("Complaint column not found in expected position.")

            if st.button("🚀 Start Categorization", type="primary"):
                analyzer = get_ai_analyzer()
                with st.spinner("Categorizing..."):
                    categorized_df = process_in_chunks(df, analyzer, column_mapping)
                    st.session_state.categorized_data = categorized_df
                    st.session_state.processing_complete = True
                    generate_statistics(categorized_df, column_mapping)

                    st.session_state.export_data = export_with_column_k(categorized_df)
                    st.session_state.export_filename = f"categorized_{datetime.now().strftime('%Y%m%d')}.xlsx"
                    st.rerun()

    if st.session_state.processing_complete and st.session_state.categorized_data is not None:
        display_results_dashboard(st.session_state.categorized_data, st.session_state.column_mapping)

        if st.session_state.export_data:
            st.download_button(
                label="⬇️ Download Categorized File",
                data=st.session_state.export_data,
                file_name=st.session_state.export_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                use_container_width=True
            )


def render_b2b_tool(provider_map=None, provider_selection=None):
    """Render the B2B Report Generator tool (Tab 2 content)"""
    # Set AI provider if provided
    if provider_map and provider_selection:
        st.session_state.ai_provider = provider_map[provider_selection]

    st.markdown("### 📑 B2B Report Automation")
    st.markdown("""
    <div style="background: rgba(0, 217, 255, 0.1); border: 1px solid var(--primary);
                border-radius: 8px; padding: 0.8rem; margin-bottom: 1rem;">
        <strong>📌 Goal:</strong> Convert raw Odoo Helpdesk export into a compliant B2B Report.
        <ul style="margin-bottom:0;">
            <li><strong>Format:</strong> Matches standard B2B Report columns (Display Name, Description, SKU, Reason)</li>
            <li><strong>SKU Logic:</strong> Auto-extracts Main SKU (e.g., <code>MOB1027</code>)</li>
            <li><strong>AI Summary:</strong> Generates detailed Reason summaries for every ticket.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### ⚙️ Data Volume / Processing Speed")
    perf_mode = st.select_slider(
        "Select Dataset Size to optimize API performance:",
        options=['Small (< 500 rows)', 'Medium (500-2,000 rows)', 'Large (2,000+ rows)'],
        value=st.session_state.b2b_perf_mode,
        key='perf_selector'
    )
    st.session_state.b2b_perf_mode = perf_mode

    if perf_mode == 'Small (< 500 rows)':
        batch_size = 10
        max_workers = 3
        st.caption("Settings: Conservative batching for max reliability.")
    elif perf_mode == 'Medium (500-2,000 rows)':
        batch_size = 25
        max_workers = 6
        st.caption("Settings: Balanced speed and concurrency.")
    else:
        batch_size = 50
        max_workers = 10
        st.caption("Settings: Aggressive parallel processing for high volume.")

    st.divider()

    b2b_file = st.file_uploader("Upload Odoo Export (CSV/Excel)", type=['csv', 'xlsx'], key="b2b_uploader")

    if b2b_file:
        b2b_df = process_b2b_file(b2b_file.read(), b2b_file.name)

        if b2b_df is not None:
            st.markdown(f"**Total Tickets Found:** {len(b2b_df):,}")

            if st.button("⚡ Generate B2B Report", type="primary"):
                analyzer = get_ai_analyzer(max_workers=max_workers)

                with st.spinner("Running AI Analysis & SKU Extraction..."):
                    final_b2b = generate_b2b_report(b2b_df, analyzer, batch_size)

                    st.session_state.b2b_processed_data = final_b2b
                    st.session_state.b2b_processing_complete = True

                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        final_b2b.to_excel(writer, index=False, sheet_name='B2B Report')

                        workbook = writer.book
                        worksheet = writer.sheets['B2B Report']

                        header_fmt = workbook.add_format({'bold': True, 'bg_color': '#00D9FF', 'font_color': 'white'})
                        for col_num, value in enumerate(final_b2b.columns.values):
                            worksheet.write(0, col_num, value, header_fmt)
                            worksheet.set_column(col_num, col_num, 30)

                    st.session_state.b2b_export_data = output.getvalue()
                    st.session_state.b2b_export_filename = f"B2B_Report_{datetime.now().strftime('%Y-%m-%d')}.xlsx"

                    st.rerun()

    if st.session_state.b2b_processing_complete and st.session_state.b2b_processed_data is not None:
        df_res = st.session_state.b2b_processed_data

        st.markdown("### 🏁 Report Dashboard")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Processed", len(df_res))
        with c2:
            sku_found_count = len(df_res[df_res['SKU'] != 'Unknown'])
            st.metric("SKUs Identified", f"{sku_found_count}", delta=f"{sku_found_count/len(df_res)*100:.1f}% coverage")
        with c3:
            unique_skus = df_res[df_res['SKU'] != 'Unknown']['SKU'].nunique()
            st.metric("Unique Products", unique_skus)

        st.markdown("#### Preview (Top 10)")
        st.dataframe(df_res.head(10), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="⬇️ Download B2B Report (.xlsx)",
                data=st.session_state.b2b_export_data,
                file_name=st.session_state.b2b_export_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                use_container_width=True
            )
        with col2:
            if st.button("🔄 Clear / Start Over", use_container_width=True):
                st.session_state.b2b_processed_data = None
                st.session_state.b2b_processing_complete = False
                st.rerun()


# --- TASK SELECTOR (Landing Page) ---

TASK_DEFINITIONS = {
    'categorize': {
        'icon': '📊',
        'title': 'Categorize Returns',
        'subtitle': 'AI-Powered Complaint Analysis',
        'description': 'Upload Amazon/customer return data and let AI categorize complaints into quality categories.',
        'keywords': ['categorize', 'complaints', 'returns', 'amazon', 'categories', 'column k'],
    },
    'b2b': {
        'icon': '📑',
        'title': 'Generate B2B Report',
        'subtitle': 'Odoo → B2B Format',
        'description': 'Convert Odoo Helpdesk exports into compliant B2B reports with AI-generated summaries.',
        'keywords': ['b2b', 'report', 'odoo', 'helpdesk', 'sku'],
    },
    'tracker': {
        'icon': '📋',
        'title': 'Quality Case Tracker',
        'subtitle': 'Manage & Export Cases',
        'description': 'Track quality cases, import from Smartsheet, and export with Leadership or Company-Wide formats.',
        'keywords': ['tracker', 'cases', 'smartsheet', 'export', 'leadership', 'track'],
    },
    'screening': {
        'icon': '🧪',
        'title': 'Screen Products',
        'subtitle': 'Flag Quality Issues',
        'description': 'Screen products for quality issues using AI analysis, statistical methods, and SOP thresholds.',
        'keywords': ['screen', 'screening', 'quality', 'flag', 'sop', 'threshold', 'anova'],
    },
    'inventory': {
        'icon': '📦',
        'title': 'Inventory Analysis',
        'subtitle': 'DOI & Reorder Planning',
        'description': 'Analyze Days of Inventory, reorder points, and integrate quality data with inventory planning.',
        'keywords': ['inventory', 'doi', 'reorder', 'days of inventory', 'stock'],
    },
    'resources': {
        'icon': '📚',
        'title': 'Resources',
        'subtitle': 'Regulatory Links & Guides',
        'description': 'Access FDA, EU MDR, UK MDR, and international regulatory resources and quality guides.',
        'keywords': ['resources', 'fda', 'regulatory', 'links', 'mdr', 'iso'],
    },
    'recalls': {
        'icon': '🌐',
        'title': 'Global Recall Surveillance',
        'subtitle': 'Worldwide Regulatory Intelligence',
        'description': 'Scan FDA, EU EMA, UK MHRA, Health Canada, ANVISA, CPSC, and global media for recalls, alerts, and safety signals affecting your products.',
        'keywords': ['recall', 'recalls', 'surveillance', 'fda', 'mhra', 'ema', 'health canada', 'anvisa', 'cpsc', 'maude', 'adverse', 'alert', 'safety', 'global', 'worldwide'],
    },
}

def match_task_from_input(user_input: str) -> str:
    """Match user text input to a task ID using keywords"""
    user_input = user_input.lower().strip()

    # Check for "all" first
    if user_input in ['all', 'show all', 'everything']:
        return 'all'

    # Score each task by keyword matches
    scores = {}
    for task_id, task_info in TASK_DEFINITIONS.items():
        score = 0
        for keyword in task_info['keywords']:
            if keyword in user_input:
                score += 1
        if score > 0:
            scores[task_id] = score

    # Return highest scoring task, or None if no match
    if scores:
        return max(scores, key=scores.get)
    return None

def render_task_selector():
    """Render the task selector landing page"""

    st.markdown("""
    <div style="text-align: center; margin: 1.5rem 0 2rem 0;">
        <h2 style="color: #004366; font-family: 'Poppins', sans-serif; margin-bottom: 0.5rem;">
            📋 What are you trying to do?
        </h2>
        <p style="color: #666; font-size: 1rem;">Select a tool below or type what you need</p>
    </div>
    """, unsafe_allow_html=True)

    # Task cards - 3x2 grid + Featured surveillance tool
    row1 = st.columns(3)
    row2 = st.columns(3)

    tasks_row1 = ['categorize', 'b2b', 'tracker']
    tasks_row2 = ['screening', 'inventory', 'resources']

    def render_task_card(col, task_id, featured=False):
        """Render a single task card"""
        task = TASK_DEFINITIONS[task_id]
        with col:
            # Card container - featured cards have special styling
            if featured:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(0,67,102,0.15) 0%, rgba(35,178,190,0.2) 100%);
                            border: 2px solid rgba(0,67,102,0.5); border-radius: 12px; padding: 1rem;
                            margin-bottom: 0.5rem; min-height: 120px; box-shadow: 0 4px 12px rgba(0,67,102,0.15);">
                    <div style="font-size: 2rem; text-align: center; margin-bottom: 0.3rem;">{task['icon']}</div>
                    <div style="font-weight: 700; color: #004366; text-align: center; font-size: 1rem;">{task['title']}</div>
                    <div style="color: #23b2be; text-align: center; font-size: 0.8rem; margin-bottom: 0.3rem; font-weight: 500;">{task['subtitle']}</div>
                    <div style="color: #666; text-align: center; font-size: 0.7rem; margin-top: 0.3rem;">FDA • EMA • MHRA • Health Canada • CPSC • Media</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(35,178,190,0.05) 0%, rgba(0,67,102,0.08) 100%);
                            border: 1px solid rgba(35,178,190,0.3); border-radius: 12px; padding: 1rem;
                            margin-bottom: 0.5rem; min-height: 120px;">
                    <div style="font-size: 2rem; text-align: center; margin-bottom: 0.3rem;">{task['icon']}</div>
                    <div style="font-weight: 600; color: #004366; text-align: center; font-size: 0.95rem;">{task['title']}</div>
                    <div style="color: #23b2be; text-align: center; font-size: 0.8rem; margin-bottom: 0.3rem;">{task['subtitle']}</div>
                </div>
                """, unsafe_allow_html=True)

            btn_type = "primary" if featured else "secondary"
            if st.button(
                f"Open {task['title']}",
                key=f"task_{task_id}",
                use_container_width=True,
                type=btn_type
            ):
                st.session_state.selected_task = task_id
                st.rerun()

    # Row 1
    for i, task_id in enumerate(tasks_row1):
        render_task_card(row1[i], task_id)

    # Row 2
    for i, task_id in enumerate(tasks_row2):
        render_task_card(row2[i], task_id)

    # Featured Row - Global Recall Surveillance (full width)
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; margin: 0.5rem 0;">
        <span style="color: #004366; font-weight: 600;">🔍 Regulatory Intelligence</span>
    </div>
    """, unsafe_allow_html=True)

    recall_col1, recall_col2, recall_col3 = st.columns([1, 2, 1])
    render_task_card(recall_col2, 'recalls', featured=True)

    st.markdown("---")

    # Quick search and Show All
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("**🔍 Quick Search:**")
        user_input = st.text_input(
            "Type keywords",
            placeholder="e.g., 'b2b', 'screen', 'tracker', 'inventory', or 'all'",
            label_visibility="collapsed",
            key="task_search_input"
        )

        col_go, col_all = st.columns(2)
        with col_go:
            if st.button("🔍 Go", use_container_width=True, type="primary"):
                if user_input:
                    matched = match_task_from_input(user_input)
                    if matched:
                        st.session_state.selected_task = matched
                        st.rerun()
                    else:
                        st.warning("No matching tool found. Try: b2b, screen, tracker, inventory, categorize, resources")
        with col_all:
            if st.button("📂 Show All Tools", use_container_width=True):
                st.session_state.selected_task = 'all'
                st.rerun()

    # Quick tips section
    with st.expander("💡 Quick Tips", expanded=False):
        st.markdown("""
        **Common Workflows:**
        - **Weekly Returns Analysis:** Categorize Returns → B2B Report
        - **Quality Investigation:** Screen Products → Quality Case Tracker
        - **Inventory Planning:** Screen Products → Inventory Analysis
        - **🆕 Proactive Surveillance:** Global Recall Surveillance → Screen similar products

        **Keyboard Shortcuts:**
        - Type `all` to see all tools in tab view
        - Type tool keywords like `b2b`, `screen`, `tracker`, `recalls` for quick access
        - Type `recall`, `fda`, `mhra`, or `maude` for Global Recall Surveillance
        """)


# --- GLOBAL RECALL SURVEILLANCE TOOL ---

def render_global_recall_surveillance():
    """
    Render the Global Recall Surveillance tool.
    Integrates FDA, EU EMA, UK MHRA, Health Canada, ANVISA, CPSC, MAUDE adverse events,
    Google Custom Search, and Google News RSS for comprehensive regulatory intelligence.
    """
    from datetime import date, timedelta
    from src.services.regulatory_service import RegulatoryService
    from src.services.adverse_event_service import AdverseEventService
    from src.services.media_service import MediaMonitoringService

    # Initialize session state for recall surveillance
    if 'recall_surveillance_results' not in st.session_state:
        st.session_state.recall_surveillance_results = pd.DataFrame()
    if 'recall_surveillance_log' not in st.session_state:
        st.session_state.recall_surveillance_log = {}

    st.markdown("""
    <div style="background: linear-gradient(135deg, #004366 0%, #23b2be 100%); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
        <h4 style="color: white; margin: 0;">🌐 Global Regulatory Intelligence</h4>
        <p style="color: rgba(255,255,255,0.85); margin: 0.3rem 0 0 0; font-size: 0.9rem;">
            Scan FDA, EU EMA, UK MHRA, Health Canada, ANVISA, CPSC, MAUDE adverse events, and global media for recalls and safety signals.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- QUICK SEARCH PRESETS ---
    st.markdown("**⚡ Quick Search Presets:**")
    preset_cols = st.columns(8)
    preset_categories = [
        ("🩺 BP Monitors", "blood pressure monitor"),
        ("🦽 Mobility", "wheelchair scooter walker"),
        ("💉 Infusion", "infusion pump syringe"),
        ("🫀 Cardiac", "pacemaker defibrillator"),
        ("🩹 Wound Care", "bandage dressing wound"),
        ("🧪 Diagnostic", "glucometer thermometer oximeter"),
        ("🛏️ Patient Care", "hospital bed mattress"),
        ("🔧 Ortho", "brace splint support"),
    ]

    # Initialize search query from session state or empty
    if 'recall_search_query' not in st.session_state:
        st.session_state.recall_search_query = ""

    for i, (label, query) in enumerate(preset_categories):
        with preset_cols[i]:
            if st.button(label, key=f"preset_{i}", use_container_width=True):
                st.session_state.recall_search_query = query
                st.rerun()

    st.markdown("---")

    # --- SEARCH CONFIGURATION ---
    with st.expander("🔍 Search Configuration", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            search_query = st.text_input(
                "Product Search",
                value=st.session_state.recall_search_query,
                placeholder="e.g., blood pressure monitor, wheelchair, infusion pump",
                help="Enter product name, category, or keywords. Synonyms are auto-expanded."
            )
            # Update session state
            st.session_state.recall_search_query = search_query

            manufacturer = st.text_input(
                "Manufacturer (Optional)",
                placeholder="e.g., MedTech Inc, Generic Corp",
                help="Filter by manufacturer/recalling firm name"
            )

        with col2:
            lookback_days = st.slider(
                "Lookback Period",
                min_value=30,
                max_value=1825,  # 5 years
                value=365,
                step=30,
                help="How far back to search (in days)"
            )
            result_limit = st.slider(
                "Max Results",
                min_value=50,
                max_value=500,
                value=200,
                step=50,
                help="Maximum results per data source"
            )

        # Region selection
        st.markdown("**Coverage Regions:**")
        region_cols = st.columns(6)
        regions = []
        with region_cols[0]:
            if st.checkbox("🇺🇸 US", value=True, key="reg_us"):
                regions.append("US")
        with region_cols[1]:
            if st.checkbox("🇪🇺 EU", value=True, key="reg_eu"):
                regions.append("EU")
        with region_cols[2]:
            if st.checkbox("🇬🇧 UK", value=True, key="reg_uk"):
                regions.append("UK")
        with region_cols[3]:
            if st.checkbox("🇨🇦 Canada", value=True, key="reg_ca"):
                regions.append("CA")
        with region_cols[4]:
            if st.checkbox("🌎 LATAM", value=False, key="reg_latam"):
                regions.append("LATAM")
        with region_cols[5]:
            if st.checkbox("🌏 APAC", value=False, key="reg_apac"):
                regions.append("APAC")

        # Search mode
        col_mode1, col_mode2 = st.columns(2)
        with col_mode1:
            search_mode = st.radio(
                "Search Mode",
                ["🎯 Comprehensive (APIs + Web + Media)", "⚡ Fast (APIs Only)"],
                index=0,
                help="Comprehensive includes web search and media monitoring"
            )
        with col_mode2:
            include_sanctions = st.checkbox(
                "Include Sanctions/Watchlists",
                value=True,
                help="Check OFAC and other sanctions lists for manufacturer"
            )
            vendor_only = st.checkbox(
                "Vendor Enforcement Only",
                value=False,
                help="Only show enforcement actions, not general recalls"
            )

        # Advanced multilingual settings
        with st.expander("🌍 Language & International Settings", expanded=False):
            st.markdown("""
            **Multi-Language Search** expands your search terms to equivalent terms in Spanish, Portuguese,
            German, French, Japanese, and Chinese to find recalls from international agencies.
            """)
            lang_col1, lang_col2 = st.columns(2)
            with lang_col1:
                multilingual_search = st.checkbox(
                    "🌐 Enable Multi-Language Search",
                    value=True,
                    help="Automatically translate search terms to find international recalls (ES, PT, DE, FR, JA, ZH)"
                )
            with lang_col2:
                translate_results = st.checkbox(
                    "🔄 Auto-Translate Results to English",
                    value=True,
                    help="Translate non-English results back to English for easier review"
                )

            st.markdown("**International Sources Searched:**")
            st.markdown("""
            - 🇺🇸 **US**: FDA Device Recalls, FDA Enforcement, MAUDE Adverse Events, CPSC
            - 🇬🇧 **UK**: MHRA Medical Device Alerts, Drug Device Alerts
            - 🇪🇺 **EU**: EMA Safety, EU Safety Gate (RAPEX), Germany BfArM, France ANSM
            - 🇨🇦 **Canada**: Health Canada Recalls (EN/FR), MedEffect Advisories
            - 🇦🇺 **APAC**: Australia TGA, Japan PMDA, Singapore HSA
            - 🌎 **LATAM**: Brazil ANVISA, Mexico COFEPRIS
            - 🌍 **Global**: WHO Medical Device Alerts, IMDRF News
            """)

    # --- RUN SURVEILLANCE ---
    btn_col1, btn_col2, _ = st.columns([1, 1, 2])
    with btn_col1:
        run_search = st.button(
            "🚀 Launch Surveillance",
            type="primary",
            use_container_width=True
        )
    with btn_col2:
        clear_results = st.button(
            "🗑️ Clear Results",
            use_container_width=True
        )

    if clear_results:
        st.session_state.recall_surveillance_results = pd.DataFrame()
        st.session_state.recall_surveillance_log = {}
        st.rerun()

    if run_search:
        if not search_query and not manufacturer:
            st.error("Please enter a product search term or manufacturer name.")
        else:
            # Calculate date range
            end_date = date.today()
            start_date = end_date - timedelta(days=lookback_days)
            mode = "powerful" if "Comprehensive" in search_mode else "fast"

            with st.status("🔍 Scanning global regulatory sources...", expanded=True) as status:
                st.write("📡 Connecting to regulatory databases...")
                if multilingual_search:
                    st.write("🌍 Expanding search to multiple languages...")

                try:
                    df, logs = RegulatoryService.search_all_sources(
                        query_term=search_query.strip(),
                        manufacturer=manufacturer.strip(),
                        regions=regions if regions else ["US", "EU", "UK", "CA"],
                        start_date=start_date,
                        end_date=end_date,
                        limit=result_limit,
                        mode=mode,
                        vendor_only=vendor_only,
                        include_sanctions=include_sanctions,
                        multilingual=multilingual_search,
                        translate_results=translate_results
                    )

                    st.session_state.recall_surveillance_results = df
                    st.session_state.recall_surveillance_log = logs

                    total_found = len(df)
                    status.update(label=f"✅ Surveillance Complete - {total_found} records found", state="complete", expanded=False)

                except Exception as e:
                    st.error(f"Search error: {str(e)}")
                    status.update(label="❌ Search failed", state="error")

    # --- DISPLAY RESULTS ---
    df = st.session_state.recall_surveillance_results
    logs = st.session_state.recall_surveillance_log

    if df is not None and not df.empty:
        st.markdown("---")

        # Summary metrics
        st.subheader(f"🚨 {len(df)} Global Alerts Found")

        metric_cols = st.columns(5)
        with metric_cols[0]:
            high_risk = len(df[df.get('Risk_Level', 'Medium') == 'High']) if 'Risk_Level' in df.columns else 0
            st.metric("🔴 High Risk", high_risk)
        with metric_cols[1]:
            medium_risk = len(df[df.get('Risk_Level', 'Medium') == 'Medium']) if 'Risk_Level' in df.columns else 0
            st.metric("🟠 Medium Risk", medium_risk)
        with metric_cols[2]:
            low_risk = len(df[df.get('Risk_Level', 'Medium') == 'Low']) if 'Risk_Level' in df.columns else 0
            st.metric("🟢 Low Risk", low_risk)
        with metric_cols[3]:
            sources = len(logs) if logs else 0
            st.metric("📊 Sources Queried", sources)
        with metric_cols[4]:
            unique_firms = df['Firm'].nunique() if 'Firm' in df.columns else 0
            st.metric("🏢 Unique Firms", unique_firms)

        # Source breakdown
        with st.expander("📈 Source Coverage Details", expanded=False):
            if logs:
                source_df = pd.DataFrame([
                    {"Source": source, "Records": count}
                    for source, count in logs.items()
                ])
                st.dataframe(source_df, use_container_width=True, hide_index=True)

        # Results tabs
        tab_smart, tab_table = st.tabs(["🧠 Smart View", "📊 Full Table"])

        with tab_smart:
            # Sort by risk level
            risk_order = {"High": 0, "Medium": 1, "Low": 2}
            if 'Risk_Level' in df.columns:
                df_sorted = df.copy()
                df_sorted['risk_sort'] = df_sorted['Risk_Level'].map(risk_order).fillna(3)
                df_sorted = df_sorted.sort_values(['risk_sort', 'Date'], ascending=[True, False])
            else:
                df_sorted = df

            for idx, row in df_sorted.iterrows():
                risk = row.get('Risk_Level', 'Medium')
                risk_icon = "🔴" if risk == "High" else "🟠" if risk == "Medium" else "🟢"
                source = row.get('Source', 'Unknown')
                product = str(row.get('Product', 'Unknown'))[:60]
                date_str = row.get('Date', 'N/A')

                with st.expander(f"{risk_icon} {source} | {product}... | {date_str}"):
                    col_left, col_right = st.columns([3, 1])
                    with col_left:
                        st.markdown(f"**Product:** {row.get('Product', 'N/A')}")
                        st.markdown(f"**Firm:** {row.get('Firm', 'N/A')}")
                        st.markdown(f"**Model Info:** {row.get('Model Info', 'N/A')}")
                        if row.get('Reason'):
                            st.info(f"**Reason/Context:** {row.get('Reason', 'N/A')}")
                        if row.get('Matched_Term'):
                            st.caption(f"Matched term: {row.get('Matched_Term')}")
                    with col_right:
                        st.markdown(f"**Date:** {date_str}")
                        st.markdown(f"**Risk:** {risk}")
                        st.markdown(f"**Status:** {row.get('Status', 'N/A')}")
                        link = row.get('Link')
                        if link and link != 'N/A':
                            st.markdown(f"[🔗 Open Source Record]({link})")

        with tab_table:
            # Configure columns for display
            display_cols = ['Source', 'Date', 'Product', 'Firm', 'Risk_Level', 'Reason', 'Link']
            available_cols = [c for c in display_cols if c in df.columns]

            st.dataframe(
                df[available_cols] if available_cols else df,
                column_config={
                    "Link": st.column_config.LinkColumn("Source Link"),
                    "Risk_Level": st.column_config.TextColumn("Risk"),
                },
                use_container_width=True,
                hide_index=True
            )

            # Export options
            st.markdown("---")
            col_exp1, col_exp2, _ = st.columns([1, 1, 2])
            with col_exp1:
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "💾 Download CSV",
                    csv_data,
                    f"regulatory_surveillance_{date.today().isoformat()}.csv",
                    "text/csv",
                    use_container_width=True
                )
            with col_exp2:
                # High-risk only export
                if 'Risk_Level' in df.columns:
                    high_risk_df = df[df['Risk_Level'] == 'High']
                    if not high_risk_df.empty:
                        csv_high = high_risk_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "🔴 Download High-Risk Only",
                            csv_high,
                            f"high_risk_alerts_{date.today().isoformat()}.csv",
                            "text/csv",
                            use_container_width=True
                        )

    elif logs:
        st.info("No records found matching your criteria. Try broadening your search or extending the date range.")

    # --- BATCH SCANNING SECTION ---
    st.markdown("---")
    with st.expander("📂 Batch Product Scan (Upload CSV)", expanded=False):
        st.markdown("""
        **Bulk Recall Check:** Upload a CSV with your product list to check all products against recall databases at once.
        """)

        batch_file = st.file_uploader(
            "Upload CSV (columns: SKU, Product Name)",
            type=['csv'],
            key="batch_recall_file"
        )

        if batch_file:
            try:
                batch_df = pd.read_csv(batch_file)
                st.dataframe(batch_df.head(10), use_container_width=True)

                # Find product name column
                name_cols = [c for c in batch_df.columns if 'product' in c.lower() or 'name' in c.lower() or 'description' in c.lower()]
                if name_cols:
                    product_col = st.selectbox("Select Product Name Column", name_cols)
                else:
                    product_col = st.selectbox("Select Product Name Column", batch_df.columns.tolist())

                batch_lookback = st.slider("Batch Lookback (days)", 30, 730, 365, key="batch_lookback")

                if st.button("🚀 Run Batch Scan", type="primary", key="run_batch_scan"):
                    products = batch_df[product_col].dropna().unique().tolist()[:50]  # Limit to 50 products
                    all_results = []
                    progress = st.progress(0, text="Starting batch scan...")

                    for i, product in enumerate(products):
                        progress.progress((i + 1) / len(products), text=f"Scanning: {product[:40]}...")
                        try:
                            end_dt = date.today()
                            start_dt = end_dt - timedelta(days=batch_lookback)
                            df_result, _ = RegulatoryService.search_all_sources(
                                query_term=str(product),
                                regions=["US", "EU", "UK", "CA"],
                                start_date=start_dt,
                                end_date=end_dt,
                                limit=20,
                                mode="fast"
                            )
                            if not df_result.empty:
                                df_result['Searched_Product'] = product
                                all_results.append(df_result)
                        except Exception:
                            pass

                    progress.empty()

                    if all_results:
                        batch_results = pd.concat(all_results, ignore_index=True)
                        st.success(f"✅ Found {len(batch_results)} potential matches across {len(all_results)} products")
                        st.dataframe(batch_results, use_container_width=True)

                        csv_batch = batch_results.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "💾 Download Batch Results",
                            csv_batch,
                            f"batch_recall_scan_{date.today().isoformat()}.csv",
                            "text/csv"
                        )
                    else:
                        st.info("No recalls found for any products in your list.")

            except Exception as e:
                st.error(f"Error reading file: {str(e)}")

    # Initial state - show guidance
    if df is None or df.empty:
        with st.expander("💡 How to Use This Tool", expanded=True):
            st.markdown("""
            **Purpose:** Search global regulatory databases to find:
            - **Product Recalls** affecting your products or similar devices
            - **Safety Alerts** from health authorities worldwide
            - **Adverse Events** (FDA MAUDE) reports
            - **Enforcement Actions** and sanctions
            - **Media Coverage** of safety issues

            **Data Sources:**
            | Region | Sources |
            |--------|---------|
            | 🇺🇸 US | FDA Device Recalls, FDA Enforcement, FDA MAUDE, CPSC |
            | 🇪🇺 EU | EMA Alerts, EU Safety Communications |
            | 🇬🇧 UK | MHRA Device Alerts |
            | 🇨🇦 Canada | Health Canada Recalls |
            | 🌎 LATAM | ANVISA (Brazil), COFEPRIS (Mexico) |
            | 🌏 APAC | TGA (Australia), PMDA (Japan), HSA (Singapore) |

            **Synonym Auto-Expansion:**
            | You Type | Also Searches |
            |----------|---------------|
            | bpm | blood pressure monitor, bp monitor, sphygmomanometer |
            | scooter | mobility scooter, powered scooter, electric scooter |
            | defibrillator | AED, ICD, implantable cardioverter |
            | glucometer | glucose meter, blood glucose monitor |
            | infusion pump | IV pump, syringe pump |

            **Pro Tips:**
            - Use product categories like "blood pressure monitor" for broad coverage
            - Add manufacturer name to find vendor-specific issues
            - Enable "Comprehensive" mode for media and web coverage
            - Check "Sanctions/Watchlists" when evaluating new suppliers
            - Use **Batch Scan** to check your entire product catalog at once
            """)


# --- MAIN APP ---

def render_single_tool(task_id: str, provider_map: dict, provider_selection: str):
    """Render a single tool with back button"""
    task = TASK_DEFINITIONS.get(task_id, {})

    # Back button at top
    col_back, col_title = st.columns([1, 10])
    with col_back:
        if st.button("← Back", key="back_to_menu", help="Return to tool selector"):
            st.session_state.selected_task = None
            st.rerun()
    with col_title:
        st.markdown(f"### {task.get('icon', '')} {task.get('title', task_id)}")

    st.markdown("---")

    # Render the appropriate tool
    if task_id == 'categorize':
        render_categorizer_tool(provider_map, provider_selection)
    elif task_id == 'b2b':
        render_b2b_tool(provider_map, provider_selection)
    elif task_id == 'tracker':
        render_quality_cases_dashboard()
    elif task_id == 'screening':
        render_quality_screening_tab()
    elif task_id == 'inventory':
        render_inventory_integration_tab()
    elif task_id == 'resources':
        render_quality_resources()
    elif task_id == 'recalls':
        render_global_recall_surveillance()


def render_all_tabs(provider_map: dict, provider_selection: str):
    """Render all tools in tab view (legacy mode)"""
    # Back button to return to selector
    if st.button("← Back to Tool Selector", key="back_from_tabs"):
        st.session_state.selected_task = None
        st.rerun()

    st.markdown("---")

    # Tabs - All tools
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Return Categorizer",
        "📑 B2B Report Generator",
        "📋 Quality Case Tracker",
        "🧪 Quality Screening",
        "📦 Inventory Integration",
        "📚 Resources",
        "🌐 Global Recalls"
    ])

    with tab1:
        render_categorizer_tool(provider_map, provider_selection)

    with tab2:
        render_b2b_tool(provider_map, provider_selection)

    with tab3:
        render_quality_cases_dashboard()

    with tab4:
        render_quality_screening_tab()

    with tab5:
        render_inventory_integration_tab()

    with tab6:
        render_quality_resources()

    with tab7:
        render_global_recall_surveillance()


def main():
    initialize_session_state()
    inject_custom_css()

    # Header
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">🏥 VIVE HEALTH QUALITY SUITE</h1>
        <p style="color: white; margin: 0.5rem 0; font-size: 1.1rem;">
            <strong>Enterprise Quality Management System v22.0</strong>
        </p>
        <p style="color: rgba(255,255,255,0.9); margin: 0; font-size: 0.9rem;">
            🤖 <strong>AI-Powered:</strong> OpenAI/Claude LLMs | TQM Methodology | Dual Export (Leadership/Company-Wide)<br/>
            🌐 <strong>Global Intelligence:</strong> FDA | EU EMA | UK MHRA | Health Canada | CPSC | Media Monitoring<br/>
            📊 <strong>Compliance:</strong> ISO 13485 | FDA 21 CFR 820 | EU MDR | UK MDR
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not AI_AVAILABLE:
        st.error("❌ AI Modules Missing. Please check deployment.")
        st.stop()

    # Add AI status indicator
    keys = check_api_keys()
    ai_status = []
    if keys.get('openai'):
        ai_status.append("✅ OpenAI Active")
    if keys.get('claude'):
        ai_status.append("✅ Claude Active")

    if ai_status:
        st.success(f"🤖 AI Status: {' | '.join(ai_status)}")
    else:
        st.warning("⚠️ No AI API keys configured. Some features will be limited.")

    # Sidebar - Always visible
    with st.sidebar:
        st.markdown("### ⚙️ Global Configuration")

        # AI Provider selection
        provider_selection = st.selectbox(
            "🤖 AI Provider",
            options=['Fastest (Claude Haiku)', 'OpenAI GPT-3.5', 'Claude Sonnet', 'Both (Consensus)'],
            index=0,
            help="Select AI model for AI-powered tools"
        )

        provider_map = {
            'Fastest (Claude Haiku)': AIProvider.FASTEST,
            'OpenAI GPT-3.5': AIProvider.OPENAI,
            'Claude Sonnet': AIProvider.CLAUDE,
            'Both (Consensus)': AIProvider.BOTH
        }

        # API Health Check
        render_api_health_check()

        st.markdown("---")

        # Help guide
        render_help_guide()

    # === TASK-BASED ROUTING ===
    selected = st.session_state.selected_task

    if selected is None:
        # Show landing page / task selector
        render_task_selector()

    elif selected == 'all':
        # Show all tabs view
        render_all_tabs(provider_map, provider_selection)

    else:
        # Show single tool view
        render_single_tool(selected, provider_map, provider_selection)


if __name__ == "__main__":
    main()
