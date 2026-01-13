"""
Vive Health Quality Suite - Version 19.0
Enhanced Quality Case Screening with Statistical Rigor

Tab 1: Return Categorizer (PRESERVED)
Tab 2: B2B Report Generator (PRESERVED)  
Tab 3: Quality Case Screening (REBUILT)

Features:
- ANOVA/MANOVA with p-values and post-hoc testing
- SPC Control Charting (CUSUM, Shewhart)
- Weighted Risk Scoring
- AI-powered cross-case correlation
- Fuzzy threshold matching
- Vendor email generation
- Investigation plan generation
- State persistence (session-based)
- Custom threshold profiles
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
    AI_AVAILABLE = True
except ImportError as e:
    AI_AVAILABLE = False
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
    'version': '19.0 (Enhanced Screening)',
    'chunk_sizes': [100, 250, 500, 1000],
    'default_chunk': 500,
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
    'muted': '#666680',
    'cost': '#50C878'
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
    """Inject custom CSS for modern UI"""
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
    }}
    
    html, body, .stApp {{
        font-family: 'Inter', sans-serif;
    }}
    
    .main-header {{
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 5px 20px rgba(0, 217, 255, 0.3);
    }}
    
    .main-title {{
        font-size: 2.2em;
        font-weight: 700;
        color: white;
        margin: 0;
    }}
    
    .info-box {{
        background: rgba(26, 26, 46, 0.8);
        border: 1px solid var(--primary);
        border-radius: 8px;
        padding: 1.2rem;
        margin: 0.8rem 0;
    }}
    
    .stButton > button {{
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        color: white;
        border: none;
        padding: 0.7rem 1.5rem;
        border-radius: 6px;
        font-weight: 600;
    }}
    
    .risk-critical {{
        background-color: #ff4b4b !important;
        color: white !important;
    }}
    
    .risk-warning {{
        background-color: #ffa500 !important;
        color: black !important;
    }}
    
    .risk-monitor {{
        background-color: #ffff00 !important;
        color: black !important;
    }}
    
    .risk-ok {{
        background-color: #00ff00 !important;
        color: black !important;
    }}
    
    .processing-log {{
        background: #1a1a2e;
        border: 1px solid #333;
        border-radius: 5px;
        padding: 10px;
        max-height: 200px;
        overflow-y: auto;
        font-family: monospace;
        font-size: 12px;
    }}
    
    .methodology-box {{
        background: #f8f9fa;
        border-left: 4px solid #4facfe;
        padding: 15px;
        margin: 10px 0;
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
    """Render the completely rebuilt Quality Case Screening tab"""

    # Enhanced Header with TQM Philosophy
    st.markdown("### 🧪 Quality Case Screening")
    st.markdown("**TQM Methodology:** *Kaizen* (改善 = Continuous Improvement) | *Jidoka* (自働化 = Smart Automation) | *Genchi Genbutsu* (現地現物 = Go & See)")
    st.caption("AI-powered quality screening compliant with ISO 13485, FDA 21 CFR 820, EU MDR, UK MDR")

    # Comprehensive User Guide at Top
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
            ["Lite (1-5 Products)", "Pro (Mass Analysis)"],
            horizontal=True,
            help="Lite: Manual entry for quick screening. Pro: Upload CSV/Excel for batch analysis."
        )
        st.session_state.qc_mode = "Lite" if "Lite" in mode else "Pro"
    
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
        st.markdown("### 📋 Tab 3 Config")
        
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
    
    if st.session_state.qc_mode == "Lite":
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
            
            # Threshold inputs in columns
            new_thresholds = {}
            cols = st.columns(4)
            categories = list(DEFAULT_CATEGORY_THRESHOLDS.keys())
            
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
                    for cat in categories:
                        st.session_state[f"thresh_{cat}"] = st.session_state.get(f"thresh_{cat}", 10) * 0.8
                    st.rerun()
            with col2:
                if st.button("Loosen All (+20%)", key="loosen"):
                    for cat in categories:
                        st.session_state[f"thresh_{cat}"] = st.session_state.get(f"thresh_{cat}", 10) * 1.2
                    st.rerun()
            with col3:
                if st.button("Reset to SOP", key="reset_sop"):
                    for cat in categories:
                        st.session_state[f"thresh_{cat}"] = DEFAULT_CATEGORY_THRESHOLDS.get(cat, 0.10) * 100
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
                    width="stretch"
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
        if st.button("❓ How to set thresholds?", key="q1", width="stretch"):
            _add_ai_response("threshold_help")
    
    with col2:
        if st.button("📊 Explain my results", key="q2", width="stretch"):
            _add_ai_response("results_help")
    
    col3, col4 = st.columns(2)
    
    with col3:
        if st.button("🎯 What should I screen?", key="q3", width="stretch"):
            _add_ai_response("screening_help")
    
    with col4:
        if st.button("⚠️ Risk score meaning?", key="q4", width="stretch"):
            _add_ai_response("risk_help")
    
    # Free text input
    st.markdown("**Ask anything:**")
    user_question = st.text_input(
        "Type your question",
        placeholder="e.g., Should I flag products under 5% return rate?",
        key="ai_chat_input",
        label_visibility="collapsed"
    )
    
    if st.button("Send", key="send_chat", width="stretch"):
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
    
    system_prompt = f"""You are a medical device quality management expert assistant. 
You help users with quality case screening, understanding return rates, setting thresholds, and interpreting results.
Be concise but helpful. Use bullet points for clarity.
Current context: {context}
Answer questions about quality management, ISO 13485, FDA QSR, return rate analysis, and the screening tool."""
    
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
        if st.button("🔍 Run AI Screening", type="primary", width="stretch", disabled=valid_count == 0):
            # Filter to valid entries only
            valid_entries = [e for e in all_entries if e.get('_valid', False)]
            
            # Remove internal _valid flag
            for e in valid_entries:
                e.pop('_valid', None)
            
            # Create DataFrame and process
            df_input = pd.DataFrame(valid_entries)
            process_screening(df_input)
    
    with col_clear:
        if st.button("🗑️ Clear All", width="stretch"):
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
        st.download_button(
            "📥 Download Example Data",
            example_csv,
            file_name="quality_screening_example.csv",
            mime="text/csv",
            help="Download example data with 5 sample products"
        )
    
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
            st.dataframe(df_input.head(10), width="stretch")
            
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
            if st.button("🚀 Run Full Screening Analysis", type="primary", width="stretch"):
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
            
            st.altair_chart(chart, width="stretch")
    
    # Claude Review (if available)
    claude_reviews = [c for c in st.session_state.ai_chat_history if c.get('role') == 'claude_review']
    if claude_reviews:
        with st.expander("🤖 Claude AI Review", expanded=True):
            st.markdown(claude_reviews[-1]['content'])
    
    # Results Table
    st.markdown("#### Detailed Results")
    
    # Add color coding based on action
    def highlight_action(row):
        if 'Immediate' in str(row.get('Action', '')):
            return ['background-color: #ff4b4b'] * len(row)
        elif 'Case' in str(row.get('Action', '')):
            return ['background-color: #ffa500'] * len(row)
        elif 'Monitor' in str(row.get('Action', '')):
            return ['background-color: #ffff99'] * len(row)
        return [''] * len(row)
    
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
        width="stretch",
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
            st.markdown("#### 📧 Generate Vendor Email")
            selected_sku = st.selectbox(
                "Select SKU",
                options=action_items['SKU'].unique(),
                key="email_sku_select"
            )
            
            email_type = st.selectbox(
                "Email Type",
                ["CAPA Request", "RCA Request", "Inspection Notice"]
            )
            
            if st.button("Generate Email", key="gen_email"):
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
                elif email_type == "RCA Request":
                    email = VendorEmailGenerator.generate_rca_request(
                        sku=row['SKU'],
                        product_name=row.get('Name', row['SKU']),
                        defect_type=row['Action'],
                        occurrence_rate=row['Return_Rate'],
                        sample_complaints=str(row.get('Complaint_Text', '')).split(',')[:5]
                    )
                else:
                    email = VendorEmailGenerator.generate_inspection_notice(
                        sku=row['SKU'],
                        product_name=row.get('Name', row['SKU']),
                        special_focus=str(row.get('Triggers', '')).split(';')
                    )
                
                st.text_area("Generated Email", email, height=400)
                st.download_button(
                    "📥 Download Email",
                    email,
                    file_name=f"vendor_email_{selected_sku}_{datetime.now().strftime('%Y%m%d')}.txt"
                )
        
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

        # TAB 2: Bulk Vendor Emails
        with tab2:
            st.markdown("#### Generate Vendor Emails for Multiple Products")
            st.caption("Create emails for all flagged products at once")

            # Select products
            selected_for_email = st.multiselect(
                "Select Products for Vendor Communication",
                options=action_items['SKU'].tolist(),
                default=action_items['SKU'].tolist()[:5],  # Default to first 5
                help="Select which products need vendor follow-up"
            )

            # Email type
            col1, col2 = st.columns(2)
            with col1:
                bulk_email_type = st.selectbox(
                    "Email Type (applies to all)",
                    ["CAPA Request", "RCA Request", "Inspection Notice", "Quality Alert"],
                    help="Same email type will be used for all selected products"
                )

            with col2:
                vendor_name = st.text_input(
                    "Vendor/Supplier Name",
                    placeholder="e.g., ABC Manufacturing Ltd.",
                    help="Vendor name for email personalization"
                )

            if st.button("📧 Generate All Emails", type="primary", key="bulk_emails"):
                if not selected_for_email:
                    st.warning("Please select at least one product")
                else:
                    with st.spinner(f"Generating {len(selected_for_email)} vendor emails..."):
                        try:
                            # Generate emails for each product
                            bulk_emails = []

                            for sku in selected_for_email:
                                row = action_items[action_items['SKU'] == sku].iloc[0]

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
                                else:
                                    email = VendorEmailGenerator.generate_inspection_notice(
                                        sku=row['SKU'],
                                        product_name=row.get('Name', row['SKU']),
                                        special_focus=str(row.get('Triggers', '')).split(';')
                                    )

                                bulk_emails.append({
                                    'SKU': sku,
                                    'Product': row.get('Name', sku),
                                    'Email_Type': bulk_email_type,
                                    'Subject': f"Quality Issue - {sku}",
                                    'Body': email,
                                    'Priority': row.get('Action', 'Monitor')
                                })

                            st.success(f"✅ Generated {len(bulk_emails)} emails!")

                            # Display preview
                            for i, email_data in enumerate(bulk_emails[:3]):  # Show first 3
                                with st.expander(f"📧 {email_data['SKU']} - {email_data['Product']}", expanded=(i==0)):
                                    st.markdown(f"**Priority:** {email_data['Priority']}")
                                    st.text_area("Email Content", email_data['Body'], height=200, key=f"preview_email_{i}")

                            if len(bulk_emails) > 3:
                                st.info(f"+ {len(bulk_emails) - 3} more emails (see CSV export)")

                            # Export option
                            email_df = pd.DataFrame(bulk_emails)
                            csv_data = email_df.to_csv(index=False)

                            st.download_button(
                                "📥 Download All Emails (CSV)",
                                csv_data,
                                file_name=f"bulk_vendor_emails_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                mime="text/csv"
                            )

                        except Exception as e:
                            st.error(f"Bulk email generation failed: {e}")
                            logger.error(f"Bulk email error: {e}")

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

            if st.button("📋 Generate All Plans", type="primary", key="bulk_plans"):
                if not selected_for_plans:
                    st.warning("Please select at least one product")
                else:
                    with st.spinner(f"Generating {len(selected_for_plans)} investigation plans..."):
                        try:
                            bulk_plans = []

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
                                    risk_score=row['Risk_Score'],
                                    investigation_method=assigned_method
                                )

                                bulk_plans.append({
                                    'SKU': sku,
                                    'Product': row.get('Name', sku),
                                    'Method': assigned_method,
                                    'Priority': row.get('Action', 'Monitor'),
                                    'Plan': InvestigationPlanGenerator.format_plan_markdown(plan),
                                    'Estimated_Days': plan.get('timeline_days', 14),
                                    'Team_Required': ', '.join(plan.get('team', []))
                                })

                            st.success(f"✅ Generated {len(bulk_plans)} investigation plans!")

                            # Display preview
                            for i, plan_data in enumerate(bulk_plans[:2]):  # Show first 2
                                with st.expander(f"📋 {plan_data['SKU']} - {plan_data['Product']}", expanded=(i==0)):
                                    st.markdown(f"**Method:** {plan_data['Method']} | **Priority:** {plan_data['Priority']}")
                                    st.markdown(plan_data['Plan'])

                            if len(bulk_plans) > 2:
                                st.info(f"+ {len(bulk_plans) - 2} more plans (see CSV export)")

                            # Export option
                            plan_df = pd.DataFrame(bulk_plans)
                            csv_data = plan_df.to_csv(index=False)

                            st.download_button(
                                "📥 Download All Plans (CSV)",
                                csv_data,
                                file_name=f"bulk_investigation_plans_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                mime="text/csv"
                            )

                        except Exception as e:
                            st.error(f"Bulk plan generation failed: {e}")
                            logger.error(f"Bulk plan error: {e}")

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
                                           width="stretch", height=400)

                            # Download options
                            col1, col2 = st.columns(2)
                            with col1:
                                csv_data = capa_plan.to_csv()
                                st.download_button(
                                    "📥 Download CSV (Smartsheet Import)",
                                    csv_data,
                                    file_name=f"CAPA_{capa_sku}_{datetime.now().strftime('%Y%m%d')}.csv",
                                    mime="text/csv",
                                    help="Import this CSV directly into Smartsheet"
                                )

                            with col2:
                                excel_data = capa_plan.to_excel()
                                st.download_button(
                                    "📥 Download Excel",
                                    excel_data,
                                    file_name=f"CAPA_{capa_sku}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )

                            # Statistics
                            total_days = max([t['Duration (Days)'] for t in capa_plan.tasks])
                            st.info(f"📅 **Estimated Timeline:** {total_days} days | **Total Tasks:** {len(capa_plan.tasks)}")

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
                                           width="stretch", height=400)

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

                            with col2:
                                excel_data = crit_plan.to_excel()
                                st.download_button(
                                    "📥 Download Excel",
                                    excel_data,
                                    file_name=f"CRITICAL_INVESTIGATION_{crit_sku}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )

                        except Exception as e:
                            st.error(f"Investigation plan generation failed: {e}")
                            logger.error(f"Investigation error: {e}")

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
                                               width="stretch", height=400)

                                # Download options
                                col1, col2 = st.columns(2)
                                with col1:
                                    csv_data = rework_plan.to_csv()
                                    st.download_button(
                                        "📥 Download CSV (Smartsheet)",
                                        csv_data,
                                        file_name=f"REWORK_{rework_sku}_{datetime.now().strftime('%Y%m%d')}.csv",
                                        mime="text/csv"
                                    )

                                with col2:
                                    excel_data = rework_plan.to_excel()
                                    st.download_button(
                                        "📥 Download Excel",
                                        excel_data,
                                        file_name=f"REWORK_{rework_sku}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                    )

                            except Exception as e:
                                st.error(f"Rework plan generation failed: {e}")
                                logger.error(f"Rework error: {e}")

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
            width="stretch"
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
            width="stretch"
        )
    
    with col3:
        # Clear results
        if st.button("🗑️ Clear Results", width="stretch"):
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
        
        st.download_button(
            "📥 Download Example Data",
            csv_buffer.getvalue(),
            file_name="example_screening_data.csv",
            mime="text/csv"
        )


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
    inv_tab1, inv_tab2, inv_tab3 = st.tabs([
        "📤 Data Upload",
        "⚙️ Configuration",
        "📊 Dashboard & Results"
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
                st.dataframe(template_df, width="stretch")

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

        st.dataframe(display_df, width="stretch", height=600)

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


# --- MAIN APP ---

def main():
    initialize_session_state()
    inject_custom_css()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">VIVE HEALTH QUALITY SUITE</h1>
        <p style="color: white; margin: 0.5rem 0;">AI-Powered Returns Analysis & Quality Screening (v19.0)</p>
    </div>
    """, unsafe_allow_html=True)

    if not AI_AVAILABLE:
        st.error("❌ AI Modules Missing. Please check deployment.")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Tab 1 & 2 AI Provider (original behavior)
        provider_t12 = st.selectbox(
            "🤖 AI Provider (Tab 1 & 2)",
            options=['Fastest (Claude Haiku)', 'OpenAI GPT-3.5', 'Claude Sonnet', 'Both (Consensus)'],
            index=0
        )
        
        # Map to enum for tabs 1 & 2
        provider_map_t12 = {
            'Fastest (Claude Haiku)': AIProvider.FASTEST,
            'OpenAI GPT-3.5': AIProvider.OPENAI,
            'Claude Sonnet': AIProvider.CLAUDE,
            'Both (Consensus)': AIProvider.BOTH
        }
        
        # API Health Check
        render_api_health_check()
        
        # Help guide
        render_help_guide()

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Return Categorizer", "📑 B2B Report Generator", "🧪 Quality Screening", "📦 Inventory Integration"])
    
    # --- TAB 1: Categorizer (PRESERVED) ---
    with tab1:
        # Use Tab 1/2 provider
        st.session_state.ai_provider = provider_map_t12[provider_t12]
        
        st.markdown("### 📁 Return Categorization (Column I → K)")
        st.markdown("""
        <div style="background: rgba(255, 183, 0, 0.1); border: 1px solid var(--accent); 
                    border-radius: 8px; padding: 0.8rem; margin-bottom: 1rem;">
            <strong>📌 Goal:</strong> Categorize complaints into standardized Quality Categories.
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload Return Data (Excel/CSV)", type=['csv', 'xlsx', 'xls', 'txt'], key="tab1_uploader")
        
        if uploaded_file:
            with st.spinner(f"Reading {uploaded_file.name}..."):
                file_content = uploaded_file.read()
                df, column_mapping = process_file_preserve_structure(file_content, uploaded_file.name)
            
            if df is not None and column_mapping:
                # Store mapping immediately
                st.session_state.column_mapping = column_mapping
                
                # Show file info
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
                        
                        # Export
                        st.session_state.export_data = export_with_column_k(categorized_df)
                        st.session_state.export_filename = f"categorized_{datetime.now().strftime('%Y%m%d')}.xlsx"
                        st.rerun()
        
        # Results Display (Tab 1)
        if st.session_state.processing_complete and st.session_state.categorized_data is not None:
            display_results_dashboard(st.session_state.categorized_data, st.session_state.column_mapping)
            
            if st.session_state.export_data:
                st.download_button(
                    label="⬇️ Download Categorized File",
                    data=st.session_state.export_data,
                    file_name=st.session_state.export_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary",
                    width="stretch"
                )

    # --- TAB 2: B2B Reports (PRESERVED) ---
    with tab2:
        # Use Tab 1/2 provider
        st.session_state.ai_provider = provider_map_t12[provider_t12]
        
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
        
        # Performance / File Size Selection
        st.markdown("#### ⚙️ Data Volume / Processing Speed")
        perf_mode = st.select_slider(
            "Select Dataset Size to optimize API performance:",
            options=['Small (< 500 rows)', 'Medium (500-2,000 rows)', 'Large (2,000+ rows)'],
            value=st.session_state.b2b_perf_mode,
            key='perf_selector'
        )
        st.session_state.b2b_perf_mode = perf_mode
        
        # Map selection to performance settings
        if perf_mode == 'Small (< 500 rows)':
            batch_size = 10
            max_workers = 3
            st.caption("Settings: Conservative batching for max reliability.")
        elif perf_mode == 'Medium (500-2,000 rows)':
            batch_size = 25
            max_workers = 6
            st.caption("Settings: Balanced speed and concurrency.")
        else:  # Large
            batch_size = 50
            max_workers = 10
            st.caption("Settings: Aggressive parallel processing for high volume.")

        st.divider()
        
        b2b_file = st.file_uploader("Upload Odoo Export (CSV/Excel)", type=['csv', 'xlsx'], key="b2b_uploader")
        
        if b2b_file:
            # Read & Preview
            b2b_df = process_b2b_file(b2b_file.read(), b2b_file.name)
            
            if b2b_df is not None:
                st.markdown(f"**Total Tickets Found:** {len(b2b_df):,}")
                
                # Process Button
                if st.button("⚡ Generate B2B Report", type="primary"):
                    # Update analyzer with new worker settings based on user choice
                    analyzer = get_ai_analyzer(max_workers=max_workers)
                    
                    with st.spinner("Running AI Analysis & SKU Extraction..."):
                        # Run the B2B pipeline
                        final_b2b = generate_b2b_report(b2b_df, analyzer, batch_size)
                        
                        # Save to session
                        st.session_state.b2b_processed_data = final_b2b
                        st.session_state.b2b_processing_complete = True
                        
                        # Prepare Download
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            final_b2b.to_excel(writer, index=False, sheet_name='B2B Report')
                            
                            # Formatting
                            workbook = writer.book
                            worksheet = writer.sheets['B2B Report']
                            
                            # Add simple formatting
                            header_fmt = workbook.add_format({'bold': True, 'bg_color': '#00D9FF', 'font_color': 'white'})
                            for col_num, value in enumerate(final_b2b.columns.values):
                                worksheet.write(0, col_num, value, header_fmt)
                                worksheet.set_column(col_num, col_num, 30)

                        st.session_state.b2b_export_data = output.getvalue()
                        st.session_state.b2b_export_filename = f"B2B_Report_{datetime.now().strftime('%Y-%m-%d')}.xlsx"
                        
                        st.rerun()

        # B2B Dashboard Results
        if st.session_state.b2b_processing_complete and st.session_state.b2b_processed_data is not None:
            df_res = st.session_state.b2b_processed_data
            
            st.markdown("### 🏁 Report Dashboard")
            
            # Dashboard Metrics
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Total Processed", len(df_res))
            with c2:
                sku_found_count = len(df_res[df_res['SKU'] != 'Unknown'])
                st.metric("SKUs Identified", f"{sku_found_count}", delta=f"{sku_found_count/len(df_res)*100:.1f}% coverage")
            with c3:
                unique_skus = df_res[df_res['SKU'] != 'Unknown']['SKU'].nunique()
                st.metric("Unique Products", unique_skus)
            
            # Preview Table
            st.markdown("#### Preview (Top 10)")
            st.dataframe(df_res.head(10), width="stretch")
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="⬇️ Download B2B Report (.xlsx)",
                    data=st.session_state.b2b_export_data,
                    file_name=st.session_state.b2b_export_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary",
                    width="stretch"
                )
            with col2:
                if st.button("🔄 Clear / Start Over", width="stretch"):
                    st.session_state.b2b_processed_data = None
                    st.session_state.b2b_processing_complete = False
                    st.rerun()

    # --- TAB 3: Quality Screening (REBUILT) ---
    with tab3:
        render_quality_screening_tab()

    # --- TAB 4: Inventory Integration ---
    with tab4:
        render_inventory_integration_tab()


if __name__ == "__main__":
    main()
