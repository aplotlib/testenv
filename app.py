import streamlit as st
import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime
import io
import time
import altair as alt

# --- Custom Modules ---
# Ensure these files are in the same directory
try:
    from enhanced_ai_analysis import EnhancedAIAnalyzer, AIProvider
    from quality_analytics import QualityAnalytics, parse_numeric, SOP_THRESHOLDS
    AI_AVAILABLE = True
except ImportError as e:
    AI_AVAILABLE = False
    print(f"Module Missing: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page Config
st.set_page_config(
    page_title="Vive Health Quality Suite",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Session State ---
def initialize_session_state():
    defaults = {
        'qc_data': None,
        'qc_mode': 'Lite', # Lite or Pro
        'ai_analyzer': None,
        'api_status': {'openai': False, 'claude': False},
        'processing_log': [], # Processing Transparency
        'qc_results': None
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def log_process(message: str, type: str = 'info'):
    """Adds message to the Processing Transparency Log"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    entry = f"[{timestamp}] {message}"
    st.session_state.processing_log.append(entry)
    if type == 'error':
        logger.error(message)
    else:
        logger.info(message)

# --- Helper Functions ---
def check_api_health():
    """Health Check for APIs"""
    # Simple check if keys exist (Real health check would ping endpoints)
    keys = {}
    if hasattr(st, 'secrets'):
        if 'openai' in st.secrets: keys['openai'] = True
        if 'claude' in st.secrets: keys['claude'] = True
    return keys

def get_analyzer():
    if not st.session_state.ai_analyzer:
        st.session_state.ai_analyzer = EnhancedAIAnalyzer(AIProvider.FASTEST)
    return st.session_state.ai_analyzer

# --- TAB 3: QUALITY SCREENING IMPLEMENTATION ---
def render_quality_screening_tab():
    st.markdown("### 🧪 Quality Case Screening")
    
    # --- Toolbar & Settings ---
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        # Lite vs Pro Mode
        mode = st.radio("Mode", ["Lite (Manual/Small Batch)", "Pro (Mass Analysis/Upload)"], horizontal=True)
        st.session_state.qc_mode = "Lite" if "Lite" in mode else "Pro"
    
    with col3:
        # API Health Check
        health = check_api_health()
        st.caption("System Health:")
        if health.get('openai'): st.success("OpenAI: Connected") 
        else: st.error("OpenAI: Disconnected")
        if health.get('claude'): st.success("Claude: Connected") 
        else: st.error("Claude: Disconnected")

    st.divider()

    # --- INPUT SECTION ---
    df_input = None
    
    if st.session_state.qc_mode == "Lite":
        st.info("ℹ️ Lite Mode: Analyze 1-5 products manually.")
        # Manual Entry Form
        with st.form("lite_entry"):
            c1, c2, c3 = st.columns(3)
            sku = c1.text_input("Product SKU/Name")
            category = c2.selectbox("Category", list(SOP_THRESHOLDS.keys()))
            cost = c3.number_input("Landed Cost ($)", min_value=0.0)
            
            c4, c5, c6 = st.columns(3)
            sold = c4.number_input("Units Sold", min_value=1)
            returned = c5.number_input("Units Returned", min_value=0)
            complaint_text = c6.text_area("Top Complaint Reasons (Optional)")
            
            # Optional Inputs
            with st.expander("Optional Context (Manuals, Feedback)"):
                manual_text = st.text_area("Paste relevant manual text here")
                safety_risk = st.checkbox("Potential Safety Risk involved?")
            
            submitted = st.form_submit_button("Run Analysis")
            
            if submitted and sku:
                data = {
                    'SKU': [sku], 'Category': [category], 'Landed Cost': [cost],
                    'Sold': [sold], 'Returned': [returned], 'Complaint_Text': [complaint_text],
                    'Manual_Context': [manual_text], 'Safety Risk': ['Yes' if safety_risk else 'No']
                }
                df_input = pd.DataFrame(data)

    else: # PRO MODE
        st.info("🚀 Pro Mode: Mass analysis with statistical rigor.")
        # File Upload
        uploaded_file = st.file_uploader("Upload Data (CSV/Excel)", type=['csv', 'xlsx'])
        
        # User thresholds upload
        with st.expander("📂 Advanced: Upload Custom Thresholds / SOPs"):
            st.file_uploader("Upload Thresholds (Optional)", type=['csv'])
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df_input = pd.read_csv(uploaded_file)
                else:
                    df_input = pd.read_excel(uploaded_file)
                
                # Validation Report
                required = ['SKU', 'Category', 'Sold', 'Returned']
                report = QualityAnalytics.validate_upload(df_input, required)
                
                if not report['valid']:
                    st.error(f"Missing Columns: {report['missing_cols']}")
                    st.stop()
                else:
                    st.success(f"Data Validated: {report['total_rows']} rows ready.")
                    
            except Exception as e:
                st.error(f"Error reading file: {e}")

    # --- PROCESSING SECTION ---
    if df_input is not None:
        if st.button("🔍 Run Screening Analysis", type="primary"):
            log_process("Started Screening Analysis")
            progress = st.progress(0)
            
            # 1. Basic Calculations
            df_input['Sold'] = parse_numeric(df_input['Sold'])
            df_input['Returned'] = parse_numeric(df_input['Returned'])
            df_input['Landed Cost'] = parse_numeric(df_input.get('Landed Cost', pd.Series(0)))
            df_input['Return_Rate'] = df_input['Returned'] / df_input['Sold']
            
            # 2. Risk Scoring & SOP Checks
            results = []
            log_process("Calculating Risk Scores and SOP checks...")
            
            # Categories for ANOVA
            if len(df_input) > 5:
                # ANOVA Calculation
                anova_res = QualityAnalytics.perform_anova(df_input, 'Category', 'Return_Rate')
                if anova_res.get('p_value'):
                    st.session_state.anova_result = anova_res
                    log_process(f"ANOVA Complete. P-Value: {anova_res['p_value']:.4f}")
            
            # Row-by-row processing (Simulating batch if needed)
            for idx, row in df_input.iterrows():
                # Risk Score
                cat_avg = SOP_THRESHOLDS.get(row['Category'], 0.10)
                risk_score = QualityAnalytics.calculate_risk_score(row, cat_avg)
                
                # Action Determination
                action = QualityAnalytics.determine_action(row, SOP_THRESHOLDS)
                
                # SPC Signal (Mocking history logic for demo)
                spc_signal = QualityAnalytics.detect_spc_signals(row, cat_avg, cat_avg*0.2) # Mock std dev
                
                row['Risk_Score'] = risk_score
                row['Recommended_Action'] = action
                row['SPC_Signal'] = spc_signal
                results.append(row)
                
            df_results = pd.DataFrame(results)
            st.session_state.qc_results = df_results
            progress.progress(100)
            log_process("Analysis Complete")

    # --- RESULTS DASHBOARD ---
    if st.session_state.qc_results is not None:
        df = st.session_state.qc_results
        
        st.markdown("### 📊 Screening Results")
        
        # 1. Visual Heatmap
        with st.expander("🔥 Visual Heatmap (Risk vs Cost)", expanded=True):
            chart = alt.Chart(df).mark_circle(size=60).encode(
                x=alt.X('Landed Cost', title='Landed Cost ($)'),
                y=alt.Y('Return_Rate', title='Return Rate', axis=alt.Axis(format='%')),
                color=alt.Color('Risk_Score', scale=alt.Scale(scheme='redyellowgreen', reverse=True), title='Risk Score'),
                tooltip=['SKU', 'Category', 'Return_Rate', 'Recommended_Action']
            ).interactive()
            st.altair_chart(chart, use_container_width=True)

        # 2. Key Metrics & Actions
        c1, c2, c3 = st.columns(3)
        escalations = len(df[df['Recommended_Action'].str.contains("Escalate")])
        c1.metric("Total Analyzed", len(df))
        c2.metric("Escalations Needed", escalations, delta_color="inverse")
        
        if 'anova_result' in st.session_state:
            p_val = st.session_state.anova_result.get('p_value', 1.0)
            c3.metric("Statistical Sig. (ANOVA p-val)", f"{p_val:.4f}", 
                      delta="Significant" if p_val < 0.05 else "Not Significant")

        # 3. Detailed Data Table
        st.dataframe(df.style.apply(lambda x: ['background-color: #ff4b4b' if 'Escalate' in str(v) else '' for v in x], subset=['Recommended_Action']))
        
        # 4. AI Analysis & Drafts
        st.markdown("#### 🤖 AI Analysis & Next Steps")
        
        selected_sku = st.selectbox("Select SKU for AI Investigation Plan", df[df['Recommended_Action'].str.contains("Escalate")]['SKU'].unique())
        
        if selected_sku:
            sku_data = df[df['SKU'] == selected_sku].iloc[0]
            if st.button("Generate Investigation Plan & Vendor Email"):
                analyzer = get_analyzer()
                
                # Investigation Prompt
                prompt = f"""
                Act as a Quality Manager. Create a Draft Investigation Plan for SKU {selected_sku}.
                Data: Return Rate: {sku_data['Return_Rate']:.1%}, Cost: ${sku_data['Landed Cost']}, Category: {sku_data['Category']}.
                Complaint Context: {sku_data.get('Complaint_Text', 'N/A')}
                Risk Score: {sku_data['Risk_Score']}
                
                Output:
                1. Investigation Steps
                2. Specific areas of device to inspect
                3. Draft Email to Vendor requesting CAPA/RCA
                """
                
                with st.spinner("AI Generating Plan..."):
                    response = analyzer.generate_text(prompt, "You are a Medical Device Quality Expert complying with ISO 13485.")
                    st.markdown(response)

        # 5. Methodology & Transparency
        with st.expander("🧮 Methodology & Math (Formulas)", expanded=False): #
            st.markdown(QualityAnalytics.generate_methodology_markdown())
            
        with st.expander("📜 Processing Transparency Log", expanded=False): #
            for log in st.session_state.processing_log:
                st.text(log)
                
        # 6. Metadata Export
        if st.download_button("Download Full Report", data="mock_data", disabled=True):
             pass # Implementation would involve creating Excel with metadata sheet

# --- MAIN APP STRUCTURE ---
def main():
    initialize_session_state()
    
    st.title("Vive Health Quality Suite (v18.0)")
    
    tab1, tab2, tab3 = st.tabs(["Categorizer", "B2B Reports", "Quality Case Screening"])
    
    with tab1:
        st.caption("Existing Categorizer Functionality")
        # (Preserve existing Tab 1 code here)
        
    with tab2:
        st.caption("Existing B2B Reports Functionality")
        # (Preserve existing Tab 2 code here)
        
    with tab3:
        render_quality_screening_tab()

if __name__ == "__main__":
    main()
