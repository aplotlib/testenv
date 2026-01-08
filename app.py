import streamlit as st
import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime, timedelta
import io
from typing import Dict, List, Any, Optional, Tuple
import time
from collections import Counter, defaultdict
import re
import os
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config with increased limits
st.set_page_config(
    page_title="Vive Health Quality & B2B Reports",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import AI module
try:
    from enhanced_ai_analysis import (
        EnhancedAIAnalyzer, AIProvider, FBA_REASON_MAP,
@@ -171,51 +172,60 @@ def initialize_session_state():
        'ai_analyzer': None,
        'processing_complete': False,
        'reason_summary': {},
        'product_summary': {},
        'total_cost': 0.0,
        'api_calls_made': 0,
        'processing_time': 0.0,
        'ai_provider': AIProvider.FASTEST,
        'batch_size': 20,
        'chunk_size': APP_CONFIG['default_chunk'],
        'processing_errors': [],
        'total_rows_processed': 0,
        'column_mapping': {},  # Track original column positions
        'show_product_analysis': False,
        'processing_speed': 0.0,
        'auto_download_triggered': False,  # Track if auto-download was triggered
        'export_data': None,  # Store export data
        'export_filename': None,  # Store filename
        
        # B2B Report specific state
        'b2b_original_data': None,
        'b2b_processed_data': None,
        'b2b_processing_complete': False,
        'b2b_export_data': None,
        'b2b_export_filename': None,
        'b2b_perf_mode': 'Small (< 500 rows)',

        # Quality Case Screening state
        'qc_manual_entries': [],
        'qc_screened_data': None,
        'qc_export_data': None,
        'qc_export_filename': None,
        'qc_ai_review': None,
        'qc_chat_history': [],
        'qc_combined_data': None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def check_api_keys():
    """Check for API keys"""
    keys_found = {}
    
    try:
        if hasattr(st, 'secrets'):
            # Check OpenAI
            for key in ['OPENAI_API_KEY', 'openai_api_key', 'openai']:
                if key in st.secrets:
                    keys_found['openai'] = str(st.secrets[key]).strip()
                    break
            
            # Check Claude
            for key in ['ANTHROPIC_API_KEY', 'anthropic_api_key', 'claude_api_key', 'claude']:
                if key in st.secrets:
                    keys_found['claude'] = str(st.secrets[key]).strip()
                    break
    except Exception as e:
        logger.warning(f"Error checking secrets: {e}")
@@ -612,50 +622,199 @@ def generate_b2b_report(df, analyzer, batch_size):
            'Description': item['full_description'],
            'SKU': item['sku'],
            'Reason': item['summary']
        })
        
    return pd.DataFrame(final_rows)

# -------------------------
# TAB 3: QUALITY CASE SCREENING
# -------------------------

CATEGORY_BENCHMARKS = {
    'B2B Products (All)': 0.025,
    'INS': 0.07,
    'RHB': 0.075,
    'LVA': 0.095,
    'MOB - Power Scooters': 0.095,
    'MOB - Walkers/Rollators/other': 0.10,
    'MOB - Wheelchairs (manual)': 0.105,
    'CSH': 0.105,
    'SUP': 0.11,
    'MOB - Wheelchairs (power)': 0.115,
    'All Others': 0.10,
}

QUALITY_CASE_COLUMNS = [
    {
        'Column': 'SKU',
        'Description': 'Unique product identifier used for escalation tracking.',
        'Requirement': 'Mandatory'
    },
    {
        'Column': 'Category',
        'Description': 'Product family/category used for benchmarks.',
        'Requirement': 'Mandatory'
    },
    {
        'Column': 'Sold',
        'Description': 'Total units sold in the period.',
        'Requirement': 'Mandatory'
    },
    {
        'Column': 'Returned',
        'Description': 'Total units returned in the period.',
        'Requirement': 'Mandatory'
    },
    {
        'Column': 'Landed Cost',
        'Description': 'Unit landed cost for immediate escalation thresholds.',
        'Requirement': 'Optional'
    },
    {
        'Column': 'Retail Price',
        'Description': 'Unit retail price for immediate escalation thresholds.',
        'Requirement': 'Optional'
    },
    {
        'Column': 'Launch Date',
        'Description': 'Launch date for new-product escalation logic (YYYY-MM-DD).',
        'Requirement': 'Optional'
    },
    {
        'Column': 'Safety Risk',
        'Description': 'Yes/No flag for safety risk.',
        'Requirement': 'Optional'
    },
    {
        'Column': 'Zero Tolerance Component',
        'Description': 'Yes/No flag for zero tolerance component.',
        'Requirement': 'Optional'
    },
    {
        'Column': 'AQL Fail',
        'Description': 'Yes/No flag for recent AQL failures.',
        'Requirement': 'Optional'
    },
    {
        'Column': 'Unique Complaint Count (30d)',
        'Description': 'Unique complaints in last 30 days.',
        'Requirement': 'Optional'
    },
    {
        'Column': 'Sales Units (30d)',
        'Description': 'Units sold in the last 30 days.',
        'Requirement': 'Optional'
    },
    {
        'Column': 'Sales Value (30d)',
        'Description': 'Sales value ($) in the last 30 days.',
        'Requirement': 'Optional'
    }
]

QUALITY_CASE_STANDARD_COLUMNS = [
    entry['Column'] for entry in QUALITY_CASE_COLUMNS
] + ['Input Source']

def quality_case_requirements_df():
    return pd.DataFrame(QUALITY_CASE_COLUMNS)

def build_quality_case_template():
    template_df = pd.DataFrame(columns=[entry['Column'] for entry in QUALITY_CASE_COLUMNS])
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        template_df.to_excel(writer, index=False, sheet_name='Quality Case Template')
    return output.getvalue()

def normalize_quality_case_data(df, mapping, source_label):
    normalized = pd.DataFrame()
    for target_col in QUALITY_CASE_STANDARD_COLUMNS:
        if target_col == 'Input Source':
            normalized[target_col] = source_label
            continue
        source_col = mapping.get(target_col)
        if source_col and source_col in df.columns:
            normalized[target_col] = df[source_col]
        else:
            normalized[target_col] = ''
    return normalized

def build_quality_case_summary(screened_df):
    action_counts = screened_df['Recommended_Action'].value_counts().to_dict()
    top_escalations = screened_df.sort_values('Return_Rate', ascending=False).head(5)
    top_rows = top_escalations[['SKU', 'Category', 'Return_Rate_Display', 'Recommended_Action']].to_dict('records')
    category_alerts = (
        screened_df.groupby('Category')['Return_Rate']
        .mean()
        .sort_values(ascending=False)
        .head(5)
        .reset_index()
        .assign(Return_Rate=lambda x: (x['Return_Rate'] * 100).round(2).astype(str) + '%')
        .to_dict('records')
    )

    return {
        'total_rows': len(screened_df),
        'action_counts': action_counts,
        'top_escalations': top_rows,
        'top_categories': category_alerts
    }

def display_quality_case_dashboard(screened_df):
    st.markdown("### 📊 Quality Case Dashboard")

    action_counts = screened_df['Recommended_Action'].value_counts().to_dict()
    total_count = len(screened_df)
    escalations = action_counts.get('Escalate to Quality Case', 0) + action_counts.get('Escalate to Quality Case (Immediate)', 0)
    monitor_count = action_counts.get('Monitor', 0)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Products", total_count)
    with col2:
        st.metric("Escalations", escalations)
    with col3:
        st.metric("Monitoring", monitor_count)

    st.markdown("#### 📌 Action Breakdown")
    breakdown_df = pd.DataFrame(
        list(action_counts.items()),
        columns=['Recommended Action', 'Count']
    ).sort_values('Count', ascending=False)
    st.dataframe(breakdown_df, use_container_width=True)

    st.markdown("#### 🔎 Highest Return Rates")
    st.dataframe(
        screened_df[['SKU', 'Category', 'Return_Rate_Display', 'Recommended_Action']]
        .assign(Return_Rate_Sort=screened_df['Return_Rate'])
        .sort_values('Return_Rate_Sort', ascending=False)
        .drop(columns=['Return_Rate_Sort'])
        .head(10),
        use_container_width=True
    )

def parse_numeric(series):
    return pd.to_numeric(series, errors='coerce').fillna(0)

def load_quality_case_file(file_content, filename):
    if filename.endswith('.csv'):
        return pd.read_csv(io.BytesIO(file_content))
    if filename.endswith(('.xlsx', '.xls')):
        return pd.read_excel(io.BytesIO(file_content))
    if filename.endswith('.txt'):
        return pd.read_csv(io.BytesIO(file_content), sep='\t')
    st.error(f"Unsupported file type: {filename}")
    return None

def one_way_anova(groups):
    all_values = np.concatenate([g for g in groups if len(g) > 0])
    if len(all_values) == 0 or len(groups) < 2:
        return None

    grand_mean = np.mean(all_values)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups if len(g) > 0)
    ss_within = sum(np.sum((g - np.mean(g)) ** 2) for g in groups if len(g) > 0)

    df_between = len(groups) - 1
    df_within = len(all_values) - len(groups)
    if df_within <= 0:
@@ -980,166 +1139,338 @@ def main():
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

    # -------------------------
    # TAB 3: Quality Case Screening
    # -------------------------
    with tab3:
        st.markdown("### 🧪 Quality Case Screening & Escalation")
        st.markdown("""
        <div style="background: rgba(255, 0, 110, 0.1); border: 1px solid var(--secondary); 
                    border-radius: 8px; padding: 0.8rem; margin-bottom: 1rem;">
            <strong>📌 Goal:</strong> Screen SKUs for Quality Case escalation using SOP thresholds,
            ANOVA/MANOVA checks, and calendar-based monitoring.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 📋 Excel Column Requirements")
        st.dataframe(quality_case_requirements_df(), use_container_width=True)
        st.download_button(
            label="⬇️ Download Quality Case Template",
            data=build_quality_case_template(),
            file_name="quality_case_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

        st.divider()

        st.markdown("#### ✍️ Manual Entry (One Product at a Time)")
        with st.form("qc_manual_entry_form", clear_on_submit=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                manual_sku = st.text_input("SKU")
                manual_category = st.text_input("Category")
            with col2:
                manual_sold = st.number_input("Units Sold", min_value=0, step=1)
                manual_returned = st.number_input("Units Returned", min_value=0, step=1)
            with col3:
                manual_landed_cost = st.number_input("Landed Cost ($)", min_value=0.0, step=1.0)
                manual_retail_price = st.number_input("Retail Price ($)", min_value=0.0, step=1.0)

            with st.expander("Optional Risk Signals"):
                col4, col5, col6 = st.columns(3)
                with col4:
                    manual_launch_date = st.text_input("Launch Date (YYYY-MM-DD)")
                    manual_safety_risk = st.selectbox("Safety Risk", options=["", "Yes", "No"])
                with col5:
                    manual_zero_tolerance = st.selectbox("Zero Tolerance Component", options=["", "Yes", "No"])
                    manual_aql_fail = st.selectbox("AQL Fail", options=["", "Yes", "No"])
                with col6:
                    manual_complaint_count = st.number_input("Unique Complaints (30d)", min_value=0, step=1)
                    manual_sales_units_30d = st.number_input("Sales Units (30d)", min_value=0, step=1)
                    manual_sales_value_30d = st.number_input("Sales Value $ (30d)", min_value=0.0, step=10.0)

            submitted = st.form_submit_button("➕ Add Product")
            if submitted:
                if not manual_sku or not manual_category:
                    st.error("SKU and Category are required for manual entry.")
                else:
                    st.session_state.qc_manual_entries.append({
                        'SKU': manual_sku,
                        'Category': manual_category,
                        'Sold': manual_sold,
                        'Returned': manual_returned,
                        'Landed Cost': manual_landed_cost,
                        'Retail Price': manual_retail_price,
                        'Launch Date': manual_launch_date.strip() if manual_launch_date else '',
                        'Safety Risk': manual_safety_risk,
                        'Zero Tolerance Component': manual_zero_tolerance,
                        'AQL Fail': manual_aql_fail,
                        'Unique Complaint Count (30d)': manual_complaint_count,
                        'Sales Units (30d)': manual_sales_units_30d,
                        'Sales Value (30d)': manual_sales_value_30d,
                        'Input Source': 'Manual'
                    })
                    st.success(f"Added {manual_sku} to the manual intake list.")

        if st.session_state.qc_manual_entries:
            st.markdown("#### ✅ Manual Entries")
            st.dataframe(pd.DataFrame(st.session_state.qc_manual_entries), use_container_width=True)
            if st.button("🧹 Clear Manual Entries", use_container_width=True):
                st.session_state.qc_manual_entries = []
                st.rerun()

        st.divider()

        st.markdown("#### ⚙️ Screening Thresholds")
        col4, col5, col6 = st.columns(3)
        with col4:
            return_rate_cap = st.number_input("Return Rate Hard Cap", value=0.25, step=0.01, format="%.2f")
            relative_threshold = st.number_input("Relative Threshold (x Category Avg)", value=1.20, step=0.05)
        with col5:
            cat_avg_delta = st.number_input("Category Avg + Delta (5% = 0.05)", value=0.05, step=0.01)
            low_volume_cutoff = st.number_input("Low Volume Cutoff (Units Sold)", value=10, step=1)
        with col6:
            use_benchmarks = st.checkbox("Use SOP Category Benchmarks", value=True)
            review_days = st.number_input("Monitor Review in X Days", value=14, step=1)

        st.markdown("#### 🚨 Immediate Escalation Criteria")
        col7, col8, col9 = st.columns(3)
        with col7:
            landed_cost_threshold = st.number_input("Landed Cost Threshold ($)", value=150.0, step=10.0)
            retail_price_threshold = st.number_input("Retail Price Threshold ($)", value=0.0, step=10.0)
        with col8:
            launch_days = st.number_input("New Launch Window (Days)", value=90, step=5)
        with col9:
            st.caption("Optional flags are mapped in the file uploader below.")

        qc_file = st.file_uploader(
            "Upload Quality Screening Data (CSV/XLSX)",
            type=['csv', 'xlsx', 'xls', 'txt'],
            key="tab3_uploader"
        )

        upload_mapping = {}
        normalized_upload = None
        if qc_file:
            qc_df = load_quality_case_file(qc_file.read(), qc_file.name)

            if qc_df is not None:
                st.markdown("#### 🧭 Column Mapping")
                columns = qc_df.columns.tolist()

                col1, col2, col3 = st.columns(3)
                with col1:
                    upload_mapping['SKU'] = st.selectbox("SKU Column", options=columns, index=0)
                    upload_mapping['Category'] = st.selectbox("Category Column", options=columns, index=min(1, len(columns) - 1))
                    upload_mapping['Sold'] = st.selectbox("Sold Column", options=columns)
                with col2:
                    upload_mapping['Returned'] = st.selectbox("Returned Column", options=columns)
                    upload_mapping['Landed Cost'] = st.selectbox("Landed Cost Column (Optional)", options=[""] + columns)
                    upload_mapping['Retail Price'] = st.selectbox("Retail Price Column (Optional)", options=[""] + columns)
                with col3:
                    upload_mapping['Launch Date'] = st.selectbox("Launch Date Column (Optional)", options=[""] + columns)
                    upload_mapping['Safety Risk'] = st.selectbox("Safety Risk Column (Optional)", options=[""] + columns)
                    upload_mapping['Zero Tolerance Component'] = st.selectbox("Zero Tolerance Component Column (Optional)", options=[""] + columns)

                st.markdown("#### 🔎 Additional Optional Columns")
                col7, col8, col9 = st.columns(3)
                with col7:
                    upload_mapping['AQL Fail'] = st.selectbox("AQL Fail Column (Optional)", options=[""] + columns)
                with col8:
                    upload_mapping['Unique Complaint Count (30d)'] = st.selectbox(
                        "Unique Complaint Count (30d) (Optional)", options=[""] + columns
                    )
                    upload_mapping['Sales Units (30d)'] = st.selectbox(
                        "Sales Units (30d) (Optional)", options=[""] + columns
                    )
                with col9:
                    upload_mapping['Sales Value (30d)'] = st.selectbox(
                        "Sales Value $ (30d) (Optional)", options=[""] + columns
                    )
                normalized_upload = normalize_quality_case_data(qc_df, upload_mapping, "Upload")

        manual_df = pd.DataFrame(st.session_state.qc_manual_entries)
        if manual_df.empty:
            manual_df = pd.DataFrame(columns=QUALITY_CASE_STANDARD_COLUMNS)

        combined_frames = []
        if normalized_upload is not None:
            combined_frames.append(normalized_upload)
        if not manual_df.empty:
            combined_frames.append(manual_df)

        if combined_frames:
            combined_df = pd.concat(combined_frames, ignore_index=True)
            st.session_state.qc_combined_data = combined_df
            st.markdown("#### 📥 Combined Intake Preview")
            st.dataframe(combined_df.head(10), use_container_width=True)
        else:
            combined_df = None

        def resolve_optional_column(name, data_frame):
            if name in data_frame.columns:
                series = data_frame[name].replace('', np.nan)
                if series.notna().any():
                    return name
            return None

        if combined_df is not None and not combined_df.empty:
            if st.button("🔍 Run Quality Case Screening", type="primary"):
                config = {
                    'sold_col': 'Sold',
                    'returned_col': 'Returned',
                    'category_col': 'Category',
                    'use_benchmarks': use_benchmarks,
                    'return_rate_cap': return_rate_cap,
                    'relative_threshold': relative_threshold,
                    'cat_avg_delta': cat_avg_delta,
                    'low_volume_cutoff': int(low_volume_cutoff),
                    'review_days': int(review_days),
                    'safety_col': resolve_optional_column('Safety Risk', combined_df),
                    'zero_tolerance_col': resolve_optional_column('Zero Tolerance Component', combined_df),
                    'landed_cost_col': resolve_optional_column('Landed Cost', combined_df),
                    'retail_price_col': resolve_optional_column('Retail Price', combined_df),
                    'launch_date_col': resolve_optional_column('Launch Date', combined_df),
                    'launch_days': int(launch_days),
                    'aql_fail_col': resolve_optional_column('AQL Fail', combined_df),
                    'landed_cost_threshold': landed_cost_threshold,
                    'retail_price_threshold': retail_price_threshold,
                    'complaint_count_col': resolve_optional_column('Unique Complaint Count (30d)', combined_df),
                    'sales_units_30d_col': resolve_optional_column('Sales Units (30d)', combined_df),
                    'sales_value_30d_col': resolve_optional_column('Sales Value (30d)', combined_df),
                }

                with st.spinner("Analyzing quality triggers..."):
                    screened = compute_quality_case_screening(combined_df, config)

                st.session_state.qc_screened_data = screened
                st.session_state.qc_ai_review = None

                export_buffer = io.BytesIO()
                with pd.ExcelWriter(export_buffer, engine='xlsxwriter') as writer:
                    screened.to_excel(writer, index=False, sheet_name='Quality Screening')
                export_buffer.seek(0)
                st.session_state.qc_export_data = export_buffer
                st.session_state.qc_export_filename = f"quality_screening_{datetime.now().strftime('%Y%m%d')}.xlsx"

        if st.session_state.qc_screened_data is not None:
            screened = st.session_state.qc_screened_data
            display_quality_case_dashboard(screened)

            st.markdown("#### ✅ Screening Results")
            st.dataframe(
                screened[
                    [
                        'SKU',
                        'Category',
                        'Return_Rate_Display',
                        'Cat_Avg_Display',
                        'Recommended_Action',
                        'Review_By',
                        'Input Source'
                    ]
                ],
                use_container_width=True
            )

            st.markdown("#### 📊 ANOVA / MANOVA Summary")
            grouped = [
                parse_numeric(screened.loc[screened['Category'] == cat, 'Return_Rate']).values
                for cat in screened['Category'].dropna().unique()
            ]
            anova_results = one_way_anova(grouped)
            if anova_results:
                st.info(
                    f"ANOVA F={anova_results['f_stat']:.4f} "
                    f"(df={anova_results['df_between']}, {anova_results['df_within']})"
                )
            else:
                st.warning("ANOVA requires at least two categories with data.")

            manova_options = [col for col in QUALITY_CASE_STANDARD_COLUMNS if col != 'Input Source']
            manova_metric_1 = st.selectbox("MANOVA Metric 1", options=manova_options, index=2)
            manova_metric_2 = st.selectbox("MANOVA Metric 2", options=manova_options, index=3)

            manova_groups = []
            for cat in screened['Category'].dropna().unique():
                subset = screened[screened['Category'] == cat][[manova_metric_1, manova_metric_2]]
                subset = subset.apply(parse_numeric)
                if len(subset) > 1:
                    manova_groups.append(subset.values)

            manova_results = manova_wilks_lambda(manova_groups, [manova_metric_1, manova_metric_2])
            if manova_results:
                st.info(f"MANOVA Wilks' Lambda={manova_results['wilks_lambda']:.4f}")
            else:
                st.warning("MANOVA requires at least two categories with 2+ rows.")

            st.markdown("#### ⬇️ Export Screening Data")
            st.download_button(
                label="Download Screening Results",
                data=st.session_state.qc_export_data,
                file_name=st.session_state.qc_export_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )

            with st.expander("💬 More (AI Review + Chat)"):
                st.markdown("#### 🤖 AI Quality Review")
                if st.button("🧠 Generate AI Review", use_container_width=True):
                    analyzer = get_ai_analyzer()
                    summary = build_quality_case_summary(screened)
                    system_prompt = (
                        "You are a quality analyst reviewing screening results. "
                        "Summarize key risks, escalations, and recommended next actions. "
                        "Keep the response structured and concise."
                    )
                    prompt = (
                        "Analyze the screening summary and provide a short executive readout.\n\n"
                        f"Summary:\n{json.dumps(summary, indent=2)}"
                    )
                    with st.spinner("Generating AI review..."):
                        response = analyzer.generate_text(prompt, system_prompt, mode='summary')
                    st.session_state.qc_ai_review = response or "AI review unavailable."

                if st.session_state.qc_ai_review:
                    st.write(st.session_state.qc_ai_review)

                st.markdown("#### 💬 Quality Case Chat")
                for message in st.session_state.qc_chat_history:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                chat_prompt = st.chat_input("Ask about screening results or column requirements...")
                if chat_prompt:
                    st.session_state.qc_chat_history.append({"role": "user", "content": chat_prompt})
                    context_summary = build_quality_case_summary(screened)
                    system_prompt = (
                        "You are a quality case screening assistant. Answer using the provided summary and "
                        "column requirements. If the user asks for details not present, say that you do not have "
                        "that data. Provide actionable, concise answers."
                    )
                    prompt = (
                        f"Column requirements:\n{json.dumps(QUALITY_CASE_COLUMNS, indent=2)}\n\n"
                        f"Screening summary:\n{json.dumps(context_summary, indent=2)}\n\n"
                        f"User question: {chat_prompt}"
                    )
                    analyzer = get_ai_analyzer()
                    with st.spinner("Thinking..."):
                        response = analyzer.generate_text(prompt, system_prompt, mode='chat')
                    assistant_message = response or "I'm unable to answer that right now."
                    st.session_state.qc_chat_history.append({"role": "assistant", "content": assistant_message})
                    st.rerun()
        elif combined_df is None or combined_df.empty:
            st.info("Add manual entries or upload a file to run screening.")

if __name__ == "__main__":
    main()
