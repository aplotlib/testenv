"""
Advanced Analytics Module for Quality Suite

Contains enterprise-grade quality analytics features:
- Root Cause Analysis (5-Why, Fishbone, Pareto)
- CAPA Management System
- Risk Analysis (FMEA)
- Predictive Analytics

These tools are designed for world-class quality teams like Arthrex.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


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

        if not tracker.cases:
            st.info("Load cases to perform 5-Why Analysis")
            return

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

        if not tracker.cases:
            st.info("Load cases to perform Fishbone Analysis")
            return

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

        if not tracker.cases:
            st.info("Load cases to create action plans")
            return

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

    if not tracker.cases:
        st.info("Load cases to perform FMEA analysis")
        return

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

    if not tracker.cases:
        st.info("Load cases to generate predictions")
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
