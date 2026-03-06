"""
B2B Zendesk Reporting Module
─────────────────────────────
Replaces the legacy Customer Service Report with a single, sortable
quality-issues table aggregated by **Parent SKU** (first 7 characters).

Integration:
    1. Import this module in app.py
    2. Add to TASK_DEFINITIONS
    3. Wire into render_single_tool / render_all_tabs
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from io import BytesIO
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────────────

PARENT_SKU_LENGTH = 7

EXCLUDED_SKUS = {"x", ".", "X", "XX", "XXX", "No SKU", ""}

MODULE_DESCRIPTION = """
<div style="background: rgba(0, 217, 255, 0.08); border: 1px solid #23b2be;
            border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 1.2rem;">
    <strong style="color:#004366;">📌 Purpose:</strong>
    Analyse B2C Zendesk quality-issue recordings and produce a consolidated
    quality report grouped by <strong>Parent SKU</strong> (first 7 characters).<br>
    <strong style="color:#004366;">📊 Output:</strong>
    One table — sorted by issue occurrence — with SKU-level breakdowns, top issues,
    ticket-type mix, and order-source distribution.  Replaces the manual
    <em>Customer Service Report</em> format.
</div>
"""

VIVE_TEAL = "#23b2be"
VIVE_NAVY = "#004366"


# ─── Data Loading & Cleaning ────────────────────────────────────────────────

def load_zendesk_data(uploaded_file) -> Optional[pd.DataFrame]:
    """Load and deduplicate the B2C Quality Issues Recordings file."""
    try:
        raw = pd.read_excel(uploaded_file)

        # Validate expected columns
        required = {"Ticket created - Date", "Ticket ID", "SKU", "Quality Issues?", "Issue", "Ticket Type"}
        missing = required - set(raw.columns)
        if missing:
            st.error(f"Missing expected columns: {', '.join(missing)}")
            return None

        # Deduplicate (source file has 28× duplication per ticket)
        df = raw.drop_duplicates(subset="Ticket ID").copy()
        logger.info(f"Loaded {len(raw):,} rows → {len(df):,} unique tickets")

        # Coerce dates
        df["Ticket created - Date"] = pd.to_datetime(df["Ticket created - Date"], errors="coerce")

        return df

    except Exception as e:
        st.error(f"Failed to read file: {e}")
        logger.error(f"load_zendesk_data error: {e}", exc_info=True)
        return None


def filter_by_date(df: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    mask = (df["Ticket created - Date"] >= pd.Timestamp(start)) & (
        df["Ticket created - Date"] <= pd.Timestamp(end) + timedelta(hours=23, minutes=59, seconds=59)
    )
    return df.loc[mask].copy()


# ─── Core Aggregation ────────────────────────────────────────────────────────

def build_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the consolidated quality-issues report.
    Returns one row per Parent SKU, sorted descending by issue count.
    """
    qi = df[df["Quality Issues?"] == True].copy()

    if qi.empty:
        return pd.DataFrame()

    # Clean SKUs — drop known junk
    qi = qi[~qi["SKU"].str.strip().isin(EXCLUDED_SKUS)]
    qi = qi[qi["SKU"].notna()]

    # Parent SKU = LEFT(SKU, 7)
    qi["Parent SKU"] = qi["SKU"].str[:PARENT_SKU_LENGTH]

    # ── Per-parent aggregations ──────────────────────────────────────────
    grouped = qi.groupby("Parent SKU", sort=False)

    agg = grouped.agg(
        Quality_Issue_Count=("Ticket ID", "count"),
        Unique_SKUs=("SKU", "nunique"),
        First_Seen=("Ticket created - Date", "min"),
        Last_Seen=("Ticket created - Date", "max"),
    ).reset_index()

    # SKU breakdown string  (e.g. "MOB1027BLU ×2, MOB1027RED ×1")
    def sku_breakdown(grp):
        counts = grp["SKU"].value_counts()
        parts = [f"{sku} ×{cnt}" for sku, cnt in counts.items()]
        return "; ".join(parts)

    sku_notes = grouped.apply(sku_breakdown, include_groups=False).reset_index()
    sku_notes.columns = ["Parent SKU", "SKU Breakdown"]

    # Top 3 issues
    def top_issues(grp, n=3):
        counts = grp["Issue"].value_counts().head(n)
        parts = [f"• {issue} ({cnt})" for issue, cnt in counts.items()]
        return "\n".join(parts)

    issue_notes = grouped.apply(top_issues, include_groups=False).reset_index()
    issue_notes.columns = ["Parent SKU", "Top Issues"]

    # Ticket-type mix
    def type_mix(grp):
        counts = grp["Ticket Type"].value_counts()
        parts = [f"{ttype}: {cnt}" for ttype, cnt in counts.items()]
        return "; ".join(parts)

    type_notes = grouped.apply(type_mix, include_groups=False).reset_index()
    type_notes.columns = ["Parent SKU", "Ticket Type Breakdown"]

    # Order source mix
    def source_mix(grp):
        counts = grp["Order source"].value_counts()
        parts = [f"{src}: {cnt}" for src, cnt in counts.items()]
        return "; ".join(parts)

    source_notes = grouped.apply(source_mix, include_groups=False).reset_index()
    source_notes.columns = ["Parent SKU", "Order Source Breakdown"]

    # Return-completed and replacement rates
    rate_agg = grouped.agg(
        Returns_Completed=("Return completed?", "sum"),
        Replacements_Sent=("Replacement SO", lambda x: (x.str.strip() != "").sum()),
    ).reset_index()

    # ── Merge everything ─────────────────────────────────────────────────
    report = (
        agg.merge(sku_notes, on="Parent SKU")
        .merge(issue_notes, on="Parent SKU")
        .merge(type_notes, on="Parent SKU")
        .merge(source_notes, on="Parent SKU")
        .merge(rate_agg, on="Parent SKU")
    )

    # Sort descending by issue count
    report = report.sort_values("Quality_Issue_Count", ascending=False).reset_index(drop=True)
    report.index = report.index + 1  # 1-based rank
    report.index.name = "Rank"

    # Clean column names for display
    report = report.rename(columns={
        "Quality_Issue_Count": "Quality Issues",
        "Unique_SKUs": "Variant Count",
        "First_Seen": "First Seen",
        "Last_Seen": "Last Seen",
        "Returns_Completed": "Returns Completed",
        "Replacements_Sent": "Replacements Sent",
    })

    return report


# ─── KPI helpers ─────────────────────────────────────────────────────────────

def compute_kpis(df: pd.DataFrame, report: pd.DataFrame) -> Dict:
    qi = df[df["Quality Issues?"] == True]
    total_tickets = len(df)
    quality_tickets = len(qi)
    quality_rate = quality_tickets / total_tickets * 100 if total_tickets else 0
    products_affected = len(report)
    top_product = report.iloc[0]["Parent SKU"] if not report.empty else "N/A"
    top_count = int(report.iloc[0]["Quality Issues"]) if not report.empty else 0

    return {
        "total_tickets": total_tickets,
        "quality_tickets": quality_tickets,
        "quality_rate": quality_rate,
        "products_affected": products_affected,
        "top_product": top_product,
        "top_count": top_count,
    }


# ─── Excel Export ────────────────────────────────────────────────────────────

def export_report_xlsx(report: pd.DataFrame, kpis: dict, date_label: str) -> bytes:
    """Export to a formatted .xlsx with openpyxl styling."""
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        report.to_excel(writer, sheet_name="Quality Issues Report", startrow=4)
        wb = writer.book
        ws = writer.sheets["Quality Issues Report"]

        # ── Header block ──
        header_fill = PatternFill("solid", fgColor="004366")
        header_font = Font(name="Arial", bold=True, color="FFFFFF", size=14)
        ws.merge_cells("A1:K1")
        ws["A1"].value = f"B2B Zendesk Quality Report — {date_label}"
        ws["A1"].font = header_font
        ws["A1"].fill = header_fill
        ws["A1"].alignment = Alignment(horizontal="center", vertical="center")
        ws.row_dimensions[1].height = 36

        sub_font = Font(name="Arial", size=10, color="004366")
        ws.merge_cells("A2:K2")
        ws["A2"].value = (
            f"Total Tickets: {kpis['total_tickets']:,}  |  "
            f"Quality Issues: {kpis['quality_tickets']:,}  ({kpis['quality_rate']:.1f}%)  |  "
            f"Products Affected: {kpis['products_affected']}  |  "
            f"Top Product: {kpis['top_product']} ({kpis['top_count']})"
        )
        ws["A2"].font = sub_font
        ws["A2"].alignment = Alignment(horizontal="center")

        ws.merge_cells("A3:K3")
        ws["A3"].value = f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        ws["A3"].font = Font(name="Arial", size=9, italic=True, color="666666")
        ws["A3"].alignment = Alignment(horizontal="center")

        # ── Column header formatting ──
        col_header_fill = PatternFill("solid", fgColor="23B2BE")
        col_header_font = Font(name="Arial", bold=True, color="FFFFFF", size=10)
        thin_border = Border(
            left=Side(style="thin", color="CCCCCC"),
            right=Side(style="thin", color="CCCCCC"),
            top=Side(style="thin", color="CCCCCC"),
            bottom=Side(style="thin", color="CCCCCC"),
        )

        for col_idx in range(1, ws.max_column + 1):
            cell = ws.cell(row=5, column=col_idx)
            cell.font = col_header_font
            cell.fill = col_header_fill
            cell.alignment = Alignment(horizontal="center", wrap_text=True)
            cell.border = thin_border

        # ── Data formatting ──
        alt_fill = PatternFill("solid", fgColor="F0FAFB")
        data_font = Font(name="Arial", size=10)
        for row_idx in range(6, ws.max_row + 1):
            for col_idx in range(1, ws.max_column + 1):
                cell = ws.cell(row=row_idx, column=col_idx)
                cell.font = data_font
                cell.border = thin_border
                cell.alignment = Alignment(vertical="top", wrap_text=True)
                if row_idx % 2 == 0:
                    cell.fill = alt_fill

        # ── Auto column widths ──
        col_widths = {
            "A": 6,   # Rank
            "B": 12,  # Parent SKU
            "C": 14,  # Quality Issues
            "D": 13,  # Variant Count
            "E": 13,  # First Seen
            "F": 13,  # Last Seen
            "G": 34,  # SKU Breakdown
            "H": 50,  # Top Issues
            "I": 36,  # Ticket Type Breakdown
            "J": 28,  # Order Source
            "K": 16,  # Returns Completed
            "L": 16,  # Replacements Sent
        }
        for col_letter, width in col_widths.items():
            ws.column_dimensions[col_letter].width = width

        ws.sheet_properties.pageSetUpPr = ws.sheet_properties.pageSetUpPr or None
        ws.freeze_panes = "A6"

    return buf.getvalue()


# ─── Streamlit UI ────────────────────────────────────────────────────────────

def render_b2b_zendesk_reporting():
    """Main render function — drop into app.py's task router."""

    st.markdown("### 🎫 B2B Zendesk Quality Reporting")
    st.markdown(MODULE_DESCRIPTION, unsafe_allow_html=True)

    # ── File upload ──────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload B2C Quality Issues Recordings (.xlsx)",
        type=["xlsx", "xls"],
        key="zendesk_uploader",
        help="Upload the B2C_QUALITY_ISSUES_RECORDINGS export from Zendesk.",
    )

    if not uploaded:
        st.info("Upload the **B2C_QUALITY_ISSUES_RECORDINGS** file to get started.")
        return

    # ── Load data ────────────────────────────────────────────────────────
    with st.spinner("Reading and deduplicating Zendesk data…"):
        df = load_zendesk_data(uploaded)

    if df is None or df.empty:
        return

    st.success(f"Loaded **{len(df):,}** unique tickets  •  "
               f"**{df['Quality Issues?'].sum():,}** flagged as quality issues")

    # ── Date range selector ──────────────────────────────────────────────
    st.markdown("#### 📅 Date Range")
    date_col = "Ticket created - Date"
    min_date = df[date_col].min().date()
    max_date = df[date_col].max().date()

    col_opt, col_range = st.columns([1, 3])
    with col_opt:
        use_all = st.checkbox("Use all dates", value=True, key="zendesk_all_dates")
    with col_range:
        if use_all:
            start_date, end_date = min_date, max_date
            st.caption(f"Full range: **{min_date}** → **{max_date}**")
        else:
            start_date, end_date = st.date_input(
                "Select range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                key="zendesk_date_range",
            )

    filtered = filter_by_date(df, start_date, end_date)
    qi_count = int(filtered["Quality Issues?"].sum())

    if qi_count == 0:
        st.warning("No quality issues in the selected date range.")
        return

    st.info(f"**{len(filtered):,}** tickets in range  •  **{qi_count}** quality issues")

    # ── Generate report ──────────────────────────────────────────────────
    if st.button("🚀 Generate Quality Report", type="primary", key="zendesk_run"):
        st.session_state["zendesk_report"] = None  # reset
        with st.spinner("Aggregating quality data by Parent SKU…"):
            report = build_quality_report(filtered)
            kpis = compute_kpis(filtered, report)
            date_label = (
                f"{start_date.strftime('%b %d')} – {end_date.strftime('%b %d, %Y')}"
                if not use_all
                else f"{min_date.strftime('%b %Y')} (All Data)"
            )
            st.session_state["zendesk_report"] = report
            st.session_state["zendesk_kpis"] = kpis
            st.session_state["zendesk_date_label"] = date_label
            st.session_state["zendesk_filtered"] = filtered

    # ── Display results ──────────────────────────────────────────────────
    if st.session_state.get("zendesk_report") is not None:
        report = st.session_state["zendesk_report"]
        kpis = st.session_state["zendesk_kpis"]
        date_label = st.session_state["zendesk_date_label"]
        filtered_data = st.session_state["zendesk_filtered"]

        if report.empty:
            st.warning("No quality issues found for the selected parameters.")
            return

        # ── KPI cards ────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown(f"#### 📊 Quality Summary — {date_label}")

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Total Tickets", f"{kpis['total_tickets']:,}")
        with k2:
            st.metric("Quality Issues", f"{kpis['quality_tickets']:,}")
        with k3:
            st.metric("Quality Rate", f"{kpis['quality_rate']:.1f}%")
        with k4:
            st.metric("Products Affected", kpis["products_affected"])

        # ── Main table ───────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 📋 Quality Issues by Product — Sorted by Occurrence")

        # Compact display columns
        display_cols = [
            "Parent SKU",
            "Quality Issues",
            "Variant Count",
            "SKU Breakdown",
            "Top Issues",
            "Ticket Type Breakdown",
            "Order Source Breakdown",
            "Returns Completed",
            "Replacements Sent",
            "First Seen",
            "Last Seen",
        ]

        st.dataframe(
            report[display_cols],
            use_container_width=True,
            height=min(600, 38 + 35 * len(report)),
            column_config={
                "Quality Issues": st.column_config.NumberColumn(
                    "Quality Issues", help="Count of quality-issue tickets", format="%d"
                ),
                "Variant Count": st.column_config.NumberColumn(
                    "Variants", help="Distinct SKUs under this parent", format="%d"
                ),
                "First Seen": st.column_config.DateColumn("First Seen", format="YYYY-MM-DD"),
                "Last Seen": st.column_config.DateColumn("Last Seen", format="YYYY-MM-DD"),
                "Returns Completed": st.column_config.NumberColumn(format="%d"),
                "Replacements Sent": st.column_config.NumberColumn(format="%d"),
                "SKU Breakdown": st.column_config.TextColumn("SKU Breakdown", width="large"),
                "Top Issues": st.column_config.TextColumn("Top Issues", width="large"),
                "Ticket Type Breakdown": st.column_config.TextColumn("Ticket Types", width="medium"),
                "Order Source Breakdown": st.column_config.TextColumn("Order Sources", width="medium"),
            },
        )

        st.caption(f"Showing **{len(report)}** parent SKUs with quality issues.")

        # ── Top offenders bar chart ──────────────────────────────────────
        if len(report) > 1:
            st.markdown("#### 🏷️ Top Quality-Issue Products")
            chart_data = report.head(15)[["Parent SKU", "Quality Issues"]].set_index("Parent SKU")
            st.bar_chart(chart_data, color=VIVE_TEAL, horizontal=True)

        # ── Expandable details ───────────────────────────────────────────
        with st.expander("🔎 Drill Down — Full Issue Detail", expanded=False):
            qi_detail = filtered_data[filtered_data["Quality Issues?"] == True].copy()
            qi_detail["Parent SKU"] = qi_detail["SKU"].str[:PARENT_SKU_LENGTH]
            detail_cols = [
                "Ticket created - Date", "Ticket ID", "Parent SKU", "SKU",
                "Issue", "Ticket Type", "Order source",
                "Return completed?", "Replacement SO",
            ]
            available = [c for c in detail_cols if c in qi_detail.columns]
            st.dataframe(qi_detail[available].sort_values("Ticket created - Date", ascending=False),
                         use_container_width=True, height=400)

        # ── Export ───────────────────────────────────────────────────────
        st.markdown("---")
        col_dl1, col_dl2, col_clear = st.columns([2, 2, 1])

        with col_dl1:
            xlsx_bytes = export_report_xlsx(report, kpis, date_label)
            st.download_button(
                "⬇️ Download Report (.xlsx)",
                data=xlsx_bytes,
                file_name=f"B2B_Zendesk_Quality_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                key="zendesk_dl_xlsx",
            )

        with col_dl2:
            csv_bytes = report.to_csv(index=True).encode("utf-8")
            st.download_button(
                "⬇️ Download Report (.csv)",
                data=csv_bytes,
                file_name=f"B2B_Zendesk_Quality_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                key="zendesk_dl_csv",
            )

        with col_clear:
            if st.button("🔄 Reset", key="zendesk_clear"):
                for k in ["zendesk_report", "zendesk_kpis", "zendesk_date_label", "zendesk_filtered"]:
                    st.session_state.pop(k, None)
                st.rerun()


# ─── Integration Helpers ─────────────────────────────────────────────────────

# Session state keys this module uses (add to initialize_session_state)
ZENDESK_SESSION_DEFAULTS = {
    "zendesk_report": None,
    "zendesk_kpis": None,
    "zendesk_date_label": None,
    "zendesk_filtered": None,
}

# Task definition to merge into TASK_DEFINITIONS
ZENDESK_TASK_DEFINITION = {
    "icon": "🎫",
    "title": "B2B Zendesk Reporting",
    "subtitle": "Quality Issue Analysis",
    "description": (
        "Analyse Zendesk B2C quality-issue recordings.  Produces a consolidated "
        "report grouped by Parent SKU (first 7 chars) sorted by issue occurrence, "
        "with SKU breakdowns, top issues, ticket types, and order sources."
    ),
    "keywords": [
        "zendesk", "b2b", "quality issues", "customer service",
        "quality report", "zendesk report", "b2c quality",
    ],
}
