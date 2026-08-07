"""
Vive Health — Returns Categorizer Suite
=======================================

A focused Streamlit app with three tools:

  1. Amazon Return Categorizer — categorizes FBA customer-return complaints into
     the standard 24-category quality taxonomy (Column I -> Column K).
  2. B2B Report — turns a raw Odoo Helpdesk export into the standard B2B Report
     (Display Name / Description / SKU / Category / Reason).
  3. Zendesk B2C Report — aggregates Zendesk quality tickets by parent SKU.

Design notes
------------
This app is a deliberate strip-down of the former 10k-line "Quality Suite"
monolith. The categorization logic is carried over VERBATIM because it is
accuracy-validated against a human-reviewed export (see validate_categorizer.py:
100% precision on the 3,024 rows the regex tier resolves, 66% of the file).
Do not "tidy" the pipeline without re-running that benchmark.

Two rules the pipeline depends on — both were previously the cause of a real
accuracy regression, so they are load-bearing:

  * The Amazon FBA reason-code column ('reason': UNWANTED_ITEM, DEFECTIVE, ...)
    must NEVER be mistaken for, or allowed to override, the free-text customer
    complaint. It is a weak hint only.
  * Complaint text is preprocessed (HTML-entity decode, contraction expansion)
    BEFORE pattern matching. Skipping this silently degrades accuracy.

Each optional module is imported independently. A failure in one tool must not
disable the others — previously a single bad import turned off AI everywhere.

Deployment: Streamlit Community Cloud. Set ANTHROPIC_API_KEY in Settings ->
Secrets. Optionally set APP_PASSCODE to require a shared passcode.
"""

import gc
import io
import logging
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Optional

import pandas as pd
import streamlit as st

from time_savings import (
    SECONDS_PER_AMAZON_RETURN,
    SECONDS_PER_SUPPORT_TICKET,
    Savings,
    estimate_amazon,
    estimate_tickets,
    format_duration,
)

# ── Core AI engine (required for categorization) ───────────────────────────────
try:
    from enhanced_ai_analysis import (
        AIProvider,
        EnhancedAIAnalyzer,
        FBA_REASON_MAP,
        MEDICAL_DEVICE_CATEGORIES,
    )
    AI_AVAILABLE = True
    AI_IMPORT_ERROR = None
except Exception as _e:  # noqa: BLE001 - must never hard-crash the app
    AI_AVAILABLE = False
    AI_IMPORT_ERROR = str(_e)
    FBA_REASON_MAP = {}
    MEDICAL_DEVICE_CATEGORIES = []
    AIProvider = None
    EnhancedAIAnalyzer = None

# ── Corrections memory (optional: enables "teach the AI" overrides) ────────────
try:
    from corrections_memory import get_corrections_memory
    CORRECTIONS_AVAILABLE = True
except Exception:  # noqa: BLE001
    CORRECTIONS_AVAILABLE = False
    get_corrections_memory = None

# ── Zendesk B2C reporting tool (optional: its own tab) ─────────────────────────
try:
    from b2b_zendesk_reporting import render_b2b_zendesk_reporting
    ZENDESK_AVAILABLE = True
    ZENDESK_IMPORT_ERROR = None
except Exception as _e:  # noqa: BLE001
    ZENDESK_AVAILABLE = False
    ZENDESK_IMPORT_ERROR = str(_e)
    render_b2b_zendesk_reporting = None

# ── Excel writer (falls back to CSV export when absent) ───────────────────────
try:
    import xlsxwriter  # noqa: F401
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Vive Health — Returns Categorizer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_CONFIG = {
    "title": "Vive Health Returns Categorizer",
    "version": "34.0",
    "chunk_sizes": [100, 250, 500, 1000],
    "default_chunk": 500,
}

COLORS = {
    "primary": "#23b2be",    # Vive turquoise
    "secondary": "#004366",  # Navy
    "accent": "#F0B323",
    "danger": "#EB3300",
}

# Category -> reporting group. Single source of truth, shared by the export
# sheets and the on-screen dashboard so the two can never disagree.
CATEGORY_GROUPS = {
    "Size: Too Small":                                "Size & Fit",
    "Size: Too Large":                                "Size & Fit",
    "Size: Doesn't Fit / Wrong Dimensions":           "Size & Fit",
    "Comfort: Causes Pain or Pressure":               "Comfort",
    "Comfort: Too Hard / Rigid":                      "Comfort",
    "Comfort: Too Soft / Lacks Support":              "Comfort",
    "Comfort: Skin Irritation or Allergic Reaction":  "Comfort",
    "Defect: Broken / Structural Failure":            "Product Defects",
    "Defect: Malfunctions / Stops Working":           "Product Defects",
    "Defect: Cosmetic Damage":                        "Product Defects",
    "Defect: Poor Material Quality":                  "Product Defects",
    "Performance: Ineffective / Doesn't Help":        "Performance & Compatibility",
    "Equipment Compatibility Issue":                  "Performance & Compatibility",
    "Stability: Shifts / Unstable / Falls":           "Stability",
    "Assembly / Usage Difficulty":                    "Assembly & Instructions",
    "Wrong Product / Not as Described":               "Order Accuracy",
    "Missing or Incomplete Components":               "Order Accuracy",
    "Customer: Changed Mind / No Longer Needed":      "Customer",
    "Customer: Ordered Wrong Size or Item":           "Customer",
    "Fulfillment: Damaged in Shipping":               "Fulfillment",
    "Fulfillment: Wrong Item Sent":                   "Fulfillment",
    "Fulfillment: Delivery Issue":                    "Fulfillment",
    "Medical / Safety Concern":                       "Medical & Safety",
    "Other / Miscellaneous":                          "Other",
}

# Categories that represent an actual product-quality problem, as opposed to a
# customer-driven or logistics return. Drives the "Is Quality Issue" column.
QUALITY_ISSUE_CATS = {
    "Size: Too Small", "Size: Too Large", "Size: Doesn't Fit / Wrong Dimensions",
    "Comfort: Causes Pain or Pressure", "Comfort: Too Hard / Rigid",
    "Comfort: Too Soft / Lacks Support", "Comfort: Skin Irritation or Allergic Reaction",
    "Defect: Broken / Structural Failure", "Defect: Malfunctions / Stops Working",
    "Defect: Cosmetic Damage", "Defect: Poor Material Quality",
    "Performance: Ineffective / Doesn't Help", "Equipment Compatibility Issue",
    "Stability: Shifts / Unstable / Falls", "Assembly / Usage Difficulty",
    "Missing or Incomplete Components", "Medical / Safety Concern",
}


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        .main-header {{
            background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
            padding: 1.4rem 1.8rem; border-radius: 10px; margin-bottom: 1.5rem;
        }}
        .main-header h1 {{ color: #fff; margin: 0; font-size: 1.7rem; }}
        .main-header p  {{ color: rgba(255,255,255,.92); margin: .35rem 0 0; font-size: .9rem; }}
        .tool-note {{
            border-left: 3px solid {COLORS['primary']};
            background: rgba(35,178,190,.08);
            padding: .75rem 1rem; border-radius: 6px; margin-bottom: 1rem; font-size: .9rem;
        }}
        div[data-testid="stMetricValue"] {{ font-size: 1.5rem; }}

        /* Time-saved hero panel — the headline number for a demo */
        .hero {{
            background: linear-gradient(135deg, #0f766e 0%, {COLORS['secondary']} 100%);
            border-radius: 14px; padding: 1.6rem 1.8rem; margin: .5rem 0 1.2rem;
            box-shadow: 0 8px 28px rgba(0,67,102,.28);
        }}
        .hero-label {{
            color: rgba(255,255,255,.82); font-size: .8rem; letter-spacing: .09em;
            text-transform: uppercase; margin: 0 0 .3rem;
        }}
        .hero-value {{
            color: #fff; font-size: 3.1rem; font-weight: 700; line-height: 1.05; margin: 0;
        }}
        .hero-sub {{ color: rgba(255,255,255,.9); font-size: .97rem; margin: .55rem 0 0; }}
        .hero-chips {{ display: flex; flex-wrap: wrap; gap: .5rem; margin-top: 1rem; }}
        .chip {{
            background: rgba(255,255,255,.16); color: #fff; border-radius: 999px;
            padding: .3rem .85rem; font-size: .85rem; white-space: nowrap;
        }}
        .chip b {{ font-weight: 700; }}
        .estimate-box {{
            border: 1px dashed {COLORS['primary']}; border-radius: 10px;
            background: rgba(35,178,190,.07); padding: .9rem 1.1rem; margin: .6rem 0 1rem;
        }}
        .estimate-box .big {{ font-size: 1.45rem; font-weight: 700; color: {COLORS['primary']}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_savings_hero(sv: Savings, *, noun: str, partial: bool = False) -> None:
    """The headline panel: how much analyst time this run replaced."""
    chips = [
        f"<span class='chip'>Manual: <b>{sv.manual_display}</b></span>",
        f"<span class='chip'>This tool: <b>{sv.actual_display}</b></span>",
    ]
    if sv.speedup > 0:
        chips.append(f"<span class='chip'><b>{sv.speedup_display}</b></span>")
    chips.append(f"<span class='chip'>{sv.items:,} {noun}</span>")
    chips.append(f"<span class='chip'>≈ <b>{sv.workdays_display}</b> of manual work</span>")

    label = "Analyst time saved (partial run)" if partial else "Analyst time saved"
    st.markdown(
        f"""
        <div class="hero">
            <p class="hero-label">⏱ {label}</p>
            <p class="hero-value">{sv.saved_display}</p>
            <p class="hero-sub">
                {sv.items:,} {noun} categorized · at {sv.seconds_per_item:g}s each by hand,
                this would have taken <b>{sv.manual_display}</b>
            </p>
            <div class="hero-chips">{''.join(chips)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pre_estimate(sv: Savings, noun: str) -> None:
    """Shown before processing: what this file is worth in manual hours."""
    st.markdown(
        f"""
        <div class="estimate-box">
            <span class="big">{sv.manual_display}</span>
            &nbsp;of manual work in this file
            &nbsp;<span style="opacity:.75">
            ({sv.items:,} {noun} × {sv.seconds_per_item:g}s each ≈ {sv.workdays_display})</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Session state / AI plumbing
# ══════════════════════════════════════════════════════════════════════════════

def initialize_session_state() -> None:
    defaults = {
        "ai_analyzer": None,
        "ai_provider": AIProvider.CLAUDE if AI_AVAILABLE else None,
        # Amazon categorizer
        "categorized_data": None,
        "processing_complete": False,
        "reason_summary": {},
        "product_summary": {},
        "batch_size": 20,
        "chunk_size": APP_CONFIG["default_chunk"],
        "processing_errors": [],
        "processing_speed": 0.0,
        "categorization_breakdown": None,
        "column_mapping": {},
        "export_data": None,
        "export_filename": None,
        # Run lifecycle: 'idle' | 'running' | 'complete' | 'interrupted'.
        # Anything found still 'running' on a fresh script run means the
        # previous run was killed (stop pressed, tab closed, session timeout).
        "run_state": "idle",
        "stop_requested": False,
        "rows_done": 0,
        "rows_target": 0,
        "processing_elapsed": 0.0,
        # index -> confidence / method, for the review queue. Kept out of the
        # DataFrame so it never leaks into the customer-facing export.
        "row_confidence": {},
        # Time-savings assumptions (adjustable so they can be defended live)
        "sec_per_return": float(SECONDS_PER_AMAZON_RETURN),
        "sec_per_ticket": float(SECONDS_PER_SUPPORT_TICKET),
        # B2B report
        "b2b_processed_data": None,
        "b2b_processing_complete": False,
        "b2b_elapsed": 0.0,
        "b2b_export_data": None,
        "b2b_export_filename": None,
        "b2b_perf_mode": "Small (< 500 rows)",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def check_api_keys() -> dict:
    """Find the Anthropic API key: Streamlit secrets first, then environment."""
    keys_found = {}
    try:
        if hasattr(st, "secrets"):
            for key in ["ANTHROPIC_API_KEY", "anthropic_api_key", "claude_api_key", "claude"]:
                try:
                    if key in st.secrets:
                        val = str(st.secrets[key]).strip()
                        if val.startswith("sk-ant-"):
                            keys_found["claude"] = val
                            os.environ["ANTHROPIC_API_KEY"] = val
                            break
                except Exception:  # noqa: BLE001
                    pass
        if "claude" not in keys_found:
            env_val = os.environ.get("ANTHROPIC_API_KEY", "").strip()
            if env_val.startswith("sk-ant-"):
                keys_found["claude"] = env_val
    except Exception as e:  # noqa: BLE001
        logger.warning("Error checking API keys: %s", e)
    if "claude" not in keys_found:
        logger.warning("ANTHROPIC_API_KEY not found — AI features disabled")
    return keys_found


def get_ai_analyzer(provider=None, max_workers: int = 3):
    """Get or create the analyzer. Returns None when AI is unavailable."""
    if not AI_AVAILABLE:
        return None
    if provider is None:
        provider = st.session_state.ai_provider

    existing = st.session_state.get("ai_analyzer")
    # Reuse only when both provider AND worker count match — a stale worker
    # count silently ignores the user's Data Volume selection.
    if (
        existing is not None
        and getattr(existing, "provider", None) == provider
        and getattr(existing, "max_workers", None) == max_workers
    ):
        return existing

    try:
        if not check_api_keys().get("claude"):
            logger.warning("AI analyzer requested but no API key found")
            return None
        st.session_state.ai_analyzer = EnhancedAIAnalyzer(provider, max_workers=max_workers)
        logger.info("AI analyzer ready: provider=%s workers=%d", provider.value, max_workers)
    except Exception as e:  # noqa: BLE001
        logger.error("AI analyzer init failed: %s", e, exc_info=True)
        st.session_state.ai_analyzer = None
    return st.session_state.ai_analyzer


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 1 — Amazon Return Categorizer
#
# Logic below is carried over verbatim from the validated implementation.
# ══════════════════════════════════════════════════════════════════════════════

def _fuzzy_find_column(cols: list, candidates: list, threshold: int = 70) -> Optional[str]:
    """Find the best-matching column name from a list of candidates."""
    try:
        from rapidfuzz import fuzz, process as rfprocess
        col_lower = [c.lower() for c in cols]
        best_col, best_score = None, 0
        for candidate in candidates:
            result = rfprocess.extractOne(candidate.lower(), col_lower, scorer=fuzz.partial_ratio)
            if result and result[1] > best_score and result[1] >= threshold:
                best_score = result[1]
                best_col = cols[col_lower.index(result[0])]
        return best_col
    except ImportError:
        return None


def process_file_preserve_structure(file_content, filename):
    """Read the upload and detect columns, preserving the original structure."""
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(file_content), dtype=str)
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(file_content), dtype=str)
        elif filename.endswith(".txt"):
            df = pd.read_csv(io.BytesIO(file_content), sep="\t", dtype=str)
        else:
            return None, None

        column_mapping = {}
        cols = df.columns.tolist()

        def _looks_like_code_column(col_name: str) -> bool:
            """True if a column mostly holds reason codes (UNWANTED_ITEM, DEFECTIVE)
            rather than free-text customer comments."""
            values = df[col_name].dropna().astype(str).str.strip()
            values = values[values != ""].head(200)
            if values.empty:
                return False
            return values.str.fullmatch(r"[A-Z0-9_\-]{2,40}").mean() >= 0.6

        # ── Detect the FBA reason-code column FIRST (it is NOT the complaint) ──
        # Amazon return reports carry a 'reason' column of codes like
        # UNWANTED_ITEM / DEFECTIVE. Categorizing those instead of the free-text
        # customer comments destroys accuracy — keep the two strictly separate.
        reason_candidates = ["reason", "return reason", "return-reason", "reason code", "reason-code"]
        cols_norm = {c: c.lower().strip().replace("-", " ").replace("_", " ") for c in cols}
        reason_col = next((c for c in cols if cols_norm[c] in reason_candidates), None)
        if reason_col:
            column_mapping["fba_reason"] = reason_col

        # ── Detect complaint column: exact match on free-text names first ──────
        complaint_candidates = [
            "customer-comments", "customer comments", "customer_comments",
            "complaint", "customer complaint", "complaint text", "comments",
            "customer feedback", "feedback", "issue description",
            "defect description", "description", "issue", "problem", "notes",
        ]
        complaint_col = None
        for cand in complaint_candidates:
            cand_norm = cand.replace("-", " ").replace("_", " ")
            match = next((c for c in cols if cols_norm[c] == cand_norm and c != reason_col), None)
            if match is not None:
                complaint_col = match
                break

        # Fuzzy fallback — never against the reason-code column
        if complaint_col is None:
            complaint_col = _fuzzy_find_column([c for c in cols if c != reason_col], complaint_candidates)

        # Reject a "complaint" column that actually contains reason codes
        if complaint_col is not None and _looks_like_code_column(complaint_col):
            logger.warning("Column '%s' looks like reason codes, not complaints — falling back", complaint_col)
            complaint_col = None

        # Fallback: position-based (column I = index 8), same sanity check
        if complaint_col is None and len(cols) > 8 and cols[8] != reason_col:
            if not _looks_like_code_column(cols[8]):
                complaint_col = cols[8]

        # ── Detect SKU column: exact, then fuzzy, then position ───────────────
        sku_candidates = [
            "sku", "imported sku", "asin", "product sku", "item sku", "product code",
            "item number", "model", "part number", "product id",
        ]
        sku_col = None
        for cand in sku_candidates:
            match = next((c for c in cols if cols_norm[c] == cand), None)
            if match is not None:
                sku_col = match
                break
        if sku_col is None:
            sku_col = _fuzzy_find_column(cols, sku_candidates)
        if sku_col is None and len(cols) > 1:
            sku_col = cols[1]

        if complaint_col:
            column_mapping["complaint"] = complaint_col
        if sku_col:
            column_mapping["sku"] = sku_col

        # ── Category output column — always column K, or append to reach it ────
        while len(df.columns) < 11:
            df[f"Column_{len(df.columns)}"] = ""
        column_mapping["category"] = df.columns[10]
        df[column_mapping["category"]] = ""

        if complaint_col:
            msg = f"Auto-detected: complaint text = **{complaint_col}**"
            if sku_col:
                msg += f", SKU = **{sku_col}**"
            if reason_col:
                msg += f", return-reason code = **{reason_col}** (hint only)"
            st.info(f"🔍 {msg}")
        elif len(cols) < 11:
            st.error("File structure not recognized. Need at least 11 columns (A–K).")
            return None, None

        return df, column_mapping
    except Exception as e:  # noqa: BLE001
        st.error(f"Error processing file: {e}")
        return None, None


def _export_filename(partial: bool = False) -> str:
    """Export filename. Partial runs are labelled in the name itself so a
    half-finished file can never be mistaken for a complete one downstream."""
    ext = "xlsx" if EXCEL_AVAILABLE else "csv"
    stamp = datetime.now().strftime("%Y%m%d")
    if partial:
        done = st.session_state.get("rows_done", 0)
        target = st.session_state.get("rows_target", 0)
        return f"categorized_PARTIAL_{done}of{target}_{stamp}.{ext}"
    return f"categorized_{stamp}.{ext}"


def count_uncategorized(df, column_mapping) -> int:
    """Eligible rows still lacking a category — what a Resume would process."""
    cat_col = column_mapping.get("category")
    comp_col = column_mapping.get("complaint")
    if not cat_col or cat_col not in df.columns:
        return 0
    empty = df[cat_col].isna() | (df[cat_col].astype(str).str.strip() == "")
    if comp_col and comp_col in df.columns:
        has_text = df[comp_col].notna() & (df[comp_col].astype(str).str.strip() != "")
        empty = empty & has_text
    return int(empty.sum())


def process_in_chunks(df, analyzer, column_mapping, chunk_size=None,
                      only_uncategorized: bool = False, limit: Optional[int] = None):
    """Categorize in chunks, reporting live progress.

    Partial work is durable: `df` is published into session state up front and
    mutated in place, and the downloadable export is regenerated after every
    chunk. So if this run is killed — Stop pressed, tab closed, session timing
    out — everything completed so far is still there and still exportable.

    only_uncategorized: skip rows that already have a category (used by Resume).
    limit:              cap the number of rows processed (used by sample runs).
    """
    if chunk_size is None:
        chunk_size = st.session_state.chunk_size

    complaint_col = column_mapping.get("complaint")
    category_col = column_mapping.get("category")

    if not category_col:
        st.error("Column mapping incomplete — could not determine the category output column.")
        return df

    fba_col = column_mapping.get("fba_reason")

    # Rows to process: any with free-text complaints, plus rows that only carry a
    # recognized Amazon reason code (categorized via the code map).
    if complaint_col:
        valid_mask = df[complaint_col].notna() & (df[complaint_col].str.strip() != "")
    else:
        valid_mask = pd.Series(False, index=df.index)
    text_rows = int(valid_mask.sum())

    code_only_rows = 0
    if fba_col and isinstance(FBA_REASON_MAP, dict) and FBA_REASON_MAP:
        code_mask = df[fba_col].isin(list(FBA_REASON_MAP.keys())) & ~valid_mask
        code_only_rows = int(code_mask.sum())
        valid_mask = valid_mask | code_mask

    # Resume: leave already-filled categories alone.
    if only_uncategorized:
        done_mask = df[category_col].notna() & (df[category_col].astype(str).str.strip() != "")
        valid_mask = valid_mask & ~done_mask

    valid_indices = df[valid_mask].index
    if limit is not None:
        valid_indices = valid_indices[: max(0, int(limit))]
    total_valid = len(valid_indices)

    if total_valid == 0:
        if only_uncategorized:
            st.success("Nothing left to categorize — every eligible row already has a category.")
        else:
            st.warning("No complaint text or recognizable return-reason codes found to process.")
        return df

    # ── Make partial work durable BEFORE the first API call ────────────────
    # df is mutated in place below, so session state sees each write as it
    # happens. This is what allows a killed run to still be exported.
    st.session_state.categorized_data = df
    st.session_state.column_mapping = column_mapping
    st.session_state.run_state = "running"
    st.session_state.stop_requested = False
    st.session_state.rows_done = 0
    st.session_state.rows_target = total_valid

    code_note = f" (+ **{code_only_rows:,}** rows with only a reason code)" if code_only_rows else ""
    scope_note = " · **resuming** (skipping already-categorized rows)" if only_uncategorized else ""
    if limit is not None:
        scope_note += f" · **sample run** capped at {total_valid:,} rows"
    st.info(
        f"📊 Categorizing **{total_valid:,}** of {text_rows:,} complaints{code_note} · "
        f"chunk size **{chunk_size}** · API batch **{st.session_state.batch_size}**{scope_note}"
    )
    st.caption(
        "⏸ Safe to stop at any time — press **Stop** below, or just close the tab. "
        "Everything already categorized is kept and stays downloadable."
    )

    # Rendered before the loop: clicking it triggers a Streamlit rerun, which
    # ends this run. Partial results survive via the in-place mutation above.
    if st.button("⏸ Stop and keep what's done", key="stop_categorization"):
        st.session_state.stop_requested = True
        st.rerun()

    progress_bar = st.progress(0)
    status_text = st.empty()
    stats_container = st.container()

    processed_count = 0
    method_counts = {"corrections": 0, "instant": 0, "ai": 0, "failed": 0}
    start_time = time.time()

    for chunk_start in range(0, total_valid, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_valid)
        chunk_indices = valid_indices[chunk_start:chunk_end]
        chunk_num = (chunk_start // chunk_size) + 1
        total_chunks = (total_valid + chunk_size - 1) // chunk_size

        batch_data = []
        for idx in chunk_indices:
            complaint = ""
            if complaint_col:
                raw_complaint = df.at[idx, complaint_col]
                if pd.notna(raw_complaint):
                    complaint = str(raw_complaint).strip()

            # FBA reason code is customer-selected and unreliable — hint only.
            fba_reason = None
            if fba_col and fba_col in df.columns:
                raw_reason = df.at[idx, fba_col]
                if pd.notna(raw_reason):
                    fba_reason = str(raw_reason).strip()

            batch_data.append({"index": idx, "complaint": complaint, "fba_reason": fba_reason})

        try:
            sub_batch_size = st.session_state.batch_size
            for i in range(0, len(batch_data), sub_batch_size):
                sub_batch = batch_data[i:i + sub_batch_size]
                results = analyzer.categorize_batch(sub_batch, mode="standard")

                for result in results:
                    idx = result["index"]
                    df.at[idx, category_col] = result.get("category", "Other / Miscellaneous")

                    # Confidence encodes which path produced the answer.
                    conf = result.get("confidence", 0)
                    if conf >= 1.0:
                        method_counts["corrections"] += 1
                    elif conf >= 0.9:
                        method_counts["instant"] += 1
                    elif conf >= 0.5:
                        method_counts["ai"] += 1
                    else:
                        method_counts["failed"] += 1
                    # Retained for the review queue — lowest confidence first.
                    st.session_state.row_confidence[idx] = conf
                    processed_count += 1

                progress_bar.progress(processed_count / total_valid)
                elapsed = time.time() - start_time
                speed = processed_count / elapsed if elapsed > 0 else 0
                remaining = (total_valid - processed_count) / speed if speed > 0 else 0

                # Publish progress every sub-batch so an interrupted run still
                # reports accurately on the next script run.
                st.session_state.rows_done = processed_count
                st.session_state.processing_elapsed = elapsed
                st.session_state.categorization_breakdown = dict(method_counts)

                sv = estimate_amazon(processed_count, actual_seconds=elapsed,
                                     seconds_per_item=st.session_state.sec_per_return)
                with stats_container:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Progress", f"{processed_count:,}/{total_valid:,}")
                    c2.metric("Speed", f"{speed:.1f}/sec")
                    c3.metric("ETA", f"{int(remaining)}s" if remaining > 0 else "Complete")
                    c4.metric("⏱ Time saved so far", sv.saved_display)

                # Throttle between sub-batches to stay inside API rate limits.
                time.sleep(0.5)

        except Exception as e:  # noqa: BLE001
            logger.error("Chunk processing error: %s", e)
            st.session_state.processing_errors.append(f"Chunk {chunk_num}: {e}")
            for item in batch_data:
                if not str(df.at[item["index"], category_col] or "").strip():
                    df.at[item["index"], category_col] = "Other / Miscellaneous"
                    method_counts["failed"] += 1

        # Refresh the downloadable export after every chunk, so a download is
        # available at all times rather than only on a clean finish.
        try:
            st.session_state.export_data = export_with_column_k(df, column_mapping)
            st.session_state.export_filename = _export_filename(
                partial=(processed_count < total_valid)
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Interim export refresh failed: %s", e)

        gc.collect()

    progress_bar.progress(1.0)
    elapsed = time.time() - start_time
    st.session_state.processing_speed = processed_count / elapsed if elapsed > 0 else 0
    st.session_state.processing_elapsed = elapsed
    st.session_state.categorization_breakdown = method_counts
    st.session_state.rows_done = processed_count
    st.session_state.run_state = "complete"

    stats_container.empty()
    status_text.success(
        f"✅ Processed {processed_count:,} returns in {elapsed:.1f}s "
        f"({st.session_state.processing_speed:.1f}/sec)"
    )

    # Surface failures loudly — silent defaulting hides accuracy problems.
    if method_counts["failed"] > 0:
        st.error(
            f"⚠️ {method_counts['failed']:,} of {total_valid:,} complaints could not be "
            f"categorized and were set to 'Other / Miscellaneous'. Check API status and re-run."
        )
    if st.session_state.processing_errors:
        with st.expander(f"🐞 Processing errors ({len(st.session_state.processing_errors)})"):
            for err in st.session_state.processing_errors[-20:]:
                st.caption(err)

    return df


def generate_statistics(df, column_mapping) -> None:
    """Summarize results into session state for the dashboard."""
    category_col = column_mapping.get("category")
    sku_col = column_mapping.get("sku")

    if not category_col:
        logger.warning("No category column in mapping; cannot generate statistics")
        return

    categorized = df[df[category_col].notna() & (df[category_col] != "")]
    if categorized.empty:
        logger.warning("No categorized data found")
        return

    st.session_state.reason_summary = categorized[category_col].value_counts().to_dict()

    if sku_col and sku_col in df.columns:
        product_summary = defaultdict(lambda: defaultdict(int))
        for _, row in categorized.iterrows():
            if pd.notna(row.get(sku_col)):
                sku = str(row[sku_col]).strip()
                if sku and sku != "nan":
                    product_summary[sku][row[category_col]] += 1
        st.session_state.product_summary = {k: dict(v) for k, v in product_summary.items()}


def export_with_column_k(df, column_mapping=None) -> bytes:
    """Export to Excel: Returns sheet + Category Summary + By SKU sheets."""
    output = io.BytesIO()
    if not EXCEL_AVAILABLE:
        df.to_csv(output, index=False)
        output.seek(0)
        return output.getvalue()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        # ── Sheet 1: Returns (original structure, column K highlighted) ────────
        df.to_excel(writer, index=False, sheet_name="Returns")
        workbook = writer.book
        worksheet = writer.sheets["Returns"]
        cat_fmt = workbook.add_format({"bg_color": "#E6F5E6", "font_color": "#006600", "bold": True})
        worksheet.set_column(10, 10, 20, cat_fmt)

        cm = column_mapping or {}
        cat_col = cm.get("category") or (df.columns[10] if len(df.columns) > 10 else None)
        sku_col = cm.get("sku")

        if not (cat_col and cat_col in df.columns):
            output.seek(0)
            return output.getvalue()

        cat_series = df[cat_col].dropna()
        cat_series = cat_series[cat_series.astype(str).str.strip() != ""]
        total = len(cat_series)
        if total == 0:
            output.seek(0)
            return output.getvalue()

        # ── Sheet 2: Category Summary ─────────────────────────────────────────
        counts = cat_series.value_counts().reset_index()
        counts.columns = ["Category", "Count"]
        counts["pct"] = counts["Count"] / total

        ws = workbook.add_worksheet("Category Summary")
        hdr_fmt = workbook.add_format(
            {"bold": True, "bg_color": "#1B2A4A", "font_color": "#00F3FF", "border": 1, "align": "center"}
        )
        qual_fmt = workbook.add_format({"bg_color": "#FFE8E8", "border": 1})
        ok_fmt = workbook.add_format({"bg_color": "#E8F5E9", "border": 1})
        pct_qual = workbook.add_format({"num_format": "0.0%", "bg_color": "#FFE8E8", "border": 1})
        pct_ok = workbook.add_format({"num_format": "0.0%", "bg_color": "#E8F5E9", "border": 1})

        for i, width in enumerate([45, 28, 10, 12, 16]):
            ws.set_column(i, i, width)
        for col_i, hdr in enumerate(["Category", "Quality Group", "Count", "% of Total", "Is Quality Issue"]):
            ws.write(0, col_i, hdr, hdr_fmt)

        for row_i, row in counts.iterrows():
            cat = row["Category"]
            qual = cat in QUALITY_ISSUE_CATS
            rfmt = qual_fmt if qual else ok_fmt
            ws.write(row_i + 1, 0, cat, rfmt)
            ws.write(row_i + 1, 1, CATEGORY_GROUPS.get(cat, "Other"), rfmt)
            ws.write(row_i + 1, 2, int(row["Count"]), rfmt)
            ws.write(row_i + 1, 3, row["pct"], pct_qual if qual else pct_ok)
            ws.write(row_i + 1, 4, "Yes" if qual else "No", rfmt)

        # Group-level rollup below the per-category table
        gap = len(counts) + 3
        grp_counts = cat_series.map(lambda c: CATEGORY_GROUPS.get(c, "Other")).value_counts().reset_index()
        grp_counts.columns = ["Quality Group", "Count"]

        grp_hdr = workbook.add_format({"bold": True, "bg_color": "#2C3E50", "font_color": "#ECF0F1", "border": 1})
        grp_dat = workbook.add_format({"border": 1, "bg_color": "#F4F6F7"})
        grp_pct = workbook.add_format({"num_format": "0.0%", "border": 1, "bg_color": "#F4F6F7"})

        for col_i, hdr in enumerate(["Quality Group", "Count", "% of Total"]):
            ws.write(gap, col_i, hdr, grp_hdr)
        for i, row in grp_counts.iterrows():
            ws.write(gap + 1 + i, 0, row["Quality Group"], grp_dat)
            ws.write(gap + 1 + i, 1, int(row["Count"]), grp_dat)
            ws.write(gap + 1 + i, 2, row["Count"] / total, grp_pct)

        # ── Sheet 3: By SKU ───────────────────────────────────────────────────
        if sku_col and sku_col in df.columns:
            sku_df = df[[sku_col, cat_col]].dropna(subset=[cat_col]).copy()
            sku_df = sku_df[sku_df[cat_col].astype(str).str.strip() != ""]
            sku_df = sku_df[~sku_df[sku_col].astype(str).str.strip().isin(["", "nan"])]

            if not sku_df.empty:
                sku_df["_sku"] = sku_df[sku_col].astype(str).str.strip()
                grp_df = sku_df.groupby(["_sku", cat_col]).size().reset_index()
                grp_df.columns = ["SKU", "Category", "Count"]
                totals = grp_df.groupby("SKU")["Count"].transform("sum")
                grp_df["% within SKU"] = (grp_df["Count"] / totals * 100).round(1)
                grp_df["Quality Group"] = grp_df["Category"].map(lambda c: CATEGORY_GROUPS.get(c, "Other"))
                grp_df["Is Quality Issue"] = grp_df["Category"].apply(
                    lambda c: "Yes" if c in QUALITY_ISSUE_CATS else "No"
                )
                grp_df = grp_df.sort_values(["SKU", "Count"], ascending=[True, False]).reset_index(drop=True)

                grp_df.to_excel(writer, index=False, sheet_name="By SKU")
                ws_sku = writer.sheets["By SKU"]
                for i, width in enumerate([20, 45, 10, 14, 28, 16]):
                    ws_sku.set_column(i, i, width)

    output.seek(0)
    return output.getvalue()


def render_corrections_editor(df, category_col) -> None:
    """Let the reviewer override categories; overrides train future runs."""
    if not CORRECTIONS_AVAILABLE:
        st.info("Corrections memory module unavailable — overrides cannot be saved.")
        return
    if not (category_col and category_col in df.columns):
        st.info("No category column found in results.")
        return

    st.caption(
        "Override any AI-assigned category below. Corrections are saved permanently and "
        "reused as examples in future runs — the more you correct, the more accurate it gets."
    )

    text_col = next(
        (c for c in ["Complaint", "complaint", "Comment", "comment", "Text", "text",
                     "Description", "description", "Body"] if c in df.columns),
        None,
    )

    # Review order matters. Showing an arbitrary first-200 wastes the reviewer's
    # time on rows the pattern tier already got right with certainty. Ranking by
    # ascending confidence puts the genuinely doubtful rows first, so corrections
    # land where they actually improve future accuracy.
    conf_map = st.session_state.get("row_confidence") or {}
    order_mode = "arbitrary (no confidence data)"
    if conf_map:
        ranked = [i for i in df.index if i in conf_map]
        ranked.sort(key=lambda i: conf_map.get(i, 1.0))
        if ranked:
            df = df.loc[ranked]
            order_mode = "least-confident first"
        low = sum(1 for c in conf_map.values() if c < 0.9)
        st.caption(
            f"Sorted **{order_mode}** — {low:,} row(s) were decided by the AI rather than "
            "an exact pattern or a previous correction. Those are listed first."
        )

    edit_df = df[[text_col, category_col]].copy() if text_col else df[[category_col]].copy()
    edit_df = edit_df.rename(
        columns={category_col: "Current Category", **({text_col: "Complaint Text"} if text_col else {})}
    )
    edit_df["Override Category"] = edit_df["Current Category"]
    edit_df = edit_df.head(200)  # cap for UI performance

    col_cfg = {
        "Override Category": st.column_config.SelectboxColumn(
            "Override Category", options=sorted(MEDICAL_DEVICE_CATEGORIES), required=True
        ),
        "Current Category": st.column_config.TextColumn("AI Category", disabled=True),
    }
    if text_col:
        col_cfg["Complaint Text"] = st.column_config.TextColumn(
            "Complaint Text", disabled=True, width="large"
        )

    edited = st.data_editor(edit_df, column_config=col_cfg, hide_index=True,
                            width="stretch", key="corrections_editor")

    if st.button("💾 Save corrections to AI memory", type="primary", key="save_corrections"):
        if not text_col:
            st.warning("Cannot save corrections without the complaint text column.")
            return
        try:
            mem = get_corrections_memory()
            saved = 0
            for _, row in edited.iterrows():
                old_cat, new_cat = row["Current Category"], row["Override Category"]
                complaint_text = str(row.get("Complaint Text", ""))
                if old_cat != new_cat and complaint_text.strip():
                    mem.add_correction(complaint_text, old_cat, new_cat)
                    saved += 1
            if saved:
                st.success(f"✅ Saved {saved} correction(s) — the AI will apply these next run.")
            else:
                st.info("No changes detected.")
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not save corrections: {e}")


def display_results_dashboard(df, column_mapping) -> None:
    """Results summary for the Amazon categorizer."""
    category_col = column_mapping.get("category")
    sku_col = column_mapping.get("sku")

    if not category_col or category_col not in df.columns:
        st.warning("No category column available to summarize.")
        return

    cats = df[category_col].dropna()
    cats = cats[cats.astype(str).str.strip() != ""]
    if cats.empty:
        st.warning("Nothing was categorized.")
        return

    st.markdown("### 📊 Results")

    total = len(cats)
    quality_count = int(cats.isin(QUALITY_ISSUE_CATS).sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Categorized", f"{total:,}")
    c2.metric("Quality issues", f"{quality_count:,}", f"{quality_count / total:.1%}")
    c3.metric("Distinct categories", cats.nunique())
    c4.metric("Speed", f"{st.session_state.processing_speed:.1f}/sec")

    tab_cat, tab_grp, tab_sku, tab_data, tab_fix = st.tabs(
        ["By category", "By quality group", "By SKU", "Data", "🧠 Correct categories"]
    )

    with tab_cat:
        counts = cats.value_counts()
        st.bar_chart(counts, color=COLORS["primary"], horizontal=True)
        summary = counts.reset_index()
        summary.columns = ["Category", "Count"]
        summary["% of Total"] = (summary["Count"] / total * 100).round(1)
        summary["Quality Issue"] = summary["Category"].apply(
            lambda c: "Yes" if c in QUALITY_ISSUE_CATS else "No"
        )
        st.dataframe(summary, width="stretch", hide_index=True)

    with tab_grp:
        groups = cats.map(lambda c: CATEGORY_GROUPS.get(c, "Other")).value_counts()
        st.bar_chart(groups, color=COLORS["secondary"], horizontal=True)
        gdf = groups.reset_index()
        gdf.columns = ["Quality Group", "Count"]
        gdf["% of Total"] = (gdf["Count"] / total * 100).round(1)
        st.dataframe(gdf, width="stretch", hide_index=True)

    with tab_sku:
        if sku_col and sku_col in df.columns:
            sku_df = df[[sku_col, category_col]].copy()
            sku_df = sku_df[sku_df[category_col].astype(str).str.strip() != ""]
            sku_df["_sku"] = sku_df[sku_col].astype(str).str.strip()
            sku_df = sku_df[~sku_df["_sku"].isin(["", "nan"])]
            if sku_df.empty:
                st.info("No SKU values found.")
            else:
                agg = (
                    sku_df.groupby("_sku")
                    .agg(
                        Returns=(category_col, "size"),
                        **{"Quality Issues": (category_col, lambda s: int(s.isin(QUALITY_ISSUE_CATS).sum()))},
                        **{"Top Category": (category_col, lambda s: s.value_counts().idxmax())},
                    )
                    .reset_index()
                    .rename(columns={"_sku": "SKU"})
                )
                agg["Quality %"] = (agg["Quality Issues"] / agg["Returns"] * 100).round(1)

                # Flag products whose returns are mostly genuine quality issues
                # AND that have enough volume to be worth investigating. A SKU
                # with 1 return at 100% is noise, not a signal.
                min_vol = max(5, int(agg["Returns"].median() or 5))
                agg["⚠"] = [
                    "⚠️" if (q >= 60 and n >= min_vol) else ""
                    for q, n in zip(agg["Quality %"], agg["Returns"])
                ]
                agg = agg.sort_values(["Returns"], ascending=False)

                flagged = int((agg["⚠"] == "⚠️").sum())
                if flagged:
                    st.warning(
                        f"⚠️ **{flagged} SKU(s)** have ≥60% quality-issue returns on "
                        f"≥{min_vol} returns — worth a CAPA look.",
                        icon="⚠️",
                    )
                st.dataframe(
                    agg[["⚠", "SKU", "Returns", "Quality Issues", "Quality %", "Top Category"]],
                    width="stretch", hide_index=True, height=420,
                    column_config={"⚠": st.column_config.TextColumn("⚠", width="small")},
                )
                st.download_button(
                    "⬇️ Download SKU breakdown (.csv)",
                    data=agg.to_csv(index=False).encode("utf-8"),
                    file_name=f"sku_breakdown_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                )
        else:
            st.info("No SKU column detected.")

    with tab_data:
        # Free-text search across the whole file — faster than exporting to
        # Excel just to find the complaints mentioning one term.
        q = st.text_input("🔎 Search complaints", key="amazon_search",
                          placeholder="e.g. strap, battery, MOB1027")
        view = df
        if q:
            hay = df.astype(str).apply(lambda c: c.str.contains(q, case=False, na=False))
            view = df[hay.any(axis=1)]
            st.caption(f"**{len(view):,}** row(s) match “{q}”.")
        st.dataframe(view.head(500), width="stretch", height=460)
        st.caption(f"Previewing first {min(500, len(view)):,} of {len(view):,} rows.")

    with tab_fix:
        render_corrections_editor(df, category_col)


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 2 — B2B Report (Odoo Helpdesk export -> B2B Report)
# ══════════════════════════════════════════════════════════════════════════════

def extract_main_sku(text) -> Optional[str]:
    """Extract the main SKU (3 uppercase letters + 4 digits).

    Matches the base SKU and ignores variant suffixes, e.g. MOB1027BLU -> MOB1027.
    """
    if not isinstance(text, str):
        return None
    match = re.search(r"\b([A-Z]{3}\d{4})", text)
    return match.group(1) if match else None


def find_sku_in_row(row) -> str:
    """Locate the main SKU, preferring explicit SKU columns over free text."""
    for col in ["Main SKU", "Main SKU/Display Name", "SKU", "Product", "Internal Reference"]:
        if col in row.index and pd.notna(row[col]):
            sku = extract_main_sku(str(row[col]))
            if sku:
                return sku

    for col in ["Display Name", "Subject", "Name"]:
        if col in row.index and pd.notna(row[col]):
            sku = extract_main_sku(str(row[col]))
            if sku:
                return sku

    for col in ["Description", "Body"]:  # last resort
        if col in row.index and pd.notna(row[col]):
            sku = extract_main_sku(str(row[col]))
            if sku:
                return sku

    return "Unknown"


def strip_html(text) -> str:
    """Strip HTML tags so the AI sees clean text. Output column keeps the original."""
    if not text or not isinstance(text, str):
        return ""
    return re.sub(re.compile("<.*?>"), " ", text).strip()


def process_b2b_file(file_content, filename):
    """Read a raw Odoo Helpdesk export."""
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(file_content), dtype=str)
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(file_content), dtype=str)
        else:
            st.error("Unsupported file format.")
            return None
        logger.info("B2B processing: %d rows loaded", len(df))
        return df
    except Exception as e:  # noqa: BLE001
        st.error(f"Error reading file: {e}")
        return None


def generate_b2b_report(df, analyzer, batch_size):
    """Build the B2B report: extracted SKU + AI category and reason per ticket."""
    display_col = "Display Name" if "Display Name" in df.columns else df.columns[0]
    desc_col = "Description" if "Description" in df.columns else None

    items_to_process = []
    for idx, row in df.iterrows():
        description = str(row.get(desc_col, "")) if desc_col else ""
        items_to_process.append({
            "index": idx,
            "subject": str(row.get(display_col, "")),
            # AI sees cleaned + capped text; the export keeps the original HTML.
            "details": strip_html(description)[:1000],
            "full_description": description,
            "sku": find_sku_in_row(row),
        })

    progress_bar = st.progress(0)
    status_text = st.empty()

    total_items = len(items_to_process)
    processed_results = []

    for i in range(0, total_items, batch_size):
        batch = items_to_process[i:i + batch_size]
        processed_results.extend(analyzer.summarize_batch(batch))
        progress_bar.progress(min((i + batch_size) / total_items, 1.0))
        status_text.text(f"⏳ Generating summaries: {min(i + batch_size, total_items)}/{total_items}")

    status_text.success("✅ AI summarization complete")
    progress_bar.empty()

    final_rows = [{
        "Display Name": item["subject"],
        "Description": item["full_description"],
        "SKU": item["sku"],
        "Category": item.get("category", ""),
        "Reason": item.get("summary", "Summary Unavailable"),
    } for item in processed_results]

    return pd.DataFrame(final_rows)


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(f"### 🏥 {APP_CONFIG['title']}")
        st.caption(f"v{APP_CONFIG['version']} · Anthropic Claude")

        st.divider()
        st.markdown("#### 🔌 API status")
        if not AI_AVAILABLE:
            st.error("AI engine failed to import.")
            with st.expander("Import error"):
                st.code(AI_IMPORT_ERROR or "unknown")
        elif check_api_keys().get("claude"):
            st.success("Anthropic API key found")
        else:
            st.error("No `ANTHROPIC_API_KEY`")
            st.caption("Add it in **Settings → Secrets** on Streamlit Cloud.")
            # Diagnostic only — lists secret KEY NAMES (never values) so a
            # naming/casing/nesting mismatch is visible without exposing the
            # key itself. check_api_keys() matches exact casing; a secret
            # saved under any other name or nested in a [section] won't be
            # found, but silently, hence needing this to see why.
            with st.expander("🔍 Diagnose: what Streamlit Cloud actually sees"):
                try:
                    names = sorted(st.secrets.keys()) if hasattr(st, "secrets") else []
                except Exception as _e:  # noqa: BLE001
                    names = None
                    st.caption(f"Could not read st.secrets: {_e}")
                if names is None:
                    pass
                elif not names:
                    st.warning(
                        "`st.secrets` is empty. Either nothing was saved, the app "
                        "hasn't rebooted since you saved it, or the secrets TOML "
                        "has a syntax error (check for a red banner elsewhere on "
                        "this page)."
                    )
                else:
                    st.write("Top-level secret names found:", names)
                    expected = {"ANTHROPIC_API_KEY", "anthropic_api_key", "claude_api_key", "claude"}
                    if not (set(names) & expected):
                        st.warning(
                            "None of these match an expected name exactly "
                            f"(`{'`, `'.join(sorted(expected))}`). TOML keys are "
                            "case-sensitive — rename the secret to exactly "
                            "`ANTHROPIC_API_KEY`, or it must be at the top level, "
                            "not nested under a `[section]`."
                        )
                    else:
                        st.warning(
                            "A matching key name was found, but its value doesn't "
                            "start with `sk-ant-` — re-check for extra quotes, "
                            "whitespace, or a truncated paste."
                        )

        if AI_AVAILABLE:
            st.divider()
            st.markdown("#### 🤖 AI model")
            provider_map = {
                "Sonnet — balanced (recommended)": AIProvider.CLAUDE,
                "Haiku — fastest, lower accuracy": AIProvider.CLAUDE_FAST,
                "Opus — maximum accuracy": AIProvider.CLAUDE_POWERFUL,
            }
            choice = st.radio(
                "Model", list(provider_map), index=0, key="model_choice", label_visibility="collapsed"
            )
            st.session_state.ai_provider = provider_map[choice]

            st.divider()
            st.markdown("#### ⏱ Time-savings assumptions")
            st.caption("Manual handling time per item — adjust to match your team.")
            st.session_state.sec_per_return = float(
                st.number_input(
                    "Seconds per Amazon return", min_value=1.0, max_value=120.0,
                    value=float(st.session_state.sec_per_return), step=1.0,
                    help="Time for an analyst to read one return comment and record a category.",
                )
            )
            st.session_state.sec_per_ticket = float(
                st.number_input(
                    "Seconds per support ticket", min_value=1.0, max_value=300.0,
                    value=float(st.session_state.sec_per_ticket), step=5.0,
                    help="Longer than a return: full ticket bodies, quoted threads, SKU lookup.",
                )
            )

            st.divider()
            st.markdown("#### ⚙️ Throughput")
            st.session_state.chunk_size = st.select_slider(
                "Chunk size", options=APP_CONFIG["chunk_sizes"], value=st.session_state.chunk_size,
                help="Rows held in memory per chunk.",
            )
            st.session_state.batch_size = st.slider(
                "API batch size", 5, 50, st.session_state.batch_size, step=5,
                help="Complaints per parallel API wave. Lower this if you hit rate limits.",
            )

            analyzer = st.session_state.get("ai_analyzer")
            if analyzer is not None:
                try:
                    cost = analyzer.get_cost_summary()
                    if cost.get("api_calls"):
                        st.divider()
                        st.markdown("#### 💰 This session")
                        st.metric("Est. API cost", f"${cost.get('total_cost', 0):.3f}")
                        st.caption(
                            f"{cost.get('ai_categorizations', 0):,} AI calls · "
                            f"{cost.get('quick_categorizations', 0):,} free pattern matches"
                        )
                except Exception:  # noqa: BLE001
                    pass


def render_amazon_tool() -> None:
    st.markdown("### 📦 Amazon Return Categorizer")
    st.markdown(
        """
        <div class="tool-note">
        <strong>Input:</strong> Amazon FBA customer-returns export (the file with complaint text in column I).<br/>
        <strong>Output:</strong> the same file with the standard quality category filled into column K,
        plus <em>Category Summary</em> and <em>By SKU</em> sheets.<br/>
        <strong>Note:</strong> the customer-selected return-reason code is used only as a weak hint —
        the free-text complaint always decides the category.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # A run still marked 'running' on a fresh script run was killed mid-flight
    # (Stop pressed, tab closed, session timed out). Partial work is intact.
    if st.session_state.run_state == "running":
        st.session_state.run_state = "interrupted"

    interrupted = st.session_state.run_state == "interrupted"
    have_data = st.session_state.categorized_data is not None

    if interrupted and have_data:
        done = st.session_state.get("rows_done", 0)
        target = st.session_state.get("rows_target", 0)
        st.warning(
            f"⏸ **Run stopped early — your work was kept.** "
            f"{done:,} of {target:,} rows were categorized before it stopped. "
            "Download the partial file below, or resume to finish the rest.",
            icon="⏸",
        )

    uploaded = st.file_uploader(
        "Upload return data", type=["csv", "xlsx", "xls", "txt"], key="amazon_uploader"
    )

    if uploaded and not have_data:
        with st.spinner(f"Reading {uploaded.name}..."):
            df, column_mapping = process_file_preserve_structure(uploaded.read(), uploaded.name)

        if df is not None and column_mapping:
            st.session_state.column_mapping = column_mapping
            complaint_col = column_mapping.get("complaint")

            if complaint_col:
                n = int((df[complaint_col].notna() & (df[complaint_col].str.strip() != "")).sum())
                st.success(f"Found **{n:,}** complaints to categorize in **{complaint_col}**.")
                render_pre_estimate(
                    estimate_amazon(n, seconds_per_item=st.session_state.sec_per_return),
                    "returns",
                )
            else:
                st.warning(
                    "⚠️ No free-text complaint column detected — only structural/code columns. "
                    "Make sure the export includes the customer comment text (e.g. 'customer-comments')."
                )

            c_run, c_sample = st.columns([2, 1])
            start = c_run.button("🚀 Categorize everything", type="primary",
                                 disabled=not complaint_col, width="stretch")
            # A cheap dry run: validates column detection and output quality on a
            # new file format before committing to thousands of API calls.
            sample = c_sample.button("🧪 Test on first 100", disabled=not complaint_col,
                                     width="stretch",
                                     help="Quick, low-cost check that this file parses correctly.")

            if start or sample:
                analyzer = get_ai_analyzer()
                if analyzer is None:
                    st.error(
                        "🚫 AI unavailable — no working `ANTHROPIC_API_KEY`. Categorization was **not** run: "
                        "without AI every complaint would be silently labeled 'Other / Miscellaneous'. "
                        "Check **API status** in the sidebar."
                    )
                else:
                    st.session_state.processing_errors = []
                    st.session_state.row_confidence = {}
                    categorized = process_in_chunks(
                        df, analyzer, column_mapping, limit=100 if sample else None
                    )
                    st.session_state.categorized_data = categorized
                    st.session_state.processing_complete = True
                    generate_statistics(categorized, column_mapping)
                    st.session_state.export_data = export_with_column_k(categorized, column_mapping)
                    st.session_state.export_filename = _export_filename(
                        partial=count_uncategorized(categorized, column_mapping) > 0
                    )
                    st.rerun()

    if have_data:
        df_done = st.session_state.categorized_data
        cm = st.session_state.column_mapping
        remaining = count_uncategorized(df_done, cm)

        # ── Headline: time saved ───────────────────────────────────────────
        done_rows = st.session_state.get("rows_done", 0)
        elapsed = st.session_state.get("processing_elapsed", 0.0)
        if done_rows:
            render_savings_hero(
                estimate_amazon(done_rows, actual_seconds=elapsed,
                                seconds_per_item=st.session_state.sec_per_return),
                noun="returns",
                partial=remaining > 0,
            )

        bd = st.session_state.get("categorization_breakdown")
        if bd:
            total_done = sum(bd.values()) or 1
            free = bd["instant"] + bd["corrections"]
            st.markdown("##### 🧭 How complaints were categorized")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("⚡ Free (no API call)", f"{free:,}", f"{free / total_done:.0%} of rows")
            c2.metric("🤖 AI (Claude)", f"{bd['ai']:,}", f"{bd['ai'] / total_done:.0%}")
            c3.metric("🧠 Learned corrections", f"{bd['corrections']:,}")
            c4.metric("⚠️ Failed", f"{bd['failed']:,}",
                      f"{bd['failed'] / total_done:.0%}", delta_color="inverse")

        # ── Export + resume ───────────────────────────────────────────────
        cols = st.columns([2, 1, 1])
        if st.session_state.export_data:
            cols[0].download_button(
                "⬇️ Download " + ("partial file" if remaining else "categorized file"),
                data=st.session_state.export_data,
                file_name=st.session_state.export_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                width="stretch",
            )
        if remaining:
            if cols[1].button(f"▶️ Resume ({remaining:,} left)", width="stretch"):
                analyzer = get_ai_analyzer()
                if analyzer is None:
                    st.error("🚫 AI unavailable — cannot resume. Check **API status** in the sidebar.")
                else:
                    process_in_chunks(df_done, analyzer, cm, only_uncategorized=True)
                    generate_statistics(df_done, cm)
                    st.session_state.export_data = export_with_column_k(df_done, cm)
                    st.session_state.export_filename = _export_filename(
                        partial=count_uncategorized(df_done, cm) > 0
                    )
                    st.rerun()
        if cols[2].button("🔄 Clear / start over", key="amazon_clear", width="stretch"):
            for k in ["categorized_data", "export_data", "export_filename",
                      "categorization_breakdown", "reason_summary", "product_summary"]:
                st.session_state[k] = None
            st.session_state.processing_complete = False
            st.session_state.run_state = "idle"
            st.session_state.rows_done = 0
            st.session_state.rows_target = 0
            st.session_state.row_confidence = {}
            st.rerun()

        display_results_dashboard(df_done, cm)


def render_b2b_tool() -> None:
    st.markdown("### 📑 B2B Report")
    st.markdown(
        """
        <div class="tool-note">
        <strong>Input:</strong> raw Odoo Helpdesk ticket export (Display Name + Description).<br/>
        <strong>Output:</strong> the standard B2B Report — Display Name, Description, SKU, Category, Reason —
        with the main SKU auto-extracted (e.g. <code>MOB1027</code>) and an AI-written reason for each ticket.
        </div>
        """,
        unsafe_allow_html=True,
    )

    perf_mode = st.select_slider(
        "Dataset size (tunes batching and concurrency)",
        options=["Small (< 500 rows)", "Medium (500-2,000 rows)", "Large (2,000+ rows)"],
        value=st.session_state.b2b_perf_mode,
        key="b2b_perf_selector",
    )
    st.session_state.b2b_perf_mode = perf_mode

    if perf_mode == "Small (< 500 rows)":
        batch_size, max_workers = 10, 2
        st.caption("Conservative batching for maximum reliability.")
    elif perf_mode == "Medium (500-2,000 rows)":
        batch_size, max_workers = 25, 3
        st.caption("Balanced speed and concurrency.")
    else:
        batch_size, max_workers = 50, 5
        st.caption("Aggressive parallel processing for high volume.")

    st.divider()
    b2b_file = st.file_uploader("Upload Odoo export", type=["csv", "xlsx", "xls"], key="b2b_uploader")

    if b2b_file:
        b2b_df = process_b2b_file(b2b_file.read(), b2b_file.name)
        if b2b_df is not None:
            st.success(f"**{len(b2b_df):,}** tickets found.")
            render_pre_estimate(
                estimate_tickets(len(b2b_df), seconds_per_item=st.session_state.sec_per_ticket),
                "tickets",
            )

            if st.button("⚡ Generate B2B report", type="primary"):
                analyzer = get_ai_analyzer(max_workers=max_workers)
                if analyzer is None:
                    st.error(
                        "🚫 AI unavailable — no working `ANTHROPIC_API_KEY`. The report was **not** generated: "
                        "every ticket would come back 'Summary Unavailable'. Check **API status** in the sidebar."
                    )
                else:
                    with st.spinner("Running AI analysis and SKU extraction..."):
                        _b2b_start = time.time()
                        final_b2b = generate_b2b_report(b2b_df, analyzer, batch_size)
                        st.session_state.b2b_elapsed = time.time() - _b2b_start
                        st.session_state.b2b_processed_data = final_b2b
                        st.session_state.b2b_processing_complete = True

                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                            final_b2b.to_excel(writer, index=False, sheet_name="B2B Report")
                            workbook = writer.book
                            worksheet = writer.sheets["B2B Report"]
                            hdr = workbook.add_format(
                                {"bold": True, "bg_color": "#00D9FF", "font_color": "white"}
                            )
                            for col_num, value in enumerate(final_b2b.columns.values):
                                worksheet.write(0, col_num, value, hdr)
                                worksheet.set_column(col_num, col_num, 30)

                        st.session_state.b2b_export_data = output.getvalue()
                        st.session_state.b2b_export_filename = (
                            f"B2B_Report_{datetime.now().strftime('%Y-%m-%d')}.xlsx"
                        )
                    st.rerun()

    if st.session_state.b2b_processing_complete and st.session_state.b2b_processed_data is not None:
        df_res = st.session_state.b2b_processed_data
        st.markdown("### 🏁 Report")

        render_savings_hero(
            estimate_tickets(len(df_res),
                             actual_seconds=st.session_state.get("b2b_elapsed", 0.0),
                             seconds_per_item=st.session_state.sec_per_ticket),
            noun="tickets",
        )

        known = df_res[df_res["SKU"] != "Unknown"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Tickets processed", f"{len(df_res):,}")
        c2.metric("SKUs identified", f"{len(known):,}",
                  f"{len(known) / len(df_res) * 100:.1f}% coverage" if len(df_res) else None)
        c3.metric("Unique products", known["SKU"].nunique())

        # Which products generate the most B2B trouble — the analyst's first
        # question after "what came in this month?".
        if not known.empty:
            with st.expander("🚨 Top products by ticket volume", expanded=True):
                top = (
                    known.groupby("SKU")
                    .agg(Tickets=("SKU", "size"),
                         **{"Top issue": ("Category", lambda s: s.value_counts().idxmax())},
                         **{"Quality issues": ("Category",
                                               lambda s: int(s.isin(QUALITY_ISSUE_CATS).sum()))})
                    .reset_index()
                    .sort_values("Tickets", ascending=False)
                    .head(15)
                )
                st.dataframe(top, width="stretch", hide_index=True)

        with st.expander("🔧 Filters"):
            f1, f2 = st.columns(2)
            sku_filter = f1.text_input("SKU contains", key="b2b_sku_filter", placeholder="e.g. MOB1027")
            cat_opts = ["All"] + sorted(df_res["Category"].dropna().unique().tolist())
            cat_filter = f2.selectbox("Category", cat_opts, key="b2b_cat_filter")

        df_display = df_res.copy()
        if sku_filter:
            df_display = df_display[df_display["SKU"].str.contains(sku_filter, case=False, na=False)]
        if cat_filter != "All":
            df_display = df_display[df_display["Category"] == cat_filter]

        st.dataframe(
            df_display,
            width="stretch",
            height=min(520, 60 + 35 * max(len(df_display), 1)),
            column_config={
                "SKU": st.column_config.TextColumn("SKU", width="small"),
                "Category": st.column_config.TextColumn("Category", width="medium"),
                "Reason": st.column_config.TextColumn("AI Reason", width="large"),
                "Description": st.column_config.TextColumn("Description (raw)", width="small"),
            },
        )
        st.caption(f"Showing **{len(df_display):,}** of **{len(df_res):,}** rows.")

        cat_counts = df_display["Category"].value_counts()
        if not cat_counts.empty:
            st.markdown("**Category distribution**")
            st.bar_chart(cat_counts, color=COLORS["primary"], horizontal=True)

        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "⬇️ Download B2B report (.xlsx)",
                data=st.session_state.b2b_export_data,
                file_name=st.session_state.b2b_export_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                width="stretch",
            )
        with c2:
            if st.button("🔄 Clear / start over", width="stretch", key="b2b_clear"):
                st.session_state.b2b_processed_data = None
                st.session_state.b2b_processing_complete = False
                st.rerun()


def render_zendesk_tool() -> None:
    st.markdown("### 🎫 Zendesk B2C Quality Report")
    if not ZENDESK_AVAILABLE:
        st.error("The Zendesk reporting module failed to import.")
        with st.expander("Import error"):
            st.code(ZENDESK_IMPORT_ERROR or "unknown")
        return
    st.markdown(
        """
        <div class="tool-note">
        <strong>Input:</strong> Zendesk quality-issues export. Required columns:
        <code>Ticket created - Date</code>, <code>Ticket ID</code>, <code>SKU</code>,
        <code>Issue</code>, <code>Ticket Type</code>.<br/>
        <strong>Output:</strong> quality report aggregated by parent SKU (first 7 characters),
        with per-category breakdowns.
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_b2b_zendesk_reporting()

    # The Zendesk tool is a self-contained module, so read back whatever it
    # left in session state to report savings without interfering with it.
    ztickets = st.session_state.get("zendesk_categorized")
    try:
        n_tickets = len(ztickets) if ztickets is not None else 0
    except TypeError:
        n_tickets = 0
    if n_tickets:
        st.divider()
        render_savings_hero(
            estimate_tickets(n_tickets, seconds_per_item=st.session_state.sec_per_ticket),
            noun="tickets",
        )


def _require_passcode() -> None:
    """Optional access gate. Set APP_PASSCODE in secrets to enable; when unset
    the gate is disabled. Uses a constant-time comparison."""
    try:
        expected = str(st.secrets.get("APP_PASSCODE", "")).strip()
    except Exception:  # noqa: BLE001
        expected = ""
    if not expected:
        expected = os.environ.get("APP_PASSCODE", "").strip()
    if not expected or st.session_state.get("_auth_ok"):
        return

    import hmac

    st.markdown("## 🔒 Vive Health Returns Categorizer")
    st.caption("This app is access-protected. Enter the team passcode to continue.")
    code = st.text_input("Passcode", type="password", key="_auth_input")
    if st.button("Enter", type="primary"):
        if hmac.compare_digest(code.strip(), expected):
            st.session_state["_auth_ok"] = True
            st.rerun()
        else:
            st.error("Incorrect passcode.")
    st.stop()


def main() -> None:
    _require_passcode()
    initialize_session_state()
    inject_css()

    st.markdown(
        f"""
        <div class="main-header">
            <h1>🏥 Vive Health — Returns Categorizer</h1>
            <p>Amazon returns · B2B reports · Zendesk B2C reports — powered by Anthropic Claude</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not AI_AVAILABLE:
        st.error(
            "⚠️ **The AI engine failed to load, so categorization is disabled.** "
            "See **API status** in the sidebar for the import error.",
            icon="⚠️",
        )
    elif not st.session_state.get("_api_key_checked"):
        st.session_state["_api_key_checked"] = True
        if not check_api_keys().get("claude"):
            st.error(
                "🔑 **Anthropic API key not found — AI features will not work.** Add `ANTHROPIC_API_KEY` "
                "to your Streamlit secrets (**Settings → Secrets** in the Streamlit Cloud dashboard).",
                icon="🔑",
            )

    render_sidebar()

    tab1, tab2, tab3 = st.tabs(["📦 Amazon Returns", "📑 B2B Report", "🎫 Zendesk B2C"])
    with tab1:
        render_amazon_tool()
    with tab2:
        render_b2b_tool()
    with tab3:
        render_zendesk_tool()


if __name__ == "__main__":
    main()
