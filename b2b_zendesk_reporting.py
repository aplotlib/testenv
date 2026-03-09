"""
B2C Zendesk Reporting Module
─────────────────────────────
Categorizes ALL Zendesk tickets by Issue text using the same
MEDICAL_DEVICE_CATEGORIES as the Return Categorizer, then reports
only quality-relevant tickets aggregated by Parent SKU (first 7 chars).

Categorization modes:
  • Keyword matching (free, instant)
  • AI for unclear tickets + random audit sample
  • Full AI mode (optional)
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from io import BytesIO
from typing import Optional, Dict, List, Tuple
import logging
import random

logger = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────────────

PARENT_SKU_LENGTH = 7

# Junk SKU values to exclude
EXCLUDED_SKUS = {"x", ".", "X", "XX", "XXX", "No SKU", "", " "}

# Issue text values that are clearly not real issues (noise)
NOISE_ISSUES = {
    "x", "X", "XX", "XXX", ".", " ", "",
    "Amazon calls.", "AMAZON CALL SIDE TASK", "Amazon calls",
    "Amazon call side tasks", "Amazon call side task",
    "Amazon call", "Amazon Call", "Amazon call.",
    "order status", "Order status", "Order status.",
    "delivery status", "delivery", "delivery SLA", "delivery sla",
    "Delivery update.", "Delivery address update.",
    "Cancelation", "discount code", "call dropped",
    "Missed chat.", "Miscellaneous.", "Miscellaneous",
    "Missed chat due back to back calls. No email to provide an answer.",
    "Call returned in 623535", "Callback not answered",
}

# ─── MEDICAL DEVICE CATEGORIES (same as Return Categorizer) ──────────────────

QUALITY_CATEGORIES = [
    'Size: Too Small',
    'Size: Too Large',
    "Size: Doesn't Fit / Wrong Dimensions",
    'Comfort: Causes Pain or Pressure',
    'Comfort: Too Hard / Rigid',
    'Comfort: Too Soft / Lacks Support',
    'Comfort: Skin Irritation or Allergic Reaction',
    'Defect: Broken / Structural Failure',
    'Defect: Malfunctions / Stops Working',
    'Defect: Cosmetic Damage',
    'Defect: Poor Material Quality',
    'Wrong Product / Not as Described',
    'Missing or Incomplete Components',
    "Performance: Ineffective / Doesn't Help",
    'Equipment Compatibility Issue',
    'Stability: Shifts / Unstable / Falls',
    'Assembly / Usage Difficulty',
    'Medical / Safety Concern',
]

NON_QUALITY_CATEGORIES = [
    'Customer: Changed Mind / No Longer Needed',
    'Customer: Ordered Wrong Size or Item',
    'Fulfillment: Damaged in Shipping',
    'Fulfillment: Wrong Item Sent',
    'Fulfillment: Delivery Issue',
    'General Inquiry / Not a Quality Issue',
    'Other / Miscellaneous',
]

ALL_CATEGORIES = QUALITY_CATEGORIES + NON_QUALITY_CATEGORIES

# ─── KEYWORD RULES (order matters — first match wins) ────────────────────────

KEYWORD_RULES: List[Tuple[str, List[str]]] = [
    # ── Medical / Safety (check first — highest priority) ──
    ('Medical / Safety Concern', [
        'injury', 'injured', 'hospital', 'emergency', 'dangerous', 'unsafe',
        'hazard', 'fell off', 'fall', 'fell', 'tipping', 'tipped over',
        'cut my leg', 'cut myself', 'hurt', 'accident',
    ]),
    # ── Defect: Broken / Structural ──
    ('Defect: Broken / Structural Failure', [
        'broke', 'broken', 'snapped', 'cracked', 'crack', 'shattered',
        'fell apart', 'ripped', 'rip', 'torn', 'tore', 'split',
        'bent', 'buckle', 'weld', 'disconnected from the frame',
        'collapsed', 'structural', 'frame broke', 'clamp broke',
        'handle broke', 'leg broke', 'arm broke', 'pin broke',
        'connector broke', 'broken clamp', 'broken connector',
        'broken handle', 'broken part', 'broken piece', 'broken battery',
        'broke off', 'snapped off',
    ]),
    # ── Defect: Malfunctions / Stops Working ──
    ('Defect: Malfunctions / Stops Working', [
        'malfunction', 'stops working', 'stopped working', 'not working',
        "won't turn on", 'wont turn on', "won't work", 'wont work',
        "doesn't work", 'doesnt work', 'not turning on', "won't operate",
        'quit working', 'dead', 'beeping', 'beeps', '2 beeps',
        'not charging', 'not taking a charge', "won't charge",
        'motor', 'control box defective', 'defective',
        'not running', "won't run", 'stops when riding',
        'stops dead', 'power issue', 'flashing', 'alarm sound',
        'not getting cold', "isn't working", 'pump not working',
        'pump quit', 'pump dead', 'pump making loud',
        'not holding air', 'losing power', 'not turning',
        'joystick', 'charger', 'not operate', 'scooter issue',
    ]),
    # ── Stability ──
    ('Stability: Shifts / Unstable / Falls', [
        'wobbly', 'wobble', 'wobbling', 'unstable', 'shifts',
        'slides', 'tipping forward', 'not stable', 'not secure',
        'tilted', 'not locking', "won't lock", 'wont lock',
        "doesn't lock", 'not stay', "won't stay",
        'seat not locking', 'lock in place',
    ]),
    # ── Defect: Poor Material Quality ──
    ('Defect: Poor Material Quality', [
        'cheap', 'poor quality', 'low quality', 'thin',
        'flimsy', 'wear out', 'worn out', 'wears out',
        'peeling', 'discolor', 'rusting', 'rust',
        'velcro stopped', 'fabric', 'material',
        'odor', 'smell', 'stain',
    ]),
    # ── Missing or Incomplete Components ──
    ('Missing or Incomplete Components', [
        'missing part', 'missing piece', 'missing hardware',
        'missing bolt', 'missing screw', 'missing nut',
        'missing washer', 'missing component', 'missing gel',
        'missing horn', 'missing tip', 'missing the',
        'missing a', 'missing item', 'parts missing',
        'incomplete', 'no instructions', 'without instructions',
        'empty package', 'package was empty', 'not included',
        'didn\'t include', 'only received 1', 'only received one',
        'missing push pin', 'missing handscrew', 'missing leg',
        'missing axle', 'missing harware', 'missing hardware kit',
    ]),
    # ── Comfort: Pain / Pressure ──
    ('Comfort: Causes Pain or Pressure', [
        'pain', 'painful', 'pressure', 'sore', 'bruise', 'bruising',
        'digs in', 'rubs', 'rubbing', 'uncomfortable', 'discomfort',
        'causes pain', 'hurts', 'numbed',
    ]),
    # ── Comfort: Too Hard / Rigid ──
    ('Comfort: Too Hard / Rigid', [
        'too hard', 'too stiff', 'too rigid', 'really stiff',
        'extremely hard', 'too firm',
    ]),
    # ── Comfort: Too Soft / Lacks Support ──
    ('Comfort: Too Soft / Lacks Support', [
        'too soft', 'no support', 'lacks support', 'deflated',
        'collapsed under', 'not enough support', 'stayed deflated',
        'sinking', 'not expand',
    ]),
    # ── Comfort: Skin Irritation ──
    ('Comfort: Skin Irritation or Allergic Reaction', [
        'rash', 'irritation', 'allergic', 'skin reaction', 'itching', 'itch',
    ]),
    # ── Size: Too Small ──
    ('Size: Too Small', [
        'too small', 'too short', 'too narrow', 'too tight',
    ]),
    # ── Size: Too Large ──
    ('Size: Too Large', [
        'too large', 'too big', 'too wide', 'too long', 'too bulky',
        'too heavy', 'oversized',
    ]),
    # ── Size: Doesn't Fit ──
    ("Size: Doesn't Fit / Wrong Dimensions", [
        "doesn't fit", "doesnt fit", "does not fit", "not fit",
        'not fitting', "won't fit", "didn't fit", 'wrong size',
        'wrong dimension', 'bad fit', 'good fit',
    ]),
    # ── Performance ──
    ("Performance: Ineffective / Doesn't Help", [
        'ineffective', "doesn't help", "doesn't do anything",
        "doesn't work for", 'not effective', 'useless',
        "didn't help", 'not able to use it',
    ]),
    # ── Equipment Compatibility ──
    ('Equipment Compatibility Issue', [
        'not compatible', 'incompatible', "doesn't attach",
        'does not fit my walker', 'does not fit my wheelchair',
        "doesn't connect", 'not pairing',
    ]),
    # ── Assembly / Usage Difficulty ──
    ('Assembly / Usage Difficulty', [
        'hard to assemble', 'difficult to assemble', 'assembly issue',
        'parts not fitting', 'unable to assemble', 'confusing instruction',
        'how to', 'help assembl', 'help putting',
        'not able to open', 'problem assembling',
    ]),
    # ── Defect: Cosmetic ──
    ('Defect: Cosmetic Damage', [
        'scratch', 'scratched', 'dent', 'dented', 'cosmetic',
        'paint', 'hubcap', 'dust on',
    ]),
    # ── Wrong Product / Not as Described ──
    ('Wrong Product / Not as Described', [
        'wrong product', 'wrong item', 'not as described',
        'incorrect product', 'incorrect item', 'received the incorrect',
        'wrong package', "didn't order", 'not what I ordered',
        'different product', 'different color',
    ]),
    # ── Fulfillment: Damaged in Shipping ──
    ('Fulfillment: Damaged in Shipping', [
        'damaged in shipping', 'damaged during shipping',
        'box arrived damaged', 'box was damaged', 'arrived damaged',
        'shipping damage', 'damaged box', 'scuffed',
        'package was damaged', 'package arrived damaged',
    ]),
    # ── Fulfillment: Wrong Item Sent ──
    ('Fulfillment: Wrong Item Sent', [
        'sent wrong', 'shipped wrong', 'wrong item sent',
        'received wrong', 'wanted the beige and received',
    ]),
    # ── Fulfillment: Delivery Issue ──
    ('Fulfillment: Delivery Issue', [
        'not delivered', 'never arrived', 'never received',
        'not received', 'lost package', 'lost order',
        'returned to sender', 'order not received',
        'has not been shipped', 'hasn\'t been shipped',
        'item not delivered',
    ]),
    # ── Customer: Changed Mind ──
    ('Customer: Changed Mind / No Longer Needed', [
        'changed mind', 'no longer need', 'no longer needed',
        'not needed', "doesn't need", 'does not need',
        'decided not to', 'not a good fit for',
        'doctor recommended not', 'doctor advised',
        'hospice', 'will not use', "won't use",
        'return product for a better price',
        'return the item', 'wants to return',
        'looking to return', 'return request',
        'return/refund request', 'return/refund operation',
        'return authorization', 'return process',
        'return and refund', 'return request not needed',
        'not meet', "didn't meet", 'expectations',
    ]),
    # ── Customer: Ordered Wrong ──
    ('Customer: Ordered Wrong Size or Item', [
        'ordered wrong', 'purchased incorrect', 'ordered the wrong',
        'he purchased the incorrect', 'bought wrong',
    ]),
]

# Brake-specific rules — brakes are structural/functional, not "changed mind"
BRAKE_KEYWORDS = [
    'brake', 'brakes', 'braking',
]

# Part request keywords that indicate an underlying quality issue
PART_REQUEST_QUALITY_KEYWORDS = [
    'replacement part', 'needs part', 'need part', 'needs the',
    'needs a new', 'replacement wheel', 'replacement tire',
    'replacement battery', 'needs replacement',
    'part request', 'parts request',
]


# ─── Keyword Categorizer ────────────────────────────────────────────────────

def categorize_by_keywords(issue_text: str) -> Tuple[str, float]:
    """
    Categorize an issue string using keyword rules.
    Returns (category, confidence).
    confidence: 1.0 = strong match, 0.5 = weak/partial match, 0.0 = no match.
    """
    if not issue_text or not isinstance(issue_text, str):
        return 'General Inquiry / Not a Quality Issue', 0.0

    text = issue_text.lower().strip()

    # Skip obvious noise
    if text in {s.lower() for s in NOISE_ISSUES} or len(text) < 3:
        return 'General Inquiry / Not a Quality Issue', 1.0

    # Brake issues → Defect: Malfunctions (brakes are functional components)
    if any(kw in text for kw in BRAKE_KEYWORDS):
        # Check if it's about brake not working vs just a return
        if any(kw in text for kw in ['not working', "doesn't", "won't", "wont",
                                      'not lock', 'not engage', 'stiff', 'hard to',
                                      'broke', 'broken', 'bent', 'issue', 'problem',
                                      'defective', 'not move', 'floppy']):
            return 'Defect: Malfunctions / Stops Working', 0.9
        return 'Defect: Malfunctions / Stops Working', 0.7

    # Part requests that imply a quality issue (something broke/wore out)
    if any(kw in text for kw in PART_REQUEST_QUALITY_KEYWORDS):
        # If text also mentions broke/worn/defective → structural
        if any(kw in text for kw in ['broke', 'broken', 'worn', 'fell off',
                                      'fell apart', 'cracked', 'snapped']):
            return 'Defect: Broken / Structural Failure', 0.8
        return 'Defect: Malfunctions / Stops Working', 0.6

    # Standard keyword rules (first match wins)
    for category, keywords in KEYWORD_RULES:
        for kw in keywords:
            if kw in text:
                return category, 0.85

    # ── Broader "missing" catch (after specific missing keywords above) ──
    if 'missing' in text:
        return 'Missing or Incomplete Components', 0.7

    # "only came with" / "only received" / "supposed to come with" — shortages
    if any(kw in text for kw in ['only came with', 'only received',
                                  'supposed to come with', 'did not receive']):
        return 'Missing or Incomplete Components', 0.7

    # Catch-all patterns
    if any(kw in text for kw in ['leak', 'leaking', 'leaks']):
        return 'Defect: Malfunctions / Stops Working', 0.7

    if any(kw in text for kw in ['noise', 'noisy', 'loud', 'grinding',
                                  'squeaking', 'clicking', 'squeaks']):
        return 'Defect: Malfunctions / Stops Working', 0.7

    # Seized / stuck / jammed → functional defect
    if any(kw in text for kw in ['seized', 'stuck', 'jammed', 'cannot be loosened',
                                  'cannot adjust']):
        return 'Defect: Malfunctions / Stops Working', 0.7

    # "non working" / "does not work" (broader forms)
    if any(kw in text for kw in ['non working', 'non-working', 'does not work',
                                  'do not work', 'not work at all']):
        return 'Defect: Malfunctions / Stops Working', 0.8

    # "too tall" → Size: Too Large
    if 'too tall' in text:
        return 'Size: Too Large', 0.8

    # "wrong color" / "wrong one" / "sent the wrong" → Wrong Product
    if any(kw in text for kw in ['wrong color', 'wrong colour', 'wrong one',
                                  'sent the wrong', 'received the wrong',
                                  'was sent the wrong']):
        return 'Wrong Product / Not as Described', 0.8

    # Battery with problem context
    if 'batter' in text:  # catches battery, batteries
        if any(kw in text for kw in ['not holding', 'not working', 'non working',
                                      'dead', 'issue', 'defective', 'problem',
                                      'not charging', 'won\'t charge', 'not taking',
                                      'pushed inside', 'port pushed']):
            return 'Defect: Malfunctions / Stops Working', 0.7

    # "folding" with problem context
    if 'fold' in text:
        if any(kw in text for kw in ['not fold', "doesn't fold", "won't fold",
                                      'issue', 'problem', 'not work', 'mechanism',
                                      'incorrectly', 'correctly']):
            return 'Defect: Malfunctions / Stops Working', 0.7

    # "issue" / "issues" with product context (but not "issue fixed" or "delivery issue")
    if any(kw in text for kw in ['issue with', 'issues with', 'having issue',
                                  'having issues', 'product issue']):
        if not any(kw in text for kw in ['delivery', 'payment', 'order', 'fixed']):
            return 'Defect: Malfunctions / Stops Working', 0.5

    # "pulls to the right/left" / alignment issues
    if any(kw in text for kw in ['pulls to the', 'not pumping', 'not inflat',
                                  'not sturdy', 'rotate too much',
                                  'not engaging', 'not locking']):
        return 'Defect: Malfunctions / Stops Working', 0.7

    # "damaged when arrived" / "arrived damaged" (broader shipping damage)
    if any(kw in text for kw in ['damaged when', 'arrived damaged', 'damaged upon',
                                  'damaged on arrival']):
        return 'Fulfillment: Damaged in Shipping', 0.7

    # "breaks" misspelling of "brakes"
    if 'breaks' in text and any(kw in text for kw in ['not', "won't", "don't",
                                                       'engage', 'lock', 'work']):
        return 'Defect: Malfunctions / Stops Working', 0.7

    # "drilled on the wrong side" / manufacturing defect language
    if any(kw in text for kw in ['drilled', 'manufactured wrong', 'wrong side',
                                  'thread', 'no thread']):
        return 'Defect: Broken / Structural Failure', 0.7

    if any(kw in text for kw in ['warranty', 'warranty claim']):
        return 'Defect: Malfunctions / Stops Working', 0.5

    # Ticket types as weak signals
    if any(kw in text for kw in ['troubleshooting', 'troubleshoot']):
        return 'Defect: Malfunctions / Stops Working', 0.5

    if any(kw in text for kw in ['refund', 'return', 'exchange', 'replacement']):
        return 'Customer: Changed Mind / No Longer Needed', 0.3

    # No match
    return 'Other / Miscellaneous', 0.0


def is_quality_category(category: str) -> bool:
    """Returns True if the category represents an actual quality issue."""
    return category in QUALITY_CATEGORIES


# ─── Data Loading & Cleaning ────────────────────────────────────────────────

def load_zendesk_data(uploaded_file) -> Optional[pd.DataFrame]:
    """Load and deduplicate the B2C Quality Issues Recordings file."""
    try:
        raw = pd.read_excel(uploaded_file)

        required = {"Ticket created - Date", "Ticket ID", "SKU", "Issue", "Ticket Type"}
        missing = required - set(raw.columns)
        if missing:
            st.error(f"Missing expected columns: {', '.join(missing)}")
            return None

        df = raw.drop_duplicates(subset="Ticket ID").copy()
        logger.info(f"Loaded {len(raw):,} rows → {len(df):,} unique tickets")
        df["Ticket created - Date"] = pd.to_datetime(df["Ticket created - Date"], errors="coerce")
        return df

    except Exception as e:
        st.error(f"Failed to read file: {e}")
        logger.error(f"load_zendesk_data error: {e}", exc_info=True)
        return None


def filter_by_date(df: pd.DataFrame, start, end) -> pd.DataFrame:
    mask = (df["Ticket created - Date"] >= pd.Timestamp(start)) & (
        df["Ticket created - Date"] <= pd.Timestamp(end) + timedelta(hours=23, minutes=59, seconds=59)
    )
    return df.loc[mask].copy()


# ─── Categorize All Tickets ─────────────────────────────────────────────────

def categorize_all_tickets(df: pd.DataFrame) -> pd.DataFrame:
    """Apply keyword categorization to every ticket. Returns df with new columns."""
    categories = []
    confidences = []

    for _, row in df.iterrows():
        issue = str(row.get("Issue", "")) if pd.notna(row.get("Issue")) else ""
        cat, conf = categorize_by_keywords(issue)
        categories.append(cat)
        confidences.append(conf)

    df = df.copy()
    df["Category"] = categories
    df["Confidence"] = confidences
    df["Is Quality Issue"] = df["Category"].apply(is_quality_category)
    return df


# ─── Core Aggregation ────────────────────────────────────────────────────────

def build_quality_report(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the consolidated quality-issues report.
    Returns (product_report, category_summary).
    """
    qi = df[df["Is Quality Issue"] == True].copy()

    if qi.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Clean SKUs
    qi = qi[qi["SKU"].notna()]
    qi = qi[~qi["SKU"].str.strip().isin(EXCLUDED_SKUS)]
    qi["Parent SKU"] = qi["SKU"].str[:PARENT_SKU_LENGTH]

    # ── Per-parent aggregations ──────────────────────────────────────────
    grouped = qi.groupby("Parent SKU", sort=False)

    agg = grouped.agg(
        Quality_Issue_Count=("Ticket ID", "count"),
        Unique_SKUs=("SKU", "nunique"),
        First_Seen=("Ticket created - Date", "min"),
        Last_Seen=("Ticket created - Date", "max"),
    ).reset_index()

    # SKU breakdown
    def sku_breakdown(grp):
        counts = grp["SKU"].value_counts()
        return "; ".join(f"{sku} ×{cnt}" for sku, cnt in counts.items())

    sku_notes = grouped.apply(sku_breakdown, include_groups=False).reset_index()
    sku_notes.columns = ["Parent SKU", "SKU Breakdown"]

    # Category breakdown per parent SKU
    def category_breakdown(grp):
        counts = grp["Category"].value_counts()
        return "; ".join(f"{cat} ({cnt})" for cat, cnt in counts.items())

    cat_notes = grouped.apply(category_breakdown, include_groups=False).reset_index()
    cat_notes.columns = ["Parent SKU", "Category Breakdown"]

    # Top 3 issues (raw text)
    def top_issues(grp, n=3):
        counts = grp["Issue"].value_counts().head(n)
        return "\n".join(f"• {issue} ({cnt})" for issue, cnt in counts.items())

    issue_notes = grouped.apply(top_issues, include_groups=False).reset_index()
    issue_notes.columns = ["Parent SKU", "Top Issues"]

    # Ticket-type mix
    def type_mix(grp):
        counts = grp["Ticket Type"].value_counts()
        return "; ".join(f"{t}: {c}" for t, c in counts.items())

    type_notes = grouped.apply(type_mix, include_groups=False).reset_index()
    type_notes.columns = ["Parent SKU", "Ticket Type Breakdown"]

    # Order source mix
    def source_mix(grp):
        if "Order source" not in grp.columns:
            return ""
        counts = grp["Order source"].value_counts()
        return "; ".join(f"{s}: {c}" for s, c in counts.items())

    source_notes = grouped.apply(source_mix, include_groups=False).reset_index()
    source_notes.columns = ["Parent SKU", "Order Source Breakdown"]

    # Return-completed and replacement rates
    rate_cols = {}
    if "Return completed?" in qi.columns:
        rate_cols["Returns_Completed"] = ("Return completed?", "sum")
    if "Replacement SO" in qi.columns:
        rate_cols["Replacements_Sent"] = ("Replacement SO", lambda x: (x.str.strip() != "").sum())

    if rate_cols:
        rate_agg = grouped.agg(**rate_cols).reset_index()
    else:
        rate_agg = agg[["Parent SKU"]].copy()
        rate_agg["Returns_Completed"] = 0
        rate_agg["Replacements_Sent"] = 0

    # ── Merge ────────────────────────────────────────────────────────────
    report = (
        agg.merge(sku_notes, on="Parent SKU")
        .merge(cat_notes, on="Parent SKU")
        .merge(issue_notes, on="Parent SKU")
        .merge(type_notes, on="Parent SKU")
        .merge(source_notes, on="Parent SKU")
        .merge(rate_agg, on="Parent SKU")
    )

    report = report.sort_values("Quality_Issue_Count", ascending=False).reset_index(drop=True)
    report.index = report.index + 1
    report.index.name = "Rank"

    report = report.rename(columns={
        "Quality_Issue_Count": "Quality Issues",
        "Unique_SKUs": "Variant Count",
        "First_Seen": "First Seen",
        "Last_Seen": "Last Seen",
        "Returns_Completed": "Returns Completed",
        "Replacements_Sent": "Replacements Sent",
    })

    # ── Category summary across all products ─────────────────────────────
    cat_summary = (
        qi.groupby("Category")
        .agg(
            Issue_Count=("Ticket ID", "count"),
            Products_Affected=("Parent SKU", "nunique"),
        )
        .sort_values("Issue_Count", ascending=False)
        .reset_index()
    )
    cat_summary.index = cat_summary.index + 1
    cat_summary.index.name = "Rank"
    cat_summary = cat_summary.rename(columns={
        "Issue_Count": "Total Issues",
        "Products_Affected": "Products Affected",
    })

    return report, cat_summary


# ─── KPI helpers ─────────────────────────────────────────────────────────────

def compute_kpis(df: pd.DataFrame, report: pd.DataFrame) -> Dict:
    total_tickets = len(df)
    quality_tickets = int(df["Is Quality Issue"].sum())
    quality_rate = quality_tickets / total_tickets * 100 if total_tickets else 0
    products_affected = len(report)
    top_product = report.iloc[0]["Parent SKU"] if not report.empty else "N/A"
    top_count = int(report.iloc[0]["Quality Issues"]) if not report.empty else 0
    low_conf = int((df["Is Quality Issue"] & (df["Confidence"] < 0.5)).sum())

    return {
        "total_tickets": total_tickets,
        "quality_tickets": quality_tickets,
        "quality_rate": quality_rate,
        "products_affected": products_affected,
        "top_product": top_product,
        "top_count": top_count,
        "low_confidence_count": low_conf,
    }


# ─── Excel Export ────────────────────────────────────────────────────────────

def export_report_xlsx(report: pd.DataFrame, cat_summary: pd.DataFrame,
                       kpis: dict, date_label: str) -> bytes:
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        # Sheet 1 — Product report
        report.to_excel(writer, sheet_name="Quality by Product", startrow=4)
        wb = writer.book
        ws = writer.sheets["Quality by Product"]

        header_fill = PatternFill("solid", fgColor="004366")
        header_font = Font(name="Arial", bold=True, color="FFFFFF", size=14)
        ws.merge_cells("A1:L1")
        ws["A1"].value = f"B2C Zendesk Quality Report — {date_label}"
        ws["A1"].font = header_font
        ws["A1"].fill = header_fill
        ws["A1"].alignment = Alignment(horizontal="center", vertical="center")
        ws.row_dimensions[1].height = 36

        sub_font = Font(name="Arial", size=10, color="004366")
        ws.merge_cells("A2:L2")
        ws["A2"].value = (
            f"Total Tickets: {kpis['total_tickets']:,}  |  "
            f"Quality Issues: {kpis['quality_tickets']:,}  ({kpis['quality_rate']:.1f}%)  |  "
            f"Products Affected: {kpis['products_affected']}  |  "
            f"Top Product: {kpis['top_product']} ({kpis['top_count']})"
        )
        ws["A2"].font = sub_font
        ws["A2"].alignment = Alignment(horizontal="center")

        ws.merge_cells("A3:L3")
        ws["A3"].value = f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        ws["A3"].font = Font(name="Arial", size=9, italic=True, color="666666")
        ws["A3"].alignment = Alignment(horizontal="center")

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

        widths = {"A": 6, "B": 12, "C": 13, "D": 11, "E": 13, "F": 13,
                  "G": 34, "H": 38, "I": 48, "J": 34, "K": 28, "L": 15, "M": 15}
        for col_letter, w in widths.items():
            ws.column_dimensions[col_letter].width = w
        ws.freeze_panes = "A6"

        # Sheet 2 — Category summary
        cat_summary.to_excel(writer, sheet_name="Quality by Category", startrow=2)
        ws2 = writer.sheets["Quality by Category"]
        ws2.merge_cells("A1:D1")
        ws2["A1"].value = "Quality Issues by Category"
        ws2["A1"].font = Font(name="Arial", bold=True, color="004366", size=13)
        ws2["A1"].alignment = Alignment(horizontal="center")
        for col_idx in range(1, 5):
            cell = ws2.cell(row=3, column=col_idx)
            cell.font = col_header_font
            cell.fill = col_header_fill
            cell.alignment = Alignment(horizontal="center")
        for col_letter, w in {"A": 6, "B": 42, "C": 14, "D": 18}.items():
            ws2.column_dimensions[col_letter].width = w

    return buf.getvalue()


# ─── Module Description ─────────────────────────────────────────────────────

MODULE_DESCRIPTION = """
<div style="background: rgba(0, 217, 255, 0.08); border: 1px solid #23b2be;
            border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 1.2rem;">
    <strong style="color:#004366;">📌 Purpose:</strong>
    Categorize <strong>all</strong> Zendesk tickets by Issue text using the same
    medical-device quality categories as the Return Categorizer, then report
    only quality-relevant issues grouped by <strong>Parent SKU</strong> (first 7 chars).<br>
    <strong style="color:#004366;">📊 Output:</strong>
    One table sorted by issue occurrence with category breakdowns per product,
    plus a cross-product category summary.  Does <strong>not</strong> rely on the
    <em>Quality Issues?</em> checkbox — classifies every ticket independently.
</div>
"""

VIVE_TEAL = "#23b2be"
VIVE_NAVY = "#004366"


# ─── Streamlit UI ────────────────────────────────────────────────────────────

def render_b2b_zendesk_reporting():
    """Main render function — drop into app.py's task router."""

    st.markdown("### 🎫 B2C Zendesk Quality Reporting")
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

    st.success(f"Loaded **{len(df):,}** unique tickets")

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
    st.info(f"**{len(filtered):,}** tickets in selected range")

    # ── Categorization mode ──────────────────────────────────────────────
    st.markdown("#### ⚙️ Categorization Mode")
    cat_mode = st.radio(
        "How should tickets be categorized?",
        options=[
            "🔑 Keyword Matching (instant, free)",
            "🔑+🤖 Keywords + AI for unclear tickets (recommended)",
            "🤖 Full AI Analysis (all tickets via API)",
        ],
        index=0,
        key="zendesk_cat_mode",
        horizontal=True,
    )

    # ── Generate report ──────────────────────────────────────────────────
    if st.button("🚀 Generate Quality Report", type="primary", key="zendesk_run"):
        st.session_state["zendesk_report"] = None

        with st.spinner("Categorizing all tickets…"):
            categorized = categorize_all_tickets(filtered)

        # Show categorization stats
        qi_count = int(categorized["Is Quality Issue"].sum())
        low_conf = int((categorized["Confidence"] < 0.5).sum())
        no_match = int((categorized["Confidence"] == 0.0).sum())

        st.toast(f"✅ Categorized {len(categorized):,} tickets → {qi_count} quality issues identified")

        if "🤖" in cat_mode and cat_mode != "🤖 Full AI Analysis (all tickets via API)":
            # AI for unclear ones
            unclear = categorized[categorized["Confidence"] < 0.5]
            if len(unclear) > 0:
                st.warning(f"⚠️ {len(unclear)} tickets had low confidence and could benefit from AI review. "
                          f"AI integration requires API keys configured in the sidebar.")
        elif "Full AI" in cat_mode:
            st.warning("⚠️ Full AI mode requires API keys configured in the sidebar. "
                      "Currently showing keyword-based results.")

        with st.spinner("Building quality report…"):
            report, cat_summary = build_quality_report(categorized)
            kpis = compute_kpis(categorized, report)
            date_label = (
                f"{start_date.strftime('%b %d')} – {end_date.strftime('%b %d, %Y')}"
                if not use_all
                else f"{min_date.strftime('%b %Y')} (All Data)"
            )
            st.session_state["zendesk_report"] = report
            st.session_state["zendesk_cat_summary"] = cat_summary
            st.session_state["zendesk_kpis"] = kpis
            st.session_state["zendesk_date_label"] = date_label
            st.session_state["zendesk_categorized"] = categorized

    # ── Display results ──────────────────────────────────────────────────
    if st.session_state.get("zendesk_report") is not None:
        report = st.session_state["zendesk_report"]
        cat_summary = st.session_state["zendesk_cat_summary"]
        kpis = st.session_state["zendesk_kpis"]
        date_label = st.session_state["zendesk_date_label"]
        categorized = st.session_state["zendesk_categorized"]

        if report.empty:
            st.warning("No quality issues found for the selected parameters.")
            return

        # ── KPI cards ────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown(f"#### 📊 Quality Summary — {date_label}")

        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            st.metric("Total Tickets", f"{kpis['total_tickets']:,}")
        with k2:
            st.metric("Quality Issues", f"{kpis['quality_tickets']:,}")
        with k3:
            st.metric("Quality Rate", f"{kpis['quality_rate']:.1f}%")
        with k4:
            st.metric("Products Affected", kpis["products_affected"])
        with k5:
            st.metric("Low Confidence", kpis["low_confidence_count"],
                      help="Tickets with confidence < 50% — may need AI review")

        # ── Category summary table ───────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 📂 Quality Issues by Category — All Products")

        col_table, col_chart = st.columns([1, 1])
        with col_table:
            st.dataframe(
                cat_summary,
                use_container_width=True,
                height=min(500, 38 + 35 * len(cat_summary)),
                column_config={
                    "Total Issues": st.column_config.NumberColumn(format="%d"),
                    "Products Affected": st.column_config.NumberColumn(format="%d"),
                },
            )
        with col_chart:
            if len(cat_summary) > 0:
                chart_df = cat_summary.set_index("Category")[["Total Issues"]]
                st.bar_chart(chart_df, color=VIVE_TEAL, horizontal=True)

        # ── Main product table ───────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 📋 Quality Issues by Product — Sorted by Occurrence")

        display_cols = [
            "Parent SKU", "Quality Issues", "Variant Count",
            "Category Breakdown", "SKU Breakdown", "Top Issues",
            "Ticket Type Breakdown", "Order Source Breakdown",
            "Returns Completed", "Replacements Sent",
            "First Seen", "Last Seen",
        ]
        available_cols = [c for c in display_cols if c in report.columns]

        st.dataframe(
            report[available_cols],
            use_container_width=True,
            height=min(600, 38 + 35 * len(report)),
            column_config={
                "Quality Issues": st.column_config.NumberColumn(format="%d"),
                "Variant Count": st.column_config.NumberColumn("Variants", format="%d"),
                "First Seen": st.column_config.DateColumn(format="YYYY-MM-DD"),
                "Last Seen": st.column_config.DateColumn(format="YYYY-MM-DD"),
                "Returns Completed": st.column_config.NumberColumn(format="%d"),
                "Replacements Sent": st.column_config.NumberColumn(format="%d"),
                "Category Breakdown": st.column_config.TextColumn("Category Breakdown", width="large"),
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
            top_chart = report.head(15)[["Parent SKU", "Quality Issues"]].set_index("Parent SKU")
            st.bar_chart(top_chart, color=VIVE_TEAL, horizontal=True)

        # ── Drill down ───────────────────────────────────────────────────
        with st.expander("🔎 Drill Down — All Categorized Tickets", expanded=False):
            qi_detail = categorized[categorized["Is Quality Issue"] == True].copy()
            if "SKU" in qi_detail.columns:
                qi_detail["Parent SKU"] = qi_detail["SKU"].str[:PARENT_SKU_LENGTH]
            detail_cols = [
                "Ticket created - Date", "Ticket ID", "Parent SKU", "SKU",
                "Category", "Confidence", "Issue", "Ticket Type",
            ]
            avail = [c for c in detail_cols if c in qi_detail.columns]
            st.dataframe(
                qi_detail[avail].sort_values("Ticket created - Date", ascending=False),
                use_container_width=True, height=400,
            )

        with st.expander("🔍 Low Confidence Tickets (may need AI review)", expanded=False):
            low_conf_df = categorized[categorized["Confidence"] < 0.5].copy()
            if not low_conf_df.empty:
                lc_cols = ["Ticket ID", "Issue", "Category", "Confidence", "Ticket Type", "SKU"]
                avail_lc = [c for c in lc_cols if c in low_conf_df.columns]
                st.dataframe(low_conf_df[avail_lc], use_container_width=True, height=300)
                st.caption(f"{len(low_conf_df)} tickets with confidence < 50%")
            else:
                st.success("All tickets categorized with high confidence.")

        with st.expander("📊 Non-Quality Tickets (excluded from report)", expanded=False):
            non_qi = categorized[categorized["Is Quality Issue"] == False].copy()
            non_summary = non_qi["Category"].value_counts().reset_index()
            non_summary.columns = ["Category", "Count"]
            st.dataframe(non_summary, use_container_width=True)
            st.caption(f"{len(non_qi)} tickets classified as non-quality")

        # ── Export ───────────────────────────────────────────────────────
        st.markdown("---")
        col_dl1, col_dl2, col_dl3, col_clear = st.columns([2, 2, 2, 1])

        with col_dl1:
            xlsx_bytes = export_report_xlsx(report, cat_summary, kpis, date_label)
            st.download_button(
                "⬇️ Download Report (.xlsx)",
                data=xlsx_bytes,
                file_name=f"B2C_Zendesk_Quality_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                key="zendesk_dl_xlsx",
            )

        with col_dl2:
            csv_bytes = report.to_csv(index=True).encode("utf-8")
            st.download_button(
                "⬇️ Product Report (.csv)",
                data=csv_bytes,
                file_name=f"B2C_Zendesk_Product_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                key="zendesk_dl_csv",
            )

        with col_dl3:
            full_csv = categorized.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Full Categorized Data (.csv)",
                data=full_csv,
                file_name=f"B2C_Zendesk_All_Categorized_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                key="zendesk_dl_full",
            )

        with col_clear:
            if st.button("🔄 Reset", key="zendesk_clear"):
                for k in ["zendesk_report", "zendesk_cat_summary", "zendesk_kpis",
                           "zendesk_date_label", "zendesk_categorized"]:
                    st.session_state.pop(k, None)
                st.rerun()
