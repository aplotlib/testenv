#!/usr/bin/env python
"""
Accuracy benchmark for the Amazon return categorizer.

Measures the categorizer against a human-reviewed "ground truth" export — i.e. a
file that has already been categorized and checked by the quality team.

Two tiers are measured separately, because they have different cost profiles:

  * regex tier (`quick_categorize`) — free, no API calls. Runs on every row.
    Reports COVERAGE (how many rows it resolves without AI) and PRECISION
    (of the rows it does resolve, how many match ground truth).
  * AI tier (`categorize_return`) — costs money. Only runs with --ai, and only
    on a random sample (--sample N).

Usage
-----
    # free: regex tier only, all rows
    python validate_categorizer.py

    # explicit files
    python validate_categorizer.py --truth "categorized_20260630.xlsx"

    # include the AI tier on a 200-row sample (needs ANTHROPIC_API_KEY)
    python validate_categorizer.py --ai --sample 200

Exit code is 1 if regex precision falls below --min-precision, so this can be
wired into CI as a regression guard.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import Counter, defaultdict

import pandas as pd

DEFAULT_TRUTH = "categorized_20260630.xlsx"
COMPLAINT_COL = "Complaint"
CATEGORY_COL = "Category"


def load_truth(path: str, complaint_col: str, category_col: str) -> pd.DataFrame:
    """Load a human-reviewed categorized export as the benchmark set."""
    xl = pd.ExcelFile(path)
    # The tool writes its output to a sheet named 'Returns'; fall back to first.
    sheet = "Returns" if "Returns" in xl.sheet_names else xl.sheet_names[0]
    df = xl.parse(sheet, dtype=str)

    missing = [c for c in (complaint_col, category_col) if c not in df.columns]
    if missing:
        sys.exit(
            f"ERROR: {path} (sheet '{sheet}') is missing column(s): {missing}\n"
            f"       Columns present: {list(df.columns)}"
        )

    df = df[df[complaint_col].notna() & (df[complaint_col].str.strip() != "")]
    df = df[df[category_col].notna() & (df[category_col].str.strip() != "")]
    return df.reset_index(drop=True)


def report(title: str, rows: list[tuple[str, str, str]], total_considered: int) -> float:
    """Print a precision report. `rows` is (complaint, predicted, actual)."""
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)

    if not rows:
        print("  no predictions made")
        return 0.0

    hits = [r for r in rows if r[1] == r[2]]
    precision = len(hits) / len(rows)
    coverage = len(rows) / total_considered if total_considered else 0.0

    print(f"  rows considered : {total_considered:,}")
    print(f"  predictions made: {len(rows):,}  (coverage {coverage:.1%})")
    print(f"  correct         : {len(hits):,}  (precision {precision:.1%})")

    misses = [r for r in rows if r[1] != r[2]]
    if misses:
        print(f"\n  -- top confusions (predicted -> actual) --")
        pairs = Counter((p, a) for _, p, a in misses)
        for (pred, actual), n in pairs.most_common(12):
            print(f"   {n:5d}  {pred}  ->  {actual}")

        print(f"\n  -- 8 example misses --")
        for complaint, pred, actual in misses[:8]:
            text = " ".join(str(complaint).split())[:110]
            print(f'   "{text}"')
            print(f"        predicted: {pred}")
            print(f"        truth    : {actual}")

    return precision


def run_regex_tier(df: pd.DataFrame, complaint_col: str, category_col: str) -> float:
    # Mirror production order exactly: categorize_return() preprocesses the text
    # (HTML-entity decode, apostrophe normalize, contraction expansion) BEFORE
    # handing it to quick_categorize. Benchmarking raw text would measure a
    # pipeline that does not exist.
    from enhanced_ai_analysis import preprocess_complaint, quick_categorize

    rows = []
    for _, r in df.iterrows():
        complaint = preprocess_complaint(r[complaint_col])
        predicted = quick_categorize(complaint)
        if predicted:
            rows.append((complaint, predicted, r[category_col]))

    return report("REGEX TIER  (quick_categorize — free, no API calls)", rows, len(df))


def run_ai_tier(df: pd.DataFrame, complaint_col: str, category_col: str, sample: int) -> float:
    from enhanced_ai_analysis import EnhancedAIAnalyzer, AIProvider

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ERROR: --ai needs ANTHROPIC_API_KEY set in the environment.")

    sub = df.sample(n=min(sample, len(df)), random_state=42).reset_index(drop=True)
    analyzer = EnhancedAIAnalyzer(AIProvider.CLAUDE, max_workers=4)

    batch = [
        {"index": i, "complaint": r[complaint_col], "fba_reason": None}
        for i, r in sub.iterrows()
    ]

    print(f"\ncalling Claude on {len(batch):,} sampled rows (this costs money)...")
    results = analyzer.categorize_batch(batch, mode="standard")

    by_index = {r["index"]: r for r in results}
    rows = []
    for i, r in sub.iterrows():
        res = by_index.get(i)
        if res:
            rows.append((r[complaint_col], res.get("category", ""), r[category_col]))

    precision = report(
        f"FULL PIPELINE  (corrections -> regex -> Claude), {len(rows):,}-row sample",
        rows,
        len(rows),
    )

    try:
        cost = analyzer.get_cost_summary()
        print(f"\n  cost this run: ${cost.get('total_cost', 0):.4f}")
    except Exception:
        pass
    return precision


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--truth", default=DEFAULT_TRUTH, help=f"reviewed categorized export (default: {DEFAULT_TRUTH})")
    p.add_argument("--complaint-col", default=COMPLAINT_COL)
    p.add_argument("--category-col", default=CATEGORY_COL)
    p.add_argument("--ai", action="store_true", help="also benchmark the AI tier (costs money)")
    p.add_argument("--sample", type=int, default=200, help="rows to sample for the AI tier (default: 200)")
    p.add_argument("--min-precision", type=float, default=0.80,
                   help="exit 1 if regex precision drops below this (default: 0.80)")
    args = p.parse_args()

    if not os.path.exists(args.truth):
        sys.exit(f"ERROR: ground-truth file not found: {args.truth}")

    random.seed(42)
    df = load_truth(args.truth, args.complaint_col, args.category_col)

    print(f"ground truth : {args.truth}")
    print(f"usable rows  : {len(df):,}  (non-empty complaint AND category)")
    print(f"distinct cats: {df[args.category_col].nunique()}")

    regex_precision = run_regex_tier(df, args.complaint_col, args.category_col)

    if args.ai:
        run_ai_tier(df, args.complaint_col, args.category_col, args.sample)

    print()
    if regex_precision < args.min_precision:
        print(f"FAIL: regex precision {regex_precision:.1%} < threshold {args.min_precision:.1%}")
        sys.exit(1)
    print(f"PASS: regex precision {regex_precision:.1%} >= threshold {args.min_precision:.1%}")


if __name__ == "__main__":
    main()
