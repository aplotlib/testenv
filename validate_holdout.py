"""Honest generalization check for the pattern tier.

The collision-resolution rules and the new regexes were designed by looking at
this dataset, so measuring on the same rows would be overfitting. This splits
50/50 on a fixed seed and reports TRAIN vs HELD-OUT separately. If the rules
generalize, the two precisions should be close. A big train/test gap means the
rules memorized this file and should not be trusted on next month's export.

Usage (from the repository root):

    python validate_holdout.py [reviewed_export.xlsx]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from enhanced_ai_analysis import preprocess_complaint, quick_categorize

TRUTH = sys.argv[1] if len(sys.argv) > 1 else "categorized_20260630.xlsx"
if not os.path.exists(TRUTH):
    sys.exit(f"ERROR: reviewed export not found: {TRUTH}")

df = pd.read_excel(TRUTH, sheet_name="Returns", dtype=str)
df = df[df["Complaint"].notna() & (df["Complaint"].str.strip() != "")]
df = df[df["Category"].notna() & (df["Category"].str.strip() != "")].reset_index(drop=True)

shuffled = df.sample(frac=1.0, random_state=1234).reset_index(drop=True)
mid = len(shuffled) // 2
splits = {"TRAIN (rules derived here)": shuffled.iloc[:mid],
          "HELD-OUT (never inspected)": shuffled.iloc[mid:]}

for name, part in splits.items():
    resolved = correct = 0
    misses = []
    for _, r in part.iterrows():
        txt = preprocess_complaint(r["Complaint"])
        pred = quick_categorize(txt)
        if pred:
            resolved += 1
            if pred == r["Category"]:
                correct += 1
            else:
                misses.append((txt, pred, r["Category"]))
    cov = resolved / len(part)
    prec = correct / resolved if resolved else 0.0
    print(f"\n{name}")
    print(f"  rows       : {len(part):,}")
    print(f"  resolved   : {resolved:,}  (coverage {cov:.1%})")
    print(f"  precision  : {prec:.2%}   ({correct:,} correct, {len(misses)} wrong)")
    for txt, p, a in misses[:6]:
        print(f'     MISS "{" ".join(txt.split())[:80]}"')
        print(f"          pred={p}  truth={a}")
