"""
Verify that the Quality Tracker exports match Smartsheet template columns exactly
"""

import pandas as pd
from quality_tracker_manager import ALL_COLUMNS_LEADERSHIP, ALL_COLUMNS_COMPANY_WIDE

# Read actual Smartsheet templates
leadership_template = pd.read_excel('Tracker_ Priority List (Leadership) (1).xlsx')
company_template = pd.read_excel('Company Wide Quality Tracker.xlsx')

# Get column lists
leadership_actual = list(leadership_template.columns)
company_actual = list(company_template.columns)

print("=" * 80)
print("LEADERSHIP EXPORT VERIFICATION")
print("=" * 80)
print(f"\nExpected columns: {len(ALL_COLUMNS_LEADERSHIP)}")
print(f"Actual columns:   {len(leadership_actual)}\n")

# Check if they match
if ALL_COLUMNS_LEADERSHIP == leadership_actual:
    print("OK PERFECT MATCH! Column order is identical.")
else:
    print("X MISMATCH DETECTED!\n")

    # Find differences
    for i, (expected, actual) in enumerate(zip(ALL_COLUMNS_LEADERSHIP, leadership_actual), 1):
        if expected != actual:
            print(f"  Position {i}: Expected '{expected}' but got '{actual}'")

    # Check for missing columns
    missing = set(ALL_COLUMNS_LEADERSHIP) - set(leadership_actual)
    if missing:
        print(f"\n  Missing from actual: {missing}")

    extra = set(leadership_actual) - set(ALL_COLUMNS_LEADERSHIP)
    if extra:
        print(f"\n  Extra in actual: {extra}")

print("\n" + "=" * 80)
print("COMPANY WIDE EXPORT VERIFICATION")
print("=" * 80)
print(f"\nExpected columns: {len(ALL_COLUMNS_COMPANY_WIDE)}")
print(f"Actual columns:   {len(company_actual)}\n")

# Check if they match
if ALL_COLUMNS_COMPANY_WIDE == company_actual:
    print("OK PERFECT MATCH! Column order is identical.")
else:
    print("X MISMATCH DETECTED!\n")

    # Find differences
    for i, (expected, actual) in enumerate(zip(ALL_COLUMNS_COMPANY_WIDE, company_actual), 1):
        if expected != actual:
            print(f"  Position {i}: Expected '{expected}' but got '{actual}'")

    # Check for missing columns
    missing = set(ALL_COLUMNS_COMPANY_WIDE) - set(company_actual)
    if missing:
        print(f"\n  Missing from actual: {missing}")

    extra = set(company_actual) - set(ALL_COLUMNS_COMPANY_WIDE)
    if extra:
        print(f"\n  Extra in actual: {extra}")

print("\n" + "=" * 80)
print("DETAILED COLUMN LISTING")
print("=" * 80)

print("\n### LEADERSHIP COLUMNS (Expected vs Actual) ###")
for i, (exp, act) in enumerate(zip(ALL_COLUMNS_LEADERSHIP, leadership_actual), 1):
    match = "+" if exp == act else "-"
    print(f"{i:2}. [{match}] {exp}")
    if exp != act:
        print(f"       ACTUAL: {act}")

print("\n### COMPANY WIDE COLUMNS (Expected vs Actual) ###")
for i, (exp, act) in enumerate(zip(ALL_COLUMNS_COMPANY_WIDE, company_actual), 1):
    match = "+" if exp == act else "-"
    print(f"{i:2}. [{match}] {exp}")
    if exp != act:
        print(f"       ACTUAL: {act}")
