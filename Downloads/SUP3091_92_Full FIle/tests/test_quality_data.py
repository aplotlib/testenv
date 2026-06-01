import pytest
from quality.data import (
    FAILURE_MODES, VARIANT_RATES, MONTHLY_TICKETS,
    MONTHLY_RETURNS, REWORK_WAREHOUSES, CORRECTIVE_ACTIONS,
    KEY_DATES, COLORS, OUTPUT_DIR, HALF_RATES,
)

VALID_STATUSES = {'VALIDATED', 'IN PROGRESS', 'DONE'}
REQUIRED_MODE_KEYS   = {'name', 'count', 'color'}
REQUIRED_VARIANT_KEYS = {'variant', 'units', 'plastic_defects', 'plastic_rate',
                          'adhesive_defects', 'adhesive_rate', 'total_rate'}
REQUIRED_WH_KEYS     = {'wh', 'sleeves'}
REQUIRED_ACTION_KEYS = {'action', 'status', 'notes'}
REQUIRED_DATE_KEYS   = {'date_str', 'label', 'annotation_type'}


def test_four_failure_modes():
    assert len(FAILURE_MODES) == 4

def test_failure_modes_have_required_keys():
    for i, mode in enumerate(FAILURE_MODES):
        missing = REQUIRED_MODE_KEYS - set(mode.keys())
        assert not missing, f"FAILURE_MODES[{i}] missing: {missing}"

def test_failure_mode_counts_are_positive():
    for mode in FAILURE_MODES:
        assert mode['count'] > 0

def test_failure_mode_colors_are_hex():
    for mode in FAILURE_MODES:
        assert mode['color'].startswith('#') and len(mode['color']) == 7

def test_failure_mode_total_is_132():
    assert sum(m['count'] for m in FAILURE_MODES) == 132

def test_four_variant_rates():
    assert len(VARIANT_RATES) == 4

def test_variant_rates_have_required_keys():
    for i, v in enumerate(VARIANT_RATES):
        missing = REQUIRED_VARIANT_KEYS - set(v.keys())
        assert not missing, f"VARIANT_RATES[{i}] missing: {missing}"

def test_variant_units_are_positive():
    for v in VARIANT_RATES:
        assert v['units'] > 0

def test_monthly_tickets_has_14_months():
    assert len(MONTHLY_TICKETS) == 14

def test_monthly_returns_has_13_months():
    assert len(MONTHLY_RETURNS) == 13

def test_total_returns_is_610():
    assert sum(MONTHLY_RETURNS.values()) == 610

def test_four_rework_warehouses():
    assert len(REWORK_WAREHOUSES) == 4

def test_rework_warehouses_have_required_keys():
    for i, wh in enumerate(REWORK_WAREHOUSES):
        missing = REQUIRED_WH_KEYS - set(wh.keys())
        assert not missing

def test_rework_total_sleeves_is_10892():
    assert sum(wh['sleeves'] for wh in REWORK_WAREHOUSES) == 10_892

def test_corrective_actions_statuses_are_valid():
    for a in CORRECTIVE_ACTIONS:
        assert a['status'] in VALID_STATUSES

def test_corrective_actions_have_required_keys():
    for i, a in enumerate(CORRECTIVE_ACTIONS):
        missing = REQUIRED_ACTION_KEYS - set(a.keys())
        assert not missing

def test_five_key_dates():
    assert len(KEY_DATES) == 5

def test_key_dates_have_required_keys():
    for i, d in enumerate(KEY_DATES):
        missing = REQUIRED_DATE_KEYS - set(d.keys())
        assert not missing

def test_colors_has_required_keys():
    required = {'plastic', 'adhesive', 'bladder', 'other', 'cold_shading',
                'qa_rec', 'decision', 'background', 'text', 'grid', 'returns_line'}
    assert not (required - set(COLORS.keys()))

def test_output_dir_is_string():
    assert isinstance(OUTPUT_DIR, str) and len(OUTPUT_DIR) > 0

def test_half_rates_has_two_entries():
    assert len(HALF_RATES) == 2
