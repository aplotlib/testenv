import os

OUTPUT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COLORS = {
    'plastic':      '#E07B39',
    'adhesive':     '#4A7DB5',
    'bladder':      '#7B5EA7',
    'other':        '#888888',
    'cold_shading': '#E07B39',
    'qa_rec':       '#2E8B7A',
    'decision':     '#C0392B',
    'background':   '#F8F9FA',
    'text':         '#2C2C2C',
    'grid':         '#DDDDDD',
    'returns_line': '#AAAAAA',
}

FAILURE_MODES = [
    {'name': 'Plastic\n(buckle / side-stay / frame)', 'count': 40, 'color': '#E07B39'},
    {'name': 'Adhesive / Velcro',                     'count': 38, 'color': '#4A7DB5'},
    {'name': 'Air Bladder / Pump',                    'count': 24, 'color': '#7B5EA7'},
    {'name': 'Other / Unknown',                       'count': 30, 'color': '#888888'},
]

VARIANT_RATES = [
    {'variant': 'SUP3091 Tall',  'units': 9_057,  'plastic_defects': 19, 'plastic_rate': 2.10, 'adhesive_defects': 14, 'adhesive_rate': 1.55, 'total_rate': 6.58},
    {'variant': 'SUP3091 Short', 'units': 19_509, 'plastic_defects': 9,  'plastic_rate': 0.46, 'adhesive_defects': 13, 'adhesive_rate': 0.67, 'total_rate': 1.58},
    {'variant': 'SUP3092 Tall',  'units': 5_902,  'plastic_defects': 7,  'plastic_rate': 1.19, 'adhesive_defects': 9,  'adhesive_rate': 1.52, 'total_rate': 2.71},
    {'variant': 'SUP3092 Short', 'units': 11_778, 'plastic_defects': 2,  'plastic_rate': 0.17, 'adhesive_defects': 1,  'adhesive_rate': 0.08, 'total_rate': 0.25},
]

MONTHLY_TICKETS = {
    "Apr '25": 1,  "May '25": 8,  "Jun '25": 4,  "Jul '25": 10,
    "Aug '25": 10, "Sep '25": 8,  "Oct '25": 15, "Nov '25": 23,
    "Dec '25": 14, "Jan '26": 14, "Feb '26": 16, "Mar '26": 14,
    "Apr '26": 23, "May '26": 13,
}

MONTHLY_RETURNS = {
    "May '25": 19, "Jun '25": 78, "Jul '25": 8,  "Aug '25": 47,
    "Sep '25": 63, "Oct '25": 105,"Nov '25": 84, "Dec '25": 14,
    "Jan '26": 28, "Feb '26": 10, "Mar '26": 76, "Apr '26": 51,
    "May '26": 27,
}

REWORK_WAREHOUSES = [
    {'wh': 'PA (Bristol)', 'sleeves': 5_080},
    {'wh': 'CA',           'sleeves': 2_852},
    {'wh': 'KS',           'sleeves': 2_264},
    {'wh': 'JX (Jacksonville)', 'sleeves': 696},
]

CORRECTIVE_ACTIONS = [
    {'action': 'Vendor switch (Pailing)',                'status': 'VALIDATED',   'notes': 'Drop test passed; rib-removal fix confirmed'},
    {'action': 'New adhesive',                           'status': 'VALIDATED',   'notes': '1,000-cycle test vs. ~100 cycles (old)'},
    {'action': 'Plastic material change\n(cold additive)', 'status': 'IN PROGRESS', 'notes': 'All variants; permanent fix'},
    {'action': 'Rework — Tall boots\n(sleeve)',          'status': 'IN PROGRESS', 'notes': '10,892 sleeves; ETA pending'},
    {'action': 'Pack station photo program',             'status': 'DONE',        'notes': 'Live at all warehouses'},
    {'action': 'Inventory reconciliation\n(all WH)',     'status': 'IN PROGRESS', 'notes': 'JX confirmed; counts underway'},
    {'action': 'Unchecked stock from sale',              'status': 'DONE',        'notes': 'Stop-ship active May 2026'},
]

KEY_DATES = [
    {'date_str': '2025-11', 'label': 'MSDS flag',        'annotation_type': 'flag'},
    {'date_str': '2026-01', 'label': 'Tall flagged',      'annotation_type': 'flag'},
    {'date_str': '2026-02', 'label': 'Testing opened',    'annotation_type': 'flag'},
    {'date_str': '2026-02', 'label': 'Freezer FAILS',     'annotation_type': 'test'},
    {'date_str': '2026-05', 'label': 'STOP-SHIP',         'annotation_type': 'stop'},
]

HALF_RATES = {
    'H1 (Jun–Nov 2025)':       1.9,
    'H2 (Dec 2025–May 2026)':  3.7,
}
