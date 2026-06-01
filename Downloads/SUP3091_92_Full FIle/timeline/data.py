from datetime import datetime

COLORS = {
    'detection':     '#E07B39',
    'rca':           '#4A7DB5',
    'decision':      '#7B5EA7',
    'containment':   '#2E8B7A',
    'positive_badge':'#27AE60',
    'spine':         '#AAAAAA',
    'text':          '#2C2C2C',
    'background':    '#F8F9FA',
}

MARKERS = {
    'detection':   'o',
    'rca':         'D',
    'decision':    's',
    'containment': 'h',
}

PHASES = [
    {'key': 'early',         'name': 'EARLY SIGNALS',       'color': '#E07B39'},
    {'key': 'investigation', 'name': 'INVESTIGATION & RCA', 'color': '#4A7DB5'},
    {'key': 'decisions',     'name': 'DECISIONS',           'color': '#7B5EA7'},
    {'key': 'containment',   'name': 'CONTAINMENT',         'color': '#2E8B7A'},
]

EVENTS = [
    # ── EARLY SIGNALS ──────────────────────────────────────────────────────────
    {
        'date':     datetime(2025, 5, 24),
        'label':    'Incoming plastic\nfractures observed',
        'type':     'detection',
        'note':     None,
        'positive': False,
        'phase':    'early',
    },
    {
        'date':     datetime(2025, 8, 30),
        'label':    'Duoyuan molder\nblacklisted (SUP3076)',
        'type':     'decision',
        'note':     '~7,000 units on related SKU affected',
        'positive': True,
        'phase':    'early',
    },
    {
        'date':     datetime(2025, 9, 15),
        'label':    'Ticket #97091:\n"93 units damaged"\nreported',
        'type':     'detection',
        'note':     None,
        'positive': False,
        'phase':    'early',
    },
    {
        'date':     datetime(2025, 9, 25),
        'label':    'Boiling/conditioning\n+ 40-bend test\nimplemented',
        'type':     'decision',
        'note':     'Team acts immediately; new incoming inspection standard',
        'positive': True,
        'phase':    'early',
    },
    {
        'date':     datetime(2025, 11, 12),
        'label':    'Adhesive/MSDS\ninvestigation opened',
        'type':     'detection',
        'note':     None,
        'positive': False,
        'phase':    'early',
    },
    # ── INVESTIGATION & RCA ────────────────────────────────────────────────────
    {
        'date':     datetime(2026, 1, 5),
        'label':    'First complaint\nof current cluster',
        'type':     'detection',
        'note':     None,
        'positive': False,
        'phase':    'investigation',
    },
    {
        'date':     datetime(2026, 1, 7),
        'label':    'Tall boots flagged;\nsleeve fix identified',
        'type':     'decision',
        'note':     'SUP3091 & SUP3092 Tall need rework; sleeve interim solution developed',
        'positive': True,
        'phase':    'investigation',
    },
    {
        'date':     datetime(2026, 1, 13),
        'label':    'Short-height problems\nsurface; new adhesive\nevaluated',
        'type':     'rca',
        'note':     None,
        'positive': False,
        'phase':    'investigation',
    },
    {
        'date':     datetime(2026, 2, 4),
        'label':    'Formal cross-variant\ntesting thread opened',
        'type':     'rca',
        'note':     'MPF & Vive teams mobilize full test program',
        'positive': True,
        'phase':    'investigation',
    },
    {
        'date':     datetime(2026, 2, 9),
        'label':    '0°C freezer test\nFAILS',
        'type':     'rca',
        'note':     'Cold-embrittlement confirmed: PA6/glass-filled PP brittle at 0°C',
        'positive': False,
        'phase':    'investigation',
    },
    {
        'date':     datetime(2026, 3, 1),
        'label':    'Cold-geography\ncorrelation locked',
        'type':     'rca',
        'note':     'Defects map to cold-weather warehouses and customers — mechanism locked',
        'positive': True,
        'phase':    'investigation',
    },
    # ── DECISIONS ─────────────────────────────────────────────────────────────
    # NB: decisions phase overlaps investigation in time; renderer sorts events by date
    {
        'date':     datetime(2026, 2, 5),
        'label':    'Vendor switch to\nPacRim/Pailing agreed',
        'type':     'decision',
        'note':     None,
        'positive': False,
        'phase':    'decisions',
    },
    {
        'date':     datetime(2026, 4, 17),
        'label':    'Rework order placed;\n~5,092 sleeves,\nphased plan',
        'type':     'decision',
        'note':     'Phased approach approved after direction changes; scope fully defined',
        'positive': False,
        'phase':    'decisions',
    },
    {
        'date':     datetime(2026, 5, 13),
        'label':    'Drop test: Pailing\noutperforms Duoyuan;\nrib-removal fix',
        'type':     'rca',
        'note':     'New molder validated; structural design improvement confirmed',
        'positive': True,
        'phase':    'decisions',
    },
    # ── CONTAINMENT ────────────────────────────────────────────────────────────
    {
        'date':     datetime(2026, 5, 20),
        'label':    'Full stop-ship:\nall SUP3091/3092',
        'type':     'containment',
        'note':     'All inventory halted; Duoyuan/PacRim pricing finalized',
        'positive': False,
        'phase':    'containment',
    },
    {
        'date':     datetime(2026, 5, 28),
        'label':    'KS warehouse: Jason\nproactively flags\nreturn risk',
        'type':     'containment',
        'note':     'KS lead identifies return-to-inventory risk; team trained; pack station photos added',
        'positive': True,
        'phase':    'containment',
    },
    {
        'date':     datetime(2026, 5, 30),
        'label':    'JX warehouse:\nQC-hold gap\ndiscovered & corrected',
        'type':     'containment',
        'note':     'Units in sellable locations (non-sellable in Odoo only); all WH leads checking inventory',
        'positive': False,
        'phase':    'containment',
    },
]
