"""Product Compliance Test — pre-Odoo field validation tool.

This form mirrors the data fields being evaluated for Odoo integration.
Fill it out with real product data to test field coverage before go-live.
"""

import streamlit as st
import json
from datetime import date, datetime


# ─── Shared constants ────────────────────────────────────────────────────────

ACCENT = '#1a7f5e'
ACCENT_LIGHT = 'rgba(26,127,94,0.08)'
ACCENT_BORDER = 'rgba(26,127,94,0.35)'

STAFF_OPTIONS = [
    '— Select owner —',
    'Alex P.',
    'Jess M.',
    'Regulatory Team',
    'Quality Team',
    'Other / TBD',
]

FDA_CLASSES = [
    '— Select —',
    'Class I',
    'Class II',
    'Class III',
    'Class I Special Controls',
    'Class II Exempt',
    'Non-Exempt',
]

MDR_CLASSES = ['— Select —', 'I', 'Is', 'Im', 'IIa', 'IIb', 'III']
INVIMA_CLASSES = ['— Select —', 'I', 'II', 'III', 'IV']
BIOCOMPAT_OPTIONS = [
    '— Select —',
    'Yes — Supported by Testing',
    'Yes — Supported by Documentation',
    'No',
]
RISK_CLASSES = ['— Select —', 'Low', 'Medium', 'High']

EU_AR_TEXT = "Vive Health EU Authorized Representative\n[Name / Address auto-populates from Odoo AR record]"
UK_RP_TEXT = "Vive Health UK Responsible Person\n[Name / Address auto-populates from Odoo RP record]"


# ─── UI helpers ──────────────────────────────────────────────────────────────

def _section(icon: str, title: str, subtitle: str = '', color: str = ACCENT):
    st.markdown(
        f"""
        <div style="background:linear-gradient(90deg,{color}12,transparent);
                    border-left:4px solid {color};padding:12px 18px;
                    border-radius:0 8px 8px 0;margin:28px 0 16px 0;">
            <div style="font-size:1.05rem;font-weight:700;color:{color};letter-spacing:0.01em;">
                {icon}&nbsp;&nbsp;{title}
            </div>
            {'<div style="color:#666;font-size:0.78rem;margin-top:3px;">'+subtitle+'</div>' if subtitle else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _subsection(title: str, color: str = '#555'):
    st.markdown(
        f'<div style="font-weight:600;color:{color};font-size:0.88rem;'
        f'margin:14px 0 6px 0;padding-bottom:4px;border-bottom:1px solid #eee;">'
        f'{title}</div>',
        unsafe_allow_html=True,
    )


def _info_card(text: str, color: str = '#e8f4ef'):
    st.markdown(
        f'<div style="background:{color};border-radius:6px;padding:10px 14px;'
        f'font-size:0.82rem;color:#444;margin:6px 0 10px 0;white-space:pre-line;">'
        f'{text}</div>',
        unsafe_allow_html=True,
    )


def _required_badge():
    st.markdown(
        '<span style="background:#EB3300;color:#fff;font-size:0.65rem;'
        'padding:2px 6px;border-radius:3px;font-weight:600;margin-left:6px;">REQUIRED</span>',
        unsafe_allow_html=True,
    )


def _yn_row(label: str, key: str, help_text: str = None, default: bool = False) -> bool:
    """Inline toggle that returns the bool value."""
    return st.toggle(label, value=default, key=key, help=help_text)


# ─── Main render function ─────────────────────────────────────────────────────

def render_product_compliance():
    st.markdown(
        f"""
        <style>
        div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stToggle"]) {{
            background: #fafafa;
            border-radius: 6px;
            padding: 4px 8px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg,{ACCENT_LIGHT},transparent);
                    border:1.5px solid {ACCENT_BORDER};border-radius:12px;
                    padding:20px 24px;margin-bottom:24px;">
            <div style="font-size:0.95rem;color:{ACCENT};font-weight:700;margin-bottom:6px;">
                📋 About This Form
            </div>
            <div style="font-size:0.82rem;color:#555;line-height:1.6;">
                This is a <strong>pre-production test</strong> of compliance data fields being evaluated for
                Vive Health's Odoo ERP. Fill in with real product data to validate field coverage,
                conditional logic, and data quality before fields go live.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Product header ────────────────────────────────────────────────────────
    _section('📦', 'Product Identification', 'Basic record header — always required')

    col1, col2, col3 = st.columns(3)
    with col1:
        product_name = st.text_input('Product Name *', placeholder='e.g. Rollator Walker 4-Wheel', key='cp_name')
    with col2:
        sku = st.text_input('SKU / Parent SKU *', placeholder='e.g. VH-100-BLK', key='cp_sku')
    with col3:
        product_line = st.text_input('Product Line / Category', placeholder='e.g. Mobility Aids', key='cp_line')

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — UNIVERSAL
    # ═══════════════════════════════════════════════════════════════════════════
    _section('🌐', 'Universal Requirements', 'Required for every product — medical device or not', ACCENT)

    # — Warnings —
    _subsection('Warnings')
    st.caption('Does this product carry required warning statements? Check all that apply.')
    wcol1, wcol2, wcol3 = st.columns(3)
    with wcol1:
        warn_product = st.checkbox('On Product label', key='cp_warn_product')
    with wcol2:
        warn_manual = st.checkbox('In Manual / IFU', key='cp_warn_manual')
    with wcol3:
        warn_package = st.checkbox('On Package / outer carton', key='cp_warn_package')

    st.text_input(
        'Artwork Folder Link',
        placeholder='https://drive.google.com/... (version-controlled)',
        key='cp_artwork_link',
        help='Link to the controlled artwork folder. Version-control note should appear in folder name or description.',
    )

    # — Regulatory flags —
    _subsection('Regulatory Flags')
    rc1, rc2 = st.columns(2)
    with rc1:
        prop65 = _yn_row(
            'Requires Prop 65 Warning?',
            'cp_prop65',
            'If YES, CA-bound label artwork must surface the appropriate Prop 65 statement.',
        )
        if prop65:
            st.info('⚠️ Prop 65 required — ensure CA-bound label artwork includes compliant warning language.', icon='⚠️')
    with rc2:
        elec_label = _yn_row(
            'Requires Electrical Labeling (IEC 60601)?',
            'cp_elec_label',
            'If YES, label artwork must include relevant IEC/ISO electrical symbols.',
        )
        if elec_label:
            st.info('⚡ IEC 60601 symbols required on label artwork.', icon='⚡')

    hsa_fsa = _yn_row('HSA / FSA Eligible?', 'cp_hsa_fsa', 'Can apply to non-medical-device products too.')

    # — Compliance ownership —
    _subsection('Compliance Ownership')
    oc1, oc2 = st.columns(2)
    with oc1:
        comp_owner = st.selectbox(
            'Compliance Owner',
            STAFF_OPTIONS,
            key='cp_owner',
            help='Staff member assigned at record creation.',
        )
    with oc2:
        review_date = st.date_input(
            'Compliance Review Date',
            value=None,
            key='cp_review_date',
            help='System will alert at 12-month anniversary.',
        )
        if review_date:
            today = date.today()
            days_until = (review_date - today).days if review_date > today else None
            if days_until is not None and days_until <= 30:
                st.warning(f'Review due in {days_until} days.', icon='📅')
            elif days_until is None:
                st.error('Review date is in the past — update required.', icon='🔴')

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — MEDICAL DEVICE GATE
    # ═══════════════════════════════════════════════════════════════════════════
    _section('🏥', 'Medical Device Gate', 'Determines whether regulatory sub-sections apply', '#b04a00')

    is_md = _yn_row(
        'Is this a Medical Device?',
        'cp_is_md',
        'If YES, all fields below become required. If NO, record can be saved without them.',
    )

    if not is_md:
        st.success('Non-medical-device — regulatory sub-sections are not required. Record can be saved.', icon='✅')
    else:
        st.markdown(
            '<div style="background:#fff8f0;border-left:3px solid #b04a00;padding:8px 14px;'
            'border-radius:0 6px 6px 0;font-size:0.82rem;color:#7a3300;margin-bottom:12px;">'
            '🏥 Medical Device confirmed — complete all sections below before saving.</div>',
            unsafe_allow_html=True,
        )

        # Intended Use
        _subsection('Intended Use & Classification')
        st.text_area(
            'Intended Use *',
            placeholder=(
                'Describe the device\'s intended purpose as a regulatory justification statement. '
                'This text pre-populates IFU and Tech File templates.'
            ),
            height=100,
            key='cp_intended_use',
        )

        # Binary product flags
        _subsection('Product Attributes')
        pa1, pa2, pa3 = st.columns(3)
        with pa1:
            sterile = _yn_row('Sterile?', 'cp_sterile')
        with pa2:
            single_use = _yn_row('Single Use?', 'cp_single_use')
        with pa3:
            latex = _yn_row('Contains Latex?', 'cp_latex')
            if latex:
                st.warning('Contains Latex — ensure labeling includes latex warning.', icon='⚠️')

        biocompat = st.selectbox(
            'Biocompatibility Required?',
            BIOCOMPAT_OPTIONS,
            key='cp_biocompat',
        )

        # Shelf life
        shelf_yn = _yn_row('Has Shelf Life / Expiration Date?', 'cp_shelf_yn')
        if shelf_yn:
            st.text_input('Shelf Life Description', placeholder='e.g. 5 years from manufacture date', key='cp_shelf_life')

        # Documentation
        _subsection('Documentation & Files')
        st.text_input(
            'Tech / Design File Link',
            placeholder='Link to Tech File folder (version-controlled)',
            key='cp_tech_file',
        )
        st.text_area(
            'Regulatory Documentation / Product Testing Links',
            placeholder=(
                'One URL per line — include electrical, biocompat, EMC, performance reports:\n'
                'https://drive.google.com/... (Electrical Safety Report)\n'
                'https://drive.google.com/... (Biocompatibility)\n'
                'https://drive.google.com/... (EMC Report)'
            ),
            height=120,
            key='cp_reg_docs',
            help='Repeating field — add as many links as needed, one per line.',
        )

        # Lot / Serial control
        _subsection('Traceability')
        tc1, tc2 = st.columns(2)
        with tc1:
            lot_ctrl = _yn_row('Lot Controlled?', 'cp_lot_ctrl')
        with tc2:
            serial_ctrl = _yn_row('Serial Controlled?', 'cp_serial_ctrl')

        # Risk
        _subsection('Risk Profile')
        risk_class = st.selectbox('Risk Class', RISK_CLASSES, key='cp_risk_class')
        if risk_class and risk_class != '— Select —' and risk_class == 'High':
            st.error('High risk class — Regulatory review required before launch.', icon='🔴')

        primary_hazards = st.text_area(
            'Primary Hazards',
            placeholder='List the primary hazards identified in the risk analysis (one per line).',
            height=80,
            key='cp_primary_hazards',
        )
        st.text_input(
            'Risk Mitigation File Link',
            placeholder='Link to risk file / FMEA / risk management report',
            key='cp_risk_file',
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — USA (FDA)
    # ═══════════════════════════════════════════════════════════════════════════
    _section('🇺🇸', 'USA — FDA Compliance', 'Triggered when Medical Device = YES and sold in US', '#002868')

    if not is_md:
        st.caption('Complete the Medical Device Gate above to unlock this section.')
    else:
        fda_class = st.selectbox('FDA Device Class *', FDA_CLASSES, key='cp_fda_class')

        if fda_class == 'Class II':
            st.markdown(
                '<div style="background:#e8f0fb;border-left:3px solid #002868;padding:8px 14px;'
                'border-radius:0 6px 6px 0;font-size:0.82rem;color:#002868;margin:8px 0;">'
                '⚠️ Class II — 510(k) clearance documentation is <strong>mandatory</strong> before this record can be saved.'
                '</div>',
                unsafe_allow_html=True,
            )
            fd1, fd2 = st.columns(2)
            with fd1:
                pmn_number = st.text_input(
                    '510(k) Number *',
                    placeholder='K######',
                    key='cp_pmn_number',
                    help='Mandatory for Class II — record cannot save without it.',
                )
            with fd2:
                pmn_link = st.text_input(
                    'Link to 510(k) Clearance Letter *',
                    placeholder='https://...',
                    key='cp_pmn_link',
                )

        _subsection('FDA Registration & Listing')
        g1, g2 = st.columns(2)
        with g1:
            product_code_fda = st.text_input(
                'Product Code (FDA)',
                placeholder='3-letter code, e.g. IYO',
                key='cp_product_code_fda',
                help='Read-only reference once populated from FDA database.',
            )
            hcpcs = st.text_input(
                'HCPCS Code',
                placeholder='e.g. E0143',
                key='cp_hcpcs',
                help='Populates on Business Portal.',
            )
            vendor_fda_reg = st.text_input(
                'Vendor FDA Establishment Registration #',
                placeholder='e.g. 3007530XXX',
                key='cp_vendor_fda_reg',
            )
        with g2:
            device_listing_mfr = st.text_input(
                'Device Listing # (Manufacturer)',
                placeholder='e.g. D123456',
                key='cp_listing_mfr',
            )
            device_listing_vive = st.text_input(
                'Device Listing # (Vive)',
                placeholder='e.g. D654321',
                key='cp_listing_vive',
            )
            gudid = _yn_row('Added to GUDID?', 'cp_gudid')

        ndc_required = _yn_row(
            'NDC Code Required?',
            'cp_ndc_required',
            'Drug / combination products only.',
        )
        if ndc_required:
            st.text_input('NDC Code *', placeholder='####-####-##', key='cp_ndc_code')

        _subsection('Barcode / UDI Requirements (USA)')
        st.caption('Check all data carriers required on US product UDI label:')
        u1, u2, u3, u4, u5 = st.columns(5)
        with u1:
            udi_gtin = st.checkbox('GTIN', value=True, key='cp_udi_gtin_us', disabled=True)
            st.caption('Required')
        with u2:
            udi_sku = st.checkbox('SKU', key='cp_udi_sku_us')
        with u3:
            udi_lot = st.checkbox('Lot #', key='cp_udi_lot_us')
        with u4:
            udi_mfg = st.checkbox('MFG Date', key='cp_udi_mfg_us')
        with u5:
            udi_serial = st.checkbox('Serial #', key='cp_udi_serial_us')

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 4 — EU (MDR)
    # ═══════════════════════════════════════════════════════════════════════════
    _section('🇪🇺', 'EU — MDR Compliance', 'Triggered when product is sold in the EU', '#003399')

    sold_eu = _yn_row('Sold in EU?', 'cp_sold_eu')

    if sold_eu:
        ce_mark = _yn_row('CE Mark?', 'cp_ce_mark')

        if ce_mark:
            mdr_class = st.selectbox('MDR Class *', MDR_CLASSES, key='cp_mdr_class')
            if mdr_class in ('IIa', 'IIb', 'III'):
                st.warning('⚠️ Notified Body involvement required — flag for Regulatory review.', icon='⚠️')

            _subsection('EU Authorized Representative')
            _info_card(EU_AR_TEXT, '#eef2fb')

            st.text_input(
                'Declaration of Conformity Link *',
                placeholder='URL or file path (version-controlled)',
                key='cp_eu_doc',
                help='Version-controlled DoC required for all CE-marked products.',
            )

        eudamed = _yn_row('Registered in EUDAMED?', 'cp_eudamed')

        _subsection('Barcode / UDI Requirements (EU)')
        eu_udi_diff = _yn_row(
            'UDI Requirements Different from USA?',
            'cp_eu_udi_diff',
            'If NO, EU UDI will mirror USA selections.',
        )
        if eu_udi_diff:
            eu1, eu2, eu3, eu4, eu5 = st.columns(5)
            with eu1:
                st.checkbox('GTIN', value=True, key='cp_udi_gtin_eu', disabled=True)
                st.caption('Required')
            with eu2:
                st.checkbox('SKU', key='cp_udi_sku_eu')
            with eu3:
                st.checkbox('Lot #', key='cp_udi_lot_eu')
            with eu4:
                st.checkbox('MFG Date', key='cp_udi_mfg_eu')
            with eu5:
                st.checkbox('Serial #', key='cp_udi_serial_eu')
        else:
            st.caption('EU UDI will copy USA selections.')

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 5 — UK (UKCA / MHRA)
    # ═══════════════════════════════════════════════════════════════════════════
    _section('🇬🇧', 'UK — UKCA / MHRA', 'Independent of CE mark — both can apply', '#C8102E')

    sold_uk = _yn_row('Sold in UK?', 'cp_sold_uk')

    if sold_uk:
        ukca_mark = _yn_row('UKCA Mark?', 'cp_ukca_mark', 'Independent of CE — both can apply simultaneously.')

        _subsection('UK Responsible Person')
        _info_card(UK_RP_TEXT, '#fdf0f2')

        st.text_input(
            'Declaration of Conformity Link (UK)',
            placeholder='Same DoC may cover EU + UK, or separate document — note which',
            key='cp_uk_doc',
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 6 — LATAM / Colombia
    # ═══════════════════════════════════════════════════════════════════════════
    _section('🌎', 'LATAM — Colombia (INVIMA)', 'Per Decree 4725', '#007A33')

    sold_co = _yn_row('Sold in Colombia?', 'cp_sold_co')

    if sold_co:
        invima_class = st.selectbox(
            'INVIMA Class *',
            INVIMA_CLASSES,
            key='cp_invima_class',
            help='Per Decreto 4725 classification.',
        )
        if invima_class != '— Select —':
            invima_reg = st.text_input(
                'INVIMA Registration # *',
                placeholder='Links to public INVIMA registry',
                key='cp_invima_reg',
                help='Mandatory when INVIMA Class is populated.',
            )

    # ═══════════════════════════════════════════════════════════════════════════
    # EXPORT / SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════
    _section('💾', 'Export Record', 'Save this compliance record for Odoo field testing', ACCENT)

    st.caption(
        'Export captures all field values as entered — use this to validate data structure '
        'before mapping to Odoo fields.'
    )

    if st.button('📥 Export Compliance Record (JSON)', key='cp_export', type='primary'):
        record = _collect_record()
        json_str = json.dumps(record, indent=2, default=str)
        st.download_button(
            label='⬇️ Download compliance_record.json',
            data=json_str,
            file_name=f'compliance_{sku or "product"}_{date.today().isoformat()}.json',
            mime='application/json',
            key='cp_download',
        )
        with st.expander('Preview record JSON', expanded=False):
            st.code(json_str, language='json')


def _collect_record() -> dict:
    """Collect all session state compliance fields into a dict for export."""
    s = st.session_state
    return {
        'meta': {
            'exported_at': datetime.now().isoformat(),
            'tool': 'Product Compliance Test (pre-Odoo)',
        },
        'product': {
            'name':          s.get('cp_name', ''),
            'sku':           s.get('cp_sku', ''),
            'product_line':  s.get('cp_line', ''),
        },
        'universal': {
            'warnings': {
                'on_product': s.get('cp_warn_product', False),
                'on_manual':  s.get('cp_warn_manual', False),
                'on_package': s.get('cp_warn_package', False),
            },
            'artwork_folder':       s.get('cp_artwork_link', ''),
            'prop_65_required':     s.get('cp_prop65', False),
            'iec_60601_required':   s.get('cp_elec_label', False),
            'hsa_fsa_eligible':     s.get('cp_hsa_fsa', False),
            'compliance_owner':     s.get('cp_owner', ''),
            'compliance_review_date': str(s.get('cp_review_date', '')),
        },
        'medical_device': {
            'is_medical_device':    s.get('cp_is_md', False),
            'intended_use':         s.get('cp_intended_use', ''),
            'sterile':              s.get('cp_sterile', False),
            'single_use':           s.get('cp_single_use', False),
            'contains_latex':       s.get('cp_latex', False),
            'biocompatibility':     s.get('cp_biocompat', ''),
            'has_shelf_life':       s.get('cp_shelf_yn', False),
            'shelf_life':           s.get('cp_shelf_life', ''),
            'tech_file_link':       s.get('cp_tech_file', ''),
            'regulatory_doc_links': s.get('cp_reg_docs', ''),
            'lot_controlled':       s.get('cp_lot_ctrl', False),
            'serial_controlled':    s.get('cp_serial_ctrl', False),
            'risk_class':           s.get('cp_risk_class', ''),
            'primary_hazards':      s.get('cp_primary_hazards', ''),
            'risk_mitigation_file': s.get('cp_risk_file', ''),
        },
        'usa_fda': {
            'fda_class':            s.get('cp_fda_class', ''),
            '510k_number':          s.get('cp_pmn_number', ''),
            '510k_letter_link':     s.get('cp_pmn_link', ''),
            'product_code_fda':     s.get('cp_product_code_fda', ''),
            'hcpcs_code':           s.get('cp_hcpcs', ''),
            'vendor_fda_reg':       s.get('cp_vendor_fda_reg', ''),
            'device_listing_mfr':   s.get('cp_listing_mfr', ''),
            'device_listing_vive':  s.get('cp_listing_vive', ''),
            'added_to_gudid':       s.get('cp_gudid', False),
            'ndc_required':         s.get('cp_ndc_required', False),
            'ndc_code':             s.get('cp_ndc_code', ''),
            'udi': {
                'gtin':       True,
                'sku':        s.get('cp_udi_sku_us', False),
                'lot':        s.get('cp_udi_lot_us', False),
                'mfg_date':   s.get('cp_udi_mfg_us', False),
                'serial':     s.get('cp_udi_serial_us', False),
            },
        },
        'eu_mdr': {
            'sold_in_eu':           s.get('cp_sold_eu', False),
            'ce_mark':              s.get('cp_ce_mark', False),
            'mdr_class':            s.get('cp_mdr_class', ''),
            'doc_link':             s.get('cp_eu_doc', ''),
            'registered_in_eudamed':s.get('cp_eudamed', False),
            'udi_different_from_us':s.get('cp_eu_udi_diff', False),
            'udi': {
                'gtin':     True,
                'sku':      s.get('cp_udi_sku_eu', False),
                'lot':      s.get('cp_udi_lot_eu', False),
                'mfg_date': s.get('cp_udi_mfg_eu', False),
                'serial':   s.get('cp_udi_serial_eu', False),
            },
        },
        'uk_ukca': {
            'sold_in_uk':   s.get('cp_sold_uk', False),
            'ukca_mark':    s.get('cp_ukca_mark', False),
            'doc_link':     s.get('cp_uk_doc', ''),
        },
        'latam_colombia': {
            'sold_in_colombia': s.get('cp_sold_co', False),
            'invima_class':     s.get('cp_invima_class', ''),
            'invima_reg':       s.get('cp_invima_reg', ''),
        },
    }
