"""
Quality Resources and External Links

Comprehensive list of tools, databases, and documentation for quality management.
"""

# Quality Resources organized by category
QUALITY_RESOURCES = {
    "Vive Quality Tools": {
        "icon": "🛠️",
        "description": "Internal Vive quality management tools and dashboards",
        "links": [
            {
                "name": "Autocapa & International Intelligence",
                "url": "https://autocapa.streamlit.app/",
                "description": "Recall alert query tool for international market monitoring"
            },
            {
                "name": "Quality Goals 2026",
                "url": "https://docs.google.com/spreadsheets/d/1f2GmZuDl83AWYb5VacpHVbII6C-emhgIQG38izbl_4k/edit?gid=285616133#gid=285616133",
                "description": "Annual quality objectives and KPI tracking"
            },
            {
                "name": "Quality Impact Tracker ($)",
                "url": "https://docs.google.com/spreadsheets/d/1OiqI2juy-atLbwJ4pUS_A5w0UNXC01-HAa13ttl3Txw/edit?gid=1309786158#gid=1309786158",
                "description": "Financial impact tracking for quality initiatives"
            },
            {
                "name": "Quality Intranet Site",
                "url": "https://sites.google.com/vivehealth.com/quality/quality-cases",
                "description": "Central hub for quality cases and documentation"
            },
            {
                "name": "All Streamlit Apps",
                "url": "https://share.streamlit.io/",
                "description": "Browse all available Streamlit applications"
            }
        ]
    },
    "Quality Standards & Calculators": {
        "icon": "📊",
        "description": "Statistical tools and quality acceptance criteria calculators",
        "links": [
            {
                "name": "AQL Calculator",
                "url": "https://tetrainspection.com/aql-calculator-acceptable-quality-limit/",
                "description": "Acceptable Quality Limit calculator for sampling plans (ISO 2859-1)"
            }
        ]
    },
    "Vive QMS Documentation": {
        "icon": "📚",
        "description": "Quality Management System documents and procedures",
        "links": [
            {
                "name": "GDrive QMS Folder",
                "url": "https://drive.google.com/drive/folders/105Tn_ngdhmNGCoqnlTJNe9Ah6wRPDi4j",
                "description": "Master folder for all QMS documentation"
            },
            {
                "name": "Quality SOPs",
                "url": "https://docs.google.com/document/d/1vmRZkIqWPrg_u7EEZ1chma6KUUevTJsw3x73E6HILeQ/edit?tab=t.0",
                "description": "Standard Operating Procedures for quality processes"
            },
            {
                "name": "Quality Manual",
                "url": "https://docs.google.com/document/d/1HUTZGkZRoeW9_9W4VQcly11546yZ_XuJIiOeyFHmLsI/edit?tab=t.0#heading=h.47pjta77rxqh",
                "description": "ISO 13485 Quality Manual - top-level QMS document"
            }
        ]
    },
    "US Regulatory Databases": {
        "icon": "🇺🇸",
        "description": "FDA databases for device clearances, recalls, and adverse events",
        "links": [
            {
                "name": "FDA 510(k) Database",
                "url": "https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm",
                "description": "Search premarket notifications (510k) for medical devices"
            },
            {
                "name": "FDA Recall Database",
                "url": "https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfres/res.cfm",
                "description": "Search recalls of medical devices and equipment"
            },
            {
                "name": "FDA MAUDE Database",
                "url": "https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfmaude/search.cfm",
                "description": "Manufacturer and User Facility Device Experience - adverse event reports"
            },
            {
                "name": "FDA Registration & Listing",
                "url": "https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfrl/rl.cfm",
                "description": "Search registered medical device establishments and listed products"
            },
            {
                "name": "FDA Warning Letters",
                "url": "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/compliance-actions-and-activities/warning-letters",
                "description": "FDA warning letters to companies for regulatory violations"
            }
        ]
    },
    "European Union (EU) Databases": {
        "icon": "🇪🇺",
        "description": "European medical device databases and regulatory resources",
        "links": [
            {
                "name": "EUDAMED (EU Medical Device Database)",
                "url": "https://ec.europa.eu/tools/eudamed/",
                "description": "European database on medical devices - UDI, registrations, certificates"
            },
            {
                "name": "EU MDR/IVDR Regulations",
                "url": "https://health.ec.europa.eu/medical-devices-sector/new-regulations_en",
                "description": "Medical Device Regulation (2017/745) and In Vitro Diagnostic Regulation"
            },
            {
                "name": "NANDO (Notified Bodies)",
                "url": "https://ec.europa.eu/growth/tools-databases/nando/",
                "description": "Database of notified bodies for EU conformity assessment"
            },
            {
                "name": "EU Safety Gate (RAPEX)",
                "url": "https://ec.europa.eu/safety-gate-alerts/screen/search",
                "description": "Rapid Alert System for dangerous products in EU"
            }
        ]
    },
    "United Kingdom Databases": {
        "icon": "🇬🇧",
        "description": "UK medical device databases post-Brexit",
        "links": [
            {
                "name": "MHRA Device Registration",
                "url": "https://www.gov.uk/guidance/register-as-a-manufacturer-to-sell-medical-devices",
                "description": "UK Medicines and Healthcare products Regulatory Agency registration"
            },
            {
                "name": "MHRA Yellow Card Scheme",
                "url": "https://yellowcard.mhra.gov.uk/",
                "description": "Report adverse incidents with medical devices in the UK"
            },
            {
                "name": "UK Medical Device Alerts",
                "url": "https://www.gov.uk/drug-device-alerts",
                "description": "Safety alerts for medical devices and medicines"
            },
            {
                "name": "UK Approved Bodies",
                "url": "https://www.gov.uk/government/publications/medical-devices-uk-approved-bodies",
                "description": "List of UK approved bodies for conformity assessment"
            }
        ]
    },
    "Canada Databases": {
        "icon": "🇨🇦",
        "description": "Health Canada medical device resources",
        "links": [
            {
                "name": "Health Canada Medical Devices Active Licence Listing (MDALL)",
                "url": "https://health-products.canada.ca/mdall-limh/",
                "description": "Database of licensed medical devices in Canada"
            },
            {
                "name": "Canada Recalls & Safety Alerts",
                "url": "https://recalls-rappels.canada.ca/en",
                "description": "Search recalls and safety alerts for medical devices"
            },
            {
                "name": "Canada Vigilance Adverse Reaction Database",
                "url": "https://www.canada.ca/en/health-canada/services/drugs-health-products/medeffect-canada/adverse-reaction-database.html",
                "description": "Reports of suspected adverse reactions to medical devices"
            }
        ]
    },
    "LATAM - Brazil": {
        "icon": "🇧🇷",
        "description": "ANVISA (Brazilian Health Regulatory Agency) resources",
        "links": [
            {
                "name": "ANVISA Medical Device Database",
                "url": "https://consultas.anvisa.gov.br/#/saude/",
                "description": "Search registered medical devices in Brazil"
            },
            {
                "name": "ANVISA Regulations (RDC 665/2022)",
                "url": "https://www.in.gov.br/en/web/dou/-/resolucao-rdc-n-665-de-30-de-marco-de-2022-389495735",
                "description": "Current medical device regulation in Brazil"
            },
            {
                "name": "ANVISA Alerts",
                "url": "https://www.gov.br/anvisa/pt-br/assuntos/fiscalizacao-e-monitoramento/alertas",
                "description": "Safety alerts and product recalls"
            }
        ]
    },
    "LATAM - Mexico": {
        "icon": "🇲🇽",
        "description": "COFEPRIS (Mexican health authority) resources",
        "links": [
            {
                "name": "COFEPRIS Medical Device Registry",
                "url": "https://www.gob.mx/cofepris/acciones-y-programas/registros-sanitarios-de-dispositivos-medicos",
                "description": "Medical device registration information"
            },
            {
                "name": "COFEPRIS Regulations",
                "url": "https://www.gob.mx/cofepris/documentos/marco-regulatorio-dispositivos-medicos",
                "description": "Regulatory framework for medical devices in Mexico"
            },
            {
                "name": "COFEPRIS Alerts & Withdrawals",
                "url": "https://www.gob.mx/cofepris/documentos/alertas-sanitarias",
                "description": "Health alerts and product withdrawals"
            }
        ]
    },
    "LATAM - Colombia": {
        "icon": "🇨🇴",
        "description": "INVIMA (Colombian health authority) resources",
        "links": [
            {
                "name": "INVIMA Medical Device Database",
                "url": "https://www.invima.gov.co/consultas-registros-sanitarios",
                "description": "Search registered medical devices in Colombia"
            },
            {
                "name": "INVIMA Regulations (Decreto 4725/2005)",
                "url": "https://www.invima.gov.co/normatividad-dispositivos-medicos",
                "description": "Medical device regulatory framework"
            },
            {
                "name": "INVIMA Alerts",
                "url": "https://www.invima.gov.co/alertas-sanitarias",
                "description": "Safety alerts and recalls"
            }
        ]
    },
    "LATAM - Chile": {
        "icon": "🇨🇱",
        "description": "ISP (Chilean health authority) resources",
        "links": [
            {
                "name": "ISP Medical Device Registry",
                "url": "https://registrosanitario.ispch.gob.cl/Ficha/BuscarProducto",
                "description": "Search registered medical devices in Chile"
            },
            {
                "name": "ISP Regulations",
                "url": "https://www.ispch.gob.cl/dispositivos-medicos/",
                "description": "Medical device regulatory information"
            },
            {
                "name": "ISP Alerts",
                "url": "https://www.ispch.gob.cl/alertas/",
                "description": "Safety alerts and product recalls"
            }
        ]
    },
    "LATAM - Argentina": {
        "icon": "🇦🇷",
        "description": "ANMAT (Argentine health authority) resources",
        "links": [
            {
                "name": "ANMAT Medical Device Database",
                "url": "https://servicios.pami.org.ar/vademecum/views/consultaPublica/listado.zul",
                "description": "Search registered medical devices in Argentina"
            },
            {
                "name": "ANMAT Disposición 2318/2002",
                "url": "http://www.anmat.gov.ar/webanmat/legislacion/medicamentos/Disposicion_ANMAT_2318-2002.pdf",
                "description": "Medical device registration regulation"
            },
            {
                "name": "ANMAT Alerts",
                "url": "http://www.anmat.gov.ar/comunicados/comunicados.asp",
                "description": "Health alerts and prohibitions"
            }
        ]
    },
    "Australia & Asia-Pacific": {
        "icon": "🌏",
        "description": "TGA and other Asia-Pacific regulatory resources",
        "links": [
            {
                "name": "TGA (Australia) ARTG Database",
                "url": "https://www.tga.gov.au/resources/artg",
                "description": "Australian Register of Therapeutic Goods"
            },
            {
                "name": "TGA Recalls Database",
                "url": "https://www.tga.gov.au/safety/recalls",
                "description": "Product recalls and safety alerts in Australia"
            },
            {
                "name": "PMDA (Japan) Device Database",
                "url": "https://www.pmda.go.jp/english/index.html",
                "description": "Pharmaceuticals and Medical Devices Agency of Japan"
            },
            {
                "name": "NMPA (China) Database",
                "url": "http://english.nmpa.gov.cn/",
                "description": "National Medical Products Administration of China"
            }
        ]
    },
    "International Standards": {
        "icon": "🌍",
        "description": "Global quality and regulatory standards organizations",
        "links": [
            {
                "name": "ISO Standards Catalogue",
                "url": "https://www.iso.org/standards.html",
                "description": "International Organization for Standardization - medical device standards"
            },
            {
                "name": "IEC Medical Standards",
                "url": "https://www.iec.ch/",
                "description": "International Electrotechnical Commission - electrical safety standards"
            },
            {
                "name": "IMDRF Documents",
                "url": "http://www.imdrf.org/documents/documents.asp",
                "description": "International Medical Device Regulators Forum - harmonization documents"
            },
            {
                "name": "WHO Medical Devices Portal",
                "url": "https://www.who.int/health-topics/medical-devices",
                "description": "World Health Organization medical device resources"
            },
            {
                "name": "MDSAP Program",
                "url": "http://www.imdrf.org/workitems/wi-mdsap.asp",
                "description": "Medical Device Single Audit Program for multiple countries"
            }
        ]
    }
}


def get_all_categories():
    """Get list of all resource categories"""
    return list(QUALITY_RESOURCES.keys())


def get_category_resources(category: str):
    """Get resources for a specific category"""
    return QUALITY_RESOURCES.get(category, {})


def get_total_link_count():
    """Get total number of resource links"""
    total = 0
    for category in QUALITY_RESOURCES.values():
        total += len(category.get("links", []))
    return total
