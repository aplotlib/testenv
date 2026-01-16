from __future__ import annotations

"""Regulatory data aggregation and normalization with multilingual support."""

from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import requests

from src.search.cpsc import cpsc_search
from src.search.health_agency_feeds import fetch_agency_alerts
from src.search.google_cse import google_search
from src.search.openfda import (
    search_device_enforcement,
    search_device_recall,
    search_by_product_code,
    get_product_codes_for_term,
)
from src.services.adverse_event_service import AdverseEventService
from src.services.media_service import MediaMonitoringService


def _as_date(value: Any) -> Optional[date]:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return None


def _safe_list(items: Optional[Iterable[str]]) -> List[str]:
    return [str(x).strip() for x in items or [] if str(x).strip()]


DEFAULT_LOOKBACK_YEARS = 3

# Supported languages for multi-language search
SUPPORTED_LANGUAGES = ["en", "es", "pt", "de", "fr", "it", "ja", "zh", "ko"]


class RegulatoryService:
    """
    Unified Regulatory Intelligence Engine.
    Integrates OpenFDA, CPSC, Google Custom Search (regulators), and Media.
    """

    REGIONAL_DOMAINS = {
        "US": ["fda.gov", "openfda.gov", "saferproducts.gov", "cpsc.gov", "cdc.gov"],
        "UK": ["gov.uk", "mhra.gov.uk", "nhs.uk"],
        "CA": ["canada.ca", "recalls-rappels.gc.ca", "healthcanada.gc.ca"],
        "EU": ["europa.eu", "ema.europa.eu", "ec.europa.eu", "echa.europa.eu"],
        "LATAM": ["anvisa.gov.br", "cofepris.gob.mx", "invima.gov.co", "minsalud.gov.co"],
        "APAC": ["tga.gov.au", "pmda.go.jp", "hsa.gov.sg"],
    }

    SANCTIONS_DOMAINS = [
        "treasury.gov",
        "hmt-sanctions.s3.eu-west-2.amazonaws.com",
        "ec.europa.eu",
        "un.org/securitycouncil",
    ]

    # Expanded synonym map with comprehensive medical device terminology
    SYNONYM_MAP = {
        # Blood Pressure Monitors - comprehensive coverage
        "bpm": ["blood pressure monitor", "bp monitor", "sphygmomanometer", "blood pressure machine", "blood pressure cuff"],
        "blood pressure monitor": ["bpm", "bp monitor", "blood pressure machine", "sphygmomanometer", "blood pressure cuff", "digital bp monitor", "automatic blood pressure", "wrist blood pressure", "arm blood pressure"],
        "blood pressure": ["blood pressure monitor", "bp monitor", "sphygmomanometer", "hypertension monitor"],
        "bp monitor": ["blood pressure monitor", "sphygmomanometer", "blood pressure machine"],
        "sphygmomanometer": ["blood pressure monitor", "bp monitor", "blood pressure cuff"],

        # Mobility Devices
        "scooter": ["mobility scooter", "powered scooter", "electric scooter", "power wheelchair"],
        "wheelchair": ["power wheelchair", "electric wheelchair", "manual wheelchair", "mobility chair", "transport chair"],
        "walker": ["rollator", "walking frame", "mobility walker", "rolling walker"],

        # Cardiac Devices
        "pacemaker": ["cardiac pacemaker", "implantable pacemaker", "heart pacemaker", "pulse generator"],
        "defibrillator": ["aed", "automated external defibrillator", "icd", "implantable cardioverter defibrillator", "cardiac defibrillator"],
        "aed": ["automated external defibrillator", "defibrillator", "heart defibrillator"],
        "icd": ["implantable cardioverter defibrillator", "implantable defibrillator"],

        # Infusion & Injection
        "infusion pump": ["iv pump", "syringe pump", "intravenous pump", "volumetric pump", "ambulatory pump", "pca pump"],
        "insulin pump": ["diabetes pump", "csii pump", "continuous insulin", "insulin delivery"],
        "syringe": ["syringe", "pre-filled syringe", "prefilled syringe", "hypodermic syringe", "injection syringe"],
        "syringe pump": ["infusion pump", "iv pump"],

        # Respiratory Devices
        "ventilator": ["respirator", "mechanical ventilator", "breathing machine", "life support ventilator"],
        "cpap": ["continuous positive airway pressure", "sleep apnea device", "cpap machine", "pap device"],
        "bipap": ["bilevel positive airway pressure", "bipap machine", "sleep therapy"],
        "nebulizer": ["nebuliser", "aerosol therapy", "breathing treatment", "inhalation device"],
        "oxygen concentrator": ["oxygen generator", "home oxygen", "portable oxygen", "o2 concentrator"],
        "pulse oximeter": ["oximeter", "spo2 monitor", "oxygen saturation", "finger oximeter"],

        # Monitoring Devices
        "glucometer": ["glucose meter", "blood glucose monitor", "blood sugar meter", "diabetes monitor", "cgm"],
        "glucose monitor": ["glucometer", "blood glucose meter", "continuous glucose monitor", "cgm"],
        "thermometer": ["digital thermometer", "infrared thermometer", "forehead thermometer", "ear thermometer", "clinical thermometer"],
        "ecg": ["electrocardiogram", "ekg", "heart monitor", "cardiac monitor"],
        "holter": ["holter monitor", "ambulatory ecg", "heart rhythm monitor"],

        # Catheters & Tubes
        "catheter": ["urinary catheter", "central line", "iv catheter", "foley catheter", "picc line"],
        "feeding tube": ["enteral feeding", "ng tube", "peg tube", "gastrostomy"],

        # Surgical & Sterilization
        "sterilizer": ["autoclave", "steam sterilizer", "sterilization equipment"],
        "surgical instrument": ["surgical tool", "operating instrument", "surgical device"],

        # Imaging
        "ultrasound": ["sonography", "ultrasound machine", "diagnostic ultrasound"],
        "x-ray": ["radiography", "x-ray machine", "radiographic"],
        "mri": ["magnetic resonance", "mri scanner", "mr imaging"],
        "ct scanner": ["computed tomography", "cat scan", "ct machine"],

        # Patient Care
        "hospital bed": ["medical bed", "patient bed", "adjustable bed", "electric bed"],
        "patient monitor": ["vital signs monitor", "bedside monitor", "multiparameter monitor"],
        "scale": ["medical scale", "patient scale", "weighing scale", "body weight scale"],

        # Orthopedic
        "brace": ["orthopedic brace", "support brace", "knee brace", "back brace", "ankle brace"],
        "splint": ["orthopedic splint", "immobilizer", "finger splint", "wrist splint"],
        "prosthetic": ["prosthesis", "artificial limb", "prosthetic device"],
        "implant": ["medical implant", "orthopedic implant", "surgical implant"],

        # Hearing
        "hearing aid": ["hearing device", "hearing amplifier", "cochlear implant"],

        # Dental
        "dental": ["dental device", "dental equipment", "dental instrument"],

        # Lab Equipment
        "analyzer": ["laboratory analyzer", "diagnostic analyzer", "blood analyzer"],
        "centrifuge": ["laboratory centrifuge", "blood centrifuge"],
    }

    @classmethod
    def search_all_sources(
        cls,
        query_term: str,
        regions: Optional[List[str]] = None,
        start_date: Any = None,
        end_date: Any = None,
        limit: int = 300,
        mode: str = "fast",
        ai_service: Any = None,
        manufacturer: Optional[str] = None,
        vendor_only: bool = False,
        include_sanctions: bool = True,
        extra_terms: Optional[Sequence[str]] = None,
        multilingual: bool = True,
        translate_results: bool = True,
    ) -> tuple[pd.DataFrame, dict]:
        """
        Main entry point for regulatory intelligence search.

        Args:
            query_term: Product name or search term
            regions: List of regions to search (US, EU, UK, CA, LATAM, APAC)
            start_date: Start of date range
            end_date: End of date range
            limit: Maximum results per source
            mode: 'fast' (APIs only) or 'powerful' (adds web/media/intl feeds)
            ai_service: Optional AI service for classification
            manufacturer: Optional manufacturer name filter
            vendor_only: Search only vendor-related actions
            include_sanctions: Include sanctions list search
            extra_terms: Additional search terms to include
            multilingual: Enable multi-language search expansion
            translate_results: Auto-translate foreign language results to English
        """
        results: List[Dict[str, Any]] = []
        status_log: Dict[str, int] = {}

        regions = regions or ["US", "EU", "UK", "CA", "LATAM", "APAC"]
        query_term = (query_term or "").strip()
        manufacturer = (manufacturer or "").strip()
        if not query_term and not manufacturer:
            return pd.DataFrame(), {"Error": 0}

        start_dt = _as_date(start_date) or date.today().replace(year=date.today().year - DEFAULT_LOOKBACK_YEARS)
        end_dt = _as_date(end_date) or date.today()
        if start_dt > end_dt:
            start_dt, end_dt = end_dt, start_dt
        is_powerful = mode == "powerful"
        max_terms = 20 if is_powerful else 15  # Increased for better coverage

        # Prepare search terms with synonym expansion
        terms = cls.prepare_terms(query_term, manufacturer, max_terms=max_terms, extra_terms=extra_terms)

        # Get multilingual terms if enabled
        multilingual_terms = {}
        if multilingual:
            multilingual_terms = cls._get_multilingual_terms(query_term)

        if not vendor_only:
            # Enhanced FDA search with product codes
            fda_recalls = cls._fetch_openfda_device_recalls(terms, limit * 2, start_dt, end_dt)
            results.extend(fda_recalls)
            status_log["FDA Device Recalls"] = len(fda_recalls)

            # Also search by product code for better coverage
            product_codes = get_product_codes_for_term(query_term)
            if product_codes:
                code_recalls = cls._fetch_openfda_by_product_code(product_codes, limit, start_dt, end_dt)
                # Filter out duplicates
                existing_ids = {r.get("ID") for r in results if r.get("ID")}
                new_code_recalls = [r for r in code_recalls if r.get("ID") not in existing_ids]
                results.extend(new_code_recalls)
                status_log["FDA Product Code Recalls"] = len(new_code_recalls)

            fda_enf = cls._fetch_openfda_enforcement(terms, limit, start_dt, end_dt)
            results.extend(fda_enf)
            status_log["FDA Enforcement"] = len(fda_enf)

            maude_service = AdverseEventService()
            # Search with multiple terms for MAUDE
            maude_hits = []
            for term in terms[:5]:  # Top 5 terms
                hits = maude_service.search_events(term, start_dt, end_dt, limit=20)
                for item in hits:
                    item["Matched_Term"] = term
                maude_hits.extend(hits)
            # Dedupe MAUDE by report number
            seen_reports = set()
            unique_maude = []
            for hit in maude_hits:
                report_id = hit.get("ID", "")
                if report_id not in seen_reports:
                    seen_reports.add(report_id)
                    unique_maude.append(hit)
            results.extend(unique_maude)
            status_log["FDA MAUDE"] = len(unique_maude)

            cpsc_hits = cls._fetch_cpsc(terms, start_dt, end_dt, limit=limit)
            results.extend(cpsc_hits)
            status_log["CPSC Recalls"] = len(cpsc_hits)

        if include_sanctions and manufacturer:
            sanctions_hits = cls._search_sanctions(manufacturer, limit=limit)
            results.extend(sanctions_hits)
            status_log["Sanctions & Watchlists"] = len(sanctions_hits)
            ofac_hits = cls._search_ofac(manufacturer, limit=limit)
            results.extend(ofac_hits)
            status_log["OFAC Sanctions"] = len(ofac_hits)

        if is_powerful:
            web_hits = cls._safe_regulatory_web_search(terms, regions, limit=limit)
            results.extend(web_hits)
            status_log["Regulatory Web"] = len(web_hits)

            # Enhanced global agency search with multilingual support
            agency_hits = cls._search_global_agencies(
                terms,
                regions,
                limit=limit,
                translate_results=translate_results,
                multilingual_search=multilingual
            )
            results.extend(agency_hits)
            status_log["Global Health Agencies"] = len(agency_hits)

            # Search media in multiple regions and languages
            media_hits = cls._search_media_multilingual(
                query_term or manufacturer,
                regions,
                multilingual_terms if multilingual else {}
            )
            results.extend(media_hits)
            status_log["Media Signals"] = len(media_hits)

        df = pd.DataFrame(results)
        if df.empty:
            return df, status_log

        df = cls._dedupe(df)
        df = cls._normalize_columns(df)
        df.sort_values(by="Date", ascending=False, inplace=True, ignore_index=True)
        return df, status_log

    @classmethod
    def _get_multilingual_terms(cls, query_term: str) -> Dict[str, List[str]]:
        """Get search terms translated to multiple languages."""
        try:
            from src.services.translation_service import get_multilingual_terms
            return get_multilingual_terms(query_term)
        except ImportError:
            return {"en": [query_term]}

    @classmethod
    def _fetch_openfda_by_product_code(
        cls,
        product_codes: List[str],
        limit: int,
        start: date,
        end: date
    ) -> List[Dict[str, Any]]:
        """Fetch FDA recalls by product code for more comprehensive results."""
        results: List[Dict[str, Any]] = []
        hits = search_by_product_code(product_codes, start, end, limit=limit)
        for hit in hits:
            recall_number = hit.get("recall_number", "")
            classification = hit.get("classification", "")
            record = {
                "Source": "FDA Device Recall (Product Code)",
                "Date": hit.get("report_date", ""),
                "Product": hit.get("product_description", ""),
                "Description": hit.get("product_description", ""),
                "Reason": hit.get("reason_for_recall", ""),
                "Firm": hit.get("recalling_firm", ""),
                "Model Info": hit.get("model_number", "") or hit.get("code_info", ""),
                "ID": recall_number,
                "Link": cls._openfda_link("recall", recall_number),
                "Status": hit.get("status", ""),
                "Risk_Level": cls._risk_from_classification(classification),
                "Matched_Term": f"Product Code: {hit.get('product_code', '')}",
            }
            results.append(record)
        return results

    @classmethod
    def _search_media_multilingual(
        cls,
        query_term: str,
        regions: Sequence[str],
        multilingual_terms: Dict[str, List[str]]
    ) -> List[Dict[str, Any]]:
        """Search media with multilingual query expansion."""
        if not query_term:
            return []
        media_svc = MediaMonitoringService()
        results: List[Dict[str, Any]] = []

        # Map regions to languages for media search
        region_languages = {
            "US": "en", "UK": "en", "CA": "en",
            "EU": "de",  # Try German for EU
            "LATAM": "es",  # Spanish for LATAM
            "APAC": "en",
        }

        for region in regions:
            # Search with English term
            results.extend(media_svc.search_media(query_term, limit=10, region=region))

            # Search with localized terms if available
            lang = region_languages.get(region, "en")
            if lang != "en" and lang in multilingual_terms:
                for local_term in multilingual_terms[lang][:2]:  # Top 2 local terms
                    hits = media_svc.search_media(local_term, limit=5, region=region)
                    for hit in hits:
                        hit["Matched_Term"] = f"{local_term} ({lang})"
                    results.extend(hits)

        return results

    @classmethod
    def search_all_sources_safe(cls, **kwargs: Any) -> tuple[pd.DataFrame, dict]:
        try:
            return cls.search_all_sources(**kwargs)
        except TypeError as exc:
            if "extra_terms" in str(exc):
                kwargs.pop("extra_terms", None)
                return cls.search_all_sources(**kwargs)
            raise

    @classmethod
    def prepare_terms(
        cls,
        query_term: str,
        manufacturer: str,
        max_terms: int = 8,
        extra_terms: Optional[Sequence[str]] = None,
    ) -> List[str]:
        base_terms = []
        if query_term:
            base_terms.append(query_term)
        if manufacturer:
            base_terms.append(manufacturer)
        if query_term and manufacturer:
            base_terms.append(f"{manufacturer} {query_term}")
            base_terms.append(f"{manufacturer} {query_term} recall")
        if query_term and extra_terms:
            for keyword in _safe_list(extra_terms):
                base_terms.append(f"{query_term} {keyword}")

        expanded: List[str] = []
        for term in base_terms:
            expanded.extend(cls._expand_terms(term))

        deduped = cls._dedupe_terms(expanded)
        return deduped[:max_terms]

    @classmethod
    def _expand_terms(cls, term: str) -> List[str]:
        if not term:
            return []
        normalized = term.strip()
        lower_term = normalized.lower()
        expanded = {normalized}

        if lower_term in cls.SYNONYM_MAP:
            expanded.update(cls.SYNONYM_MAP[lower_term])

        for key, synonyms in cls.SYNONYM_MAP.items():
            if key in lower_term:
                expanded.update(synonyms)

        if "-" in normalized:
            expanded.add(normalized.replace("-", " "))
        if " " in normalized:
            expanded.add(normalized.replace(" ", "-"))

        return [t for t in expanded if t]

    @staticmethod
    def _dedupe_terms(terms: Sequence[str]) -> List[str]:
        seen = set()
        out = []
        for term in terms:
            key = term.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(term.strip())
        return out

    @classmethod
    def _fetch_openfda_device_recalls(cls, terms: Sequence[str], limit: int, start: date, end: date) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for term in terms:
            hits = search_device_recall(term, start, end, limit=limit)
            for hit in hits:
                recall_number = hit.get("recall_number", "")
                reason = hit.get("reason_for_recall", "")
                classification = hit.get("classification", "")
                record = {
                    "Source": "FDA Device Recall",
                    "Date": hit.get("report_date", ""),
                    "Product": hit.get("product_description", ""),
                    "Description": hit.get("product_description", ""),
                    "Reason": reason,
                    "Firm": hit.get("recalling_firm", ""),
                    "Model Info": hit.get("model_number", "") or hit.get("code_info", ""),
                    "ID": recall_number,
                    "Link": cls._openfda_link("recall", recall_number),
                    "Status": hit.get("status", ""),
                    "Risk_Level": cls._risk_from_classification(classification),
                    "Matched_Term": term,
                }
                results.append(record)
            if len(results) >= limit:
                break
        return results

    @classmethod
    def _fetch_openfda_enforcement(cls, terms: Sequence[str], limit: int, start: date, end: date) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for term in terms:
            hits = search_device_enforcement(term, start, end, limit=limit)
            for hit in hits:
                recall_number = hit.get("recall_number", "")
                reason = hit.get("reason_for_recall", "")
                classification = hit.get("classification", "")
                record = {
                    "Source": "FDA Enforcement",
                    "Date": hit.get("report_date", ""),
                    "Product": hit.get("product_description", ""),
                    "Description": hit.get("product_description", ""),
                    "Reason": reason,
                    "Firm": hit.get("recalling_firm", ""),
                    "Model Info": hit.get("model_number", "") or hit.get("code_info", ""),
                    "ID": recall_number,
                    "Link": cls._openfda_link("enforcement", recall_number),
                    "Status": hit.get("status", ""),
                    "Risk_Level": cls._risk_from_classification(classification),
                    "Matched_Term": term,
                }
                results.append(record)
            if len(results) >= limit:
                break
        return results

    @classmethod
    def _fetch_cpsc(cls, terms: Sequence[str], start: date, end: date, limit: int = 100) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for term in terms:
            hits = cpsc_search(term, start, end, limit=limit)
            for hit in hits:
                record = {
                    "Source": "CPSC Recall",
                    "Date": hit.get("RecallDate", ""),
                    "Product": hit.get("Title", ""),
                    "Description": hit.get("Title", ""),
                    "Reason": hit.get("Description", ""),
                    "Firm": hit.get("Manufacturer", ""),
                    "Model Info": hit.get("ProductID", ""),
                    "ID": hit.get("RecallID", ""),
                    "Link": hit.get("URL", ""),
                    "Status": hit.get("Status", ""),
                    "Risk_Level": "Medium",
                    "Matched_Term": term,
                }
                results.append(record)
            if len(results) >= limit:
                break
        return results

    @classmethod
    def _search_sanctions(cls, manufacturer: str, limit: int = 50) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for domain in cls.SANCTIONS_DOMAINS:
            results.extend(cls._google_search(f'"{manufacturer}" site:{domain}', category="Sanctions", num=limit))
        return results[:limit]

    @classmethod
    def _search_ofac(cls, manufacturer: str, limit: int = 50) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        url = "https://www.treasury.gov/ofac/downloads/sdn.csv"
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            lines = response.text.splitlines()
        except Exception:
            return results

        for line in lines[1:]:
            fields = line.split(",")
            name = fields[1] if len(fields) > 1 else ""
            if manufacturer.lower() in name.lower():
                link = "https://www.treasury.gov/ofac/downloads/sdn.csv"
                results.append(
                    {
                        "Source": "OFAC Sanctions",
                        "Date": "",
                        "Product": name,
                        "Description": name,
                        "Reason": "OFAC sanctions list match",
                        "Firm": name,
                        "Model Info": "",
                        "ID": f"OFAC:{name}",
                        "Link": link,
                        "Status": "Listed",
                        "Risk_Level": "High",
                        "Matched_Term": manufacturer,
                    }
                )
            if len(results) >= limit:
                break
        return results

    @classmethod
    def _search_media(cls, query_term: str, regions: Sequence[str]) -> List[Dict[str, Any]]:
        if not query_term:
            return []
        media_svc = MediaMonitoringService()
        results: List[Dict[str, Any]] = []
        for region in regions:
            results.extend(media_svc.search_media(query_term, limit=10, region=region))
        return results

    @classmethod
    def _safe_regulatory_web_search(
        cls,
        terms: Sequence[str],
        regions: Sequence[str],
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        search_fn = getattr(cls, "_search_regulatory_web", None)
        if not callable(search_fn):
            return []
        try:
            return search_fn(terms, regions, limit=limit)
        except AttributeError:
            return []

    @classmethod
    def _search_regulatory_web(
        cls,
        terms: Sequence[str],
        regions: Sequence[str],
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        if not terms:
            return []
        results: List[Dict[str, Any]] = []
        query_specs: List[tuple[str, str]] = []
        for term in terms:
            for region in regions:
                for domain in cls.REGIONAL_DOMAINS.get(region, []):
                    query_specs.append((f"\"{term}\" site:{domain}", region))

        for query, region in query_specs:
            remaining = limit - len(results)
            if remaining <= 0:
                break
            per_query_limit = min(10, remaining)
            results.extend(
                cls._google_search(
                    query,
                    category=f"Regulatory Web ({region})",
                    num=per_query_limit,
                )
            )
        return results[:limit]

    @classmethod
    def _search_global_agencies(
        cls,
        terms: Sequence[str],
        regions: Sequence[str],
        limit: int = 50,
        translate_results: bool = True,
        multilingual_search: bool = True,
    ) -> List[Dict[str, Any]]:
        if not terms:
            return []
        return fetch_agency_alerts(
            terms,
            regions,
            limit=limit,
            translate_results=translate_results,
            multilingual_search=multilingual_search
        )

    @staticmethod
    def _google_search(query: str, category: str = "Web Search", num: int = 10) -> List[Dict[str, Any]]:
        hits = google_search(query, num=min(max(num, 1), 10), pages=2)
        return RegulatoryService._google_hits_to_records(hits, category, query)

    @staticmethod
    def _google_hits_to_records(
        hits: Sequence[Dict[str, Any]],
        source_label: str,
        matched_term: str,
    ) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for item in hits:
            link = item.get("link", "")
            snippet = item.get("snippet") or ""
            title = item.get("title") or snippet
            risk_level = RegulatoryService._risk_from_keywords(f"{title} {snippet}")
            records.append(
                {
                    "Source": source_label,
                    "Date": "",
                    "Product": title,
                    "Description": title,
                    "Reason": snippet,
                    "Firm": item.get("displayLink") or "",
                    "Model Info": "",
                    "ID": link,
                    "Link": link,
                    "Status": "Published",
                    "Risk_Level": risk_level,
                    "Matched_Term": matched_term,
                }
            )
        return records

    @staticmethod
    def _risk_from_keywords(text: str) -> str:
        if not text:
            return "Medium"
        normalized = text.lower()
        high_risk_terms = [
            "class i",
            "class ii",
            "class iii",
            "recall",
            "safety alert",
            "field safety",
            "warning",
            "urgent",
        ]
        if any(term in normalized for term in high_risk_terms):
            return "High"
        return "Medium"

    @staticmethod
    def _risk_from_classification(classification: str) -> str:
        normalized = classification.strip().upper().replace("CLASS ", "")
        if normalized == "I":
            return "High"
        if normalized == "II":
            return "Medium"
        if normalized == "III":
            return "Low"
        return "Medium"

    @staticmethod
    def _openfda_link(category: str, recall_number: str) -> str:
        if not recall_number:
            return ""
        endpoint = "device/recall" if category == "recall" else "device/enforcement"
        return f"https://api.fda.gov/{endpoint}.json?search=recall_number:{recall_number}"

    @staticmethod
    def _dedupe(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        id_series = df["ID"].fillna("").astype(str) if "ID" in df.columns else pd.Series("", index=df.index)
        link_series = df["Link"].fillna("").astype(str) if "Link" in df.columns else pd.Series("", index=df.index)
        product_series = df["Product"].fillna("").astype(str) if "Product" in df.columns else pd.Series("", index=df.index)

        dedupe_key = id_series.where(id_series != "", link_series.where(link_series != "", product_series))
        df["__dedupe_key"] = dedupe_key
        df = df.drop_duplicates(subset=["__dedupe_key"])
        df.drop(columns="__dedupe_key", inplace=True)
        return df

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "Manufacturer" not in df.columns and "Firm" in df.columns:
            df["Manufacturer"] = df["Firm"]
        if "Model Info" not in df.columns and "Model_Numbers" in df.columns:
            df["Model Info"] = df["Model_Numbers"]
        for col in ["Date", "Product", "Reason", "Firm", "Link"]:
            if col not in df.columns:
                df[col] = ""
        return df
