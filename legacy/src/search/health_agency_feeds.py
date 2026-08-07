from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Iterable, List, Optional, Dict
from urllib.parse import urlparse, urlunparse
import xml.etree.ElementTree as ET

import requests


@dataclass(frozen=True)
class AgencyFeed:
    name: str
    region: str
    url: str
    language: str = "en"  # ISO 639-1 code
    feed_type: str = "rss"  # rss, atom, or json


# Comprehensive international regulatory feeds
FEEDS: List[AgencyFeed] = [
    # === UNITED STATES ===
    AgencyFeed("FDA MedWatch Safety", "US", "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/medwatch/rss.xml", "en"),
    AgencyFeed("FDA Medical Device Recalls", "US", "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/medical-devices/rss.xml", "en"),

    # === UNITED KINGDOM ===
    AgencyFeed("UK MHRA Alerts", "UK", "https://www.gov.uk/drug-device-alerts.atom", "en", "atom"),
    AgencyFeed("UK MHRA Medical Device Alerts", "UK", "https://www.gov.uk/government/publications.atom?departments%5B%5D=medicines-and-healthcare-products-regulatory-agency&publication_filter_option=alerts-and-recalls", "en", "atom"),

    # === EUROPEAN UNION ===
    AgencyFeed("EU EMA News", "EU", "https://www.ema.europa.eu/en/rss.xml", "en"),
    AgencyFeed("EU Safety Gate (RAPEX)", "EU", "https://ec.europa.eu/consumers/consumers_safety/safety_products/rapex/alerts/repository/content/blobs/latestweeklyoverview.xml", "en"),

    # === GERMANY ===
    AgencyFeed("Germany BfArM Safety", "EU", "https://www.bfarm.de/SiteGlobals/Functions/RSSFeed/DE/rss_Kundeninfo.xml", "de"),

    # === FRANCE ===
    AgencyFeed("France ANSM Alerts", "EU", "https://ansm.sante.fr/flux-rss", "fr"),

    # === CANADA ===
    AgencyFeed("Canada Health Recalls EN", "CA", "https://recalls-rappels.canada.ca/en/feed/rss2.xml", "en"),
    AgencyFeed("Canada Health Recalls FR", "CA", "https://recalls-rappels.canada.ca/fr/flux/rss2.xml", "fr"),
    AgencyFeed("Health Canada Advisories", "CA", "https://www.canada.ca/content/canadasite/en/health-canada/services/drugs-health-products/medeffect-canada.atom", "en", "atom"),

    # === AUSTRALIA ===
    AgencyFeed("Australia TGA Alerts", "APAC", "https://www.tga.gov.au/news/rss.xml", "en"),
    AgencyFeed("Australia TGA Safety", "APAC", "https://www.tga.gov.au/ws/tga/rss/medicine-safety-updates/all", "en"),
    AgencyFeed("Australia Product Recalls", "APAC", "https://www.productsafety.gov.au/rss/recalls", "en"),

    # === JAPAN ===
    AgencyFeed("Japan PMDA Recalls", "APAC", "https://www.pmda.go.jp/rss/recall.xml", "ja"),
    AgencyFeed("Japan PMDA Safety", "APAC", "https://www.pmda.go.jp/rss/safety.xml", "ja"),

    # === SINGAPORE ===
    AgencyFeed("Singapore HSA Alerts", "APAC", "https://www.hsa.gov.sg/announcements/-/media/HSA/announcements/rss.xml", "en"),

    # === LATIN AMERICA ===
    AgencyFeed("Brazil ANVISA Alerts", "LATAM", "https://www.gov.br/anvisa/pt-br/assuntos/fiscalizacao-e-monitoramento/monitoramento/alertas/alertas-de-seguranca/rss", "pt"),
    AgencyFeed("Mexico COFEPRIS", "LATAM", "https://www.gob.mx/cofepris/archivo/prensa.rss", "es"),

    # === INTERNATIONAL ===
    AgencyFeed("WHO Medical Device Alerts", "GLOBAL", "https://www.who.int/feeds/entity/medical_devices/en/rss.xml", "en"),
    AgencyFeed("IMDRF News", "GLOBAL", "https://www.imdrf.org/rss.xml", "en"),
]

# Language-specific search term mappings for international feeds
LANGUAGE_SEARCH_TERMS: Dict[str, Dict[str, List[str]]] = {
    "recall": {
        "en": ["recall", "withdrawal", "alert", "safety"],
        "de": ["rückruf", "warnung", "sicherheit", "zurückziehung"],
        "fr": ["rappel", "retrait", "alerte", "sécurité"],
        "es": ["retiro", "retirada", "alerta", "seguridad"],
        "pt": ["recall", "recolhimento", "alerta", "segurança"],
        "ja": ["リコール", "回収", "警告", "安全"],
        "zh": ["召回", "撤回", "警告", "安全"],
    },
    "blood pressure": {
        "en": ["blood pressure", "bp monitor", "sphygmomanometer"],
        "de": ["blutdruck", "blutdruckmessgerät"],
        "fr": ["tension artérielle", "tensiomètre"],
        "es": ["presión arterial", "tensiómetro"],
        "pt": ["pressão arterial", "esfigmomanômetro"],
        "ja": ["血圧", "血圧計"],
        "zh": ["血压", "血压计"],
    },
}


def fetch_agency_alerts(
    terms: Iterable[str],
    regions: Iterable[str],
    limit: int = 50,
    translate_results: bool = True,
    multilingual_search: bool = True
) -> List[dict]:
    """
    Fetch alerts from international health agencies.

    Args:
        terms: Search terms (in English)
        regions: Region codes to search (US, UK, EU, CA, LATAM, APAC, GLOBAL)
        limit: Maximum results to return
        translate_results: Auto-translate non-English results to English
        multilingual_search: Expand search terms to other languages
    """
    selected_regions = {r.upper() for r in regions}
    # Add GLOBAL to all searches since WHO/IMDRF are relevant everywhere
    if any(r in selected_regions for r in ["US", "UK", "EU", "CA", "LATAM", "APAC"]):
        selected_regions.add("GLOBAL")

    normalized_terms = _normalize_terms(terms)
    if not normalized_terms:
        return []

    # Expand terms to multiple languages if enabled
    search_terms_by_lang = _expand_terms_multilingual(normalized_terms) if multilingual_search else {"en": normalized_terms}

    results: List[dict] = []
    seen_links: set[str] = set()

    for feed in FEEDS:
        if feed.region.upper() not in selected_regions:
            continue
        remaining = limit - len(results)
        if remaining <= 0:
            break

        # Get search terms for this feed's language
        feed_search_terms = search_terms_by_lang.get(feed.language, normalized_terms)
        # Also include English terms as fallback
        if feed.language != "en":
            feed_search_terms = list(set(feed_search_terms + normalized_terms))

        items = _fetch_feed(feed)
        for item in items:
            if len(results) >= limit:
                break
            if not _matches_terms_multilingual(item, feed_search_terms, feed.language):
                continue
            link = item.link or ""
            normalized_link = _normalize_link(link)
            if normalized_link in seen_links:
                continue
            seen_links.add(normalized_link)

            # Translate non-English content if enabled
            title = item.title or ""
            summary = item.summary or ""
            original_lang = feed.language

            if translate_results and feed.language != "en":
                title, summary = _translate_content(title, summary, feed.language)

            results.append(
                {
                    "Source": feed.name,
                    "Date": item.date_str,
                    "Product": title,
                    "Description": title,
                    "Reason": summary,
                    "Firm": feed.name,
                    "Model Info": "",
                    "ID": normalized_link or title or summary or "",
                    "Link": link,
                    "Status": "Published",
                    "Risk_Level": "High" if _looks_high_risk(item.title, item.summary) else "Medium",
                    "Matched_Term": item.matched_term,
                    "Original_Language": original_lang,
                }
            )
    return results


def _expand_terms_multilingual(terms: List[str]) -> Dict[str, List[str]]:
    """Expand English search terms to equivalent terms in other languages."""
    result: Dict[str, List[str]] = {"en": terms.copy()}

    # Try to import translation service
    try:
        from src.services.translation_service import get_multilingual_terms
        for term in terms:
            translations = get_multilingual_terms(term)
            for lang, translated_terms in translations.items():
                if lang not in result:
                    result[lang] = []
                result[lang].extend(translated_terms)
    except ImportError:
        # Fallback to built-in mappings
        for term in terms:
            for base_term, lang_dict in LANGUAGE_SEARCH_TERMS.items():
                if base_term in term.lower():
                    for lang, lang_terms in lang_dict.items():
                        if lang not in result:
                            result[lang] = []
                        result[lang].extend(lang_terms)

    # Deduplicate each language's terms
    for lang in result:
        result[lang] = list(set(result[lang]))

    return result


def _matches_terms_multilingual(item: "FeedItem", terms: List[str], language: str) -> bool:
    """Check if item matches any search terms, considering language."""
    haystack = f"{item.title} {item.summary}".lower()

    for term in terms:
        term_lower = term.lower()
        if term_lower in haystack:
            item.matched_term = term
            return True

    return False


def _translate_content(title: str, summary: str, source_lang: str) -> tuple:
    """Translate title and summary to English using translation service."""
    try:
        from src.services.translation_service import translate_to_english
        translated_title, _ = translate_to_english(title, source_lang)
        translated_summary, _ = translate_to_english(summary, source_lang)
        return translated_title, translated_summary
    except ImportError:
        return title, summary  # Return originals if translation unavailable


def _normalize_terms(terms: Iterable[str]) -> List[str]:
    normalized = []
    for term in terms:
        cleaned = (term or "").strip().lower()
        if cleaned and cleaned not in normalized:
            normalized.append(cleaned)
    return normalized


def _matches_terms(item: "FeedItem", terms: List[str]) -> bool:
    haystack = f"{item.title} {item.summary}".lower()
    for term in terms:
        if term in haystack:
            item.matched_term = term
            return True
    return False


def _looks_high_risk(title: str, summary: str) -> bool:
    text = f"{title} {summary}".lower()
    keywords = ["recall", "safety", "alert", "warning", "field safety", "withdrawal", "urgent", "class i"]
    return any(keyword in text for keyword in keywords)


def _fetch_feed(feed: AgencyFeed) -> List["FeedItem"]:
    try:
        response = requests.get(feed.url, timeout=12)
        response.raise_for_status()
    except requests.RequestException:
        return []

    content = response.content.strip()
    if not content.startswith(b"<"):
        return []

    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return []

    items: List[FeedItem] = []
    if root.tag.endswith("feed"):
        items.extend(_parse_atom_feed(root))
    else:
        items.extend(_parse_rss_feed(root))
    return items


class FeedItem:
    def __init__(self, title: str, link: str, summary: str, date_str: str):
        self.title = title
        self.link = link
        self.summary = summary
        self.date_str = date_str
        self.matched_term = ""


def _parse_atom_feed(root: ET.Element) -> List[FeedItem]:
    items: List[FeedItem] = []
    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        title = _text(entry.find("{http://www.w3.org/2005/Atom}title"))
        link = _atom_link(entry)
        summary = _text(entry.find("{http://www.w3.org/2005/Atom}summary")) or _text(
            entry.find("{http://www.w3.org/2005/Atom}content")
        )
        date_str = _format_date(_text(entry.find("{http://www.w3.org/2005/Atom}updated")))
        items.append(FeedItem(title, link, summary, date_str))
    return items


def _parse_rss_feed(root: ET.Element) -> List[FeedItem]:
    items: List[FeedItem] = []
    for item in root.findall(".//item"):
        title = _text(item.find("title"))
        link = _text(item.find("link"))
        summary = _text(item.find("description"))
        date_str = _format_date(_text(item.find("pubDate")))
        items.append(FeedItem(title, link, summary, date_str))
    return items


def _atom_link(entry: ET.Element) -> str:
    link_elem = entry.find("{http://www.w3.org/2005/Atom}link")
    if link_elem is None:
        return ""
    return link_elem.attrib.get("href", "")


def _text(element: Optional[ET.Element]) -> str:
    if element is None or element.text is None:
        return ""
    return element.text.strip()


def _format_date(value: str) -> str:
    if not value:
        return ""
    try:
        parsed = parsedate_to_datetime(value)
        return parsed.strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        pass
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return value


def _normalize_link(link: str) -> str:
    if not link:
        return ""
    parsed = urlparse(link)
    clean = parsed._replace(fragment="", query="")
    return urlunparse(clean).rstrip("/")
