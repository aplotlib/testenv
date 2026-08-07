# src/services/translation_service.py
"""
Multi-language translation service for international regulatory intelligence.
Uses free translation APIs with fallback support.
"""
from __future__ import annotations
import requests
from typing import Dict, List, Optional, Tuple
import re
from functools import lru_cache


# Medical device term translations for international searches
MULTILINGUAL_TERMS: Dict[str, Dict[str, List[str]]] = {
    # Blood Pressure Monitors
    "blood pressure monitor": {
        "en": ["blood pressure monitor", "bp monitor", "sphygmomanometer", "blood pressure machine"],
        "es": ["monitor de presión arterial", "tensiómetro", "esfigmomanómetro", "medidor de presión"],
        "pt": ["monitor de pressão arterial", "esfigmomanômetro", "aparelho de pressão", "medidor de pressão"],
        "de": ["blutdruckmessgerät", "blutdruckmonitor", "sphygmomanometer", "blutdruckmesser"],
        "fr": ["tensiomètre", "moniteur de tension artérielle", "sphygmomanomètre", "appareil de tension"],
        "it": ["misuratore di pressione", "sfigmomanometro", "monitor pressione sanguigna"],
        "ja": ["血圧計", "血圧モニター", "電子血圧計"],
        "zh": ["血压计", "血压监测仪", "电子血压计"],
        "ko": ["혈압계", "혈압 모니터"],
    },
    # Infusion Pumps
    "infusion pump": {
        "en": ["infusion pump", "iv pump", "syringe pump", "intravenous pump"],
        "es": ["bomba de infusión", "bomba intravenosa", "bomba de jeringa"],
        "pt": ["bomba de infusão", "bomba intravenosa", "bomba de seringa"],
        "de": ["infusionspumpe", "spritzenpumpe", "iv-pumpe"],
        "fr": ["pompe à perfusion", "pompe intraveineuse", "pousse-seringue"],
        "it": ["pompa per infusione", "pompa a siringa"],
        "ja": ["輸液ポンプ", "注入ポンプ", "シリンジポンプ"],
        "zh": ["输液泵", "注射泵"],
        "ko": ["주입 펌프", "수액 펌프"],
    },
    # Pacemakers
    "pacemaker": {
        "en": ["pacemaker", "cardiac pacemaker", "implantable pacemaker"],
        "es": ["marcapasos", "marcapasos cardíaco", "marcapasos implantable"],
        "pt": ["marcapasso", "marca-passo cardíaco", "pacemaker implantável"],
        "de": ["herzschrittmacher", "schrittmacher", "implantierbarer schrittmacher"],
        "fr": ["stimulateur cardiaque", "pacemaker", "stimulateur implantable"],
        "it": ["pacemaker", "stimolatore cardiaco"],
        "ja": ["ペースメーカー", "心臓ペースメーカー"],
        "zh": ["心脏起搏器", "起搏器"],
        "ko": ["심장 박동기", "페이스메이커"],
    },
    # Defibrillators
    "defibrillator": {
        "en": ["defibrillator", "aed", "automated external defibrillator", "icd"],
        "es": ["desfibrilador", "dea", "desfibrilador automático externo"],
        "pt": ["desfibrilador", "dea", "desfibrilador externo automático"],
        "de": ["defibrillator", "aed", "automatischer externer defibrillator"],
        "fr": ["défibrillateur", "dae", "défibrillateur automatisé externe"],
        "it": ["defibrillatore", "dae", "defibrillatore automatico esterno"],
        "ja": ["除細動器", "AED", "自動体外式除細動器"],
        "zh": ["除颤器", "AED", "自动体外除颤器"],
        "ko": ["제세동기", "AED", "자동 심장충격기"],
    },
    # Ventilators
    "ventilator": {
        "en": ["ventilator", "respirator", "mechanical ventilator", "breathing machine"],
        "es": ["ventilador", "respirador", "ventilador mecánico"],
        "pt": ["ventilador", "respirador", "ventilador mecânico"],
        "de": ["beatmungsgerät", "respirator", "beatmungsmaschine"],
        "fr": ["ventilateur", "respirateur", "ventilateur mécanique"],
        "it": ["ventilatore", "respiratore"],
        "ja": ["人工呼吸器", "ベンチレーター"],
        "zh": ["呼吸机", "通气机"],
        "ko": ["인공호흡기", "환기기"],
    },
    # Wheelchairs
    "wheelchair": {
        "en": ["wheelchair", "mobility scooter", "powered wheelchair", "electric wheelchair"],
        "es": ["silla de ruedas", "scooter de movilidad", "silla eléctrica"],
        "pt": ["cadeira de rodas", "scooter de mobilidade", "cadeira motorizada"],
        "de": ["rollstuhl", "elektrorollstuhl", "mobilitätsroller"],
        "fr": ["fauteuil roulant", "scooter de mobilité", "fauteuil électrique"],
        "it": ["sedia a rotelle", "carrozzina elettrica"],
        "ja": ["車椅子", "電動車椅子", "モビリティスクーター"],
        "zh": ["轮椅", "电动轮椅"],
        "ko": ["휠체어", "전동휠체어"],
    },
    # Glucose Monitors
    "glucometer": {
        "en": ["glucometer", "glucose meter", "blood glucose monitor", "glucose monitor"],
        "es": ["glucómetro", "medidor de glucosa", "monitor de glucosa"],
        "pt": ["glicosímetro", "medidor de glicose", "monitor de glicemia"],
        "de": ["blutzuckermessgerät", "glukometer", "blutzuckermesser"],
        "fr": ["glucomètre", "lecteur de glycémie", "moniteur de glucose"],
        "it": ["glucometro", "misuratore di glicemia"],
        "ja": ["血糖値計", "グルコメーター"],
        "zh": ["血糖仪", "血糖监测仪"],
        "ko": ["혈당계", "혈당측정기"],
    },
    # Pulse Oximeters
    "pulse oximeter": {
        "en": ["pulse oximeter", "oxygen saturation monitor", "spo2 monitor"],
        "es": ["oxímetro de pulso", "pulsioxímetro", "saturómetro"],
        "pt": ["oxímetro de pulso", "oxímetro", "saturímetro"],
        "de": ["pulsoximeter", "sauerstoffsättigungsmesser"],
        "fr": ["oxymètre de pouls", "saturomètre"],
        "it": ["pulsossimetro", "ossimetro"],
        "ja": ["パルスオキシメーター", "酸素濃度計"],
        "zh": ["脉搏血氧仪", "血氧仪"],
        "ko": ["맥박 산소 측정기", "산소포화도측정기"],
    },
    # CPAP/BiPAP
    "cpap": {
        "en": ["cpap", "continuous positive airway pressure", "sleep apnea device", "bipap"],
        "es": ["cpap", "dispositivo de apnea del sueño", "bipap"],
        "pt": ["cpap", "aparelho de apneia", "bipap"],
        "de": ["cpap", "cpap-gerät", "schlafapnoe-gerät", "bipap"],
        "fr": ["cpap", "ppc", "appareil apnée du sommeil", "bipap"],
        "it": ["cpap", "dispositivo apnea notturna"],
        "ja": ["CPAP", "シーパップ", "睡眠時無呼吸治療器"],
        "zh": ["CPAP", "睡眠呼吸机", "呼吸机"],
        "ko": ["CPAP", "수면무호흡치료기"],
    },
    # Thermometers
    "thermometer": {
        "en": ["thermometer", "digital thermometer", "infrared thermometer", "forehead thermometer"],
        "es": ["termómetro", "termómetro digital", "termómetro infrarrojo"],
        "pt": ["termômetro", "termômetro digital", "termômetro infravermelho"],
        "de": ["thermometer", "fieberthermometer", "infrarot-thermometer"],
        "fr": ["thermomètre", "thermomètre numérique", "thermomètre infrarouge"],
        "it": ["termometro", "termometro digitale", "termometro a infrarossi"],
        "ja": ["体温計", "電子体温計", "非接触体温計"],
        "zh": ["体温计", "电子体温计", "红外体温计"],
        "ko": ["체온계", "전자체온계"],
    },
    # Generic recall terms
    "recall": {
        "en": ["recall", "safety alert", "warning", "withdrawal"],
        "es": ["retiro", "retirada", "alerta de seguridad", "advertencia", "aviso de seguridad"],
        "pt": ["recall", "recolhimento", "alerta de segurança", "aviso"],
        "de": ["rückruf", "sicherheitswarnung", "warnung", "rückrufaktion"],
        "fr": ["rappel", "retrait", "alerte de sécurité", "avertissement"],
        "it": ["richiamo", "ritiro", "avviso di sicurezza"],
        "ja": ["リコール", "回収", "安全警告"],
        "zh": ["召回", "安全警告", "撤回"],
        "ko": ["리콜", "회수", "안전 경고"],
    },
    # Medical device generic terms
    "medical device": {
        "en": ["medical device", "medical equipment", "healthcare device"],
        "es": ["dispositivo médico", "equipo médico", "aparato médico"],
        "pt": ["dispositivo médico", "equipamento médico"],
        "de": ["medizinprodukt", "medizingerät", "medizintechnik"],
        "fr": ["dispositif médical", "équipement médical", "appareil médical"],
        "it": ["dispositivo medico", "apparecchio medico"],
        "ja": ["医療機器", "医療器具"],
        "zh": ["医疗器械", "医疗设备"],
        "ko": ["의료기기", "의료장비"],
    },
}

# Language detection patterns
LANGUAGE_PATTERNS = {
    "ja": re.compile(r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]'),  # Japanese
    "zh": re.compile(r'[\u4e00-\u9fff]'),  # Chinese
    "ko": re.compile(r'[\uac00-\ud7af]'),  # Korean
    "de": re.compile(r'\b(der|die|das|und|ist|ein|eine|für|mit|auf|bei)\b', re.I),
    "fr": re.compile(r'\b(le|la|les|de|du|des|et|est|un|une|pour|avec|sur)\b', re.I),
    "es": re.compile(r'\b(el|la|los|las|de|del|y|es|un|una|para|con)\b', re.I),
    "pt": re.compile(r'\b(o|a|os|as|de|do|da|e|é|um|uma|para|com)\b', re.I),
    "it": re.compile(r'\b(il|la|lo|gli|le|di|del|e|è|un|una|per|con)\b', re.I),
}


def detect_language(text: str) -> str:
    """Detect the language of text. Returns ISO 639-1 code."""
    if not text:
        return "en"

    # Check for CJK first (non-Latin scripts)
    if LANGUAGE_PATTERNS["ja"].search(text) and (
        re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text)  # Hiragana/Katakana
    ):
        return "ja"
    if LANGUAGE_PATTERNS["ko"].search(text):
        return "ko"
    if LANGUAGE_PATTERNS["zh"].search(text):
        return "zh"

    # Count pattern matches for European languages
    scores = {}
    for lang, pattern in LANGUAGE_PATTERNS.items():
        if lang in ("ja", "zh", "ko"):
            continue
        matches = pattern.findall(text.lower())
        scores[lang] = len(matches)

    if scores:
        best_lang = max(scores, key=scores.get)
        if scores[best_lang] > 2:  # Require at least 3 matches
            return best_lang

    return "en"  # Default


def get_multilingual_terms(base_term: str, languages: Optional[List[str]] = None) -> Dict[str, List[str]]:
    """
    Get translations of a search term in multiple languages.

    Args:
        base_term: The base search term in English
        languages: List of ISO 639-1 language codes. None for all languages.

    Returns:
        Dictionary mapping language codes to lists of equivalent terms
    """
    result: Dict[str, List[str]] = {}
    base_lower = base_term.lower().strip()

    # Find the best matching key in our dictionary
    matched_key = None
    for key in MULTILINGUAL_TERMS:
        if key == base_lower or key in base_lower or base_lower in key:
            matched_key = key
            break

    if matched_key:
        all_translations = MULTILINGUAL_TERMS[matched_key]
        if languages:
            for lang in languages:
                if lang in all_translations:
                    result[lang] = all_translations[lang]
        else:
            result = all_translations.copy()
    else:
        # Return original term for English if no translation found
        result["en"] = [base_term]

    return result


def expand_search_terms_multilingual(
    term: str,
    target_languages: Optional[List[str]] = None
) -> List[Tuple[str, str]]:
    """
    Expand a search term into multiple languages.

    Args:
        term: The search term to expand
        target_languages: Languages to include (None = all supported)

    Returns:
        List of (translated_term, language_code) tuples
    """
    if target_languages is None:
        target_languages = ["en", "es", "pt", "de", "fr", "it", "ja", "zh", "ko"]

    result: List[Tuple[str, str]] = []
    translations = get_multilingual_terms(term, target_languages)

    for lang, terms in translations.items():
        for t in terms:
            result.append((t, lang))

    # Always include the original term as English
    if not any(t == term for t, _ in result):
        result.append((term, "en"))

    return result


@lru_cache(maxsize=1000)
def translate_text_libre(text: str, source_lang: str = "auto", target_lang: str = "en") -> str:
    """
    Translate text using LibreTranslate (free, open-source).
    Caches results for performance.

    Falls back to original text if translation fails.
    """
    if not text or source_lang == target_lang:
        return text

    # List of public LibreTranslate instances
    libre_instances = [
        "https://libretranslate.com",
        "https://translate.argosopentech.com",
        "https://translate.terraprint.co",
    ]

    for instance in libre_instances:
        try:
            response = requests.post(
                f"{instance}/translate",
                json={
                    "q": text,
                    "source": source_lang if source_lang != "auto" else "auto",
                    "target": target_lang,
                    "format": "text",
                },
                timeout=10,
            )
            if response.status_code == 200:
                data = response.json()
                translated = data.get("translatedText", text)
                if translated:
                    return translated
        except requests.RequestException:
            continue

    return text  # Return original if all instances fail


def translate_text_mymemory(text: str, source_lang: str = "auto", target_lang: str = "en") -> str:
    """
    Translate text using MyMemory API (free tier: 1000 words/day).
    """
    if not text or len(text) > 500:  # MyMemory has length limits
        return text

    try:
        lang_pair = f"{source_lang}|{target_lang}" if source_lang != "auto" else f"auto|{target_lang}"
        response = requests.get(
            "https://api.mymemory.translated.net/get",
            params={"q": text, "langpair": lang_pair},
            timeout=10,
        )
        if response.status_code == 200:
            data = response.json()
            if data.get("responseStatus") == 200:
                translated = data.get("responseData", {}).get("translatedText", text)
                if translated and translated.lower() != "invalid language pair":
                    return translated
    except requests.RequestException:
        pass

    return text


def translate_to_english(text: str, source_lang: Optional[str] = None) -> Tuple[str, str]:
    """
    Translate text to English.

    Args:
        text: Text to translate
        source_lang: Source language code (auto-detect if None)

    Returns:
        Tuple of (translated_text, detected_language)
    """
    if not text:
        return text, "en"

    # Detect source language if not provided
    if source_lang is None:
        source_lang = detect_language(text)

    # Skip if already English
    if source_lang == "en":
        return text, "en"

    # Try LibreTranslate first (more reliable)
    translated = translate_text_libre(text, source_lang, "en")

    # Fallback to MyMemory if LibreTranslate failed (returned original)
    if translated == text:
        translated = translate_text_mymemory(text, source_lang, "en")

    return translated, source_lang


def batch_translate_to_english(texts: List[str]) -> List[Tuple[str, str, str]]:
    """
    Translate multiple texts to English.

    Returns:
        List of (original_text, translated_text, source_language) tuples
    """
    results = []
    for text in texts:
        translated, source_lang = translate_to_english(text)
        results.append((text, translated, source_lang))
    return results


# Regional search query builders
def build_regional_queries(term: str, regions: List[str]) -> Dict[str, List[str]]:
    """
    Build search queries optimized for different regions/languages.

    Args:
        term: Base search term in English
        regions: List of region codes (US, EU, UK, LATAM, APAC, etc.)

    Returns:
        Dictionary mapping regions to list of search queries
    """
    region_languages = {
        "US": ["en"],
        "UK": ["en"],
        "CA": ["en", "fr"],  # Canada: English + French
        "EU": ["en", "de", "fr", "it", "es"],  # Major EU languages
        "LATAM": ["es", "pt"],  # Latin America: Spanish + Portuguese
        "APAC": ["en", "ja", "zh", "ko"],  # Asia-Pacific
        "GLOBAL": ["en", "es", "pt", "de", "fr", "ja", "zh"],
    }

    queries: Dict[str, List[str]] = {}

    for region in regions:
        languages = region_languages.get(region.upper(), ["en"])
        region_queries = []

        for lang in languages:
            translations = get_multilingual_terms(term, [lang])
            for lang_terms in translations.values():
                region_queries.extend(lang_terms)

        # Deduplicate while preserving order
        seen = set()
        unique_queries = []
        for q in region_queries:
            if q.lower() not in seen:
                seen.add(q.lower())
                unique_queries.append(q)

        queries[region] = unique_queries

    return queries
