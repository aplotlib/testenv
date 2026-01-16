# src/search/openfda.py
from __future__ import annotations
import requests
from datetime import date
from typing import Any, Dict, List, Optional, Sequence
import re

DEVICE_RECALL_ENDPOINT = "https://api.fda.gov/device/recall.json"
DEVICE_ENF_ENDPOINT    = "https://api.fda.gov/device/enforcement.json"

# Product code mappings for common medical devices (more precise FDA searches)
PRODUCT_CODES = {
    "blood pressure": ["DXN", "DXQ", "DXP"],  # Sphygmomanometers (various types)
    "bp monitor": ["DXN", "DXQ", "DXP"],
    "sphygmomanometer": ["DXN", "DXQ", "DXP"],
    "infusion pump": ["FRN", "MEA", "MEB"],
    "wheelchair": ["IRL", "IRN", "ITI"],
    "pacemaker": ["DXY", "DTB", "LWP"],
    "defibrillator": ["MKJ", "MQP", "DRY"],
    "ventilator": ["BTL", "CBK", "MNT"],
    "glucometer": ["NBW", "NBX"],
    "pulse oximeter": ["DQA", "DPZ"],
    "thermometer": ["FLL", "FLK"],
    "nebulizer": ["NBZ", "CAH"],
    "cpap": ["MNR", "MNQ"],
    "oxygen concentrator": ["CAF", "CBK"],
}


def _yyyymmdd(d: date) -> str:
    return d.strftime("%Y%m%d")


def _openfda(endpoint: str, search: str, limit: int = 100, skip: int = 0) -> List[Dict[str, Any]]:
    """Execute OpenFDA query with pagination support."""
    params = {"search": search, "limit": min(max(limit, 1), 1000)}
    if skip > 0:
        params["skip"] = skip
    try:
        r = requests.get(endpoint, params=params, timeout=30)
        # openFDA returns 404 when no results; treat as empty.
        if r.status_code == 404:
            return []
        r.raise_for_status()
        return r.json().get("results", []) or []
    except requests.RequestException:
        return []


def _openfda_paginated(endpoint: str, search: str, limit: int = 100) -> List[Dict[str, Any]]:
    """Fetch results with pagination to get more comprehensive data."""
    all_results: List[Dict[str, Any]] = []
    page_size = min(limit, 100)  # FDA max is 1000 but 100 is more reliable
    skip = 0

    while len(all_results) < limit:
        remaining = limit - len(all_results)
        batch_size = min(page_size, remaining)
        results = _openfda(endpoint, search, batch_size, skip)
        if not results:
            break
        all_results.extend(results)
        if len(results) < batch_size:
            break  # No more results
        skip += len(results)

    return all_results[:limit]


def _build_search_query(
    product_name: str,
    start: date,
    end: date,
    use_wildcards: bool = True,
    search_all_fields: bool = True
) -> str:
    """Build comprehensive search query with multiple field coverage."""
    # Clean and prepare the search term
    clean_name = product_name.strip()

    # Split into words for wildcard expansion
    words = clean_name.split()

    # Build field-specific queries
    field_queries = []

    # Primary fields to search
    primary_fields = [
        "product_description",
        "reason_for_recall",
        "recalling_firm",
    ]

    # Additional fields for comprehensive coverage
    extra_fields = [
        "product_code",
        "code_info",
        "distribution_pattern",
        "openfda.device_name",
        "openfda.brand_name",
    ] if search_all_fields else []

    all_fields = primary_fields + extra_fields

    for field in all_fields:
        # Exact phrase match
        field_queries.append(f'{field}:"{clean_name}"')

        # Wildcard matches for partial terms
        if use_wildcards and len(words) >= 1:
            for word in words:
                if len(word) >= 3:  # Only wildcard meaningful words
                    field_queries.append(f'{field}:{word}*')

    # Check for product codes
    lower_name = clean_name.lower()
    for key, codes in PRODUCT_CODES.items():
        if key in lower_name:
            for code in codes:
                field_queries.append(f'product_code:"{code}"')

    # Combine all field queries with OR
    combined = " OR ".join(field_queries)

    # Add date range
    date_query = f"report_date:[{_yyyymmdd(start)} TO {_yyyymmdd(end)}]"

    return f"({combined}) AND {date_query}"


def search_device_recall(
    product_name: str,
    start: date,
    end: date,
    limit: int = 100,
    use_wildcards: bool = True,
    paginate: bool = True
) -> List[Dict[str, Any]]:
    """
    Search FDA device recalls with enhanced query capabilities.

    Args:
        product_name: Product to search for
        start: Start date for recall reports
        end: End date for recall reports
        limit: Maximum results to return
        use_wildcards: Enable wildcard matching for broader results
        paginate: Enable pagination for more comprehensive results
    """
    search_query = _build_search_query(product_name, start, end, use_wildcards)

    if paginate:
        return _openfda_paginated(DEVICE_RECALL_ENDPOINT, search_query, limit)
    return _openfda(DEVICE_RECALL_ENDPOINT, search_query, limit)


def search_device_enforcement(
    product_name: str,
    start: date,
    end: date,
    limit: int = 100,
    use_wildcards: bool = True,
    paginate: bool = True
) -> List[Dict[str, Any]]:
    """
    Search FDA enforcement actions with enhanced query capabilities.
    """
    search_query = _build_search_query(product_name, start, end, use_wildcards)

    if paginate:
        return _openfda_paginated(DEVICE_ENF_ENDPOINT, search_query, limit)
    return _openfda(DEVICE_ENF_ENDPOINT, search_query, limit)


def search_by_product_code(
    product_codes: Sequence[str],
    start: date,
    end: date,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    Search recalls directly by FDA product codes for maximum precision.
    """
    if not product_codes:
        return []

    code_queries = [f'product_code:"{code}"' for code in product_codes]
    combined = " OR ".join(code_queries)
    date_query = f"report_date:[{_yyyymmdd(start)} TO {_yyyymmdd(end)}]"
    search_query = f"({combined}) AND {date_query}"

    return _openfda_paginated(DEVICE_RECALL_ENDPOINT, search_query, limit)


def get_product_codes_for_term(term: str) -> List[str]:
    """Get FDA product codes associated with a search term."""
    lower_term = term.lower()
    codes = []
    for key, code_list in PRODUCT_CODES.items():
        if key in lower_term or lower_term in key:
            codes.extend(code_list)
    return list(set(codes))
