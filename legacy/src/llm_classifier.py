from __future__ import annotations
import json
import os
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

SYSTEM = """You are a safety/compliance analyst.
Classify whether a web/recall record likely pertains to the given product.
Be conservative: if unclear, say "uncertain" and lower confidence.
Return ONLY valid JSON matching the schema."""

SCHEMA_DESCRIPTION = """Return a JSON object with exactly these fields:
- relation: one of "likely_match", "similar_product", "not_related", "uncertain"
- category: one of "recall", "regulatory_action", "injury", "lawsuit", "negative_press", "other"
- severity: one of "low", "medium", "high"
- confidence: a number between 0 and 1
- why: a brief explanation string"""


def _get_anthropic_key() -> Optional[str]:
    """Resolve Anthropic API key from Streamlit secrets or environment."""
    try:
        import streamlit as st
        if hasattr(st, 'secrets'):
            for key_name in ["ANTHROPIC_API_KEY", "anthropic_api_key", "claude_api_key", "claude"]:
                if key_name in st.secrets:
                    val = str(st.secrets[key_name]).strip()
                    if val:
                        return val
    except Exception:
        pass
    return os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY")


def classify_hit(sku: str, product_name: str, hit: Dict[str, Any]) -> Dict[str, Any]:
    """Classify a recall/regulatory hit against a product using Claude."""
    import requests

    api_key = _get_anthropic_key()
    if not api_key:
        logger.warning("No Anthropic API key available for llm_classifier")
        return {
            "relation": "uncertain",
            "category": "other",
            "severity": "low",
            "confidence": 0.0,
            "why": "AI classifier unavailable — no API key configured"
        }

    user_content = (
        f"Classify this record for product SKU={sku} ({product_name}):\n"
        f"{json.dumps(hit, ensure_ascii=False)[:2000]}\n\n"
        f"{SCHEMA_DESCRIPTION}"
    )

    payload = {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 256,
        "system": SYSTEM,
        "messages": [{"role": "user", "content": user_content}],
    }

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        text = response.json()["content"][0]["text"].strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        return json.loads(text)

    except (json.JSONDecodeError, KeyError, IndexError) as e:
        logger.error(f"llm_classifier parse error: {e}")
        return {"relation": "uncertain", "category": "other", "severity": "low",
                "confidence": 0.0, "why": f"Parse error: {e}"}
    except Exception as e:
        logger.error(f"llm_classifier API error: {e}")
        return {"relation": "uncertain", "category": "other", "severity": "low",
                "confidence": 0.0, "why": f"API error: {e}"}
