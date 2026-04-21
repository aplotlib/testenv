"""
Corrections Memory — Persistent Learning from User Category Overrides
Version 1.0

When a user overrides an AI-assigned category, the correction is saved to disk.
On subsequent analyses, top corrections are injected as few-shot examples
into the AI prompt, so the model improves over time without retraining.

Storage: ~/.quality_app/corrections.json  (user home dir, portable)
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
# Prefer ~/.quality_app; fall back to app directory if home isn't writable
# (e.g. Streamlit Cloud container — corrections persist within a deployment)
_HOME_DIR = Path(os.path.expanduser("~/.quality_app"))
_APP_DIR = Path(__file__).parent / ".quality_app"
MEMORY_DIR = _HOME_DIR if os.access(str(_HOME_DIR.parent), os.W_OK) else _APP_DIR
CORRECTIONS_FILE = MEMORY_DIR / "corrections.json"
MAX_FEW_SHOT = 20        # Max corrections injected into prompt
MIN_COUNT_TO_USE = 2     # Corrections seen at least N times get priority
KEY_MAX_LEN = 200        # Max chars used for dict key


# ──────────────────────────────────────────────────────────────────────────────
# Core class
# ──────────────────────────────────────────────────────────────────────────────
class CorrectionsMemory:
    """
    Persistent store for user category corrections.

    Each correction maps a complaint text fragment to the user's preferred
    category.  Frequently confirmed corrections are prioritised as few-shot
    examples so the AI learns your specific product taxonomy over time.
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = path or CORRECTIONS_FILE
        # key → {"text_fragment", "old_category", "new_category", "count",
        #          "first_seen", "last_seen"}
        self.corrections: Dict[str, dict] = {}
        self._load()

    # ── Persistence ────────────────────────────────────────────────────────

    def _load(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists():
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.corrections = data.get("corrections", {})
                logger.info(
                    f"CorrectionsMemory: loaded {len(self.corrections)} entries"
                )
        except Exception as exc:
            logger.warning(f"CorrectionsMemory load failed: {exc}")
            self.corrections = {}

    def save(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "corrections": self.corrections,
                "last_updated": datetime.now().isoformat(),
                "version": "1.0",
            }
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.error(f"CorrectionsMemory save failed: {exc}")

    # ── Recording corrections ───────────────────────────────────────────────

    def add_correction(
        self, complaint_text: str, old_category: str, new_category: str
    ):
        """Record a user-confirmed correction.  No-op if categories are equal."""
        if not complaint_text or old_category == new_category:
            return
        key = self._make_key(complaint_text)
        if key in self.corrections:
            entry = self.corrections[key]
            entry["count"] += 1
            entry["new_category"] = new_category  # accept latest override
            entry["last_seen"] = datetime.now().isoformat()
        else:
            self.corrections[key] = {
                "text_fragment": complaint_text[:KEY_MAX_LEN],
                "old_category": old_category,
                "new_category": new_category,
                "count": 1,
                "first_seen": datetime.now().isoformat(),
                "last_seen": datetime.now().isoformat(),
            }
        self.save()

    def bulk_add(self, corrections: List[dict]):
        """Add multiple corrections at once.  Each dict: text, old_cat, new_cat."""
        for c in corrections:
            self.add_correction(
                c.get("text", ""),
                c.get("old_category", ""),
                c.get("new_category", ""),
            )

    # ── Retrieval for prompt injection ─────────────────────────────────────

    def get_few_shot_examples(self, n: int = MAX_FEW_SHOT) -> List[dict]:
        """
        Return top corrections sorted by frequency, for injection into AI prompts.
        High-count corrections (confirmed multiple times) rank first.
        """
        eligible = list(self.corrections.values())
        # Sort: high-count first, then most recent
        eligible.sort(key=lambda x: (x["count"], x["last_seen"]), reverse=True)
        return eligible[:n]

    def build_few_shot_block(self) -> str:
        """
        Build a formatted prompt block from stored corrections.
        Returns empty string if no corrections exist yet.
        """
        examples = self.get_few_shot_examples()
        if not examples:
            return ""
        lines = [
            "LEARNED CORRECTIONS FROM USER FEEDBACK",
            "(These corrections come from previous analyst reviews — "
            "treat them as ground truth and prioritise them over your defaults):",
            "",
        ]
        for ex in examples:
            text = ex["text_fragment"][:120].replace("\n", " ")
            lines.append(f'  Text: "{text}"')
            lines.append(f'  ✓ Correct category : {ex["new_category"]}')
            if ex["count"] >= MIN_COUNT_TO_USE:
                lines.append(f'    (confirmed {ex["count"]}× — high confidence)')
            lines.append("")
        return "\n".join(lines)

    def get_direct_match(self, complaint_text: str) -> Optional[str]:
        """
        If complaint_text exactly matches a stored key, return the corrected
        category immediately (no AI call needed).
        """
        entry = self.corrections.get(self._make_key(complaint_text))
        return entry.get("new_category") if entry else None

    # ── Stats / admin ───────────────────────────────────────────────────────

    def stats(self) -> dict:
        total = len(self.corrections)
        high_conf = sum(
            1 for v in self.corrections.values() if v["count"] >= MIN_COUNT_TO_USE
        )
        categories_corrected = len(
            {v["new_category"] for v in self.corrections.values()}
        )
        return {
            "total_corrections": total,
            "high_confidence": high_conf,
            "categories_covered": categories_corrected,
            "storage_path": str(self.path),
        }

    def clear(self):
        """Wipe all stored corrections (irreversible)."""
        self.corrections = {}
        self.save()

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _make_key(text: str) -> str:
        return text.strip().lower()[:KEY_MAX_LEN]


# ──────────────────────────────────────────────────────────────────────────────
# Singleton accessor
# ──────────────────────────────────────────────────────────────────────────────
_instance: Optional[CorrectionsMemory] = None


def get_corrections_memory() -> CorrectionsMemory:
    """Return the shared CorrectionsMemory instance (created on first call)."""
    global _instance
    if _instance is None:
        _instance = CorrectionsMemory()
    return _instance


# ──────────────────────────────────────────────────────────────────────────────
# Streamlit UI helper
# ──────────────────────────────────────────────────────────────────────────────
def render_corrections_panel():
    """
    Render a small Streamlit expander showing correction stats and
    a button to clear the memory.  Meant to be embedded in the sidebar
    or settings area.
    """
    import streamlit as st

    mem = get_corrections_memory()
    s = mem.stats()

    with st.expander("🧠 AI Memory", expanded=False):
        col1, col2, col3 = st.columns(3)
        col1.metric("Corrections", s["total_corrections"])
        col2.metric("High Confidence", s["high_confidence"])
        col3.metric("Categories", s["categories_covered"])

        if s["total_corrections"] > 0:
            examples = mem.get_few_shot_examples(5)
            st.caption("**Recent corrections (sample):**")
            for ex in examples:
                st.caption(
                    f"• \"{ex['text_fragment'][:60]}…\" → **{ex['new_category']}** "
                    f"(×{ex['count']})"
                )

        col_a, col_b = st.columns(2)
        if col_a.button("🗑️ Clear Memory", key="clear_corrections", type="secondary"):
            mem.clear()
            st.success("Memory cleared.")
            st.rerun()
        col_b.caption(f"📁 {s['storage_path']}")


__all__ = [
    "CorrectionsMemory",
    "get_corrections_memory",
    "render_corrections_panel",
    "MAX_FEW_SHOT",
    "MIN_COUNT_TO_USE",
]
