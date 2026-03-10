"""
AI Quality Analyst — Claude-Powered Chat Agent with Tool Use
Version 1.0

Features:
- Natural language Q&A about your loaded quality data (returns, Zendesk, B2B)
- Tool use: Claude calls Python functions to query REAL session data
- Extended thinking toggle for deep root-cause analysis
- Streaming response output for real-time feel
- Prompt caching on system prompt (cheaper, faster repeated calls)
- Regulatory Signal Watcher (optional FDA MAUDE check)

Architecture:
  1. User asks a question
  2. Claude decides which tool(s) to call
  3. We execute the Python tool with live session_state data
  4. Tool results go back to Claude
  5. Claude streams its final answer

All data access goes through session_data dict passed in at call time —
no direct Streamlit state access inside the agent core.
"""

import json
import logging
import time
from typing import Any, Dict, Generator, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Tool schemas (sent to Claude as JSON)
# ──────────────────────────────────────────────────────────────────────────────
ANALYST_TOOLS = [
    {
        "name": "get_data_summary",
        "description": (
            "Returns a summary of ALL data currently loaded in the app: "
            "how many rows, which modules have data, and what types of analysis "
            "have been run. Call this first to understand what data is available."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_top_issues",
        "description": (
            "Returns the top N SKUs or categories with the most quality issues, "
            "sorted by count, return rate, or safety flags. Use to find biggest problems."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "n": {
                    "type": "integer",
                    "description": "How many top items to return (default 10)",
                    "default": 10,
                },
                "sort_by": {
                    "type": "string",
                    "enum": ["count", "safety"],
                    "description": "Sort by return count or safety flag count",
                },
                "group_by": {
                    "type": "string",
                    "enum": ["sku", "category", "parent_sku"],
                    "description": "Group results by SKU, category, or parent SKU",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_category_breakdown",
        "description": (
            "Returns the distribution of quality issue categories across all "
            "loaded data, with counts and percentages. Optionally filter by SKU."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sku_filter": {
                    "type": "string",
                    "description": "Optional: only count records matching this SKU string",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_kpis",
        "description": (
            "Returns overall KPIs: total returns, top category, safety incident count, "
            "and any trend data available."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_sku_details",
        "description": (
            "Returns a detailed breakdown for one specific SKU: category counts, "
            "sample complaint texts, and any safety flags."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sku": {
                    "type": "string",
                    "description": "The SKU or Parent SKU to look up (partial match OK)",
                }
            },
            "required": ["sku"],
        },
    },
    {
        "name": "check_regulatory_signals",
        "description": (
            "Queries the FDA MAUDE adverse event database for reports matching "
            "the categories or keywords in the loaded data. Returns any matching "
            "recent adverse events that may be relevant to your products."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "Product keyword to search in FDA MAUDE (e.g. 'blood pressure monitor')",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (default 5)",
                    "default": 5,
                },
            },
            "required": ["keyword"],
        },
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Agent class
# ──────────────────────────────────────────────────────────────────────────────
class QualityAnalystAgent:
    """
    Claude agent with tool use for answering quality data questions.

    Usage:
        agent = QualityAnalystAgent(claude_key="sk-ant-...")
        session_snapshot = {
            "categorized_data": df,         # pandas DataFrame or None
            "zendesk_data": df,             # pandas DataFrame or None
            "zendesk_kpis": {...},          # dict or None
            "b2b_report_data": df,          # pandas DataFrame or None
        }
        for chunk in agent.stream(messages, session_snapshot):
            print(chunk, end="", flush=True)
    """

    MAX_TOOL_ROUNDS = 6
    DEFAULT_MODEL = "claude-sonnet-4-6"
    THINKING_MODEL = "claude-sonnet-4-6"  # extended thinking compatible

    def __init__(self, claude_key: str, model: str = DEFAULT_MODEL):
        self.claude_key = claude_key
        self.model = model
        self._session: Dict[str, Any] = {}

    # ── Public streaming entry point ────────────────────────────────────────

    def stream(
        self,
        messages: List[Dict],
        session_data: Dict[str, Any],
        use_extended_thinking: bool = False,
        thinking_budget: int = 8000,
    ) -> Generator[str, None, None]:
        """
        Run the agent loop with tool use, yielding text chunks for display.

        Args:
            messages: conversation history [{"role": "user"/"assistant", "content": ...}]
            session_data: snapshot of relevant session_state data for tools
            use_extended_thinking: enable Claude's extended thinking
            thinking_budget: token budget for extended thinking
        """
        self._session = session_data

        system_prompt = self._build_system_prompt()

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.claude_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "prompt-caching-2024-07-31",
        }

        base_payload: Dict[str, Any] = {
            "model": self.THINKING_MODEL if use_extended_thinking else self.model,
            "max_tokens": (thinking_budget + 4096) if use_extended_thinking else 4096,
            "system": [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "tools": ANALYST_TOOLS,
        }

        if use_extended_thinking:
            base_payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }
            yield "🧠 *Extended thinking enabled — deeper analysis in progress…*\n\n"

        current_messages = list(messages)

        try:
            import requests as req
        except ImportError:
            yield "❌ `requests` library not available."
            return

        for round_idx in range(self.MAX_TOOL_ROUNDS):
            payload = {**base_payload, "messages": current_messages}

            try:
                resp = req.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload,
                    timeout=90,
                )
            except Exception as exc:
                yield f"\n❌ Request failed: {exc}"
                return

            if resp.status_code != 200:
                yield f"\n❌ API error {resp.status_code}: {resp.text[:300]}"
                return

            result = resp.json()
            stop_reason = result.get("stop_reason")
            content_blocks = result.get("content", [])

            # Separate blocks by type
            tool_calls = [b for b in content_blocks if b.get("type") == "tool_use"]
            text_blocks = [b for b in content_blocks if b.get("type") == "text"]
            # thinking blocks exist but we don't display them to keep UI clean

            # Yield any text from this round
            for b in text_blocks:
                text = b.get("text", "")
                if text:
                    yield text

            # If no tool calls or final answer, we're done
            if stop_reason == "end_turn" or not tool_calls:
                return

            # Execute tool calls
            assistant_msg = {"role": "assistant", "content": content_blocks}
            tool_results = []

            for tc in tool_calls:
                tool_name = tc.get("name", "unknown")
                tool_input = tc.get("input", {})
                tool_id = tc.get("id", "")

                yield f"\n\n🔧 *{self._tool_display_name(tool_name)}…*\n"

                result_text = self._execute_tool(tool_name, tool_input)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": result_text,
                    }
                )

            current_messages = current_messages + [
                assistant_msg,
                {"role": "user", "content": tool_results},
            ]

        yield "\n\n*(reached max tool rounds)*"

    def complete(
        self,
        messages: List[Dict],
        session_data: Dict[str, Any],
        use_extended_thinking: bool = False,
    ) -> str:
        """Non-streaming version — returns full response as a string."""
        return "".join(self.stream(messages, session_data, use_extended_thinking))

    # ── Tool executor ───────────────────────────────────────────────────────

    def _execute_tool(self, name: str, args: dict) -> str:
        try:
            dispatch = {
                "get_data_summary": self._t_data_summary,
                "get_top_issues": self._t_top_issues,
                "get_category_breakdown": self._t_category_breakdown,
                "get_kpis": self._t_kpis,
                "get_sku_details": self._t_sku_details,
                "check_regulatory_signals": self._t_regulatory,
            }
            fn = dispatch.get(name)
            if fn is None:
                return f"Unknown tool: {name}"
            return fn(**args)
        except Exception as exc:
            logger.exception(f"Tool {name} raised: {exc}")
            return f"Tool error ({name}): {exc}"

    # ── Individual tool implementations ────────────────────────────────────

    def _t_data_summary(self) -> str:
        parts = []

        cat = self._session.get("categorized_data")
        if cat is not None and not _empty(cat):
            sku_col = _find_col(cat, ["sku", "parent_sku", "asin"])
            unique_skus = cat[sku_col].nunique() if sku_col else "?"
            parts.append(
                f"• Return Categorizer: {len(cat)} rows, {unique_skus} unique SKUs"
            )

        z = self._session.get("zendesk_data")
        if z is not None and not _empty(z):
            parts.append(f"• Zendesk Tickets: {len(z)} tickets")

        b2b = self._session.get("b2b_report_data")
        if b2b is not None and not _empty(b2b):
            parts.append(f"• B2B Report: {len(b2b)} rows")

        kpis = self._session.get("zendesk_kpis")
        if kpis:
            parts.append(f"• Zendesk KPIs computed: yes")

        if not parts:
            return (
                "No data is currently loaded. "
                "Please upload files in the Return Categorizer, B2C Zendesk Reporting, "
                "or B2B Report tools first."
            )
        return "Currently loaded data:\n" + "\n".join(parts)

    def _t_top_issues(
        self, n: int = 10, sort_by: str = "count", group_by: str = "sku"
    ) -> str:
        df = self._get_best_df()
        if df is None:
            return "No data loaded."

        sku_col = _find_col(df, ["parent_sku", "sku", "asin", "product_id"])
        cat_col = _find_col(df, ["category", "ai_category", "issue_type"])
        safety_col = _find_col(
            df, ["has_safety_concern", "safety_flag", "injury", "safety"]
        )

        group_col = (
            cat_col if group_by == "category" else (sku_col or df.columns[0])
        )

        grouped = df.groupby(group_col).size().reset_index(name="count")

        if sort_by == "safety" and safety_col:
            safe_df = df[
                df[safety_col]
                .astype(str)
                .str.lower()
                .isin(["true", "yes", "1", "true"])
            ]
            sc = safe_df.groupby(group_col).size().reset_index(name="safety_count")
            grouped = grouped.merge(sc, on=group_col, how="left").fillna(0)
            grouped["safety_count"] = grouped["safety_count"].astype(int)
            grouped = grouped.sort_values("safety_count", ascending=False)
        else:
            grouped = grouped.sort_values("count", ascending=False)

        top = grouped.head(n)
        lines = [f"Top {len(top)} by {sort_by} (grouped by {group_by}):"]
        for _, row in top.iterrows():
            line = f"  {row[group_col]}: {int(row['count'])} occurrences"
            if "safety_count" in row and row["safety_count"] > 0:
                line += f" ⚠️ {int(row['safety_count'])} safety flags"
            lines.append(line)
        return "\n".join(lines)

    def _t_category_breakdown(self, sku_filter: str = None) -> str:
        df = self._get_best_df()
        if df is None:
            return "No data loaded."

        cat_col = _find_col(
            df, ["category", "ai_category", "issue_type", "category_name"]
        )
        if not cat_col:
            return "No category column found in the data."

        if sku_filter:
            sku_col = _find_col(df, ["sku", "parent_sku", "asin"])
            if sku_col:
                df = df[
                    df[sku_col]
                    .astype(str)
                    .str.contains(sku_filter, case=False, na=False)
                ]

        total = len(df)
        if total == 0:
            return f"No records found for SKU filter '{sku_filter}'."

        breakdown = df[cat_col].value_counts()
        lines = [f"Category breakdown ({total} total{f' — SKU filter: {sku_filter}' if sku_filter else ''}):"]
        for cat, cnt in breakdown.items():
            pct = cnt / total * 100
            lines.append(f"  {cat}: {cnt} ({pct:.1f}%)")
        return "\n".join(lines)

    def _t_kpis(self) -> str:
        # Prefer pre-computed Zendesk KPIs
        kpis = self._session.get("zendesk_kpis")
        if kpis:
            lines = ["Zendesk KPIs:"]
            for k, v in kpis.items():
                lines.append(f"  {k}: {v}")
            return "\n".join(lines)

        df = self._get_best_df()
        if df is None:
            return "No data loaded."

        total = len(df)
        lines = [f"Total records: {total}"]

        cat_col = _find_col(df, ["category", "ai_category"])
        if cat_col and not df[cat_col].empty:
            top_cat = df[cat_col].mode()[0]
            top_cnt = int(df[cat_col].value_counts().iloc[0])
            lines.append(f"Top category: {top_cat} ({top_cnt} occurrences)")

        safety_col = _find_col(
            df, ["has_safety_concern", "safety_flag", "injury"]
        )
        if safety_col:
            safety_cnt = (
                df[safety_col]
                .astype(str)
                .str.lower()
                .isin(["true", "yes", "1"])
                .sum()
            )
            lines.append(f"Safety concerns: {int(safety_cnt)}")

        sku_col = _find_col(df, ["parent_sku", "sku", "asin"])
        if sku_col:
            lines.append(f"Unique SKUs: {df[sku_col].nunique()}")

        return "\n".join(lines)

    def _t_sku_details(self, sku: str) -> str:
        df = self._get_best_df()
        if df is None:
            return "No data loaded."

        sku_col = _find_col(df, ["sku", "parent_sku", "asin"])
        if not sku_col:
            return "No SKU column found."

        filtered = df[
            df[sku_col].astype(str).str.contains(sku, case=False, na=False)
        ]
        if filtered.empty:
            return f"No records found matching SKU '{sku}'."

        lines = [f"Details for '{sku}' — {len(filtered)} records:"]

        cat_col = _find_col(filtered, ["category", "ai_category"])
        if cat_col:
            lines.append("  Categories:")
            for cat, cnt in filtered[cat_col].value_counts().head(10).items():
                lines.append(f"    {cat}: {cnt}")

        safety_col = _find_col(
            filtered, ["has_safety_concern", "safety_flag", "injury"]
        )
        if safety_col:
            s_cnt = (
                filtered[safety_col]
                .astype(str)
                .str.lower()
                .isin(["true", "yes", "1"])
                .sum()
            )
            if s_cnt > 0:
                lines.append(f"  ⚠️ Safety flags: {int(s_cnt)}")

        text_col = _find_col(
            filtered, ["complaint", "comment", "text", "description", "subject", "body"]
        )
        if text_col:
            samples = filtered[text_col].dropna().astype(str).head(3).tolist()
            lines.append("  Sample complaints:")
            for s in samples:
                lines.append(f'    "{s[:120]}"')

        return "\n".join(lines)

    def _t_regulatory(self, keyword: str, limit: int = 5) -> str:
        """Query FDA MAUDE adverse event database."""
        try:
            import requests as req
            url = (
                f"https://api.fda.gov/device/event.json"
                f"?search=device.generic_name:{keyword.replace(' ', '+')}"
                f"&limit={limit}&sort=date_received:desc"
            )
            resp = req.get(url, timeout=10)
            if resp.status_code != 200:
                return f"FDA MAUDE query failed (status {resp.status_code})."
            data = resp.json()
            results = data.get("results", [])
            if not results:
                return f"No FDA MAUDE adverse events found for '{keyword}'."
            lines = [
                f"FDA MAUDE: {data.get('meta', {}).get('results', {}).get('total', '?')} total events for '{keyword}'. Recent {len(results)}:"
            ]
            for r in results:
                date = r.get("date_received", "?")
                event_type = r.get("event_type", "?")
                device = r.get("device", [{}])
                device_name = device[0].get("generic_name", "?") if device else "?"
                mfr = device[0].get("manufacturer_d_name", "?") if device else "?"
                lines.append(f"  [{date}] {event_type} — {device_name} by {mfr}")
            return "\n".join(lines)
        except Exception as exc:
            return f"FDA MAUDE query error: {exc}"

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _get_best_df(self):
        """Return the richest available DataFrame from session data."""
        for key in ["categorized_data", "zendesk_data", "b2b_report_data"]:
            df = self._session.get(key)
            if df is not None and not _empty(df):
                return df
        return None

    @staticmethod
    def _tool_display_name(name: str) -> str:
        labels = {
            "get_data_summary": "Checking loaded data",
            "get_top_issues": "Finding top issues",
            "get_category_breakdown": "Computing category breakdown",
            "get_kpis": "Pulling KPIs",
            "get_sku_details": "Looking up SKU details",
            "check_regulatory_signals": "Querying FDA MAUDE",
        }
        return labels.get(name, f"Running {name}")

    def _build_system_prompt(self) -> str:
        try:
            from corrections_memory import get_corrections_memory
            few_shot = get_corrections_memory().build_few_shot_block()
        except Exception as exc:
            logger.debug(f"Could not load corrections memory: {exc}")
            few_shot = ""

        base = (
            "You are an expert Quality Analyst AI embedded in a medical device quality "
            "management platform. You have access to tools that query real-time data "
            "loaded by the user (product returns, Zendesk tickets, B2B reports).\n\n"
            "GUIDELINES:\n"
            "- ALWAYS call tools to get real data before making claims about specific numbers\n"
            "- Start with get_data_summary if you're unsure what's loaded\n"
            "- Be specific: name SKUs, categories, counts\n"
            "- Proactively flag safety concerns and potential adverse events\n"
            "- Recommend actionable next steps: CAPA, vendor contact, regulatory reporting\n"
            "- If asked to draft emails, give a complete draft, not just bullet points\n"
            "- Keep answers concise but actionable — lead with the key insight\n"
        )
        if few_shot:
            base += f"\n\n{few_shot}"
        return base


# ──────────────────────────────────────────────────────────────────────────────
# Streamlit Chat UI
# ──────────────────────────────────────────────────────────────────────────────

def render_quality_analyst_chat(claude_key: str, session_data: Dict[str, Any]):
    """
    Full-page Streamlit UI for the Quality Analyst Chat.

    Args:
        claude_key: Anthropic API key string
        session_data: dict snapshot from st.session_state for tool access
    """
    import streamlit as st

    # ── Init session state ─────────────────────────────────────────────────
    if "analyst_messages" not in st.session_state:
        st.session_state.analyst_messages = []
    if "analyst_thinking" not in st.session_state:
        st.session_state.analyst_thinking = False

    agent = QualityAnalystAgent(claude_key)

    # ── Header ─────────────────────────────────────────────────────────────
    col_h1, col_h2 = st.columns([4, 1])
    with col_h1:
        st.markdown("### 🤖 Quality Analyst AI")
        st.caption(
            "Ask anything about your loaded data — top issues, category breakdowns, "
            "draft emails, safety flags, regulatory signals."
        )
    with col_h2:
        thinking = st.toggle(
            "🧠 Deep Think",
            value=st.session_state.analyst_thinking,
            key="analyst_thinking_toggle",
            help="Enables Claude's extended thinking for complex root-cause analysis (slower, more thorough)",
        )
        st.session_state.analyst_thinking = thinking

    st.markdown("---")

    # ── Quick-action buttons ────────────────────────────────────────────────
    st.caption("**Quick questions:**")
    qcols = st.columns(4)
    quick_questions = [
        ("🔝 Top Issues", "What are the top 10 SKUs by return count? Flag any safety concerns."),
        ("📊 Category Breakdown", "Give me the full category breakdown of all loaded data with percentages."),
        ("⚠️ Safety First", "Are there any safety concerns or potential injury risks in the data? List them."),
        ("📋 KPI Summary", "Give me a full KPI summary of the current data and highlight anything urgent."),
    ]
    for col, (label, question) in zip(qcols, quick_questions):
        if col.button(label, key=f"quick_{label}", width='stretch'):
            st.session_state.analyst_messages.append(
                {"role": "user", "content": question}
            )
            st.rerun()

    st.markdown("")

    # ── Chat history ────────────────────────────────────────────────────────
    for msg in st.session_state.analyst_messages:
        with st.chat_message(msg["role"]):
            content = msg["content"]
            # content may be a string or a list of blocks (from assistant with tools)
            if isinstance(content, str):
                st.markdown(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        st.markdown(block.get("text", ""))

    # ── Chat input ──────────────────────────────────────────────────────────
    if prompt := st.chat_input(
        "Ask about your quality data…",
        key="analyst_input",
    ):
        st.session_state.analyst_messages.append(
            {"role": "user", "content": prompt}
        )
        with st.chat_message("user"):
            st.markdown(prompt)

        # Build messages for API (only string content for user turns)
        api_messages = _build_api_messages(st.session_state.analyst_messages)

        with st.chat_message("assistant"):
            response_text = st.write_stream(
                agent.stream(
                    api_messages,
                    session_data,
                    use_extended_thinking=st.session_state.analyst_thinking,
                )
            )

        # Strip tool-progress markers before saving to history — they are UI
        # artifacts (🔧 *Querying…*) and must not be sent back to the API.
        import re as _re
        clean_response = _re.sub(r"\n*🔧 \*[^*]+…\*\n*", "\n", response_text or "").strip()

        # Save assistant response
        st.session_state.analyst_messages.append(
            {"role": "assistant", "content": clean_response or response_text}
        )

    # ── Footer controls ─────────────────────────────────────────────────────
    st.markdown("---")
    col_clr, col_info = st.columns([1, 3])
    with col_clr:
        if st.button("🗑️ Clear Chat", key="analyst_clear"):
            st.session_state.analyst_messages = []
            st.rerun()
    with col_info:
        try:
            from corrections_memory import get_corrections_memory
            s = get_corrections_memory().stats()
            st.caption(
                f"🧠 AI Memory: **{s['total_corrections']}** corrections stored "
                f"({s['high_confidence']} high-confidence) — "
                "the more you correct, the smarter it gets."
            )
        except Exception:
            st.caption("🧠 AI Memory active — correct categories to improve accuracy over time.")


# ──────────────────────────────────────────────────────────────────────────────
# Regulatory Signal Watcher (sidebar widget)
# ──────────────────────────────────────────────────────────────────────────────

def render_regulatory_watcher_sidebar(session_data: Dict[str, Any]):
    """
    Optional sidebar widget that checks FDA MAUDE for signals matching
    the top categories in the loaded data.  Manual trigger + optional auto.
    """
    import streamlit as st

    with st.sidebar.expander("🚨 Regulatory Watcher", expanded=False):
        st.caption("Checks FDA MAUDE for adverse events matching your data.")

        auto = st.toggle("Auto-check on load", value=False, key="reg_auto")
        keyword_override = st.text_input(
            "Search keyword (leave blank = auto-detect)",
            key="reg_keyword",
            placeholder="e.g. blood pressure monitor",
        )

        run_check = st.button("🔍 Check Now", key="reg_check_now", type="primary")

        # Auto trigger on first load
        if auto and "reg_last_result" not in st.session_state:
            run_check = True

        if run_check:
            keyword = keyword_override.strip()
            if not keyword:
                # Auto-detect from loaded data
                keyword = _auto_detect_keyword(session_data)
            if not keyword:
                st.warning("No data loaded — can't auto-detect keyword.")
            else:
                with st.spinner(f"Querying FDA MAUDE for '{keyword}'…"):
                    result = _query_fda_maude(keyword, limit=5)
                st.session_state["reg_last_result"] = (keyword, result)

        if "reg_last_result" in st.session_state:
            kw, res = st.session_state["reg_last_result"]
            st.caption(f"Last check: **{kw}**")
            st.text(res)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _query_fda_maude(keyword: str, limit: int = 5) -> str:
    """Standalone FDA MAUDE query — no class instance needed."""
    try:
        import requests as req
        url = (
            f"https://api.fda.gov/device/event.json"
            f"?search=device.generic_name:{keyword.replace(' ', '+')}"
            f"&limit={limit}&sort=date_received:desc"
        )
        resp = req.get(url, timeout=10)
        if resp.status_code != 200:
            return f"FDA MAUDE query failed (status {resp.status_code})."
        data = resp.json()
        results = data.get("results", [])
        if not results:
            return f"No FDA MAUDE adverse events found for '{keyword}'."
        total = data.get("meta", {}).get("results", {}).get("total", "?")
        lines = [f"FDA MAUDE: {total} total events for '{keyword}'. Most recent {len(results)}:"]
        for r in results:
            date = r.get("date_received", "?")
            event_type = r.get("event_type", "?")
            device = r.get("device", [{}])
            device_name = device[0].get("generic_name", "?") if device else "?"
            mfr = device[0].get("manufacturer_d_name", "?") if device else "?"
            lines.append(f"  [{date}] {event_type} — {device_name} by {mfr}")
        return "\n".join(lines)
    except Exception as exc:
        return f"FDA MAUDE query error: {exc}"


def _empty(df) -> bool:
    try:
        return df is None or len(df) == 0
    except Exception:
        return True


def _find_col(df, candidates: list) -> Optional[str]:
    """Return first matching column, case-insensitive, with partial match fallback."""
    cols_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_map:
            return cols_map[cand.lower()]
    for cand in candidates:
        for col_lower, col in cols_map.items():
            if cand.lower() in col_lower:
                return col
    return None


def _build_api_messages(messages: list) -> list:
    """Convert chat history to API-compatible messages list."""
    api_msgs = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        if isinstance(content, str):
            api_msgs.append({"role": role, "content": content})
        elif isinstance(content, list):
            # Keep only text blocks to avoid tool_use in non-current turns
            text = " ".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
            if text:
                api_msgs.append({"role": role, "content": text})
    return api_msgs


def _auto_detect_keyword(session_data: dict) -> str:
    """Guess best keyword from loaded data for MAUDE query."""
    for key in ["categorized_data", "zendesk_data"]:
        df = session_data.get(key)
        if df is not None and not _empty(df):
            cat_col = _find_col(df, ["category", "ai_category"])
            if cat_col:
                top = df[cat_col].mode()
                if not top.empty:
                    return str(top.iloc[0]).replace("_", " ").lower()
    return ""


__all__ = [
    "QualityAnalystAgent",
    "render_quality_analyst_chat",
    "render_regulatory_watcher_sidebar",
    "ANALYST_TOOLS",
]
