"""
AI Helper Factory — Anthropic Claude Migration
Primary AI: Claude (Anthropic)
Key source: ANTHROPIC_API_KEY in Streamlit secrets
"""

import streamlit as st


class AIHelperFactory:
    """
    Factory to initialize AI helper classes (Claude-backed).
    Call initialize_ai_helpers() once at app startup after the API key is confirmed.
    """

    @staticmethod
    def initialize_ai_helpers(api_key: str):
        """
        Initialize shared AI helpers and store them in session state.

        Args:
            api_key: Anthropic API key. If empty, falls back to
                     ANTHROPIC_API_KEY in Streamlit secrets.
        """
        if st.session_state.get("ai_helpers_initialized", False):
            return

        try:
            # Resolve key: prefer explicit arg, then secrets
            resolved_key = api_key
            if not resolved_key:
                for name in ("ANTHROPIC_API_KEY", "anthropic_api_key", "claude_api_key"):
                    if name in st.secrets:
                        resolved_key = str(st.secrets[name]).strip()
                        if resolved_key:
                            break

            if not resolved_key:
                st.warning(
                    "⚠️ ANTHROPIC_API_KEY not found in Streamlit secrets. "
                    "Add it under Settings → Secrets to enable AI features."
                )
                return

            # Store the resolved key for lazy initialization in get_ai_service()
            if "api_key" not in st.session_state:
                st.session_state.api_key = resolved_key

            st.session_state.ai_helpers_initialized = True

        except Exception as e:
            st.error(f"Failed to initialize AI helpers: {e}")
