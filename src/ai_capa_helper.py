"""
AI CAPA Helper — Anthropic Claude Migration
Replaced: openai SDK → anthropic SDK
Primary model: claude-sonnet-4-6 (reasoning)
Fast model:    claude-haiku-4-5-20251001

Note: Audio transcription (Whisper) is an OpenAI-only feature.
      transcribe_audio() now returns an informative message directing
      users to manually enter text. All other functionality is unchanged.
"""

import json
from typing import Dict, Optional
import anthropic
import src.prompts as prompts
from src.utils import retry_with_backoff


class AICAPAHelper:
    """AI assistant for generating CAPA form suggestions using Anthropic Claude."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize with Anthropic API key.

        Looks for key in:
          1. api_key argument
          2. Streamlit secrets (ANTHROPIC_API_KEY)
          3. Environment variable ANTHROPIC_API_KEY
        """
        self.client: Optional[anthropic.Anthropic] = None
        self.fast_model = "claude-haiku-4-5-20251001"   # Fast, cheap — for polish/refine
        self.reasoning_model = "claude-sonnet-4-6"       # Balanced — for CAPA synthesis

        resolved_key = api_key or self._resolve_key()

        if resolved_key:
            try:
                self.client = anthropic.Anthropic(api_key=resolved_key)
            except Exception as e:
                print(f"Failed to initialize Anthropic client: {e}")

    @staticmethod
    def _resolve_key() -> Optional[str]:
        """Try Streamlit secrets then environment variable."""
        # Streamlit secrets
        try:
            import streamlit as st
            for name in ("ANTHROPIC_API_KEY", "anthropic_api_key", "claude_api_key"):
                if name in st.secrets:
                    val = str(st.secrets[name]).strip()
                    if val:
                        return val
        except Exception:
            pass

        # Environment variable
        import os
        for name in ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY"):
            val = os.environ.get(name, "").strip()
            if val:
                return val

        return None

    # ------------------------------------------------------------------
    # Internal call wrapper
    # ------------------------------------------------------------------
    @retry_with_backoff(retries=3, backoff_in_seconds=2)
    def _generate_unsafe(
        self,
        model: str,
        system: str,
        user_message: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> anthropic.types.Message:
        """
        Internal retryable method that wraps client.messages.create().

        NOTE: Anthropic API does not accept a 'system' role inside the
        messages list — it must be passed as the top-level 'system' param.
        """
        return self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        )

    @staticmethod
    def _extract_text(message: anthropic.types.Message) -> str:
        """Extract text content from an Anthropic Message response."""
        if message.content and len(message.content) > 0:
            return message.content[0].text.strip()
        return ""

    # ------------------------------------------------------------------
    # Public methods — same interface as the original OpenAI version
    # ------------------------------------------------------------------
    def transcribe_audio(self, audio_file) -> str:
        """
        Audio transcription is not supported by Anthropic's API.
        Returns a user-friendly message asking for manual text entry.

        If you require audio transcription, keep a separate OpenAI client
        using whisper-1 and pass the transcript text into refine_capa_input().
        """
        return (
            "Audio transcription is not available with the Claude API. "
            "Please type your notes directly into the text field, or use a "
            "separate transcription tool and paste the result here."
        )

    def refine_capa_input(self, field_name: str, rough_input: str, product_context: str) -> str:
        """
        Uses the FAST model (Haiku) to quickly polish rough text for a CAPA field.
        """
        if not self.client:
            return "AI client not initialized — check ANTHROPIC_API_KEY in Streamlit secrets."
        if not rough_input or len(rough_input) < 3:
            return rough_input

        system = prompts.CAPA_REFINE_SYSTEM.format(field_name=field_name)
        user_msg = (
            f"**Product Context:** {product_context}\n"
            f"**Rough Input:** {rough_input}\n"
            f"**Refined Output:**"
        )

        try:
            message = self._generate_unsafe(
                model=self.fast_model,
                system=system,
                user_message=user_msg,
                max_tokens=300,
                temperature=0.3,
            )
            return self._extract_text(message)
        except Exception as e:
            return f"Error refining input: {e}"

    def generate_capa_suggestions(
        self, issue_summary: str, analysis_results: Dict
    ) -> Dict[str, str]:
        """
        Uses the REASONING model (Sonnet) for complex CAPA synthesis.
        Returns a JSON dict of CAPA field suggestions.
        """
        if not self.client:
            return {"error": "AI client not initialized — check ANTHROPIC_API_KEY in Streamlit secrets."}

        # Build user prompt from template
        summary = (
            analysis_results.get("return_summary", {}).iloc[0]
            if not analysis_results.get("return_summary", {}).empty
            else {}
        )

        user_prompt = prompts.CAPA_SUGGESTION_USER_TEMPLATE.format(
            issue_summary=issue_summary,
            sku=summary.get("sku", "N/A"),
            return_rate=summary.get("return_rate", 0),
            total_returns=int(summary.get("total_returned", 0)),
        )

        # Ensure system prompt instructs JSON output
        system_content = prompts.CAPA_SUGGESTION_SYSTEM
        if "json" not in system_content.lower():
            system_content += (
                "\n\nIMPORTANT: Respond with valid JSON only — no markdown fences, "
                "no preamble, no commentary outside the JSON object."
            )
        else:
            # Reinforce JSON-only even if already mentioned
            system_content += (
                "\n\nRespond with valid JSON only — no markdown fences or extra commentary."
            )

        try:
            message = self._generate_unsafe(
                model=self.reasoning_model,
                system=system_content,
                user_message=user_prompt,
                max_tokens=1500,
                temperature=0.3,
            )
            raw = self._extract_text(message)

            # Strip markdown fences Claude might add despite instructions
            import re
            cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()

            return json.loads(cleaned)

        except json.JSONDecodeError as e:
            print(f"JSON decode error in generate_capa_suggestions: {e}\nRaw response: {raw[:500]}")
            return {"error": f"Failed to parse JSON from Claude response: {e}"}
        except Exception as e:
            print(f"Error generating CAPA suggestions: {e}")
            return {"error": f"Failed to generate CAPA suggestions: {e}"}
