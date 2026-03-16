"""
AI Services — Anthropic Claude Migration
=========================================
Replaced: openai SDK (OpenAI) → anthropic SDK (Claude)
Primary models:
  - Fast:      claude-haiku-4-5-20251001
  - Reasoning: claude-sonnet-4-6
  - Powerful:  claude-opus-4-6  (optional — swap reasoning_model to use)

Gemini support is retained as an OPTIONAL alternate path using Google's
OpenAI-compatible endpoint (requires the openai package).  To use Gemini,
set provider="gemini" and supply a valid GEMINI_API_KEY / GOOGLE_API_KEY.

Audio transcription (Whisper) was an OpenAI-exclusive feature and has been
removed. transcribe_and_structure() now processes plain text input.

Migration Notes:
  - anthropic.Anthropic() replaces openai.OpenAI()
  - client.messages.create() replaces client.chat.completions.create()
  - response.content[0].text replaces response.choices[0].message.content
  - 'system' is a top-level parameter, not a messages[] entry
  - response_format={"type":"json_object"} removed; JSON enforced via prompts
  - JSON markdown fences stripped from Claude responses
"""

import re
import json
import logging
import os
from typing import Optional, Dict, Any, List, Tuple

import anthropic
import streamlit as st

from src.utils import retry_with_backoff

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_json_fences(text: str) -> str:
    """Remove markdown code fences that Claude may add around JSON."""
    return re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()


def _resolve_anthropic_key(explicit_key: Optional[str] = None) -> Optional[str]:
    """Resolve Anthropic API key from argument → Streamlit secrets → env."""
    if explicit_key:
        return explicit_key

    # Streamlit secrets
    try:
        for name in ("ANTHROPIC_API_KEY", "anthropic_api_key", "claude_api_key", "claude"):
            if name in st.secrets:
                val = str(st.secrets[name]).strip()
                if val:
                    return val
    except Exception:
        pass

    # Environment variables
    for name in ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY"):
        val = os.environ.get(name, "").strip()
        if val:
            return val

    return None


# ---------------------------------------------------------------------------
# Base Service
# ---------------------------------------------------------------------------

class AIServiceBase:
    """
    Base class for all AI services — Anthropic Claude primary, Gemini optional.

    Interface is backward-compatible with the previous OpenAI-backed version.
    """

    # Default token budgets
    FAST_MAX_TOKENS = 512
    REASONING_MAX_TOKENS = 2048

    def __init__(
        self,
        api_key: str,
        provider: str = "claude",
        model_overrides: Optional[Dict[str, str]] = None,
    ):
        model_overrides = model_overrides or {}
        self.provider = provider
        self.client = None          # Anthropic client (primary)
        self._gemini_client = None  # OpenAI-compat client for Gemini (optional)

        if provider == "gemini":
            self._init_gemini(api_key, model_overrides)
        else:
            self._init_claude(api_key, model_overrides)

    # ---- Initializers ----

    def _init_claude(self, api_key: str, model_overrides: Dict):
        resolved = _resolve_anthropic_key(api_key)
        if not resolved:
            logger.warning("AIServiceBase: no Anthropic API key found.")
            return
        try:
            self.client = anthropic.Anthropic(api_key=resolved)
            self.fast_model = model_overrides.get("fast", "claude-haiku-4-5-20251001")
            self.reasoning_model = model_overrides.get("reasoning", "claude-sonnet-4-6")
            logger.info(f"Claude initialized — fast={self.fast_model}, reasoning={self.reasoning_model}")
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {e}")

    def _init_gemini(self, api_key: str, model_overrides: Dict):
        """Google Gemini via OpenAI-compatible endpoint (requires openai package)."""
        try:
            import openai as _openai  # type: ignore
            if api_key and api_key.startswith("sk-"):
                logger.warning("Gemini provider selected but key looks like a Claude/Anthropic key (sk- prefix).")
            gemini_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
            self._gemini_client = _openai.OpenAI(
                api_key=gemini_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
            self.fast_model = model_overrides.get("fast", "gemini-1.5-flash")
            self.reasoning_model = model_overrides.get("reasoning", "gemini-1.5-pro")
            logger.info("Gemini initialized via OpenAI-compatible endpoint.")
        except ImportError:
            logger.error("openai package required for Gemini support. Run: pip install openai")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")

    # ---- Low-level call methods ----

    @retry_with_backoff(retries=3, backoff_in_seconds=2)
    def _generate_with_retry(
        self,
        model: str,
        system: str,
        user_message: str,
        max_tokens: int = 1024,
        temperature: float = 0.4,
    ):
        """Execute a Claude messages.create() call with retry logic."""
        if self.provider == "gemini" and self._gemini_client:
            # Route through OpenAI-compat endpoint
            return self._gemini_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

        # Anthropic (primary)
        return self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        )

    def _extract_content(self, response) -> str:
        """Extract text from either an Anthropic Message or OpenAI ChatCompletion."""
        if self.provider == "gemini":
            return response.choices[0].message.content or ""
        # Anthropic
        if response.content and len(response.content) > 0:
            return response.content[0].text.strip()
        return ""

    # ---- High-level helpers ----

    def _generate_json(
        self,
        prompt: str,
        system_instruction: str = None,
        use_reasoning: bool = False,
    ) -> Dict[str, Any]:
        """Generate a JSON response, handling fence stripping and parse errors."""
        if not (self.client or self._gemini_client):
            return {"error": "AI client not initialized (missing API key)."}

        system = system_instruction or "You are a helpful assistant."
        # Ensure JSON instruction present
        if "json" not in system.lower():
            system += "\n\nRespond with valid JSON only — no markdown fences, no commentary outside the JSON object."
        else:
            system += "\n\nRespond with valid JSON only — no markdown fences."

        model = self.reasoning_model if use_reasoning else self.fast_model
        max_tokens = self.REASONING_MAX_TOKENS if use_reasoning else self.FAST_MAX_TOKENS

        try:
            response = self._generate_with_retry(
                model=model,
                system=system,
                user_message=prompt,
                max_tokens=max_tokens,
                temperature=0.3,
            )
            raw = self._extract_content(response)
            cleaned = _strip_json_fences(raw)
            return json.loads(cleaned)

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e} | raw snippet: {raw[:300] if 'raw' in dir() else 'N/A'}")
            return {"error": "Failed to decode AI response as JSON"}
        except Exception as e:
            logger.error(f"AI JSON generation error: {e}")
            return {"error": str(e)}

    def _generate_text(
        self,
        prompt: str,
        system_instruction: str = None,
        use_reasoning: bool = False,
    ) -> str:
        """Generate a plain-text response."""
        if not (self.client or self._gemini_client):
            return "Error: AI client not initialized (missing API key)."

        system = system_instruction or "You are a helpful assistant."
        model = self.reasoning_model if use_reasoning else self.fast_model
        max_tokens = self.REASONING_MAX_TOKENS if use_reasoning else self.FAST_MAX_TOKENS

        try:
            response = self._generate_with_retry(
                model=model,
                system=system,
                user_message=prompt,
                max_tokens=max_tokens,
                temperature=0.4,
            )
            return self._extract_content(response)
        except Exception as e:
            logger.error(f"AI text generation error: {e}")
            return f"Error: {str(e)}"

    # ---- Verbosity helpers (unchanged interface) ----

    @staticmethod
    def _verbosity_instruction(level: str) -> str:
        if level == "Pithy":
            return "Be extremely concise. Use 3-5 bullets, max 120 words."
        if level == "Verbose":
            return "Be thorough and structured with headings and bullet points. Include key context and caveats."
        return "Be concise but clear. Use short paragraphs or bullets."

    def generate_text_with_verbosity(
        self,
        prompt: str,
        system_instruction: str,
        verbosity: str,
        use_reasoning: bool = False,
    ) -> str:
        verbosity_note = self._verbosity_instruction(verbosity)
        merged = f"{system_instruction}\n{verbosity_note}".strip()
        return self._generate_text(prompt, system_instruction=merged, use_reasoning=use_reasoning)

    def generate_dual_responses(
        self,
        prompt: str,
        system_instruction: str,
        concise_label: str = "Pithy",
        verbose_label: str = "Verbose",
        use_reasoning: bool = False,
    ) -> Tuple[str, str]:
        concise = self.generate_text_with_verbosity(
            prompt, system_instruction, concise_label, use_reasoning=use_reasoning
        )
        verbose = self.generate_text_with_verbosity(
            prompt, system_instruction, verbose_label, use_reasoning=use_reasoning
        )
        return concise, verbose


# ---------------------------------------------------------------------------
# Specialized Services (same public interface as before)
# ---------------------------------------------------------------------------

class AIService(AIServiceBase):
    """General-purpose AI service — used for recalls, screening, analysis."""

    def analyze_text(self, prompt: str, system_instruction: str = None) -> str:
        return self._generate_text(prompt, system_instruction)

    def transcribe_and_structure(self, text_input: str, context: str = "") -> Dict[str, str]:
        """
        Extract structured CAPA fields from plain text (meeting notes, typed observations).

        NOTE: Audio transcription (Whisper) was removed — Claude does not have a
        speech-to-text endpoint.  Pass already-transcribed text here instead.
        """
        if not (self.client or self._gemini_client):
            return {"error": "AI client not initialized."}

        system_prompt = "You are a Quality Assurance Assistant. Extract fields from text. Respond with valid JSON only — no fences."
        user_prompt = f"""
CONTEXT: {context}
TEXT: {text_input}

TASK:
Extract the following fields from the text and return as a JSON object:
- "Issue Description": What went wrong
- "Root Cause": Why it happened
- "Immediate Actions": What was done immediately
"""
        return self._generate_json(user_prompt, system_prompt, use_reasoning=True)

    def assess_relevance_json(self, my_context: str, record_text: str) -> Dict[str, str]:
        """Analyze a recall record for relevance to the user's product context."""
        system = (
            "You are a Regulatory Safety Officer. Analyze the recall for relevance to "
            "the user's product. Respond with valid JSON only — no fences."
        )
        prompt = f"""
USER CONTEXT (My Product):
{my_context}

RECALL RECORD:
{record_text}

TASK:
1. Compare 'My Firm' and 'My Model' against the Recall Record.
2. If they match, Risk is HIGH.
3. If the recall is for a competitor but same device type, Risk is MEDIUM (market surveillance).
4. If unrelated, Risk is LOW.
5. If the record is not in English, TRANSLATE the key issue in the analysis.

Return JSON strictly:
{{
    "risk": "High" | "Medium" | "Low",
    "analysis": "Short explanation. Mention if translation was applied."
}}
"""
        return self._generate_json(prompt, system, use_reasoning=True)

    def analyze_meeting_transcript(self, transcript_text: str) -> Dict[str, str]:
        system = "You are a QA Expert. Extract CAPA details (issue, root cause, actions) from notes. Respond with valid JSON only — no fences."
        user = f"Analyze this transcript and return a JSON object:\n{transcript_text}"
        return self._generate_json(user, system, use_reasoning=True)

    def screen_recalls(self, product_description: str) -> str:
        system = "You are a Regulatory Expert. Screen device against FDA/MDR recall databases."
        user = f"Screen this device: {product_description}. List common recall reasons and keywords."
        return self._generate_text(user, system)

    def generate_search_keywords(self, product_name: str, description: str) -> list:
        system = "You are a Regulatory Search Expert. Generate search terms for recall databases. Respond with valid JSON only — no fences."
        prompt = f"""
Product: {product_name}
Description: {description}

Task:
1. Identify the core device type (e.g., "infusion pump").
2. Identify 3-4 synonyms or related medical terms.
3. Identify potential hazard keywords.

Return JSON: {{ "keywords": ["term1", "term2", "term3"] }}
"""
        response = self._generate_json(prompt, system, use_reasoning=True)
        return response.get("keywords", [])

    def assess_relevance(self, product_desc: str, recall_text: str) -> str:
        system = "You are a Safety Engineer. Assess if a recall is relevant to our product."
        prompt = f"""
Our Product: {product_desc}
Recall Data: {recall_text}

Is this recall relevant to our product's technology or risks?
Answer with 'High', 'Medium', or 'Low' followed by a 1-sentence explanation.
"""
        return self._generate_text(prompt, system)


class DesignControlsTriager(AIServiceBase):
    def generate_design_controls(
        self, name: str, ifu: str, user_needs: str, tech_reqs: str, risks: str
    ) -> Dict[str, str]:
        system = (
            "You are a Medical Device Systems Engineer (ISO 13485). Generate Design Control "
            "documentation. Respond with valid JSON only — no fences."
        )
        prompt = f"""
Product: {name}
Description: {ifu}

Inputs:
- User Needs: {user_needs}
- Tech Reqs: {tech_reqs}
- Risks: {risks}

Generate a JSON with keys: 'traceability_matrix', 'inputs', 'outputs', 'verification',
'validation', 'plan', 'transfer', 'dhf'.
Each value should be a markdown string suitable for a report.
"""
        return self._generate_json(prompt, system, use_reasoning=True)


class UrraGenerator(AIServiceBase):
    def generate_urra(
        self, product_name: str, product_desc: str, user: str, environment: str
    ) -> Dict[str, Any]:
        system = (
            "You are a Usability Engineer (IEC 62366). Generate a Use-Related Risk Analysis (URRA). "
            "Respond with valid JSON only — no fences."
        )
        prompt = f"""
Device: {product_name}
Context: {product_desc}
User: {user}
Env: {environment}

Generate a list of 5-7 critical usability risks.
Return JSON with key 'urra_rows' containing a list of objects with keys:
'Task', 'Hazard', 'Severity' (1-5), 'Probability' (1-5), 'Risk Level' (Low/Med/High), 'Mitigation'.
"""
        return self._generate_json(prompt, system, use_reasoning=True)


class ManualWriter(AIServiceBase):
    def generate_manual_section(
        self,
        section_title: str,
        product_name: str,
        product_ifu: str,
        user_inputs: Dict,
        target_language: str,
    ) -> str:
        system = f"You are a Technical Writer. Write a user manual section in {target_language}."
        prompt = f"""
Section: {section_title}
Product: {product_name}
Context: {product_ifu}
Key Details: {user_inputs}

Write professional, clear, compliant content for this section using Markdown.
"""
        return self._generate_text(prompt, system)


class ProjectCharterHelper(AIServiceBase):
    def generate_charter_draft(
        self, product_name: str, problem_statement: str, target_user: str
    ) -> Dict[str, Any]:
        system = (
            "You are a Project Manager in MedTech. Draft a Project Charter. "
            "Respond with valid JSON only — no fences."
        )
        prompt = f"""
Project: {product_name}
Problem: {problem_statement}
User: {target_user}

Return JSON with keys: 'project_goal', 'scope', 'device_classification',
'applicable_standards' (list), 'stakeholders'.
"""
        return self._generate_json(prompt, system, use_reasoning=True)


class VendorEmailDrafter(AIServiceBase):
    def draft_vendor_email(
        self,
        goal: str,
        analysis_results: Any,
        sku: str,
        vendor: str,
        contact: str,
        english_level: int,
    ) -> str:
        system = "You are a Quality Manager writing to a supplier. Be professional and data-driven."
        prompt = f"""
Vendor: {vendor}, Contact: {contact}
SKU: {sku}
Goal: {goal}
Recipient English Level: {english_level}/5 (Adjust complexity accordingly).

Draft the email body.
"""
        return self._generate_text(prompt, system)


class HumanFactorsHelper(AIServiceBase):
    def generate_hf_report_from_answers(
        self, name: str, ifu: str, answers: Dict
    ) -> Dict[str, str]:
        system = (
            "You are a Human Factors Engineer (IEC 62366). Draft an HFE Report. "
            "Respond with valid JSON only — no fences."
        )
        prompt = f"""
Device: {name}
User Profile: {answers.get('user_profile')}
Critical Tasks: {answers.get('critical_tasks')}
Harms: {answers.get('potential_harms')}

Return JSON with keys: 'conclusion_statement', 'descriptions', 'device_interface',
'known_problems', 'hazards_analysis', 'preliminary_analyses', 'critical_tasks', 'validation_testing'.
"""
        return self._generate_json(prompt, system, use_reasoning=True)


class MedicalDeviceClassifier(AIServiceBase):
    def classify_device(self, description: str) -> Dict[str, str]:
        system = (
            "You are a Regulatory Affairs Specialist. Classify medical devices (FDA/MDR). "
            "Respond with valid JSON only — no fences."
        )
        prompt = f"""
Device Description: {description}

Return JSON with keys:
'classification' (e.g., 'Class II / Class IIa'),
'rationale' (Why?),
'product_code' (FDA Product Code if applicable).
"""
        return self._generate_json(prompt, system, use_reasoning=True)


# ---------------------------------------------------------------------------
# Multi-provider service (Claude primary + Gemini alternate)
# ---------------------------------------------------------------------------

class MultiProviderAIService:
    """
    Dual-provider service: Claude (primary) + Gemini (alternate).
    Falls back to Claude for all calls by default.
    """

    def __init__(
        self,
        claude_key: str,
        gemini_key: str,
        model_overrides: Optional[Dict[str, Dict[str, str]]] = None,
    ):
        model_overrides = model_overrides or {}
        self.claude = AIService(
            claude_key, provider="claude", model_overrides=model_overrides.get("claude")
        )
        self.gemini = AIService(
            gemini_key, provider="gemini", model_overrides=model_overrides.get("gemini")
        )
        self.default_provider = "claude"

    def generate_dual_responses(
        self, prompt: str, system_instruction: str
    ) -> Tuple[str, str]:
        """Concise response from Claude, verbose from Gemini."""
        concise = self.claude.generate_text_with_verbosity(
            prompt, system_instruction, "Pithy", use_reasoning=True
        )
        verbose = self.gemini.generate_text_with_verbosity(
            prompt, system_instruction, "Verbose", use_reasoning=True
        )
        return concise, verbose

    def _base(self) -> AIService:
        return self.claude if self.default_provider == "claude" else self.gemini

    def _generate_text(
        self, prompt: str, system_instruction: str = None, use_reasoning: bool = False
    ) -> str:
        return self._base()._generate_text(
            prompt, system_instruction=system_instruction, use_reasoning=use_reasoning
        )

    def _generate_json(
        self, prompt: str, system_instruction: str = None, use_reasoning: bool = False
    ) -> Dict[str, Any]:
        return self._base()._generate_json(
            prompt, system_instruction=system_instruction, use_reasoning=use_reasoning
        )

    def assess_relevance_json(self, my_context: str, record_text: str) -> Dict[str, str]:
        return self._base().assess_relevance_json(my_context, record_text)

    def generate_text_with_verbosity(
        self,
        prompt: str,
        system_instruction: str,
        verbosity: str,
        use_reasoning: bool = False,
    ) -> str:
        return self._base().generate_text_with_verbosity(
            prompt,
            system_instruction=system_instruction,
            verbosity=verbosity,
            use_reasoning=use_reasoning,
        )


# ---------------------------------------------------------------------------
# Singleton management
# ---------------------------------------------------------------------------

def get_ai_service() -> Optional[AIService]:
    """
    Return the shared AIService from session state.
    Initializes lazily using keys stored in session state or Streamlit secrets.

    Provider precedence:
      1. "claude"  — uses ANTHROPIC_API_KEY from secrets/env (default)
      2. "gemini"  — uses GEMINI_API_KEY / GOOGLE_API_KEY
      3. "both"    — returns a MultiProviderAIService
    """
    if "ai_service" not in st.session_state:
        provider = st.session_state.get("provider", "claude")
        model_overrides = st.session_state.get("model_overrides", {})

        if provider == "both":
            # Expect both keys to be present
            claude_key = _resolve_anthropic_key(st.session_state.get("api_key"))
            gemini_key = st.session_state.get("gemini_api_key", "")
            if claude_key and gemini_key:
                st.session_state.ai_service = MultiProviderAIService(
                    claude_key, gemini_key, model_overrides=model_overrides
                )
            else:
                return None

        elif provider == "gemini":
            api_key = st.session_state.get("api_key", "")
            if api_key:
                st.session_state.ai_service = AIService(
                    api_key,
                    provider="gemini",
                    model_overrides=model_overrides.get("gemini"),
                )
            else:
                return None

        else:
            # Default: Claude
            api_key = _resolve_anthropic_key(st.session_state.get("api_key"))
            if api_key:
                st.session_state.ai_service = AIService(
                    api_key,
                    provider="claude",
                    model_overrides=model_overrides.get("claude"),
                )
            else:
                return None

    return st.session_state.get("ai_service")
