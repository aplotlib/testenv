"""
Enhanced AI Analysis Module — Claude (Anthropic) Only
Version 32.0

Key Features:
- Claude (Anthropic) as sole AI provider
- Claude Haiku 4.5 for fast categorization/summarization
- Claude Sonnet 4.6 / Opus 4.6 for complex cases
- Batch processing with parallel API calls
- Prompt caching and extended thinking support
- Dynamic worker scaling
"""

import logging
import os
import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Safe imports
def safe_import(module_name):
    try:
        return __import__(module_name), True
    except ImportError:
        logger.warning(f"Module {module_name} not available")
        return None, False

# Check for dependencies
requests, has_requests = safe_import('requests')

# API Configuration
API_TIMEOUT = 45
MAX_RETRIES = 2

# Token configurations by mode
TOKEN_LIMITS = {
    'standard': 60,      # Single category name response — small by design
    'enhanced': 200,
    'extreme': 400,
    'chat': 1000,
    'summary': 500
}

# =====================================================================
# MODEL CONFIGURATIONS — Current Claude 4.x / 4.5 family (March 2026)
# =====================================================================
MODELS = {
    'claude': {
        'fast':      'claude-haiku-4-5-20251001',   # Fastest, cheapest
        'standard':  'claude-sonnet-4-6',             # Balanced performance
        'enhanced':  'claude-sonnet-4-6',             # Quality tasks
        'extreme':   'claude-opus-4-6',               # Maximum capability
        'powerful':  'claude-opus-4-6',               # Alias for extreme
        'chat':      'claude-sonnet-4-6',             # Conversational
        'summary':   'claude-haiku-4-5-20251001'      # Fast summaries
    }
}

# Updated pricing per 1K tokens (March 2026)
PRICING = {
    # Claude — current family
    'claude-haiku-4-5-20251001':  {'input': 0.00080, 'output': 0.00400},
    'claude-sonnet-4-6':          {'input': 0.00300, 'output': 0.01500},
    'claude-opus-4-6':            {'input': 0.01500, 'output': 0.07500},
    # Claude 3.x — legacy (kept for reference)
    'claude-3-5-haiku-20241022':  {'input': 0.00100, 'output': 0.00500},
    'claude-3-5-sonnet-20241022': {'input': 0.00300, 'output': 0.01500},
    'claude-3-opus-20240229':     {'input': 0.01500, 'output': 0.07500},
}

# Medical Device Return Categories — granular taxonomy for actionable QA analysis
# Size/Fit split into directional sub-types for root cause clarity.
# Comfort split by failure mode (pressure vs. rigidity vs. softness vs. skin reaction).
# Defects split by physical failure type.
MEDICAL_DEVICE_CATEGORIES = [
    # ── Quality categories (aligned with Return Categorizer) ──────────────────
    'Product Defects/Quality',       # Broken, malfunctioning, poor material, cosmetic damage, not as described
    'Performance/Effectiveness',     # Doesn't help therapeutically, compatibility issues, assembly/usage difficulty
    'Missing Components',            # Missing parts, incomplete, no instructions
    'Design/Material Issues',        # Size/fit problems, comfort issues, skin irritation, material design flaws
    'Stability/Positioning Issues',  # Wobbles, tips over, shifts, slides out of position
    'Medical/Health Concerns',       # Injury, safety hazard, adverse medical reaction
    # ── Non-quality categories ─────────────────────────────────────────────────
    'Customer: Changed Mind / No Longer Needed',
    'Customer: Ordered Wrong Size or Item',
    'Fulfillment: Damaged in Shipping',
    'Fulfillment: Wrong Item Sent',
    'Fulfillment: Delivery Issue',
    'General Inquiry / Not a Quality Issue',
    'Other / Miscellaneous',
]

# Legacy category map — maps old granular strings to unified categories for backward compatibility
LEGACY_CATEGORY_MAP = {
    # Old granular v1/v2 → unified
    'Size: Too Small':                               'Design/Material Issues',
    'Size: Too Large':                               'Design/Material Issues',
    "Size: Doesn't Fit / Wrong Dimensions":          'Design/Material Issues',
    'Comfort: Causes Pain or Pressure':              'Design/Material Issues',
    'Comfort: Too Hard / Rigid':                     'Design/Material Issues',
    'Comfort: Too Soft / Lacks Support':             'Design/Material Issues',
    'Comfort: Skin Irritation or Allergic Reaction': 'Design/Material Issues',
    'Defect: Broken / Structural Failure':           'Product Defects/Quality',
    'Defect: Malfunctions / Stops Working':          'Product Defects/Quality',
    'Defect: Cosmetic Damage':                       'Product Defects/Quality',
    'Defect: Poor Material Quality':                 'Product Defects/Quality',
    'Wrong Product / Not as Described':              'Product Defects/Quality',
    'Missing or Incomplete Components':              'Missing Components',
    "Performance: Ineffective / Doesn't Help":       'Performance/Effectiveness',
    'Equipment Compatibility Issue':                 'Performance/Effectiveness',
    'Stability: Shifts / Unstable / Falls':          'Stability/Positioning Issues',
    'Assembly / Usage Difficulty':                   'Performance/Effectiveness',
    'Medical / Safety Concern':                      'Medical/Health Concerns',
    # Old broad names
    'Size/Fit Issues':                               'Design/Material Issues',
    'Comfort Issues':                                'Design/Material Issues',
    'Product Defects/Quality':                       'Product Defects/Quality',
    'Performance/Effectiveness':                     'Performance/Effectiveness',
    'Stability/Positioning Issues':                  'Stability/Positioning Issues',
    'Equipment Compatibility':                       'Performance/Effectiveness',
    'Design/Material Issues':                        'Design/Material Issues',
    'Wrong Product/Misunderstanding':                'Product Defects/Quality',
    'Missing Components':                            'Missing Components',
    'Customer Error/Changed Mind':                   'Customer: Changed Mind / No Longer Needed',
    'Shipping/Fulfillment Issues':                   'Fulfillment: Damaged in Shipping',
    'Assembly/Usage Difficulty':                     'Performance/Effectiveness',
    'Medical/Health Concerns':                       'Medical/Health Concerns',
    'Price/Value':                                   'Other / Miscellaneous',
    'Other/Miscellaneous':                           'Other / Miscellaneous',
    'Other / Miscellaneous':                         'Other / Miscellaneous',
}

# FBA reason code mapping — unified to Return Categorizer categories
FBA_REASON_MAP = {
    'NOT_COMPATIBLE':                'Performance/Effectiveness',
    'DAMAGED_BY_FC':                 'Product Defects/Quality',
    'DAMAGED_BY_CARRIER':            'Fulfillment: Damaged in Shipping',
    'DEFECTIVE':                     'Product Defects/Quality',
    'NOT_AS_DESCRIBED':              'Product Defects/Quality',
    'WRONG_ITEM':                    'Fulfillment: Wrong Item Sent',
    'MISSING_PARTS':                 'Missing Components',
    'QUALITY_NOT_ADEQUATE':          'Product Defects/Quality',
    'UNWANTED_ITEM':                 'Customer: Changed Mind / No Longer Needed',
    'UNAUTHORIZED_PURCHASE':         'Customer: Changed Mind / No Longer Needed',
    'CUSTOMER_DAMAGED':              'Customer: Changed Mind / No Longer Needed',
    'SWITCHEROO':                    'Fulfillment: Wrong Item Sent',
    'EXPIRED_ITEM':                  'Product Defects/Quality',
    'DAMAGED_GLASS_VIAL':            'Product Defects/Quality',
    'DIFFERENT_PRODUCT':             'Fulfillment: Wrong Item Sent',
    'MISSING_ITEM':                  'Missing Components',
    'NOT_DELIVERED':                 'Fulfillment: Delivery Issue',
    'ORDERED_WRONG_ITEM':            'Customer: Ordered Wrong Size or Item',
    'UNNEEDED_ITEM':                 'Customer: Changed Mind / No Longer Needed',
    'BAD_GIFT':                      'Customer: Changed Mind / No Longer Needed',
    'INACCURATE_WEBSITE_DESCRIPTION':'Product Defects/Quality',
    'BETTER_PRICE_AVAILABLE':        'Other / Miscellaneous',
    'DOES_NOT_FIT':                  'Design/Material Issues',
    'NOT_COMPATIBLE_WITH_DEVICE':    'Performance/Effectiveness',
    'UNSATISFACTORY_PRODUCT':        'Performance/Effectiveness',
    'ARRIVED_LATE':                  'Other / Miscellaneous',
    'TOO_SMALL':                     'Design/Material Issues',
    'TOO_LARGE':                     'Design/Material Issues',
    'UNCOMFORTABLE':                 'Design/Material Issues',
    'DIFFICULT_TO_USE':              'Performance/Effectiveness',
    'DAMAGED':                       'Product Defects/Quality',
    'BROKEN':                        'Product Defects/Quality',
    'POOR_QUALITY':                  'Product Defects/Quality',
    'NOT_WORKING':                   'Product Defects/Quality',
    'DOESNT_WORK':                   'Product Defects/Quality',
}

# Quick categorization patterns for speed — unified to Return Categorizer categories
QUICK_PATTERNS = {
    # Safety first — always wins
    'Medical/Health Concerns': [
        r'\binjur(y|ed|ies)\b', r'\bhospital(ized)?\b', r'\bemergency\b',
        r'\bdangerous\b', r'\bunsafe\b', r'\bhazard\b',
        r'\bdeath\b', r'\bdied?\b', r'\bfatal\b', r'\bserious (harm|injury)\b',
    ],
    # Stability — checked before defects so "tipping/falling" wins over "broken"
    'Stability/Positioning Issues': [
        r'\bunstable\b', r'\btips? (over|easily)\b', r'\bfalls? over\b',
        r'\bwobble?s?\b', r'\bshifts? (out|around|constantly)\b',
        r'\bslides? (out|off|around)\b', r'\bwon[\']?t stay (in place|on|put)\b',
        r'\bkeeps? (moving|shifting|sliding|falling)\b',
    ],
    # Design/Material — size, comfort, fit, skin issues
    'Design/Material Issues': [
        r'\btoo (small|tight|narrow|short)\b',
        r'\btoo (big|large|wide|long|bulky|loose|baggy)\b',
        r"\bdoesn[\']?t fit\b", r'\bwon[\']?t fit\b', r'\bwrong (size|fit)\b',
        r'\bdoes not fit\b', r'\bfit (poorly|badly|incorrectly)\b',
        r'\bcauses? (pain|sores?|blisters?|bruising|chafing)\b',
        r'\bhurts?\b', r'\bpainful\b', r'\bsore\b', r'\bdigs? in\b', r'\buncomfortable\b',
        r'\btoo (hard|stiff|rigid|firm)\b', r'\btoo (soft|flimsy|weak|floppy)\b',
        r'\blacks? support\b', r'\bno support\b', r'\bcollapses?\b',
        r'\birritati(on|ng|es?)\b', r'\brash\b', r'\ballerg(y|ic)\b',
    ],
    # Product Defects/Quality — breaks, malfunctions, material, wrong product
    'Product Defects/Quality': [
        r'\bbroken\b', r'\bsnapped\b', r'\bcracked\b', r'\bfell? apart\b', r'\bdetached\b',
        r"\bdoesn[\']?t work\b", r'\bdoes not work\b', r'\bstopped? working\b', r'\bmalfunctions?\b',
        r'\bscratched\b', r'\bpaint (peeling|chipping)\b', r'\bpeeling\b',
        r'\bpoor (quality|material|construction)\b', r'\bcheap (material|plastic|fabric)\b',
        r'\blow quality\b', r'\bwrong (item|product|color|model)\b',
        r'\bnot as (described|advertised|shown)\b',
    ],
    # Missing Components
    'Missing Components': [
        r'\bmissing (part|piece|component|accessory|hardware|screw|bolt|nut)\b',
        r'\bincomplete\b', r'\bparts? (missing|absent|not included)\b', r'\bno instructions?\b',
        r'\bnot (all|everything) (included|in box)\b',
    ],
    # Performance/Effectiveness — doesn't help, compatibility, assembly
    'Performance/Effectiveness': [
        r'\bineffective\b', r"\bdoesn[\']?t help\b", r'\bnot effective\b', r'\buseless\b',
        r"\bdoesn[\']?t do anything\b",
        r'\bnot compatible (with|for)\b', r'\bincompatible\b',
        r"\bdoesn[\']?t (fit|work) (with|on|for) (my|the|a)\b",
        r'\bdifficult (to (assemble|use|adjust|put together|set up))\b',
        r'\bhard (to (assemble|use|put together|adjust))\b',
        r'\bconfusing instructions?\b', r'\bimpossible to (assemble|use)\b',
    ],
    # Non-quality
    'Customer: Changed Mind / No Longer Needed': [
        r'\bchanged (my )?mind\b', r'\bno longer (need|want|require)\b',
        r'\bdon[\']?t need (it|anymore|this)\b', r'\bnot needed\b',
        r'\bdecided (not to|against)\b', r'\bdonating\b',
    ],
    'Customer: Ordered Wrong Size or Item': [
        r'\bordered (the )?wrong (size|item|product)\b', r'\bmy (mistake|error|fault)\b',
        r'\baccidentally (ordered|bought|purchased)\b',
    ],
    'Fulfillment: Damaged in Shipping': [
        r'\bdamaged (in|during|by) (shipping|transit|delivery)\b',
        r'\barrived (damaged|broken|crushed|dented|wet)\b',
        r'\bbox (damaged|crushed|wet|torn)\b', r'\bshipping damage\b',
    ],
    'Fulfillment: Wrong Item Sent': [
        r'\bsent (the )?wrong (item|product|size|color)\b',
        r'\breceived (the )?wrong\b', r'\bpackaged incorrectly\b',
    ],
}

# Compile patterns for speed
COMPILED_PATTERNS = {
    category: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    for category, patterns in QUICK_PATTERNS.items()
}


class AIProvider(Enum):
    CLAUDE = "claude"
    CLAUDE_FAST = "claude_fast"
    CLAUDE_POWERFUL = "claude_powerful"
    FASTEST = "fastest"  # Auto-select fastest available model


@dataclass
class CostEstimate:
    """Cost estimation data class"""
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float

    def to_dict(self):
        return {
            'provider': self.provider,
            'model': self.model,
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'input_cost': self.input_cost,
            'output_cost': self.output_cost,
            'total_cost': self.total_cost
        }


def estimate_tokens(text: str) -> int:
    """Estimate token count (rough approximation)"""
    return max(len(text) // 4, len(text.split()) * 4 // 3)


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> CostEstimate:
    """Calculate cost for API usage"""
    if model not in PRICING:
        logger.warning(f"Model {model} not in pricing table")
        return CostEstimate("unknown", model, input_tokens, output_tokens, 0, 0, 0)

    pricing = PRICING[model]
    input_cost = (input_tokens / 1000) * pricing['input']
    output_cost = (output_tokens / 1000) * pricing['output']
    total_cost = input_cost + output_cost

    provider = 'claude'

    return CostEstimate(
        provider=provider,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost=total_cost
    )


def quick_categorize(complaint: str, fba_reason: str = None) -> Optional[str]:
    """Quick pattern-based categorization for speed"""
    if not complaint:
        return None

    if fba_reason and fba_reason in FBA_REASON_MAP:
        return FBA_REASON_MAP[fba_reason]

    complaint_lower = complaint.lower()

    for category, patterns in COMPILED_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(complaint_lower):
                return category

    return None


def detect_severity(complaint: str, category: str) -> str:
    """Detect severity level based on complaint text and unified category."""
    complaint_lower = complaint.lower()

    # Critical — safety/injury signals always override category
    critical_keywords = ['injury', 'injured', 'hospital', 'emergency', 'dangerous',
                         'unsafe', 'hazard', 'broke while', 'failed while', 'collapsed while']
    if any(kw in complaint_lower for kw in critical_keywords):
        return 'critical'
    if category == 'Medical/Health Concerns':
        return 'critical'

    # Major — functional failures and stability issues
    major_categories = {'Product Defects/Quality', 'Stability/Positioning Issues'}
    major_keywords = ['unusable', 'cannot use', "can't use", 'completely broken',
                      'fell apart', 'stopped working', 'failed', 'malfunction']
    if category in major_categories or any(kw in complaint_lower for kw in major_keywords):
        return 'major'

    # Moderate — fit, comfort, performance issues affecting usability
    moderate_categories = {'Design/Material Issues', 'Performance/Effectiveness'}
    if category in moderate_categories:
        return 'moderate'

    return 'minor'


class CostTracker:
    """Track API costs across sessions, including prompt cache savings."""

    def __init__(self):
        self.session_costs = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.api_calls = 0
        self.start_time = datetime.now()
        self.quick_categorizations = 0
        self.ai_categorizations = 0
        # Prompt caching metrics
        self.cache_read_tokens = 0       # Tokens served from cache (cheap)
        self.cache_creation_tokens = 0   # Tokens written to cache (first call)
        self.estimated_cache_savings = 0.0  # USD saved vs non-cached

    def add_cost(self, cost_estimate: CostEstimate):
        self.session_costs.append(cost_estimate)
        self.total_input_tokens += cost_estimate.input_tokens
        self.total_output_tokens += cost_estimate.output_tokens
        self.total_cost += cost_estimate.total_cost
        self.api_calls += 1

    def record_cache_usage(self, model: str, cache_read: int, cache_creation: int):
        """Record prompt cache hit/miss stats from API response usage block."""
        self.cache_read_tokens += cache_read
        self.cache_creation_tokens += cache_creation
        # Cache reads cost ~10% of normal input price — calculate savings
        pricing = PRICING.get(model, {})
        input_price = pricing.get('input', 0)
        if input_price and cache_read > 0:
            normal_cost = cache_read * input_price / 1000
            cache_cost = cache_read * input_price * 0.1 / 1000
            self.estimated_cache_savings += (normal_cost - cache_cost)

    def add_quick_categorization(self):
        self.quick_categorizations += 1

    def add_ai_categorization(self):
        self.ai_categorizations += 1

    def get_summary(self) -> Dict[str, Any]:
        duration = (datetime.now() - self.start_time).total_seconds() / 60
        return {
            'total_cost': round(self.total_cost, 4),
            'api_calls': self.api_calls,
            'total_tokens': self.total_input_tokens + self.total_output_tokens,
            'quick_categorizations': self.quick_categorizations,
            'ai_categorizations': self.ai_categorizations,
            'speed_improvement': f"{self.quick_categorizations / max(1, self.quick_categorizations + self.ai_categorizations) * 100:.1f}%",
            'average_cost_per_call': round(self.total_cost / max(1, self.api_calls), 4),
            'duration_minutes': round(duration, 1),
            'breakdown_by_provider': self._get_provider_breakdown(),
            'cache_read_tokens': self.cache_read_tokens,
            'cache_creation_tokens': self.cache_creation_tokens,
            'estimated_cache_savings': round(self.estimated_cache_savings, 4),
        }

    def _get_provider_breakdown(self) -> Dict[str, Dict]:
        breakdown = {'claude': {'calls': 0, 'cost': 0}}
        for cost in self.session_costs:
            provider = cost.provider
            if provider in breakdown:
                breakdown[provider]['calls'] += 1
                breakdown[provider]['cost'] += cost.total_cost
        return breakdown


class EnhancedAIAnalyzer:
    """
    Main AI analyzer — Claude (Anthropic) only.
    Uses direct HTTP requests (requests library).
    """

    def __init__(self, provider: AIProvider = AIProvider.CLAUDE, max_workers: int = 5):
        self.provider = provider
        self.max_workers = max_workers
        self.claude_key = self._get_api_key('claude')

        self.cost_tracker = CostTracker()

        self.claude_configured = bool(self.claude_key and has_requests)

        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

        self.session = None
        if has_requests:
            self.session = requests.Session()

        logger.info(
            f"AI Analyzer initialized — Claude: {self.claude_configured}, "
            f"Mode: {provider.value}, Workers: {self.max_workers}"
        )

    def _get_api_key(self, provider: str) -> Optional[str]:
        """Get API key from Streamlit secrets or environment variables."""
        # Streamlit secrets first
        try:
            import streamlit as st
            if hasattr(st, 'secrets'):
                for key_name in ["ANTHROPIC_API_KEY", "anthropic_api_key", "claude_api_key", "claude"]:
                    if key_name in st.secrets:
                        key_value = str(st.secrets[key_name]).strip()
                        if key_value:
                            logger.info(f"Found Claude key in Streamlit secrets ({key_name})")
                            return key_value
        except Exception as e:
            logger.debug(f"Streamlit secrets not available: {e}")

        # Environment variables fallback
        for env_name in ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"]:
            api_key = os.environ.get(env_name, '').strip()
            if api_key:
                logger.info(f"Found Claude key in environment ({env_name})")
                return api_key

        logger.warning("No Anthropic API key found")
        return None

    def get_api_status(self) -> Dict[str, Any]:
        status = {
            'available': self.claude_configured,
            'claude_configured': self.claude_configured,
            'primary_provider': 'claude',
            'provider': self.provider.value,
            'cost_summary': self.cost_tracker.get_summary(),
            'message': ''
        }
        if self.claude_configured:
            status['message'] = 'Claude API configured'
        else:
            status['message'] = 'No API configured — add ANTHROPIC_API_KEY to Streamlit secrets'
        return status

    # ----------------------------------------------------------------
    # Claude API call (direct HTTP — Anthropic Messages API)
    # ----------------------------------------------------------------
    def _call_claude(
        self,
        prompt: str,
        system_prompt: str,
        mode: str = 'standard',
        use_extended_thinking: bool = False,
        thinking_budget: int = 8000,
    ) -> Tuple[Optional[str], Optional[CostEstimate]]:
        """
        Call Anthropic Claude API with cost tracking.

        Features:
        - Prompt caching: system_prompt is cached after first call (saves ~90% on
          repeated calls with the same system prompt — no accuracy change)
        - Extended thinking: optional deep reasoning for complex analyses
        """
        if not self.claude_configured:
            return None, None

        model = MODELS['claude'].get(mode, MODELS['claude']['standard'])
        max_tokens = TOKEN_LIMITS.get(mode, TOKEN_LIMITS['standard'])
        input_tokens = estimate_tokens(system_prompt + prompt)

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.claude_key,
            "anthropic-version": "2023-06-01",
            # Enable prompt caching (stable, no accuracy impact)
            "anthropic-beta": "prompt-caching-2024-07-31",
        }

        # System prompt as list with cache_control — eligible for caching when
        # the text is >= 1024 tokens (Anthropic requirement). Short prompts are
        # passed through normally; the API ignores cache_control if too short.
        payload: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": prompt}],
        }

        # Extended thinking — uses a compatible model and larger token budget
        if use_extended_thinking:
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }
            # max_tokens must exceed thinking_budget
            payload["max_tokens"] = max(max_tokens, thinking_budget + 1000)

        for attempt in range(MAX_RETRIES):
            try:
                response = (self.session or requests).post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload,
                    timeout=API_TIMEOUT,
                )

                if response.status_code == 200:
                    result = response.json()

                    # Extract text from content blocks (may include thinking blocks)
                    content_blocks = result.get("content", [])
                    text_parts = [
                        b["text"] for b in content_blocks if b.get("type") == "text"
                    ]
                    content = " ".join(text_parts).strip()
                    if not content:
                        content = ""

                    usage = result.get("usage", {})
                    actual_input = usage.get("input_tokens", input_tokens)
                    actual_output = usage.get("output_tokens", len(content.split()))

                    # Track prompt cache stats
                    cache_read = usage.get("cache_read_input_tokens", 0)
                    cache_creation = usage.get("cache_creation_input_tokens", 0)
                    if cache_read or cache_creation:
                        self.cost_tracker.record_cache_usage(
                            model, cache_read, cache_creation
                        )

                    cost = calculate_cost(model, actual_input, actual_output)
                    self.cost_tracker.add_cost(cost)
                    return content, cost

                elif response.status_code == 429:
                    wait_time = min(2 ** attempt, 10)
                    logger.warning(f"Claude rate limited, waiting {wait_time}s")
                    time.sleep(wait_time)

                elif response.status_code == 529:
                    wait_time = min(2 ** attempt * 2, 20)
                    logger.warning(f"Claude overloaded, waiting {wait_time}s")
                    time.sleep(wait_time)

                else:
                    logger.error(
                        f"Claude API error {response.status_code}: {response.text[:200]}"
                    )
                    return None, None

            except Exception as e:
                logger.error(f"Claude call error: {e}")
                if attempt == MAX_RETRIES - 1:
                    return None, None
                time.sleep(1)

        return None, None

    def _call_claude_stream(
        self,
        prompt: str,
        system_prompt: str,
        mode: str = 'chat',
    ):
        """
        Stream Claude response via Server-Sent Events.
        Yields text chunks as they arrive — use with st.write_stream().

        Uses prompt caching on the system prompt automatically.
        """
        if not self.claude_configured:
            yield "AI not configured."
            return

        model = MODELS['claude'].get(mode, MODELS['claude']['chat'])
        max_tokens = max(TOKEN_LIMITS.get(mode, 1500), 1500)

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.claude_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "prompt-caching-2024-07-31",
        }

        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "stream": True,
            "system": [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": prompt}],
        }

        try:
            with (self.session or requests).post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
                timeout=90,
                stream=True,
            ) as response:
                if response.status_code != 200:
                    yield f"API error {response.status_code}"
                    return

                for raw_line in response.iter_lines():
                    if not raw_line:
                        continue
                    try:
                        line = (
                            raw_line.decode("utf-8", errors="replace")
                            if isinstance(raw_line, bytes)
                            else raw_line
                        )
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        data = json.loads(data_str)
                        if data.get("type") == "content_block_delta":
                            delta = data.get("delta", {})
                            if delta.get("type") == "text_delta":
                                yield delta.get("text", "")
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
                    except Exception as parse_exc:
                        logger.debug(f"SSE parse error: {parse_exc}")
                        continue
        except Exception as exc:
            yield f"\nStreaming error: {exc}"

    # ----------------------------------------------------------------
    # Internal routing helper
    # ----------------------------------------------------------------
    def _route_call(
        self, prompt: str, system_prompt: str, mode: str
    ) -> Tuple[Optional[str], Optional[CostEstimate]]:
        """Route API call to Claude."""
        p = self.provider

        if p == AIProvider.CLAUDE_FAST:
            return self._call_claude(prompt, system_prompt, 'fast')

        if p == AIProvider.CLAUDE_POWERFUL:
            return self._call_claude(prompt, system_prompt, 'powerful')

        if p == AIProvider.FASTEST:
            return self._call_claude(prompt, system_prompt, 'fast')

        # Default: Claude standard
        return self._call_claude(prompt, system_prompt, mode)

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------
    def generate_text(self, prompt: str, system_prompt: str, mode: str = 'chat') -> Optional[str]:
        """Generate a single response for general analysis or chat use cases."""
        response, _ = self._route_call(prompt, system_prompt, mode)
        return response

    def summarize_batch(self, items: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Summarize and categorize a batch of tickets for B2B reports."""
        categories_list = '\n'.join(f'  - {cat}' for cat in MEDICAL_DEVICE_CATEGORIES)
        system_prompt = f"""You are a medical device quality engineer analyzing customer service tickets.
For each ticket, provide TWO things separated by a pipe character (|):
1. CATEGORY: Assign exactly one category from the list below.
2. SUMMARY: A concise but complete description of the issue.

AVAILABLE CATEGORIES:
{categories_list}

DECISION RULES:
- Product Defects/Quality: broken, snapped, cracked, stopped working, malfunctions, poor material, wrong product, not as described.
- Performance/Effectiveness: product doesn't achieve its purpose, compatibility with equipment, assembly difficulty, confusing instructions.
- Missing Components: parts or accessories absent from box, incomplete package, no instructions.
- Design/Material Issues: size too small/large/doesn't fit, comfort problems (pain, pressure, too hard/soft), skin irritation, allergic reaction.
- Stability/Positioning Issues: product wobbles, tips over, shifts out of position, slides, won't stay in place.
- Medical/Health Concerns: injury, safety hazard, hospital visit, dangerous condition.
- Customer: Changed Mind / No Longer Needed: customer returning by choice, no product fault.
- Fulfillment: Damaged in Shipping: damaged by carrier in transit.
- Fulfillment: Wrong Item Sent: warehouse sent incorrect product.
- Other / Miscellaneous: no clear issue or general inquiry.

FORMAT YOUR RESPONSE EXACTLY AS: Category Name | Summary text here
Example: Product Defects/Quality | Scooter battery not holding charge after 3 months of use."""

        futures = []
        for item in items:
            prompt = f"Subject: {item.get('subject', '')}\nDetails: {item.get('details', '')}\nCATEGORY | SUMMARY:"
            future = self.executor.submit(self._call_claude, prompt, system_prompt, 'summary')
            futures.append((future, item))

        results = []
        for future, item in futures:
            summary = "Summary Unavailable"
            category = "Other / Miscellaneous"
            try:
                resp, _ = future.result(timeout=API_TIMEOUT)
                if resp:
                    if '|' in resp:
                        parts = resp.split('|', 1)
                        raw_cat = parts[0].strip()
                        summary = parts[1].strip() if len(parts) > 1 else resp
                        matched = self._clean_category_response(raw_cat)
                        if matched and matched != 'Other / Miscellaneous':
                            category = matched
                        else:
                            category = raw_cat if raw_cat in MEDICAL_DEVICE_CATEGORIES else 'Other / Miscellaneous'
                    else:
                        summary = resp
            except Exception as e:
                logger.error(f"Summary error: {e}")

            result_item = item.copy()
            result_item['summary'] = summary
            result_item['category'] = category
            results.append(result_item)

        return results

    def categorize_return(
        self, complaint: str, fba_reason: str = None, mode: str = 'standard'
    ) -> Tuple[str, float, str, str]:
        """Categorize return with speed optimization and learned corrections."""
        if not complaint or not complaint.strip():
            return 'Other/Miscellaneous', 0.1, 'none', 'en'

        # 1. Check persistent corrections memory — exact match = instant, free
        try:
            from corrections_memory import get_corrections_memory
            mem = get_corrections_memory()
            direct = mem.get_direct_match(complaint)
            if direct:
                self.cost_tracker.add_quick_categorization()
                severity = detect_severity(complaint, direct)
                return direct, 1.0, severity, 'en'
        except Exception:
            mem = None

        # 2. Quick pattern match
        quick_category = quick_categorize(complaint, fba_reason)
        if quick_category:
            self.cost_tracker.add_quick_categorization()
            severity = detect_severity(complaint, quick_category)
            return quick_category, 0.9, severity, 'en'

        # 3. AI categorization
        self.cost_tracker.add_ai_categorization()

        # Build few-shot block from corrections memory (injected into system prompt)
        few_shot_block = ""
        try:
            if mem is not None:
                few_shot_block = mem.build_few_shot_block()
        except Exception:
            pass

        system_prompt = f"""You are a medical device quality engineer with 15+ years of experience in returns analysis and CAPA investigations. Your job is to assign EXACTLY ONE category from the provided list to a customer return complaint.

DECISION RULES — read carefully before categorizing:
1. PRODUCT DEFECTS/QUALITY: Physical breaks (snapped, cracked, fell apart), malfunctions/stops working, poor material quality, cosmetic damage, product not as described/wrong product.
2. PERFORMANCE/EFFECTIVENESS: Product works mechanically but doesn't achieve its therapeutic purpose. Also: compatibility with other equipment, assembly difficulty, confusing instructions.
3. MISSING COMPONENTS: Parts, accessories, or instructions absent from the box.
4. DESIGN/MATERIAL ISSUES: Size problems (too small, too large, doesn't fit), comfort problems (causes pain, too hard, too soft, skin irritation, allergic reaction). These are design/material flaws, not defects.
5. STABILITY/POSITIONING ISSUES: Product slides, tips over, wobbles, shifts out of position, won't stay in place.
6. MEDICAL/HEALTH CONCERNS: Injury, safety hazard, hospital visit, dangerous condition. Always the highest priority category.
7. CUSTOMER CAUSED: Only use Customer categories when the customer explicitly states it was their own mistake or decision.

EXAMPLES (use these to calibrate your judgment):
- "Brace is way too small, couldn't get it past my knee" → Design/Material Issues
- "Way too big, slides right off my leg" → Design/Material Issues
- "Cuts into my skin after 20 minutes" → Design/Material Issues
- "Caused a rash on my arm after 2 days" → Design/Material Issues
- "Buckle snapped in half on first use" → Product Defects/Quality
- "Worked fine for a week then motor stopped" → Product Defects/Quality
- "Paint peeling, velcro wore out after 2 weeks" → Product Defects/Quality
- "Received a completely different product than ordered" → Product Defects/Quality
- "Doesn't attach to my rollator" → Performance/Effectiveness
- "Hard to assemble, instructions make no sense" → Performance/Effectiveness
- "Doesn't seem to help my knee pain at all" → Performance/Effectiveness
- "Missing the leg support piece" → Missing Components
- "Keeps sliding off the seat cushion" → Stability/Positioning Issues
- "Tipped over and I fell" → Medical/Health Concerns
- "I ordered the wrong size, my fault" → Customer: Ordered Wrong Size or Item
- "Decided I don't need it after all" → Customer: Changed Mind / No Longer Needed
- "Arrived with the frame bent from the box being crushed" → Fulfillment: Damaged in Shipping

Respond with ONLY the exact category name from the list. No explanation, no punctuation, no quotes."""

        # Append learned corrections block if available
        if few_shot_block:
            system_prompt = system_prompt + f"\n\n{few_shot_block}"

        categories_list = '\n'.join(f'  {cat}' for cat in MEDICAL_DEVICE_CATEGORIES)
        user_prompt = (
            f'AVAILABLE CATEGORIES:\n{categories_list}\n\n'
            f'COMPLAINT: "{complaint}"\n\n'
            f'CATEGORY:'
        )

        # Use standard mode (Sonnet) for AI categorization — Haiku misses nuance on size/comfort splits
        response, _ = self._route_call(user_prompt, system_prompt, 'standard')

        if response:
            category = self._clean_category_response(response)
            severity = detect_severity(complaint, category)
            return category, 0.85, severity, 'en'

        return 'Other/Miscellaneous', 0.3, 'none', 'en'

    def categorize_batch(
        self, complaints: List[Dict[str, Any]], mode: str = 'standard'
    ) -> List[Dict[str, Any]]:
        """Categorize multiple complaints in parallel for speed."""
        results = []
        futures = []

        for item in complaints:
            future = self.executor.submit(
                self.categorize_return,
                item.get('complaint', ''),
                item.get('fba_reason'),
                mode
            )
            futures.append((future, item))

        for future, item in futures:
            try:
                category, confidence, severity, language = future.result(timeout=API_TIMEOUT)
                result = item.copy()
                result.update({'category': category, 'confidence': confidence, 'severity': severity, 'language': language})
                results.append(result)
            except Exception as e:
                logger.error(f"Batch categorization error: {e}")
                result = item.copy()
                result.update({'category': 'Other/Miscellaneous', 'confidence': 0.1, 'severity': 'none', 'language': 'en'})
                results.append(result)

        return results

    def _clean_category_response(self, response: str) -> str:
        """
        Clean Claude's response to extract the exact category name.
        Handles: extra whitespace, quotes, common preamble phrases,
        and partial/prefix matches for the 'Prefix: Description' naming convention.
        """
        # Strip fences and quotes
        response = response.strip().strip('`').strip('"').strip("'").strip()

        # Strip common preamble phrases Claude may add despite instructions
        for prefix in ['Category:', 'The category is:', 'Answer:', 'Result:',
                       'Classification:', 'Return category:']:
            if response.lower().startswith(prefix.lower()):
                response = response[len(prefix):].strip()

        # Take only the first line (in case Claude added an explanation on line 2)
        response = response.splitlines()[0].strip()

        # 1. Exact match (case-insensitive)
        for valid_cat in MEDICAL_DEVICE_CATEGORIES:
            if response.lower() == valid_cat.lower():
                return valid_cat

        # 2. Substring match — valid category name appears anywhere in response
        response_lower = response.lower()
        for valid_cat in MEDICAL_DEVICE_CATEGORIES:
            if valid_cat.lower() in response_lower:
                return valid_cat

        # 3. Prefix match — for 'Size: Too Small' style names, match by prefix alone
        #    e.g. Claude says "Size: Too Small (the product was too tight)" → match prefix
        for valid_cat in MEDICAL_DEVICE_CATEGORIES:
            prefix = valid_cat.split(':')[0].strip().lower()
            suffix_words = [w for w in valid_cat.lower().split() if len(w) > 3]
            if prefix and prefix in response_lower:
                # Verify at least one meaningful word from full name also matches
                if any(w in response_lower for w in suffix_words):
                    return valid_cat

        # 4. Legacy name remapping — if old-style category comes back, upgrade it
        response_stripped = response.strip()
        if response_stripped in LEGACY_CATEGORY_MAP:
            return LEGACY_CATEGORY_MAP[response_stripped]

        # 5. Keyword overlap fallback
        for valid_cat in MEDICAL_DEVICE_CATEGORIES:
            # Use meaningful tokens only (>3 chars, ignore 'and', 'the', etc.)
            cat_tokens = {w for w in re.split(r'[\s/:\-]+', valid_cat.lower()) if len(w) > 3}
            resp_tokens = {w for w in re.split(r'[\s/:\-]+', response_lower) if len(w) > 3}
            overlap = cat_tokens & resp_tokens
            if len(overlap) >= 2:
                return valid_cat

        return 'Other / Miscellaneous'

    def get_cost_summary(self) -> Dict[str, Any]:
        return self.cost_tracker.get_summary()

    def estimate_remaining_cost(self, remaining_items: int) -> float:
        quick_rate = self.cost_tracker.quick_categorizations / max(
            1, self.cost_tracker.quick_categorizations + self.cost_tracker.ai_categorizations
        )
        ai_items = remaining_items * (1 - quick_rate)
        if self.cost_tracker.api_calls > 0:
            avg_cost = self.cost_tracker.total_cost / self.cost_tracker.api_calls
            return round(avg_cost * ai_items, 2)
        return 0.0

    def __del__(self):
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
        if hasattr(self, 'session') and self.session:
            self.session.close()


# ---------------------------------------------------------------------------
# Batch processing helper
# ---------------------------------------------------------------------------
def process_dataframe_in_batches(df, analyzer, batch_size=20):
    """Process dataframe in batches for speed."""
    total_rows = len(df)
    results = []

    for i in range(0, total_rows, batch_size):
        batch = df.iloc[i:i + batch_size]
        batch_data = []
        for idx, row in batch.iterrows():
            batch_data.append({
                'index': idx,
                'complaint': str(row.get('Complaint', '')),
                'fba_reason': str(row.get('FBA_Reason_Code', '')) if 'FBA_Reason_Code' in row else None
            })
        batch_results = analyzer.categorize_batch(batch_data)
        results.extend(batch_results)

    return results


# =============================================================================
# DEEP DIVE ANALYSIS — Investigation Method Recommendations
# =============================================================================

class DeepDiveAnalyzer:
    """Advanced AI analysis for flagged products with investigation recommendations."""

    def __init__(self, ai_analyzer: 'EnhancedAIAnalyzer'):
        self.ai = ai_analyzer
        self.investigation_methods = {
            '5_whys': {
                'name': '5 Whys Analysis',
                'best_for': 'Simple, linear problems with clear cause-effect relationships',
                'use_when': 'Problem has a clear starting point and you need to dig deep into root cause',
                'example': 'Product defect → Why? → Manufacturing issue → Why? → Machine calibration → Why? (repeat 5x)'
            },
            'fishbone': {
                'name': 'Fishbone Diagram (Ishikawa)',
                'best_for': 'Complex problems with multiple potential contributing factors',
                'use_when': 'Many possible causes from different categories (people, process, materials, equipment)',
                'example': 'Analyzing all potential causes of product quality issues across manufacturing, design, materials, etc.'
            },
            'rca': {
                'name': 'Root Cause Analysis (Formal RCA)',
                'best_for': 'Critical/high-impact issues requiring comprehensive investigation',
                'use_when': 'Safety concerns, regulatory issues, or high-value/high-volume problems',
                'example': 'Medical device failure with potential patient impact — requires full documentation'
            },
            'fmea': {
                'name': 'FMEA (Failure Mode Effects Analysis)',
                'best_for': 'Proactive risk assessment of potential failures',
                'use_when': 'New product launches, design changes, or preventing future issues',
                'example': 'Analyzing all ways a product could fail and prioritizing prevention efforts'
            },
            '8d': {
                'name': '8D Problem Solving',
                'best_for': 'Team-based problem solving with customer impact',
                'use_when': 'Customer complaints requiring cross-functional investigation and containment',
                'example': 'Batch quality issue affecting multiple customers — requires immediate containment + long-term fix'
            },
            'pareto': {
                'name': 'Pareto Analysis (80/20 Rule)',
                'best_for': 'Prioritizing which issues to tackle first',
                'use_when': 'Multiple issues and you need to focus resources on the biggest impact areas',
                'example': 'Identifying that 20% of defect types cause 80% of returns'
            }
        }

    def analyze_flagged_product(
        self, product_data: Dict[str, Any], product_docs: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Deep dive analysis of a flagged product with AI recommendations."""
        context_parts = [
            f"Product: {product_data.get('product_name', 'Unknown')}",
            f"SKU: {product_data.get('sku', 'Unknown')}",
            f"Category: {product_data.get('category', 'Unknown')}",
            f"Return Rate: {product_data.get('return_rate', 0):.1%}",
            f"Category Threshold: {product_data.get('category_threshold', 0):.1%}",
            f"Units Sold: {product_data.get('units_sold', 0):,}",
            f"Units Returned: {product_data.get('units_returned', 0):,}",
            f"Landed Cost: ${product_data.get('landed_cost', 0):.2f}",
        ]

        if product_data.get('triggers'):
            context_parts.append(f"Triggers: {', '.join(product_data['triggers'])}")

        doc_context = ""
        if product_docs:
            if 'manual' in product_docs:
                doc_context += f"\n\nProduct Manual Excerpt:\n{product_docs['manual'][:2000]}"
            if 'amazon_listing' in product_docs:
                doc_context += f"\n\nAmazon Listing:\n{product_docs['amazon_listing'][:1000]}"
            if 'ifu' in product_docs:
                doc_context += f"\n\nInstructions for Use:\n{product_docs['ifu'][:1000]}"

        prompt = f"""Analyze this flagged medical device product and provide investigation guidance:

{chr(10).join(context_parts)}
{doc_context}

Provide a comprehensive analysis with:

1. PROBLEM SUMMARY: What is the core issue based on the data?

2. RECOMMENDED INVESTIGATION METHOD: Choose the BEST investigation method from:
   - 5 Whys: For simple, linear problems
   - Fishbone Diagram: For complex, multi-factor problems
   - Formal RCA: For critical/safety issues
   - FMEA: For proactive risk assessment
   - 8D Problem Solving: For customer-facing issues requiring team response
   - Pareto Analysis: For prioritizing multiple issues

3. INTENDED USE QUESTIONS: What critical questions should we ask about:
   - How customers are actually using this product?
   - What is the intended vs actual use case?
   - Are there use environment factors we're missing?

4. KEY INVESTIGATION AREAS: What specific areas should the investigation focus on?

5. IMMEDIATE ACTIONS: What should happen right now?

6. RISK LEVEL: Rate the urgency (Low/Medium/High/Critical) and explain why.

Format your response as structured JSON."""

        system_prompt = (
            "You are a quality investigation expert. Analyze the provided quality issue and "
            "recommend the best investigation approach. Respond in JSON format."
        )

        try:
            response = self.ai.generate_text(prompt, system_prompt, mode='chat')
            if not response:
                raise Exception("No response from AI")

            try:
                analysis = json.loads(response)
            except json.JSONDecodeError:
                # Claude sometimes wraps JSON in markdown fences
                cleaned = re.sub(r'```(?:json)?\s*|\s*```', '', response).strip()
                try:
                    analysis = json.loads(cleaned)
                except json.JSONDecodeError:
                    analysis = {
                        'raw_analysis': response,
                        'recommended_method': self._extract_method_from_text(response),
                        'risk_level': self._extract_risk_level(response)
                    }
            return analysis

        except Exception as e:
            logger.error(f"Deep dive analysis failed: {e}")
            return {'error': str(e), 'recommended_method': 'rca', 'risk_level': 'Medium'}

    def _extract_method_from_text(self, text: str) -> str:
        text_lower = text.lower()
        for method_key in self.investigation_methods:
            if method_key.replace('_', ' ') in text_lower or \
               self.investigation_methods[method_key]['name'].lower() in text_lower:
                return method_key
        return 'rca'

    def _extract_risk_level(self, text: str) -> str:
        text_lower = text.lower()
        if 'critical' in text_lower or 'immediate' in text_lower or 'urgent' in text_lower:
            return 'Critical'
        elif 'high' in text_lower:
            return 'High'
        elif 'low' in text_lower:
            return 'Low'
        return 'Medium'

    def get_method_details(self, method_key: str) -> Dict[str, str]:
        return self.investigation_methods.get(method_key, self.investigation_methods['rca'])


# =============================================================================
# BULK OPERATIONS — Multiple Products
# =============================================================================

class BulkOperationsManager:
    """Handles bulk generation of vendor emails and investigation plans."""

    def __init__(self, vendor_email_generator, investigation_plan_generator):
        self.vendor_gen = vendor_email_generator
        self.investigation_gen = investigation_plan_generator

    def generate_bulk_vendor_emails(
        self, flagged_products: List[Dict[str, Any]], vendor_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        emails = []
        for product in flagged_products:
            try:
                email_result = self.vendor_gen.generate_email(
                    product_data=product, vendor_info=vendor_info
                )
                emails.append({
                    'sku': product.get('sku'),
                    'product_name': product.get('product_name'),
                    'subject': email_result.get('subject', ''),
                    'body': email_result.get('body', ''),
                    'priority': product.get('action', 'Monitor'),
                    'return_rate': product.get('return_rate', 0),
                    'units_affected': product.get('units_returned', 0)
                })
            except Exception as e:
                logger.error(f"Failed to generate email for {product.get('sku')}: {e}")
                emails.append({'sku': product.get('sku'), 'product_name': product.get('product_name'), 'error': str(e)})
        return emails

    def generate_bulk_investigation_plans(
        self, flagged_products: List[Dict[str, Any]], investigation_methods: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        plans = []
        for product in flagged_products:
            try:
                sku = product.get('sku')
                method = investigation_methods.get(sku) if investigation_methods else None
                plan_result = self.investigation_gen.generate_plan(
                    product_data=product, investigation_method=method
                )
                plans.append({
                    'sku': sku,
                    'product_name': product.get('product_name'),
                    'method': method or 'rca',
                    'plan': plan_result.get('plan', ''),
                    'timeline': plan_result.get('timeline', ''),
                    'team_required': plan_result.get('team', []),
                    'priority': product.get('action', 'Monitor')
                })
            except Exception as e:
                logger.error(f"Failed to generate plan for {product.get('sku')}: {e}")
                plans.append({'sku': product.get('sku'), 'product_name': product.get('product_name'), 'error': str(e)})
        return plans

    def export_bulk_emails_to_csv(self, emails: List[Dict[str, Any]]) -> str:
        import pandas as pd
        import io
        df = pd.DataFrame(emails)
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        return buf.getvalue()

    def export_bulk_plans_to_csv(self, plans: List[Dict[str, Any]]) -> str:
        import pandas as pd
        import io
        df = pd.DataFrame(plans)
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        return buf.getvalue()


# Export all components
__all__ = [
    'EnhancedAIAnalyzer',
    'AIProvider',
    'MEDICAL_DEVICE_CATEGORIES',
    'FBA_REASON_MAP',
    'detect_severity',
    'CostEstimate',
    'CostTracker',
    'estimate_tokens',
    'calculate_cost',
    'process_dataframe_in_batches',
    'quick_categorize',
    'DeepDiveAnalyzer',
    'BulkOperationsManager',
    'MODELS',
    'PRICING',
]
