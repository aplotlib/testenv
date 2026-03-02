"""
Enhanced AI Analysis Module - Claude-Primary with Speed Optimization
Version 16.0 - Claude API Migration

Key Features:
- Claude (Anthropic) as primary AI provider
- Dual AI support with intelligent routing
- Batch processing for speed
- Claude Haiku for fast categorization/summarization
- Claude Sonnet/Opus for complex cases
- Parallel API calls
- Dynamic Worker Scaling

Migration Notes:
- Migrated from OpenAI to Anthropic Claude as primary provider
- All Claude API calls use direct HTTP requests (requests library)
- Claude model strings updated to current Claude 4.x / 4.5 family
- OpenAI kept as optional fallback if key is present
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
    'standard': 100,
    'enhanced': 200,
    'extreme': 400,
    'chat': 1000,       # Increased for Claude's more verbose responses
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
    },
    # OpenAI kept as optional fallback
    'openai': {
        'fast':      'gpt-4o-mini',
        'standard':  'gpt-4o',
        'enhanced':  'gpt-4o',
        'extreme':   'gpt-4o',
        'powerful':  'gpt-4o',
        'chat':      'gpt-4o',
        'summary':   'gpt-4o-mini'
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
    # OpenAI — optional fallback
    'gpt-4o-mini':   {'input': 0.00015, 'output': 0.00060},
    'gpt-4o':        {'input': 0.00250, 'output': 0.01000},
    'gpt-3.5-turbo': {'input': 0.00050, 'output': 0.00150},
}

# Medical Device Return Categories
MEDICAL_DEVICE_CATEGORIES = [
    'Size/Fit Issues',
    'Comfort Issues',
    'Product Defects/Quality',
    'Performance/Effectiveness',
    'Stability/Positioning Issues',
    'Equipment Compatibility',
    'Design/Material Issues',
    'Wrong Product/Misunderstanding',
    'Missing Components',
    'Customer Error/Changed Mind',
    'Shipping/Fulfillment Issues',
    'Assembly/Usage Difficulty',
    'Medical/Health Concerns',
    'Price/Value',
    'Other/Miscellaneous'
]

# FBA reason code mapping
FBA_REASON_MAP = {
    'NOT_COMPATIBLE': 'Equipment Compatibility',
    'DAMAGED_BY_FC': 'Product Defects/Quality',
    'DAMAGED_BY_CARRIER': 'Shipping/Fulfillment Issues',
    'DEFECTIVE': 'Product Defects/Quality',
    'NOT_AS_DESCRIBED': 'Wrong Product/Misunderstanding',
    'WRONG_ITEM': 'Wrong Product/Misunderstanding',
    'MISSING_PARTS': 'Missing Components',
    'QUALITY_NOT_ADEQUATE': 'Product Defects/Quality',
    'UNWANTED_ITEM': 'Customer Error/Changed Mind',
    'UNAUTHORIZED_PURCHASE': 'Customer Error/Changed Mind',
    'CUSTOMER_DAMAGED': 'Customer Error/Changed Mind',
    'SWITCHEROO': 'Wrong Product/Misunderstanding',
    'EXPIRED_ITEM': 'Product Defects/Quality',
    'DAMAGED_GLASS_VIAL': 'Product Defects/Quality',
    'DIFFERENT_PRODUCT': 'Wrong Product/Misunderstanding',
    'MISSING_ITEM': 'Missing Components',
    'NOT_DELIVERED': 'Shipping/Fulfillment Issues',
    'ORDERED_WRONG_ITEM': 'Customer Error/Changed Mind',
    'UNNEEDED_ITEM': 'Customer Error/Changed Mind',
    'BAD_GIFT': 'Customer Error/Changed Mind',
    'INACCURATE_WEBSITE_DESCRIPTION': 'Wrong Product/Misunderstanding',
    'BETTER_PRICE_AVAILABLE': 'Price/Value',
    'DOES_NOT_FIT': 'Size/Fit Issues',
    'NOT_COMPATIBLE_WITH_DEVICE': 'Equipment Compatibility',
    'UNSATISFACTORY_PRODUCT': 'Performance/Effectiveness',
    'ARRIVED_LATE': 'Shipping/Fulfillment Issues',
    'TOO_SMALL': 'Size/Fit Issues',
    'TOO_LARGE': 'Size/Fit Issues',
    'UNCOMFORTABLE': 'Comfort Issues',
    'DIFFICULT_TO_USE': 'Assembly/Usage Difficulty',
    'DAMAGED': 'Product Defects/Quality',
    'BROKEN': 'Product Defects/Quality',
    'POOR_QUALITY': 'Product Defects/Quality',
    'NOT_WORKING': 'Product Defects/Quality',
    'DOESNT_WORK': 'Product Defects/Quality'
}

# Quick categorization patterns for speed
QUICK_PATTERNS = {
    'Size/Fit Issues': [
        r'too (small|large|big|tight|loose)', r'doesn[\']?t fit', r'wrong size',
        r'size issue', r'(small|large)r than expected', r'fit issue'
    ],
    'Product Defects/Quality': [
        r'defect', r'broken', r'damaged', r'poor quality', r'doesn[\']?t work',
        r'stopped working', r'malfunction', r'fell apart', r'ripped', r'torn'
    ],
    'Wrong Product/Misunderstanding': [
        r'wrong (item|product)', r'not as described', r'different than',
        r'thought it was', r'expected', r'not what I ordered'
    ],
    'Customer Error/Changed Mind': [
        r'changed mind', r'don[\']?t need', r'ordered by mistake',
        r'accidentally', r'no longer need', r'bought wrong'
    ],
    'Comfort Issues': [
        r'uncomfort', r'hurts', r'painful', r'too (hard|soft|firm)',
        r'causes pain', r'irritat'
    ],
    'Equipment Compatibility': [
        r'doesn[\']?t fit (my|the)', r'not compatible', r'incompatible',
        r'doesn[\']?t work with', r'won[\']?t fit'
    ]
}

# Compile patterns for speed
COMPILED_PATTERNS = {
    category: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    for category, patterns in QUICK_PATTERNS.items()
}


class AIProvider(Enum):
    OPENAI = "openai"
    OPENAI_FAST = "openai_fast"
    OPENAI_POWERFUL = "openai_powerful"
    CLAUDE = "claude"
    CLAUDE_FAST = "claude_fast"
    CLAUDE_POWERFUL = "claude_powerful"
    BOTH = "both"
    FASTEST = "fastest"  # Auto-select fastest available (Claude-first)


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

    provider = 'claude' if 'claude' in model else 'openai'

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
    """Detect severity level of complaint"""
    complaint_lower = complaint.lower()

    critical_keywords = ['injury', 'injured', 'hospital', 'emergency', 'dangerous', 'unsafe', 'hazard']
    if any(keyword in complaint_lower for keyword in critical_keywords):
        return 'critical'

    if category == 'Medical/Health Concerns':
        return 'critical'

    major_keywords = ['defective', 'broken', 'malfunction', 'unusable', 'failed', 'stopped working']
    if any(keyword in complaint_lower for keyword in major_keywords):
        return 'major'

    if category in ['Product Defects/Quality', 'Performance/Effectiveness']:
        return 'major'

    return 'minor'


class CostTracker:
    """Track API costs across sessions"""

    def __init__(self):
        self.session_costs = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.api_calls = 0
        self.start_time = datetime.now()
        self.quick_categorizations = 0
        self.ai_categorizations = 0

    def add_cost(self, cost_estimate: CostEstimate):
        self.session_costs.append(cost_estimate)
        self.total_input_tokens += cost_estimate.input_tokens
        self.total_output_tokens += cost_estimate.output_tokens
        self.total_cost += cost_estimate.total_cost
        self.api_calls += 1

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
            'breakdown_by_provider': self._get_provider_breakdown()
        }

    def _get_provider_breakdown(self) -> Dict[str, Dict]:
        breakdown = {'openai': {'calls': 0, 'cost': 0}, 'claude': {'calls': 0, 'cost': 0}}
        for cost in self.session_costs:
            provider = cost.provider
            if provider in breakdown:
                breakdown[provider]['calls'] += 1
                breakdown[provider]['cost'] += cost.total_cost
        return breakdown


class EnhancedAIAnalyzer:
    """
    Main AI analyzer — Claude (Anthropic) primary, OpenAI optional fallback.
    Uses direct HTTP requests (requests library) for both providers.
    """

    def __init__(self, provider: AIProvider = AIProvider.CLAUDE, max_workers: int = 5):
        self.provider = provider
        self.max_workers = max_workers
        self.claude_key = self._get_api_key('claude')
        self.openai_key = self._get_api_key('openai')

        self.cost_tracker = CostTracker()

        self.claude_configured = bool(self.claude_key and has_requests)
        self.openai_configured = bool(self.openai_key and has_requests)

        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

        self.session = None
        if has_requests:
            self.session = requests.Session()

        logger.info(
            f"AI Analyzer initialized — Claude: {self.claude_configured}, "
            f"OpenAI: {self.openai_configured}, Mode: {provider.value}, Workers: {self.max_workers}"
        )

    def _get_api_key(self, provider: str) -> Optional[str]:
        """Get API key from Streamlit secrets or environment variables."""
        # Streamlit secrets first
        try:
            import streamlit as st
            if hasattr(st, 'secrets'):
                if provider == 'claude':
                    for key_name in ["ANTHROPIC_API_KEY", "anthropic_api_key", "claude_api_key", "claude"]:
                        if key_name in st.secrets:
                            key_value = str(st.secrets[key_name]).strip()
                            if key_value:
                                logger.info(f"Found Claude key in Streamlit secrets ({key_name})")
                                return key_value
                elif provider == 'openai':
                    for key_name in ["OPENAI_API_KEY", "openai_api_key", "openai"]:
                        if key_name in st.secrets:
                            key_value = str(st.secrets[key_name]).strip()
                            if key_value and key_value.startswith('sk-'):
                                logger.info(f"Found OpenAI key in Streamlit secrets")
                                return key_value
        except Exception as e:
            logger.debug(f"Streamlit secrets not available: {e}")

        # Environment variables fallback
        env_vars = {
            'claude': ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"],
            'openai': ["OPENAI_API_KEY", "OPENAI_API"],
        }
        for env_name in env_vars.get(provider, []):
            api_key = os.environ.get(env_name, '').strip()
            if api_key:
                logger.info(f"Found {provider} key in environment ({env_name})")
                return api_key

        logger.warning(f"No {provider} API key found")
        return None

    def get_api_status(self) -> Dict[str, Any]:
        status = {
            'available': self.claude_configured or self.openai_configured,
            'claude_configured': self.claude_configured,
            'openai_configured': self.openai_configured,
            'primary_provider': 'claude',
            'provider': self.provider.value,
            'cost_summary': self.cost_tracker.get_summary(),
            'message': ''
        }
        if self.claude_configured and self.openai_configured:
            status['message'] = 'Claude (primary) and OpenAI (fallback) both configured'
        elif self.claude_configured:
            status['message'] = 'Claude API configured'
        elif self.openai_configured:
            status['message'] = 'OpenAI API configured (Claude key missing — add ANTHROPIC_API_KEY to secrets)'
        else:
            status['message'] = 'No APIs configured — add ANTHROPIC_API_KEY to Streamlit secrets'
        return status

    # ----------------------------------------------------------------
    # Claude API call (direct HTTP — Anthropic Messages API)
    # ----------------------------------------------------------------
    def _call_claude(
        self, prompt: str, system_prompt: str, mode: str = 'standard'
    ) -> Tuple[Optional[str], Optional[CostEstimate]]:
        """Call Anthropic Claude API with cost tracking."""
        if not self.claude_configured:
            return None, None

        model = MODELS['claude'].get(mode, MODELS['claude']['standard'])
        max_tokens = TOKEN_LIMITS.get(mode, TOKEN_LIMITS['standard'])
        input_tokens = estimate_tokens(system_prompt + prompt)

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.claude_key,
            "anthropic-version": "2023-06-01"
        }

        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": prompt}]
        }

        for attempt in range(MAX_RETRIES):
            try:
                response = (self.session or requests).post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload,
                    timeout=API_TIMEOUT
                )

                if response.status_code == 200:
                    result = response.json()
                    content = result["content"][0]["text"].strip()

                    usage = result.get("usage", {})
                    actual_input = usage.get("input_tokens", input_tokens)
                    actual_output = usage.get("output_tokens", len(content.split()))

                    cost = calculate_cost(model, actual_input, actual_output)
                    self.cost_tracker.add_cost(cost)
                    return content, cost

                elif response.status_code == 429:
                    wait_time = min(2 ** attempt, 10)
                    logger.warning(f"Claude rate limited, waiting {wait_time}s")
                    time.sleep(wait_time)

                elif response.status_code == 529:
                    # Anthropic overloaded
                    wait_time = min(2 ** attempt * 2, 20)
                    logger.warning(f"Claude overloaded, waiting {wait_time}s")
                    time.sleep(wait_time)

                else:
                    logger.error(f"Claude API error {response.status_code}: {response.text[:200]}")
                    return None, None

            except Exception as e:
                logger.error(f"Claude call error: {e}")
                if attempt == MAX_RETRIES - 1:
                    return None, None
                time.sleep(1)

        return None, None

    # ----------------------------------------------------------------
    # OpenAI API call — optional fallback
    # ----------------------------------------------------------------
    def _call_openai(
        self, prompt: str, system_prompt: str, mode: str = 'standard'
    ) -> Tuple[Optional[str], Optional[CostEstimate]]:
        """Call OpenAI API (optional fallback) with cost tracking."""
        if not self.openai_configured:
            return None, None

        model = MODELS['openai'].get(mode, MODELS['openai']['standard'])
        max_tokens = TOKEN_LIMITS.get(mode, TOKEN_LIMITS['standard'])
        input_tokens = estimate_tokens(system_prompt + prompt)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_key}"
        }

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": max_tokens
        }

        for attempt in range(MAX_RETRIES):
            try:
                response = (self.session or requests).post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=API_TIMEOUT
                )

                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"].strip()

                    usage = result.get("usage", {})
                    actual_input = usage.get("prompt_tokens", input_tokens)
                    actual_output = usage.get("completion_tokens", len(content.split()))

                    cost = calculate_cost(model, actual_input, actual_output)
                    self.cost_tracker.add_cost(cost)
                    return content, cost

                elif response.status_code == 429:
                    wait_time = min(2 ** attempt, 10)
                    logger.warning(f"OpenAI rate limited, waiting {wait_time}s")
                    time.sleep(wait_time)

                else:
                    logger.error(f"OpenAI API error {response.status_code}")
                    return None, None

            except Exception as e:
                logger.error(f"OpenAI call error: {e}")
                if attempt == MAX_RETRIES - 1:
                    return None, None
                time.sleep(1)

        return None, None

    # ----------------------------------------------------------------
    # Internal routing helper
    # ----------------------------------------------------------------
    def _route_call(
        self, prompt: str, system_prompt: str, mode: str
    ) -> Tuple[Optional[str], Optional[CostEstimate]]:
        """Route API call to the configured provider, Claude-first."""
        p = self.provider

        if p in (AIProvider.CLAUDE, AIProvider.CLAUDE_FAST):
            eff_mode = 'fast' if p == AIProvider.CLAUDE_FAST else mode
            return self._call_claude(prompt, system_prompt, eff_mode)

        if p == AIProvider.CLAUDE_POWERFUL:
            return self._call_claude(prompt, system_prompt, 'powerful')

        if p in (AIProvider.OPENAI, AIProvider.OPENAI_FAST):
            eff_mode = 'fast' if p == AIProvider.OPENAI_FAST else mode
            return self._call_openai(prompt, system_prompt, eff_mode)

        if p == AIProvider.OPENAI_POWERFUL:
            return self._call_openai(prompt, system_prompt, 'powerful')

        if p == AIProvider.FASTEST:
            # Claude-first
            if self.claude_configured:
                result, cost = self._call_claude(prompt, system_prompt, 'fast')
                if result:
                    return result, cost
            if self.openai_configured:
                return self._call_openai(prompt, system_prompt, 'fast')
            return None, None

        if p == AIProvider.BOTH:
            # Parallel calls, take longer/better response
            futures = []
            if self.claude_configured:
                futures.append(self.executor.submit(self._call_claude, prompt, system_prompt, mode))
            if self.openai_configured:
                futures.append(self.executor.submit(self._call_openai, prompt, system_prompt, mode))

            results = []
            for future in as_completed(futures, timeout=API_TIMEOUT):
                try:
                    resp, cost = future.result()
                    if resp:
                        results.append((resp, cost))
                except Exception as e:
                    logger.error(f"Parallel call failed: {e}")

            if results:
                return max(results, key=lambda x: len(x[0]))
            return None, None

        # Default: Claude
        return self._call_claude(prompt, system_prompt, mode)

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------
    def generate_text(self, prompt: str, system_prompt: str, mode: str = 'chat') -> Optional[str]:
        """Generate a single response for general analysis or chat use cases."""
        response, _ = self._route_call(prompt, system_prompt, mode)
        return response

    def summarize_batch(self, items: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Summarize a batch of tickets for B2B reports."""
        system_prompt = (
            "You are a customer service analyst. Summarize the return/replacement reason. "
            "Provide an accurate, detailed description of the 'Why' (e.g., 'Product defective', "
            "'Customer changed mind', 'Wrong item sent'). Use as many words as needed to fully "
            "capture the issue."
        )

        futures = []
        for item in items:
            prompt = f"Subject: {item.get('subject', '')}\nDetails: {item.get('details', '')}\nSummary:"
            future = self.executor.submit(self._call_claude, prompt, system_prompt, 'summary')
            futures.append((future, item))

        results = []
        for future, item in futures:
            summary = "Summary Unavailable"
            try:
                resp, _ = future.result(timeout=API_TIMEOUT)
                if resp:
                    summary = resp
            except Exception as e:
                logger.error(f"Summary error: {e}")

            result_item = item.copy()
            result_item['summary'] = summary
            results.append(result_item)

        return results

    def categorize_return(
        self, complaint: str, fba_reason: str = None, mode: str = 'standard'
    ) -> Tuple[str, float, str, str]:
        """Categorize return with speed optimization."""
        if not complaint or not complaint.strip():
            return 'Other/Miscellaneous', 0.1, 'none', 'en'

        # Quick pattern match first
        quick_category = quick_categorize(complaint, fba_reason)
        if quick_category:
            self.cost_tracker.add_quick_categorization()
            severity = detect_severity(complaint, quick_category)
            return quick_category, 0.9, severity, 'en'

        self.cost_tracker.add_ai_categorization()

        system_prompt = (
            "You are a medical device quality expert. Categorize this return into exactly one "
            "category from the provided list. Respond with ONLY the category name, nothing else."
        )

        categories_list = '\n'.join(f'- {cat}' for cat in MEDICAL_DEVICE_CATEGORIES)
        user_prompt = f'Complaint: "{complaint}"\n\nCategories:\n{categories_list}\n\nCategory:'

        response, _ = self._route_call(user_prompt, system_prompt, mode)

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
        """Clean AI response to extract category name."""
        response = response.strip().strip('"').strip("'")

        for prefix in ['Category:', 'The category is:', 'Answer:']:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()

        for valid_cat in MEDICAL_DEVICE_CATEGORIES:
            if response.lower() == valid_cat.lower():
                return valid_cat

        response_lower = response.lower()
        for valid_cat in MEDICAL_DEVICE_CATEGORIES:
            if valid_cat.lower() in response_lower:
                return valid_cat

        for valid_cat in MEDICAL_DEVICE_CATEGORIES:
            cat_words = set(valid_cat.lower().split('/'))
            response_words = set(response_lower.split())
            if cat_words & response_words:
                return valid_cat

        return 'Other/Miscellaneous'

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
