"""
Enhanced AI Analysis Module - With Cost Tracking & Chat Support
Version 13.0 - OpenAI + Claude with Cost Estimation

Key Features:
- Dual AI support (OpenAI GPT-4 + Claude Sonnet)
- Real-time cost tracking and estimation
- Chat support for Q&A about results
- Token usage optimization
- Quality pattern recognition
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
MAX_RETRIES = 3
BATCH_SIZE = 50

# Token configurations by mode
TOKEN_LIMITS = {
    'standard': 150,     # Reduced for cost savings
    'enhanced': 300,     # Moderate
    'extreme': 800,      # High accuracy
    'chat': 500          # For chat responses
}

# Model configurations with latest pricing (as of 2024)
MODELS = {
    'openai': {
        'standard': 'gpt-3.5-turbo',
        'enhanced': 'gpt-4',
        'extreme': 'gpt-4',
        'chat': 'gpt-3.5-turbo'
    },
    'claude': {
        'standard': 'claude-3-haiku-20240307',
        'enhanced': 'claude-3-sonnet-20240229',
        'extreme': 'claude-3-sonnet-20240229',
        'chat': 'claude-3-haiku-20240307'
    }
}

# Updated pricing per 1K tokens (in USD)
PRICING = {
    # OpenAI
    'gpt-3.5-turbo': {'input': 0.0005, 'output': 0.0015},
    'gpt-4': {'input': 0.03, 'output': 0.06},
    'gpt-4-turbo': {'input': 0.01, 'output': 0.03},
    # Claude (Anthropic)
    'claude-3-haiku-20240307': {'input': 0.00025, 'output': 0.00125},
    'claude-3-sonnet-20240229': {'input': 0.003, 'output': 0.015},
    'claude-3-opus-20240229': {'input': 0.015, 'output': 0.075}
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
    'ARRIVED_LATE': 'Shipping/Fulfillment Issues'
}

# Quality patterns
QUALITY_PATTERNS = {
    'Material Failure': [
        'velcro', 'strap', 'fabric', 'material', 'stitching', 'seam', 'tear', 'rip', 'worn'
    ],
    'Component Failure': [
        'button', 'buckle', 'wheel', 'handle', 'pump', 'valve', 'motor', 'battery'
    ],
    'Design Flaw': [
        'too heavy', 'hard to use', 'difficult to adjust', 'poor design', 'awkward'
    ],
    'Manufacturing Defect': [
        'broken on arrival', 'defective', 'missing parts', 'not assembled correctly'
    ],
    'Durability Issue': [
        'broke after', 'lasted only', 'stopped working', 'fell apart', 'wore out'
    ],
    'Safety Concern': [
        'unsafe', 'dangerous', 'injury', 'hurt', 'accident', 'risk', 'hazard'
    ]
}

# Severity keywords
SEVERITY_KEYWORDS = {
    'critical': [
        'injury', 'injured', 'hospital', 'emergency', 'doctor', 'dangerous', 'unsafe'
    ],
    'major': [
        'defective', 'broken', 'malfunction', 'unusable', 'failed', 'stopped working'
    ],
    'minor': [
        'uncomfortable', 'difficult', 'confusing', 'disappointed', 'not ideal'
    ]
}

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

class AIProvider(Enum):
    OPENAI = "openai"
    CLAUDE = "claude"
    BOTH = "both"

def estimate_tokens(text: str) -> int:
    """Estimate token count (rough approximation)"""
    # Rough estimate: 1 token ≈ 4 characters or 0.75 words
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
    
    provider = 'openai' if 'gpt' in model else 'claude'
    
    return CostEstimate(
        provider=provider,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost=total_cost
    )

def detect_language(text: str) -> str:
    """Simple language detection"""
    spanish_indicators = ['ñ', 'á', 'é', 'í', 'ó', 'ú', ' es ', ' la ', ' el ']
    spanish_count = sum(1 for indicator in spanish_indicators if indicator in text.lower())
    return 'es' if spanish_count >= 3 else 'en'

def calculate_confidence(complaint: str, category: str, language: str = 'en', consensus: bool = False) -> float:
    """Calculate confidence score for categorization"""
    confidence = 0.5
    
    if consensus:
        confidence = 0.95
    else:
        confidence = 0.85
    
    word_count = len(complaint.split())
    if word_count > 20:
        confidence += 0.05
    elif word_count < 5:
        confidence -= 0.1
    
    if category != 'Other/Miscellaneous':
        confidence += 0.05
    else:
        confidence -= 0.1
    
    if language != 'en':
        confidence -= 0.05
    
    return max(0.1, min(1.0, confidence))

def detect_severity(complaint: str, category: str) -> str:
    """Detect severity level of complaint"""
    complaint_lower = complaint.lower()
    
    for keyword in SEVERITY_KEYWORDS['critical']:
        if keyword in complaint_lower:
            return 'critical'
    
    if category == 'Medical/Health Concerns':
        return 'critical'
    
    for keyword in SEVERITY_KEYWORDS['major']:
        if keyword in complaint_lower:
            return 'major'
    
    if category in ['Product Defects/Quality', 'Performance/Effectiveness']:
        return 'major'
    
    for keyword in SEVERITY_KEYWORDS['minor']:
        if keyword in complaint_lower:
            return 'minor'
    
    return 'none'

def extract_quality_patterns(complaint: str, category: str) -> Dict[str, Any]:
    """Extract quality patterns from complaints"""
    patterns_found = []
    complaint_lower = complaint.lower()
    
    quality_categories = [
        'Product Defects/Quality',
        'Performance/Effectiveness',
        'Design/Material Issues',
        'Stability/Positioning Issues',
        'Medical/Health Concerns'
    ]
    
    if category not in quality_categories:
        return {'patterns': [], 'root_cause': None, 'is_safety_critical': False}
    
    for pattern_name, keywords in QUALITY_PATTERNS.items():
        for keyword in keywords:
            if keyword in complaint_lower:
                patterns_found.append({
                    'pattern': pattern_name,
                    'keyword': keyword,
                    'position': complaint_lower.find(keyword)
                })
    
    root_cause = None
    if patterns_found:
        patterns_found.sort(key=lambda x: x['position'])
        root_cause = patterns_found[0]['pattern']
    
    is_safety_critical = any(p['pattern'] == 'Safety Concern' for p in patterns_found)
    
    return {
        'patterns': patterns_found,
        'root_cause': root_cause,
        'is_safety_critical': is_safety_critical
    }

class CostTracker:
    """Track API costs across sessions"""
    
    def __init__(self):
        self.session_costs = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.api_calls = 0
        self.start_time = datetime.now()
    
    def add_cost(self, cost_estimate: CostEstimate):
        """Add cost to tracking"""
        self.session_costs.append(cost_estimate)
        self.total_input_tokens += cost_estimate.input_tokens
        self.total_output_tokens += cost_estimate.output_tokens
        self.total_cost += cost_estimate.total_cost
        self.api_calls += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get cost summary"""
        duration = (datetime.now() - self.start_time).total_seconds() / 60  # minutes
        
        return {
            'total_cost': round(self.total_cost, 4),
            'api_calls': self.api_calls,
            'total_tokens': self.total_input_tokens + self.total_output_tokens,
            'input_tokens': self.total_input_tokens,
            'output_tokens': self.total_output_tokens,
            'average_cost_per_call': round(self.total_cost / max(1, self.api_calls), 4),
            'duration_minutes': round(duration, 1),
            'breakdown_by_provider': self._get_provider_breakdown()
        }
    
    def _get_provider_breakdown(self) -> Dict[str, Dict]:
        """Get cost breakdown by provider"""
        breakdown = {'openai': {'calls': 0, 'cost': 0}, 'claude': {'calls': 0, 'cost': 0}}
        
        for cost in self.session_costs:
            provider = cost.provider
            if provider in breakdown:
                breakdown[provider]['calls'] += 1
                breakdown[provider]['cost'] += cost.total_cost
        
        return breakdown
    
    def estimate_remaining_cost(self, remaining_items: int) -> float:
        """Estimate cost for remaining items"""
        if self.api_calls == 0:
            return 0.0
        
        avg_cost = self.total_cost / self.api_calls
        return round(avg_cost * remaining_items, 2)

class EnhancedAIAnalyzer:
    """Main AI analyzer with cost tracking and chat support"""
    
    def __init__(self, provider: AIProvider = AIProvider.BOTH):
        self.provider = provider
        self.openai_key = self._get_api_key('openai')
        self.claude_key = self._get_api_key('claude')
        
        # Initialize tracking
        self.cost_tracker = CostTracker()
        
        # Initialize API availability
        self.openai_configured = bool(self.openai_key and has_requests)
        self.claude_configured = bool(self.claude_key and has_requests)
        
        # Chat context
        self.chat_context = []
        
        logger.info(f"AI Analyzer initialized - OpenAI: {self.openai_configured}, Claude: {self.claude_configured}")
    
    def _get_api_key(self, provider: str) -> Optional[str]:
        """Get API key from multiple sources"""
        # Try Streamlit secrets first
        try:
            import streamlit as st
            if hasattr(st, 'secrets'):
                if provider == 'openai':
                    for key_name in ["OPENAI_API_KEY", "openai_api_key", "openai"]:
                        if key_name in st.secrets:
                            key_value = str(st.secrets[key_name]).strip()
                            if key_value and (provider == 'openai' and key_value.startswith('sk-')):
                                logger.info(f"Found {provider} key in Streamlit secrets")
                                return key_value
                elif provider == 'claude':
                    for key_name in ["ANTHROPIC_API_KEY", "anthropic_api_key", "claude_api_key", "claude"]:
                        if key_name in st.secrets:
                            key_value = str(st.secrets[key_name]).strip()
                            if key_value and (provider == 'claude' and 'ant' in key_value):
                                logger.info(f"Found {provider} key in Streamlit secrets")
                                return key_value
        except Exception as e:
            logger.debug(f"Streamlit secrets not available: {e}")
        
        # Try environment variables
        env_vars = {
            'openai': ["OPENAI_API_KEY", "OPENAI_API"],
            'claude': ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"]
        }
        
        for env_name in env_vars.get(provider, []):
            api_key = os.environ.get(env_name, '').strip()
            if api_key:
                logger.info(f"Found {provider} key in environment")
                return api_key
        
        logger.warning(f"No {provider} API key found")
        return None
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get API status with cost summary"""
        status = {
            'available': self.openai_configured or self.claude_configured,
            'openai_configured': self.openai_configured,
            'claude_configured': self.claude_configured,
            'dual_ai_available': self.openai_configured and self.claude_configured,
            'provider': self.provider.value,
            'cost_summary': self.cost_tracker.get_summary(),
            'message': ''
        }
        
        if status['dual_ai_available']:
            status['message'] = 'Both OpenAI and Claude APIs are configured'
        elif self.openai_configured:
            status['message'] = 'OpenAI API is configured (Claude not available)'
        elif self.claude_configured:
            status['message'] = 'Claude API is configured (OpenAI not available)'
        else:
            status['message'] = 'No APIs configured'
        
        return status
    
    def estimate_analysis_cost(self, num_items: int, mode: str = 'standard') -> Dict[str, float]:
        """Estimate cost before running analysis"""
        # Average tokens per complaint analysis
        avg_input_tokens = 200 + (50 if mode == 'enhanced' else 0) + (100 if mode == 'extreme' else 0)
        avg_output_tokens = TOKEN_LIMITS[mode] // 2  # Assume half of limit used
        
        estimates = {}
        
        if self.openai_configured:
            model = MODELS['openai'][mode]
            cost = calculate_cost(model, avg_input_tokens * num_items, avg_output_tokens * num_items)
            estimates['openai'] = round(cost.total_cost, 2)
        
        if self.claude_configured:
            model = MODELS['claude'][mode]
            cost = calculate_cost(model, avg_input_tokens * num_items, avg_output_tokens * num_items)
            estimates['claude'] = round(cost.total_cost, 2)
        
        if self.provider == AIProvider.BOTH and len(estimates) == 2:
            estimates['both'] = round(estimates['openai'] + estimates['claude'], 2)
        
        return estimates
    
    def _call_openai(self, prompt: str, system_prompt: str, mode: str = 'standard') -> Tuple[Optional[str], Optional[CostEstimate]]:
        """Call OpenAI API with cost tracking"""
        if not self.openai_configured:
            return None, None
        
        model = MODELS['openai'][mode]
        max_tokens = TOKEN_LIMITS[mode]
        
        # Estimate input tokens
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
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=API_TIMEOUT
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"].strip()
                    
                    # Get actual token usage
                    usage = result.get("usage", {})
                    actual_input = usage.get("prompt_tokens", input_tokens)
                    actual_output = usage.get("completion_tokens", len(content.split()))
                    
                    # Calculate cost
                    cost = calculate_cost(model, actual_input, actual_output)
                    self.cost_tracker.add_cost(cost)
                    
                    return content, cost
                
                elif response.status_code == 429:
                    wait_time = min(2 ** attempt * 2, 30)
                    logger.warning(f"OpenAI rate limited, waiting {wait_time}s")
                    time.sleep(wait_time)
                    continue
                
                else:
                    logger.error(f"OpenAI API error {response.status_code}")
                    return None, None
                    
            except Exception as e:
                logger.error(f"OpenAI call error: {e}")
                if attempt == MAX_RETRIES - 1:
                    return None, None
                time.sleep(2 ** attempt)
        
        return None, None
    
    def _call_claude(self, prompt: str, system_prompt: str, mode: str = 'standard') -> Tuple[Optional[str], Optional[CostEstimate]]:
        """Call Claude API with cost tracking"""
        if not self.claude_configured:
            return None, None
        
        model = MODELS['claude'][mode]
        max_tokens = TOKEN_LIMITS[mode]
        
        # Estimate input tokens
        input_tokens = estimate_tokens(system_prompt + prompt)
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.claude_key,
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "system": system_prompt,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload,
                    timeout=API_TIMEOUT
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result["content"][0]["text"].strip()
                    
                    # Get actual token usage
                    usage = result.get("usage", {})
                    actual_input = usage.get("input_tokens", input_tokens)
                    actual_output = usage.get("output_tokens", len(content.split()))
                    
                    # Calculate cost
                    cost = calculate_cost(model, actual_input, actual_output)
                    self.cost_tracker.add_cost(cost)
                    
                    return content, cost
                
                else:
                    logger.error(f"Claude API error {response.status_code}: {response.text}")
                    return None, None
                    
            except Exception as e:
                logger.error(f"Claude call error: {e}")
                if attempt == MAX_RETRIES - 1:
                    return None, None
                time.sleep(2 ** attempt)
        
        return None, None
    
    def categorize_return(self, complaint: str, fba_reason: str = None, mode: str = 'standard') -> Tuple[str, float, str, str]:
        """Categorize return with cost tracking"""
        if not complaint or not complaint.strip():
            return 'Other/Miscellaneous', 0.1, 'none', 'en'
        
        # Quick FBA mapping
        if fba_reason and fba_reason in FBA_REASON_MAP:
            category = FBA_REASON_MAP[fba_reason]
            return category, 0.95, detect_severity(complaint, category), 'en'
        
        # Detect language
        language = detect_language(complaint)
        
        # Build prompts
        system_prompt = """You are a medical device quality expert. Categorize this return into exactly one category from the provided list. Respond with ONLY the category name."""
        
        categories_list = '\n'.join(f'- {cat}' for cat in MEDICAL_DEVICE_CATEGORIES)
        
        user_prompt = f"""Complaint: "{complaint}"

Categories:
{categories_list}

Choose the most appropriate category."""
        
        # Get AI responses
        openai_result = None
        claude_result = None
        
        if self.provider in [AIProvider.OPENAI, AIProvider.BOTH] and self.openai_configured:
            openai_response, _ = self._call_openai(user_prompt, system_prompt, mode)
            if openai_response:
                openai_result = self._clean_category_response(openai_response)
        
        if self.provider in [AIProvider.CLAUDE, AIProvider.BOTH] and self.claude_configured:
            claude_response, _ = self._call_claude(user_prompt, system_prompt, mode)
            if claude_response:
                claude_result = self._clean_category_response(claude_response)
        
        # Determine final category
        if openai_result and claude_result:
            if openai_result == claude_result:
                category = openai_result
                confidence = 0.95
            else:
                category = openai_result if openai_result != 'Other/Miscellaneous' else claude_result
                confidence = 0.8
        elif openai_result:
            category = openai_result
            confidence = 0.85
        elif claude_result:
            category = claude_result
            confidence = 0.85
        else:
            category = 'Other/Miscellaneous'
            confidence = 0.3
        
        severity = detect_severity(complaint, category)
        
        return category, confidence, severity, language
    
    def _clean_category_response(self, response: str) -> str:
        """Clean AI response to extract category"""
        response = response.strip().strip('"').strip("'")
        
        # Try exact match
        for valid_cat in MEDICAL_DEVICE_CATEGORIES:
            if response == valid_cat or response.lower() == valid_cat.lower():
                return valid_cat
        
        # Try partial match
        for valid_cat in MEDICAL_DEVICE_CATEGORIES:
            if valid_cat.lower() in response.lower() or response.lower() in valid_cat.lower():
                return valid_cat
        
        return 'Other/Miscellaneous'
    
    def chat(self, user_message: str, context: Dict[str, Any] = None) -> Tuple[str, float]:
        """Chat about analysis results with cost tracking"""
        system_prompt = """You are a helpful assistant specializing in medical device quality management. 
        Answer questions about return categorization results, quality insights, and provide actionable recommendations.
        Be concise but thorough."""
        
        # Add context if available
        if context:
            context_str = f"\n\nContext:\n- Total returns analyzed: {context.get('total_returns', 0)}\n"
            if context.get('quality_rate'):
                context_str += f"- Quality issue rate: {context['quality_rate']:.1f}%\n"
            if context.get('top_categories'):
                context_str += f"- Top return categories: {', '.join(context['top_categories'][:3])}\n"
            
            system_prompt += context_str
        
        # Get response
        response, cost = self._call_openai(user_message, system_prompt, 'chat')
        
        if not response and self.claude_configured:
            response, cost = self._call_claude(user_message, system_prompt, 'chat')
        
        if not response:
            return "I apologize, but I'm unable to process your request at the moment.", 0.0
        
        return response, cost.total_cost if cost else 0.0
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get detailed cost summary"""
        return self.cost_tracker.get_summary()
    
    def estimate_remaining_cost(self, remaining_items: int) -> float:
        """Estimate cost for remaining items"""
        return self.cost_tracker.estimate_remaining_cost(remaining_items)

def generate_quality_insights(df, reason_summary: Dict, product_summary: Dict) -> Dict[str, Any]:
    """Generate quality insights with cost awareness"""
    
    total_returns = len(df)
    quality_issues = {cat: count for cat, count in reason_summary.items() 
                     if cat in ['Product Defects/Quality', 'Performance/Effectiveness',
                               'Design/Material Issues', 'Stability/Positioning Issues',
                               'Medical/Health Concerns']}
    
    total_quality = sum(quality_issues.values())
    quality_rate = (total_quality / total_returns * 100) if total_returns > 0 else 0
    
    # Risk assessment
    if quality_rate > 30:
        risk_level = "HIGH"
    elif quality_rate > 15:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
    
    # Top issues
    top_quality_issues = sorted(quality_issues.items(), key=lambda x: x[1], reverse=True)[:3]
    
    # Action items
    action_items = []
    for category, count in top_quality_issues:
        if count > 0:
            severity = "HIGH" if count > total_returns * 0.1 else "MEDIUM"
            action_items.append({
                'severity': severity,
                'issue': category,
                'frequency': count,
                'percentage': round(count / total_returns * 100, 1),
                'recommendation': f"Investigate root cause of {category.lower()} issues affecting {count} returns",
                'estimated_impact': f"Could reduce returns by {count} units"
            })
    
    # Product analysis
    top_risk_products = []
    for product, issues in product_summary.items():
        quality_count = sum(count for cat, count in issues.items() 
                          if cat in quality_issues.keys())
        if quality_count > 0:
            top_risk_products.append({
                'product': product,
                'total_issues': sum(issues.values()),
                'quality_issues': quality_count,
                'quality_rate': round(quality_count / sum(issues.values()) * 100, 1),
                'primary_issue': max(issues.items(), key=lambda x: x[1])[0]
            })
    
    top_risk_products.sort(key=lambda x: x['quality_issues'], reverse=True)
    
    return {
        'risk_assessment': {
            'overall_risk_level': risk_level,
            'quality_rate': quality_rate,
            'total_quality_issues': total_quality,
            'top_risk_products': top_risk_products[:10]
        },
        'action_items': action_items,
        'cost_benefit': {
            'potential_return_reduction': total_quality,
            'estimated_savings': f"${total_quality * 15:.2f}",  # Assuming $15 avg return cost
            'roi_timeframe': '3-6 months'
        }
    }

# Simplified exports for backward compatibility
class APIClient:
    """Backward compatible API client"""
    def __init__(self, provider: AIProvider = AIProvider.BOTH):
        self.analyzer = EnhancedAIAnalyzer(provider)
    
    def categorize_return(self, complaint: str, fba_reason: str = None, 
                         use_both: bool = True, max_tokens: int = 300) -> str:
        mode = 'extreme' if max_tokens > 500 else 'enhanced' if max_tokens > 200 else 'standard'
        category, _, _, _ = self.analyzer.categorize_return(complaint, fba_reason, mode)
        return category
    
    def get_usage_summary(self) -> Dict[str, Any]:
        return self.analyzer.get_cost_summary()

# Export all components
__all__ = [
    'EnhancedAIAnalyzer',
    'APIClient',
    'AIProvider',
    'MEDICAL_DEVICE_CATEGORIES',
    'FBA_REASON_MAP',
    'detect_language',
    'calculate_confidence',
    'detect_severity',
    'extract_quality_patterns',
    'generate_quality_insights',
    'CostEstimate',
    'CostTracker',
    'estimate_tokens',
    'calculate_cost'
]
