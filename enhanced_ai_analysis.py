"""
Enhanced AI Analysis Module - Multi-Provider Edition
Supports both OpenAI and Claude for medical device return categorization

Version: 7.0 - Medical Device Categories
Author: Vive Health Quality Team
"""

import logging
import os
import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
import time

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
anthropic, has_anthropic = safe_import('anthropic')

# API Configuration
API_TIMEOUT = 30
MAX_RETRIES = 3

# Model configurations
MODELS = {
    'openai': {
        'fast': 'gpt-3.5-turbo',
        'accurate': 'gpt-4',
        'default': 'gpt-3.5-turbo'
    },
    'claude': {
        'fast': 'claude-3-haiku-20240307',
        'accurate': 'claude-3-sonnet-20240229',
        'default': 'claude-3-haiku-20240307'
    }
}

# Pricing per 1K tokens (approximate)
PRICING = {
    'gpt-3.5-turbo': {'input': 0.0005, 'output': 0.0015},
    'gpt-4': {'input': 0.03, 'output': 0.06},
    'claude-3-haiku-20240307': {'input': 0.00025, 'output': 0.00125},
    'claude-3-sonnet-20240229': {'input': 0.003, 'output': 0.015}
}

# Medical Device Return Categories (15 total)
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

# FBA reason code mapping to medical device categories
FBA_REASON_MAP = {
    'NOT_COMPATIBLE': 'Equipment Compatibility',
    'DAMAGED_BY_FC': 'Product Defects/Quality',
    'DAMAGED_BY_CARRIER': 'Shipping/Fulfillment Issues',
    'DEFECTIVE': 'Product Defects/Quality',
    'NOT_AS_DESCRIBED': 'Wrong Product/Misunderstanding',
    'WRONG_ITEM': 'Wrong Product/Misunderstanding',
    'MISSING_PARTS': 'Missing Components',
    'QUALITY_NOT_ADEQUATE': 'Performance/Effectiveness',
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

class AIProvider(Enum):
    OPENAI = "openai"
    CLAUDE = "claude"
    BOTH = "both"

class APIClient:
    """Multi-provider API client supporting OpenAI and Claude"""
    
    def __init__(self, provider: AIProvider = AIProvider.OPENAI):
        self.provider = provider
        self.openai_key = self._get_api_key('openai')
        self.claude_key = self._get_api_key('claude')
        
        # Initialize clients
        self.openai_client = None
        self.claude_client = None
        
        if self.openai_key and has_requests:
            self.openai_client = OpenAIClient(self.openai_key)
        
        if self.claude_key and has_anthropic:
            self.claude_client = ClaudeClient(self.claude_key)
        
        # Usage tracking
        self.usage_stats = {
            'openai': {
                'total_calls': 0,
                'total_input_tokens': 0,
                'total_output_tokens': 0,
                'total_cost': 0.0
            },
            'claude': {
                'total_calls': 0,
                'total_input_tokens': 0,
                'total_output_tokens': 0,
                'total_cost': 0.0
            }
        }
        
        logger.info(f"API Client initialized with provider: {provider.value}")
    
    def _get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for specified provider"""
        # Environment variable names
        env_names = {
            'openai': ['OPENAI_API_KEY', 'OPENAI_API'],
            'claude': ['ANTHROPIC_API_KEY', 'CLAUDE_API_KEY']
        }
        
        # Try Streamlit secrets first
        try:
            import streamlit as st
            if hasattr(st, 'secrets'):
                # Check various secret names
                secret_names = {
                    'openai': ['openai_api_key', 'OPENAI_API_KEY', 'openai'],
                    'claude': ['anthropic_api_key', 'claude_api_key', 'ANTHROPIC_API_KEY']
                }
                
                for key_name in secret_names.get(provider, []):
                    if key_name in st.secrets:
                        api_key = str(st.secrets[key_name]).strip()
                        logger.info(f"Found {provider} API key in Streamlit secrets")
                        return api_key
        except Exception as e:
            logger.debug(f"Streamlit secrets not available: {e}")
        
        # Try environment variables
        for env_name in env_names.get(provider, []):
            api_key = os.environ.get(env_name, '').strip()
            if api_key:
                logger.info(f"Found {provider} API key in environment variable '{env_name}'")
                return api_key
        
        logger.warning(f"No {provider} API key found")
        return None
    
    def is_available(self) -> bool:
        """Check if at least one provider is available"""
        if self.provider == AIProvider.OPENAI:
            return self.openai_client is not None
        elif self.provider == AIProvider.CLAUDE:
            return self.claude_client is not None
        else:  # BOTH
            return self.openai_client is not None or self.claude_client is not None
    
    def call_api(self, messages: List[Dict[str, str]], 
                model: str = None,
                temperature: float = 0.3,
                max_tokens: int = 100,
                use_specific_provider: str = None) -> Dict[str, Any]:
        """Make API call to selected provider"""
        
        # Determine which provider to use
        if use_specific_provider:
            provider_to_use = use_specific_provider
        elif self.provider == AIProvider.OPENAI:
            provider_to_use = 'openai'
        elif self.provider == AIProvider.CLAUDE:
            provider_to_use = 'claude'
        else:  # BOTH - choose based on availability and cost
            if self.openai_client and self.claude_client:
                # Choose cheaper option for simple categorization
                provider_to_use = 'claude' if max_tokens < 200 else 'openai'
            elif self.openai_client:
                provider_to_use = 'openai'
            elif self.claude_client:
                provider_to_use = 'claude'
            else:
                return {
                    "success": False,
                    "error": "No API providers available",
                    "result": None
                }
        
        # Make the call
        if provider_to_use == 'openai' and self.openai_client:
            result = self.openai_client.call(messages, model, temperature, max_tokens)
            if result['success']:
                self._track_usage('openai', result.get('usage', {}), model or MODELS['openai']['default'])
            return result
        elif provider_to_use == 'claude' and self.claude_client:
            result = self.claude_client.call(messages, model, temperature, max_tokens)
            if result['success']:
                self._track_usage('claude', result.get('usage', {}), model or MODELS['claude']['default'])
            return result
        else:
            return {
                "success": False,
                "error": f"Provider {provider_to_use} not available",
                "result": None
            }
    
    def _track_usage(self, provider: str, usage: Dict, model: str):
        """Track API usage and costs"""
        if provider in self.usage_stats and usage:
            stats = self.usage_stats[provider]
            stats['total_calls'] += 1
            
            input_tokens = usage.get('prompt_tokens', 0) or usage.get('input_tokens', 0)
            output_tokens = usage.get('completion_tokens', 0) or usage.get('output_tokens', 0)
            
            stats['total_input_tokens'] += input_tokens
            stats['total_output_tokens'] += output_tokens
            
            # Calculate cost
            if model in PRICING:
                cost = (input_tokens / 1000 * PRICING[model]['input'] + 
                       output_tokens / 1000 * PRICING[model]['output'])
                stats['total_cost'] += cost
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get usage summary across all providers"""
        total_cost = sum(stats['total_cost'] for stats in self.usage_stats.values())
        total_calls = sum(stats['total_calls'] for stats in self.usage_stats.values())
        
        return {
            'total_cost': total_cost,
            'total_calls': total_calls,
            'openai': self.usage_stats['openai'],
            'claude': self.usage_stats['claude']
        }

class OpenAIClient:
    """OpenAI-specific client"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def call(self, messages: List[Dict[str, str]], 
             model: str = None,
             temperature: float = 0.3,
             max_tokens: int = 100) -> Dict[str, Any]:
        """Make OpenAI API call"""
        
        model = model or MODELS['openai']['default']
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=API_TIMEOUT
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    
                    return {
                        "success": True,
                        "result": content,
                        "usage": result.get("usage", {}),
                        "model": model,
                        "provider": "openai"
                    }
                elif response.status_code == 429:
                    # Rate limit - wait and retry
                    time.sleep(2 ** attempt)
                    continue
                else:
                    error_data = response.json() if response.text else {}
                    return {
                        "success": False,
                        "error": error_data.get('error', {}).get('message', f'API error {response.status_code}'),
                        "result": None
                    }
                    
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    return {
                        "success": False,
                        "error": f"API call failed: {str(e)}",
                        "result": None
                    }
                time.sleep(2 ** attempt)
        
        return {
            "success": False,
            "error": "Max retries exceeded",
            "result": None
        }

class ClaudeClient:
    """Anthropic Claude-specific client"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        if has_anthropic:
            self.client = anthropic.Anthropic(api_key=api_key)
        else:
            self.client = None
            self.fallback_mode = True
            self.base_url = "https://api.anthropic.com/v1/messages"
            self.headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01"
            }
    
    def call(self, messages: List[Dict[str, str]], 
             model: str = None,
             temperature: float = 0.3,
             max_tokens: int = 100) -> Dict[str, Any]:
        """Make Claude API call"""
        
        model = model or MODELS['claude']['default']
        
        # Convert OpenAI format to Claude format
        system_message = None
        claude_messages = []
        
        for msg in messages:
            if msg['role'] == 'system':
                system_message = msg['content']
            else:
                claude_messages.append({
                    'role': msg['role'],
                    'content': msg['content']
                })
        
        try:
            if self.client:
                # Use anthropic SDK
                response = self.client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_message,
                    messages=claude_messages
                )
                
                return {
                    "success": True,
                    "result": response.content[0].text,
                    "usage": {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens
                    },
                    "model": model,
                    "provider": "claude"
                }
            else:
                # Fallback to direct API call
                payload = {
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": claude_messages
                }
                if system_message:
                    payload["system"] = system_message
                
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=API_TIMEOUT
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return {
                        "success": True,
                        "result": result["content"][0]["text"],
                        "usage": result.get("usage", {}),
                        "model": model,
                        "provider": "claude"
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Claude API error {response.status_code}",
                        "result": None
                    }
                    
        except Exception as e:
            return {
                "success": False,
                "error": f"Claude API call failed: {str(e)}",
                "result": None
            }

class EnhancedAIAnalyzer:
    """Main AI analyzer with multi-provider support"""
    
    def __init__(self, provider: AIProvider = AIProvider.BOTH):
        self.provider = provider
        self.api_client = APIClient(provider)
        logger.info(f"Enhanced AI Analyzer initialized with provider: {provider.value}")
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get detailed API status for all providers"""
        status = {
            'available': self.api_client.is_available(),
            'providers': {
                'openai': {
                    'configured': bool(self.api_client.openai_key),
                    'available': self.api_client.openai_client is not None
                },
                'claude': {
                    'configured': bool(self.api_client.claude_key),
                    'available': self.api_client.claude_client is not None
                }
            },
            'message': ''
        }
        
        if status['available']:
            active_providers = []
            if status['providers']['openai']['available']:
                active_providers.append('OpenAI')
            if status['providers']['claude']['available']:
                active_providers.append('Claude')
            status['message'] = f"AI ready with: {', '.join(active_providers)}"
        else:
            status['message'] = 'No AI providers configured'
        
        return status
    
    def fallback_categorization(self, complaint: str, fba_reason: str = None) -> str:
        """Fallback keyword-based categorization when AI is unavailable"""
        
        # First check FBA reason mapping
        if fba_reason and fba_reason in FBA_REASON_MAP:
            return FBA_REASON_MAP[fba_reason]
        
        complaint_lower = complaint.lower() if complaint else ""
        
        # Define keyword mappings for medical device categories
        keyword_map = {
            'Size/Fit Issues': ['small', 'large', 'size', 'fit', 'tight', 'loose', 'narrow', 'wide', 'big', 'tiny'],
            'Comfort Issues': ['uncomfortable', 'comfort', 'hurts', 'painful', 'pressure', 'sore', 'pain', 'ache'],
            'Product Defects/Quality': ['defective', 'broken', 'damaged', 'quality', 'malfunction', 'faulty', 'poor quality', 'defect', 'crack', 'tear', 'rip'],
            'Performance/Effectiveness': ['not work', 'ineffective', 'useless', 'performance', 'doesn\'t work', 'does not work', 'not effective'],
            'Stability/Positioning Issues': ['unstable', 'slides', 'moves', 'position', 'falls', 'tips', 'wobbly', 'shift'],
            'Equipment Compatibility': ['compatible', 'fit toilet', 'fit wheelchair', 'walker', 'doesn\'t fit', 'not compatible', 'incompatible'],
            'Design/Material Issues': ['heavy', 'bulky', 'material', 'design', 'flimsy', 'thin', 'cheap material'],
            'Wrong Product/Misunderstanding': ['wrong', 'different', 'not as described', 'expected', 'not what', 'incorrect', 'mistake'],
            'Missing Components': ['missing', 'incomplete', 'no instructions', 'parts missing', 'not included'],
            'Customer Error/Changed Mind': ['mistake', 'changed mind', 'no longer', 'patient died', 'don\'t need', 'ordered wrong'],
            'Shipping/Fulfillment Issues': ['shipping', 'damaged arrival', 'late', 'package', 'delivery', 'arrived damaged'],
            'Assembly/Usage Difficulty': ['difficult', 'hard to', 'confusing', 'complicated', 'instructions', 'assembly', 'setup'],
            'Medical/Health Concerns': ['doctor', 'medical', 'health', 'allergic', 'reaction', 'injury', 'condition'],
            'Price/Value': ['price', 'expensive', 'value', 'cheaper', 'cost', 'overpriced']
        }
        
        # Score each category
        scores = {}
        for category, keywords in keyword_map.items():
            score = sum(1 for keyword in keywords if keyword in complaint_lower)
            if score > 0:
                scores[category] = score
        
        # Return highest scoring category
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        
        return 'Other/Miscellaneous'
    
    def categorize_return(self, complaint: str, return_reason: str = None, 
                         fba_reason: str = None, use_both: bool = False) -> Union[str, Dict[str, str]]:
        """Categorize a return using AI"""
        
        # Build prompt
        prompt = self._build_categorization_prompt(complaint, return_reason, fba_reason)
        
        messages = [
            {
                "role": "system", 
                "content": "You are a quality management expert categorizing medical device returns. Always respond with exactly one return reason from the provided list."
            },
            {"role": "user", "content": prompt}
        ]
        
        if use_both and self.provider == AIProvider.BOTH:
            # Get results from both providers
            results = {}
            
            # Try OpenAI
            if self.api_client.openai_client:
                openai_result = self.api_client.call_api(
                    messages, 
                    temperature=0.1, 
                    max_tokens=50,
                    use_specific_provider='openai'
                )
                if openai_result['success']:
                    results['openai'] = self._parse_categorization_result(openai_result['result'])
            
            # Try Claude
            if self.api_client.claude_client:
                claude_result = self.api_client.call_api(
                    messages, 
                    temperature=0.1, 
                    max_tokens=50,
                    use_specific_provider='claude'
                )
                if claude_result['success']:
                    results['claude'] = self._parse_categorization_result(claude_result['result'])
            
            return results if len(results) > 1 else results.get('openai', results.get('claude', 'other'))
        else:
            # Normal single-provider categorization
            result = self.api_client.call_api(messages, temperature=0.1, max_tokens=50)
            
            if result['success']:
                return self._parse_categorization_result(result['result'])
            else:
                logger.error(f"Categorization failed: {result.get('error')}")
                return 'other'
    
    def _build_categorization_prompt(self, complaint: str, return_reason: str, fba_reason: str) -> str:
        """Build categorization prompt"""
        
        # Medical device return categories
        reasons_list = "\n".join([f"- {reason}" for reason in MEDICAL_DEVICE_CATEGORIES])
        
        # Add context
        context = ""
        if return_reason:
            context += f"\nOriginal return reason: {return_reason}"
        if fba_reason and fba_reason in FBA_REASON_MAP:
            suggested = FBA_REASON_MAP[fba_reason]
            context += f"\nFBA reason code: {fba_reason} (typically indicates: {suggested})"
        elif fba_reason:
            context += f"\nFBA reason code: {fba_reason}"
        
        return f"""You are a quality management expert for medical devices. Analyze this return and select the SINGLE MOST APPROPRIATE category.

Complaint: {complaint}{context}

Available Medical Device Return Categories:
{reasons_list}

Instructions:
1. Consider medical device quality and safety implications
2. Choose the ONE category that best matches the primary issue
3. If multiple categories apply, choose the most specific one
4. Only use "Other/Miscellaneous" if no other category fits

Respond with ONLY the exact category text from the list."""
    
    def _parse_categorization_result(self, result: str) -> str:
        """Parse and validate categorization result"""
        
        # Clean the result
        result = result.strip()
        
        # Find exact match (case sensitive first)
        if result in MEDICAL_DEVICE_CATEGORIES:
            return result
        
        # Try case-insensitive match
        for valid in MEDICAL_DEVICE_CATEGORIES:
            if result.lower() == valid.lower():
                return valid
        
        # Try partial match
        for valid in MEDICAL_DEVICE_CATEGORIES:
            if valid.lower() in result.lower() or result.lower() in valid.lower():
                return valid
        
        return 'Other/Miscellaneous'
    
    def generate_quality_insights(self, category_summary: Dict[str, int],
                                product_summary: Dict[str, Dict[str, int]],
                                total_returns: int,
                                data_sources: List[str]) -> str:
        """Generate quality management insights"""
        
        prompt = f"""As a medical device quality manager, analyze these return patterns and provide actionable insights.

RETURN SUMMARY:
Total Returns: {total_returns}
Data Sources: {', '.join(data_sources)}

CATEGORY BREAKDOWN:
{self._format_category_summary(category_summary)}

TOP PRODUCTS BY RETURNS:
{self._format_product_summary(product_summary)}

QUALITY CATEGORIES OF CONCERN:
- Product Defects/Quality
- Performance/Effectiveness
- Missing Components
- Design/Material Issues
- Stability/Positioning Issues
- Medical/Health Concerns

Provide analysis in this format:

## QUALITY MANAGEMENT SUMMARY
[Overview of return patterns and key findings]

## CRITICAL QUALITY ISSUES
[List top 3-5 quality concerns requiring immediate attention, focusing on medical device safety]

## ROOT CAUSE ANALYSIS
[Potential root causes for main return categories]

## IMMEDIATE ACTION ITEMS
1. [Most urgent quality action]
2. [Second priority]
3. [Third priority]

## CAPA RECOMMENDATIONS
[Specific corrective and preventive actions for quality issues]

## FDA CONSIDERATIONS
[Any patterns requiring MDR evaluation or regulatory action]

Focus on medical device quality, patient safety, and regulatory compliance."""

        messages = [
            {
                "role": "system",
                "content": "You are an expert medical device quality manager. Provide specific, actionable insights for quality improvement and regulatory compliance."
            },
            {"role": "user", "content": prompt}
        ]
        
        result = self.api_client.call_api(messages, temperature=0.3, max_tokens=1500)
        
        if result['success']:
            return result['result']
        else:
            return self._generate_fallback_insights(category_summary, product_summary, total_returns)
    
    def _format_category_summary(self, summary: Dict[str, int]) -> str:
        """Format category summary for prompt"""
        sorted_categories = sorted(summary.items(), key=lambda x: x[1], reverse=True)
        return "\n".join([f"{cat}: {count} returns" for cat, count in sorted_categories[:10]])
    
    def _format_product_summary(self, summary: Dict[str, Dict[str, int]]) -> str:
        """Format product summary for prompt"""
        product_totals = [(prod, sum(cats.values())) for prod, cats in summary.items()]
        top_products = sorted(product_totals, key=lambda x: x[1], reverse=True)[:5]
        
        formatted = []
        for product, total in top_products:
            top_issue = max(summary[product].items(), key=lambda x: x[1])
            formatted.append(f"{product}: {total} returns (top issue: {top_issue[0]})")
        
        return "\n".join(formatted)
    
    def _generate_fallback_insights(self, category_summary: Dict[str, int],
                                  product_summary: Dict[str, Dict[str, int]],
                                  total_returns: int) -> str:
        """Generate basic insights when AI is unavailable"""
        
        # Calculate quality-related returns
        quality_categories = ['Product Defects/Quality', 'Performance/Effectiveness', 
                            'Missing Components', 'Design/Material Issues', 
                            'Stability/Positioning Issues']
        quality_count = sum(count for cat, count in category_summary.items() 
                          if cat in quality_categories)
        quality_pct = (quality_count / total_returns * 100) if total_returns > 0 else 0
        
        # Get top category
        top_category = max(category_summary.items(), key=lambda x: x[1]) if category_summary else ('Unknown', 0)
        
        return f"""## QUALITY MANAGEMENT SUMMARY

Total Returns Analyzed: {total_returns}
Quality-Related Returns: {quality_count} ({quality_pct:.1f}%)
Top Return Category: {top_category[0]} ({top_category[1]} returns)

## CRITICAL QUALITY ISSUES

Based on the data, the following quality issues require immediate attention:
1. {quality_pct:.1f}% of returns are quality-related
2. Top return reason: {top_category[0]}
3. Multiple products showing consistent quality patterns

## IMMEDIATE ACTION ITEMS

1. Investigate all {quality_count} quality-related returns
2. Review top products for common failure modes
3. Update inspection criteria based on return reasons
4. Document findings for potential CAPA

## RECOMMENDATIONS

- Implement enhanced quality controls for identified issues
- Consider design modifications for comfort/fit problems
- Update IFUs to address usage difficulties
- Monitor improvement metrics after interventions"""

# Export classes
__all__ = ['EnhancedAIAnalyzer', 'APIClient', 'AIProvider']
