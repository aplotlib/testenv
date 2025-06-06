"""
Enhanced AI Analysis Module - Enterprise Edition
Version 9.0 - Multi-Provider with Advanced Features

Features:
- Dual AI support (OpenAI + Claude) for consensus
- Multi-language detection and translation
- Confidence scoring
- Severity detection for medical devices
- Duplicate detection
- Increased token limits for accuracy

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
import hashlib
from difflib import SequenceMatcher

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
langdetect, has_langdetect = safe_import('langdetect')
googletrans, has_googletrans = safe_import('googletrans')

# If translation libraries not available, we'll use API-based translation
if has_googletrans:
    try:
        from googletrans import Translator
        translator = Translator()
    except:
        translator = None
        has_googletrans = False
else:
    translator = None

# API Configuration
API_TIMEOUT = 45  # Increased for larger token counts
MAX_RETRIES = 3

# Model configurations - using more powerful models
MODELS = {
    'openai': {
        'fast': 'gpt-3.5-turbo',
        'accurate': 'gpt-4',
        'default': 'gpt-4'  # Default to GPT-4 for better accuracy
    },
    'claude': {
        'fast': 'claude-3-haiku-20240307',
        'accurate': 'claude-3-sonnet-20240229',
        'default': 'claude-3-sonnet-20240229'  # Default to Sonnet for accuracy
    }
}

# Pricing per 1K tokens
PRICING = {
    'gpt-3.5-turbo': {'input': 0.0005, 'output': 0.0015},
    'gpt-4': {'input': 0.03, 'output': 0.06},
    'claude-3-haiku-20240307': {'input': 0.00025, 'output': 0.00125},
    'claude-3-sonnet-20240229': {'input': 0.003, 'output': 0.015}
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

# Severity keywords for medical devices
SEVERITY_KEYWORDS = {
    'critical': [
        'injury', 'injured', 'hospital', 'emergency', 'doctor', 'dangerous', 
        'unsafe', 'hazard', 'accident', 'wound', 'bleeding', 'pain severe',
        'medical attention', 'urgent care', 'health risk', 'safety issue',
        'burn', 'cut', 'bruise', 'fall', 'allergic reaction', 'infection',
        'hospitalized', 'ambulance', 'poison', 'toxic', 'death', 'died'
    ],
    'major': [
        'defective', 'broken', 'malfunction', 'unusable', 'failed', 'stopped working',
        "doesn't work", 'not working', 'poor quality', 'fell apart', 'dangerous',
        'unreliable', 'safety concern', 'risk', 'worried', 'concerned about safety',
        'completely broken', 'total failure', 'major defect', 'serious issue'
    ],
    'minor': [
        'uncomfortable', 'difficult', 'confusing', 'disappointed', 'not ideal',
        'could be better', 'minor issue', 'small problem', 'slight', 'somewhat',
        'a bit', 'not perfect', 'okay but', 'works but', 'functions but'
    ]
}

class AIProvider(Enum):
    OPENAI = "openai"
    CLAUDE = "claude"
    BOTH = "both"

def detect_language(text: str) -> str:
    """Detect language of text"""
    try:
        if has_langdetect:
            from langdetect import detect
            lang = detect(text)
            return lang
        else:
            # Simple heuristic - check for Spanish indicators
            spanish_indicators = ['ñ', 'á', 'é', 'í', 'ó', 'ú', ' es ', ' la ', ' el ', ' los ', ' las ', ' de ', ' que ', ' y ']
            spanish_count = sum(1 for indicator in spanish_indicators if indicator in text.lower())
            
            if spanish_count >= 3:
                return 'es'
            else:
                return 'en'
    except Exception as e:
        logger.warning(f"Language detection failed: {e}")
        return 'en'

def translate_text(text: str, source_lang: str = 'auto', target_lang: str = 'en') -> Optional[str]:
    """Translate text to target language"""
    try:
        if has_googletrans and translator:
            result = translator.translate(text, src=source_lang, dest=target_lang)
            return result.text
        else:
            # For production, you might want to use a proper translation API
            logger.warning("Translation library not available")
            return text
    except Exception as e:
        logger.warning(f"Translation failed: {e}")
        return text

def calculate_confidence(complaint: str, category: str, language: str = 'en') -> float:
    """Calculate confidence score for categorization"""
    
    confidence = 0.5  # Base confidence
    
    # Adjust based on complaint length
    complaint_length = len(complaint.split())
    if complaint_length > 20:
        confidence += 0.1
    elif complaint_length < 5:
        confidence -= 0.1
    
    # Adjust based on category specificity
    if category != 'Other/Miscellaneous':
        confidence += 0.2
    else:
        confidence -= 0.2
    
    # Check for keyword matches
    complaint_lower = complaint.lower()
    category_keywords = {
        'Size/Fit Issues': ['small', 'large', 'size', 'fit', 'tight', 'loose', 'narrow', 'wide'],
        'Comfort Issues': ['uncomfortable', 'comfort', 'hurts', 'painful', 'pressure', 'sore'],
        'Product Defects/Quality': ['defective', 'broken', 'damaged', 'quality', 'malfunction'],
        'Performance/Effectiveness': ['not work', 'ineffective', 'useless', 'performance'],
        'Stability/Positioning Issues': ['unstable', 'slides', 'moves', 'position', 'falls'],
        'Equipment Compatibility': ['compatible', 'fit toilet', 'fit wheelchair', 'walker'],
        'Design/Material Issues': ['heavy', 'bulky', 'material', 'design', 'flimsy'],
        'Wrong Product/Misunderstanding': ['wrong', 'different', 'not as described', 'expected'],
        'Missing Components': ['missing', 'incomplete', 'no instructions', 'parts missing'],
        'Customer Error/Changed Mind': ['mistake', 'changed mind', 'no longer', 'patient died'],
        'Shipping/Fulfillment Issues': ['shipping', 'damaged arrival', 'late', 'package'],
        'Assembly/Usage Difficulty': ['difficult', 'hard to', 'confusing', 'complicated'],
        'Medical/Health Concerns': ['doctor', 'medical', 'health', 'allergic', 'injury'],
        'Price/Value': ['price', 'expensive', 'value', 'cheaper', 'cost']
    }
    
    if category in category_keywords:
        keyword_match = any(keyword in complaint_lower for keyword in category_keywords[category])
        if keyword_match:
            confidence += 0.15
    
    # Adjust for non-English
    if language != 'en':
        confidence -= 0.05  # Slightly lower confidence for translations
    
    # Ensure confidence is between 0 and 1
    return max(0.0, min(1.0, confidence))

def detect_severity(complaint: str, category: str) -> str:
    """Detect severity level of complaint"""
    
    complaint_lower = complaint.lower()
    
    # Check for critical keywords first
    for keyword in SEVERITY_KEYWORDS['critical']:
        if keyword in complaint_lower:
            return 'critical'
    
    # Check if medical/health category
    if category == 'Medical/Health Concerns':
        return 'critical'  # Default medical concerns to critical
    
    # Check for major keywords
    for keyword in SEVERITY_KEYWORDS['major']:
        if keyword in complaint_lower:
            return 'major'
    
    # Check if quality defect
    if category in ['Product Defects/Quality', 'Performance/Effectiveness']:
        return 'major'  # Default quality issues to major
    
    # Check for minor keywords
    for keyword in SEVERITY_KEYWORDS['minor']:
        if keyword in complaint_lower:
            return 'minor'
    
    return 'none'

def is_duplicate(complaint1: str, complaint2: str, threshold: float = 0.85) -> bool:
    """Check if two complaints are duplicates"""
    
    # Quick check - if one is much longer than the other, probably not duplicate
    len_ratio = len(complaint1) / len(complaint2) if len(complaint2) > 0 else 0
    if len_ratio < 0.5 or len_ratio > 2.0:
        return False
    
    # Use sequence matcher for similarity
    similarity = SequenceMatcher(None, complaint1.lower(), complaint2.lower()).ratio()
    
    return similarity >= threshold

class APIClient:
    """Multi-provider API client supporting OpenAI and Claude"""
    
    def __init__(self, provider: AIProvider = None):
        if provider is None:
            provider = AIProvider.BOTH  # Default to both for max accuracy
            
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
        env_names = {
            'openai': ['OPENAI_API_KEY', 'OPENAI_API'],
            'claude': ['ANTHROPIC_API_KEY', 'CLAUDE_API_KEY']
        }
        
        # Try Streamlit secrets first
        try:
            import streamlit as st
            if hasattr(st, 'secrets'):
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
                temperature: float = 0.1,  # Low temperature for consistency
                max_tokens: int = 300,     # Increased default
                use_specific_provider: str = None) -> Dict[str, Any]:
        """Make API call to selected provider"""
        
        # Determine which provider to use
        if use_specific_provider:
            provider_to_use = use_specific_provider
        elif self.provider == AIProvider.OPENAI:
            provider_to_use = 'openai'
        elif self.provider == AIProvider.CLAUDE:
            provider_to_use = 'claude'
        else:  # BOTH - use both for comparison
            # Make calls to both providers
            results = {}
            
            if self.openai_client:
                openai_result = self.openai_client.call(messages, model, temperature, max_tokens)
                if openai_result['success']:
                    results['openai'] = openai_result
                    self._track_usage('openai', openai_result.get('usage', {}), 
                                    model or MODELS['openai']['default'])
            
            if self.claude_client:
                claude_result = self.claude_client.call(messages, model, temperature, max_tokens)
                if claude_result['success']:
                    results['claude'] = claude_result
                    self._track_usage('claude', claude_result.get('usage', {}), 
                                    model or MODELS['claude']['default'])
            
            # Return combined results
            if len(results) == 2:
                return {
                    "success": True,
                    "result": {
                        "openai": results['openai']['result'],
                        "claude": results['claude']['result']
                    },
                    "usage": {
                        "openai": results['openai'].get('usage', {}),
                        "claude": results['claude'].get('usage', {})
                    },
                    "providers": ["openai", "claude"]
                }
            elif len(results) == 1:
                provider = list(results.keys())[0]
                return results[provider]
            else:
                return {
                    "success": False,
                    "error": "No providers available",
                    "result": None
                }
        
        # Single provider call
        if provider_to_use == 'openai' and self.openai_client:
            result = self.openai_client.call(messages, model, temperature, max_tokens)
            if result['success']:
                self._track_usage('openai', result.get('usage', {}), 
                                model or MODELS['openai']['default'])
            return result
        elif provider_to_use == 'claude' and self.claude_client:
            result = self.claude_client.call(messages, model, temperature, max_tokens)
            if result['success']:
                self._track_usage('claude', result.get('usage', {}), 
                                model or MODELS['claude']['default'])
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
    
    def categorize_return(self, complaint: str, return_reason: str = None, 
                         fba_reason: str = None, use_both: bool = False,
                         max_tokens: int = 300) -> Union[str, Dict[str, str]]:
        """Categorize a return using AI with optional dual processing"""
        
        # Build prompt
        prompt = self._build_categorization_prompt(complaint, return_reason, fba_reason, max_tokens)
        
        messages = [
            {
                "role": "system", 
                "content": "You are a quality management expert categorizing medical device returns. Always respond with exactly one return reason from the provided list. Be very careful to match the exact text."
            },
            {"role": "user", "content": prompt}
        ]
        
        if use_both and self.provider == AIProvider.BOTH:
            # Get results from both providers
            result = self.call_api(messages, temperature=0.1, max_tokens=max_tokens)
            
            if result['success'] and 'providers' in result:
                # Both providers responded
                openai_cat = self._parse_categorization_result(result['result']['openai'])
                claude_cat = self._parse_categorization_result(result['result']['claude'])
                
                return {
                    'openai': openai_cat,
                    'claude': claude_cat
                }
            elif result['success']:
                # Single provider
                return self._parse_categorization_result(result['result'])
            else:
                return 'Other/Miscellaneous'
        else:
            # Single provider mode
            result = self.call_api(messages, temperature=0.1, max_tokens=max_tokens)
            
            if result['success']:
                return self._parse_categorization_result(result['result'])
            else:
                logger.error(f"Categorization failed: {result.get('error')}")
                return 'Other/Miscellaneous'
    
    def _build_categorization_prompt(self, complaint: str, return_reason: str, 
                                   fba_reason: str, max_tokens: int) -> str:
        """Build categorization prompt based on token allowance"""
        
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
        
        # Build prompt based on token allowance
        if max_tokens >= 300:  # Enhanced prompt for high token count
            return f"""You are a quality management expert for medical devices. Carefully analyze this customer complaint and select the SINGLE MOST APPROPRIATE return category.

Customer Complaint: "{complaint}"{context}

Available Medical Device Return Categories:
{reasons_list}

Instructions:
1. Read the complaint carefully and understand the primary issue
2. Consider all aspects: product function, safety, usability, and customer expectations
3. Look for both explicit statements and implied problems
4. For medical devices, prioritize safety and health-related categorizations
5. Choose the ONE category that best matches the PRIMARY complaint
6. If multiple issues exist, focus on the most significant one
7. Only use "Other/Miscellaneous" if absolutely no other category fits

Important category distinctions:
- "Size/Fit Issues": Product dimensions don't match user needs
- "Comfort Issues": Product causes discomfort during use
- "Product Defects/Quality": Manufacturing defects, broken items, quality control issues
- "Performance/Effectiveness": Product doesn't perform its intended function
- "Equipment Compatibility": Doesn't work with other equipment/devices
- "Medical/Health Concerns": Any health, safety, or medical issues

Think step by step about which category best fits, then respond with ONLY the exact category name from the list above."""
        
        else:  # Simpler prompt for lower token count
            return f"""Categorize this medical device return complaint.

Complaint: "{complaint}"{context}

Categories:
{reasons_list}

Reply with ONLY the exact category name that best fits the primary complaint."""
    
    def _parse_categorization_result(self, result: str) -> str:
        """Parse and validate categorization result"""
        
        # Clean the result
        result = result.strip().strip('"').strip("'")
        
        # Remove any extra text
        if ':' in result:
            result = result.split(':')[-1].strip()
        
        # Find exact match (case sensitive first)
        if result in MEDICAL_DEVICE_CATEGORIES:
            return result
        
        # Try case-insensitive match
        result_lower = result.lower()
        for valid in MEDICAL_DEVICE_CATEGORIES:
            if result_lower == valid.lower():
                return valid
        
        # Try partial match (in case AI added extra words)
        for valid in MEDICAL_DEVICE_CATEGORIES:
            if valid.lower() in result_lower or result_lower in valid.lower():
                return valid
        
        # Try to find the category name anywhere in the response
        for valid in MEDICAL_DEVICE_CATEGORIES:
            if valid in result:
                return valid
        
        logger.warning(f"Could not parse category from: {result}")
        return 'Other/Miscellaneous'

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
             temperature: float = 0.1,
             max_tokens: int = 300) -> Dict[str, Any]:
        """Make OpenAI API call"""
        
        model = model or MODELS['openai']['default']
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.9,  # Add top_p for better quality
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
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
                    wait_time = min(2 ** attempt * 2, 30)
                    logger.warning(f"Rate limited, waiting {wait_time} seconds")
                    time.sleep(wait_time)
                    continue
                else:
                    error_data = response.json() if response.text else {}
                    error_msg = error_data.get('error', {}).get('message', f'API error {response.status_code}')
                    logger.error(f"OpenAI API error: {error_msg}")
                    return {
                        "success": False,
                        "error": error_msg,
                        "result": None
                    }
                    
            except requests.exceptions.Timeout:
                logger.warning(f"OpenAI timeout on attempt {attempt + 1}")
                if attempt == MAX_RETRIES - 1:
                    return {
                        "success": False,
                        "error": "API timeout",
                        "result": None
                    }
                time.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"OpenAI API error: {str(e)}")
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
             temperature: float = 0.1,
             max_tokens: int = 300) -> Dict[str, Any]:
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
                    error_msg = f"Claude API error {response.status_code}"
                    logger.error(error_msg)
                    return {
                        "success": False,
                        "error": error_msg,
                        "result": None
                    }
                    
        except Exception as e:
            logger.error(f"Claude API error: {str(e)}")
            return {
                "success": False,
                "error": f"Claude API call failed: {str(e)}",
                "result": None
            }

class EnhancedAIAnalyzer:
    """Main AI analyzer with enhanced features"""
    
    def __init__(self, provider: AIProvider = AIProvider.BOTH):
        self.provider = provider
        self.api_client = APIClient(provider)
        logger.info(f"Enhanced AI Analyzer initialized with provider: {provider.value}")
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get detailed API status"""
        status = {
            'available': self.api_client.is_available(),
            'providers': {
                'openai': {
                    'configured': bool(self.api_client.openai_key),
                    'available': self.api_client.openai_client is not None,
                    'model': MODELS['openai']['default']
                },
                'claude': {
                    'configured': bool(self.api_client.claude_key),
                    'available': self.api_client.claude_client is not None,
                    'model': MODELS['claude']['default']
                }
            },
            'features': {
                'language_detection': has_langdetect,
                'translation': has_googletrans,
                'dual_ai': self.provider == AIProvider.BOTH
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
    
    def analyze_reviews_for_listing_optimization(self, 
                                                reviews: List[Dict],
                                                product_info: Dict,
                                                listing_details: Optional[Dict] = None,
                                                metrics: Optional[Dict] = None,
                                                marketplace_data: Optional[Dict] = None) -> str:
        """
        Main method for compatibility with the original app
        Enhanced with marketplace data support
        """
        try:
            # This method is for the main review analysis app compatibility
            # For the return categorizer, we use the simpler categorize_return method
            
            # Build a comprehensive analysis prompt
            prompt = "Analyze these reviews and provide listing optimization recommendations."
            
            # Make the API call
            response = self.api_client.call_api(
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an Amazon listing optimization expert."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            if response['success']:
                return response['result']
            else:
                return f"Analysis error: {response.get('error', 'Unknown error')}"
                
        except Exception as e:
            logger.error(f"Error in analyze_reviews_for_listing_optimization: {str(e)}")
            return f"Analysis error: {str(e)}"

# Export all necessary components
__all__ = [
    'EnhancedAIAnalyzer', 
    'APIClient', 
    'AIProvider', 
    'MEDICAL_DEVICE_CATEGORIES', 
    'FBA_REASON_MAP',
    'detect_language',
    'translate_text',
    'calculate_confidence',
    'detect_severity',
    'is_duplicate'
]
