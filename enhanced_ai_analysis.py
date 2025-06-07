"""
Enhanced AI Analysis Module - Return Reason Categorization
Version 11.0 - Fixed for Vive Health Quality Management

Key Features:
- AI-first categorization using GPT-4 (with GPT-3.5 fallback)
- Medical device specific categorization
- Clear error messages and fallback logic
- FBA reason code mapping
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

# API Configuration
API_TIMEOUT = 30
MAX_RETRIES = 3

# Medical Device Return Categories (from your specification)
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

# Category keywords for fallback matching
CATEGORY_KEYWORDS = {
    'Size/Fit Issues': ['small', 'large', 'big', 'loose', 'tight', 'size', 'fit', 'narrow', 'wide', 'short', 'tall', 'thin'],
    'Comfort Issues': ['uncomfortable', 'comfort', 'hurts', 'painful', 'pain', 'pressure', 'sore', 'firm', 'hard', 'soft', 'stiff'],
    'Product Defects/Quality': ['defective', 'broken', 'damaged', 'quality', 'malfunction', 'defect', 'ripped', 'torn', 'not working', 'does not work'],
    'Performance/Effectiveness': ['ineffective', 'not work', 'useless', 'performance', 'not as expected', 'does not meet', 'poor support', 'not enough'],
    'Stability/Positioning Issues': ['unstable', 'slides', 'moves', 'position', 'falls', 'stay in place', 'slippery', 'slides around'],
    'Equipment Compatibility': ['not compatible', 'does not fit', 'fit toilet', 'fit wheelchair', 'walker', 'machine', 'device'],
    'Design/Material Issues': ['heavy', 'bulky', 'material', 'design', 'flimsy', 'thick', 'thin', 'grip'],
    'Wrong Product/Misunderstanding': ['wrong', 'different', 'not as described', 'expected', 'thought it was', 'not as advertised', 'misunderstanding'],
    'Missing Components': ['missing', 'incomplete', 'no instructions', 'parts missing', 'pieces', 'accessories'],
    'Customer Error/Changed Mind': ['mistake', 'changed mind', 'no longer', 'patient died', 'ordered wrong', 'bought by mistake', 'unauthorized'],
    'Shipping/Fulfillment Issues': ['shipping', 'damaged arrival', 'late', 'package', 'never arrived', 'delivery'],
    'Assembly/Usage Difficulty': ['difficult', 'hard to', 'confusing', 'complicated', 'assembly', 'instructions', 'setup'],
    'Medical/Health Concerns': ['doctor', 'medical', 'health', 'allergic', 'injury', 'hospital', 'reaction'],
    'Price/Value': ['price', 'expensive', 'value', 'cheaper', 'cost', 'money', 'better price']
}

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

class AIProvider(Enum):
    OPENAI = "openai"
    BOTH = "both"

def detect_language(text: str) -> str:
    """Simple language detection"""
    # Basic detection for Spanish
    spanish_indicators = ['ñ', 'á', 'é', 'í', 'ó', 'ú', ' es ', ' la ', ' el ', ' los ', ' las ']
    spanish_count = sum(1 for indicator in spanish_indicators if indicator in text.lower())
    return 'es' if spanish_count >= 2 else 'en'

def translate_text(text: str, source_lang: str = 'auto', target_lang: str = 'en') -> str:
    """Simple passthrough - implement translation if needed"""
    return text

def calculate_confidence(complaint: str, category: str, language: str = 'en') -> float:
    """Calculate confidence score for categorization"""
    # Always return high confidence for simplicity
    return 0.9

def detect_severity(complaint: str, category: str) -> str:
    """Detect severity level of complaint"""
    complaint_lower = complaint.lower()
    
    # Critical keywords
    critical_keywords = ['injury', 'injured', 'hospital', 'emergency', 'doctor', 'dangerous', 
                        'unsafe', 'accident', 'bleeding', 'burn', 'fall', 'allergic']
    
    for keyword in critical_keywords:
        if keyword in complaint_lower:
            return 'critical'
    
    # Major keywords  
    if category in ['Product Defects/Quality', 'Medical/Health Concerns']:
        return 'major'
    
    return 'none'

def is_duplicate(complaint1: str, complaint2: str) -> bool:
    """Removed - no duplicate detection"""
    return False

class EnhancedAIAnalyzer:
    """Main AI analyzer for medical device return categorization"""
    
    def __init__(self, provider: AIProvider = AIProvider.OPENAI):
        self.provider = provider
        self.api_key = self._get_api_key()
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}" if self.api_key else ""
        }
        
        logger.info(f"AI Analyzer initialized - API key found: {bool(self.api_key)}")
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key from multiple sources"""
        # Try Streamlit secrets first
        try:
            import streamlit as st
            if hasattr(st, 'secrets'):
                for key_name in ["openai_api_key", "OPENAI_API_KEY", "openai", "api_key"]:
                    if key_name in st.secrets:
                        logger.info(f"Found API key in Streamlit secrets under '{key_name}'")
                        return str(st.secrets[key_name])
        except Exception as e:
            logger.debug(f"Streamlit secrets not available: {e}")
        
        # Try environment variable
        for env_name in ["OPENAI_API_KEY", "OPENAI_API", "API_KEY"]:
            api_key = os.environ.get(env_name)
            if api_key:
                logger.info(f"Found API key in environment variable '{env_name}'")
                return api_key
        
        logger.warning("No OpenAI API key found")
        return None
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get API availability status"""
        return {
            'available': bool(self.api_key),
            'configured': bool(self.api_key),
            'message': 'API ready' if self.api_key else 'API key not configured'
        }
    
    def categorize_return(self, complaint: str, fba_reason: str = None) -> Tuple[str, float, str, str]:
        """Categorize a return reason using AI first"""
        
        # If no API key, check FBA mapping or use keywords as fallback
        if not self.api_key:
            # Check FBA reason mapping
            if fba_reason and fba_reason in FBA_REASON_MAP:
                category = FBA_REASON_MAP[fba_reason]
                return category, 0.95, 'none', 'en'
            
            # Fallback to keywords
            complaint_lower = complaint.lower()
            for category, keywords in CATEGORY_KEYWORDS.items():
                if any(keyword in complaint_lower for keyword in keywords):
                    severity = detect_severity(complaint, category)
                    return category, 0.85, severity, 'en'
            
            return 'Other/Miscellaneous', 0.5, 'none', 'en'
        
        # Use AI for categorization (PRIMARY METHOD)
        try:
            # Build comprehensive prompt
            prompt = f"""You are a quality management expert for medical devices. Categorize this return complaint.

Complaint: "{complaint}"
"""
            
            # Add FBA context if available
            if fba_reason:
                prompt += f"\nFBA Reason Code: {fba_reason}"
                if fba_reason in FBA_REASON_MAP:
                    prompt += f" (typically indicates: {FBA_REASON_MAP[fba_reason]})"
            
            prompt += f"""

IMPORTANT: Choose exactly ONE category from this list:
{chr(10).join(f'- {cat}' for cat in MEDICAL_DEVICE_CATEGORIES)}

Instructions:
1. Read the complaint carefully
2. Consider the primary issue the customer is reporting
3. If multiple issues exist, focus on the most significant one
4. Be specific - avoid "Other/Miscellaneous" unless absolutely necessary

Category distinctions:
- "Size/Fit Issues": Product dimensions don't match user needs (too small/large/tight/loose)
- "Comfort Issues": Product causes discomfort during use
- "Product Defects/Quality": Manufacturing defects, broken items, poor quality
- "Performance/Effectiveness": Product doesn't perform its intended function
- "Equipment Compatibility": Doesn't work with other equipment/devices
- "Customer Error/Changed Mind": Customer mistake, no longer needed, patient died
- "Medical/Health Concerns": Health reactions, safety issues, doctor concerns

Your response must be EXACTLY one of the category names listed above. Do not add any other words, punctuation, or explanation."""

            payload = {
                "model": "gpt-4",  # Use GPT-4 for better accuracy
                "messages": [
                    {
                        "role": "system", 
                        "content": f"""You are a quality expert categorizing medical device returns. 

You MUST respond with EXACTLY one of these categories (copy exactly as shown):
{chr(10).join(f'"{cat}"' for cat in MEDICAL_DEVICE_CATEGORIES)}

Do not add any other text, explanation, or punctuation. Just the category name."""
                    },
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,  # Low temperature for consistency
                "max_tokens": 30,    # Reduced to ensure only category name
                "top_p": 0.95,
                "frequency_penalty": 0,
                "presence_penalty": 0
            }
            
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=API_TIMEOUT
            )
            
            if response.status_code == 200:
                result = response.json()
                category_result = result["choices"][0]["message"]["content"].strip()
                
                # Clean the response - remove quotes and extra whitespace
                category_result = category_result.strip()
                # Remove surrounding quotes if present
                if category_result.startswith('"') and category_result.endswith('"'):
                    category_result = category_result[1:-1]
                if category_result.startswith("'") and category_result.endswith("'"):
                    category_result = category_result[1:-1]
                category_result = category_result.strip()
                
                # Find exact match first
                for valid_cat in MEDICAL_DEVICE_CATEGORIES:
                    if category_result == valid_cat:
                        severity = detect_severity(complaint, valid_cat)
                        logger.info(f"AI categorized '{complaint[:50]}...' as '{valid_cat}'")
                        return valid_cat, 0.9, severity, 'en'
                
                # Try case-insensitive match
                for valid_cat in MEDICAL_DEVICE_CATEGORIES:
                    if category_result.lower() == valid_cat.lower():
                        severity = detect_severity(complaint, valid_cat)
                        return valid_cat, 0.9, severity, 'en'
                
                # Try partial match
                for valid_cat in MEDICAL_DEVICE_CATEGORIES:
                    if valid_cat in category_result or category_result in valid_cat:
                        severity = detect_severity(complaint, valid_cat)
                        return valid_cat, 0.85, severity, 'en'
                
                # If no match found, log and return Other
                logger.warning(f"AI returned unrecognized category: '{category_result}' for complaint: '{complaint[:50]}...'")
                return 'Other/Miscellaneous', 0.6, 'none', 'en'
                
            else:
                # Log detailed error
                error_msg = f"API error {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg += f": {error_data.get('error', {}).get('message', response.text)}"
                except:
                    error_msg += f": {response.text[:200]}"
                
                logger.error(error_msg)
                
                # If GPT-4 fails, try GPT-3.5-turbo as fallback
                if response.status_code in [404, 403] and payload["model"] == "gpt-4":
                    logger.info("Falling back to GPT-3.5-turbo")
                    payload["model"] = "gpt-3.5-turbo"
                    
                    fallback_response = requests.post(
                        self.base_url,
                        headers=self.headers,
                        json=payload,
                        timeout=API_TIMEOUT
                    )
                    
                    if fallback_response.status_code == 200:
                        result = fallback_response.json()
                        category_result = result["choices"][0]["message"]["content"].strip()
                        
                        # Process result same as above
                        # Clean the response - remove quotes and extra whitespace
                        category_result = category_result.strip()
                        if category_result.startswith('"') and category_result.endswith('"'):
                            category_result = category_result[1:-1]
                        if category_result.startswith("'") and category_result.endswith("'"):
                            category_result = category_result[1:-1]
                        category_result = category_result.strip()
                        
                        for valid_cat in MEDICAL_DEVICE_CATEGORIES:
                            if category_result == valid_cat:
                                severity = detect_severity(complaint, valid_cat)
                                return valid_cat, 0.85, severity, 'en'
                        
                        for valid_cat in MEDICAL_DEVICE_CATEGORIES:
                            if category_result.lower() == valid_cat.lower():
                                severity = detect_severity(complaint, valid_cat)
                                return valid_cat, 0.85, severity, 'en'
                
                # Fallback to FBA mapping if available
                if fba_reason and fba_reason in FBA_REASON_MAP:
                    category = FBA_REASON_MAP[fba_reason]
                    return category, 0.8, 'none', 'en'
                
                # Last resort - keyword matching
                complaint_lower = complaint.lower()
                for category, keywords in CATEGORY_KEYWORDS.items():
                    if any(keyword in complaint_lower for keyword in keywords):
                        severity = detect_severity(complaint, category)
                        return category, 0.7, severity, 'en'
                
                return 'Other/Miscellaneous', 0.5, 'none', 'en'
                
        except Exception as e:
            logger.error(f"Categorization error: {e}")
            
            # Fallback to keyword matching
            complaint_lower = complaint.lower()
            for category, keywords in CATEGORY_KEYWORDS.items():
                if any(keyword in complaint_lower for keyword in keywords):
                    severity = detect_severity(complaint, category)
                    return category, 0.7, severity, 'en'
            
            return 'Other/Miscellaneous', 0.5, 'none', 'en'

# Simplified API Client for compatibility
class APIClient:
    """Simplified API client for the app"""
    
    def __init__(self):
        self.analyzer = EnhancedAIAnalyzer()
    
    def categorize_return(self, complaint: str, fba_reason: str = None, **kwargs) -> str:
        """Simple categorization method - AI first"""
        # This returns just the category string for compatibility
        category, _, _, _ = self.analyzer.categorize_return(complaint, fba_reason)
        return category
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get usage summary"""
        return {
            'total_cost': 0.0,
            'total_calls': 0
        }

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
