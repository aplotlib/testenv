"""
Enhanced AI Analysis Module - Medical Device Return Categorization
Version: 7.0 - Streamlined for Quality Management

Provides AI-powered analysis for categorizing returns from:
- PDF files (Amazon Seller Central)
- FBA Return Reports
- Product Complaints Ledger

Focuses on medical device quality management and FDA compliance.
"""

import logging
import os
import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
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
DEFAULT_MAX_TOKENS = 2000

# Medical device return categories
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

class APIClient:
    """Robust OpenAI API client with error handling"""
    
    def __init__(self):
        self.api_key = self._get_api_key()
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}" if self.api_key else ""
        }
        
        # Log API key status
        if self.api_key:
            logger.info(f"API key configured (first 10 chars): {self.api_key[:10]}...")
        else:
            logger.warning("No API key found - AI features will be disabled")
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key from multiple sources"""
        # Try Streamlit secrets first
        try:
            import streamlit as st
            if hasattr(st, 'secrets'):
                # Try multiple possible key names
                for key_name in ["openai_api_key", "OPENAI_API_KEY", "openai", "api_key"]:
                    if key_name in st.secrets:
                        api_key = str(st.secrets[key_name]).strip()
                        if api_key and api_key.startswith('sk-'):
                            logger.info(f"Found API key in Streamlit secrets under '{key_name}'")
                            return api_key
                    
                # Check nested secrets
                if "openai" in st.secrets and isinstance(st.secrets.get("openai"), dict):
                    if "api_key" in st.secrets["openai"]:
                        api_key = str(st.secrets["openai"]["api_key"]).strip()
                        if api_key and api_key.startswith('sk-'):
                            logger.info("Found API key in nested Streamlit secrets")
                            return api_key
        except Exception as e:
            logger.debug(f"Streamlit secrets not available: {e}")
        
        # Try environment variable
        for env_name in ["OPENAI_API_KEY", "OPENAI_API", "API_KEY"]:
            api_key = os.environ.get(env_name, '').strip()
            if api_key and api_key.startswith('sk-'):
                logger.info(f"Found API key in environment variable '{env_name}'")
                return api_key
        
        logger.warning("No OpenAI API key found in Streamlit secrets or environment")
        return None
    
    def is_available(self) -> bool:
        """Check if API is available"""
        return bool(self.api_key and has_requests)
    
    def call_api(self, messages: List[Dict[str, str]], 
                model: str = "gpt-4o-mini",
                temperature: float = 0.3,
                max_tokens: int = DEFAULT_MAX_TOKENS) -> Dict[str, Any]:
        """Make API call with retry logic"""
        
        if not self.is_available():
            return {
                "success": False,
                "error": "API not available - missing key or requests module",
                "result": "AI analysis requires OpenAI API key. Please add OPENAI_API_KEY to your Streamlit secrets or environment variables."
            }
        
        # Ensure we're using valid model names
        model_map = {
            'gpt-4o': 'gpt-4o-mini',
            'gpt-4': 'gpt-4-turbo-preview',
            'gpt-3.5': 'gpt-3.5-turbo'
        }
        
        if model in model_map:
            model = model_map[model]
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        last_error = None
        
        for attempt in range(MAX_RETRIES):
            try:
                logger.info(f"Making API call to {model} (attempt {attempt + 1}/{MAX_RETRIES})")
                
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=API_TIMEOUT
                )
                
                logger.info(f"API response status: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    logger.info(f"API call successful, response length: {len(content)} chars")
                    
                    return {
                        "success": True,
                        "result": content,
                        "usage": result.get("usage", {}),
                        "model": model
                    }
                    
                elif response.status_code == 401:
                    error_msg = "Invalid API key. Please check your OpenAI API key configuration."
                    logger.error(error_msg)
                    return {
                        "success": False,
                        "error": error_msg,
                        "result": None
                    }
                    
                elif response.status_code == 429:
                    # Rate limit - wait and retry
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limited, waiting {wait_time} seconds")
                    time.sleep(wait_time)
                    continue
                    
                elif response.status_code == 404:
                    # Model not found, try fallback
                    if model != 'gpt-3.5-turbo':
                        logger.warning(f"Model {model} not found, trying gpt-3.5-turbo")
                        model = 'gpt-3.5-turbo'
                        payload['model'] = model
                        continue
                    
                else:
                    error_data = response.json() if response.text else {}
                    error_msg = error_data.get('error', {}).get('message', f'API error {response.status_code}')
                    last_error = error_msg
                    logger.error(f"API error: {error_msg}")
                    
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(2 ** attempt)
                        continue
                    
            except requests.exceptions.Timeout:
                last_error = "Request timed out"
                logger.warning(f"API timeout on attempt {attempt + 1}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                    continue
                    
            except Exception as e:
                last_error = str(e)
                logger.error(f"API call error: {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                    continue
        
        return {
            "success": False,
            "error": last_error or "Max retries exceeded",
            "result": None
        }

class EnhancedAIAnalyzer:
    """Main AI analyzer class optimized for medical device return categorization"""
    
    def __init__(self):
        self.api_client = APIClient()
        logger.info(f"Enhanced AI Analyzer initialized - API available: {self.api_client.is_available()}")
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get API availability status"""
        is_available = self.api_client.is_available()
        
        status = {
            'available': is_available,
            'configured': bool(self.api_client.api_key)
        }
        
        if not self.api_client.api_key:
            status['error'] = 'API key not configured'
            status['message'] = 'Add OPENAI_API_KEY to Streamlit secrets or environment'
        elif not has_requests:
            status['error'] = 'Requests module not available'
            status['message'] = 'Install requests: pip install requests'
        else:
            # Try a test call with minimal tokens
            test_response = self.api_client.call_api(
                [{"role": "user", "content": "Hi"}],
                max_tokens=10
            )
            
            if test_response['success']:
                status['message'] = 'API is working correctly'
                status['model'] = test_response.get('model', 'gpt-4o-mini')
            else:
                status['available'] = False
                status['error'] = test_response.get('error', 'Unknown error')
                status['message'] = f"API test failed: {status['error']}"
        
        logger.info(f"API status check: {status}")
        return status
    
    def categorize_return(self, complaint: str, return_reason: str = None, fba_reason: str = None) -> str:
        """
        Categorize a single return into medical device categories
        
        Args:
            complaint: Customer complaint or comment text
            return_reason: Return reason from PDF or ledger
            fba_reason: FBA reason code
            
        Returns:
            Category name from MEDICAL_DEVICE_CATEGORIES
        """
        
        # Build context from all available information
        context_parts = []
        
        if return_reason and return_reason.strip():
            context_parts.append(f"Return Reason: {return_reason}")
        
        if fba_reason and fba_reason.strip():
            # Map FBA codes to human-readable descriptions
            fba_descriptions = {
                'NOT_COMPATIBLE': 'Not compatible with equipment',
                'DAMAGED_BY_FC': 'Damaged at fulfillment center',
                'DAMAGED_BY_CARRIER': 'Damaged during shipping',
                'DEFECTIVE': 'Product is defective',
                'NOT_AS_DESCRIBED': 'Not as described on listing',
                'WRONG_ITEM': 'Wrong item received',
                'MISSING_PARTS': 'Missing parts or components',
                'QUALITY_NOT_ADEQUATE': 'Quality not adequate',
                'UNWANTED_ITEM': 'Customer no longer wants item',
                'CUSTOMER_DAMAGED': 'Damaged by customer',
                'UNAUTHORIZED_PURCHASE': 'Unauthorized purchase',
                'BETTER_PRICE_AVAILABLE': 'Found better price elsewhere'
            }
            
            fba_desc = fba_descriptions.get(fba_reason, fba_reason)
            context_parts.append(f"FBA Code: {fba_reason} ({fba_desc})")
        
        context = "\n".join(context_parts) if context_parts else ""
        
        # Create categorization prompt
        categories_list = "\n".join([f"{i+1}. {cat}" for i, cat in enumerate(MEDICAL_DEVICE_CATEGORIES)])
        
        prompt = f"""You are a quality management expert for medical devices. Categorize this return into exactly ONE category.

Customer Complaint/Comment: {complaint}
{context}

Categories:
{categories_list}

Rules:
1. Choose the SINGLE most appropriate category based on the root cause
2. For medical devices, prioritize safety and quality issues
3. If multiple issues exist, choose the primary/most serious one
4. Consider FDA reportability - quality defects are high priority
5. Use FBA codes as hints but complaint text takes precedence

Respond with ONLY the category name exactly as shown in the list."""

        response = self.api_client.call_api(
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical device quality expert. Always respond with exactly one category name from the provided list, nothing else."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature for consistency
            max_tokens=50
        )
        
        if response['success']:
            category = response['result'].strip()
            
            # Validate the category
            if category in MEDICAL_DEVICE_CATEGORIES:
                return category
            
            # Try to match if response includes number
            for i, cat in enumerate(MEDICAL_DEVICE_CATEGORIES):
                if str(i+1) in category or cat.lower() in category.lower():
                    return cat
        
        # Log the failure for debugging
        logger.warning(f"AI categorization failed or returned invalid category: {response}")
        
        # Fallback to rule-based categorization
        return self._fallback_categorization(complaint, return_reason, fba_reason)
    
    def _fallback_categorization(self, complaint: str, return_reason: str = None, fba_reason: str = None) -> str:
        """Rule-based fallback when AI is unavailable"""
        
        # FBA reason code mapping
        fba_map = {
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
            'BETTER_PRICE_AVAILABLE': 'Price/Value',
            'DOES_NOT_FIT': 'Size/Fit Issues',
            'ARRIVED_LATE': 'Shipping/Fulfillment Issues'
        }
        
        # Check FBA mapping first
        if fba_reason and fba_reason in fba_map:
            return fba_map[fba_reason]
        
        # Combine all text
        all_text = f"{complaint} {return_reason or ''} {fba_reason or ''}".lower()
        
        # Keyword-based rules
        rules = [
            (['small', 'large', 'size', 'fit', 'tight', 'loose'], 'Size/Fit Issues'),
            (['uncomfortable', 'comfort', 'hurts', 'painful'], 'Comfort Issues'),
            (['defective', 'broken', 'damaged', 'quality', 'malfunction'], 'Product Defects/Quality'),
            (['not work', 'ineffective', 'performance'], 'Performance/Effectiveness'),
            (['unstable', 'slides', 'position'], 'Stability/Positioning Issues'),
            (['compatible', 'fit toilet', 'wheelchair'], 'Equipment Compatibility'),
            (['heavy', 'material', 'design'], 'Design/Material Issues'),
            (['wrong', 'different', 'expected'], 'Wrong Product/Misunderstanding'),
            (['missing', 'incomplete'], 'Missing Components'),
            (['mistake', 'changed mind', 'no longer'], 'Customer Error/Changed Mind'),
            (['shipping', 'package', 'late'], 'Shipping/Fulfillment Issues'),
            (['difficult', 'hard to', 'confusing'], 'Assembly/Usage Difficulty'),
            (['doctor', 'medical', 'health'], 'Medical/Health Concerns'),
            (['price', 'expensive', 'cheaper'], 'Price/Value')
        ]
        
        for keywords, category in rules:
            if any(keyword in all_text for keyword in keywords):
                return category
        
        return 'Other/Miscellaneous'
    
    def generate_quality_insights(self, 
                                category_summary: Dict[str, int],
                                product_summary: Dict[str, Dict[str, int]],
                                total_returns: int,
                                data_sources: List[str]) -> str:
        """
        Generate comprehensive quality management insights
        
        Args:
            category_summary: Count of returns by category
            product_summary: Returns by product and category
            total_returns: Total number of returns
            data_sources: List of data sources used (PDF, FBA, Ledger)
            
        Returns:
            Formatted insights report
        """
        
        # Calculate quality metrics
        quality_categories = ['Product Defects/Quality', 'Performance/Effectiveness', 
                            'Missing Components', 'Design/Material Issues']
        quality_count = sum(category_summary.get(cat, 0) for cat in quality_categories)
        quality_percentage = (quality_count / total_returns * 100) if total_returns > 0 else 0
        
        # Top categories
        sorted_categories = sorted(category_summary.items(), key=lambda x: x[1], reverse=True)
        
        # Top products with issues
        product_totals = {
            product: sum(cats.values()) 
            for product, cats in product_summary.items()
        }
        top_products = sorted(product_totals.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Build prompt
        prompt = f"""As a medical device quality management expert, analyze these return patterns and provide actionable insights for FDA compliance and quality improvement.

RETURN DATA SUMMARY:
- Total Returns: {total_returns}
- Data Sources: {', '.join(data_sources)}
- Quality-Related Returns: {quality_count} ({quality_percentage:.1f}%)

TOP RETURN CATEGORIES:
{chr(10).join([f"- {cat}: {count} ({count/total_returns*100:.1f}%)" for cat, count in sorted_categories[:10]])}

TOP 5 PRODUCTS BY RETURN VOLUME:
{chr(10).join([f"- {prod}: {count} returns" for prod, count in top_products if prod and str(prod).strip()])}

CRITICAL CONSIDERATIONS:
- Product Defects/Quality issues may require FDA reporting (MDRs)
- Medical/Health Concerns need immediate investigation
- Pattern analysis for potential recalls or field actions
- Customer safety is paramount

Provide a comprehensive report in this format:

## EXECUTIVE SUMMARY
[2-3 sentences for quality leadership on key findings and urgency]

## CRITICAL QUALITY ISSUES
1. [Most serious quality issue with specific data]
2. [Second priority issue]
3. [Third priority issue]

## FDA/REGULATORY CONSIDERATIONS
[Identify any patterns suggesting reportable events, potential recalls, or compliance issues]

## ROOT CAUSE ANALYSIS
[For top 3 categories, identify likely root causes based on the data]

## IMMEDIATE ACTIONS (Within 48 Hours)
1. [Most urgent action with clear steps]
2. [Second priority with responsible party]
3. [Third priority with timeline]

## 30-DAY QUALITY IMPROVEMENT PLAN
[Week 1]: [Specific actions]
[Week 2-3]: [Implementation steps]
[Week 4]: [Verification activities]

## PRODUCT-SPECIFIC RECOMMENDATIONS
[For each of the top 3 products with returns, provide targeted actions]

## METRICS FOR SUCCESS
[Define KPIs to track improvement - target <2% quality return rate]

Focus on patient safety, FDA compliance, and sustainable quality improvements."""

        response = self.api_client.call_api(
            messages=[
                {
                    "role": "system",
                    "content": "You are a senior medical device quality management expert with deep knowledge of FDA regulations, ISO 13485, and quality systems. Provide specific, actionable recommendations."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        if response['success']:
            return response['result']
        else:
            # Fallback insights
            return self._generate_fallback_insights(
                category_summary, product_summary, total_returns, 
                quality_count, quality_percentage, sorted_categories
            )
    
    def _generate_fallback_insights(self, category_summary, product_summary, 
                                   total_returns, quality_count, quality_percentage,
                                   sorted_categories) -> str:
        """Generate basic insights when AI is unavailable"""
        
        top_category = sorted_categories[0] if sorted_categories else ('Unknown', 0)
        
        insights = f"""## QUALITY MANAGEMENT SUMMARY

**Analysis Date:** {datetime.now().strftime('%B %d, %Y')}
**Total Returns Analyzed:** {total_returns}
**Quality-Related Returns:** {quality_count} ({quality_percentage:.1f}%)

## KEY FINDINGS

1. **Primary Return Category:** {top_category[0]} ({top_category[1]} returns, {top_category[1]/total_returns*100:.1f}%)
2. **Quality Impact:** {quality_percentage:.1f}% of returns are quality-related
3. **Multi-Source Analysis:** Data unified from PDF, FBA, and Ledger sources

## IMMEDIATE ACTIONS REQUIRED

1. **Quality Investigation**
   - Review all {quality_count} quality-related returns
   - Identify potential patterns for MDR reporting
   - Document findings in quality system

2. **Product Focus**
   - Prioritize top products with highest return rates
   - Conduct root cause analysis
   - Update inspection criteria

3. **Customer Safety**
   - Review Medical/Health Concerns category immediately
   - Assess need for customer notifications
   - Document in complaint files

## RECOMMENDATIONS

- Implement enhanced incoming inspection for top return categories
- Update IFUs to address usage difficulty issues  
- Consider design modifications for comfort/fit problems
- Track improvement metrics after interventions

## NEXT STEPS

1. Schedule quality review meeting within 48 hours
2. Assign CAPA owners for top 3 issues
3. Report findings to management
4. Monitor trends weekly for 30 days

*Note: This is an automated analysis. Please review with quality team for final decisions.*
"""
        
        return insights

# Export main classes
__all__ = ['EnhancedAIAnalyzer', 'APIClient']
