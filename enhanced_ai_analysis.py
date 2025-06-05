"""
Enhanced AI Analysis Module - Return Categorization Edition

Provides AI-powered analysis for Amazon return reasons and quality management insights.

Author: Assistant
Version: 6.0 - Quality Management Focus
"""

import logging
import os
import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

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
MAX_TOKENS = 9700

# Return categories for AI understanding
RETURN_CATEGORY_DEFINITIONS = """
SIZE_FIT_ISSUES: Product doesn't fit properly, too small/large, wrong dimensions
QUALITY_DEFECTS: Product is broken, defective, damaged, malfunctioning, poor quality
WRONG_PRODUCT: Received different item than ordered, not as described, incorrect product
BUYER_MISTAKE: Customer ordered by mistake, accidentally purchased, wrong selection
NO_LONGER_NEEDED: Customer no longer needs item, changed mind, patient died, plans changed
FUNCTIONALITY_ISSUES: Product hard to use, uncomfortable, unstable, difficult to operate
COMPATIBILITY_ISSUES: Product doesn't fit with other items, incompatible, wrong type
"""

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
                        logger.info(f"Found API key in Streamlit secrets under '{key_name}'")
                        return str(st.secrets[key_name])
                    
                # Check nested secrets
                if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
                    logger.info("Found API key in nested Streamlit secrets")
                    return str(st.secrets["openai"]["api_key"])
        except Exception as e:
            logger.debug(f"Streamlit secrets not available: {e}")
        
        # Try environment variable
        for env_name in ["OPENAI_API_KEY", "OPENAI_API", "API_KEY"]:
            api_key = os.environ.get(env_name)
            if api_key:
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
                max_tokens: int = MAX_TOKENS) -> Dict[str, Any]:
        """Make API call with retry logic"""
        
        if not self.is_available():
            return {
                "success": False,
                "error": "API not available - missing key or requests module",
                "result": "AI analysis requires OpenAI API key. Please add OPENAI_API_KEY to your Streamlit secrets or environment variables."
            }
        
        # Ensure we're using the correct model name
        if model == "gpt-4o":
            model = "gpt-4o-mini"  # Use the more cost-effective version
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
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
                    import time
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limited, waiting {wait_time} seconds")
                    time.sleep(wait_time)
                    continue
                    
                else:
                    error_data = response.json() if response.text else {}
                    error_msg = error_data.get('error', {}).get('message', f'API error {response.status_code}')
                    logger.error(f"API error: {error_msg}")
                    return {
                        "success": False,
                        "error": error_msg,
                        "result": None
                    }
                    
            except requests.exceptions.Timeout:
                logger.warning(f"API timeout on attempt {attempt + 1}")
                if attempt == MAX_RETRIES - 1:
                    return {
                        "success": False,
                        "error": "API timeout after multiple attempts",
                        "result": None
                    }
            except Exception as e:
                logger.error(f"API call error: {str(e)}")
                if attempt == MAX_RETRIES - 1:
                    return {
                        "success": False,
                        "error": f"API call failed: {str(e)}",
                        "result": None
                    }
        
        return {
            "success": False,
            "error": "Max retries exceeded",
            "result": None
        }

class EnhancedAIAnalyzer:
    """Main AI analyzer class optimized for return categorization and quality management"""
    
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
                max_tokens=150
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
    
    def categorize_return(self, return_reason: str, customer_comment: str) -> str:
        """Categorize a single return using AI"""
        prompt = f"""Categorize this Amazon product return into exactly ONE of these categories:

{RETURN_CATEGORY_DEFINITIONS}

Return reason: {return_reason}
Customer comment: {customer_comment}

Respond with ONLY the category name (e.g., SIZE_FIT_ISSUES)."""

        response = self.api_client.call_api(
            messages=[
                {
                    "role": "system",
                    "content": "You are a quality management expert categorizing product returns. Always respond with exactly one category name."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=750
        )
        
        if response['success']:
            category = response['result'].strip().upper().replace(' ', '_')
            # Validate category
            valid_categories = ['SIZE_FIT_ISSUES', 'QUALITY_DEFECTS', 'WRONG_PRODUCT', 
                              'BUYER_MISTAKE', 'NO_LONGER_NEEDED', 'FUNCTIONALITY_ISSUES', 
                              'COMPATIBILITY_ISSUES']
            
            if category in valid_categories:
                return category
        
        return 'UNCATEGORIZED'
    
    def analyze_return_patterns(self, categorized_returns: Dict[str, List], 
                              product_info: Optional[Dict] = None) -> str:
        """Analyze return patterns and provide quality management insights"""
        
        # Calculate statistics
        total_returns = sum(len(returns) for returns in categorized_returns.values())
        category_stats = {}
        
        for category, returns in categorized_returns.items():
            if returns:
                category_stats[category] = {
                    'count': len(returns),
                    'percentage': (len(returns) / total_returns * 100) if total_returns > 0 else 0
                }
        
        # Build analysis prompt
        stats_summary = "\n".join([
            f"{cat}: {stats['count']} returns ({stats['percentage']:.1f}%)"
            for cat, stats in category_stats.items()
        ])
        
        # Sample returns for context
        sample_returns = []
        for category, returns in categorized_returns.items():
            for ret in returns[:2]:  # Take 2 samples from each category
                sample_returns.append(f"- {category}: {ret.get('return_reason', '')} | {ret.get('buyer_comment', '')}")
        
        prompt = f"""As a quality management expert for medical devices, analyze these return patterns and provide actionable insights.

RETURN STATISTICS:
Total Returns: {total_returns}

CATEGORY BREAKDOWN:
{stats_summary}

SAMPLE RETURNS:
{chr(10).join(sample_returns[:10])}

Provide analysis in this format:

## QUALITY MANAGEMENT SUMMARY
[Brief overview of return patterns and quality concerns]

## CRITICAL QUALITY ISSUES
[List top 3 quality-related concerns based on the data]

## ROOT CAUSE ANALYSIS
[Identify potential root causes for the main return categories]

## IMMEDIATE ACTION ITEMS
1. [Most urgent action for quality team]
2. [Second priority action]
3. [Third priority action]

## LONG-TERM IMPROVEMENTS
[Strategic recommendations for reducing returns]

## PRODUCT-SPECIFIC INSIGHTS
[Any patterns related to specific products/ASINs if apparent]

Focus on medical device quality, safety, compliance, and user experience."""

        response = self.api_client.call_api(
            messages=[
                {
                    "role": "system",
                    "content": "You are a quality management expert for medical devices. Provide specific, actionable insights based on return data to improve product quality and reduce returns."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=9500
        )
        
        if response['success']:
            return response['result']
        else:
            return self._generate_fallback_analysis(category_stats)
    
    def _generate_fallback_analysis(self, category_stats: Dict) -> str:
        """Generate basic analysis when AI is unavailable"""
        analysis = "## QUALITY MANAGEMENT SUMMARY\n\n"
        
        # Find top issues
        sorted_categories = sorted(
            category_stats.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )
        
        if sorted_categories:
            top_category = sorted_categories[0][0]
            analysis += f"Primary return reason: {top_category.replace('_', ' ').title()} "
            analysis += f"({sorted_categories[0][1]['percentage']:.1f}% of returns)\n\n"
        
        # Quality alerts
        quality_returns = category_stats.get('QUALITY_DEFECTS', {}).get('percentage', 0)
        if quality_returns > 20:
            analysis += "⚠️ CRITICAL: Over 20% of returns are quality-related\n\n"
        
        analysis += "## IMMEDIATE ACTION ITEMS\n"
        analysis += "1. Review quality control processes\n"
        analysis += "2. Analyze specific defects mentioned in returns\n"
        analysis += "3. Update product documentation if needed\n"
        
        return analysis
    
    def analyze_reviews_for_listing_optimization(self, 
                                                reviews: List[Dict],
                                                product_info: Dict,
                                                listing_details: Optional[Dict] = None,
                                                metrics: Optional[Dict] = None,
                                                marketplace_data: Optional[Dict] = None) -> str:
        """
        Analyze reviews with marketplace data integration
        Enhanced to understand return patterns from marketplace data
        """
        try:
            # Check if we have return data in marketplace_data
            return_insights = ""
            if marketplace_data and 'return_patterns' in marketplace_data:
                return_patterns = marketplace_data['return_patterns']
                total_returns = sum(
                    data.get('count', 0) 
                    for data in return_patterns.values() 
                    if isinstance(data, dict)
                )
                
                if total_returns > 0:
                    return_insights = f"\n\nRETURN DATA INSIGHTS:\nTotal Returns: {total_returns}\n"
                    
                    # Analyze return reasons
                    for return_type, data in return_patterns.items():
                        if isinstance(data, dict) and data.get('count', 0) > 0:
                            return_insights += f"\n{return_type.upper()}: {data['count']} returns"
                            if 'reasons' in data and data['reasons']:
                                top_reasons = list(data['reasons'].items())[:3]
                                return_insights += "\nTop reasons: " + ", ".join([f"{r[0]} ({r[1]})" for r in top_reasons])
            
            # Prepare review summaries
            review_texts = []
            for i, review in enumerate(reviews[:30], 1):
                rating = review.get('rating', 3)
                title = review.get('title', '')[:100]
                body = review.get('body', '')[:300]
                verified = " [Verified]" if review.get('verified') else ""
                
                review_text = f"Review {i} ({rating}/5){verified}: {title} - {body}"
                review_texts.append(review_text)
            
            # Build the comprehensive prompt
            listing_context = ""
            if listing_details:
                listing_context = f"""
CURRENT LISTING DETAILS:
Title: {listing_details.get('title', 'Not provided')}
Brand: {listing_details.get('brand', 'Not provided')}
Category: {listing_details.get('category', 'Not provided')}
ASIN: {listing_details.get('asin', 'Not provided')}

Current Bullet Points:
{chr(10).join([f'• {b}' for b in listing_details.get('bullet_points', []) if b.strip()])}
"""
            
            # Medical device context for quality management
            medical_device_context = """
Note: This is a medical device product. Pay special attention to:
- Safety concerns and adverse events
- Quality and reliability issues  
- Usability and user training needs
- Regulatory compliance implications
- Documentation and instruction clarity
- Return patterns indicating quality issues
"""
            
            prompt = f"""You are an expert Amazon listing optimization specialist with medical device expertise.
Analyze these customer reviews and return data to provide SPECIFIC, ACTIONABLE listing improvements.

PRODUCT INFORMATION:
ASIN: {product_info.get('asin', 'Unknown')}
Total Reviews Analyzed: {len(reviews)}
{listing_context}
{medical_device_context}
{return_insights}

CUSTOMER REVIEWS:
{chr(10).join(review_texts)}

Provide optimization recommendations in this EXACT format:

## TITLE OPTIMIZATION
Current issues identified from reviews and returns:
[List specific problems mentioned]

Recommended new title (max 200 chars):
[Exact title incorporating customer language and addressing concerns]

## BULLET POINT REWRITE
Based on customer feedback and return reasons, rewrite all 5 bullet points:

• Bullet 1 (Address #1 concern): [Complete bullet point text]
• Bullet 2 (Highlight safety/quality): [Complete bullet point text]
• Bullet 3 (Usability/ease of use): [Complete bullet point text]
• Bullet 4 (Specifications/compatibility): [Complete bullet point text]
• Bullet 5 (Support/warranty): [Complete bullet point text]

## QUALITY & SAFETY PRIORITIES
[Based on reviews and returns, list quality improvements needed]

## RETURN REDUCTION STRATEGY
[Specific recommendations to reduce returns based on patterns]

Focus on medical device quality, safety, and compliance while improving conversion."""

            # Make the API call
            response = self.api_client.call_api(
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an Amazon listing optimization expert specializing in medical devices. Provide specific, implementable recommendations based on customer feedback and return data."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=9000
            )
            
            if response['success']:
                logger.info("AI analysis completed successfully")
                return response['result']
            else:
                error_msg = response.get('error', 'Unknown error')
                logger.error(f"AI analysis failed: {error_msg}")
                
                # Return a helpful error message
                return f"""## AI Analysis Error

{error_msg}

### Troubleshooting Steps:
1. Check that your OpenAI API key is correctly configured
2. Verify the API key has sufficient credits
3. Ensure you're using a valid API key that starts with 'sk-'

### Manual Analysis Recommendations:
While AI is unavailable, focus on:
- Addressing the most common complaints in reviews
- Reviewing return reasons for quality issues
- Adding safety and quality assurances to bullet points
- Including customer keywords in your title
- Updating backend search terms with review language"""
                
        except Exception as e:
            logger.error(f"Error in analyze_reviews_for_listing_optimization: {str(e)}")
            return f"""## Analysis Error

An error occurred during analysis: {str(e)}

Please check your configuration and try again."""

# Export main class
__all__ = ['EnhancedAIAnalyzer', 'APIClient']
