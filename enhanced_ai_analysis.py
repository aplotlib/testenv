"""
Enhanced AI Analysis Module - Amazon Listing Optimization Edition

**STABLE AI INTEGRATION FOR AMAZON SELLERS**

Provides robust AI-powered analysis using OpenAI GPT-4o with comprehensive
error handling and Amazon-specific optimization focus.

Author: Assistant
Version: 5.1 - Medical Device Quality Management Edition
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
MAX_RETRIES = 2
MAX_TOKENS = 2000  # Increased for comprehensive analysis

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
                model: str = "gpt-4o-mini",  # Using mini for cost efficiency
                temperature: float = 0.3,
                max_tokens: int = MAX_TOKENS) -> Dict[str, Any]:
        """Make API call with retry logic - matches main app's expectations"""
        
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
    """Main AI analyzer class optimized for Amazon listings and medical devices"""
    
    def __init__(self):
        self.api_client = APIClient()
        logger.info(f"Enhanced AI Analyzer initialized - API available: {self.api_client.is_available()}")
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get API availability status - matches main app's expectations"""
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
                max_tokens=5
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
    
    def analyze_reviews_for_listing_optimization(self, 
                                                reviews: List[Dict],
                                                product_info: Dict,
                                                listing_details: Optional[Dict] = None) -> str:
        """
        Main method called by the app for AI analysis
        Returns a formatted string with optimization recommendations
        """
        try:
            # Prepare review summaries
            review_texts = []
            for i, review in enumerate(reviews[:30], 1):  # Limit to 30 reviews
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

Current Description Preview:
{listing_details.get('description', 'Not provided')[:500]}
"""
            
            # Medical device context for quality management
            medical_device_context = """
Note: This is a medical device product. Pay special attention to:
- Safety concerns and adverse events
- Quality and reliability issues
- Usability and user training needs
- Regulatory compliance implications
- Documentation and instruction clarity
"""
            
            prompt = f"""You are an expert Amazon listing optimization specialist with medical device expertise.
Analyze these customer reviews to provide SPECIFIC, ACTIONABLE listing improvements.

PRODUCT INFORMATION:
ASIN: {product_info.get('asin', 'Unknown')}
Total Reviews Analyzed: {len(reviews)}
Average Rating: {sum(r.get('rating', 3) for r in reviews) / len(reviews):.1f}/5

{listing_context}

{medical_device_context}

CUSTOMER REVIEWS:
{chr(10).join(review_texts)}

Provide optimization recommendations in this EXACT format:

## TITLE OPTIMIZATION
Current issues identified from reviews:
[List specific problems mentioned]

Recommended new title (max 200 chars):
[Exact title incorporating customer language and addressing concerns]

## BULLET POINT REWRITE
Based on customer feedback, rewrite all 5 bullet points:

• Bullet 1 (Address #1 concern): [Complete bullet point text]
• Bullet 2 (Highlight safety/quality): [Complete bullet point text]
• Bullet 3 (Usability/ease of use): [Complete bullet point text]
• Bullet 4 (Specifications/compatibility): [Complete bullet point text]
• Bullet 5 (Support/warranty): [Complete bullet point text]

## A9 ALGORITHM OPTIMIZATION
Backend keywords extracted from customer language:
[Comma-separated list, max 250 chars]

## IMMEDIATE QUICK WINS
1. [Most critical change based on negative reviews]
2. [Second priority addressing common complaints]
3. [Third priority for conversion improvement]

Focus on medical device quality, safety, and compliance while improving conversion."""

            # Make the API call
            response = self.api_client.call_api(
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an Amazon listing optimization expert specializing in medical devices. Provide specific, implementable recommendations based on customer feedback."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
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
