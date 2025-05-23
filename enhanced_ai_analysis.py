"""
Enhanced AI Analysis Module - Amazon Listing Optimization Edition

**STABLE AI INTEGRATION FOR AMAZON SELLERS**

Provides robust AI-powered analysis using OpenAI GPT-4o with comprehensive
error handling and Amazon-specific optimization focus.

Author: Assistant
Version: 5.0 - Amazon Optimization Edition
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
MAX_TOKENS = 1500

class APIClient:
    """Robust OpenAI API client with error handling"""
    
    def __init__(self):
        self.api_key = self._get_api_key()
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}" if self.api_key else ""
        }
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key from multiple sources"""
        # Try Streamlit secrets first
        try:
            import streamlit as st
            if hasattr(st, 'secrets'):
                for key_name in ["openai_api_key", "OPENAI_API_KEY"]:
                    if key_name in st.secrets:
                        logger.info("Found API key in Streamlit secrets")
                        return st.secrets[key_name]
        except:
            pass
        
        # Try environment variable
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            logger.info("Found API key in environment")
            return api_key
        
        logger.warning("No OpenAI API key found")
        return None
    
    def is_available(self) -> bool:
        """Check if API is available"""
        return bool(self.api_key and has_requests)
    
    def call_api(self, messages: List[Dict[str, str]], 
                model: str = "gpt-4o",
                temperature: float = 0.1,
                max_tokens: int = MAX_TOKENS) -> Dict[str, Any]:
        """Make API call with retry logic"""
        
        if not self.is_available():
            return {
                "success": False,
                "error": "API not available - missing key or requests module",
                "result": None
            }
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        for attempt in range(MAX_RETRIES):
            try:
                logger.info(f"Making API call (attempt {attempt + 1}/{MAX_RETRIES})")
                
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
                        "result": result["choices"][0]["message"]["content"],
                        "usage": result.get("usage", {}),
                        "model": model
                    }
                elif response.status_code == 429:
                    # Rate limit - wait and retry
                    import time
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limited, waiting {wait_time} seconds")
                    time.sleep(wait_time)
                    continue
                else:
                    error_msg = f"API error {response.status_code}: {response.text[:200]}"
                    logger.error(error_msg)
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

class PromptTemplates:
    """AI prompt templates optimized for Amazon listing analysis"""
    
    @staticmethod
    def create_review_analysis_prompt(product_info: Dict[str, Any], 
                                    review_summaries: List[str]) -> str:
        """Create prompt for Amazon-specific review analysis"""
        
        product_name = product_info.get('name', 'Unknown Product')
        product_category = product_info.get('category', 'Amazon Product')
        asin = product_info.get('asin', 'Unknown ASIN')
        
        return f"""You are an elite Amazon listing optimization expert analyzing customer reviews for immediate listing improvements.

PRODUCT: {product_name}
CATEGORY: {product_category}
ASIN: {asin}
TOTAL REVIEWS: {len(review_summaries)}

CUSTOMER REVIEWS:
{chr(10).join(review_summaries)}

Provide AMAZON-SPECIFIC analysis in this EXACT format:

## TITLE OPTIMIZATION OPPORTUNITIES
[Specific keywords from reviews that should be in title]
[Exact new title suggestion - 200 chars max]

## BULLET POINT GAPS
[Top 5 customer concerns not addressed in current bullets]
[Specific language customers use that should be in bullets]

## A9 ALGORITHM INSIGHTS
[Hidden keywords from reviews competitors likely miss]
[Long-tail search terms customers actually use]
[Backend search term recommendations]

## MAIN IMAGE REQUIREMENTS
[Visual concerns mentioned in reviews]
[What customers need to see to buy]

## CONVERSION KILLERS
[Top 3 issues causing customers to not purchase]
[Exact copy to address each concern]

## COMPETITOR ADVANTAGES
[What competitors are mentioned and why]
[How to position against them]

## IMMEDIATE ACTIONS (DO TODAY)
1. [Most impactful change - be specific]
2. [Second priority - exact implementation]
3. [Third priority - copy to use]

## REVIEW RESPONSE TEMPLATES
[Template for 1-2 star reviews]
[Key phrases to use consistently]

Focus on actionable changes that will improve Best Seller Rank and conversion rate within 7 days."""
    
    @staticmethod
    def create_listing_optimization_prompt(metrics: Dict[str, Any], 
                                         top_issues: List[str]) -> str:
        """Create prompt for listing optimization recommendations"""
        
        return f"""As an Amazon listing optimization expert, create specific listing improvements based on this data:

METRICS:
- Average Rating: {metrics.get('avg_rating', 'N/A')}/5
- Negative Review %: {metrics.get('negative_pct', 'N/A')}%
- Main Issues: {', '.join(top_issues[:5])}

Provide EXACT copy for:

## NEW TITLE (200 chars max)
[Include main keywords, brand, key feature, and size/count]

## OPTIMIZED BULLETS (5 bullets)
• [Bullet 1 - Address biggest concern]
• [Bullet 2 - Highlight unique benefit]
• [Bullet 3 - Size/compatibility info]
• [Bullet 4 - Quality/durability]
• [Bullet 5 - Guarantee/support]

## A+ CONTENT MODULES
1. [Module type and content focus]
2. [Module type and content focus]
3. [Module type and content focus]

## BACKEND KEYWORDS
[Comma-separated list of 250 chars]

## FAQ SECTION (Top 5)
Q1: [Question]
A1: [Answer]
[Continue for Q2-Q5]

Be specific. Use emotional triggers. Address every major concern."""
    
    @staticmethod
    def create_competitive_analysis_prompt(competitor_mentions: List[str],
                                         product_strengths: List[str]) -> str:
        """Create prompt for competitive positioning"""
        
        return f"""Analyze these competitor mentions from reviews and create differentiation strategy:

COMPETITOR MENTIONS:
{chr(10).join(competitor_mentions[:10])}

OUR STRENGTHS:
{chr(10).join(product_strengths[:5])}

Provide:

## COMPETITIVE POSITIONING STATEMENT
[2-3 sentences for product description]

## COMPARISON CHART ELEMENTS
[5 key differentiators to highlight]

## COUNTER-MESSAGING
[How to address each competitor advantage]

## PRICING STRATEGY
[Price positioning recommendations]

## UNIQUE VALUE PROPS
[Top 3 USPs to emphasize everywhere]

Make it compelling and specific to Amazon buyers."""

class ResponseParser:
    """Parse and structure AI responses for Amazon optimization"""
    
    @staticmethod
    def parse_review_analysis(content: str) -> Dict[str, Any]:
        """Parse review analysis response"""
        
        parsed = {
            'title_optimization': '',
            'bullet_gaps': [],
            'a9_insights': '',
            'image_requirements': '',
            'conversion_killers': [],
            'competitor_advantages': '',
            'immediate_actions': [],
            'review_templates': ''
        }
        
        try:
            # Extract sections using regex
            sections = {
                'title_optimization': r'## TITLE OPTIMIZATION OPPORTUNITIES\s*\n(.*?)(?=## |$)',
                'bullet_gaps': r'## BULLET POINT GAPS\s*\n(.*?)(?=## |$)',
                'a9_insights': r'## A9 ALGORITHM INSIGHTS\s*\n(.*?)(?=## |$)',
                'image_requirements': r'## MAIN IMAGE REQUIREMENTS\s*\n(.*?)(?=## |$)',
                'conversion_killers': r'## CONVERSION KILLERS\s*\n(.*?)(?=## |$)',
                'competitor_advantages': r'## COMPETITOR ADVANTAGES\s*\n(.*?)(?=## |$)',
                'immediate_actions': r'## IMMEDIATE ACTIONS.*?\s*\n(.*?)(?=## |$)',
                'review_templates': r'## REVIEW RESPONSE TEMPLATES\s*\n(.*?)(?=## |$)'
            }
            
            for section_name, pattern in sections.items():
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    section_content = match.group(1).strip()
                    
                    # Parse lists for certain sections
                    if section_name in ['bullet_gaps', 'conversion_killers', 'immediate_actions']:
                        # Extract numbered or bulleted lists
                        items = []
                        lines = section_content.split('\n')
                        for line in lines:
                            line = line.strip()
                            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                                # Clean up the line
                                cleaned = re.sub(r'^[\d\.\-•\s]+', '', line).strip()
                                if cleaned:
                                    items.append(cleaned)
                        parsed[section_name] = items if items else [section_content]
                    else:
                        parsed[section_name] = section_content
            
            return parsed
            
        except Exception as e:
            logger.error(f"Error parsing review analysis: {str(e)}")
            return parsed
    
    @staticmethod
    def parse_listing_optimization(content: str) -> Dict[str, Any]:
        """Parse listing optimization response"""
        
        parsed = {
            'new_title': '',
            'optimized_bullets': [],
            'a_plus_modules': [],
            'backend_keywords': '',
            'faq_section': []
        }
        
        try:
            sections = {
                'new_title': r'## NEW TITLE.*?\s*\n(.*?)(?=## |$)',
                'optimized_bullets': r'## OPTIMIZED BULLETS.*?\s*\n(.*?)(?=## |$)',
                'a_plus_modules': r'## A\+ CONTENT MODULES\s*\n(.*?)(?=## |$)',
                'backend_keywords': r'## BACKEND KEYWORDS\s*\n(.*?)(?=## |$)',
                'faq_section': r'## FAQ SECTION.*?\s*\n(.*?)(?=## |$)'
            }
            
            for section_name, pattern in sections.items():
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    section_content = match.group(1).strip()
                    
                    if section_name in ['optimized_bullets', 'a_plus_modules']:
                        # Extract bullet points
                        items = []
                        lines = section_content.split('\n')
                        for line in lines:
                            line = line.strip()
                            if line and (line.startswith('•') or line.startswith('-') or line[0].isdigit()):
                                cleaned = re.sub(r'^[•\-\d\.\s]+', '', line).strip()
                                if cleaned:
                                    items.append(cleaned)
                        parsed[section_name] = items
                    elif section_name == 'faq_section':
                        # Parse Q&A pairs
                        qa_pairs = []
                        lines = section_content.split('\n')
                        current_q = None
                        current_a = None
                        
                        for line in lines:
                            line = line.strip()
                            if line.startswith('Q') and ':' in line:
                                if current_q and current_a:
                                    qa_pairs.append({'question': current_q, 'answer': current_a})
                                current_q = line.split(':', 1)[1].strip()
                                current_a = None
                            elif line.startswith('A') and ':' in line and current_q:
                                current_a = line.split(':', 1)[1].strip()
                        
                        if current_q and current_a:
                            qa_pairs.append({'question': current_q, 'answer': current_a})
                        
                        parsed[section_name] = qa_pairs
                    else:
                        parsed[section_name] = section_content
            
            return parsed
            
        except Exception as e:
            logger.error(f"Error parsing listing optimization: {str(e)}")
            return parsed

class EnhancedAIAnalyzer:
    """Main AI analyzer class optimized for Amazon listings"""
    
    def __init__(self):
        self.api_client = APIClient()
        self.prompt_templates = PromptTemplates()
        self.response_parser = ResponseParser()
        
        if self.api_client.is_available():
            logger.info("Enhanced AI Analyzer initialized - API available")
        else:
            logger.warning("Enhanced AI Analyzer initialized - API not available")
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get API availability status"""
        if not self.api_client.api_key:
            return {
                'available': False,
                'error': 'API key not configured',
                'suggestions': [
                    'Add OPENAI_API_KEY to environment variables',
                    'Add openai_api_key to Streamlit secrets'
                ]
            }
        
        if not has_requests:
            return {
                'available': False,
                'error': 'Requests module not available',
                'suggestions': ['Install requests module: pip install requests']
            }
        
        # Test API with minimal call
        test_response = self.api_client.call_api([
            {"role": "user", "content": "Test connection"}
        ], max_tokens=10)
        
        return {
            'available': test_response['success'],
            'error': test_response.get('error'),
            'model': test_response.get('model', 'gpt-4o')
        }
    
    def analyze_reviews_comprehensive(self, product_info: Dict[str, Any],
                                    reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Comprehensive AI analysis of reviews for Amazon optimization"""
        
        if not self.api_client.is_available():
            return {
                'success': False,
                'error': 'AI analysis not available',
                'ai_analysis_available': False
            }
        
        try:
            # Prepare review summaries
            review_summaries = self._prepare_review_summaries(reviews)
            
            if not review_summaries:
                return {
                    'success': False,
                    'error': 'No review data to analyze',
                    'ai_analysis_available': False
                }
            
            # Create analysis prompt
            prompt = self.prompt_templates.create_review_analysis_prompt(
                product_info, review_summaries
            )
            
            # Make API call with Amazon-specific system prompt
            response = self.api_client.call_api([
                {"role": "system", "content": """You are the top Amazon listing optimization expert with deep knowledge of:
                - A9 algorithm and ranking factors
                - Conversion rate optimization for Amazon
                - Keyword research from customer language
                - Competitive positioning on Amazon
                - Review management strategies
                
                Always provide specific, actionable recommendations that can be implemented immediately.
                Focus on changes that will improve BSR and conversion rate within days."""},
                {"role": "user", "content": prompt}
            ])
            
            if not response['success']:
                return {
                    'success': False,
                    'error': response['error'],
                    'ai_analysis_available': False
                }
            
            # Parse response
            analysis_results = self.response_parser.parse_review_analysis(response['result'])
            
            # Add metadata
            analysis_results.update({
                'success': True,
                'ai_analysis_available': True,
                'analysis_timestamp': datetime.now().isoformat(),
                'reviews_analyzed': len(reviews),
                'model_used': response.get('model', 'gpt-4o'),
                'raw_response': response['result']  # For debugging
            })
            
            logger.info(f"AI analysis completed for {len(reviews)} reviews")
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'ai_analysis_available': False
            }
    
    def generate_listing_optimization(self, metrics: Dict[str, Any],
                                    top_issues: List[str]) -> Dict[str, Any]:
        """Generate specific listing optimization recommendations"""
        
        if not self.api_client.is_available():
            return {
                'success': False,
                'error': 'AI optimization not available'
            }
        
        try:
            # Create optimization prompt
            prompt = self.prompt_templates.create_listing_optimization_prompt(
                metrics, top_issues
            )
            
            # Make API call
            response = self.api_client.call_api([
                {"role": "system", "content": """You are an Amazon copywriting expert who writes listing copy that:
                - Ranks on page 1 for target keywords
                - Converts browsers into buyers
                - Addresses customer objections preemptively
                - Uses emotional triggers effectively
                - Follows Amazon's style guidelines
                
                Every word must serve a purpose. Be specific and compelling."""},
                {"role": "user", "content": prompt}
            ])
            
            if not response['success']:
                return {
                    'success': False,
                    'error': response['error']
                }
            
            # Parse response
            optimization_results = self.response_parser.parse_listing_optimization(response['result'])
            
            # Add metadata
            optimization_results.update({
                'success': True,
                'generation_timestamp': datetime.now().isoformat(),
                'model_used': response.get('model', 'gpt-4o')
            })
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Error in listing optimization: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def analyze_competitive_position(self, competitor_mentions: List[str],
                                   product_strengths: List[str]) -> Dict[str, Any]:
        """Analyze competitive positioning"""
        
        if not self.api_client.is_available():
            return {
                'success': False,
                'error': 'AI competitive analysis not available'
            }
        
        try:
            # Create competitive analysis prompt
            prompt = self.prompt_templates.create_competitive_analysis_prompt(
                competitor_mentions, product_strengths
            )
            
            # Make API call
            response = self.api_client.call_api([
                {"role": "system", "content": """You are an Amazon competitive strategy expert who understands:
                - How to differentiate products on Amazon
                - Price-value positioning
                - Creating compelling comparison content
                - Winning the Buy Box
                - Building brand loyalty on Amazon
                
                Create strategies that make competitors irrelevant."""},
                {"role": "user", "content": prompt}
            ])
            
            if not response['success']:
                return {
                    'success': False,
                    'error': response['error']
                }
            
            # For now, return raw response (can add parser later)
            return {
                'success': True,
                'analysis': response['result'],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in competitive analysis: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _prepare_review_summaries(self, reviews: List[Dict[str, Any]]) -> List[str]:
        """Prepare review summaries for AI analysis"""
        summaries = []
        
        # Sort by rating (lowest first for issue identification)
        sorted_reviews = sorted(reviews, key=lambda x: x.get('rating', 3))
        
        # Take up to 15 reviews (balanced sample)
        selected_reviews = sorted_reviews[:15]
        
        for i, review in enumerate(selected_reviews, 1):
            try:
                text = review.get('text', '').strip()
                if not text:
                    continue
                
                # Limit text length
                text = text[:400]
                
                # Add rating if available
                rating_text = ""
                if review.get('rating') is not None:
                    rating_text = f" (Rating: {review['rating']}/5)"
                
                # Add verification if available
                verified_text = ""
                if review.get('verified') == 'Verified Purchase':
                    verified_text = " [Verified]"
                
                summary = f"Review {i}{rating_text}{verified_text}: {text}"
                summaries.append(summary)
                
            except Exception as e:
                logger.warning(f"Error preparing review summary: {str(e)}")
                continue
        
        return summaries
    
    def analyze_product_comprehensive(self, product_info: Dict[str, Any],
                                    reviews: List[Dict[str, Any]] = None,
                                    returns: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Comprehensive product analysis (main interface)"""
        
        # Combine reviews and returns
        all_feedback = []
        
        if reviews:
            all_feedback.extend(reviews)
        
        if returns:
            # Convert returns to review format
            for return_item in returns:
                feedback_item = {
                    'text': return_item.get('return_reason', ''),
                    'type': 'return',
                    'rating': 1,  # Assume low satisfaction for returns
                    'date': return_item.get('date', ''),
                    'source': 'return_data'
                }
                all_feedback.append(feedback_item)
        
        if not all_feedback:
            return {
                'success': False,
                'error': 'No feedback data provided',
                'ai_analysis_available': False
            }
        
        # Run comprehensive analysis
        return self.analyze_reviews_comprehensive(product_info, all_feedback)

# Export main class
__all__ = ['EnhancedAIAnalyzer', 'APIClient']
