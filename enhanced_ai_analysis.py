"""
Enhanced AI Analysis Module - Dual AI with Extreme Mode
Version 12.0 - OpenAI + Claude with Batch Processing

Key Features:
- Dual AI support (OpenAI GPT-4 + Claude Sonnet)
- Extreme mode for difficult categorizations (2000+ tokens)
- Batch processing with checkpoints for stability
- Quality pattern recognition
- Consensus-based categorization
"""

import logging
import os
import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
import time
import pickle

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
API_TIMEOUT = 60  # Increased for extreme mode
MAX_RETRIES = 3
BATCH_SIZE = 50  # Smaller batches for stability

# Token configurations by mode
TOKEN_LIMITS = {
    'standard': 300,
    'enhanced': 800,
    'extreme': 2500  # Increased for very detailed analysis
}

# Model configurations
MODELS = {
    'openai': {
        'standard': 'gpt-3.5-turbo',
        'enhanced': 'gpt-4',
        'extreme': 'gpt-4'  # Always use GPT-4 for extreme mode
    },
    'claude': {
        'standard': 'claude-3-haiku-20240307',
        'enhanced': 'claude-3-sonnet-20240229',
        'extreme': 'claude-3-sonnet-20240229'
    }
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

# Quality patterns for root cause analysis
QUALITY_PATTERNS = {
    'Material Failure': [
        'velcro', 'strap', 'fabric', 'material', 'stitching', 'seam', 'tear', 'rip', 'worn', 
        'fraying', 'deteriorate', 'degraded', 'cloth', 'leather', 'vinyl', 'plastic cracked'
    ],
    'Component Failure': [
        'button', 'buckle', 'wheel', 'handle', 'pump', 'valve', 'motor', 'battery', 'suction',
        'mechanism', 'joint', 'hinge', 'lock', 'latch', 'connector', 'tubing', 'cord'
    ],
    'Design Flaw': [
        'too heavy', 'hard to use', 'difficult to adjust', 'poor design', 'awkward', 'cumbersome',
        'not intuitive', 'poorly designed', 'bad ergonomics', 'uncomfortable shape', 'wrong angle'
    ],
    'Manufacturing Defect': [
        'broken on arrival', 'defective', 'missing parts', 'not assembled correctly', 'crooked',
        'uneven', 'loose screws', 'poor assembly', 'manufacturing error', 'quality control'
    ],
    'Durability Issue': [
        'broke after', 'lasted only', 'stopped working', 'fell apart', 'wore out', 'degraded',
        'short lifespan', 'not durable', 'cheaply made', 'fragile', 'flimsy construction'
    ],
    'Safety Concern': [
        'unsafe', 'dangerous', 'injury', 'hurt', 'accident', 'risk', 'hazard', 'sharp edge',
        'cut', 'pinch', 'trap', 'unstable', 'tip over', 'collapse', 'malfunction danger'
    ]
}

# Severity keywords for medical devices
SEVERITY_KEYWORDS = {
    'critical': [
        'injury', 'injured', 'hospital', 'emergency', 'doctor', 'dangerous', 'unsafe', 
        'hazard', 'accident', 'wound', 'bleeding', 'pain severe', 'medical attention',
        'urgent care', 'health risk', 'safety issue', 'burn', 'cut', 'bruise', 'fall',
        'allergic reaction', 'infection', 'hospitalized', 'ambulance', 'poison', 'toxic'
    ],
    'major': [
        'defective', 'broken', 'malfunction', 'unusable', 'failed', 'stopped working',
        "doesn't work", 'not working', 'poor quality', 'fell apart', 'unreliable',
        'safety concern', 'risk', 'worried', 'concerned about safety', 'completely broken'
    ],
    'minor': [
        'uncomfortable', 'difficult', 'confusing', 'disappointed', 'not ideal',
        'could be better', 'minor issue', 'small problem', 'slight', 'somewhat'
    ]
}

class AIProvider(Enum):
    OPENAI = "openai"
    CLAUDE = "claude"
    BOTH = "both"

def detect_language(text: str) -> str:
    """Simple language detection"""
    spanish_indicators = ['ñ', 'á', 'é', 'í', 'ó', 'ú', ' es ', ' la ', ' el ', ' los ', ' las ', ' de ', ' que ', ' y ']
    spanish_count = sum(1 for indicator in spanish_indicators if indicator in text.lower())
    return 'es' if spanish_count >= 3 else 'en'

def translate_text(text: str, source_lang: str = 'auto', target_lang: str = 'en') -> str:
    """Placeholder for translation - implement if needed"""
    return text

def calculate_confidence(complaint: str, category: str, language: str = 'en', consensus: bool = False) -> float:
    """Calculate confidence score for categorization"""
    confidence = 0.5  # Base confidence
    
    # Boost for consensus
    if consensus:
        confidence = 0.95
    else:
        confidence = 0.85
    
    # Adjust based on complaint length
    word_count = len(complaint.split())
    if word_count > 20:
        confidence += 0.05
    elif word_count < 5:
        confidence -= 0.1
    
    # Adjust for category specificity
    if category != 'Other/Miscellaneous':
        confidence += 0.05
    else:
        confidence -= 0.1
    
    # Adjust for non-English
    if language != 'en':
        confidence -= 0.05
    
    return max(0.1, min(1.0, confidence))

def detect_severity(complaint: str, category: str) -> str:
    """Detect severity level of complaint"""
    complaint_lower = complaint.lower()
    
    # Check for critical keywords first
    for keyword in SEVERITY_KEYWORDS['critical']:
        if keyword in complaint_lower:
            return 'critical'
    
    # Check if medical/health category
    if category == 'Medical/Health Concerns':
        return 'critical'
    
    # Check for major keywords
    for keyword in SEVERITY_KEYWORDS['major']:
        if keyword in complaint_lower:
            return 'major'
    
    # Check if quality defect
    if category in ['Product Defects/Quality', 'Performance/Effectiveness']:
        return 'major'
    
    # Check for minor keywords
    for keyword in SEVERITY_KEYWORDS['minor']:
        if keyword in complaint_lower:
            return 'minor'
    
    return 'none'

def is_duplicate(complaint1: str, complaint2: str, threshold: float = 0.85) -> bool:
    """Placeholder - no duplicate detection as requested"""
    return False

def extract_quality_patterns(complaint: str, category: str) -> Dict[str, Any]:
    """Extract specific quality patterns from complaints"""
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
        # Sort by position and take the first one
        patterns_found.sort(key=lambda x: x['position'])
        root_cause = patterns_found[0]['pattern']
    
    is_safety_critical = any(p['pattern'] == 'Safety Concern' for p in patterns_found)
    
    return {
        'patterns': patterns_found,
        'root_cause': root_cause,
        'is_safety_critical': is_safety_critical
    }

def get_quality_recommendation(root_cause: str, frequency: int, products: List[str]) -> str:
    """Generate specific quality recommendations based on root cause"""
    product_list = ', '.join(products[:3]) + (f' (+{len(products)-3} more)' if len(products) > 3 else '')
    
    recommendations = {
        'Material Failure': f"IMMEDIATE ACTION: Conduct material testing and supplier audit for {product_list}. {frequency} failures indicate potential batch/supplier issue requiring vendor corrective action.",
        
        'Component Failure': f"ENGINEERING REVIEW: Component design verification required for {product_list}. Implement incoming inspection protocols. {frequency} failures suggest design tolerance or supplier quality issue.",
        
        'Design Flaw': f"DESIGN CONTROL: Initiate formal design review per ISO 13485 for {product_list}. Consider usability study and risk analysis update. {frequency} complaints indicate user interface improvement needed.",
        
        'Manufacturing Defect': f"PRODUCTION CONTROL: Investigation of manufacturing processes required. Review last {frequency} production lots for {product_list}. Implement enhanced process monitoring and operator training.",
        
        'Durability Issue': f"VERIFICATION & VALIDATION: Conduct accelerated life testing per relevant standards. Review warranty data trends. {frequency} durability failures may require material upgrade or design reinforcement.",
        
        'Safety Concern': f"URGENT - RISK MANAGEMENT: Immediate risk assessment per ISO 14971 required. {frequency} safety complaints for {product_list} may trigger post-market surveillance action or field safety notice evaluation."
    }
    
    return recommendations.get(root_cause, f"Investigate {frequency} occurrences of {root_cause} pattern affecting {product_list}.")

class BatchProcessor:
    """Handle batch processing with checkpoints for large datasets"""
    
    def __init__(self, batch_size: int = BATCH_SIZE):
        self.batch_size = batch_size
        self.checkpoint_dir = "checkpoints"
        self._ensure_checkpoint_dir()
    
    def _ensure_checkpoint_dir(self):
        """Ensure checkpoint directory exists"""
        try:
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)
        except Exception as e:
            logger.warning(f"Could not create checkpoint directory: {e}")
    
    def save_checkpoint(self, data: Dict, checkpoint_file: str):
        """Save processing checkpoint"""
        try:
            filepath = os.path.join(self.checkpoint_dir, checkpoint_file)
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved checkpoint: {len(data.get('processed', []))} items")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(self, checkpoint_file: str) -> Optional[Dict]:
        """Load processing checkpoint"""
        try:
            filepath = os.path.join(self.checkpoint_dir, checkpoint_file)
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                logger.info(f"Loaded checkpoint: {len(data.get('processed', []))} items")
                return data
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
        return None
    
    def get_checkpoint_file(self, prefix: str = "categorization") -> str:
        """Generate checkpoint filename with timestamp"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{prefix}_checkpoint_{timestamp}.pkl"
    
    def process_in_batches(self, items: List, processor_func, checkpoint_interval: int = 5):
        """Process items in batches with periodic checkpoints"""
        total_items = len(items)
        processed_items = []
        checkpoint_file = self.get_checkpoint_file()
        batch_count = 0
        
        for i in range(0, total_items, self.batch_size):
            batch = items[i:i + self.batch_size]
            batch_results = []
            
            # Process batch
            for item in batch:
                try:
                    result = processor_func(item)
                    batch_results.append(result)
                except Exception as e:
                    logger.error(f"Error processing item: {e}")
                    batch_results.append(None)
            
            processed_items.extend(batch_results)
            batch_count += 1
            
            # Save checkpoint periodically
            if batch_count % checkpoint_interval == 0:
                checkpoint_data = {
                    'processed': processed_items,
                    'total_items': total_items,
                    'batch_count': batch_count,
                    'timestamp': datetime.now().isoformat()
                }
                self.save_checkpoint(checkpoint_data, checkpoint_file)
        
        return processed_items

class EnhancedAIAnalyzer:
    """Main AI analyzer with dual AI support and extreme mode"""
    
    def __init__(self, provider: AIProvider = AIProvider.BOTH):
        self.provider = provider
        self.openai_key = self._get_api_key('openai')
        self.claude_key = self._get_api_key('claude')
        
        # Initialize API clients
        self.openai_configured = bool(self.openai_key and has_requests)
        self.claude_configured = bool(self.claude_key and has_anthropic)
        
        if self.claude_configured:
            try:
                self.claude_client = anthropic.Anthropic(api_key=self.claude_key)
            except Exception as e:
                logger.error(f"Failed to initialize Claude client: {e}")
                self.claude_configured = False
                self.claude_client = None
        else:
            self.claude_client = None
        
        # Usage tracking
        self.usage_stats = {
            'total_calls': 0,
            'openai_calls': 0,
            'claude_calls': 0,
            'consensus_achieved': 0,
            'fallback_used': 0
        }
        
        logger.info(f"AI Analyzer initialized - OpenAI: {self.openai_configured}, Claude: {self.claude_configured}, Provider: {provider.value}")
    
    def _get_api_key(self, provider: str) -> Optional[str]:
        """Get API key from multiple sources with comprehensive search"""
        # Try Streamlit secrets first
        try:
            import streamlit as st
            if hasattr(st, 'secrets'):
                if provider == 'openai':
                    for key_name in ["openai_api_key", "OPENAI_API_KEY", "openai", "api_key"]:
                        if key_name in st.secrets:
                            key_value = str(st.secrets[key_name]).strip()
                            if key_value.startswith('sk-'):
                                logger.info(f"Found OpenAI key in Streamlit secrets ({key_name})")
                                return key_value
                elif provider == 'claude':
                    for key_name in ["anthropic_api_key", "ANTHROPIC_API_KEY", "claude_api_key", "claude"]:
                        if key_name in st.secrets:
                            key_value = str(st.secrets[key_name]).strip()
                            if key_value.startswith('sk-ant-'):
                                logger.info(f"Found Claude key in Streamlit secrets ({key_name})")
                                return key_value
        except Exception as e:
            logger.debug(f"Streamlit secrets not available: {e}")
        
        # Try environment variables
        if provider == 'openai':
            for env_name in ["OPENAI_API_KEY", "OPENAI_API", "API_KEY"]:
                api_key = os.environ.get(env_name, '').strip()
                if api_key and api_key.startswith('sk-'):
                    logger.info(f"Found OpenAI key in environment ({env_name})")
                    return api_key
        elif provider == 'claude':
            for env_name in ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"]:
                api_key = os.environ.get(env_name, '').strip()
                if api_key and api_key.startswith('sk-ant-'):
                    logger.info(f"Found Claude key in environment ({env_name})")
                    return api_key
        
        logger.warning(f"No valid {provider} API key found")
        return None
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get comprehensive API availability status"""
        status = {
            'available': self.openai_configured or self.claude_configured,
            'openai_configured': self.openai_configured,
            'claude_configured': self.claude_configured,
            'dual_ai_available': self.openai_configured and self.claude_configured,
            'provider': self.provider.value,
            'extreme_mode_available': self.openai_configured or self.claude_configured,
            'message': '',
            'capabilities': []
        }
        
        if status['dual_ai_available']:
            status['message'] = 'Dual AI ready: OpenAI + Claude consensus mode available'
            status['capabilities'] = ['Consensus categorization', 'Extreme mode', 'Quality pattern recognition']
        elif self.openai_configured:
            status['message'] = 'OpenAI ready (Claude not configured)'
            status['capabilities'] = ['OpenAI categorization', 'Extreme mode', 'Quality patterns']
        elif self.claude_configured:
            status['message'] = 'Claude ready (OpenAI not configured)'
            status['capabilities'] = ['Claude categorization', 'Extreme mode', 'Quality patterns']
        else:
            status['message'] = 'No APIs configured - unable to process'
            status['available'] = False
        
        return status
    
    def _call_openai(self, prompt: str, system_prompt: str, mode: str = 'standard') -> Optional[str]:
        """Call OpenAI API with comprehensive error handling"""
        if not self.openai_configured:
            return None
        
        model = MODELS['openai'][mode]
        max_tokens = TOKEN_LIMITS[mode]
        
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
            "max_tokens": max_tokens,
            "top_p": 0.95,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
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
                    self.usage_stats['openai_calls'] += 1
                    self.usage_stats['total_calls'] += 1
                    return content
                    
                elif response.status_code == 429:
                    # Rate limit - wait and retry
                    wait_time = min(2 ** attempt * 2, 30)
                    logger.warning(f"OpenAI rate limited, waiting {wait_time}s")
                    time.sleep(wait_time)
                    continue
                    
                else:
                    logger.error(f"OpenAI API error {response.status_code}: {response.text}")
                    if attempt == MAX_RETRIES - 1:
                        return None
                    time.sleep(2 ** attempt)
                    
            except requests.exceptions.Timeout:
                logger.warning(f"OpenAI timeout on attempt {attempt + 1}")
                if attempt == MAX_RETRIES - 1:
                    return None
                time.sleep(2 ** attempt)
                
            except Exception as e:
                logger.error(f"OpenAI call error: {e}")
                if attempt == MAX_RETRIES - 1:
                    return None
                time.sleep(2 ** attempt)
        
        return None
    
    def _call_claude(self, prompt: str, system_prompt: str, mode: str = 'standard') -> Optional[str]:
        """Call Claude API with comprehensive error handling"""
        if not self.claude_configured or not self.claude_client:
            return None
        
        model = MODELS['claude'][mode]
        max_tokens = TOKEN_LIMITS[mode]
        
        for attempt in range(MAX_RETRIES):
            try:
                response = self.claude_client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=0.1,
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                content = response.content[0].text.strip()
                self.usage_stats['claude_calls'] += 1
                self.usage_stats['total_calls'] += 1
                return content
                
            except Exception as e:
                logger.error(f"Claude call error (attempt {attempt + 1}): {e}")
                if attempt == MAX_RETRIES - 1:
                    return None
                time.sleep(2 ** attempt)
        
        return None
    
    def categorize_return(self, complaint: str, fba_reason: str = None, mode: str = 'standard') -> Tuple[str, float, str, str]:
        """
        Main categorization method with dual AI consensus
        
        Returns:
            Tuple[category, confidence, severity, language]
        """
        if not complaint or not complaint.strip():
            return 'Other/Miscellaneous', 0.1, 'none', 'en'
        
        # Detect language
        language = detect_language(complaint)
        
        # Check FBA mapping first for quick categorization
        if fba_reason and fba_reason in FBA_REASON_MAP:
            category = FBA_REASON_MAP[fba_reason]
            severity = detect_severity(complaint, category)
            confidence = calculate_confidence(complaint, category, language)
            logger.info(f"FBA mapping used: {fba_reason} -> {category}")
            return category, confidence, severity, language
        
        # Determine processing mode based on complaint complexity
        if mode == 'standard' and len(complaint.split()) > 50:
            mode = 'enhanced'  # Auto-upgrade for complex complaints
        
        # Build prompts
        system_prompt = self._build_system_prompt(mode)
        user_prompt = self._build_user_prompt(complaint, fba_reason, mode)
        
        # Get responses from available AIs
        openai_result = None
        claude_result = None
        
        if self.provider in [AIProvider.OPENAI, AIProvider.BOTH] and self.openai_configured:
            openai_result = self._call_openai(user_prompt, system_prompt, mode)
            if openai_result:
                openai_result = self._clean_category_response(openai_result)
        
        if self.provider in [AIProvider.CLAUDE, AIProvider.BOTH] and self.claude_configured:
            claude_result = self._call_claude(user_prompt, system_prompt, mode)
            if claude_result:
                claude_result = self._clean_category_response(claude_result)
        
        # Determine final category with consensus logic
        category, confidence = self._determine_consensus(openai_result, claude_result, complaint, language)
        severity = detect_severity(complaint, category)
        
        # Log result for debugging
        if openai_result and claude_result:
            consensus = openai_result == claude_result
            if consensus:
                self.usage_stats['consensus_achieved'] += 1
            logger.info(f"Dual AI: OpenAI='{openai_result}', Claude='{claude_result}', Consensus={consensus}, Final='{category}'")
        
        return category, confidence, severity, language
    
    def _build_system_prompt(self, mode: str) -> str:
        """Build system prompt based on processing mode"""
        base_prompt = """You are a medical device quality expert specializing in categorizing product returns.

CRITICAL INSTRUCTIONS:
1. You MUST respond with EXACTLY one category name from the provided list
2. Copy the category name EXACTLY as written (including capitalization and punctuation)
3. Do not add quotes, explanations, or any other text
4. Choose the MOST SPECIFIC category that fits the primary complaint"""
        
        if mode == 'extreme':
            base_prompt += """

EXTREME ANALYSIS MODE:
- Perform deep linguistic analysis of the complaint
- Consider multiple interpretations and contexts
- Analyze sentiment, intent, and underlying issues
- Apply medical device regulatory knowledge
- Consider patient safety implications
- Use advanced reasoning to handle ambiguous cases"""
        
        return base_prompt
    
    def _build_user_prompt(self, complaint: str, fba_reason: str, mode: str) -> str:
        """Build user prompt based on processing mode"""
        
        categories_list = '\n'.join(f'- {cat}' for cat in MEDICAL_DEVICE_CATEGORIES)
        
        if mode == 'extreme':
            prompt = f"""MEDICAL DEVICE RETURN ANALYSIS - EXTREME MODE

COMPLAINT TEXT: "{complaint}"
"""
            if fba_reason:
                prompt += f"AMAZON FBA REASON CODE: {fba_reason}\n"
            
            prompt += f"""
AVAILABLE CATEGORIES:
{categories_list}

COMPREHENSIVE ANALYSIS FRAMEWORK:

1. LINGUISTIC ANALYSIS:
   - Parse the complaint for explicit and implicit meanings
   - Identify primary vs. secondary issues mentioned
   - Analyze emotional context and severity indicators

2. MEDICAL DEVICE CONTEXT:
   - Consider the medical/therapeutic purpose
   - Evaluate patient safety implications
   - Assess impact on intended use

3. ROOT CAUSE IDENTIFICATION:
   - Distinguish between product, user, or external factors
   - Identify whether issue is design, manufacturing, or usage-related
   - Consider regulatory classification implications

4. CATEGORY MAPPING DECISION TREE:
   - If physical dimensions mentioned → Size/Fit Issues
   - If discomfort during use → Comfort Issues
   - If broken/defective/malfunction → Product Defects/Quality
   - If doesn't achieve therapeutic goal → Performance/Effectiveness
   - If product moves/slides/unstable → Stability/Positioning Issues
   - If incompatible with other equipment → Equipment Compatibility
   - If design criticism or material issues → Design/Material Issues
   - If wrong item or misunderstood → Wrong Product/Misunderstanding
   - If missing parts/instructions → Missing Components
   - If customer error or changed mind → Customer Error/Changed Mind
   - If shipping/delivery problems → Shipping/Fulfillment Issues
   - If setup/usage difficulties → Assembly/Usage Difficulty
   - If health/safety/medical concerns → Medical/Health Concerns
   - If price/value related → Price/Value
   - If none clearly fit → Other/Miscellaneous

5. CONFIDENCE VALIDATION:
   - Verify the selected category against complaint details
   - Ensure no better-fitting category exists
   - Consider edge cases and overlapping categories

Based on this comprehensive analysis, respond with the EXACT category name that best fits the primary complaint."""
        
        else:
            # Standard or enhanced mode
            prompt = f"""Categorize this medical device return complaint.

COMPLAINT: "{complaint}"
"""
            if fba_reason:
                prompt += f"FBA REASON: {fba_reason}\n"
            
            prompt += f"""
CATEGORIES:
{categories_list}

Instructions:
- Choose the category that best matches the PRIMARY reason for return
- If multiple issues exist, focus on the most significant one
- For medical devices, consider safety and intended use
- Respond with ONLY the exact category name"""
        
        return prompt
    
    def _clean_category_response(self, response: str) -> str:
        """Clean and validate AI response to extract valid category"""
        if not response:
            return 'Other/Miscellaneous'
        
        # Remove common prefixes and suffixes
        response = response.strip()
        response = re.sub(r'^(Category:|The category is:?|Answer:|Response:)\s*', '', response, flags=re.IGNORECASE)
        response = re.sub(r'^["\'-]+|["\'-]+$', '', response)  # Remove quotes
        response = response.strip()
        
        # Try exact match first
        for valid_cat in MEDICAL_DEVICE_CATEGORIES:
            if response == valid_cat:
                return valid_cat
        
        # Try case-insensitive match
        response_lower = response.lower()
        for valid_cat in MEDICAL_DEVICE_CATEGORIES:
            if response_lower == valid_cat.lower():
                return valid_cat
        
        # Try partial matching
        for valid_cat in MEDICAL_DEVICE_CATEGORIES:
            # Check if the valid category is contained in the response
            if valid_cat.lower() in response_lower:
                return valid_cat
            # Check if the response is contained in the valid category
            if response_lower in valid_cat.lower():
                return valid_cat
        
        # Try fuzzy matching for common variations
        category_variations = {
            'size': 'Size/Fit Issues',
            'comfort': 'Comfort Issues',
            'quality': 'Product Defects/Quality',
            'defect': 'Product Defects/Quality',
            'performance': 'Performance/Effectiveness',
            'stability': 'Stability/Positioning Issues',
            'compatibility': 'Equipment Compatibility',
            'design': 'Design/Material Issues',
            'wrong': 'Wrong Product/Misunderstanding',
            'missing': 'Missing Components',
            'customer': 'Customer Error/Changed Mind',
            'shipping': 'Shipping/Fulfillment Issues',
            'assembly': 'Assembly/Usage Difficulty',
            'medical': 'Medical/Health Concerns',
            'price': 'Price/Value'
        }
        
        for keyword, category in category_variations.items():
            if keyword in response_lower:
                return category
        
        logger.warning(f"Could not match AI response to valid category: '{response}'")
        return 'Other/Miscellaneous'
    
    def _determine_consensus(self, openai_result: Optional[str], claude_result: Optional[str], 
                           complaint: str, language: str) -> Tuple[str, float]:
        """Determine final category based on AI consensus"""
        
        if openai_result and claude_result:
            # Both AIs responded
            if openai_result == claude_result:
                # Perfect consensus
                confidence = calculate_confidence(complaint, openai_result, language, consensus=True)
                return openai_result, confidence
            else:
                # Disagreement - use logic to resolve
                logger.info(f"AI disagreement: OpenAI='{openai_result}', Claude='{claude_result}'")
                
                # Prefer non-"Other" categories
                if openai_result != 'Other/Miscellaneous' and claude_result == 'Other/Miscellaneous':
                    category = openai_result
                elif claude_result != 'Other/Miscellaneous' and openai_result == 'Other/Miscellaneous':
                    category = claude_result
                else:
                    # Both are specific categories - prefer OpenAI (can be customized)
                    category = openai_result
                
                confidence = calculate_confidence(complaint, category, language, consensus=False) * 0.8  # Reduced confidence
                return category, confidence
        
        elif openai_result:
            # Only OpenAI responded
            confidence = calculate_confidence(complaint, openai_result, language)
            return openai_result, confidence
        
        elif claude_result:
            # Only Claude responded
            confidence = calculate_confidence(complaint, claude_result, language)
            return claude_result, confidence
        
        else:
            # No AI responses - use fallback
            self.usage_stats['fallback_used'] += 1
            logger.warning(f"No AI responses, using fallback for: '{complaint[:50]}...'")
            return self._fallback_categorize(complaint), 0.3
    
    def _fallback_categorize(self, complaint: str) -> str:
        """Simple keyword-based fallback categorization"""
        complaint_lower = complaint.lower()
        
        # Simple keyword mapping
        keyword_map = {
            'size': 'Size/Fit Issues',
            'small': 'Size/Fit Issues', 
            'large': 'Size/Fit Issues',
            'fit': 'Size/Fit Issues',
            'comfort': 'Comfort Issues',
            'uncomfortable': 'Comfort Issues',
            'pain': 'Comfort Issues',
            'broken': 'Product Defects/Quality',
            'defective': 'Product Defects/Quality',
            'quality': 'Product Defects/Quality',
            'work': 'Performance/Effectiveness',
            'wrong': 'Wrong Product/Misunderstanding',
            'mistake': 'Customer Error/Changed Mind',
            'shipping': 'Shipping/Fulfillment Issues'
        }
        
        for keyword, category in keyword_map.items():
            if keyword in complaint_lower:
                return category
        
        return 'Other/Miscellaneous'
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        total_calls = self.usage_stats['total_calls']
        return {
            'total_calls': total_calls,
            'openai_calls': self.usage_stats['openai_calls'],
            'claude_calls': self.usage_stats['claude_calls'],
            'consensus_rate': (self.usage_stats['consensus_achieved'] / max(1, total_calls)) * 100,
            'fallback_rate': (self.usage_stats['fallback_used'] / max(1, total_calls)) * 100,
            'dual_ai_enabled': self.openai_configured and self.claude_configured
        }

def generate_quality_insights(df, reason_summary: Dict, product_summary: Dict) -> Dict[str, Any]:
    """Generate comprehensive quality insights from categorized data"""
    
    insights = {
        'risk_assessment': {},
        'root_cause_distribution': {},
        'action_items': [],
        'product_analysis': {},
        'safety_alerts': []
    }
    
    quality_categories = [
        'Product Defects/Quality',
        'Performance/Effectiveness',
        'Design/Material Issues',
        'Stability/Positioning Issues',
        'Medical/Health Concerns'
    ]
    
    # Filter to quality-related complaints
    quality_complaints = df[df['Category'].isin(quality_categories)]
    total_returns = len(df)
    quality_returns = len(quality_complaints)
    
    # Root cause analysis
    root_causes = {}
    safety_critical_count = 0
    
    for idx, row in quality_complaints.iterrows():
        complaint = str(row.get('Complaint', ''))
        category = row.get('Category', '')
        product = str(row.get('Product Identifier Tag', 'Unknown Product'))
        
        # Extract quality patterns
        pattern_data = extract_quality_patterns(complaint, category)
        
        if pattern_data['root_cause']:
            root_cause = pattern_data['root_cause']
            
            if root_cause not in root_causes:
                root_causes[root_cause] = {
                    'count': 0,
                    'products': set(),
                    'examples': [],
                    'safety_critical': 0
                }
            
            root_causes[root_cause]['count'] += 1
            root_causes[root_cause]['products'].add(product)
            
            if len(root_causes[root_cause]['examples']) < 3:
                root_causes[root_cause]['examples'].append(complaint[:100])
            
            if pattern_data['is_safety_critical']:
                root_causes[root_cause]['safety_critical'] += 1
                safety_critical_count += 1
                
                # Add to safety alerts
                insights['safety_alerts'].append({
                    'product': product,
                    'complaint': complaint[:200],
                    'root_cause': root_cause,
                    'severity': 'CRITICAL'
                })
    
    # Convert sets to lists for JSON serialization
    for root_cause, data in root_causes.items():
        data['products'] = list(data['products'])
    
    insights['root_cause_distribution'] = root_causes
    
    # Generate action items based on root causes
    for root_cause, data in root_causes.items():
        if data['count'] >= 3:  # Only create action items for patterns with multiple occurrences
            severity = 'HIGH' if root_cause in ['Safety Concern', 'Component Failure', 'Material Failure'] else 'MEDIUM'
            
            if data['safety_critical'] > 0:
                severity = 'HIGH'
            
            action = {
                'severity': severity,
                'issue': root_cause,
                'frequency': data['count'],
                'affected_products': data['products'][:5],
                'safety_critical': data['safety_critical'],
                'recommendation': get_quality_recommendation(root_cause, data['count'], data['products']),
                'examples': data['examples']
            }
            insights['action_items'].append(action)
    
    # Sort action items by priority
    insights['action_items'].sort(
        key=lambda x: (x['severity'] == 'HIGH', x['safety_critical'], x['frequency']), 
        reverse=True
    )
    
    # Risk assessment
    quality_rate = (quality_returns / total_returns * 100) if total_returns > 0 else 0
    
    if safety_critical_count > 10 or quality_rate > 25:
        risk_level = 'HIGH'
    elif safety_critical_count > 5 or quality_rate > 15:
        risk_level = 'MEDIUM'
    else:
        risk_level = 'LOW'
    
    # Product-specific analysis
    high_risk_products = []
    for product, categories in product_summary.items():
        quality_issues = sum(count for cat, count in categories.items() if cat in quality_categories)
        total_issues = sum(categories.values())
        
        if quality_issues >= 5:  # Products with significant quality issues
            # Get safety critical count for this product
            product_items = df[df['Product Identifier Tag'] == product]
            safety_count = 0
            primary_root_cause = 'Unknown'
            
            for _, row in product_items.iterrows():
                if row['Category'] in quality_categories:
                    pattern = extract_quality_patterns(str(row.get('Complaint', '')), row['Category'])
                    if pattern['is_safety_critical']:
                        safety_count += 1
                    if pattern['root_cause'] and primary_root_cause == 'Unknown':
                        primary_root_cause = pattern['root_cause']
            
            risk_score = quality_issues + (safety_count * 5)  # Weight safety issues more heavily
            
            high_risk_products.append({
                'product': product,
                'total_issues': total_issues,
                'quality_issues': quality_issues,
                'safety_issues': safety_count,
                'risk_score': risk_score,
                'primary_root_cause': primary_root_cause,
                'quality_rate': (quality_issues / total_issues * 100) if total_issues > 0 else 0
            })
    
    # Sort by risk score
    high_risk_products.sort(key=lambda x: x['risk_score'], reverse=True)
    
    insights['risk_assessment'] = {
        'overall_risk_level': risk_level,
        'quality_rate': quality_rate,
        'safety_critical_count': safety_critical_count,
        'total_returns': total_returns,
        'quality_returns': quality_returns,
        'top_risk_products': high_risk_products[:10]
    }
    
    return insights

# Simplified API Client for backward compatibility
class APIClient:
    """Simplified API client maintaining compatibility with existing code"""
    
    def __init__(self, provider: AIProvider = AIProvider.BOTH):
        self.analyzer = EnhancedAIAnalyzer(provider)
    
    def categorize_return(self, complaint: str, fba_reason: str = None, 
                         use_both: bool = True, max_tokens: int = 300) -> str:
        """Simple categorization method for compatibility"""
        mode = 'extreme' if max_tokens > 1000 else 'enhanced' if max_tokens > 500 else 'standard'
        category, _, _, _ = self.analyzer.categorize_return(complaint, fba_reason, mode)
        return category
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get usage summary"""
        return self.analyzer.get_usage_stats()

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
    'is_duplicate',
    'extract_quality_patterns',
    'generate_quality_insights',
    'BatchProcessor',
    'TOKEN_LIMITS',
    'BATCH_SIZE'
]
