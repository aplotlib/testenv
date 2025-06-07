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
API_TIMEOUT = 45
MAX_RETRIES = 3
BATCH_SIZE = 100  # Process in batches for stability

# Token configurations by mode
TOKEN_LIMITS = {
    'standard': 300,
    'enhanced': 800,
    'extreme': 2000  # For difficult cases
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

# Category keywords for fallback
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

# Quality patterns for root cause analysis
QUALITY_PATTERNS = {
    'Material Failure': ['velcro', 'strap', 'fabric', 'material', 'stitching', 'tear', 'rip', 'worn', 'fraying'],
    'Component Failure': ['button', 'buckle', 'wheel', 'handle', 'pump', 'valve', 'motor', 'battery', 'suction'],
    'Design Flaw': ['too heavy', 'hard to use', 'difficult to adjust', 'poor design', 'awkward', 'cumbersome', 'not intuitive'],
    'Manufacturing Defect': ['broken on arrival', 'defective', 'missing parts', 'not assembled correctly', 'crooked', 'uneven'],
    'Durability Issue': ['broke after', 'lasted only', 'stopped working', 'fell apart', 'wore out', 'degraded'],
    'Safety Concern': ['unsafe', 'dangerous', 'injury', 'hurt', 'accident', 'risk', 'hazard', 'sharp edge']
}

class AIProvider(Enum):
    OPENAI = "openai"
    CLAUDE = "claude"
    BOTH = "both"

def detect_language(text: str) -> str:
    """Simple language detection"""
    spanish_indicators = ['ñ', 'á', 'é', 'í', 'ó', 'ú', ' es ', ' la ', ' el ', ' los ', ' las ']
    spanish_count = sum(1 for indicator in spanish_indicators if indicator in text.lower())
    return 'es' if spanish_count >= 2 else 'en'

def translate_text(text: str, source_lang: str = 'auto', target_lang: str = 'en') -> str:
    """Simple passthrough - implement translation if needed"""
    return text

def calculate_confidence(complaint: str, category: str, consensus: bool = False) -> float:
    """Calculate confidence score for categorization"""
    if consensus:
        return 0.95  # High confidence when both AIs agree
    return 0.85  # Standard confidence

def detect_severity(complaint: str, category: str) -> str:
    """Detect severity level of complaint"""
    complaint_lower = complaint.lower()
    
    critical_keywords = ['injury', 'injured', 'hospital', 'emergency', 'doctor', 'dangerous', 
                        'unsafe', 'accident', 'bleeding', 'burn', 'fall', 'allergic']
    
    for keyword in critical_keywords:
        if keyword in complaint_lower:
            return 'critical'
    
    if category in ['Product Defects/Quality', 'Medical/Health Concerns']:
        return 'major'
    
    return 'none'

def is_duplicate(complaint1: str, complaint2: str) -> bool:
    """No duplicate detection as requested"""
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
        return {'patterns': [], 'root_cause': None}
    
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
    
    return {
        'patterns': patterns_found,
        'root_cause': root_cause,
        'is_safety_critical': any(p['pattern'] == 'Safety Concern' for p in patterns_found)
    }

def get_quality_recommendation(root_cause: str, frequency: int) -> str:
    """Generate specific quality recommendations based on root cause"""
    recommendations = {
        'Material Failure': f"Conduct material testing on affected components. Consider supplier quality audit. {frequency} cases indicate potential batch issue.",
        'Component Failure': f"Engineering review required for component design/specification. Implement incoming inspection for this component.",
        'Design Flaw': f"Initiate design review with engineering team. Consider user study to improve ergonomics. Pattern appears in {frequency} returns.",
        'Manufacturing Defect': f"Investigate production line QC procedures. Possible process control issue. Review last {frequency} production batches.",
        'Durability Issue': f"Accelerated life testing recommended. Review warranty data. Consider material upgrade or design reinforcement.",
        'Safety Concern': f"URGENT: Conduct risk assessment per ISO 14971. Review {frequency} safety-related complaints for potential recall evaluation."
    }
    
    return recommendations.get(root_cause, f"Investigate {frequency} occurrences of {root_cause} pattern.")

class BatchProcessor:
    """Handle batch processing with checkpoints"""
    
    @staticmethod
    def save_checkpoint(data: Dict, checkpoint_file: str):
        """Save processing checkpoint"""
        try:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved checkpoint to {checkpoint_file}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    @staticmethod
    def load_checkpoint(checkpoint_file: str) -> Optional[Dict]:
        """Load processing checkpoint"""
        try:
            if os.path.exists(checkpoint_file):
                with open(checkpoint_file, 'rb') as f:
                    data = pickle.load(f)
                logger.info(f"Loaded checkpoint from {checkpoint_file}")
                return data
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
        return None
    
    @staticmethod
    def get_checkpoint_file() -> str:
        """Generate checkpoint filename"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"categorization_checkpoint_{timestamp}.pkl"

class EnhancedAIAnalyzer:
    """Main AI analyzer with dual AI support and extreme mode"""
    
    def __init__(self, provider: AIProvider = AIProvider.BOTH):
        self.provider = provider
        self.openai_key = self._get_api_key('openai')
        self.claude_key = self._get_api_key('claude')
        
        # Initialize API clients
        self.openai_configured = bool(self.openai_key)
        self.claude_configured = bool(self.claude_key) and has_anthropic
        
        if self.claude_configured:
            self.claude_client = anthropic.Anthropic(api_key=self.claude_key)
        else:
            self.claude_client = None
        
        logger.info(f"AI Analyzer initialized - OpenAI: {self.openai_configured}, Claude: {self.claude_configured}")
    
    def _get_api_key(self, provider: str) -> Optional[str]:
        """Get API key from multiple sources"""
        # Try Streamlit secrets first
        try:
            import streamlit as st
            if hasattr(st, 'secrets'):
                if provider == 'openai':
                    for key_name in ["openai_api_key", "OPENAI_API_KEY", "openai", "api_key"]:
                        if key_name in st.secrets:
                            logger.info(f"Found OpenAI key in Streamlit secrets")
                            return str(st.secrets[key_name])
                elif provider == 'claude':
                    for key_name in ["anthropic_api_key", "ANTHROPIC_API_KEY", "claude_api_key", "claude"]:
                        if key_name in st.secrets:
                            logger.info(f"Found Claude key in Streamlit secrets")
                            return str(st.secrets[key_name])
        except Exception as e:
            logger.debug(f"Streamlit secrets not available: {e}")
        
        # Try environment variables
        if provider == 'openai':
            for env_name in ["OPENAI_API_KEY", "OPENAI_API", "API_KEY"]:
                api_key = os.environ.get(env_name)
                if api_key:
                    logger.info(f"Found OpenAI key in environment")
                    return api_key
        elif provider == 'claude':
            for env_name in ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"]:
                api_key = os.environ.get(env_name)
                if api_key:
                    logger.info(f"Found Claude key in environment")
                    return api_key
        
        logger.warning(f"No {provider} API key found")
        return None
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get API availability status"""
        status = {
            'available': self.openai_configured or self.claude_configured,
            'openai_configured': self.openai_configured,
            'claude_configured': self.claude_configured,
            'dual_ai_available': self.openai_configured and self.claude_configured,
            'message': ''
        }
        
        if status['dual_ai_available']:
            status['message'] = 'Both OpenAI and Claude APIs ready'
        elif self.openai_configured:
            status['message'] = 'OpenAI API ready (Claude not configured)'
        elif self.claude_configured:
            status['message'] = 'Claude API ready (OpenAI not configured)'
        else:
            status['message'] = 'No APIs configured'
        
        return status
    
    def _call_openai(self, prompt: str, system_prompt: str, mode: str = 'standard') -> Optional[str]:
        """Call OpenAI API"""
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
            "top_p": 0.95
        }
        
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=API_TIMEOUT
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            else:
                logger.error(f"OpenAI API error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"OpenAI call error: {e}")
            return None
    
    def _call_claude(self, prompt: str, system_prompt: str, mode: str = 'standard') -> Optional[str]:
        """Call Claude API"""
        if not self.claude_configured or not self.claude_client:
            return None
        
        model = MODELS['claude'][mode]
        max_tokens = TOKEN_LIMITS[mode]
        
        try:
            response = self.claude_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=0.1,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text.strip()
            
        except Exception as e:
            logger.error(f"Claude call error: {e}")
            return None
    
    def categorize_return(self, complaint: str, fba_reason: str = None, mode: str = 'standard') -> Tuple[str, float, str, str]:
        """Categorize using dual AI with consensus"""
        
        # Check FBA mapping first
        if fba_reason and fba_reason in FBA_REASON_MAP:
            category = FBA_REASON_MAP[fba_reason]
            return category, 0.95, detect_severity(complaint, category), 'en'
        
        # Build prompts
        system_prompt = f"""You are a quality expert categorizing medical device returns. 

You MUST respond with EXACTLY one of these categories (copy exactly as shown):
{chr(10).join(f'"{cat}"' for cat in MEDICAL_DEVICE_CATEGORIES)}

Do not add any other text, explanation, or punctuation. Just the category name."""
        
        if mode == 'extreme':
            # Extreme mode - very detailed prompt
            prompt = f"""Carefully analyze this medical device return complaint using deep reasoning.

COMPLAINT: "{complaint}"
"""
            if fba_reason:
                prompt += f"\nFBA REASON CODE: {fba_reason}"
            
            prompt += f"""

STEP-BY-STEP ANALYSIS REQUIRED:
1. Identify ALL issues mentioned in the complaint
2. Determine the PRIMARY reason for return
3. Consider medical device context and safety implications
4. Match to the most specific category possible

DETAILED CATEGORY DEFINITIONS:
- "Size/Fit Issues": Product physical dimensions don't match user's body/equipment (too small, large, tight, loose, narrow, wide)
- "Comfort Issues": Product causes physical discomfort, pain, pressure points during normal use
- "Product Defects/Quality": Manufacturing defects, broken components, quality control failures, damaged items
- "Performance/Effectiveness": Product fails to perform its intended medical function or therapeutic benefit
- "Stability/Positioning Issues": Product moves, slides, falls, or doesn't maintain proper position during use
- "Equipment Compatibility": Doesn't properly interface with wheelchairs, walkers, beds, toilets, or other equipment
- "Design/Material Issues": Inherent design problems, material choices that affect usability (weight, bulkiness, material quality)
- "Wrong Product/Misunderstanding": Customer received or ordered incorrect item, product not as described/expected
- "Missing Components": Package incomplete, missing parts, accessories, or instructions
- "Customer Error/Changed Mind": Customer mistake, no longer needed, patient condition changed, unauthorized purchase
- "Shipping/Fulfillment Issues": Delivery problems, packaging damage, late arrival
- "Assembly/Usage Difficulty": Hard to set up, confusing instructions, difficult to operate
- "Medical/Health Concerns": Allergic reactions, medical complications, doctor disapproval, safety issues
- "Price/Value": Cost concerns, found cheaper alternative, not worth the price
- "Other/Miscellaneous": ONLY if absolutely no other category fits

IMPORTANT DISTINCTIONS:
- If product is "too small/large" = Size/Fit Issues (NOT Design/Material)
- If product "doesn't work" = Performance/Effectiveness (NOT Product Defects)
- If "hard to use" = Assembly/Usage Difficulty (NOT Design/Material)
- If "broken on arrival" = Product Defects/Quality (NOT Shipping)
- If "uncomfortable" = Comfort Issues (NOT Size/Fit)

Analyze the complaint thoroughly, then respond with the single best matching category name."""
        
        else:
            # Standard mode prompt
            prompt = f"""Categorize this medical device return complaint.

Complaint: "{complaint}"
"""
            if fba_reason:
                prompt += f"\nFBA Reason Code: {fba_reason}"
            
            prompt += f"""

Choose exactly ONE category from:
{chr(10).join(f'- {cat}' for cat in MEDICAL_DEVICE_CATEGORIES)}

Respond with ONLY the category name."""
        
        # Get responses from both AIs if available
        openai_result = None
        claude_result = None
        
        if self.provider in [AIProvider.OPENAI, AIProvider.BOTH] and self.openai_configured:
            openai_result = self._call_openai(prompt, system_prompt, mode)
            if openai_result:
                openai_result = self._clean_category_response(openai_result)
        
        if self.provider in [AIProvider.CLAUDE, AIProvider.BOTH] and self.claude_configured:
            claude_result = self._call_claude(prompt, system_prompt, mode)
            if claude_result:
                claude_result = self._clean_category_response(claude_result)
        
        # Determine final category
        if openai_result and claude_result:
            # Both responded - check consensus
            if openai_result == claude_result:
                # Consensus!
                severity = detect_severity(complaint, openai_result)
                logger.info(f"AI consensus: '{complaint[:50]}...' -> {openai_result}")
                return openai_result, 0.95, severity, 'en'
            else:
                # Disagreement - prefer non-Other category
                logger.warning(f"AI disagreement - OpenAI: {openai_result}, Claude: {claude_result}")
                if openai_result != 'Other/Miscellaneous':
                    category = openai_result
                elif claude_result != 'Other/Miscellaneous':
                    category = claude_result
                else:
                    category = openai_result  # Default to OpenAI
                
                severity = detect_severity(complaint, category)
                return category, 0.75, severity, 'en'
        
        elif openai_result:
            # Only OpenAI responded
            severity = detect_severity(complaint, openai_result)
            return openai_result, 0.85, severity, 'en'
        
        elif claude_result:
            # Only Claude responded
            severity = detect_severity(complaint, claude_result)
            return claude_result, 0.85, severity, 'en'
        
        else:
            # No AI responses - fallback to keywords
            logger.warning(f"No AI responses for: '{complaint[:50]}...'")
            complaint_lower = complaint.lower()
            for category, keywords in CATEGORY_KEYWORDS.items():
                if any(keyword in complaint_lower for keyword in keywords):
                    severity = detect_severity(complaint, category)
                    return category, 0.7, severity, 'en'
            
            return 'Other/Miscellaneous', 0.5, 'none', 'en'
    
    def _clean_category_response(self, response: str) -> str:
        """Clean and validate category response"""
        # Remove quotes and whitespace
        response = response.strip()
        if response.startswith('"') and response.endswith('"'):
            response = response[1:-1]
        if response.startswith("'") and response.endswith("'"):
            response = response[1:-1]
        response = response.strip()
        
        # Find exact match
        for valid_cat in MEDICAL_DEVICE_CATEGORIES:
            if response == valid_cat:
                return valid_cat
        
        # Try case-insensitive
        for valid_cat in MEDICAL_DEVICE_CATEGORIES:
            if response.lower() == valid_cat.lower():
                return valid_cat
        
        # Try partial match
        for valid_cat in MEDICAL_DEVICE_CATEGORIES:
            if valid_cat in response or response in valid_cat:
                return valid_cat
        
        logger.warning(f"Could not match category: '{response}'")
        return 'Other/Miscellaneous'

def generate_quality_insights(df, reason_summary: Dict, product_summary: Dict) -> Dict[str, Any]:
    """Generate actionable quality insights from categorized data"""
    insights = {
        'critical_patterns': {},
        'product_specific_issues': {},
        'root_cause_distribution': {},
        'action_items': [],
        'risk_assessment': {}
    }
    
    quality_categories = [
        'Product Defects/Quality',
        'Performance/Effectiveness',
        'Design/Material Issues',
        'Stability/Positioning Issues',
        'Medical/Health Concerns'
    ]
    
    quality_complaints = df[df['Category'].isin(quality_categories)]
    
    for idx, row in quality_complaints.iterrows():
        complaint = str(row.get('Complaint', ''))
        category = row.get('Category', '')
        product = str(row.get('Product Identifier Tag', 'Unknown'))
        
        pattern_data = extract_quality_patterns(complaint, category)
        
        if pattern_data['root_cause']:
            root_cause = pattern_data['root_cause']
            if root_cause not in insights['root_cause_distribution']:
                insights['root_cause_distribution'][root_cause] = {
                    'count': 0,
                    'products': set(),
                    'examples': []
                }
            
            insights['root_cause_distribution'][root_cause]['count'] += 1
            insights['root_cause_distribution'][root_cause]['products'].add(product)
            
            if len(insights['root_cause_distribution'][root_cause]['examples']) < 3:
                insights['root_cause_distribution'][root_cause]['examples'].append(complaint[:100])
            
            if product not in insights['product_specific_issues']:
                insights['product_specific_issues'][product] = {
                    'total_quality_issues': 0,
                    'root_causes': {},
                    'safety_critical': 0
                }
            
            insights['product_specific_issues'][product]['total_quality_issues'] += 1
            
            if root_cause not in insights['product_specific_issues'][product]['root_causes']:
                insights['product_specific_issues'][product]['root_causes'][root_cause] = 0
            insights['product_specific_issues'][product]['root_causes'][root_cause] += 1
            
            if pattern_data['is_safety_critical']:
                insights['product_specific_issues'][product]['safety_critical'] += 1
    
    # Convert sets to lists
    for root_cause, data in insights['root_cause_distribution'].items():
        data['products'] = list(data['products'])
    
    # Generate action items
    for root_cause, data in insights['root_cause_distribution'].items():
        if data['count'] >= 5:
            severity = 'HIGH' if root_cause in ['Safety Concern', 'Component Failure'] else 'MEDIUM'
            
            action = {
                'severity': severity,
                'issue': root_cause,
                'frequency': data['count'],
                'affected_products': data['products'][:5],
                'recommendation': get_quality_recommendation(root_cause, data['count']),
                'examples': data['examples']
            }
            insights['action_items'].append(action)
    
    insights['action_items'].sort(key=lambda x: (x['severity'] == 'HIGH', x['frequency']), reverse=True)
    
    # Risk assessment
    total_returns = len(df)
    quality_returns = len(quality_complaints)
    safety_issues = sum(1 for _, row in quality_complaints.iterrows() 
                       if extract_quality_patterns(str(row.get('Complaint', '')), row.get('Category', ''))['is_safety_critical'])
    
    insights['risk_assessment'] = {
        'quality_rate': (quality_returns / total_returns * 100) if total_returns > 0 else 0,
        'safety_critical_count': safety_issues,
        'top_risk_products': [],
        'overall_risk_level': 'LOW'
    }
    
    risk_products = []
    for product, data in insights['product_specific_issues'].items():
        if data['safety_critical'] > 0 or data['total_quality_issues'] >= 10:
            risk_score = data['safety_critical'] * 10 + data['total_quality_issues']
            risk_products.append({
                'product': product,
                'risk_score': risk_score,
                'safety_issues': data['safety_critical'],
                'total_issues': data['total_quality_issues'],
                'primary_root_cause': max(data['root_causes'].items(), key=lambda x: x[1])[0] if data['root_causes'] else 'Unknown'
            })
    
    risk_products.sort(key=lambda x: x['risk_score'], reverse=True)
    insights['risk_assessment']['top_risk_products'] = risk_products[:10]
    
    if safety_issues > 5 or insights['risk_assessment']['quality_rate'] > 20:
        insights['risk_assessment']['overall_risk_level'] = 'HIGH'
    elif safety_issues > 2 or insights['risk_assessment']['quality_rate'] > 10:
        insights['risk_assessment']['overall_risk_level'] = 'MEDIUM'
    
    return insights

# Simplified API Client for app compatibility
class APIClient:
    """Simplified API client for the app"""
    
    def __init__(self):
        self.analyzer = EnhancedAIAnalyzer()
    
    def categorize_return(self, complaint: str, fba_reason: str = None, mode: str = 'standard') -> str:
        """Simple categorization method"""
        category, _, _, _ = self.analyzer.categorize_return(complaint, fba_reason, mode)
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
    'is_duplicate',
    'extract_quality_patterns',
    'generate_quality_insights',
    'BatchProcessor',
    'TOKEN_LIMITS',
    'BATCH_SIZE'
]
