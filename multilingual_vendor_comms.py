"""
Multilingual AI-Powered Vendor Communication Module

Generates vendor emails and reports with:
- AI-powered content generation
- Multi-language support (Chinese, Spanish, Portuguese, Hindi, etc.)
- English proficiency level adjustment
- Cultural context awareness
- Translation with original English text
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class EnglishLevel(Enum):
    """English proficiency levels for recipients"""
    NATIVE = "native"
    FLUENT = "fluent"
    INTERMEDIATE = "intermediate"
    BASIC = "basic"
    MINIMAL = "minimal"


class TargetLanguage(Enum):
    """Supported target languages"""
    ENGLISH = "en"
    CHINESE_SIMPLIFIED = "zh-CN"
    CHINESE_TRADITIONAL = "zh-TW"
    SPANISH = "es"
    PORTUGUESE = "pt"
    HINDI = "hi"
    GERMAN = "de"
    FRENCH = "fr"
    ITALIAN = "it"


# Language metadata
LANGUAGE_INFO = {
    TargetLanguage.ENGLISH: {"name": "English", "region": "Global"},
    TargetLanguage.CHINESE_SIMPLIFIED: {"name": "Chinese (Simplified)", "region": "China"},
    TargetLanguage.CHINESE_TRADITIONAL: {"name": "Chinese (Traditional)", "region": "Taiwan/Hong Kong"},
    TargetLanguage.SPANISH: {"name": "Spanish", "region": "Spain/LATAM"},
    TargetLanguage.PORTUGUESE: {"name": "Portuguese", "region": "Brazil/Portugal"},
    TargetLanguage.HINDI: {"name": "Hindi", "region": "India"},
    TargetLanguage.GERMAN: {"name": "German", "region": "Germany/EU"},
    TargetLanguage.FRENCH: {"name": "French", "region": "France/EU"},
    TargetLanguage.ITALIAN: {"name": "Italian", "region": "Italy/EU"},
}


class MultilingualVendorCommunicator:
    """
    AI-powered vendor communication generator with multilingual support
    """

    def __init__(self, ai_analyzer=None):
        """
        Initialize with optional AI analyzer for enhanced content generation

        Args:
            ai_analyzer: EnhancedAIAnalyzer instance for AI-powered generation
        """
        self.ai_analyzer = ai_analyzer

    def generate_capa_email(
        self,
        sku: str,
        product_name: str,
        issue_summary: str,
        return_rate: float,
        defect_description: str,
        units_affected: int,
        severity: str = 'major',
        english_level: EnglishLevel = EnglishLevel.INTERMEDIATE,
        target_language: TargetLanguage = TargetLanguage.ENGLISH,
        vendor_region: str = "China",
        additional_context: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Generate AI-powered CAPA request email with multi-language support

        Returns:
            Dict with 'english', 'translated', 'subject_english', 'subject_translated'
        """

        # Build AI prompt based on parameters
        prompt = self._build_capa_prompt(
            sku, product_name, issue_summary, return_rate, defect_description,
            units_affected, severity, english_level, vendor_region, additional_context
        )

        # Generate English version using AI
        english_email = self._generate_english_email_ai(
            prompt, english_level, vendor_region
        )

        # Generate translation if needed
        if target_language != TargetLanguage.ENGLISH and self.ai_analyzer:
            translated_email = self._translate_email(
                english_email['body'], target_language, vendor_region
            )
            translated_subject = self._translate_email(
                english_email['subject'], target_language, vendor_region
            )
        else:
            translated_email = None
            translated_subject = None

        return {
            'subject_english': english_email['subject'],
            'subject_translated': translated_subject,
            'body_english': english_email['body'],
            'body_translated': translated_email,
            'language': LANGUAGE_INFO[target_language]['name'],
            'reference': f"QC-{datetime.now().strftime('%Y%m%d')}-{sku}"
        }

    def generate_rca_request(
        self,
        sku: str,
        product_name: str,
        defect_type: str,
        occurrence_rate: float,
        sample_complaints: List[str],
        english_level: EnglishLevel = EnglishLevel.INTERMEDIATE,
        target_language: TargetLanguage = TargetLanguage.ENGLISH,
        vendor_region: str = "China"
    ) -> Dict[str, str]:
        """Generate AI-powered RCA request email"""

        prompt = self._build_rca_prompt(
            sku, product_name, defect_type, occurrence_rate,
            sample_complaints, english_level, vendor_region
        )

        english_email = self._generate_english_email_ai(
            prompt, english_level, vendor_region
        )

        if target_language != TargetLanguage.ENGLISH and self.ai_analyzer:
            translated_email = self._translate_email(
                english_email['body'], target_language, vendor_region
            )
            translated_subject = self._translate_email(
                english_email['subject'], target_language, vendor_region
            )
        else:
            translated_email = None
            translated_subject = None

        return {
            'subject_english': english_email['subject'],
            'subject_translated': translated_subject,
            'body_english': english_email['body'],
            'body_translated': translated_email,
            'language': LANGUAGE_INFO[target_language]['name'],
            'reference': f"RCA-{datetime.now().strftime('%Y%m%d')}-{sku}"
        }

    def generate_quality_report(
        self,
        products: List[Dict[str, Any]],
        report_type: str = "monthly",
        english_level: EnglishLevel = EnglishLevel.INTERMEDIATE,
        target_language: TargetLanguage = TargetLanguage.ENGLISH,
        vendor_region: str = "China"
    ) -> Dict[str, str]:
        """Generate comprehensive quality report for vendor"""

        prompt = f"""Generate a {report_type} quality performance report for a vendor in {vendor_region}.

English Level: {english_level.value}
Products covered: {len(products)}

Include:
1. Executive Summary
2. Performance Metrics
3. Top Issues Identified
4. Improvement Recommendations
5. Next Steps

Product data:
{self._format_products_for_prompt(products)}

Make the language {'simple and clear' if english_level in [EnglishLevel.BASIC, EnglishLevel.MINIMAL] else 'professional'}.
Use {'high-context, partnership-focused' if vendor_region in ['China', 'Japan', 'Korea'] else 'direct and concise'} communication style.
"""

        if self.ai_analyzer:
            system_prompt = f"You are a quality management professional writing to a vendor in {vendor_region}. Adjust your language complexity for {english_level.value} English proficiency."

            response = self.ai_analyzer.generate_text(prompt, system_prompt, mode='chat')

            if response:
                english_report = response
            else:
                english_report = self._fallback_quality_report(products, report_type)
        else:
            english_report = self._fallback_quality_report(products, report_type)

        # Translate if needed
        if target_language != TargetLanguage.ENGLISH and self.ai_analyzer:
            translated_report = self._translate_email(
                english_report, target_language, vendor_region
            )
        else:
            translated_report = None

        return {
            'subject_english': f"Quality Performance Report - {report_type.title()}",
            'body_english': english_report,
            'body_translated': translated_report,
            'language': LANGUAGE_INFO[target_language]['name'],
            'reference': f"QR-{datetime.now().strftime('%Y%m%d')}"
        }

    def _build_capa_prompt(
        self, sku, product_name, issue_summary, return_rate, defect_description,
        units_affected, severity, english_level, vendor_region, additional_context
    ) -> str:
        """Build AI prompt for CAPA email"""

        complexity_guide = {
            EnglishLevel.NATIVE: "Use professional business English with industry terminology",
            EnglishLevel.FLUENT: "Use clear professional language, avoid idioms",
            EnglishLevel.INTERMEDIATE: "Use simple, direct sentences. Avoid complex grammar",
            EnglishLevel.BASIC: "Use very simple words. Short sentences. No compound sentences",
            EnglishLevel.MINIMAL: "Use only basic words. Very short sentences. Like speaking to beginner"
        }

        cultural_guide = {
            "China": "Use high-context, relationship-focused language. Emphasize partnership. Allow face-saving. Be indirect about blame",
            "India": "Be formal and respectful. Use clear structure. Acknowledge partnership",
            "LATAM": "Be warm and relationship-focused. Use formal respect (usted)",
            "USA": "Be direct and concise. Focus on facts and action items",
            "EU": "Be formal and structured. Focus on compliance and standards"
        }

        prompt = f"""Generate a professional CAPA (Corrective and Preventive Action) request email to a vendor in {vendor_region}.

**Product Information:**
- SKU: {sku}
- Product Name: {product_name}
- Issue: {issue_summary}
- Return Rate: {return_rate:.1%}
- Units Affected: {units_affected}
- Severity: {severity}

**Defect Description:**
{defect_description}

{'**Additional Context:**' + additional_context if additional_context else ''}

**Email Requirements:**
- English Level: {english_level.value} - {complexity_guide[english_level]}
- Cultural Context: {cultural_guide.get(vendor_region, cultural_guide['USA'])}
- Tone: Professional but partnership-focused
- Structure: Clear sections with headers

**Required Content:**
1. Friendly greeting acknowledging partnership
2. Clear description of the quality issue
3. Specific data (return rate, units affected)
4. Request for:
   - Root Cause Analysis
   - Corrective Actions
   - Preventive Measures
   - Timeline
5. Offer of support/samples if needed
6. Professional closing

Generate only the email body (no meta-commentary). Include a subject line at the top.
"""
        return prompt

    def _build_rca_prompt(
        self, sku, product_name, defect_type, occurrence_rate,
        sample_complaints, english_level, vendor_region
    ) -> str:
        """Build AI prompt for RCA request email"""

        complaints_text = "\n".join(f"- {c}" for c in sample_complaints[:5])

        prompt = f"""Generate a Root Cause Analysis (RCA) request email to a vendor in {vendor_region}.

**Product Information:**
- SKU: {sku}
- Product Name: {product_name}
- Defect Type: {defect_type}
- Occurrence Rate: {occurrence_rate:.1%}

**Sample Customer Complaints:**
{complaints_text}

**Requirements:**
- English Level: {english_level.value}
- Region: {vendor_region}
- Request specific RCA deliverables
- Timeline: 5-7 business days
- Offer assistance/samples

Generate the email with subject line.
"""
        return prompt

    def _generate_english_email_ai(
        self, prompt: str, english_level: EnglishLevel, vendor_region: str
    ) -> Dict[str, str]:
        """Generate English email using AI"""

        if self.ai_analyzer:
            system_prompt = f"You are a quality management professional writing vendor communications. Adjust language for {english_level.value} English proficiency and {vendor_region} cultural context."

            try:
                response = self.ai_analyzer.generate_text(prompt, system_prompt, mode='chat')

                if response:
                    # Extract subject and body
                    if "Subject:" in response:
                        parts = response.split("\n", 1)
                        subject = parts[0].replace("Subject:", "").strip()
                        body = parts[1].strip() if len(parts) > 1 else response
                    else:
                        subject = f"Quality Matter - Product Quality Review"
                        body = response

                    return {'subject': subject, 'body': body}
            except Exception as e:
                logger.error(f"AI email generation failed: {e}")

        # Fallback to template
        return self._fallback_email_template(english_level, vendor_region)

    def _translate_email(
        self, text: str, target_language: TargetLanguage, vendor_region: str
    ) -> str:
        """Translate email using AI"""

        if not self.ai_analyzer:
            return None

        language_name = LANGUAGE_INFO[target_language]['name']

        prompt = f"""Translate the following business email into {language_name}.

**Requirements:**
- Maintain professional business tone
- Keep cultural context appropriate for {vendor_region}
- Preserve formatting (line breaks, bullet points, sections)
- Ensure technical terms are accurately translated

**Original Email:**
{text}

Provide only the translated text, no explanations.
"""

        system_prompt = f"You are a professional translator specializing in business and quality management communications for {vendor_region}."

        try:
            translation = self.ai_analyzer.generate_text(prompt, system_prompt, mode='chat')
            return translation if translation else None
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return None

    def _format_products_for_prompt(self, products: List[Dict[str, Any]]) -> str:
        """Format product list for AI prompt"""
        formatted = []
        for p in products[:10]:  # Limit to 10 for token management
            formatted.append(
                f"- {p.get('sku', 'N/A')}: {p.get('product_name', 'N/A')} | "
                f"Return Rate: {p.get('return_rate', 0):.1%} | "
                f"Issue: {p.get('issue_summary', 'N/A')}"
            )
        return "\n".join(formatted)

    def _fallback_email_template(self, english_level: EnglishLevel, vendor_region: str) -> Dict[str, str]:
        """Fallback email template when AI is not available"""

        if english_level in [EnglishLevel.BASIC, EnglishLevel.MINIMAL]:
            subject = "Product Quality - Need Your Help"
            body = """Dear Partner,

We have a product quality issue.

We need your help to fix it.

Please:
1. Look at the problem
2. Find the cause
3. Tell us how to fix it
4. Tell us when it will be fixed

We will send more information soon.

Thank you.

Best regards,
Quality Team"""
        else:
            subject = "Quality Improvement Request - Partnership Collaboration"
            body = """Dear Partner,

Thank you for your continued partnership with our company.

We are writing to request your assistance with a product quality matter that has come to our attention through customer feedback.

We would like to work together to investigate this issue and implement improvements that will benefit both our companies and our customers.

Please provide the following within 7 business days:
1. Root cause analysis of the issue
2. Proposed corrective actions
3. Preventive measures for future production
4. Implementation timeline

We value our partnership and are committed to working together to resolve this matter. Please let us know if you need any additional information, samples, or support from our team.

Thank you for your attention and cooperation.

Best regards,
Quality Team"""

        return {'subject': subject, 'body': body}

    def _fallback_quality_report(self, products: List[Dict[str, Any]], report_type: str) -> str:
        """Fallback quality report when AI is not available"""

        total_products = len(products)
        avg_return_rate = sum(p.get('return_rate', 0) for p in products) / total_products if total_products > 0 else 0

        report = f"""QUALITY PERFORMANCE REPORT
{report_type.upper()} REVIEW
Generated: {datetime.now().strftime('%Y-%m-%d')}

=====================================

EXECUTIVE SUMMARY

Products Reviewed: {total_products}
Average Return Rate: {avg_return_rate:.1%}

=====================================

TOP PERFORMING PRODUCTS

"""
        # Add top 5 products by return rate
        sorted_products = sorted(products, key=lambda x: x.get('return_rate', 0))[:5]
        for i, p in enumerate(sorted_products, 1):
            report += f"{i}. {p.get('sku', 'N/A')}: {p.get('return_rate', 0):.1%}\n"

        report += """
=====================================

RECOMMENDED ACTIONS

1. Continue monitoring quality metrics
2. Implement corrective actions for high-return products
3. Share best practices from top performers
4. Schedule follow-up review

=====================================

Thank you for your partnership.
"""
        return report


# Export main class
__all__ = ['MultilingualVendorCommunicator', 'EnglishLevel', 'TargetLanguage', 'LANGUAGE_INFO']
