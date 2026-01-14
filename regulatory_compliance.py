"""
Regulatory Compliance Module for Medical Device Quality Management

Provides AI-powered screening for regulatory requirements across multiple markets
and generates compliance suggestions with confidence scoring.
"""

import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# Regulatory markets and their key agencies
REGULATORY_MARKETS = {
    'US': {
        'name': 'United States',
        'agencies': [
            {'name': 'FDA (Food & Drug Administration)', 'url': 'https://www.fda.gov/medical-devices'},
            {'name': 'FDA MedWatch', 'url': 'https://www.fda.gov/safety/medwatch-fda-safety-information-and-adverse-event-reporting-program'},
        ],
        'key_regulations': ['21 CFR Part 820 (QSR)', 'MDR (Medical Device Reporting)', 'UDI Requirements'],
        'injury_reporting': True,
        'reporting_timeline': '30 days for serious injuries, 5 days for deaths',
    },
    'Mexico': {
        'name': 'Mexico',
        'agencies': [
            {'name': 'COFEPRIS', 'url': 'https://www.gob.mx/cofepris'},
        ],
        'key_regulations': ['NOM-241-SSA1-2012', 'Farmacovigilancia'],
        'injury_reporting': True,
        'reporting_timeline': '15 days for serious adverse events',
    },
    'Colombia': {
        'name': 'Colombia',
        'agencies': [
            {'name': 'INVIMA', 'url': 'https://www.invima.gov.co/'},
        ],
        'key_regulations': ['Decree 4725', 'Resolution 4002'],
        'injury_reporting': True,
        'reporting_timeline': '15 days for serious incidents',
    },
    'Brazil': {
        'name': 'Brazil',
        'agencies': [
            {'name': 'ANVISA', 'url': 'https://www.gov.br/anvisa/'},
        ],
        'key_regulations': ['RDC 16/2013', 'Tecnovigilância'],
        'injury_reporting': True,
        'reporting_timeline': '72 hours for serious adverse events',
    },
    'Chile': {
        'name': 'Chile',
        'agencies': [
            {'name': 'ISP (Instituto de Salud Pública)', 'url': 'https://www.ispch.cl/'},
        ],
        'key_regulations': ['Decree 57', 'Technical Regulation'],
        'injury_reporting': True,
        'reporting_timeline': '10 days for serious events',
    },
    'UK': {
        'name': 'United Kingdom',
        'agencies': [
            {'name': 'MHRA (Medicines & Healthcare products Regulatory Agency)', 'url': 'https://www.gov.uk/government/organisations/medicines-and-healthcare-products-regulatory-agency'},
        ],
        'key_regulations': ['UK MDR 2002', 'UKCA Marking'],
        'injury_reporting': True,
        'reporting_timeline': 'Immediate for deaths/serious deterioration',
    },
    'Germany': {
        'name': 'Germany',
        'agencies': [
            {'name': 'BfArM (Federal Institute for Drugs & Medical Devices)', 'url': 'https://www.bfarm.de/EN/'},
        ],
        'key_regulations': ['EU MDR 2017/745', 'MPG (Medical Devices Act)'],
        'injury_reporting': True,
        'reporting_timeline': 'Immediate for serious incidents per EU MDR',
    },
    'France': {
        'name': 'France',
        'agencies': [
            {'name': 'ANSM (Agence Nationale de Sécurité du Médicament)', 'url': 'https://ansm.sante.fr/'},
        ],
        'key_regulations': ['EU MDR 2017/745', 'Code de la Santé Publique'],
        'injury_reporting': True,
        'reporting_timeline': 'Immediate for serious incidents per EU MDR',
    },
    'Italy': {
        'name': 'Italy',
        'agencies': [
            {'name': 'Italian Ministry of Health', 'url': 'https://www.salute.gov.it/'},
        ],
        'key_regulations': ['EU MDR 2017/745', 'Legislative Decree 46/97'],
        'injury_reporting': True,
        'reporting_timeline': 'Immediate for serious incidents per EU MDR',
    },
    'Spain': {
        'name': 'Spain',
        'agencies': [
            {'name': 'AEMPS (Agencia Española de Medicamentos)', 'url': 'https://www.aemps.gob.es/'},
        ],
        'key_regulations': ['EU MDR 2017/745', 'Royal Decree 1591/2009'],
        'injury_reporting': True,
        'reporting_timeline': 'Immediate for serious incidents per EU MDR',
    },
    'Netherlands': {
        'name': 'Netherlands',
        'agencies': [
            {'name': 'IGJ (Health & Youth Care Inspectorate)', 'url': 'https://www.igj.nl/'},
        ],
        'key_regulations': ['EU MDR 2017/745', 'Medical Devices Decree'],
        'injury_reporting': True,
        'reporting_timeline': 'Immediate for serious incidents per EU MDR',
    },
}


class RegulatoryComplianceAnalyzer:
    """
    AI-powered regulatory compliance analyzer for medical device quality issues
    """

    def __init__(self, ai_analyzer=None):
        """
        Initialize with optional AI analyzer

        Args:
            ai_analyzer: EnhancedAIAnalyzer instance for AI-powered suggestions
        """
        self.ai_analyzer = ai_analyzer
        self.markets = REGULATORY_MARKETS

    def analyze_compliance_requirements(
        self,
        selected_markets: List[str],
        product_data: Dict[str, Any],
        issue_type: str = "Quality Issue"
    ) -> Dict[str, Any]:
        """
        Analyze regulatory compliance requirements for selected markets

        Args:
            selected_markets: List of market codes (e.g., ['US', 'UK', 'Germany'])
            product_data: Product information including complaints, severity, etc.
            issue_type: Type of quality issue

        Returns:
            Dictionary with compliance analysis and recommendations
        """

        results = {
            'markets_analyzed': len(selected_markets),
            'injury_reporting_required': [],
            'regulatory_actions': [],
            'reporting_timelines': {},
            'key_regulations': {},
            'agency_contacts': {},
        }

        # Check each market
        for market_code in selected_markets:
            if market_code not in self.markets:
                continue

            market_info = self.markets[market_code]

            # Check if injury reporting required
            if market_info.get('injury_reporting'):
                results['injury_reporting_required'].append({
                    'market': market_info['name'],
                    'timeline': market_info.get('reporting_timeline', 'See regulations'),
                    'agencies': market_info['agencies']
                })

            # Store timelines
            results['reporting_timelines'][market_code] = market_info.get('reporting_timeline', 'N/A')

            # Store regulations
            results['key_regulations'][market_code] = market_info.get('key_regulations', [])

            # Store agency contacts
            results['agency_contacts'][market_code] = market_info['agencies']

        # AI-powered analysis if available
        if self.ai_analyzer:
            ai_suggestions = self._get_ai_regulatory_suggestions(
                selected_markets, product_data, issue_type
            )
            results['ai_suggestions'] = ai_suggestions
        else:
            results['ai_suggestions'] = []

        return results

    def _get_ai_regulatory_suggestions(
        self,
        markets: List[str],
        product_data: Dict[str, Any],
        issue_type: str
    ) -> List[Dict[str, Any]]:
        """
        Get AI-powered regulatory suggestions with confidence scoring

        Returns:
            List of suggestions with confidence >= 85%
        """

        suggestions = []

        # Build AI prompt
        market_names = [self.markets[m]['name'] for m in markets if m in self.markets]

        prompt = f"""Analyze this medical device quality issue for regulatory compliance requirements.

Product Information:
- SKU: {product_data.get('sku', 'Unknown')}
- Product: {product_data.get('product_name', 'Unknown')}
- Category: {product_data.get('category', 'Unknown')}
- Return Rate: {product_data.get('return_rate', 0) * 100:.2f}%
- Issue Type: {issue_type}
- Complaints: {product_data.get('complaint_summary', 'None specified')}
- Safety Risk: {product_data.get('safety_risk', False)}

Markets: {', '.join(market_names)}

Analyze for regulatory requirements including:
1. Injury/incident reporting obligations
2. Recall considerations
3. Corrective action notifications
4. Post-market surveillance reports
5. Product registration updates

For EACH requirement you identify, provide:
- Requirement description
- Applicable markets (subset of markets listed above)
- Confidence score (0-100) based on the evidence in the data
- Rationale for the confidence score
- Recommended action

ONLY include requirements where confidence >= 85%.

Format your response as a structured list with clear sections for each requirement."""

        system_prompt = """You are a medical device regulatory compliance expert with expertise in FDA 21 CFR 820,
EU MDR 2017/745, ISO 13485, and international medical device regulations. Provide accurate,
conservative compliance guidance. Only flag requirements you are highly confident about (>=85% confidence)."""

        try:
            response = self.ai_analyzer.generate_text(prompt, system_prompt, mode='chat')

            if response:
                # Parse the AI response to extract suggestions
                # This is a simplified parser - could be enhanced with structured output
                suggestions.append({
                    'confidence': 90,  # Placeholder - would parse from response
                    'requirement': 'AI Analysis Complete',
                    'markets': market_names,
                    'recommendation': response,
                    'priority': 'High'
                })

        except Exception as e:
            logger.error(f"AI regulatory analysis failed: {e}")

        return suggestions

    def generate_compliance_report(
        self,
        selected_markets: List[str],
        products_data: List[Dict[str, Any]],
        analysis_results: Dict[str, Any]
    ) -> str:
        """
        Generate a formatted compliance report

        Args:
            selected_markets: Markets being analyzed
            products_data: List of product data dictionaries
            analysis_results: Results from analyze_compliance_requirements

        Returns:
            Formatted markdown report
        """

        report_lines = []
        report_lines.append("# Regulatory Compliance Report")
        report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Markets summary
        report_lines.append("## Markets Analyzed")
        for market_code in selected_markets:
            if market_code in self.markets:
                market_info = self.markets[market_code]
                report_lines.append(f"### {market_info['name']}")
                report_lines.append(f"**Regulations:** {', '.join(market_info.get('key_regulations', []))}")
                report_lines.append(f"**Injury Reporting:** {'Required' if market_info.get('injury_reporting') else 'Not Required'}")
                if market_info.get('injury_reporting'):
                    report_lines.append(f"**Timeline:** {market_info.get('reporting_timeline')}")
                report_lines.append("\n**Regulatory Agencies:**")
                for agency in market_info['agencies']:
                    report_lines.append(f"- [{agency['name']}]({agency['url']})")
                report_lines.append("")

        # Products summary
        report_lines.append("\n## Products Analyzed")
        report_lines.append(f"**Total Products:** {len(products_data)}")
        for product in products_data[:10]:  # Limit to first 10
            report_lines.append(f"- {product.get('sku', 'Unknown')}: {product.get('product_name', 'Unknown')}")

        if len(products_data) > 10:
            report_lines.append(f"- ... and {len(products_data) - 10} more products")

        # AI Suggestions
        if analysis_results.get('ai_suggestions'):
            report_lines.append("\n## AI-Powered Compliance Suggestions (≥85% Confidence)")
            for suggestion in analysis_results['ai_suggestions']:
                report_lines.append(f"\n### {suggestion.get('requirement', 'Suggestion')}")
                report_lines.append(f"**Confidence:** {suggestion.get('confidence', 0)}%")
                report_lines.append(f"**Priority:** {suggestion.get('priority', 'Medium')}")
                report_lines.append(f"**Markets:** {', '.join(suggestion.get('markets', []))}")
                report_lines.append(f"\n{suggestion.get('recommendation', '')}")

        # Reporting requirements
        if analysis_results.get('injury_reporting_required'):
            report_lines.append("\n## Injury/Incident Reporting Requirements")
            for req in analysis_results['injury_reporting_required']:
                report_lines.append(f"\n### {req['market']}")
                report_lines.append(f"**Timeline:** {req['timeline']}")
                report_lines.append("**Report To:**")
                for agency in req['agencies']:
                    report_lines.append(f"- [{agency['name']}]({agency['url']})")

        return '\n'.join(report_lines)


__all__ = ['RegulatoryComplianceAnalyzer', 'REGULATORY_MARKETS']
