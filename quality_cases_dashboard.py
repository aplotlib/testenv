"""
Quality Cases Dashboard Module

Manages the top 12 quality cases derived from three key reports:
1. Returns Analysis (trailing 12-month data)
2. B2B Sales Feedback
3. Carolina's Reviews Analysis

Each case is generated based on resulting criteria specific to each report type.
Tracks monthly progress and impact metrics.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


# Resulting Criteria for Each Report Type
REPORT_CRITERIA = {
    'Returns Analysis': {
        'description': 'Trailing 12-month Amazon return data analyzed for patterns',
        'criteria': [
            {
                'name': 'Category Return Rate Threshold',
                'logic': 'Product return rate > category-specific threshold',
                'thresholds': {
                    'SUP (Support Products)': '11-20%',
                    'RHB (Rehabilitation)': '10.5-24%',
                    'LVA (Living Aids)': '9-18.5%',
                    'CSH (Cushions)': '11-18%',
                    'MOB (Mobility)': '10-15%',
                    'INS (Insoles)': '12-14%',
                    'CPAP Masks': '8-12%',
                    'CPAP Machines': '6-10%',
                    'POC (Portable Oxygen)': '7-11%',
                }
            },
            {
                'name': 'Safety Risk Flag',
                'logic': 'Complaints contain keywords: brake, fall, collapse, unstable, shock, burn, injury'
            },
            {
                'name': 'High Cost Impact',
                'logic': 'Return value > $10,000/month (units returned × landed cost)'
            },
            {
                'name': 'Statistical Outlier',
                'logic': 'Product exceeds 2σ or 3σ control limits in SPC analysis'
            },
            {
                'name': 'Complaint Volume',
                'logic': '>50 returns per month with similar complaint patterns'
            }
        ],
        'case_trigger': 'Any criteria met = Case generated'
    },
    'B2B Sales Feedback': {
        'description': 'Feedback from B2B sales team (distributors, healthcare facilities)',
        'criteria': [
            {
                'name': 'Multiple Customer Complaints',
                'logic': '≥3 different customers report same issue within 30 days'
            },
            {
                'name': 'Large Order Rejection',
                'logic': 'B2B customer returns >25% of order due to quality concerns'
            },
            {
                'name': 'Distributor Escalation',
                'logic': 'Distributor formally requests CAPA or quality meeting'
            },
            {
                'name': 'Clinical Feedback',
                'logic': 'Healthcare provider reports patient safety or efficacy concern'
            },
            {
                'name': 'Lost Account Risk',
                'logic': 'Customer threatens to discontinue SKU due to quality'
            }
        ],
        'case_trigger': 'Any criteria met = Case generated'
    },
    'Reviews Analysis': {
        'description': "Carolina's systematic analysis of Amazon reviews and customer feedback",
        'criteria': [
            {
                'name': 'Star Rating Threshold',
                'logic': 'Product average rating drops below 3.8 stars with >20 reviews'
            },
            {
                'name': 'Recent Negative Trend',
                'logic': '≥5 new 1-2 star reviews in past 30 days mentioning same issue'
            },
            {
                'name': 'Competitor Comparison',
                'logic': 'Product rated 0.5+ stars lower than similar competitor products'
            },
            {
                'name': 'Safety Mention in Reviews',
                'logic': 'Reviews mention safety concerns, injury, or dangerous conditions'
            },
            {
                'name': 'Sentiment Analysis Alert',
                'logic': 'AI sentiment analysis flags deteriorating customer perception'
            }
        ],
        'case_trigger': 'Any criteria met = Case generated'
    }
}


# Priority Scoring System
PRIORITY_WEIGHTS = {
    'safety_risk': 40,           # Safety always highest priority
    'financial_impact': 25,       # Cost of returns + lost sales
    'customer_impact': 20,        # Number of customers affected
    'regulatory_risk': 10,        # Potential FDA/regulatory issues
    'brand_reputation': 5         # Public perception risk
}


class QualityCase:
    """Represents a single quality case being tracked"""

    def __init__(
        self,
        case_id: str,
        product_sku: str,
        product_name: str,
        source_report: str,
        criteria_triggered: List[str],
        opened_date: datetime,
        priority: str,
        assigned_to: str,
        category: str = "General"
    ):
        self.case_id = case_id
        self.product_sku = product_sku
        self.product_name = product_name
        self.source_report = source_report
        self.criteria_triggered = criteria_triggered
        self.opened_date = opened_date
        self.priority = priority  # Critical, High, Medium
        self.assigned_to = assigned_to
        self.category = category
        self.status = "Open"
        self.target_close_date = None
        self.actual_close_date = None

        # Progress tracking
        self.milestones = []
        self.monthly_updates = []

        # Impact metrics
        self.baseline_metrics = {}
        self.current_metrics = {}
        self.impact_realized = {}

    def add_milestone(self, milestone: str, due_date: datetime, completed: bool = False):
        """Add a milestone to track case progress"""
        self.milestones.append({
            'milestone': milestone,
            'due_date': due_date,
            'completed': completed,
            'completed_date': datetime.now() if completed else None
        })

    def add_monthly_update(self, month: str, progress: str, metrics: Dict[str, Any]):
        """Add monthly progress update"""
        self.monthly_updates.append({
            'month': month,
            'progress': progress,
            'metrics': metrics,
            'timestamp': datetime.now()
        })

    def calculate_priority_score(self) -> int:
        """Calculate numerical priority score (0-100)"""
        score = 0

        # Safety risk
        if 'Safety Risk Flag' in self.criteria_triggered or 'safety' in str(self.criteria_triggered).lower():
            score += PRIORITY_WEIGHTS['safety_risk']

        # Financial impact (from baseline metrics)
        monthly_cost = self.baseline_metrics.get('monthly_return_cost', 0)
        if monthly_cost > 20000:
            score += PRIORITY_WEIGHTS['financial_impact']
        elif monthly_cost > 10000:
            score += PRIORITY_WEIGHTS['financial_impact'] * 0.7
        elif monthly_cost > 5000:
            score += PRIORITY_WEIGHTS['financial_impact'] * 0.4

        # Customer impact
        customers_affected = self.baseline_metrics.get('customers_affected', 0)
        if customers_affected > 100:
            score += PRIORITY_WEIGHTS['customer_impact']
        elif customers_affected > 50:
            score += PRIORITY_WEIGHTS['customer_impact'] * 0.6

        # Regulatory risk
        if self.category in ['CPAP Machines', 'POC', 'Mobility']:
            score += PRIORITY_WEIGHTS['regulatory_risk']

        # Brand reputation
        if 'Reviews Analysis' in self.source_report:
            score += PRIORITY_WEIGHTS['brand_reputation']

        return min(score, 100)

    def to_dict(self) -> Dict[str, Any]:
        """Convert case to dictionary for display"""
        return {
            'Case ID': self.case_id,
            'SKU': self.product_sku,
            'Product': self.product_name,
            'Category': self.category,
            'Source Report': self.source_report,
            'Criteria': ', '.join(self.criteria_triggered[:2]) + ('...' if len(self.criteria_triggered) > 2 else ''),
            'Priority': self.priority,
            'Status': self.status,
            'Opened': self.opened_date.strftime('%Y-%m-%d'),
            'Assigned To': self.assigned_to,
            'Days Open': (datetime.now() - self.opened_date).days,
            'Priority Score': self.calculate_priority_score()
        }


class QualityCasesDashboard:
    """Manages the top 12 quality cases and their tracking"""

    def __init__(self):
        self.cases: List[QualityCase] = []
        self.max_active_cases = 12

    def add_case(self, case: QualityCase):
        """Add a new quality case"""
        self.cases.append(case)

    def get_top_cases(self, limit: int = 12) -> List[QualityCase]:
        """Get top N cases sorted by priority score"""
        sorted_cases = sorted(
            [c for c in self.cases if c.status == 'Open'],
            key=lambda x: x.calculate_priority_score(),
            reverse=True
        )
        return sorted_cases[:limit]

    def get_cases_by_category(self, category: str) -> List[QualityCase]:
        """Get all cases for a specific category"""
        return [c for c in self.cases if c.category == category and c.status == 'Open']

    def get_monthly_summary(self, month: Optional[str] = None) -> Dict[str, Any]:
        """Generate monthly summary report"""
        if month is None:
            month = datetime.now().strftime('%Y-%m')

        summary = {
            'month': month,
            'total_open_cases': len([c for c in self.cases if c.status == 'Open']),
            'total_closed_cases': len([c for c in self.cases if c.status == 'Closed']),
            'critical_cases': len([c for c in self.cases if c.priority == 'Critical' and c.status == 'Open']),
            'high_cases': len([c for c in self.cases if c.priority == 'High' and c.status == 'Open']),
            'cases_by_source': {},
            'cases_by_category': {},
            'total_impact_realized': {}
        }

        # Count by source
        for case in self.cases:
            if case.status == 'Open':
                source = case.source_report
                summary['cases_by_source'][source] = summary['cases_by_source'].get(source, 0) + 1

                category = case.category
                summary['cases_by_category'][category] = summary['cases_by_category'].get(category, 0) + 1

        # Calculate total impact
        for case in self.cases:
            if case.impact_realized:
                for metric, value in case.impact_realized.items():
                    summary['total_impact_realized'][metric] = summary['total_impact_realized'].get(metric, 0) + value

        return summary

    def generate_cases_dataframe(self) -> pd.DataFrame:
        """Generate DataFrame of all cases for display"""
        if not self.cases:
            return pd.DataFrame()

        cases_data = [case.to_dict() for case in self.get_top_cases()]
        return pd.DataFrame(cases_data)


# Demo Data: Top 12 Quality Cases (January 2026)
def generate_demo_cases() -> QualityCasesDashboard:
    """Generate realistic demo cases for the dashboard"""

    dashboard = QualityCasesDashboard()

    # Case 1: CPAP Mask - Critical
    case1 = QualityCase(
        case_id="QC-2026-001",
        product_sku="CPAP-MASK-001",
        product_name="ComfortGel Nasal CPAP Mask",
        source_report="Returns Analysis + B2B Sales Feedback",
        criteria_triggered=["Category Return Rate Threshold", "Multiple Customer Complaints", "Safety Risk Flag"],
        opened_date=datetime.now() - timedelta(days=45),
        priority="Critical",
        assigned_to="Sarah Chen",
        category="CPAP Masks"
    )
    case1.baseline_metrics = {
        'return_rate': 0.145,
        'monthly_returns': 87,
        'monthly_return_cost': 18750,
        'customers_affected': 87,
        'avg_star_rating': 3.2
    }
    case1.add_milestone("Root cause analysis complete", datetime.now() + timedelta(days=7), completed=True)
    case1.add_milestone("Vendor CAPA issued", datetime.now() + timedelta(days=14), completed=True)
    case1.add_milestone("Improved units arrive", datetime.now() + timedelta(days=30), completed=False)
    case1.add_monthly_update("2026-01", "RCA identified seal material degradation. Vendor working on improved silicone compound.", {
        'return_rate': 0.132,
        'monthly_returns': 76,
        'improvement': '12.6% reduction'
    })
    dashboard.add_case(case1)

    # Case 2: Portable Oxygen Concentrator - Critical
    case2 = QualityCase(
        case_id="QC-2026-002",
        product_sku="POC-5L-ADV",
        product_name="5L Portable Oxygen Concentrator Advanced",
        source_report="B2B Sales Feedback + Reviews Analysis",
        criteria_triggered=["Distributor Escalation", "Clinical Feedback", "Recent Negative Trend"],
        opened_date=datetime.now() - timedelta(days=38),
        priority="Critical",
        assigned_to="Michael Rodriguez",
        category="POC"
    )
    case2.baseline_metrics = {
        'return_rate': 0.098,
        'monthly_returns': 23,
        'monthly_return_cost': 34500,
        'customers_affected': 23,
        'avg_star_rating': 3.5,
        'b2b_complaints': 4
    }
    case2.add_milestone("Engineering review battery performance", datetime.now() + timedelta(days=5), completed=True)
    case2.add_milestone("Field testing improved battery", datetime.now() + timedelta(days=20), completed=False)
    case2.add_monthly_update("2026-01", "Battery capacity testing shows 15% degradation after 50 cycles vs. spec of 10%. Evaluating new battery supplier.", {
        'return_rate': 0.098,
        'status': 'Investigation ongoing'
    })
    dashboard.add_case(case2)

    # Case 3: CPAP Machine - Critical
    case3 = QualityCase(
        case_id="QC-2026-003",
        product_sku="CPAP-AUTO-PRO",
        product_name="Auto-Adjusting CPAP Machine Pro Series",
        source_report="Returns Analysis + Reviews Analysis",
        criteria_triggered=["Category Return Rate Threshold", "Star Rating Threshold", "High Cost Impact"],
        opened_date=datetime.now() - timedelta(days=52),
        priority="Critical",
        assigned_to="Sarah Chen",
        category="CPAP Machines"
    )
    case3.baseline_metrics = {
        'return_rate': 0.087,
        'monthly_returns': 31,
        'monthly_return_cost': 27900,
        'customers_affected': 31,
        'avg_star_rating': 3.6
    }
    case3.status = "Closed"
    case3.actual_close_date = datetime.now() - timedelta(days=5)
    case3.impact_realized = {
        'return_rate_reduction': 0.032,
        'cost_savings_monthly': 8500,
        'star_rating_improvement': 0.4
    }
    case3.add_monthly_update("2025-12", "Firmware update released. Motor noise reduced by 40dB.", {
        'return_rate': 0.055,
        'improvement': '36.8% reduction'
    })
    dashboard.add_case(case3)

    # Case 4: Shoulder Brace - High Priority
    case4 = QualityCase(
        case_id="QC-2026-004",
        product_sku="SUP1041BGEFBM",
        product_name="Shoulder Brace - Rotator Cuff",
        source_report="Returns Analysis",
        criteria_triggered=["Category Return Rate Threshold", "Statistical Outlier", "Complaint Volume"],
        opened_date=datetime.now() - timedelta(days=28),
        priority="High",
        assigned_to="Jennifer Wu",
        category="SUP"
    )
    case4.baseline_metrics = {
        'return_rate': 0.6296,
        'monthly_returns': 68,
        'monthly_return_cost': 3694,
        'customers_affected': 68
    }
    case4.add_milestone("Quality inspection of current inventory", datetime.now() + timedelta(days=3), completed=True)
    case4.add_milestone("Vendor 8D report due", datetime.now() + timedelta(days=15), completed=False)
    dashboard.add_case(case4)

    # Case 5: Portable Stand Assist - High Priority (Safety)
    case5 = QualityCase(
        case_id="QC-2026-005",
        product_sku="LVA3016BLK",
        product_name="Portable Stand Assist (Black)",
        source_report="Returns Analysis + Reviews Analysis",
        criteria_triggered=["Safety Risk Flag", "Category Return Rate Threshold", "Safety Mention in Reviews"],
        opened_date=datetime.now() - timedelta(days=35),
        priority="High",
        assigned_to="Michael Rodriguez",
        category="LVA"
    )
    case5.baseline_metrics = {
        'return_rate': 0.5001,
        'monthly_returns': 178,
        'monthly_return_cost': 13962,
        'customers_affected': 178,
        'safety_incidents': 3
    }
    case5.add_milestone("Safety risk assessment complete", datetime.now() + timedelta(days=2), completed=True)
    case5.add_milestone("Design review for stability improvements", datetime.now() + timedelta(days=10), completed=False)
    dashboard.add_case(case5)

    # Case 6: Folding Mobility Scooter - High Priority
    case6 = QualityCase(
        case_id="QC-2026-006",
        product_sku="MOB1058BLKFBM",
        product_name="Folding Mobility Scooter",
        source_report="B2B Sales Feedback + Returns Analysis",
        criteria_triggered=["Large Order Rejection", "Category Return Rate Threshold"],
        opened_date=datetime.now() - timedelta(days=21),
        priority="High",
        assigned_to="David Park",
        category="MOB"
    )
    case6.baseline_metrics = {
        'return_rate': 0.1538,
        'monthly_returns': 6,
        'monthly_return_cost': 4135,
        'b2b_order_rejection': 1,
        'customers_affected': 6
    }
    dashboard.add_case(case6)

    # Case 7: CPAP Mask Headgear - High Priority
    case7 = QualityCase(
        case_id="QC-2026-007",
        product_sku="CPAP-GEAR-002",
        product_name="Universal CPAP Headgear Replacement",
        source_report="Reviews Analysis + Returns Analysis",
        criteria_triggered=["Recent Negative Trend", "Category Return Rate Threshold"],
        opened_date=datetime.now() - timedelta(days=18),
        priority="High",
        assigned_to="Sarah Chen",
        category="CPAP Masks"
    )
    case7.baseline_metrics = {
        'return_rate': 0.112,
        'monthly_returns': 45,
        'monthly_return_cost': 1350,
        'customers_affected': 45,
        'avg_star_rating': 3.7,
        'recent_1star_reviews': 7
    }
    dashboard.add_case(case7)

    # Case 8: Knee Walker - Medium Priority
    case8 = QualityCase(
        case_id="QC-2026-008",
        product_sku="MOB1019BLKFBM",
        product_name="All Terrain Knee Walker",
        source_report="Returns Analysis",
        criteria_triggered=["Category Return Rate Threshold", "Safety Risk Flag"],
        opened_date=datetime.now() - timedelta(days=14),
        priority="Medium",
        assigned_to="David Park",
        category="MOB"
    )
    case8.baseline_metrics = {
        'return_rate': 0.1102,
        'monthly_returns': 42,
        'monthly_return_cost': 3757,
        'customers_affected': 42
    }
    dashboard.add_case(case8)

    # Case 9: Post OP Shoe - Medium Priority
    case9 = QualityCase(
        case_id="QC-2026-009",
        product_sku="RHB2096BLKL",
        product_name="Closed Toe Post OP Shoe",
        source_report="Returns Analysis + B2B Sales Feedback",
        criteria_triggered=["Category Return Rate Threshold", "Multiple Customer Complaints"],
        opened_date=datetime.now() - timedelta(days=12),
        priority="Medium",
        assigned_to="Jennifer Wu",
        category="RHB"
    )
    case9.baseline_metrics = {
        'return_rate': 0.3052,
        'monthly_returns': 1023,
        'monthly_return_cost': 34996,
        'customers_affected': 1023
    }
    dashboard.add_case(case9)

    # Case 10: Bathtub Rail - Medium Priority (Safety)
    case10 = QualityCase(
        case_id="QC-2026-010",
        product_sku="LVA1021FBM",
        product_name="Bathtub Rail",
        source_report="Reviews Analysis + Returns Analysis",
        criteria_triggered=["Safety Mention in Reviews", "Category Return Rate Threshold"],
        opened_date=datetime.now() - timedelta(days=9),
        priority="Medium",
        assigned_to="Michael Rodriguez",
        category="LVA"
    )
    case10.baseline_metrics = {
        'return_rate': 0.1915,
        'monthly_returns': 86,
        'monthly_return_cost': 2569,
        'customers_affected': 86,
        'safety_mentions': 5
    }
    dashboard.add_case(case10)

    # Case 11: Hinged Knee Brace - Medium Priority
    case11 = QualityCase(
        case_id="QC-2026-011",
        product_sku="SUP1046BLKM",
        product_name="Hinged Knee Brace",
        source_report="Returns Analysis",
        criteria_triggered=["Category Return Rate Threshold", "Complaint Volume"],
        opened_date=datetime.now() - timedelta(days=7),
        priority="Medium",
        assigned_to="Jennifer Wu",
        category="SUP"
    )
    case11.baseline_metrics = {
        'return_rate': 0.2045,
        'monthly_returns': 382,
        'monthly_return_cost': 4874,
        'customers_affected': 382
    }
    dashboard.add_case(case11)

    # Case 12: Transfer Belt - Medium Priority
    case12 = QualityCase(
        case_id="QC-2026-012",
        product_sku="RHB1011L",
        product_name="Transfer Belt with Leg Loops",
        source_report="B2B Sales Feedback + Reviews Analysis",
        criteria_triggered=["Clinical Feedback", "Star Rating Threshold"],
        opened_date=datetime.now() - timedelta(days=5),
        priority="Medium",
        assigned_to="Michael Rodriguez",
        category="RHB"
    )
    case12.baseline_metrics = {
        'return_rate': 0.1702,
        'monthly_returns': 289,
        'monthly_return_cost': 4208,
        'customers_affected': 289,
        'avg_star_rating': 3.7,
        'clinical_feedback_count': 2
    }
    dashboard.add_case(case12)

    return dashboard


# Product Development Focus Areas
PRODUCT_DEVELOPMENT_FOCUS = {
    'CPAP Masks': {
        'why_focus': 'High regulatory scrutiny (FDA Class II), customer comfort critical for therapy compliance',
        'key_quality_areas': [
            'Seal integrity and leak prevention',
            'Skin irritation and pressure points',
            'Headgear durability and adjustability',
            'Material biocompatibility',
            'Size/fit accuracy across diverse populations'
        ],
        'upfront_quality_actions': [
            'Wear testing with 20+ users before launch',
            'Dermatological testing for skin contact materials',
            'Leak testing at multiple pressure settings',
            'FDA 510(k) pre-submission consultation',
            'Competitive teardown and benchmarking'
        ],
        'success_metrics': {
            'target_return_rate': '< 8%',
            'target_star_rating': '> 4.2',
            'target_first_pass_yield': '> 98%'
        }
    },
    'CPAP Machines': {
        'why_focus': 'FDA Class II device, high cost, critical for patient safety and therapy efficacy',
        'key_quality_areas': [
            'Pressure accuracy and stability',
            'Noise levels (< 30 dBA)',
            'Motor reliability and lifespan',
            'Data logging and connectivity',
            'Electrical safety and EMC compliance'
        ],
        'upfront_quality_actions': [
            'Design FMEA with clinical team input',
            'Accelerated life testing (ALT) for motor',
            'Clinical validation study (n=50+)',
            'Full EMC and electrical safety testing',
            'Reliability demonstration test (RDT)',
            'FDA 510(k) clearance before launch'
        ],
        'success_metrics': {
            'target_return_rate': '< 6%',
            'target_star_rating': '> 4.3',
            'target_MTBF': '> 5 years'
        }
    },
    'POC (Portable Oxygen Concentrators)': {
        'why_focus': 'FDA Class II, life-sustaining device, high cost, complex technology',
        'key_quality_areas': [
            'Oxygen concentration accuracy (90%±3%)',
            'Battery performance and runtime',
            'Altitude performance',
            'Flow sensor accuracy',
            'Alarm system reliability'
        ],
        'upfront_quality_actions': [
            'Clinical evaluation per ISO 80601-2-69',
            'Battery cycle life testing (500+ cycles)',
            'Altitude chamber testing',
            'Oxygen purity verification (multiple units)',
            'FAA approval for aircraft use',
            'FDA 510(k) clearance required'
        ],
        'success_metrics': {
            'target_return_rate': '< 7%',
            'target_star_rating': '> 4.4',
            'target_oxygen_purity': '90% ± 3% at all settings'
        }
    }
}


__all__ = [
    'QualityCase',
    'QualityCasesDashboard',
    'REPORT_CRITERIA',
    'PRIORITY_WEIGHTS',
    'PRODUCT_DEVELOPMENT_FOCUS',
    'generate_demo_cases'
]
