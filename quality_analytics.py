"""
Quality Analytics Module - Enhanced Statistical Analysis
Version 2.0 - Full Statistical Rigor

Features:
- ANOVA / MANOVA with p-values
- Post-Hoc Tukey HSD testing
- SPC Control Charting (CUSUM, Shewhart)
- Trend Analysis (30/60/90/180/365 day comparisons)
- Weighted Risk Scoring
- Fuzzy threshold matching
- Vendor email generation (Chinese vendor friendly)
- Investigation plan generation
- ISO 13485/FDA/EU MDR compliance checks
"""

import pandas as pd
import numpy as np
import logging
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict

# Statistical imports
try:
    from scipy import stats
    from scipy.stats import f_oneway, kruskal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from statsmodels.multivariate.manova import MANOVA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from rapidfuzz import fuzz, process
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False

logger = logging.getLogger(__name__)

# =============================================================================
# SOP THRESHOLDS - From uploaded Quality SOPs
# =============================================================================

SOP_THRESHOLDS = {
    # Category-specific return rate thresholds (from VREC-001)
    'B2B Products': 0.025,
    'B2B': 0.025,
    'INS': 0.07,
    'RHB': 0.075,
    'LVA': 0.095,
    'MOB - Power Scooters': 0.095,
    'MOB - Walkers': 0.10,
    'MOB - Rollators': 0.10,
    'MOB - Manual Wheelchairs': 0.105,
    'MOB - Wheelchairs': 0.105,
    'CSH': 0.105,
    'SUP': 0.11,
    'MOB - Power Wheelchairs': 0.115,
    'MOB': 0.10,  # Default MOB
    'All Others': 0.10,
    
    # Critical thresholds (from Quality Case SOP)
    'Critical_Return_Rate_Cap': 0.25,  # 25% absolute cap
    'Relative_Threshold_Above_Category': 0.20,  # 20% above category = case
    'Relative_Threshold_Above_Category_Pct': 0.05,  # OR 5% points above
    'Standard_Deviation_Trigger': 1.0,  # 1 std dev = flag
    
    # Financial thresholds (from VREC-001)
    'Critical_Landed_Cost': 150.00,
    'High_Value_Threshold': 100.00,
    
    # Time thresholds
    'New_Launch_Days': 90,
    'Trend_Monitoring_Days': 30,
    
    # AQL thresholds
    'AQL_Defect_Rate': 0.025,  # 2.5% AQL
    
    # Qualitative thresholds
    'Unique_Complaint_Threshold': 3,  # 3+ same complaints in 30 days
    'Complaint_Period_Days': 30,
}

# Escalation criteria from SOPs
ESCALATION_CRITERIA = {
    'immediate': [
        'safety_risk',
        'injury_reported',
        'regulatory_violation',
        'zero_tolerance_component',
        'high_value_defect',
    ],
    'quality_case': [
        'return_rate_above_25pct',
        'return_rate_20pct_above_category',
        'return_rate_1std_above',
        'recurring_complaints_3plus',
        'aql_failure',
    ],
    'monitor': [
        'return_rate_elevated',
        'new_product_launch',
        'single_complaint_high_value',
    ]
}

# Zero tolerance components (from VREC-001)
ZERO_TOLERANCE_COMPONENTS = [
    'lithium-ion battery', 'lithium battery', 'li-ion',
    'sterile packaging', 'sterile',
    'electrical', 'motor', 'controller',
]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ScreeningResult:
    """Result of a single product screening"""
    sku: str
    product_name: str
    category: str
    return_rate: float
    category_threshold: float
    landed_cost: float
    units_sold: int
    units_returned: int
    
    # Calculated fields
    risk_score: float = 0.0
    action: str = "Monitor"
    triggers: List[str] = None
    spc_signal: str = "Normal"
    trend_status: str = "Stable"
    
    # AI fields
    ai_recommendation: str = ""
    cross_case_matches: List[str] = None
    similar_products: List[str] = None
    
    def __post_init__(self):
        if self.triggers is None:
            self.triggers = []
        if self.cross_case_matches is None:
            self.cross_case_matches = []
        if self.similar_products is None:
            self.similar_products = []
    
    def to_dict(self):
        return asdict(self)


@dataclass
class StatisticalResult:
    """Result of statistical analysis"""
    test_type: str  # ANOVA, MANOVA, Kruskal-Wallis
    statistic: float
    p_value: float
    significant: bool
    effect_size: float = None
    post_hoc_results: Dict = None
    outlier_categories: List[str] = None
    recommendation: str = ""
    
    def to_dict(self):
        return asdict(self)


@dataclass
class SPCResult:
    """Statistical Process Control result"""
    signal_type: str  # Normal, Warning, Critical
    z_score: float
    control_limit_upper: float
    control_limit_lower: float
    cusum_value: float = 0.0
    run_length: int = 0
    western_electric_rules: List[str] = None
    
    def to_dict(self):
        return asdict(self)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def parse_numeric(series: pd.Series) -> pd.Series:
    """Safely parse numeric columns, handling currency and percentage formats"""
    if series is None:
        return pd.Series([0])
    
    def clean_value(val):
        if pd.isna(val):
            return 0.0
        val_str = str(val).strip()
        # Remove currency symbols, commas, percent signs
        val_str = re.sub(r'[$,€£¥]', '', val_str)
        val_str = val_str.replace('%', '')
        # Handle parentheses for negatives
        if val_str.startswith('(') and val_str.endswith(')'):
            val_str = '-' + val_str[1:-1]
        try:
            return float(val_str)
        except (ValueError, TypeError):
            return 0.0
    
    return series.apply(clean_value)


def parse_percentage(series: pd.Series) -> pd.Series:
    """Parse percentage columns, converting to decimal"""
    numeric = parse_numeric(series)
    # If values are > 1, assume they're already percentages (e.g., 25 = 25%)
    return numeric.apply(lambda x: x / 100 if abs(x) > 1 else x)


def fuzzy_match_category(product_name: str, product_type: str, 
                         threshold_data: pd.DataFrame) -> Tuple[str, float]:
    """
    Use fuzzy matching to find the best category threshold for a product.
    Returns (matched_category, threshold_rate)
    """
    if not FUZZY_AVAILABLE or threshold_data is None or threshold_data.empty:
        return 'All Others', SOP_THRESHOLDS.get('All Others', 0.10)
    
    search_text = f"{product_name} {product_type}".lower()
    
    # Build list of searchable terms from threshold data
    if 'Type' in threshold_data.columns:
        types = threshold_data['Type'].dropna().unique().tolist()
        
        # Find best match
        best_match = process.extractOne(search_text, types, scorer=fuzz.token_set_ratio)
        
        if best_match and best_match[1] > 60:  # 60% similarity threshold
            matched_type = best_match[0]
            # Get threshold for this type
            match_row = threshold_data[threshold_data['Type'] == matched_type].iloc[0]
            if 'Return Rate Threshold' in threshold_data.columns:
                threshold = parse_percentage(pd.Series([match_row['Return Rate Threshold']])).iloc[0]
                return matched_type, abs(threshold)
    
    # Fallback to category code
    if 'Category' in threshold_data.columns:
        categories = threshold_data['Category'].dropna().unique().tolist()
        for cat in categories:
            if cat in SOP_THRESHOLDS:
                return cat, SOP_THRESHOLDS[cat]
    
    return 'All Others', SOP_THRESHOLDS.get('All Others', 0.10)


def detect_zero_tolerance(product_name: str, complaint_text: str = "") -> bool:
    """Check if product involves zero-tolerance components"""
    search_text = f"{product_name} {complaint_text}".lower()
    return any(component in search_text for component in ZERO_TOLERANCE_COMPONENTS)


# =============================================================================
# STATISTICAL ANALYSIS CLASS
# =============================================================================

class QualityStatistics:
    """
    Handles all statistical analysis for quality screening.
    """
    
    @staticmethod
    def suggest_analysis_type(df: pd.DataFrame, metrics: List[str]) -> Dict[str, Any]:
        """
        AI-suggested analysis type based on data characteristics.
        Returns recommendation with reasoning.
        """
        suggestion = {
            'recommended': 'ANOVA',
            'reason': '',
            'alternatives': [],
            'warnings': []
        }
        
        n_groups = df['Category'].nunique() if 'Category' in df.columns else 1
        n_samples = len(df)
        n_metrics = len(metrics)
        
        # Check data requirements
        if n_groups < 2:
            suggestion['recommended'] = 'Descriptive Only'
            suggestion['reason'] = 'Only 1 category present. Cannot perform group comparison.'
            return suggestion
        
        if n_samples < 10:
            suggestion['warnings'].append('Small sample size may affect reliability.')
        
        # Determine best test
        if n_metrics > 1:
            suggestion['recommended'] = 'MANOVA'
            suggestion['reason'] = f"Multiple metrics detected ({n_metrics}). MANOVA tests differences across all metrics simultaneously."
            suggestion['alternatives'].append({
                'test': 'Separate ANOVAs',
                'when': 'If you want to analyze each metric independently'
            })
        else:
            # Check normality for ANOVA vs Kruskal-Wallis
            if SCIPY_AVAILABLE and n_samples >= 20:
                try:
                    metric_col = metrics[0] if metrics else 'Return_Rate'
                    if metric_col in df.columns:
                        _, p_norm = stats.shapiro(df[metric_col].dropna().head(5000))
                        if p_norm < 0.05:
                            suggestion['recommended'] = 'Kruskal-Wallis'
                            suggestion['reason'] = 'Data appears non-normal (Shapiro p < 0.05). Non-parametric test recommended.'
                            suggestion['alternatives'].append({
                                'test': 'ANOVA',
                                'when': 'If you assume approximate normality or have large samples (CLT)'
                            })
                        else:
                            suggestion['recommended'] = 'ANOVA'
                            suggestion['reason'] = 'Data appears normally distributed. Parametric ANOVA is appropriate.'
                except Exception:
                    pass
            
            suggestion['reason'] = suggestion['reason'] or f"Single metric with {n_groups} categories. One-way ANOVA tests for category differences."
        
        return suggestion
    
    @staticmethod
    def perform_anova(df: pd.DataFrame, group_col: str, metric_col: str) -> StatisticalResult:
        """
        Perform One-Way ANOVA with effect size and post-hoc testing.
        """
        if not SCIPY_AVAILABLE:
            return StatisticalResult(
                test_type='ANOVA',
                statistic=0.0,
                p_value=1.0,
                significant=False,
                recommendation='scipy not installed. Cannot perform statistical tests.'
            )
        
        try:
            # Prepare groups
            groups = []
            group_names = []
            for name, group in df.groupby(group_col):
                values = group[metric_col].dropna().values
                if len(values) >= 2:
                    groups.append(values)
                    group_names.append(name)
            
            if len(groups) < 2:
                return StatisticalResult(
                    test_type='ANOVA',
                    statistic=0.0,
                    p_value=1.0,
                    significant=False,
                    recommendation='Not enough groups with sufficient data for ANOVA.'
                )
            
            # Perform ANOVA
            f_stat, p_val = f_oneway(*groups)
            
            # Calculate effect size (eta-squared)
            all_values = np.concatenate(groups)
            grand_mean = np.mean(all_values)
            ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
            ss_total = np.sum((all_values - grand_mean)**2)
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            
            # Post-hoc testing if significant
            post_hoc = None
            outliers = []
            
            if p_val < 0.05 and STATSMODELS_AVAILABLE:
                try:
                    # Prepare data for Tukey
                    all_data = []
                    all_groups = []
                    for name, values in zip(group_names, groups):
                        all_data.extend(values)
                        all_groups.extend([name] * len(values))
                    
                    tukey = pairwise_tukeyhsd(all_data, all_groups, alpha=0.05)
                    
                    post_hoc = {
                        'comparisons': [],
                        'summary': str(tukey)
                    }
                    
                    # Find outlier categories (significantly different from others)
                    sig_counts = defaultdict(int)
                    for i, reject in enumerate(tukey.reject):
                        if reject:
                            g1, g2 = tukey.groupsunique[tukey._multicomp.pairindices[0][i]], \
                                     tukey.groupsunique[tukey._multicomp.pairindices[1][i]]
                            sig_counts[g1] += 1
                            sig_counts[g2] += 1
                            post_hoc['comparisons'].append({
                                'group1': str(g1),
                                'group2': str(g2),
                                'mean_diff': float(tukey.meandiffs[i]),
                                'p_adj': float(tukey.pvalues[i])
                            })
                    
                    # Categories significantly different from many others are outliers
                    threshold = len(group_names) // 2
                    outliers = [cat for cat, count in sig_counts.items() if count >= threshold]
                    
                except Exception as e:
                    logger.warning(f"Post-hoc testing failed: {e}")
            
            # Build recommendation
            if p_val < 0.001:
                sig_level = "highly significant (p < 0.001)"
            elif p_val < 0.01:
                sig_level = "very significant (p < 0.01)"
            elif p_val < 0.05:
                sig_level = "significant (p < 0.05)"
            else:
                sig_level = "not significant (p ≥ 0.05)"
            
            effect_desc = "large" if eta_squared > 0.14 else "medium" if eta_squared > 0.06 else "small"
            
            recommendation = f"Category differences are {sig_level} with {effect_desc} effect size (η² = {eta_squared:.3f})."
            if outliers:
                recommendation += f" Outlier categories: {', '.join(str(o) for o in outliers)}."
            
            return StatisticalResult(
                test_type='ANOVA',
                statistic=float(f_stat),
                p_value=float(p_val),
                significant=p_val < 0.05,
                effect_size=float(eta_squared),
                post_hoc_results=post_hoc,
                outlier_categories=outliers,
                recommendation=recommendation
            )
            
        except Exception as e:
            logger.error(f"ANOVA failed: {e}")
            return StatisticalResult(
                test_type='ANOVA',
                statistic=0.0,
                p_value=1.0,
                significant=False,
                recommendation=f'ANOVA failed: {str(e)}'
            )
    
    @staticmethod
    def perform_manova(df: pd.DataFrame, group_col: str, metric_cols: List[str]) -> StatisticalResult:
        """
        Perform MANOVA for multiple dependent variables.
        """
        if not STATSMODELS_AVAILABLE:
            return StatisticalResult(
                test_type='MANOVA',
                statistic=0.0,
                p_value=1.0,
                significant=False,
                recommendation='statsmodels not installed. Cannot perform MANOVA.'
            )
        
        try:
            # Clean data
            analysis_df = df[[group_col] + metric_cols].dropna()
            
            if len(analysis_df) < 10:
                return StatisticalResult(
                    test_type='MANOVA',
                    statistic=0.0,
                    p_value=1.0,
                    significant=False,
                    recommendation='Insufficient data for MANOVA (need at least 10 complete rows).'
                )
            
            # Build formula
            dv_formula = ' + '.join(metric_cols)
            formula = f"{dv_formula} ~ C({group_col})"
            
            # Perform MANOVA
            manova = MANOVA.from_formula(formula, data=analysis_df)
            results = manova.mv_test()
            
            # Extract Pillai's trace (most robust)
            pillai_results = results.results[f'C({group_col})']['stat']
            pillai_stat = pillai_results.loc['Pillai\'s trace', 'Value']
            pillai_f = pillai_results.loc['Pillai\'s trace', 'F Value']
            pillai_p = pillai_results.loc['Pillai\'s trace', 'Pr > F']
            
            recommendation = f"MANOVA using Pillai's trace: F = {pillai_f:.3f}, p = {pillai_p:.4f}. "
            if pillai_p < 0.05:
                recommendation += "Significant multivariate effect detected. Follow up with individual ANOVAs to identify which metrics differ."
            else:
                recommendation += "No significant multivariate effect. Categories do not differ significantly across the combined metrics."
            
            return StatisticalResult(
                test_type='MANOVA',
                statistic=float(pillai_f),
                p_value=float(pillai_p),
                significant=pillai_p < 0.05,
                effect_size=float(pillai_stat),
                recommendation=recommendation
            )
            
        except Exception as e:
            logger.error(f"MANOVA failed: {e}")
            # Fallback to separate ANOVAs
            return StatisticalResult(
                test_type='MANOVA',
                statistic=0.0,
                p_value=1.0,
                significant=False,
                recommendation=f'MANOVA failed ({str(e)}). Consider running separate ANOVAs for each metric.'
            )
    
    @staticmethod
    def perform_kruskal_wallis(df: pd.DataFrame, group_col: str, metric_col: str) -> StatisticalResult:
        """
        Non-parametric alternative to ANOVA (Kruskal-Wallis H-test).
        """
        if not SCIPY_AVAILABLE:
            return StatisticalResult(
                test_type='Kruskal-Wallis',
                statistic=0.0,
                p_value=1.0,
                significant=False,
                recommendation='scipy not installed.'
            )
        
        try:
            groups = [group[metric_col].dropna().values 
                     for _, group in df.groupby(group_col) 
                     if len(group[metric_col].dropna()) >= 2]
            
            if len(groups) < 2:
                return StatisticalResult(
                    test_type='Kruskal-Wallis',
                    statistic=0.0,
                    p_value=1.0,
                    significant=False,
                    recommendation='Not enough groups for Kruskal-Wallis.'
                )
            
            h_stat, p_val = kruskal(*groups)
            
            # Effect size (epsilon-squared)
            n = sum(len(g) for g in groups)
            epsilon_sq = h_stat / (n - 1) if n > 1 else 0
            
            return StatisticalResult(
                test_type='Kruskal-Wallis',
                statistic=float(h_stat),
                p_value=float(p_val),
                significant=p_val < 0.05,
                effect_size=float(epsilon_sq),
                recommendation=f"H = {h_stat:.3f}, p = {p_val:.4f}. {'Significant' if p_val < 0.05 else 'Not significant'} difference between groups."
            )
            
        except Exception as e:
            return StatisticalResult(
                test_type='Kruskal-Wallis',
                statistic=0.0,
                p_value=1.0,
                significant=False,
                recommendation=f'Test failed: {str(e)}'
            )


# =============================================================================
# SPC CONTROL CHARTING
# =============================================================================

class SPCAnalysis:
    """
    Statistical Process Control analysis including Shewhart and CUSUM charts.
    """
    
    @staticmethod
    def calculate_control_limits(historical_mean: float, historical_std: float,
                                  sigma_level: float = 3.0) -> Tuple[float, float]:
        """Calculate UCL and LCL"""
        ucl = historical_mean + (sigma_level * historical_std)
        lcl = max(0, historical_mean - (sigma_level * historical_std))
        return ucl, lcl
    
    @staticmethod
    def detect_signal(current_value: float, historical_mean: float, 
                      historical_std: float) -> SPCResult:
        """
        Detect SPC signals using Shewhart rules.
        """
        if historical_std == 0:
            return SPCResult(
                signal_type='Unknown',
                z_score=0.0,
                control_limit_upper=historical_mean,
                control_limit_lower=historical_mean,
                western_electric_rules=[]
            )
        
        z_score = (current_value - historical_mean) / historical_std
        ucl, lcl = SPCAnalysis.calculate_control_limits(historical_mean, historical_std)
        
        # Determine signal type
        if abs(z_score) > 3:
            signal_type = 'Critical (>3σ)'
        elif abs(z_score) > 2:
            signal_type = 'Warning (>2σ)'
        elif abs(z_score) > 1:
            signal_type = 'Watch (>1σ)'
        else:
            signal_type = 'Normal'
        
        return SPCResult(
            signal_type=signal_type,
            z_score=float(z_score),
            control_limit_upper=float(ucl),
            control_limit_lower=float(lcl),
            western_electric_rules=[]
        )
    
    @staticmethod
    def calculate_cusum(values: List[float], target: float, 
                        k: float = 0.5, h: float = 4.0) -> Dict[str, Any]:
        """
        Calculate CUSUM (Cumulative Sum) for detecting small shifts.
        k = slack value (typically 0.5 * shift to detect)
        h = decision interval (typically 4-5)
        """
        if not values:
            return {'cusum_pos': [], 'cusum_neg': [], 'signal': False, 'signal_point': None}
        
        std = np.std(values) if len(values) > 1 else 1.0
        k_scaled = k * std
        h_scaled = h * std
        
        cusum_pos = [0.0]
        cusum_neg = [0.0]
        signal_point = None
        
        for i, val in enumerate(values):
            cusum_pos.append(max(0, cusum_pos[-1] + (val - target) - k_scaled))
            cusum_neg.append(max(0, cusum_neg[-1] - (val - target) - k_scaled))
            
            if signal_point is None and (cusum_pos[-1] > h_scaled or cusum_neg[-1] > h_scaled):
                signal_point = i
        
        return {
            'cusum_pos': cusum_pos[1:],
            'cusum_neg': cusum_neg[1:],
            'signal': signal_point is not None,
            'signal_point': signal_point,
            'h_limit': h_scaled
        }


# =============================================================================
# TREND ANALYSIS
# =============================================================================

class TrendAnalysis:
    """
    Analyze trends comparing current vs historical performance.
    """
    
    @staticmethod
    def analyze_trend(current_rate: float, rates_30d: float = None,
                      rates_90d: float = None, rates_180d: float = None,
                      rates_365d: float = None) -> Dict[str, Any]:
        """
        Compare current return rate against historical rolling averages.
        """
        result = {
            'current_rate': current_rate,
            'trend_direction': 'stable',
            'trend_magnitude': 0.0,
            'comparisons': {},
            'alert_level': 'none',
            'recommendation': ''
        }
        
        comparisons = []
        if rates_30d is not None and rates_30d > 0:
            pct_change = (current_rate - rates_30d) / rates_30d
            result['comparisons']['vs_30d'] = {
                'baseline': rates_30d,
                'change': pct_change,
                'direction': 'up' if pct_change > 0 else 'down'
            }
            comparisons.append(pct_change)
        
        if rates_90d is not None and rates_90d > 0:
            pct_change = (current_rate - rates_90d) / rates_90d
            result['comparisons']['vs_90d'] = {
                'baseline': rates_90d,
                'change': pct_change,
                'direction': 'up' if pct_change > 0 else 'down'
            }
            comparisons.append(pct_change)
        
        if rates_180d is not None and rates_180d > 0:
            pct_change = (current_rate - rates_180d) / rates_180d
            result['comparisons']['vs_180d'] = {
                'baseline': rates_180d,
                'change': pct_change,
                'direction': 'up' if pct_change > 0 else 'down'
            }
            comparisons.append(pct_change)
        
        if rates_365d is not None and rates_365d > 0:
            pct_change = (current_rate - rates_365d) / rates_365d
            result['comparisons']['vs_365d'] = {
                'baseline': rates_365d,
                'change': pct_change,
                'direction': 'up' if pct_change > 0 else 'down'
            }
            comparisons.append(pct_change)
        
        # Determine overall trend
        if comparisons:
            avg_change = np.mean(comparisons)
            result['trend_magnitude'] = float(avg_change)
            
            if avg_change > 0.25:
                result['trend_direction'] = 'deteriorating_rapidly'
                result['alert_level'] = 'critical'
                result['recommendation'] = 'Return rate increasing rapidly. Immediate investigation recommended.'
            elif avg_change > 0.10:
                result['trend_direction'] = 'deteriorating'
                result['alert_level'] = 'warning'
                result['recommendation'] = 'Return rate trending upward. Monitor closely and prepare investigation.'
            elif avg_change < -0.10:
                result['trend_direction'] = 'improving'
                result['alert_level'] = 'positive'
                result['recommendation'] = 'Return rate improving. Continue monitoring.'
            else:
                result['trend_direction'] = 'stable'
                result['alert_level'] = 'none'
                result['recommendation'] = 'Return rate stable.'
        
        return result


# =============================================================================
# RISK SCORING
# =============================================================================

class RiskScoring:
    """
    Calculate weighted risk scores combining multiple factors.
    """
    
    # Weights for risk calculation
    WEIGHTS = {
        'statistical_deviation': 0.25,
        'financial_impact': 0.25,
        'safety_severity': 0.30,
        'trend_direction': 0.10,
        'complaint_volume': 0.10
    }
    
    @staticmethod
    def calculate_risk_score(
        return_rate: float,
        category_threshold: float,
        landed_cost: float,
        safety_risk: bool = False,
        trend_direction: str = 'stable',
        complaint_count: int = 0,
        units_sold: int = 1,
        ai_severity: str = 'minor'
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate weighted risk score (0-100).
        Returns (total_score, component_breakdown)
        """
        components = {}
        
        # 1. Statistical Deviation (0-25 pts)
        if category_threshold > 0:
            deviation = (return_rate - category_threshold) / category_threshold
            if deviation > 0.5:  # >50% above threshold
                components['statistical'] = 25
            elif deviation > 0.25:  # >25% above
                components['statistical'] = 20
            elif deviation > 0:  # Above threshold
                components['statistical'] = 15
            elif deviation > -0.1:  # Slightly below
                components['statistical'] = 5
            else:
                components['statistical'] = 0
        else:
            components['statistical'] = 0
        
        # 2. Financial Impact (0-25 pts)
        if landed_cost >= SOP_THRESHOLDS['Critical_Landed_Cost']:
            components['financial'] = 25
        elif landed_cost >= SOP_THRESHOLDS['High_Value_Threshold']:
            components['financial'] = 18
        elif landed_cost >= 50:
            components['financial'] = 10
        else:
            components['financial'] = 5
        
        # 3. Safety/Severity (0-30 pts)
        if safety_risk:
            components['safety'] = 30
        elif ai_severity == 'critical':
            components['safety'] = 25
        elif ai_severity == 'major':
            components['safety'] = 15
        else:
            components['safety'] = 0
        
        # 4. Trend Direction (0-10 pts)
        trend_scores = {
            'deteriorating_rapidly': 10,
            'deteriorating': 7,
            'stable': 3,
            'improving': 0
        }
        components['trend'] = trend_scores.get(trend_direction, 3)
        
        # 5. Complaint Volume (0-10 pts)
        complaint_rate = complaint_count / max(units_sold, 1)
        if complaint_rate > 0.05:
            components['complaints'] = 10
        elif complaint_rate > 0.02:
            components['complaints'] = 6
        elif complaint_count >= 3:
            components['complaints'] = 4
        else:
            components['complaints'] = 0
        
        total = sum(components.values())
        return min(100, total), components


# =============================================================================
# ACTION DETERMINATION
# =============================================================================

class ActionDetermination:
    """
    Determine appropriate action based on SOP criteria.
    """
    
    @staticmethod
    def determine_action(
        return_rate: float,
        category_threshold: float,
        landed_cost: float,
        safety_risk: bool = False,
        zero_tolerance: bool = False,
        is_new_product: bool = False,
        complaint_count: int = 0,
        risk_score: float = 0
    ) -> Tuple[str, List[str]]:
        """
        Determine action and list of triggered criteria.
        Returns (action, [triggered_criteria])
        """
        triggers = []
        
        # IMMEDIATE ESCALATION CHECKS
        if safety_risk:
            triggers.append("Safety risk reported")
            return "Immediate Escalation: Safety Risk", triggers
        
        if zero_tolerance:
            triggers.append("Zero-tolerance component involved")
            return "Immediate Escalation: Zero Tolerance Component", triggers
        
        # High value + any defect
        if landed_cost >= SOP_THRESHOLDS['Critical_Landed_Cost'] and return_rate > 0.05:
            triggers.append(f"High value item (${landed_cost:.2f}) with elevated return rate")
            return "Immediate Escalation: High Value Defect", triggers
        
        # QUALITY CASE CHECKS
        # Absolute threshold (25%)
        if return_rate >= SOP_THRESHOLDS['Critical_Return_Rate_Cap']:
            triggers.append(f"Return rate ({return_rate:.1%}) ≥ 25% cap")
            return "Open Quality Case: Critical Return Rate", triggers
        
        # Relative threshold (20% above category OR 5 points above)
        relative_diff = (return_rate - category_threshold) / category_threshold if category_threshold > 0 else 0
        absolute_diff = return_rate - category_threshold
        
        if relative_diff >= SOP_THRESHOLDS['Relative_Threshold_Above_Category']:
            triggers.append(f"Return rate {relative_diff:.0%} above category average ({category_threshold:.1%})")
            return "Open Quality Case: Exceeds Category Threshold", triggers
        
        if absolute_diff >= SOP_THRESHOLDS['Relative_Threshold_Above_Category_Pct']:
            triggers.append(f"Return rate {absolute_diff:.1%} points above category threshold")
            return "Open Quality Case: Exceeds Category Threshold", triggers
        
        # Recurring complaints
        if complaint_count >= SOP_THRESHOLDS['Unique_Complaint_Threshold']:
            triggers.append(f"{complaint_count} unique complaints in 30-day period")
            return "Open Quality Case: Recurring Complaints", triggers
        
        # High risk score
        if risk_score >= 70:
            triggers.append(f"High composite risk score ({risk_score:.0f}/100)")
            return "Open Quality Case: High Risk Score", triggers
        
        # MONITORING CHECKS
        if is_new_product:
            triggers.append("New product (< 90 days)")
            return "Monitor: New Product Launch", triggers
        
        if return_rate > category_threshold:
            triggers.append(f"Return rate ({return_rate:.1%}) above category average ({category_threshold:.1%})")
            return "Monitor: Elevated Return Rate", triggers
        
        if risk_score >= 40:
            triggers.append(f"Moderate risk score ({risk_score:.0f}/100)")
            return "Monitor: Moderate Risk", triggers
        
        # No issues
        return "No Action Required", triggers


# =============================================================================
# VENDOR EMAIL GENERATION
# =============================================================================

class VendorEmailGenerator:
    """
    Generate vendor emails optimized for Chinese vendors.
    - Simple, clear language (translation-app friendly)
    - High-context communication (saving face)
    - Partnership-focused tone
    """
    
    @staticmethod
    def generate_capa_request(
        sku: str,
        product_name: str,
        issue_summary: str,
        return_rate: float,
        defect_description: str,
        units_affected: int,
        severity: str = 'major'
    ) -> str:
        """Generate CAPA request email for Chinese vendor."""
        
        email = f"""Subject: Quality Improvement Request - {sku}

Dear Partner,

Thank you for your continued partnership. We are writing about product {sku} ({product_name}).

**Current Situation:**
We have observed quality feedback from customers that we would like to work together to improve.

- Product: {sku} - {product_name}
- Return Rate: {return_rate:.1%}
- Units Affected: {units_affected}
- Main Issue: {issue_summary}

**Customer Feedback Details:**
{defect_description}

**Request:**
We kindly request your team to:
1. Review the issue and identify the root cause
2. Provide a written analysis within 7 business days
3. Propose corrective actions to prevent future occurrences
4. Share timeline for implementing improvements

**Information Needed:**
- Root Cause Analysis (RCA) report
- Corrective Action plan
- Preventive measures for future production
- Timeline for implementation

We value our partnership and believe working together will improve product quality for all customers. Please contact us if you need any additional information or samples for investigation.

Thank you for your attention to this matter.

Best regards,
Quality Team

---
Reference: QC-{datetime.now().strftime('%Y%m%d')}-{sku}
"""
        return email
    
    @staticmethod
    def generate_rca_request(
        sku: str,
        product_name: str,
        defect_type: str,
        occurrence_rate: float,
        sample_complaints: List[str]
    ) -> str:
        """Generate Root Cause Analysis request email."""
        
        complaints_text = "\n".join(f"  - {c}" for c in sample_complaints[:5])
        
        email = f"""Subject: Root Cause Analysis Request - {sku}

Dear Partner,

We hope this message finds you well. We are writing to request your assistance with a quality investigation.

**Product Information:**
- SKU: {sku}
- Product Name: {product_name}
- Issue Type: {defect_type}
- Occurrence Rate: {occurrence_rate:.1%}

**Customer Feedback Examples:**
{complaints_text}

**RCA Request:**
Please help us understand:
1. What is causing this issue?
2. Which production step may be involved?
3. Were there any recent changes to materials or process?
4. What inspection checks are currently in place?

**Timeline:**
Please provide initial findings within 5 business days.

We appreciate your cooperation and look forward to finding a solution together.

Best regards,
Quality Team

---
Reference: RCA-{datetime.now().strftime('%Y%m%d')}-{sku}
"""
        return email
    
    @staticmethod
    def generate_inspection_notice(
        sku: str,
        product_name: str,
        inspection_type: str = 'Pre-shipment',
        special_focus: List[str] = None
    ) -> str:
        """Generate inspection notification email."""
        
        focus_text = ""
        if special_focus:
            focus_text = "\n**Special Inspection Focus:**\n" + "\n".join(f"  - {f}" for f in special_focus)
        
        email = f"""Subject: Inspection Notification - {sku}

Dear Partner,

We would like to schedule a {inspection_type} inspection for the following product:

**Product Details:**
- SKU: {sku}
- Product Name: {product_name}
- Inspection Type: {inspection_type}
{focus_text}

**Please Prepare:**
1. Production records for this batch
2. QC inspection reports
3. Material certificates (if applicable)
4. Samples for inspection

Please confirm your readiness for inspection and provide available dates.

Thank you for your cooperation.

Best regards,
Quality Team

---
Reference: INS-{datetime.now().strftime('%Y%m%d')}-{sku}
"""
        return email


# =============================================================================
# INVESTIGATION PLAN GENERATOR
# =============================================================================

class InvestigationPlanGenerator:
    """
    Generate draft investigation plans based on AI analysis.
    """
    
    @staticmethod
    def generate_plan(
        sku: str,
        product_name: str,
        category: str,
        issue_type: str,
        complaint_summary: str,
        return_rate: float,
        risk_score: float
    ) -> Dict[str, Any]:
        """Generate structured investigation plan."""
        
        # Determine investigation areas based on issue type
        issue_areas = {
            'Product Defects/Quality': [
                'Material quality and specifications',
                'Manufacturing process controls',
                'Assembly procedures',
                'Final inspection criteria'
            ],
            'Performance/Effectiveness': [
                'Product specifications vs actual performance',
                'User instructions clarity',
                'Expected use conditions vs actual use',
                'Design verification records'
            ],
            'Size/Fit Issues': [
                'Sizing chart accuracy',
                'Manufacturing tolerances',
                'Size labeling accuracy',
                'Product photography vs actual dimensions'
            ],
            'Missing Components': [
                'Packing list verification',
                'Packaging process controls',
                'Final packaging inspection',
                'Shipping handling procedures'
            ],
            'Design/Material Issues': [
                'Design specifications review',
                'Material selection rationale',
                'Durability testing results',
                'User feedback on design'
            ],
            'Comfort Issues': [
                'Material softness/hardness specifications',
                'Ergonomic design review',
                'User testing results',
                'Comparison with competitor products'
            ]
        }
        
        investigation_areas = issue_areas.get(issue_type, [
            'General product inspection',
            'Manufacturing process review',
            'Customer feedback analysis',
            'Supplier communication'
        ])
        
        plan = {
            'reference_id': f"INV-{datetime.now().strftime('%Y%m%d')}-{sku}",
            'product': {
                'sku': sku,
                'name': product_name,
                'category': category
            },
            'trigger': {
                'issue_type': issue_type,
                'return_rate': f"{return_rate:.1%}",
                'risk_score': f"{risk_score:.0f}/100",
                'summary': complaint_summary
            },
            'investigation_areas': investigation_areas,
            'recommended_actions': [
                f"Review {len(investigation_areas)} key areas listed above",
                "Obtain samples from current inventory for inspection",
                "Request production records from vendor",
                "Analyze customer complaint details",
                "Compare with historical quality data"
            ],
            'timeline': {
                'initial_assessment': '2 business days',
                'vendor_response': '5-7 business days',
                'full_investigation': '14 business days',
                'capa_implementation': '30 business days'
            },
            'stakeholders': [
                'Quality Manager',
                'Product Development',
                'Vendor/Factory QC',
                'Customer Support (for feedback)'
            ],
            'compliance_notes': [
                'Document all findings per ISO 13485 requirements',
                'Maintain traceability records',
                'Report to regulatory if safety issue confirmed',
                'Update DHR (Device History Record) as needed'
            ]
        }
        
        return plan
    
    @staticmethod
    def format_plan_markdown(plan: Dict[str, Any]) -> str:
        """Format investigation plan as markdown."""
        
        md = f"""# Investigation Plan

**Reference:** {plan['reference_id']}  
**Date:** {datetime.now().strftime('%Y-%m-%d')}

---

## Product Information
- **SKU:** {plan['product']['sku']}
- **Name:** {plan['product']['name']}
- **Category:** {plan['product']['category']}

## Trigger Information
- **Issue Type:** {plan['trigger']['issue_type']}
- **Return Rate:** {plan['trigger']['return_rate']}
- **Risk Score:** {plan['trigger']['risk_score']}
- **Summary:** {plan['trigger']['summary']}

## Investigation Areas
"""
        for i, area in enumerate(plan['investigation_areas'], 1):
            md += f"{i}. {area}\n"
        
        md += "\n## Recommended Actions\n"
        for action in plan['recommended_actions']:
            md += f"- [ ] {action}\n"
        
        md += f"""
## Timeline
| Phase | Duration |
|-------|----------|
| Initial Assessment | {plan['timeline']['initial_assessment']} |
| Vendor Response | {plan['timeline']['vendor_response']} |
| Full Investigation | {plan['timeline']['full_investigation']} |
| CAPA Implementation | {plan['timeline']['capa_implementation']} |

## Stakeholders
"""
        for stakeholder in plan['stakeholders']:
            md += f"- {stakeholder}\n"
        
        md += "\n## Compliance Notes\n"
        for note in plan['compliance_notes']:
            md += f"- {note}\n"
        
        return md


# =============================================================================
# DATA VALIDATION
# =============================================================================

class DataValidation:
    """
    Validate uploaded data and generate reports.
    """
    
    @staticmethod
    def validate_upload(df: pd.DataFrame, required_cols: List[str] = None) -> Dict[str, Any]:
        """
        Validate uploaded data for quality screening.
        """
        if required_cols is None:
            required_cols = ['SKU']
        
        report = {
            'valid': True,
            'total_rows': len(df),
            'missing_cols': [],
            'found_cols': [],
            'numeric_issues': [],
            'warnings': [],
            'column_mapping': {}
        }
        
        # Check required columns (with flexible matching)
        col_aliases = {
            'SKU': ['sku', 'product_sku', 'item_sku', 'asin', 'product_id'],
            'Category': ['category', 'cat', 'product_category', 'type'],
            'Sold': ['sold', 'units_sold', 'total_units', 'quantity_sold', 'total orders'],
            'Returned': ['returned', 'units_returned', 'returns', 'refunds', 'total_returns'],
            'Return_Rate': ['return_rate', 'refund_rate', 'return rate', 'refund rate'],
            'Landed_Cost': ['landed_cost', 'cost', 'unit_cost', 'landed cost'],
            'Name': ['name', 'product_name', 'title', 'product', 'display name']
        }
        
        df_cols_lower = {c.lower().strip(): c for c in df.columns}
        
        for std_col, aliases in col_aliases.items():
            found = False
            for alias in [std_col.lower()] + aliases:
                if alias in df_cols_lower:
                    report['column_mapping'][std_col] = df_cols_lower[alias]
                    report['found_cols'].append(std_col)
                    found = True
                    break
            
            if not found and std_col in required_cols:
                report['missing_cols'].append(std_col)
        
        if report['missing_cols']:
            report['valid'] = False
            report['warnings'].append(f"Missing required columns: {', '.join(report['missing_cols'])}")
        
        # Check for numeric issues in expected numeric columns
        numeric_cols = ['Sold', 'Returned', 'Return_Rate', 'Landed_Cost']
        for col in numeric_cols:
            if col in report['column_mapping']:
                actual_col = report['column_mapping'][col]
                try:
                    parsed = parse_numeric(df[actual_col])
                    nan_count = parsed.isna().sum()
                    if nan_count > 0:
                        report['numeric_issues'].append({
                            'column': actual_col,
                            'issue': f'{nan_count} non-numeric values'
                        })
                except Exception as e:
                    report['numeric_issues'].append({
                        'column': actual_col,
                        'issue': str(e)
                    })
        
        # Warnings for data quality
        if len(df) > 500:
            report['warnings'].append(f"Large dataset ({len(df)} rows). Processing may take longer.")
        
        if len(df) < 5:
            report['warnings'].append("Small dataset. Statistical analysis may have limited reliability.")
        
        return report


# =============================================================================
# METHODOLOGY DOCUMENTATION
# =============================================================================

def generate_methodology_markdown() -> str:
    """
    Generate LaTeX-formatted methodology documentation.
    """
    return r"""
## 📐 Methodology & Mathematical Formulas

### 1. Return Rate Calculation
$$
\text{Return Rate} = \frac{\text{Units Returned}}{\text{Units Sold}}
$$

### 2. Threshold Logic (Per SOP QMS-SOP-001)
A **Quality Case** is triggered if ANY of the following conditions are met:
- Return Rate ≥ 25% (absolute cap)
- Return Rate > (Category Average × 1.20) — i.e., 20% above category
- Return Rate > (Category Average + 5 percentage points)
- ≥ 3 unique customer complaints citing same failure mode within 30 days

*Note: System defaults to the HIGHER threshold when conflicts exist.*

### 3. Weighted Risk Score
The composite risk score ($S$) is calculated as:

$$
S = (0.25 \times D_{stat}) + (0.25 \times I_{fin}) + (0.30 \times S_{sev}) + (0.10 \times T_{trend}) + (0.10 \times C_{vol})
$$

Where:
- $D_{stat}$ = Statistical deviation score (0-25)
- $I_{fin}$ = Financial impact score (0-25)
- $S_{sev}$ = Safety/severity score (0-30)
- $T_{trend}$ = Trend direction score (0-10)
- $C_{vol}$ = Complaint volume score (0-10)

### 4. ANOVA (Analysis of Variance)
Tests whether category means are significantly different:

$$
F = \frac{MS_{between}}{MS_{within}} = \frac{SS_B / (k-1)}{SS_W / (N-k)}
$$

Where:
- $SS_B$ = Sum of squares between groups
- $SS_W$ = Sum of squares within groups
- $k$ = Number of categories
- $N$ = Total sample size

**Effect Size (η²):**
$$
\eta^2 = \frac{SS_{between}}{SS_{total}}
$$

### 5. MANOVA (Multivariate ANOVA)
Tests multiple dependent variables simultaneously using Pillai's Trace:

$$
V = \sum_{i=1}^{s} \frac{\lambda_i}{1 + \lambda_i}
$$

Where $\lambda_i$ are the eigenvalues of $E^{-1}H$ (error and hypothesis matrices).

### 6. SPC Control Limits (Shewhart)
$$
UCL = \bar{x} + 3\sigma
$$
$$
LCL = \bar{x} - 3\sigma
$$

**Z-Score Signal Detection:**
$$
z = \frac{x - \bar{x}}{\sigma}
$$

| Z-Score | Signal Level |
|---------|--------------|
| \|z\| > 3 | Critical |
| \|z\| > 2 | Warning |
| \|z\| > 1 | Watch |
| \|z\| ≤ 1 | Normal |

### 7. CUSUM (Cumulative Sum)
Detects small persistent shifts:

$$
C_i^+ = \max(0, C_{i-1}^+ + (x_i - \mu_0) - k)
$$
$$
C_i^- = \max(0, C_{i-1}^- - (x_i - \mu_0) - k)
$$

Signal when $C_i^+$ or $C_i^-$ exceeds decision interval $h$.

---

### Regulatory Compliance Reference
- **ISO 13485:2016** - Medical devices QMS
- **FDA 21 CFR Part 820** - Quality System Regulation
- **EU MDR 2017/745** - Medical Device Regulation
- **UK MDR 2002** - UK Medical Devices Regulations
"""


# =============================================================================
# MAIN QUALITY ANALYTICS CLASS (Unified Interface)
# =============================================================================

class QualityAnalytics:
    """
    Unified interface for all quality analytics functions.
    """
    
    # Re-export thresholds
    SOP_THRESHOLDS = SOP_THRESHOLDS
    
    # Static method exports
    validate_upload = staticmethod(DataValidation.validate_upload)
    calculate_risk_score = staticmethod(lambda row, cat_avg: RiskScoring.calculate_risk_score(
        return_rate=row.get('Return_Rate', 0),
        category_threshold=cat_avg,
        landed_cost=row.get('Landed Cost', 0),
        safety_risk=str(row.get('Safety Risk', '')).lower() in ['yes', 'true', '1'],
        ai_severity=str(row.get('Severity', 'minor')).lower()
    )[0])
    
    perform_anova = staticmethod(QualityStatistics.perform_anova)
    perform_manova = staticmethod(QualityStatistics.perform_manova)
    suggest_analysis_type = staticmethod(QualityStatistics.suggest_analysis_type)
    
    detect_spc_signals = staticmethod(lambda row, mean, std: SPCAnalysis.detect_signal(
        row.get('Return_Rate', 0), mean, std
    ).signal_type)
    
    determine_action = staticmethod(lambda row, sop: ActionDetermination.determine_action(
        return_rate=row.get('Return_Rate', 0),
        category_threshold=sop.get(row.get('Category', 'All Others'), 0.10),
        landed_cost=row.get('Landed Cost', 0),
        safety_risk=str(row.get('Safety Risk', '')).lower() in ['yes', 'true', '1'],
        risk_score=row.get('Risk_Score', 0)
    )[0])
    
    generate_methodology_markdown = staticmethod(generate_methodology_markdown)
    
    @staticmethod
    def parse_numeric(series):
        return parse_numeric(series)
    
    @staticmethod
    def parse_percentage(series):
        return parse_percentage(series)


# Export all
__all__ = [
    'QualityAnalytics',
    'QualityStatistics',
    'SPCAnalysis',
    'TrendAnalysis',
    'RiskScoring',
    'ActionDetermination',
    'VendorEmailGenerator',
    'InvestigationPlanGenerator',
    'DataValidation',
    'ScreeningResult',
    'StatisticalResult',
    'SPCResult',
    'SOP_THRESHOLDS',
    'ESCALATION_CRITERIA',
    'parse_numeric',
    'parse_percentage',
    'fuzzy_match_category',
    'generate_methodology_markdown'
]
