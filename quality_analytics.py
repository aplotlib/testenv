import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json

# Try to import statistical libraries
try:
    from scipy import stats
    STATS_AVAILABLE = True
except ImportError:
    STATS_AVAILABLE = False

try:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from statsmodels.multivariate.manova import MANOVA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

logger = logging.getLogger(__name__)

# SOP Thresholds (Derived from uploaded SOPs)
SOP_THRESHOLDS = {
    'B2B Products': 0.025,
    'INS': 0.07,
    'RHB': 0.075,
    'LVA': 0.095,
    'MOB - Power Scooters': 0.095,
    'MOB - Walkers': 0.10,
    'MOB - Manual Wheelchairs': 0.105,
    'CSH': 0.105,
    'SUP': 0.11,
    'MOB - Power Wheelchairs': 0.115,
    'All Others': 0.10,
    'Critical_Return_Rate_Cap': 0.25,
    'Critical_Landed_Cost': 150.00,
    'New_Launch_Days': 90
}

def parse_numeric(series: pd.Series) -> pd.Series:
    """Helper to safely parse numeric columns"""
    return pd.to_numeric(series.astype(str).str.replace(r'[$,%]', '', regex=True), errors='coerce').fillna(0)

class QualityAnalytics:
    """
    Handles statistical analysis, risk scoring, and SOP compliance checks.
    """
    
    @staticmethod
    def validate_upload(
        df: pd.DataFrame,
        required_cols: List[str],
        numeric_cols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Validates uploaded data for missing fields and numeric errors.
        """
        report = {
            'valid': True,
            'missing_cols': [],
            'non_numeric_rows': [],
            'total_rows': len(df)
        }
        
        # Check columns
        for col in required_cols:
            if col not in df.columns:
                report['missing_cols'].append(col)
                report['valid'] = False

        if numeric_cols:
            for col in numeric_cols:
                if col in df.columns:
                    series = df[col]
                    numeric = pd.to_numeric(series.astype(str).str.replace(r'[$,%]', '', regex=True), errors='coerce')
                    bad_rows = df[series.notna() & numeric.isna()].index.tolist()
                    if bad_rows:
                        report['non_numeric_rows'].append({'column': col, 'rows': bad_rows})
                        report['valid'] = False
        
        return report

    @staticmethod
    def calculate_weighted_risk_score(
        row: pd.Series,
        category_avg: float,
        category_std: float = 0.0,
        ai_severity_score: float = 0.0
    ) -> float:
        """
        Calculates Weighted Risk Score.
        Factors:
        - Statistical deviation (30%)
        - Financial impact (Landed Cost) (30%)
        - Safety/Severity (40%)
        """
        score = 0.0
        
        # 1. Statistical Deviation (30 pts)
        rr = row.get('Return_Rate', 0)
        if category_std > 0 and rr > (category_avg + 2 * category_std):
            score += 30
        elif rr > (category_avg * 1.5):
            score += 30
        elif rr > category_avg:
            score += 15
            
        # 2. Financial Impact (30 pts)
        cost = row.get('Landed Cost', 0)
        if cost > 150: # High value threshold
            score += 30
        elif cost > 50:
            score += 15
            
        # 3. Severity (40 pts)
        # Assumes AI has tagged severity or user manual input
        severity = str(row.get('Safety Risk', '')).lower()
        if severity in ['yes', 'critical', 'high']:
            score += 40
        elif severity in ['major', 'medium']:
            score += 20
        else:
            score += min(max(ai_severity_score, 0), 40)
            
        return score

    @staticmethod
    def perform_anova(df: pd.DataFrame, category_col: str, metric_col: str) -> Dict[str, Any]:
        """
        Performs One-Way ANOVA and returns F-score and p-value.
        """
        if not STATS_AVAILABLE:
            return {'error': 'Scipy not installed. Cannot calculate p-values.'}
            
        try:
            groups = [group[metric_col].dropna().values for name, group in df.groupby(category_col)]
            if len(groups) < 2:
                return {'error': 'Not enough categories for ANOVA'}
                
            f_stat, p_val = stats.f_oneway(*groups)
            
            # Post-Hoc Logic (Simplified Tukey equivalent)
            # Identify outliers if ANOVA is significant
            outliers = []
            if p_val < 0.05:
                grand_mean = df[metric_col].mean()
                grand_std = df[metric_col].std()
                
                cat_stats = df.groupby(category_col)[metric_col].agg(['mean', 'count'])
                # Flag categories > 1 std dev from grand mean
                outliers = cat_stats[cat_stats['mean'] > (grand_mean + grand_std)].index.tolist()

            return {
                'f_statistic': f_stat,
                'p_value': p_val,
                'significant': p_val < 0.05,
                'outlier_categories': outliers
            }
        except Exception as e:
            return {'error': str(e)}

    @staticmethod
    def perform_tukey_hsd(df: pd.DataFrame, category_col: str, metric_col: str) -> Dict[str, Any]:
        """
        Runs Tukey HSD if statsmodels is available.
        """
        if not STATSMODELS_AVAILABLE:
            return {'error': 'statsmodels not installed. Cannot calculate Tukey HSD.'}

        try:
            data = df[[category_col, metric_col]].dropna()
            if data[category_col].nunique() < 2:
                return {'error': 'Not enough categories for Tukey HSD.'}
            tukey = pairwise_tukeyhsd(endog=data[metric_col], groups=data[category_col], alpha=0.05)
            results = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
            significant = results[results['reject'] == True]
            return {
                'results': results,
                'significant_pairs': significant
            }
        except Exception as e:
            return {'error': str(e)}

    @staticmethod
    def perform_manova(df: pd.DataFrame, category_col: str, metric_cols: List[str]) -> Dict[str, Any]:
        """
        Performs MANOVA and returns p-values when possible.
        """
        if not STATSMODELS_AVAILABLE:
            return {'error': 'statsmodels not installed. Cannot calculate MANOVA.'}

        try:
            metrics = [col for col in metric_cols if col in df.columns]
            if len(metrics) < 2:
                return {'error': 'Not enough metrics for MANOVA.'}
            formula = f"{' + '.join(metrics)} ~ {category_col}"
            maov = MANOVA.from_formula(formula, data=df.dropna(subset=metrics + [category_col]))
            test_results = maov.mv_test()
            stats_table = test_results.results[category_col]['stat']
            p_value = stats_table.loc['Wilks\' lambda', 'Pr > F']
            return {
                'p_value': float(p_value),
                'stat_table': stats_table.reset_index()
            }
        except Exception as e:
            return {'error': str(e)}

    @staticmethod
    def analyze_trend(row: pd.Series) -> Dict[str, Any]:
        """
        Compares 30-day return rate to 6M and 12M rolling averages.
        """
        current = row.get('Return_Rate_30D', row.get('Return_Rate', None))
        avg_6m = row.get('Return_Rate_6M', None)
        avg_12m = row.get('Return_Rate_12M', None)
        trend = "Unknown"
        flags = []

        if current is None:
            return {'trend': trend, 'flags': flags}

        if avg_6m is not None:
            if current > avg_6m:
                flags.append("Above 6M Avg")
            elif current < avg_6m:
                flags.append("Below 6M Avg")
        if avg_12m is not None:
            if current > avg_12m:
                flags.append("Above 12M Avg")
            elif current < avg_12m:
                flags.append("Below 12M Avg")

        if flags:
            trend = ", ".join(flags)
        return {'trend': trend, 'flags': flags}

    @staticmethod
    def detect_spc_signals(row: pd.Series, history_mean: float, history_std: float) -> str:
        """
        Control Charting (SPC) Logic.
        Detects if current rate > 3 sigma from historical mean.
        """
        if history_std == 0:
            return "No Signal"
            
        current = row.get('Return_Rate', 0)
        z_score = (current - history_mean) / history_std
        
        if z_score > 3:
            return "Critical Signal (>3σ)"
        elif z_score > 2:
            return "Warning Signal (>2σ)"
        return "Normal Control"

    @staticmethod
    def determine_action(row: pd.Series, sop_benchmarks: Dict) -> str:
        """
        Determines action based on SOP Logic and Hierarchy.
        """
        # 1. Immediate Safety/Regulatory Checks
        if str(row.get('Safety Risk', '')).lower() in ['yes', 'true', '1']:
            return "Escalate: Safety Risk (Immediate)"
        
        if str(row.get('Zero Tolerance Component', '')).lower() in ['yes', 'true', '1']:
            return "Escalate: Zero Tolerance"
            
        # 2. Financial/Cost Checks
        cost = row.get('Landed Cost', 0)
        critical_cost = sop_benchmarks.get('Critical_Landed_Cost', SOP_THRESHOLDS['Critical_Landed_Cost'])
        if cost >= critical_cost:
            # For high cost, we are stricter
            if row.get('Return_Rate', 0) > 0.05: # Arbitrary strict threshold for high value
                return "Escalate: High Value Defect"
                
        # 3. Return Rate Thresholds (Logic from SOPs)
        # Use higher threshold if conflict exists
        rr = row.get('Return_Rate', 0)
        cat = row.get('Category', 'All Others')
        benchmark = sop_benchmarks.get(cat, SOP_THRESHOLDS['All Others'])
        
        # Hard Cap
        critical_cap = sop_benchmarks.get('Critical_Return_Rate_Cap', SOP_THRESHOLDS['Critical_Return_Rate_Cap'])
        if rr >= critical_cap:
             return "Escalate: Critical Return Rate (>25%)"
             
        # Relative to Benchmark
        if rr > (benchmark + 0.05): # 5% above category
            return f"Escalate: >5% above Cat Avg ({benchmark:.1%})"
            
        return "Monitor"

    @staticmethod
    def generate_methodology_markdown() -> str:
        """
        Returns LaTeX formatted methodology.
        """
        return r"""
        ### 🧮 Methodology & Math

        **1. Return Rate Calculation**
        $$
        \text{Return Rate} = \frac{\text{Units Returned}}{\text{Units Sold}}
        $$

        **2. Relative Threshold Logic**
        Based on SOP QMS-SOP-001 (5), a case is triggered if:
        $$
        \text{Return Rate} > (\text{Category Average} + 0.05)
        $$
        *Note: The system defaults to the higher threshold when conflicts exist.*

        **3. Weighted Risk Score**
        The risk score ($S$) is calculated as:
        $$
        S = (0.3 \times \text{StatDev}) + (0.3 \times \text{Cost}) + (0.4 \times \text{Severity})
        $$
        Where *Severity* is AI-determined from complaint text.

        **4. ANOVA (Analysis of Variance)**
        Used to determine if category differences are statistically significant.
        $$
        F = \frac{\text{Between-Group Variance}}{\text{Within-Group Variance}}
        $$
        If $p < 0.05$, we perform Post-Hoc testing to identify outlier categories.

        **5. Trend Analysis**
        Compare current 30-day rate to 6-month and 12-month rolling averages.
        $$
        \Delta_{6M} = \text{RR}_{30D} - \text{RR}_{6M}, \quad \Delta_{12M} = \text{RR}_{30D} - \text{RR}_{12M}
        $$

        **6. SPC Signal Detection (Shewhart)**
        $$
        Z = \frac{\text{RR}_{current} - \mu}{\sigma}
        $$
        Signal if $Z > 3$.

        **7. MANOVA (Multivariate ANOVA)**
        Used when multiple metrics (e.g., Return Rate + Landed Cost) are available to test category effects.
        """
