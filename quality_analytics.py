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
    def validate_upload(df: pd.DataFrame, required_cols: List[str]) -> Dict[str, Any]:
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
        
        return report

    @staticmethod
    def calculate_risk_score(row: pd.Series, category_avg: float) -> float:
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
        if rr > (category_avg * 1.5):
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
        if cost >= SOP_THRESHOLDS['Critical_Landed_Cost']:
            # For high cost, we are stricter
            if row.get('Return_Rate', 0) > 0.05: # Arbitrary strict threshold for high value
                return "Escalate: High Value Defect"
                
        # 3. Return Rate Thresholds (Logic from SOPs)
        # Use higher threshold if conflict exists
        rr = row.get('Return_Rate', 0)
        cat = row.get('Category', 'All Others')
        benchmark = sop_benchmarks.get(cat, SOP_THRESHOLDS['All Others'])
        
        # Hard Cap
        if rr >= SOP_THRESHOLDS['Critical_Return_Rate_Cap']:
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
        """
