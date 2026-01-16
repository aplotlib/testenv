"""
Enhanced VoC Analysis Integration - v2.0
Supports multiple file formats and multi-period comparison
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import re


# Amazon Return Rate Thresholds (2026 Policy)
AMAZON_RETURN_RATE_THRESHOLDS = {
    "Baby Products": 0.088,
    "Beauty & Personal Care": 0.048,
    "Clothing & Accessories": 0.088,
    "Electronics": 0.048,
    "Grocery & Gourmet Food": 0.029,
    "Health & Household": 0.048,
    "Home & Kitchen": 0.048,
    "Kitchen & Dining": 0.048,
    "Office Products": 0.048,
    "Pet Supplies": 0.048,
    "Sports & Outdoors": 0.048,
    "Tools & Home Improvement": 0.048,
    "Toys & Games": 0.088,
    "Video Games": 0.128,
    "Everything Else": 0.048
}


@dataclass
class PeriodData:
    """Sales and quality metrics for a specific time period"""
    period_name: str
    date_start: Optional[datetime]
    date_end: Optional[datetime]
    total_orders: int
    ncx_orders: int = 0
    ncx_rate: float = 0.0
    return_rate: float = 0.0
    star_rating: Optional[float] = None
    cx_health: str = "Unknown"
    return_badge_displayed: bool = False
    top_ncx_reason: Optional[str] = None

    # Additional metrics for totals format
    category_counts: Dict[str, int] = field(default_factory=dict)
    category_percentages: Dict[str, float] = field(default_factory=dict)


@dataclass
class MultiPeriodTrendAnalysis:
    """Trend analysis across multiple periods (3+)"""
    product_name: str
    sku: str
    asin: Optional[str]

    # All periods (chronological, oldest to newest)
    periods: List[PeriodData]

    # Trend metrics (comparing most recent to oldest)
    overall_sales_change_pct: Optional[float]
    overall_sales_trend: str  # "Increasing", "Decreasing", "Stable", "Volatile"
    overall_return_rate_change: Optional[float]
    overall_return_trend: str  # "Improving", "Worsening", "Stable", "Volatile"

    # Period-over-period changes
    period_changes: List[Dict[str, Any]]  # List of {period, sales_change, return_change}

    # Amazon fee thresholds
    amazon_category: str
    amazon_threshold: float
    above_threshold: bool
    periods_above_threshold: int  # How many periods exceeded threshold
    fee_risk_units: int
    estimated_fee_impact: Optional[float]

    # Risk flags
    risk_flags: List[str]
    action_required: bool
    priority_level: str  # "Critical", "High", "Medium", "Low"


class EnhancedVoCAnalysisService:
    """Enhanced VoC Analysis with multi-format and multi-period support"""

    @staticmethod
    def detect_file_format(file_path_or_df) -> str:
        """
        Auto-detect VoC file format

        Returns:
            "voc_workbook" - Multi-sheet VoC Analysis.xlsx with dated tabs
            "totals_csv" - Analysis Totals Sheet CSV with horizontal periods
            "standard_csv" - Standard return data CSV
            "unknown" - Unable to determine format
        """
        try:
            # Handle file path or DataFrame
            if isinstance(file_path_or_df, str):
                if file_path_or_df.endswith('.xlsx') or file_path_or_df.endswith('.xls'):
                    import openpyxl
                    wb = openpyxl.load_workbook(file_path_or_df, read_only=True)
                    # Check for dated sheets
                    months = ['January', 'February', 'March', 'April', 'May', 'June',
                             'July', 'August', 'September', 'October', 'November', 'December']
                    has_dated_sheets = any(
                        any(month in sheet for month in months)
                        for sheet in wb.sheetnames
                    )
                    if has_dated_sheets:
                        return "voc_workbook"

                # Try reading as CSV
                df = pd.read_csv(file_path_or_df, nrows=20)
            else:
                df = file_path_or_df.head(20)

            # Check for totals CSV format (has Category: column with return categories)
            if 'Category:' in df.columns or (len(df.columns) > 5 and 'SKU' in df.columns):
                # Check if there are many unnamed columns (horizontal periods)
                unnamed_cols = [col for col in df.columns if 'Unnamed' in str(col) or col == '']
                if len(unnamed_cols) > 10:
                    return "totals_csv"

            # Check for standard VoC columns
            voc_columns = {'SKU', 'Product name', 'Return rate', 'NCX rate', 'Total orders'}
            if len(voc_columns.intersection(set(df.columns))) >= 3:
                return "standard_csv"

            return "unknown"

        except Exception as e:
            print(f"Format detection error: {e}")
            return "unknown"

    @staticmethod
    def parse_totals_csv(file_path: str) -> Dict[str, MultiPeriodTrendAnalysis]:
        """
        Parse Analysis Totals Sheet CSV with return category breakdown

        Format:
        - Row 2: Product info (ID, name, SKU, start date, end date)
        - Row 5+: Return categories with counts, percentages, and individual comments

        Returns:
            Dictionary mapping SKU to MultiPeriodTrendAnalysis
        """
        df = pd.read_csv(file_path)

        # Extract product info from row 1 (index 0)
        product_row = df.iloc[0]
        product_id = product_row.iloc[0] if not pd.isna(product_row.iloc[0]) else "Unknown"
        product_name = product_row.iloc[1] if not pd.isna(product_row.iloc[1]) else "Unknown Product"
        sku = product_row.iloc[2] if not pd.isna(product_row.iloc[2]) else "UNKNOWN"
        start_date_str = product_row.iloc[3] if not pd.isna(product_row.iloc[3]) else None
        end_date_str = product_row.iloc[4] if not pd.isna(product_row.iloc[4]) else None

        # Parse dates
        start_date = EnhancedVoCAnalysisService._parse_date(start_date_str) if start_date_str else None
        end_date = EnhancedVoCAnalysisService._parse_date(end_date_str) if end_date_str else None

        # Extract category data (starting from row 5, index 4)
        category_data = df.iloc[4:].copy()
        category_data = category_data.dropna(how='all')

        # Parse return categories
        # Row structure: empty, "Category:", count, percentage, individual comments...
        category_counts = {}
        category_percentages = {}
        total_returns = 0

        for _, row in category_data.iterrows():
            # Category name is in column index 1
            category_name = row.iloc[1] if len(row) > 1 and not pd.isna(row.iloc[1]) else None

            if category_name and category_name != "Category:":
                # Count is in column index 2
                try:
                    count = int(row.iloc[2]) if len(row) > 2 and not pd.isna(row.iloc[2]) else 0
                    percentage_str = row.iloc[3] if len(row) > 3 and not pd.isna(row.iloc[3]) else "0%"

                    # Parse percentage
                    percentage = float(percentage_str.replace('%', '')) / 100 if '%' in str(percentage_str) else 0.0

                    category_counts[category_name] = count
                    category_percentages[category_name] = percentage
                    total_returns += count
                except (ValueError, TypeError):
                    pass

        # Calculate defect rate (quality issues vs. customer error)
        defect_categories = [
            "Product Defects/Quality",
            "Performance/Effectiveness",
            "Design/Material Issues",
            "Stability/Positioning Issues",
            "Comfort Issues",
            "Size/Fit Issues"
        ]

        non_defect_categories = [
            "Customer Error/Changed Mind",
            "Shipping/Fulfillment Issues",
            "Wrong Product/Misunderstanding",
            "Medical/Health Concerns"
        ]

        defect_returns = sum(category_counts.get(cat, 0) for cat in defect_categories)
        customer_error_returns = sum(category_counts.get(cat, 0) for cat in non_defect_categories)

        # Calculate return rate (defect returns / total returns)
        return_rate = defect_returns / total_returns if total_returns > 0 else 0.0

        # Determine top issue categories
        sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        top_issue = sorted_categories[0][0] if sorted_categories else "Unknown"
        top_ncx_reason = f"{top_issue} ({sorted_categories[0][1]} returns)" if sorted_categories else None

        # Create single period (this CSV represents one time period)
        period_name = "Analysis Period"
        if start_date and end_date:
            period_name = f"{start_date.strftime('%b %Y')} - {end_date.strftime('%b %Y')}"

        period = PeriodData(
            period_name=period_name,
            date_start=start_date,
            date_end=end_date,
            total_orders=total_returns,  # Using total returns as proxy for orders
            ncx_orders=defect_returns,
            ncx_rate=return_rate,
            return_rate=return_rate,
            star_rating=None,
            cx_health="Unknown",
            return_badge_displayed=False,
            top_ncx_reason=top_ncx_reason,
            category_counts=category_counts,
            category_percentages=category_percentages
        )

        periods = [period]

        # Determine Amazon category
        amazon_category = "Health & Household"
        amazon_threshold = AMAZON_RETURN_RATE_THRESHOLDS[amazon_category]

        # Check threshold
        above_threshold = return_rate > amazon_threshold

        # Calculate fee risk
        if above_threshold:
            fee_risk_units = int(total_returns * (return_rate - amazon_threshold))
            estimated_fee_impact = max(0, fee_risk_units * 0.50)
        else:
            fee_risk_units = 0
            estimated_fee_impact = 0.0

        # Generate risk flags
        risk_flags = []

        if above_threshold:
            risk_flags.append("🚨 Above Amazon Return Rate Threshold - Fee Risk")

        # High defect rate
        if return_rate > 0.15:
            risk_flags.append(f"⚠️ High Quality Issue Rate ({return_rate*100:.1f}%)")

        # Dominant category issues
        if sorted_categories:
            top_cat, top_count = sorted_categories[0]
            if top_count > total_returns * 0.40:  # >40% in one category
                risk_flags.append(f"🔍 Dominant Issue: {top_cat} ({top_count}/{total_returns})")

        # Size/fit issues (design problem)
        size_fit = category_counts.get("Size/Fit Issues", 0)
        if size_fit > total_returns * 0.30:
            risk_flags.append(f"📏 High Size/Fit Issues - Design Review Needed")

        # Product defects
        defect_count = category_counts.get("Product Defects/Quality", 0)
        if defect_count > total_returns * 0.15:
            risk_flags.append(f"🔧 Quality Defects Detected - Manufacturing Review")

        # Determine priority
        priority = "Low"
        if len(risk_flags) >= 3 or above_threshold:
            priority = "High"
        elif len(risk_flags) >= 1:
            priority = "Medium"

        analysis = MultiPeriodTrendAnalysis(
            product_name=product_name,
            sku=sku,
            asin=None,
            periods=periods,
            overall_sales_change_pct=None,  # Single period
            overall_sales_trend="Single Period",
            overall_return_rate_change=None,
            overall_return_trend="Baseline",
            period_changes=[],
            amazon_category=amazon_category,
            amazon_threshold=amazon_threshold,
            above_threshold=above_threshold,
            periods_above_threshold=1 if above_threshold else 0,
            fee_risk_units=fee_risk_units,
            estimated_fee_impact=estimated_fee_impact,
            risk_flags=risk_flags,
            action_required=len(risk_flags) > 0,
            priority_level=priority
        )

        return {sku: analysis}

    @staticmethod
    def generate_root_cause_recommendations(analysis: MultiPeriodTrendAnalysis) -> List[Dict[str, str]]:
        """
        Generate actionable recommendations based on return category analysis

        Returns:
            List of recommendations with priority, action, and rationale
        """
        recommendations = []

        if not analysis.periods or not analysis.periods[0].category_counts:
            return recommendations

        latest = analysis.periods[0]
        total_returns = latest.total_orders

        # Analyze each category
        for category, count in sorted(latest.category_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = count / total_returns if total_returns > 0 else 0

            # Size/Fit Issues - Design problem
            if "Size/Fit" in category and percentage > 0.25:
                recommendations.append({
                    "priority": "High",
                    "category": category,
                    "issue": f"{percentage*100:.1f}% of returns are size/fit related",
                    "action": "Review sizing chart accuracy and consider adding detailed measurements to product page",
                    "rationale": "Customers are receiving products that don't fit expectations"
                })

            # Product Defects - Quality control
            if "Defect" in category or "Quality" in category:
                if percentage > 0.15:
                    recommendations.append({
                        "priority": "Critical",
                        "category": category,
                        "issue": f"{percentage*100:.1f}% of returns are quality defects",
                        "action": "Conduct manufacturing quality audit and implement additional QC checkpoints",
                        "rationale": "High defect rate indicates quality control issues in production"
                    })

            # Performance Issues
            if "Performance" in category or "Effectiveness" in category:
                if percentage > 0.20:
                    recommendations.append({
                        "priority": "High",
                        "category": category,
                        "issue": f"{percentage*100:.1f}% of returns cite performance issues",
                        "action": "Review product specifications and marketing claims for accuracy",
                        "rationale": "Product may not meet customer expectations set by marketing"
                    })

            # Design/Material Issues
            if "Design" in category or "Material" in category:
                if percentage > 0.20:
                    recommendations.append({
                        "priority": "Medium",
                        "category": category,
                        "issue": f"{percentage*100:.1f}% of returns are design-related",
                        "action": "Consider design revision or enhanced product photos showing design details",
                        "rationale": "Customers are dissatisfied with design/material choices"
                    })

            # Comfort Issues
            if "Comfort" in category and percentage > 0.15:
                recommendations.append({
                    "priority": "Medium",
                    "category": category,
                    "issue": f"{percentage*100:.1f}% of returns cite comfort issues",
                    "action": "Add comfort-related information to product page and consider ergonomic improvements",
                    "rationale": "Comfort is subjective but can be addressed through better customer education"
                })

            # Customer Error - Educational opportunity
            if "Customer Error" in category or "Changed Mind" in category:
                if percentage > 0.30:
                    recommendations.append({
                        "priority": "Low",
                        "category": category,
                        "issue": f"{percentage*100:.1f}% of returns are customer error/changed mind",
                        "action": "Enhance product descriptions, add FAQ section, and consider adding comparison charts",
                        "rationale": "Better product education can reduce buyer's remorse and wrong purchases"
                    })

            # Shipping Issues
            if "Shipping" in category or "Fulfillment" in category:
                if percentage > 0.10:
                    recommendations.append({
                        "priority": "Medium",
                        "category": category,
                        "issue": f"{percentage*100:.1f}% of returns are shipping-related",
                        "action": "Review packaging methods and shipping carrier performance",
                        "rationale": "Shipping issues damage customer experience and increase costs"
                    })

        return recommendations

    @staticmethod
    def generate_comparison_report(analyses: List[MultiPeriodTrendAnalysis]) -> pd.DataFrame:
        """
        Generate side-by-side comparison of multiple products

        Args:
            analyses: List of product trend analyses to compare

        Returns:
            DataFrame with comparison metrics
        """
        comparison_data = []

        for analysis in analyses:
            latest = analysis.periods[0] if analysis.periods else None
            if not latest:
                continue

            comparison_data.append({
                "Product": analysis.product_name,
                "SKU": analysis.sku,
                "Total Returns": latest.total_orders,
                "Defect Rate": f"{latest.return_rate*100:.1f}%",
                "Amazon Threshold": f"{analysis.amazon_threshold*100:.1f}%",
                "Above Threshold": "Yes" if analysis.above_threshold else "No",
                "Priority": analysis.priority_level,
                "Risk Flags": len(analysis.risk_flags),
                "Top Issue": latest.top_ncx_reason or "N/A",
                "Fee Risk ($)": f"${analysis.estimated_fee_impact:.2f}" if analysis.estimated_fee_impact else "$0.00"
            })

        df = pd.DataFrame(comparison_data)

        # Sort by priority and defect rate
        priority_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
        df["_priority_sort"] = df["Priority"].map(priority_order)
        df = df.sort_values(["_priority_sort", "Risk Flags"], ascending=[True, False])
        df = df.drop(columns=["_priority_sort"])

        return df

    @staticmethod
    def detect_emerging_issues(current_analysis: MultiPeriodTrendAnalysis,
                              historical_analyses: List[MultiPeriodTrendAnalysis]) -> List[str]:
        """
        Detect emerging quality issues by comparing current analysis to historical data

        Returns:
            List of alert messages for emerging issues
        """
        alerts = []

        if not current_analysis.periods or not historical_analyses:
            return alerts

        current = current_analysis.periods[0]

        # Compare to historical average
        historical_return_rates = []
        historical_defect_counts = []

        for hist_analysis in historical_analyses:
            if hist_analysis.periods:
                hist_period = hist_analysis.periods[0]
                historical_return_rates.append(hist_period.return_rate)
                historical_defect_counts.append(hist_period.ncx_orders)

        if historical_return_rates:
            avg_historical_rate = np.mean(historical_return_rates)
            std_historical_rate = np.std(historical_return_rates)

            # Spike detection
            if current.return_rate > avg_historical_rate + 2 * std_historical_rate:
                alerts.append(f"🚨 SPIKE DETECTED: Return rate ({current.return_rate*100:.1f}%) is 2+ standard deviations above historical average ({avg_historical_rate*100:.1f}%)")

            # Trend acceleration
            if len(historical_return_rates) >= 3:
                recent_trend = np.mean(historical_return_rates[-3:])
                older_trend = np.mean(historical_return_rates[:-3]) if len(historical_return_rates) > 3 else recent_trend

                if recent_trend > older_trend * 1.5:
                    alerts.append(f"📈 ACCELERATING TREND: Recent return rate increasing faster than historical trend")

        # Category-specific alerts
        if current.category_counts:
            # Check for sudden emergence of new categories
            for category, count in current.category_counts.items():
                if count > current.total_orders * 0.15:  # >15% in this category
                    # Check if this category was minimal historically
                    historical_category_avg = 0
                    for hist_analysis in historical_analyses:
                        if hist_analysis.periods and hist_analysis.periods[0].category_counts:
                            hist_count = hist_analysis.periods[0].category_counts.get(category, 0)
                            hist_total = hist_analysis.periods[0].total_orders
                            if hist_total > 0:
                                historical_category_avg += hist_count / hist_total

                    if len(historical_analyses) > 0:
                        historical_category_avg /= len(historical_analyses)

                        if count / current.total_orders > historical_category_avg * 3:
                            alerts.append(f"⚠️ EMERGING ISSUE: '{category}' is 3x higher than historical average")

        return alerts

    @staticmethod
    def _parse_date(date_str: str) -> Optional[datetime]:
        """Parse date string with emoji support"""
        if not date_str:
            return None

        # Remove emoji
        date_str = date_str.replace('📅', '').strip()

        try:
            return datetime.strptime(date_str, '%m/%d/%Y')
        except:
            try:
                return datetime.strptime(date_str, '%m/%d/%y')
            except:
                return None

    @staticmethod
    def _categorize_multi_period_trend(values: List[float]) -> str:
        """Categorize trend across multiple periods"""
        if len(values) < 2:
            return "Insufficient Data"

        # Calculate linear regression slope
        x = list(range(len(values)))
        slope = np.polyfit(x, values, 1)[0]

        # Calculate coefficient of variation to detect volatility
        cv = np.std(values) / np.mean(values) if np.mean(values) > 0 else 0

        if cv > 0.3:  # High variability
            return "Volatile"
        elif slope > 10:
            return "Increasing"
        elif slope < -10:
            return "Decreasing"
        else:
            return "Stable"

    @staticmethod
    def _categorize_return_trend_multi(values: List[float]) -> str:
        """Categorize return rate trend across multiple periods"""
        if len(values) < 2:
            return "Baseline"

        # Calculate slope
        x = list(range(len(values)))
        slope = np.polyfit(x, values, 1)[0]

        # Calculate volatility
        cv = np.std(values) / np.mean(values) if np.mean(values) > 0 else 0

        if cv > 0.4:
            return "Volatile"
        elif slope < -0.02:  # Improving
            return "Improving"
        elif slope > 0.02:  # Worsening
            return "Worsening"
        else:
            return "Stable"

    @staticmethod
    def _generate_multi_period_risk_flags(
        periods: List[PeriodData],
        above_threshold: bool,
        sales_change: Optional[float],
        return_change: Optional[float],
        periods_above_threshold: int
    ) -> List[str]:
        """Generate risk flags for multi-period analysis"""
        flags = []

        if not periods:
            return flags

        latest = periods[-1]

        # Amazon threshold
        if above_threshold:
            if periods_above_threshold == len(periods):
                flags.append("🚨 ALWAYS Above Amazon Threshold - Chronic Fee Risk")
            elif periods_above_threshold >= len(periods) // 2:
                flags.append("🚨 Frequently Above Amazon Threshold - High Fee Risk")
            else:
                flags.append("⚠️ Above Amazon Threshold - Fee Risk")

        # Badge
        if latest.return_badge_displayed:
            flags.append("⚠️ Amazon Return Badge Displayed")

        # Sales declining with high returns
        if sales_change is not None and sales_change < -15 and latest.return_rate > 0.08:
            flags.append("📉 Sales Declining + High Returns - Critical")

        # Return rate worsening across periods
        if return_change is not None and return_change > 0.05:
            flags.append(f"📈 Returns Increased {return_change*100:.1f}% - Worsening Quality")

        # Volatility check
        if len(periods) >= 3:
            sales_values = [p.total_orders for p in periods]
            return_values = [p.return_rate for p in periods]

            sales_cv = np.std(sales_values) / np.mean(sales_values) if np.mean(sales_values) > 0 else 0
            return_cv = np.std(return_values) / np.mean(return_values) if np.mean(return_values) > 0 else 0

            if sales_cv > 0.3:
                flags.append("⚡ Volatile Sales Pattern - Unstable Demand")
            if return_cv > 0.4:
                flags.append("⚡ Volatile Return Rate - Inconsistent Quality")

        # CX health
        if latest.cx_health == "At risk":
            flags.append("⚠️ Customer Experience At Risk")

        # Star rating
        if latest.star_rating is not None and latest.star_rating < 3.5:
            flags.append(f"⭐ Low Star Rating ({latest.star_rating:.1f}/5.0)")

        # High NCX
        if latest.ncx_rate > 0.10:
            flags.append(f"❌ High NCX Rate ({latest.ncx_rate*100:.1f}%)")

        return flags

    @staticmethod
    def _calculate_priority(
        risk_flag_count: int,
        above_threshold: bool,
        periods_above_threshold: int,
        total_periods: int
    ) -> str:
        """Calculate priority level based on risk factors"""
        if risk_flag_count >= 4 or (above_threshold and periods_above_threshold == total_periods):
            return "Critical"
        elif risk_flag_count >= 2 or (above_threshold and periods_above_threshold >= total_periods // 2):
            return "High"
        elif risk_flag_count >= 1:
            return "Medium"
        else:
            return "Low"

    @staticmethod
    def convert_to_screening_dataframe(
        trend_analyses: Dict[str, MultiPeriodTrendAnalysis]
    ) -> pd.DataFrame:
        """Convert multi-period analyses to screening DataFrame"""
        rows = []

        for sku, analysis in trend_analyses.items():
            if not analysis.periods:
                continue

            latest = analysis.periods[-1]

            # Calculate returned units
            returned_units = int(latest.total_orders * latest.return_rate) if latest.total_orders > 0 else 0

            # Build period summary
            period_summary = " → ".join([
                f"{p.period_name}: {p.total_orders} sold"
                for p in analysis.periods[-3:]  # Last 3 periods
            ])

            row = {
                'SKU': sku,
                'Name': analysis.product_name,
                'ASIN': analysis.asin or '',
                'Category': analysis.amazon_category,
                'Sold': latest.total_orders,
                'Returned': returned_units,
                'Return_Rate': latest.return_rate,
                'Landed Cost': 0.0,

                # Multi-period metrics
                'Periods_Analyzed': len(analysis.periods),
                'Period_Summary': period_summary,
                'Overall_Sales_Change_Pct': analysis.overall_sales_change_pct,
                'Overall_Sales_Trend': analysis.overall_sales_trend,
                'Overall_Return_Change': analysis.overall_return_rate_change,
                'Overall_Return_Trend': analysis.overall_return_trend,

                # Amazon metrics
                'Amazon_Threshold': analysis.amazon_threshold,
                'Above_Threshold': analysis.above_threshold,
                'Periods_Above_Threshold': analysis.periods_above_threshold,
                'Fee_Risk_Units': analysis.fee_risk_units,
                'Estimated_Fee_Impact': analysis.estimated_fee_impact or 0.0,

                # Quality metrics
                'CX_Health': latest.cx_health,
                'Star_Rating': latest.star_rating or 0.0,
                'NCX_Rate': latest.ncx_rate,
                'NCX_Orders': latest.ncx_orders,
                'Top_NCX_Reason': latest.top_ncx_reason or '',
                'Return_Badge_Displayed': 'Yes' if latest.return_badge_displayed else 'No',

                # Risk assessment
                'Risk_Flags': ' | '.join(analysis.risk_flags),
                'Priority_Level': analysis.priority_level,
                'Action_Required': 'Yes' if analysis.action_required else 'No',
                'Risk_Flag_Count': len(analysis.risk_flags)
            }

            rows.append(row)

        df = pd.DataFrame(rows)

        # Sort by priority and risk
        priority_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
        df['Priority_Rank'] = df['Priority_Level'].map(priority_order)
        df = df.sort_values(['Priority_Rank', 'Risk_Flag_Count', 'Sold'], ascending=[True, False, False])
        df = df.drop('Priority_Rank', axis=1)

        return df
