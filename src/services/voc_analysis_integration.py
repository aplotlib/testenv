"""
VoC Analysis Integration Module
Provides period-over-period sales trend analysis and Amazon return rate fee monitoring
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


# Amazon Return Rate Thresholds (2026 Policy)
# Source: https://litcommerce.com/blog/amazon-return-policy-change/
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
    period_name: str  # e.g., "January 2026" or "L30D"
    date_start: Optional[datetime]
    date_end: Optional[datetime]
    total_orders: int
    ncx_orders: int
    ncx_rate: float
    return_rate: float
    star_rating: Optional[float]
    cx_health: str  # "Excellent", "Good", "At risk", etc.
    return_badge_displayed: bool
    top_ncx_reason: Optional[str]


@dataclass
class ProductTrendAnalysis:
    """Trend analysis for a product across multiple periods"""
    product_name: str
    sku: str
    asin: Optional[str]
    current_period: PeriodData
    previous_period: Optional[PeriodData]

    # Trend metrics
    sales_change_pct: Optional[float]
    sales_trend: str  # "Increasing", "Decreasing", "Stable", "New"
    return_rate_change: Optional[float]
    return_rate_trend: str  # "Improving", "Worsening", "Stable"

    # Amazon fee thresholds
    amazon_category: str
    amazon_threshold: float
    above_threshold: bool
    fee_risk_units: int  # Units subject to return processing fee
    estimated_fee_impact: Optional[float]

    # Risk flags
    risk_flags: List[str]
    action_required: bool


class VoCAnalysisService:
    """Service for analyzing VoC data with period-over-period comparisons"""

    @staticmethod
    def parse_voc_sheet(df: pd.DataFrame, sheet_name: str) -> Dict[str, PeriodData]:
        """
        Parse a VoC Analysis sheet into structured PeriodData objects

        Args:
            df: DataFrame from VoC Analysis sheet
            sheet_name: Sheet name (e.g., "January_2026_01162026")

        Returns:
            Dictionary mapping SKU to PeriodData
        """
        period_data_map = {}

        # Extract period name from sheet name
        period_name = sheet_name.split('_')[0] + ' ' + sheet_name.split('_')[1] if '_' in sheet_name else sheet_name

        for _, row in df.iterrows():
            sku = str(row.get('SKU', '')).strip()
            if not sku or pd.isna(sku):
                continue

            # Parse return rate
            return_rate_raw = row.get('Return rate', 0)
            if pd.isna(return_rate_raw) or return_rate_raw == 'Not available':
                return_rate = 0.0
            else:
                try:
                    return_rate = float(return_rate_raw)
                except (ValueError, TypeError):
                    return_rate = 0.0

            # Parse NCX rate
            ncx_rate_raw = row.get('NCX rate', 0)
            if pd.isna(ncx_rate_raw):
                ncx_rate = 0.0
            else:
                try:
                    ncx_rate = float(ncx_rate_raw)
                except (ValueError, TypeError):
                    ncx_rate = 0.0

            # Parse star rating
            star_rating_raw = row.get('Star rating', None)
            if pd.isna(star_rating_raw):
                star_rating = None
            else:
                try:
                    star_rating = float(star_rating_raw)
                except (ValueError, TypeError):
                    star_rating = None

            # Parse return badge
            badge_raw = row.get('Return Badge Displayed', '--')
            return_badge = badge_raw not in ['--', None, np.nan, '']

            period_data = PeriodData(
                period_name=period_name,
                date_start=None,  # Could extract from sheet name if needed
                date_end=None,
                total_orders=int(row.get('Total orders', 0)) if not pd.isna(row.get('Total orders', 0)) else 0,
                ncx_orders=int(row.get('NCX orders', 0)) if not pd.isna(row.get('NCX orders', 0)) else 0,
                ncx_rate=ncx_rate,
                return_rate=return_rate,
                star_rating=star_rating,
                cx_health=str(row.get('CX Health', 'Unknown')),
                return_badge_displayed=return_badge,
                top_ncx_reason=str(row.get('Top NCX reason', '')) if not pd.isna(row.get('Top NCX reason', '')) else None
            )

            period_data_map[sku] = period_data

        return period_data_map

    @staticmethod
    def parse_voc_workbook(file_path: str, current_sheet_name: str,
                          previous_sheet_name: Optional[str] = None) -> Dict[str, ProductTrendAnalysis]:
        """
        Parse VoC Analysis workbook with period-over-period comparison

        Args:
            file_path: Path to VoC Analysis.xlsx
            current_sheet_name: Name of current period sheet
            previous_sheet_name: Name of previous period sheet (optional)

        Returns:
            Dictionary mapping SKU to ProductTrendAnalysis
        """
        # Read current period
        df_current = pd.read_excel(file_path, sheet_name=current_sheet_name)
        current_data = VoCAnalysisService.parse_voc_sheet(df_current, current_sheet_name)

        # Read previous period if specified
        previous_data = {}
        if previous_sheet_name:
            try:
                df_previous = pd.read_excel(file_path, sheet_name=previous_sheet_name)
                previous_data = VoCAnalysisService.parse_voc_sheet(df_previous, previous_sheet_name)
            except Exception:
                pass  # Previous period not available

        # Build trend analysis
        trend_analyses = {}

        for sku, current in current_data.items():
            previous = previous_data.get(sku)

            # Calculate trends
            if previous:
                sales_change_pct = VoCAnalysisService._calculate_change_pct(
                    previous.total_orders, current.total_orders
                )
                sales_trend = VoCAnalysisService._categorize_sales_trend(sales_change_pct)

                return_rate_change = current.return_rate - previous.return_rate
                return_rate_trend = VoCAnalysisService._categorize_return_trend(return_rate_change)
            else:
                sales_change_pct = None
                sales_trend = "New"
                return_rate_change = None
                return_rate_trend = "Baseline"

            # Determine Amazon category (simplified - would need product metadata)
            amazon_category = "Everything Else"  # Default
            amazon_threshold = AMAZON_RETURN_RATE_THRESHOLDS[amazon_category]

            # Check if above threshold
            above_threshold = current.return_rate > amazon_threshold

            # Calculate fee risk units
            if above_threshold and current.total_orders > 0:
                # Units above threshold over 3-month period
                fee_risk_units = int(current.total_orders * (current.return_rate - amazon_threshold))
            else:
                fee_risk_units = 0

            # Estimate fee impact (simplified - actual fees vary by size/weight)
            estimated_fee_impact = fee_risk_units * 0.50 if fee_risk_units > 0 else None  # $0.50 avg estimate

            # Generate risk flags
            risk_flags = VoCAnalysisService._generate_risk_flags(
                current, previous, above_threshold, sales_change_pct, return_rate_change
            )

            action_required = len(risk_flags) > 0

            # Get product name from current data
            try:
                product_name = df_current[df_current['SKU'] == sku]['Product name'].iloc[0]
            except (KeyError, IndexError):
                product_name = "Unknown Product"

            # Get ASIN
            try:
                asin = df_current[df_current['SKU'] == sku]['ASIN'].iloc[0]
            except (KeyError, IndexError):
                asin = None

            trend_analysis = ProductTrendAnalysis(
                product_name=product_name,
                sku=sku,
                asin=asin,
                current_period=current,
                previous_period=previous,
                sales_change_pct=sales_change_pct,
                sales_trend=sales_trend,
                return_rate_change=return_rate_change,
                return_rate_trend=return_rate_trend,
                amazon_category=amazon_category,
                amazon_threshold=amazon_threshold,
                above_threshold=above_threshold,
                fee_risk_units=fee_risk_units,
                estimated_fee_impact=estimated_fee_impact,
                risk_flags=risk_flags,
                action_required=action_required
            )

            trend_analyses[sku] = trend_analysis

        return trend_analyses

    @staticmethod
    def _calculate_change_pct(previous_value: float, current_value: float) -> Optional[float]:
        """Calculate percentage change between periods"""
        if previous_value == 0:
            return None
        return ((current_value - previous_value) / previous_value) * 100

    @staticmethod
    def _categorize_sales_trend(change_pct: Optional[float]) -> str:
        """Categorize sales trend based on percentage change"""
        if change_pct is None:
            return "Unknown"
        elif change_pct > 10:
            return "Increasing"
        elif change_pct < -10:
            return "Decreasing"
        else:
            return "Stable"

    @staticmethod
    def _categorize_return_trend(change: Optional[float]) -> str:
        """Categorize return rate trend"""
        if change is None:
            return "Baseline"
        elif change < -0.02:  # 2% improvement
            return "Improving"
        elif change > 0.02:  # 2% worsening
            return "Worsening"
        else:
            return "Stable"

    @staticmethod
    def _generate_risk_flags(current: PeriodData, previous: Optional[PeriodData],
                            above_threshold: bool, sales_change_pct: Optional[float],
                            return_rate_change: Optional[float]) -> List[str]:
        """Generate risk flags based on various conditions"""
        flags = []

        # Amazon return rate threshold
        if above_threshold:
            flags.append("🚨 Above Amazon Return Rate Threshold - Fee Risk")

        # Return badge displayed
        if current.return_badge_displayed:
            flags.append("⚠️ Amazon Return Badge Displayed - Visibility Impact")

        # Declining sales + high returns
        if sales_change_pct is not None and sales_change_pct < -15 and current.return_rate > 0.08:
            flags.append("📉 Sales Declining with High Return Rate")

        # Worsening return rate
        if return_rate_change is not None and return_rate_change > 0.05:
            flags.append("📈 Return Rate Increased >5% from Previous Period")

        # Poor CX health
        if current.cx_health == "At risk":
            flags.append("⚠️ CX Health At Risk")

        # Low star rating
        if current.star_rating is not None and current.star_rating < 3.5:
            flags.append("⭐ Low Star Rating (<3.5)")

        # High NCX rate
        if current.ncx_rate > 0.10:
            flags.append("❌ High Negative Customer Experience Rate (>10%)")

        return flags

    @staticmethod
    def convert_to_screening_dataframe(trend_analyses: Dict[str, ProductTrendAnalysis]) -> pd.DataFrame:
        """
        Convert ProductTrendAnalysis objects to DataFrame for Quality Screening tool

        Returns DataFrame with columns needed for screening:
        - SKU, Name, Category, Sold, Returned, Return_Rate, Landed Cost (if available)
        - Plus VoC-specific fields: Sales_Trend, Return_Trend, Risk_Flags, Fee_Risk, etc.
        """
        rows = []

        for sku, analysis in trend_analyses.items():
            current = analysis.current_period

            # Calculate returned units
            returned_units = int(current.total_orders * current.return_rate) if current.total_orders > 0 else 0

            row = {
                'SKU': sku,
                'Name': analysis.product_name,
                'ASIN': analysis.asin or '',
                'Category': analysis.amazon_category,
                'Sold': current.total_orders,
                'Returned': returned_units,
                'Return_Rate': current.return_rate,
                'Landed Cost': 0.0,  # Not available in VoC data

                # Period comparison
                'Current_Period': analysis.current_period.period_name,
                'Previous_Period': analysis.previous_period.period_name if analysis.previous_period else 'N/A',
                'Sales_Change_Pct': analysis.sales_change_pct,
                'Sales_Trend': analysis.sales_trend,
                'Return_Rate_Change': analysis.return_rate_change,
                'Return_Trend': analysis.return_rate_trend,

                # Amazon metrics
                'Amazon_Threshold': analysis.amazon_threshold,
                'Above_Threshold': analysis.above_threshold,
                'Fee_Risk_Units': analysis.fee_risk_units,
                'Estimated_Fee_Impact': analysis.estimated_fee_impact or 0.0,

                # Quality metrics
                'CX_Health': current.cx_health,
                'Star_Rating': current.star_rating or 0.0,
                'NCX_Rate': current.ncx_rate,
                'NCX_Orders': current.ncx_orders,
                'Top_NCX_Reason': current.top_ncx_reason or '',
                'Return_Badge_Displayed': 'Yes' if current.return_badge_displayed else 'No',

                # Risk assessment
                'Risk_Flags': ' | '.join(analysis.risk_flags),
                'Action_Required': 'Yes' if analysis.action_required else 'No',
                'Risk_Flag_Count': len(analysis.risk_flags)
            }

            rows.append(row)

        df = pd.DataFrame(rows)

        # Sort by risk and sales volume
        df = df.sort_values(['Risk_Flag_Count', 'Sold'], ascending=[False, False])

        return df

    @staticmethod
    def get_available_periods(file_path: str) -> List[Tuple[str, str]]:
        """
        Get list of available dated periods from VoC Analysis workbook

        Returns:
            List of tuples (sheet_name, display_name)
        """
        import openpyxl

        wb = openpyxl.load_workbook(file_path, read_only=True)
        periods = []

        for sheet_name in wb.sheetnames:
            # Look for dated sheets (e.g., "January_2026_01162026")
            if any(month in sheet_name for month in ['January', 'February', 'March', 'April',
                                                     'May', 'June', 'July', 'August',
                                                     'September', 'October', 'November', 'December']):
                # Create display name
                parts = sheet_name.split('_')
                if len(parts) >= 2:
                    display_name = f"{parts[0]} {parts[1]}"
                else:
                    display_name = sheet_name

                periods.append((sheet_name, display_name))

        return periods

    @staticmethod
    def generate_period_comparison_summary(trend_analyses: Dict[str, ProductTrendAnalysis]) -> Dict[str, Any]:
        """
        Generate summary statistics for period comparison

        Returns:
            Dictionary with summary metrics
        """
        total_products = len(trend_analyses)

        # Count trends
        increasing_sales = sum(1 for t in trend_analyses.values() if t.sales_trend == "Increasing")
        decreasing_sales = sum(1 for t in trend_analyses.values() if t.sales_trend == "Decreasing")
        stable_sales = sum(1 for t in trend_analyses.values() if t.sales_trend == "Stable")
        new_products = sum(1 for t in trend_analyses.values() if t.sales_trend == "New")

        improving_returns = sum(1 for t in trend_analyses.values() if t.return_rate_trend == "Improving")
        worsening_returns = sum(1 for t in trend_analyses.values() if t.return_rate_trend == "Worsening")

        # Amazon threshold violations
        above_threshold = sum(1 for t in trend_analyses.values() if t.above_threshold)
        total_fee_risk_units = sum(t.fee_risk_units for t in trend_analyses.values())
        total_estimated_fees = sum(t.estimated_fee_impact or 0 for t in trend_analyses.values())

        # Badge impact
        with_badge = sum(1 for t in trend_analyses.values() if t.current_period.return_badge_displayed)

        # Action required
        action_required = sum(1 for t in trend_analyses.values() if t.action_required)

        return {
            'total_products': total_products,
            'sales_trends': {
                'increasing': increasing_sales,
                'decreasing': decreasing_sales,
                'stable': stable_sales,
                'new': new_products
            },
            'return_trends': {
                'improving': improving_returns,
                'worsening': worsening_returns
            },
            'amazon_thresholds': {
                'above_threshold': above_threshold,
                'fee_risk_units': total_fee_risk_units,
                'estimated_fees': total_estimated_fees
            },
            'badges': {
                'with_badge': with_badge
            },
            'actions': {
                'action_required': action_required
            }
        }
