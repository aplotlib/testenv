"""
Quality Tracker Manager

Manages quality cases with manual entry, AI summaries, and dual exports:
- Leadership Export (31 columns with financials)
- Company Wide Export (25 columns, sanitized)

Based on actual tracker templates from Vive Health.
"""

import pandas as pd
from datetime import datetime, date
from typing import Dict, List, Any, Optional
import logging
import io

logger = logging.getLogger(__name__)


# Leadership-only column names (6 sensitive columns)
LEADERSHIP_ONLY_COLUMNS = [
    'Priority',
    'Total orders (t30)',
    'Flag Source 1',
    'Cost of Refunds (Annualized)',
    '12m Savings Captured (based on rr% reduction)',
    'Case Status'
]

# All 31 column names in exact order (Leadership version)
ALL_COLUMNS_LEADERSHIP = [
    'Priority',  # 1 - LEADERSHIP ONLY
    'Product name',  # 2
    'Main Sales Channel (by Volume)',  # 3
    'ASIN',  # 4
    'SKU',  # 5
    'Fulfilled by',  # 6
    'NCX rate',  # 7
    'NCX orders',  # 8
    'Total orders (t30)',  # 9 - LEADERSHIP ONLY
    'Star Rating Amazon',  # 10
    'Return rate Amazon',  # 11
    'Return Rate B2B',  # 12
    'Flag Source 1',  # 13 - LEADERSHIP ONLY
    'Return Badge Displayed Amazon',  # 14
    'Notification/Notes',  # 15
    'Top Issue(s)',  # 16
    'Cost of Refunds (Annualized)',  # 17 - LEADERSHIP ONLY (FINANCIAL)
    '12m Savings Captured (based on rr% reduction)',  # 18 - LEADERSHIP ONLY (FINANCIAL)
    'Action Taken',  # 19
    'Date Action Taken',  # 20
    'Listing Manager Notified?',  # 21
    'Product Dev Notified?',  # 22
    'Flag Source',  # 23
    'Follow Up Date',  # 24
    'Result 1 (rr%)',  # 25
    'Result Check Date 1',  # 26
    'Result 2 (rr%)',  # 27
    'Result 2 Date',  # 28
    'Top Issue(s) Change',  # 29
    'Top Issue(s) Change Date',  # 30
    'Case Status'  # 31 - LEADERSHIP ONLY
]

# 25 columns for Company Wide (excludes 6 leadership columns)
ALL_COLUMNS_COMPANY_WIDE = [col for col in ALL_COLUMNS_LEADERSHIP if col not in LEADERSHIP_ONLY_COLUMNS]


class QualityTrackerCase:
    """Represents a single quality tracker case"""

    def __init__(self):
        # Initialize all fields as empty/None
        self.priority: Optional[int] = None
        self.product_name: str = ""
        self.main_sales_channel: str = ""
        self.asin: str = ""
        self.sku: str = ""
        self.fulfilled_by: str = ""
        self.ncx_rate: Optional[float] = None
        self.ncx_orders: Optional[int] = None
        self.total_orders_t30: Optional[int] = None
        self.star_rating_amazon: Optional[float] = None
        self.return_rate_amazon: Optional[float] = None
        self.return_rate_b2b: Optional[float] = None
        self.flag_source_1: str = ""
        self.return_badge_displayed: str = ""
        self.notification_notes: str = ""
        self.top_issues: str = ""
        self.cost_of_refunds_annualized: Optional[float] = None
        self.savings_captured_12m: Optional[float] = None
        self.action_taken: str = ""
        self.date_action_taken: Optional[date] = None
        self.listing_manager_notified: str = ""
        self.product_dev_notified: str = ""
        self.flag_source: str = ""
        self.follow_up_date: Optional[date] = None
        self.result_1_rr: Optional[float] = None
        self.result_check_date_1: Optional[date] = None
        self.result_2_rr: Optional[float] = None
        self.result_2_date: Optional[date] = None
        self.top_issues_change: str = ""
        self.top_issues_change_date: Optional[date] = None
        self.case_status: str = ""

    def to_dict_leadership(self) -> Dict[str, Any]:
        """Convert to dict with all 31 columns (Leadership version)"""
        return {
            'Priority': self.priority,
            'Product name': self.product_name,
            'Main Sales Channel (by Volume)': self.main_sales_channel,
            'ASIN': self.asin,
            'SKU': self.sku,
            'Fulfilled by': self.fulfilled_by,
            'NCX rate': self.ncx_rate,
            'NCX orders': self.ncx_orders,
            'Total orders (t30)': self.total_orders_t30,
            'Star Rating Amazon': self.star_rating_amazon,
            'Return rate Amazon': self.return_rate_amazon,
            'Return Rate B2B': self.return_rate_b2b,
            'Flag Source 1': self.flag_source_1,
            'Return Badge Displayed Amazon': self.return_badge_displayed,
            'Notification/Notes': self.notification_notes,
            'Top Issue(s)': self.top_issues,
            'Cost of Refunds (Annualized)': self.cost_of_refunds_annualized,
            '12m Savings Captured (based on rr% reduction)': self.savings_captured_12m,
            'Action Taken': self.action_taken,
            'Date Action Taken': self.date_action_taken,
            'Listing Manager Notified?': self.listing_manager_notified,
            'Product Dev Notified?': self.product_dev_notified,
            'Flag Source': self.flag_source,
            'Follow Up Date': self.follow_up_date,
            'Result 1 (rr%)': self.result_1_rr,
            'Result Check Date 1': self.result_check_date_1,
            'Result 2 (rr%)': self.result_2_rr,
            'Result 2 Date': self.result_2_date,
            'Top Issue(s) Change': self.top_issues_change,
            'Top Issue(s) Change Date': self.top_issues_change_date,
            'Case Status': self.case_status
        }

    def to_dict_company_wide(self) -> Dict[str, Any]:
        """Convert to dict with 25 columns (Company Wide version - excludes 6 sensitive columns)"""
        full_dict = self.to_dict_leadership()
        # Remove leadership-only columns
        return {k: v for k, v in full_dict.items() if k not in LEADERSHIP_ONLY_COLUMNS}


class QualityTrackerManager:
    """Manages quality tracker cases with manual entry and dual exports"""

    def __init__(self, ai_analyzer=None):
        self.cases: List[QualityTrackerCase] = []
        self.ai_analyzer = ai_analyzer

    def add_case(self, case: QualityTrackerCase):
        """Add a case to the tracker"""
        self.cases.append(case)

    def generate_ai_summary(self, case: QualityTrackerCase) -> str:
        """Generate pithy AI summary of the case"""
        if not self.ai_analyzer:
            return "AI analyzer not available"

        prompt = f"""Summarize this quality case in 1-2 sentences (max 100 words):

Product: {case.product_name} ({case.sku})
Issue: {case.top_issues}
Return Rate Amazon: {case.return_rate_amazon * 100:.2f}% if case.return_rate_amazon else 'N/A'
Return Rate B2B: {case.return_rate_b2b * 100:.2f}% if case.return_rate_b2b else 'N/A'
Star Rating: {case.star_rating_amazon or 'N/A'}
Action Taken: {case.action_taken or 'None yet'}
Status: {case.case_status or 'Open'}
Notes: {case.notification_notes or 'None'}

Provide a pithy, executive-level summary focusing on the issue severity, business impact, and current status."""

        system_prompt = "You are a quality management expert. Provide concise, actionable summaries."

        try:
            summary = self.ai_analyzer.generate_text(prompt, system_prompt, mode='chat')
            return summary if summary else "Summary generation failed"
        except Exception as e:
            logger.error(f"AI summary generation failed: {e}")
            return f"Error generating summary: {str(e)}"

    def export_leadership_excel(self) -> io.BytesIO:
        """Export to Leadership Excel format (31 columns with financials)"""
        if not self.cases:
            # Return empty template
            df = pd.DataFrame(columns=ALL_COLUMNS_LEADERSHIP)
        else:
            # Convert all cases to dicts
            data = [case.to_dict_leadership() for case in self.cases]
            df = pd.DataFrame(data, columns=ALL_COLUMNS_LEADERSHIP)

        # Create Excel file
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            # Main tracker sheet
            df.to_excel(writer, sheet_name='Tracker_ Priority List (Leaders)', index=False)

            # Comments sheet
            comments_df = pd.DataFrame(columns=['Comments'])
            comments_df.to_excel(writer, sheet_name='Comments', index=False)

            # Format the main sheet
            workbook = writer.book
            worksheet = writer.sheets['Tracker_ Priority List (Leaders)']

            # Bold headers
            for cell in worksheet[1]:
                cell.font = cell.font.copy(bold=True)

            # Auto-fit columns
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width

            # Enable auto-filter
            worksheet.auto_filter.ref = worksheet.dimensions

        excel_buffer.seek(0)
        return excel_buffer

    def export_company_wide_excel(self) -> io.BytesIO:
        """Export to Company Wide Excel format (25 columns, no financials)"""
        if not self.cases:
            # Return empty template
            df = pd.DataFrame(columns=ALL_COLUMNS_COMPANY_WIDE)
        else:
            # Convert all cases to dicts (company wide version)
            data = [case.to_dict_company_wide() for case in self.cases]
            df = pd.DataFrame(data, columns=ALL_COLUMNS_COMPANY_WIDE)

        # Create Excel file
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            # Main tracker sheet
            df.to_excel(writer, sheet_name='Company Wide Quality Tracker', index=False)

            # Comments sheet
            comments_df = pd.DataFrame(columns=['Comments'])
            comments_df.to_excel(writer, sheet_name='Comments', index=False)

            # Format the main sheet
            workbook = writer.book
            worksheet = writer.sheets['Company Wide Quality Tracker']

            # Bold headers
            for cell in worksheet[1]:
                cell.font = cell.font.copy(bold=True)

            # Auto-fit columns
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width

            # Enable auto-filter
            worksheet.auto_filter.ref = worksheet.dimensions

        excel_buffer.seek(0)
        return excel_buffer

    def export_leadership_csv(self) -> str:
        """Export to Leadership CSV format (31 columns)"""
        if not self.cases:
            df = pd.DataFrame(columns=ALL_COLUMNS_LEADERSHIP)
        else:
            data = [case.to_dict_leadership() for case in self.cases]
            df = pd.DataFrame(data, columns=ALL_COLUMNS_LEADERSHIP)

        return df.to_csv(index=False)

    def export_company_wide_csv(self) -> str:
        """Export to Company Wide CSV format (25 columns)"""
        if not self.cases:
            df = pd.DataFrame(columns=ALL_COLUMNS_COMPANY_WIDE)
        else:
            data = [case.to_dict_company_wide() for case in self.cases]
            df = pd.DataFrame(data, columns=ALL_COLUMNS_COMPANY_WIDE)

        return df.to_csv(index=False)

    def get_cases_dataframe(self, leadership_version: bool = False) -> pd.DataFrame:
        """Get DataFrame of all cases"""
        if not self.cases:
            columns = ALL_COLUMNS_LEADERSHIP if leadership_version else ALL_COLUMNS_COMPANY_WIDE
            return pd.DataFrame(columns=columns)

        if leadership_version:
            data = [case.to_dict_leadership() for case in self.cases]
            return pd.DataFrame(data, columns=ALL_COLUMNS_LEADERSHIP)
        else:
            data = [case.to_dict_company_wide() for case in self.cases]
            return pd.DataFrame(data, columns=ALL_COLUMNS_COMPANY_WIDE)


# Demo data generator
def generate_demo_cases() -> List[QualityTrackerCase]:
    """Generate 3 sample cases for demonstration"""
    cases = []

    # Demo Case 1: High priority with financials
    case1 = QualityTrackerCase()
    case1.priority = 1
    case1.product_name = "Vive Mobility Walker"
    case1.main_sales_channel = "Amazon"
    case1.asin = "B07XAMPLE1"
    case1.sku = "VMW-001"
    case1.fulfilled_by = "FBA"
    case1.ncx_rate = 0.0234
    case1.ncx_orders = 45
    case1.total_orders_t30 = 1923
    case1.star_rating_amazon = 4.5
    case1.return_rate_amazon = 0.0812
    case1.return_rate_b2b = 0.0345
    case1.flag_source_1 = "High Return Rate"
    case1.return_badge_displayed = "Yes"
    case1.notification_notes = "Customer reports handle issue"
    case1.top_issues = "Handle durability - plastic handle cracks after 2-3 months of use"
    case1.cost_of_refunds_annualized = 25340.00
    case1.savings_captured_12m = 8250.00
    case1.action_taken = "Redesigned handle mechanism with reinforced metal core"
    case1.date_action_taken = datetime(2024, 11, 15).date()
    case1.listing_manager_notified = "Yes"
    case1.product_dev_notified = "Yes"
    case1.flag_source = "Analytics"
    case1.follow_up_date = datetime(2025, 1, 15).date()
    case1.result_1_rr = 0.0612
    case1.result_check_date_1 = datetime(2024, 12, 15).date()
    case1.result_2_rr = 0.0498
    case1.result_2_date = datetime(2025, 1, 10).date()
    case1.top_issues_change = "Reduced handle complaints by 38%"
    case1.top_issues_change_date = datetime(2025, 1, 10).date()
    case1.case_status = "Monitoring"
    cases.append(case1)

    # Demo Case 2: Safety concern
    case2 = QualityTrackerCase()
    case2.priority = 2
    case2.product_name = "Vive Knee Walker"
    case2.main_sales_channel = "Amazon"
    case2.asin = "B08EXAMPLE2"
    case2.sku = "VKW-002"
    case2.fulfilled_by = "FBA"
    case2.ncx_rate = 0.0189
    case2.ncx_orders = 32
    case2.total_orders_t30 = 1695
    case2.star_rating_amazon = 4.7
    case2.return_rate_amazon = 0.0645
    case2.return_rate_b2b = 0.0289
    case2.flag_source_1 = "Customer Safety Concern"
    case2.return_badge_displayed = "No"
    case2.notification_notes = "Brake mechanism reported as stiff, difficult to engage"
    case2.top_issues = "Brake functionality - users report brakes require excessive force"
    case2.cost_of_refunds_annualized = 18900.00
    case2.savings_captured_12m = 0.00
    case2.action_taken = "Investigating with manufacturer - sent 5 units for testing"
    case2.date_action_taken = datetime(2025, 1, 5).date()
    case2.listing_manager_notified = "Yes"
    case2.product_dev_notified = "Yes"
    case2.flag_source = "Customer Service"
    case2.follow_up_date = datetime(2025, 2, 5).date()
    case2.case_status = "Active Investigation"
    cases.append(case2)

    # Demo Case 3: B2B issue
    case3 = QualityTrackerCase()
    case3.priority = 3
    case3.product_name = "Vive Rollator Walker"
    case3.main_sales_channel = "B2B"
    case3.asin = ""
    case3.sku = "VRW-003"
    case3.fulfilled_by = "FBM"
    case3.ncx_rate = 0.0056
    case3.ncx_orders = 8
    case3.total_orders_t30 = 1428
    case3.star_rating_amazon = 4.8
    case3.return_rate_amazon = 0.0234
    case3.return_rate_b2b = 0.0567
    case3.flag_source_1 = "B2B Return Rate Spike"
    case3.return_badge_displayed = "No"
    case3.notification_notes = "B2B customers report packaging damage during bulk shipping"
    case3.top_issues = "Packaging insufficient for bulk shipping - units arrive with cosmetic damage"
    case3.cost_of_refunds_annualized = 6750.00
    case3.savings_captured_12m = 0.00
    case3.action_taken = "Upgraded packaging for B2B orders with double-wall corrugated boxes"
    case3.date_action_taken = datetime(2025, 1, 8).date()
    case3.listing_manager_notified = "No"
    case3.product_dev_notified = "Yes"
    case3.flag_source = "B2B Reports"
    case3.follow_up_date = datetime(2025, 2, 15).date()
    case3.case_status = "Action Taken - Monitoring"
    cases.append(case3)

    return cases


__all__ = [
    'QualityTrackerCase',
    'QualityTrackerManager',
    'ALL_COLUMNS_LEADERSHIP',
    'ALL_COLUMNS_COMPANY_WIDE',
    'LEADERSHIP_ONLY_COLUMNS',
    'generate_demo_cases'
]
