"""
Inventory Integration Module for Quality Management System

Integrates Odoo inventory data with B2B return reports to calculate:
- Days of Inventory (DOI) - planning and conservative views
- Reorder points and lead time windows
- Corrective action windows before reordering
- At-risk pipeline exposure
- Quality hold scenarios

This is medical device quality management software. All calculations must be accurate.
DO NOT HALLUCINATE OR FABRICATE DATA.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import re
import io


class OdooInventoryParser:
    """
    Parses Odoo Inventory Forecast files with special handling for first-row headers.

    Critical: Odoo files have actual headers in the first DATA row, not Excel header row.
    """

    # Core required columns for calculations (relaxed from full spec)
    REQUIRED_COLUMNS = [
        'SKU', 'Product Title',
        'On Hand', 'Total Units',
        'Total Daily rate', 'Unit Cost'
    ]

    # Optional columns that enhance functionality but aren't required
    OPTIONAL_COLUMNS = [
        'ASIN', 'Supplier', 'Status', 'Amazon Status',
        'On Order', 'Shipments in Transit', 'FBA Inbound',
        'DOI', 'Warehouse DOI',
        'Last Sale (days)', 'Amazon OOS', 'Warehouse OOS', 'Total OOS'
    ]

    def __init__(self):
        self.raw_df = None
        self.parsed_df = None
        self.column_mapping = {}  # Maps found columns to standard names

    def parse_file(self, file_content) -> pd.DataFrame:
        """
        Parse Odoo inventory file with first-row header promotion.

        Args:
            file_content: File uploaded via Streamlit file_uploader

        Returns:
            DataFrame with canonical schema
        """
        try:
            # First, read without header to inspect structure
            df_raw = pd.read_excel(file_content, sheet_name='Sheet1', header=None, nrows=10)

            # Find which row contains the headers by looking for 'SKU' column
            header_row = None
            for idx in range(min(5, len(df_raw))):  # Check first 5 rows
                row_values = df_raw.iloc[idx].astype(str).str.strip().str.lower()
                if 'sku' in row_values.values:
                    header_row = idx
                    break

            if header_row is None:
                raise ValueError("Could not find header row containing 'SKU' column")

            # Now read with correct header row
            file_content.seek(0)  # Reset file pointer
            df = pd.read_excel(file_content, sheet_name='Sheet1', header=header_row)

            # Clean column names (strip whitespace, handle encoding issues)
            df.columns = df.columns.str.strip()

            # Store raw for debugging
            self.raw_df = df.copy()

            # Validate required columns exist
            missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
            if missing_cols:
                # Provide helpful debugging info
                actual_cols = list(df.columns)
                raise ValueError(
                    f"Missing required columns: {missing_cols}\n"
                    f"Found columns: {actual_cols}\n"
                    f"Header row detected at row {header_row}"
                )

            # Add missing optional columns with default values
            for col in self.OPTIONAL_COLUMNS:
                if col not in df.columns:
                    if col in ['On Order', 'Shipments in Transit', 'FBA Inbound',
                              'Last Sale (days)', 'Amazon OOS', 'Warehouse OOS', 'Total OOS']:
                        df[col] = 0
                    elif col in ['DOI', 'Warehouse DOI']:
                        df[col] = np.nan
                    else:
                        df[col] = ''

            # Clean and validate data types
            df = self._clean_data(df)

            # Store parsed result
            self.parsed_df = df

            return df

        except Exception as e:
            raise ValueError(f"Failed to parse Odoo inventory file: {str(e)}")

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate Odoo data types."""

        # Numeric columns that should be integers
        int_cols = ['On Hand', 'On Order', 'Shipments in Transit', 'FBA Inbound',
                   'Total Units', 'Last Sale (days)', 'Amazon OOS', 'Warehouse OOS', 'Total OOS']

        # Numeric columns that should be floats
        float_cols = ['Total Daily rate', 'Unit Cost', 'DOI', 'Warehouse DOI']

        # Convert integer columns
        for col in int_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

        # Convert float columns
        for col in float_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Handle 9999 as null indicator for DOI when no demand
                if 'DOI' in col:
                    df[col] = df[col].replace(9999, np.nan)

        # String columns - strip whitespace
        str_cols = ['SKU', 'ASIN', 'Product Title', 'Supplier', 'Status', 'Amazon Status']
        for col in str_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        return df


class PivotReturnReportParser:
    """
    Parses Pivot Return Report with hierarchical structure and bracket SKU extraction.

    Critical: Report has B2B returns only (NOT Amazon returns).
    Structure: Metadata rows, month headers, then SKU rows with format "[SKU] Product Name"
    """

    def __init__(self):
        self.raw_df = None
        self.parsed_df = None

    def parse_file(self, file_content) -> pd.DataFrame:
        """
        Parse Pivot Return Report and extract SKU-level B2B return quantities.

        Args:
            file_content: File uploaded via Streamlit file_uploader

        Returns:
            DataFrame with columns: SKU, ReturnQty, Month (optional)
        """
        try:
            # Read Excel file
            df = pd.read_excel(file_content, sheet_name='Return Report', header=None)

            # Store raw for debugging
            self.raw_df = df.copy()

            # Extract SKU rows and aggregate
            result = self._extract_sku_returns(df)

            # Store parsed result
            self.parsed_df = result

            return result

        except Exception as e:
            raise ValueError(f"Failed to parse Pivot Return Report: {str(e)}")

    def _extract_sku_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract SKU and return quantities from pivot structure.

        Logic:
        - Look for rows with pattern [SKU] Product Name
        - Extract SKU from brackets
        - Sum all return quantities for each SKU across months
        """
        records = []

        # Iterate through all rows
        for idx, row in df.iterrows():
            # Check first column for bracket pattern
            first_col = str(row.iloc[0]) if pd.notna(row.iloc[0]) else ""

            # Match pattern: [SKU] Product Name
            match = re.search(r'\[([^\]]+)\]', first_col)

            if match:
                sku = match.group(1).strip()

                # Sum all numeric values in this row (return quantities across months)
                # Skip first column (SKU/name) and sum the rest
                return_qty = 0
                for val in row.iloc[1:]:
                    if pd.notna(val) and isinstance(val, (int, float)):
                        return_qty += val

                records.append({
                    'SKU': sku,
                    'B2B_ReturnQty': int(return_qty)
                })

        # Convert to DataFrame and aggregate by SKU
        if records:
            result_df = pd.DataFrame(records)
            result_df = result_df.groupby('SKU', as_index=False)['B2B_ReturnQty'].sum()
        else:
            result_df = pd.DataFrame(columns=['SKU', 'B2B_ReturnQty'])

        return result_df


class InventoryConfiguration:
    """
    Manages configuration with precedence: UI overrides > per-SKU upload > global defaults.
    """

    def __init__(self):
        # Global defaults
        self.global_lead_time_days = 45
        self.global_safety_stock_days = 14

        # Per-SKU config (uploaded via CSV)
        self.sku_config = {}

        # UI overrides (temporary, per-session)
        self.ui_overrides = {}

    def load_sku_config(self, file_content) -> None:
        """
        Load per-SKU configuration from CSV upload.

        Expected columns: SKU, LeadTimeDays, SafetyStockDays
        """
        try:
            df = pd.read_csv(file_content)

            # Validate columns
            required_cols = ['SKU', 'LeadTimeDays', 'SafetyStockDays']
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns in SKU config: {missing}")

            # Convert to dict
            for _, row in df.iterrows():
                sku = str(row['SKU']).strip()
                self.sku_config[sku] = {
                    'lead_time_days': int(row['LeadTimeDays']),
                    'safety_stock_days': int(row['SafetyStockDays'])
                }

        except Exception as e:
            raise ValueError(f"Failed to load SKU configuration: {str(e)}")

    def set_ui_override(self, sku: str, lead_time: Optional[int] = None,
                       safety_stock: Optional[int] = None) -> None:
        """Set UI override for specific SKU."""
        if sku not in self.ui_overrides:
            self.ui_overrides[sku] = {}

        if lead_time is not None:
            self.ui_overrides[sku]['lead_time_days'] = lead_time
        if safety_stock is not None:
            self.ui_overrides[sku]['safety_stock_days'] = safety_stock

    def get_config(self, sku: str) -> Dict[str, int]:
        """
        Get configuration for SKU with precedence handling.

        Returns:
            Dict with keys: lead_time_days, safety_stock_days
        """
        # Start with global defaults
        config = {
            'lead_time_days': self.global_lead_time_days,
            'safety_stock_days': self.global_safety_stock_days
        }

        # Apply per-SKU config if exists
        if sku in self.sku_config:
            config.update(self.sku_config[sku])

        # Apply UI overrides if exist (highest precedence)
        if sku in self.ui_overrides:
            config.update(self.ui_overrides[sku])

        return config


class InventoryCalculator:
    """
    Performs all inventory calculations:
    - DOI (planning and conservative views)
    - Reorder points
    - Days to reorder
    - Corrective action windows
    - At-risk exposure
    """

    def __init__(self, config: InventoryConfiguration):
        self.config = config

    def calculate_inventory_metrics(self, odoo_df: pd.DataFrame,
                                    returns_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Calculate all inventory metrics for each SKU.

        Args:
            odoo_df: Parsed Odoo inventory data
            returns_df: Optional parsed B2B returns data

        Returns:
            DataFrame with all calculated metrics
        """
        result = odoo_df.copy()

        # Merge B2B returns if provided
        if returns_df is not None and not returns_df.empty:
            result = result.merge(returns_df, on='SKU', how='left')
            result['B2B_ReturnQty'] = result['B2B_ReturnQty'].fillna(0).astype(int)
        else:
            result['B2B_ReturnQty'] = 0

        # Calculate metrics for each SKU
        metrics_list = []
        for _, row in result.iterrows():
            sku = row['SKU']
            metrics = self._calculate_sku_metrics(row, sku)
            metrics_list.append(metrics)

        # Convert to DataFrame and merge with original data
        metrics_df = pd.DataFrame(metrics_list)
        result = pd.concat([result.reset_index(drop=True), metrics_df], axis=1)

        return result

    def _calculate_sku_metrics(self, row: pd.Series, sku: str) -> Dict:
        """Calculate all metrics for a single SKU."""

        # Get configuration for this SKU
        config = self.config.get_config(sku)
        lead_time = config['lead_time_days']
        safety_stock = config['safety_stock_days']

        # Extract values from row
        total_units = row['Total Units']
        on_hand = row['On Hand']
        daily_rate = row['Total Daily rate']
        unit_cost = row['Unit Cost']
        on_order = row['On Order']
        in_transit = row['Shipments in Transit']
        fba_inbound = row['FBA Inbound']

        # Handle zero or null daily rate
        if pd.isna(daily_rate) or daily_rate <= 0:
            # No demand - set DOI to None (will display as N/A)
            doi_planning = None
            doi_conservative = None
            days_to_reorder = None
            must_order_by = None
            ca_window_before_po = None
            ca_window_before_arrival = None
            reorder_point = None
            at_risk_units = 0
            at_risk_dollars = 0.0
        else:
            # DOI Planning View: Total Units / Daily Rate
            doi_planning = total_units / daily_rate

            # DOI Conservative View: On Hand / Daily Rate
            doi_conservative = on_hand / daily_rate

            # Reorder Point: (Lead Time × Daily Rate) + Safety Stock
            reorder_point = (lead_time * daily_rate) + (safety_stock * daily_rate)

            # Days to Reorder: (On Hand - Reorder Point) / Daily Rate
            days_to_reorder = (on_hand - reorder_point) / daily_rate

            # Must Order By: Today + Days to Reorder
            if days_to_reorder > 0:
                must_order_by_date = datetime.now() + timedelta(days=days_to_reorder)
                must_order_by = must_order_by_date.strftime('%Y-%m-%d')
            else:
                must_order_by = "OVERDUE"

            # Corrective Action Window (Before PO): Days to Reorder
            ca_window_before_po = max(0, days_to_reorder)

            # Corrective Action Window (Before Arrival): Days to Reorder + Lead Time
            ca_window_before_arrival = max(0, days_to_reorder + lead_time)

            # At-Risk Pipeline Exposure
            pipeline_units = on_order + in_transit + fba_inbound
            at_risk_units = pipeline_units
            at_risk_dollars = pipeline_units * unit_cost

        return {
            'LeadTimeDays': lead_time,
            'SafetyStockDays': safety_stock,
            'DOI_Planning': doi_planning,
            'DOI_Conservative': doi_conservative,
            'ReorderPoint': reorder_point,
            'DaysToReorder': days_to_reorder,
            'MustOrderBy': must_order_by,
            'CA_Window_BeforePO': ca_window_before_po,
            'CA_Window_BeforeArrival': ca_window_before_arrival,
            'AtRiskUnits': at_risk_units,
            'AtRiskDollars': at_risk_dollars
        }

    def calculate_quality_hold_scenario(self, row: pd.Series, hold_duration_days: int) -> Dict:
        """
        Calculate what happens if product is put on quality hold.

        Args:
            row: Single SKU row with all metrics
            hold_duration_days: Duration of quality hold

        Returns:
            Dict with projected impact
        """
        daily_rate = row['Total Daily rate']
        on_hand = row['On Hand']
        days_to_reorder = row['DaysToReorder']

        if pd.isna(daily_rate) or daily_rate <= 0:
            return {
                'ProjectedStockout': False,
                'DaysUntilStockout': None,
                'UnitsShortfall': 0,
                'RecommendedAction': 'No active demand'
            }

        # Calculate days of inventory after hold
        remaining_doi = (on_hand / daily_rate) - hold_duration_days

        if remaining_doi < 0:
            # Will stock out during hold
            days_until_stockout = on_hand / daily_rate
            units_shortfall = abs(remaining_doi * daily_rate)
            recommended_action = "CRITICAL: Expedite or find substitute - will stock out during hold"
            projected_stockout = True
        elif days_to_reorder - hold_duration_days < 0:
            # Won't stock out during hold, but will miss reorder window
            days_until_stockout = None
            units_shortfall = 0
            recommended_action = "WARNING: Quality hold will delay reorder - expedite production"
            projected_stockout = False
        else:
            # Sufficient buffer
            days_until_stockout = None
            units_shortfall = 0
            recommended_action = "OK: Sufficient inventory buffer for quality hold"
            projected_stockout = False

        return {
            'ProjectedStockout': projected_stockout,
            'DaysUntilStockout': days_until_stockout,
            'UnitsShortfall': units_shortfall,
            'RecommendedAction': recommended_action
        }


class IntegratedAnalyzer:
    """
    Integrates quality screening with inventory management to provide unified recommendations.

    Considers both quality issues AND inventory/reorder constraints.
    """

    def __init__(self):
        pass

    def generate_integrated_recommendation(self, row: pd.Series,
                                          quality_flagged: bool = False,
                                          quality_severity: str = 'None') -> Dict:
        """
        Generate recommendation considering both quality and inventory factors.

        Args:
            row: SKU data with all metrics
            quality_flagged: Whether product is flagged in quality screening
            quality_severity: Severity level (Low/Medium/High/Critical)

        Returns:
            Dict with integrated recommendation
        """
        sku = row['SKU']
        days_to_reorder = row['DaysToReorder']
        ca_window_before_po = row['CA_Window_BeforePO']
        ca_window_before_arrival = row['CA_Window_BeforeArrival']

        # Determine status flags
        status_flags = []
        priority = "Normal"
        recommended_actions = []

        # Check inventory urgency
        if pd.notna(days_to_reorder):
            if days_to_reorder < 0:
                status_flags.append("🔴 PAST REORDER POINT")
                priority = "Critical"
                recommended_actions.append("Place emergency PO immediately")
            elif days_to_reorder < 7:
                status_flags.append("🟡 REORDER SOON")
                priority = "High"
                recommended_actions.append("Prepare PO this week")
            elif days_to_reorder < 14:
                status_flags.append("🟢 MONITOR")
                priority = "Medium"

        # Check quality issues
        if quality_flagged:
            status_flags.append(f"⚠️ QUALITY ISSUE - {quality_severity}")

            if quality_severity in ['Critical', 'High']:
                priority = "Critical"

                # Check if we have time for corrective action
                if pd.notna(ca_window_before_po) and ca_window_before_po > 14:
                    recommended_actions.append(
                        f"INVESTIGATE & FIX: {ca_window_before_po:.1f} days before PO needed"
                    )
                elif pd.notna(ca_window_before_arrival) and ca_window_before_arrival > 30:
                    recommended_actions.append(
                        f"FIX IN PRODUCTION: {ca_window_before_arrival:.1f} days before arrival"
                    )
                else:
                    recommended_actions.append(
                        "URGENT: Quality issue but reorder imminent - consider expedited fix or temporary hold"
                    )
            else:
                # Low/Medium severity
                if pd.notna(ca_window_before_po) and ca_window_before_po > 7:
                    recommended_actions.append(
                        f"PLAN CORRECTION: {ca_window_before_po:.1f} days available before reorder"
                    )
                else:
                    recommended_actions.append(
                        "Monitor quality - may need to order with known issue and fix in next batch"
                    )

        # No recommendation if neither urgent nor quality flagged
        if not status_flags:
            status_flags.append("✅ HEALTHY")
            recommended_actions.append("Continue monitoring")

        return {
            'SKU': sku,
            'Priority': priority,
            'StatusFlags': ' | '.join(status_flags),
            'RecommendedActions': ' | '.join(recommended_actions)
        }
