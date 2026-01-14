# Quick Reference Guide: Manual Entry Form & Export

## Column Specifications for Form Fields

### Field Groups

#### Group 1: Product Identification (Always Required)
```python
fields = [
    {'name': 'Product name', 'type': 'text', 'required': True, 'max_length': 200},
    {'name': 'Main Sales Channel (by Volume)', 'type': 'dropdown', 'options': ['Amazon', 'B2B', 'Direct', 'Walmart', 'Other']},
    {'name': 'ASIN', 'type': 'text', 'pattern': r'^[B][0-9A-Z]{9}$', 'placeholder': 'B07XXXXXXX'},
    {'name': 'SKU', 'type': 'text', 'required': True, 'placeholder': 'VMW-001'},
    {'name': 'Fulfilled by', 'type': 'dropdown', 'options': ['FBA', 'FBM', 'Merchant']},
]
```

#### Group 2: Quality Metrics
```python
fields = [
    {'name': 'NCX rate', 'type': 'number', 'min': 0, 'max': 1, 'step': 0.0001, 'placeholder': '0.0234'},
    {'name': 'NCX orders', 'type': 'integer', 'min': 0, 'placeholder': '45'},
    {'name': 'Star Rating Amazon', 'type': 'number', 'min': 0, 'max': 5, 'step': 0.1, 'placeholder': '4.5'},
    {'name': 'Return rate Amazon', 'type': 'number', 'min': 0, 'max': 1, 'step': 0.0001, 'placeholder': '0.0812'},
    {'name': 'Return Rate B2B', 'type': 'number', 'min': 0, 'max': 1, 'step': 0.0001, 'placeholder': '0.0345'},
]
```

#### Group 3: Leadership Only (Conditional Display)
```python
# Only show if user has leadership role
leadership_fields = [
    {'name': 'Priority', 'type': 'integer', 'min': 1, 'placeholder': '1', 'leadership_only': True},
    {'name': 'Total orders (t30)', 'type': 'integer', 'min': 0, 'placeholder': '1923', 'leadership_only': True},
    {'name': 'Flag Source 1', 'type': 'text', 'placeholder': 'High Return Rate', 'leadership_only': True},
    {'name': 'Cost of Refunds (Annualized)', 'type': 'currency', 'placeholder': '25340.00', 'leadership_only': True},
    {'name': '12m Savings Captured (based on rr% reduction)', 'type': 'currency', 'placeholder': '8250.00', 'leadership_only': True},
    {'name': 'Case Status', 'type': 'dropdown', 'options': ['Active Investigation', 'Action Taken - Monitoring', 'Monitoring', 'Closed'], 'leadership_only': True},
]
```

#### Group 4: Issue Documentation
```python
fields = [
    {'name': 'Return Badge Displayed Amazon', 'type': 'dropdown', 'options': ['Yes', 'No', 'N/A']},
    {'name': 'Notification/Notes', 'type': 'textarea', 'max_length': 500, 'placeholder': 'Customer reports handle issue'},
    {'name': 'Top Issue(s)', 'type': 'textarea', 'required': True, 'max_length': 200, 'placeholder': 'Handle durability'},
]
```

#### Group 5: Corrective Actions
```python
fields = [
    {'name': 'Action Taken', 'type': 'textarea', 'max_length': 500, 'placeholder': 'Redesigned handle mechanism'},
    {'name': 'Date Action Taken', 'type': 'date', 'auto_today': True},
    {'name': 'Listing Manager Notified?', 'type': 'dropdown', 'options': ['Yes', 'No', 'N/A']},
    {'name': 'Product Dev Notified?', 'type': 'dropdown', 'options': ['Yes', 'No', 'N/A']},
    {'name': 'Flag Source', 'type': 'dropdown', 'options': ['Analytics', 'Customer Service', 'B2B Reports', 'Returns Data', 'Vendor Report', 'Internal Audit']},
    {'name': 'Follow Up Date', 'type': 'date', 'min_date': 'today'},
]
```

#### Group 6: Results Tracking
```python
fields = [
    {'name': 'Result 1 (rr%)', 'type': 'number', 'min': 0, 'max': 1, 'step': 0.0001, 'placeholder': '0.0612'},
    {'name': 'Result Check Date 1', 'type': 'date'},
    {'name': 'Result 2 (rr%)', 'type': 'number', 'min': 0, 'max': 1, 'step': 0.0001, 'placeholder': '0.0498'},
    {'name': 'Result 2 Date', 'type': 'date'},
    {'name': 'Top Issue(s) Change', 'type': 'textarea', 'max_length': 200, 'placeholder': 'Reduced handle complaints'},
    {'name': 'Top Issue(s) Change Date', 'type': 'date'},
]
```

---

## Export Function Specifications

### Python Export Function Template

```python
def export_quality_tracker(df, user_role='standard'):
    """
    Export quality tracker to Excel with role-based column filtering

    Args:
        df: pandas DataFrame with all quality data
        user_role: 'leadership' or 'standard' (default)

    Returns:
        BytesIO object containing Excel file
    """
    import pandas as pd
    import io
    from datetime import datetime

    # Define column order for Leadership version (31 columns)
    leadership_columns = [
        'Priority',
        'Product name',
        'Main Sales Channel (by Volume)',
        'ASIN',
        'SKU',
        'Fulfilled by',
        'NCX rate',
        'NCX orders',
        'Total orders (t30)',
        'Star Rating Amazon',
        'Return rate Amazon',
        'Return Rate B2B',
        'Flag Source 1',
        'Return Badge Displayed Amazon',
        'Notification/Notes',
        'Top Issue(s)',
        'Cost of Refunds (Annualized)',
        '12m Savings Captured (based on rr% reduction)',
        'Action Taken',
        'Date Action Taken',
        'Listing Manager Notified?',
        'Product Dev Notified?',
        'Flag Source',
        'Follow Up Date',
        'Result 1 (rr%)',
        'Result Check Date 1',
        'Result 2 (rr%)',
        'Result 2 Date',
        'Top Issue(s) Change',
        'Top Issue(s) Change Date',
        'Case Status'
    ]

    # Define columns to exclude for Company Wide version (6 columns)
    leadership_only_columns = [
        'Priority',
        'Total orders (t30)',
        'Flag Source 1',
        'Cost of Refunds (Annualized)',
        '12m Savings Captured (based on rr% reduction)',
        'Case Status'
    ]

    # Select columns based on user role
    if user_role == 'leadership':
        export_columns = leadership_columns
        filename = 'Tracker_ Priority List (Leadership).xlsx'
        sheet_name = 'Tracker_ Priority List (Leaders)'
    else:
        export_columns = [col for col in leadership_columns if col not in leadership_only_columns]
        filename = 'Company Wide Quality Tracker.xlsx'
        sheet_name = 'Company Wide Quality Tracker'

    # Filter and reorder DataFrame
    df_export = df[export_columns].copy()

    # Create Excel file
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Write main tracker sheet
        df_export.to_excel(writer, sheet_name=sheet_name, index=False)

        # Write comments sheet
        pd.DataFrame({'': []}).to_excel(writer, sheet_name='Comments', index=False)

        # Get workbook and worksheet for formatting
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]

        # Format header row
        for cell in worksheet[1]:
            cell.font = openpyxl.styles.Font(bold=True)
            cell.fill = openpyxl.styles.PatternFill(start_color='D3D3D3', end_color='D3D3D3', fill_type='solid')

        # Auto-adjust column widths
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

    output.seek(0)
    return output, filename
```

---

## Streamlit Form Implementation

```python
import streamlit as st
import pandas as pd
from datetime import datetime

def render_quality_entry_form(user_role='standard'):
    """Render manual entry form for quality tracker"""

    st.header("Quality Tracker - Manual Entry")

    with st.form("quality_entry_form"):
        # Product Identification
        st.subheader("1. Product Identification")
        col1, col2 = st.columns(2)

        with col1:
            product_name = st.text_input("Product name*", max_chars=200)
            asin = st.text_input("ASIN", placeholder="B07XXXXXXX")
            sku = st.text_input("SKU*", placeholder="VMW-001")

        with col2:
            sales_channel = st.selectbox(
                "Main Sales Channel (by Volume)",
                ["", "Amazon", "B2B", "Direct", "Walmart", "Other"]
            )
            fulfilled_by = st.selectbox("Fulfilled by", ["", "FBA", "FBM", "Merchant"])

        # Leadership Fields (conditional)
        if user_role == 'leadership':
            st.subheader("2. Leadership Metrics")
            col1, col2, col3 = st.columns(3)

            with col1:
                priority = st.number_input("Priority", min_value=1, step=1)
                total_orders = st.number_input("Total orders (t30)", min_value=0, step=1)

            with col2:
                flag_source_1 = st.text_input("Flag Source 1", placeholder="High Return Rate")
                case_status = st.selectbox(
                    "Case Status",
                    ["", "Active Investigation", "Action Taken - Monitoring", "Monitoring", "Closed"]
                )

            with col3:
                cost_refunds = st.number_input("Cost of Refunds (Annualized)", min_value=0.0, format="%.2f")
                savings_captured = st.number_input("12m Savings Captured", min_value=0.0, format="%.2f")

        # Quality Metrics
        st.subheader("3. Quality Metrics")
        col1, col2, col3 = st.columns(3)

        with col1:
            ncx_rate = st.number_input("NCX rate", min_value=0.0, max_value=1.0, format="%.4f")
            ncx_orders = st.number_input("NCX orders", min_value=0, step=1)

        with col2:
            star_rating = st.number_input("Star Rating Amazon", min_value=0.0, max_value=5.0, step=0.1)
            return_rate_amazon = st.number_input("Return rate Amazon*", min_value=0.0, max_value=1.0, format="%.4f")

        with col3:
            return_rate_b2b = st.number_input("Return Rate B2B", min_value=0.0, max_value=1.0, format="%.4f")
            return_badge = st.selectbox("Return Badge Displayed Amazon", ["", "Yes", "No", "N/A"])

        # Issue Documentation
        st.subheader("4. Issue Documentation")
        notification_notes = st.text_area("Notification/Notes", max_chars=500, placeholder="Customer reports handle issue")
        top_issues = st.text_area("Top Issue(s)*", max_chars=200, placeholder="Handle durability")

        # Corrective Actions
        st.subheader("5. Corrective Actions")
        col1, col2 = st.columns(2)

        with col1:
            action_taken = st.text_area("Action Taken", max_chars=500, placeholder="Redesigned handle mechanism")
            date_action = st.date_input("Date Action Taken", value=datetime.today())
            listing_notified = st.selectbox("Listing Manager Notified?", ["", "Yes", "No", "N/A"])

        with col2:
            product_dev_notified = st.selectbox("Product Dev Notified?", ["", "Yes", "No", "N/A"])
            flag_source = st.selectbox(
                "Flag Source",
                ["", "Analytics", "Customer Service", "B2B Reports", "Returns Data", "Vendor Report", "Internal Audit"]
            )
            follow_up_date = st.date_input("Follow Up Date", min_value=datetime.today())

        # Results Tracking
        st.subheader("6. Results Tracking (Optional)")
        col1, col2 = st.columns(2)

        with col1:
            result1_rr = st.number_input("Result 1 (rr%)", min_value=0.0, max_value=1.0, format="%.4f")
            result1_date = st.date_input("Result Check Date 1")
            result2_rr = st.number_input("Result 2 (rr%)", min_value=0.0, max_value=1.0, format="%.4f")

        with col2:
            result2_date = st.date_input("Result 2 Date")
            top_issue_change = st.text_area("Top Issue(s) Change", max_chars=200)
            top_issue_change_date = st.date_input("Top Issue(s) Change Date")

        # Submit button
        submitted = st.form_submit_button("Add Entry")

        if submitted:
            # Validate required fields
            if not product_name or not sku or not top_issues:
                st.error("Please fill in all required fields (marked with *)")
                return None

            # Build data dictionary
            entry_data = {
                'Product name': product_name,
                'Main Sales Channel (by Volume)': sales_channel,
                'ASIN': asin,
                'SKU': sku,
                'Fulfilled by': fulfilled_by,
                'NCX rate': ncx_rate,
                'NCX orders': ncx_orders,
                'Star Rating Amazon': star_rating,
                'Return rate Amazon': return_rate_amazon,
                'Return Rate B2B': return_rate_b2b,
                'Return Badge Displayed Amazon': return_badge,
                'Notification/Notes': notification_notes,
                'Top Issue(s)': top_issues,
                'Action Taken': action_taken,
                'Date Action Taken': date_action,
                'Listing Manager Notified?': listing_notified,
                'Product Dev Notified?': product_dev_notified,
                'Flag Source': flag_source,
                'Follow Up Date': follow_up_date,
                'Result 1 (rr%)': result1_rr if result1_rr > 0 else None,
                'Result Check Date 1': result1_date if result1_rr > 0 else None,
                'Result 2 (rr%)': result2_rr if result2_rr > 0 else None,
                'Result 2 Date': result2_date if result2_rr > 0 else None,
                'Top Issue(s) Change': top_issue_change,
                'Top Issue(s) Change Date': top_issue_change_date if top_issue_change else None,
            }

            # Add leadership fields if applicable
            if user_role == 'leadership':
                entry_data.update({
                    'Priority': priority,
                    'Total orders (t30)': total_orders,
                    'Flag Source 1': flag_source_1,
                    'Cost of Refunds (Annualized)': cost_refunds,
                    '12m Savings Captured (based on rr% reduction)': savings_captured,
                    'Case Status': case_status,
                })

            st.success("Entry added successfully!")
            return entry_data

    return None
```

---

## Column Order Cheat Sheet

### Leadership (31 columns) - IN ORDER
1. Priority [L]
2. Product name
3. Main Sales Channel (by Volume)
4. ASIN
5. SKU
6. Fulfilled by
7. NCX rate
8. NCX orders
9. Total orders (t30) [L]
10. Star Rating Amazon
11. Return rate Amazon
12. Return Rate B2B
13. Flag Source 1 [L]
14. Return Badge Displayed Amazon
15. Notification/Notes
16. Top Issue(s)
17. Cost of Refunds (Annualized) [L] [$]
18. 12m Savings Captured (based on rr% reduction) [L] [$]
19. Action Taken
20. Date Action Taken
21. Listing Manager Notified?
22. Product Dev Notified?
23. Flag Source
24. Follow Up Date
25. Result 1 (rr%)
26. Result Check Date 1
27. Result 2 (rr%)
28. Result 2 Date
29. Top Issue(s) Change
30. Top Issue(s) Change Date
31. Case Status [L]

**Legend**: [L] = Leadership Only, [$] = Financial

### Company Wide (25 columns) - Remove these 6
- Priority
- Total orders (t30)
- Flag Source 1
- Cost of Refunds (Annualized)
- 12m Savings Captured (based on rr% reduction)
- Case Status

---

## File Naming Convention

```python
def get_filename(user_role):
    if user_role == 'leadership':
        return 'Tracker_ Priority List (Leadership).xlsx'
    else:
        return 'Company Wide Quality Tracker.xlsx'

def get_sheet_name(user_role):
    if user_role == 'leadership':
        return 'Tracker_ Priority List (Leaders)'
    else:
        return 'Company Wide Quality Tracker'
```

---

## Data Validation Rules

```python
VALIDATION_RULES = {
    'Product name': {'required': True, 'max_length': 200},
    'SKU': {'required': True, 'max_length': 50},
    'ASIN': {'pattern': r'^[B][0-9A-Z]{9}$', 'optional': True},
    'NCX rate': {'min': 0.0, 'max': 1.0},
    'NCX orders': {'min': 0, 'type': 'integer'},
    'Total orders (t30)': {'min': 0, 'type': 'integer'},
    'Star Rating Amazon': {'min': 0.0, 'max': 5.0},
    'Return rate Amazon': {'required': True, 'min': 0.0, 'max': 1.0},
    'Return Rate B2B': {'min': 0.0, 'max': 1.0},
    'Top Issue(s)': {'required': True, 'max_length': 200},
    'Cost of Refunds (Annualized)': {'min': 0.0, 'type': 'currency'},
    '12m Savings Captured (based on rr% reduction)': {'min': 0.0, 'type': 'currency'},
    'Date Action Taken': {'type': 'date'},
    'Follow Up Date': {'type': 'date', 'min_date': 'today'},
}
```
