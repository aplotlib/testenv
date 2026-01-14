# Excel Files Structure Analysis

## Files Analyzed
1. `c:\Users\Alex\code\testenv\Tracker_ Priority List (Leadership).xlsx`
2. `c:\Users\Alex\code\testenv\Company Wide Quality Tracker.xlsx`

---

## Executive Summary

The **Leadership version** contains **31 columns** with sensitive financial and strategic data, while the **Company Wide version** contains **25 columns** with the same operational data but excludes 6 sensitive columns. The Company Wide version is essentially a filtered subset designed for broader distribution.

### Key Differences:
- **6 columns** exist ONLY in Leadership version
- **2 financial columns** (Cost of Refunds, Savings Captured)
- **2 strategic metric columns** (Priority, Total orders)
- **2 internal tracking columns** (Flag Source 1, Case Status)

---

## File 1: Leadership Version (31 Columns)

**File**: `Tracker_ Priority List (Leadership).xlsx`

### Sheet Structure
- **Sheet 1**: "Tracker_ Priority List (Leaders)" - 31 columns
- **Sheet 2**: "Comments" - 1 column (for additional notes)
- **Current Data**: Headers only (1 row), no data rows present

### Complete Column List

| # | Column Name | Sample Data | Data Type | Category | Leadership Only |
|---|-------------|-------------|-----------|----------|-----------------|
| 1 | Priority | 1 | Integer | Strategic | YES |
| 2 | Product name | Vive Mobility Walker | Text | Product Info | |
| 3 | Main Sales Channel (by Volume) | Amazon | Text | Sales | |
| 4 | ASIN | B07XAMPLE1 | Text | Product ID | |
| 5 | SKU | VMW-001 | Text | Product ID | |
| 6 | Fulfilled by | FBA | Text | Operations | |
| 7 | NCX rate | 0.0234 | Decimal | Quality Metrics | |
| 8 | NCX orders | 45 | Integer | Quality Metrics | |
| 9 | Total orders (t30) | 1923 | Integer | Strategic Metrics | YES |
| 10 | Star Rating Amazon | 4.5 | Decimal | Customer Feedback | |
| 11 | Return rate Amazon | 0.0812 | Decimal | Quality Metrics | |
| 12 | Return Rate B2B | 0.0345 | Decimal | Quality Metrics | |
| 13 | Flag Source 1 | High Return Rate | Text | Internal Tracking | YES |
| 14 | Return Badge Displayed Amazon | Yes | Text/Boolean | Amazon Status | |
| 15 | Notification/Notes | Customer reports handle issue | Text | Notes | |
| 16 | Top Issue(s) | Handle durability | Text | Quality Analysis | |
| 17 | Cost of Refunds (Annualized) | $25,340 | Currency | Financial | YES |
| 18 | 12m Savings Captured (based on rr% reduction) | $8,250 | Currency | Financial | YES |
| 19 | Action Taken | Redesigned handle mechanism | Text | Corrective Action | |
| 20 | Date Action Taken | 2024-11-15 | Date | Corrective Action | |
| 21 | Listing Manager Notified? | Yes | Text/Boolean | Notifications | |
| 22 | Product Dev Notified? | Yes | Text/Boolean | Notifications | |
| 23 | Flag Source | Analytics | Text | Tracking | |
| 24 | Follow Up Date | 2025-01-15 | Date | Tracking | |
| 25 | Result 1 (rr%) | 0.0612 | Decimal | Results | |
| 26 | Result Check Date 1 | 2024-12-15 | Date | Results | |
| 27 | Result 2 (rr%) | 0.0498 | Decimal | Results | |
| 28 | Result 2 Date | 2025-01-10 | Date | Results | |
| 29 | Top Issue(s) Change | Reduced handle complaints | Text | Results | |
| 30 | Top Issue(s) Change Date | 2025-01-10 | Date | Results | |
| 31 | Case Status | Monitoring | Text | Internal Tracking | YES |

---

## File 2: Company Wide Version (25 Columns)

**File**: `Company Wide Quality Tracker.xlsx`

### Sheet Structure
- **Sheet 1**: "Company Wide Quality Tracker" - 25 columns
- **Sheet 2**: "Comments" - 1 column (for additional notes)
- **Current Data**: Headers only (1 row), no data rows present

### Column List (Same as Leadership EXCEPT the 6 excluded columns)

All columns from Leadership version 1-31, EXCLUDING:
- Column 1: Priority
- Column 9: Total orders (t30)
- Column 13: Flag Source 1
- Column 17: Cost of Refunds (Annualized)
- Column 18: 12m Savings Captured (based on rr% reduction)
- Column 31: Case Status

The remaining 25 columns appear in the same relative order as Leadership.

---

## Detailed Analysis: Leadership-Only Columns

### 1. Priority
- **Position**: Column 1 (first column)
- **Type**: Integer/Ranking
- **Sample**: 1, 2, 3
- **Purpose**: Strategic prioritization of quality issues
- **Category**: Strategic Information
- **Why Leadership Only**: Reveals company priorities and resource allocation

### 2. Total orders (t30)
- **Position**: Column 9
- **Type**: Integer
- **Sample**: 1923
- **Purpose**: Total order volume in last 30 days
- **Category**: Strategic Metrics
- **Why Leadership Only**: Reveals product sales volume and business performance

### 3. Flag Source 1
- **Position**: Column 13
- **Type**: Text
- **Sample**: "High Return Rate", "Customer Complaints"
- **Purpose**: Primary reason for flagging this issue
- **Category**: Internal Tracking
- **Why Leadership Only**: Internal operational information

### 4. Cost of Refunds (Annualized)
- **Position**: Column 17
- **Type**: Currency (USD)
- **Sample**: $25,340
- **Purpose**: Projected annual cost of refunds for this product
- **Category**: FINANCIAL - SENSITIVE
- **Why Leadership Only**: Confidential financial impact data

### 5. 12m Savings Captured (based on rr% reduction)
- **Position**: Column 18
- **Type**: Currency (USD)
- **Sample**: $8,250
- **Purpose**: Savings achieved from return rate reduction over 12 months
- **Category**: FINANCIAL - SENSITIVE
- **Why Leadership Only**: Confidential financial performance data

### 6. Case Status
- **Position**: Column 31 (last column)
- **Type**: Text
- **Sample**: "Monitoring", "Active", "Closed"
- **Purpose**: Current status of the quality case
- **Category**: Internal Tracking
- **Why Leadership Only**: Internal operational workflow information

---

## Common Columns (Both Files)

These 25 columns appear in both files:

### Product Identification (5 columns)
- Product name
- Main Sales Channel (by Volume)
- ASIN
- SKU
- Fulfilled by

### Quality Metrics (5 columns)
- NCX rate
- NCX orders
- Star Rating Amazon
- Return rate Amazon
- Return Rate B2B

### Amazon Status (1 column)
- Return Badge Displayed Amazon

### Issue Documentation (2 columns)
- Notification/Notes
- Top Issue(s)

### Corrective Actions (4 columns)
- Action Taken
- Date Action Taken
- Listing Manager Notified?
- Product Dev Notified?

### Tracking (2 columns)
- Flag Source
- Follow Up Date

### Results Tracking (6 columns)
- Result 1 (rr%)
- Result Check Date 1
- Result 2 (rr%)
- Result 2 Date
- Top Issue(s) Change
- Top Issue(s) Change Date

---

## Sample Data Rows

### Sample Row 1 (Leadership Version - All 31 columns)
```
Priority: 1
Product name: Vive Mobility Walker
Main Sales Channel: Amazon
ASIN: B07XAMPLE1
SKU: VMW-001
Fulfilled by: FBA
NCX rate: 0.0234
NCX orders: 45
Total orders (t30): 1923
Star Rating Amazon: 4.5
Return rate Amazon: 0.0812
Return Rate B2B: 0.0345
Flag Source 1: High Return Rate
Return Badge Displayed Amazon: Yes
Notification/Notes: Customer reports handle issue
Top Issue(s): Handle durability
Cost of Refunds (Annualized): $25,340
12m Savings Captured: $8,250
Action Taken: Redesigned handle mechanism
Date Action Taken: 2024-11-15
Listing Manager Notified?: Yes
Product Dev Notified?: Yes
Flag Source: Analytics
Follow Up Date: 2025-01-15
Result 1 (rr%): 0.0612
Result Check Date 1: 2024-12-15
Result 2 (rr%): 0.0498
Result 2 Date: 2025-01-10
Top Issue(s) Change: Reduced handle complaints
Top Issue(s) Change Date: 2025-01-10
Case Status: Monitoring
```

### Sample Row 1 (Company Wide Version - 25 columns)
```
Product name: Vive Mobility Walker
Main Sales Channel: Amazon
ASIN: B07XAMPLE1
SKU: VMW-001
Fulfilled by: FBA
NCX rate: 0.0234
NCX orders: 45
Star Rating Amazon: 4.5
Return rate Amazon: 0.0812
Return Rate B2B: 0.0345
Return Badge Displayed Amazon: Yes
Notification/Notes: Customer reports handle issue
Top Issue(s): Handle durability
Action Taken: Redesigned handle mechanism
Date Action Taken: 2024-11-15
Listing Manager Notified?: Yes
Product Dev Notified?: Yes
Flag Source: Analytics
Follow Up Date: 2025-01-15
Result 1 (rr%): 0.0612
Result Check Date 1: 2024-12-15
Result 2 (rr%): 0.0498
Result 2 Date: 2025-01-10
Top Issue(s) Change: Reduced handle complaints
Top Issue(s) Change Date: 2025-01-10
```

### Sample Row 2 (Leadership Version)
```
Priority: 2
Product name: Vive Knee Walker
Main Sales Channel: Amazon
ASIN: B08EXAMPLE2
SKU: VKW-002
Fulfilled by: FBA
NCX rate: 0.0189
NCX orders: 32
Total orders (t30): 1695
Star Rating Amazon: 4.7
Return rate Amazon: 0.0645
Return Rate B2B: 0.0289
Flag Source 1: Customer Safety Concern
Return Badge Displayed Amazon: No
Notification/Notes: Brake mechanism reported as stiff
Top Issue(s): Brake functionality
Cost of Refunds (Annualized): $18,900
12m Savings Captured: $0
Action Taken: Investigating with manufacturer
Date Action Taken: 2025-01-05
Listing Manager Notified?: Yes
Product Dev Notified?: Yes
Flag Source: Customer Service
Follow Up Date: 2025-02-05
Result 1 (rr%): [pending]
Result Check Date 1: [pending]
Result 2 (rr%): [pending]
Result 2 Date: [pending]
Top Issue(s) Change: [none]
Top Issue(s) Change Date: [none]
Case Status: Active Investigation
```

### Sample Row 3 (Leadership Version)
```
Priority: 3
Product name: Vive Rollator Walker
Main Sales Channel: B2B
ASIN: [N/A]
SKU: VRW-003
Fulfilled by: FBM
NCX rate: 0.0056
NCX orders: 8
Total orders (t30): 1428
Star Rating Amazon: 4.8
Return rate Amazon: 0.0234
Return Rate B2B: 0.0567
Flag Source 1: B2B Return Rate Spike
Return Badge Displayed Amazon: No
Notification/Notes: B2B customers report packaging damage
Top Issue(s): Packaging insufficient for bulk shipping
Cost of Refunds (Annualized): $6,750
12m Savings Captured: $0
Action Taken: Upgraded packaging for B2B orders
Date Action Taken: 2025-01-08
Listing Manager Notified?: No
Product Dev Notified?: Yes
Flag Source: B2B Reports
Follow Up Date: 2025-02-15
Result 1 (rr%): [pending]
Result Check Date 1: [pending]
Result 2 (rr%): [pending]
Result 2 Date: [pending]
Top Issue(s) Change: [none]
Top Issue(s) Change Date: [none]
Case Status: Action Taken - Monitoring
```

---

## Data Types by Column

### Text/String Columns
- Product name, Main Sales Channel, ASIN, SKU, Fulfilled by
- Flag Source 1, Return Badge Displayed Amazon
- Notification/Notes, Top Issue(s), Action Taken
- Listing Manager Notified?, Product Dev Notified?
- Flag Source, Top Issue(s) Change, Case Status

### Numeric Columns
#### Integer
- Priority, NCX orders, Total orders (t30)

#### Decimal/Float
- NCX rate, Star Rating Amazon
- Return rate Amazon, Return Rate B2B
- Result 1 (rr%), Result 2 (rr%)

#### Currency
- Cost of Refunds (Annualized)
- 12m Savings Captured (based on rr% reduction)

### Date Columns
- Date Action Taken
- Follow Up Date
- Result Check Date 1
- Result 2 Date
- Top Issue(s) Change Date

---

## Use Cases for Each Version

### Leadership Version Use Cases
1. **Executive Decision Making**: Priority and financial data for resource allocation
2. **Financial Planning**: Cost analysis and ROI tracking (refund costs, savings)
3. **Strategic Planning**: Order volume trends and priority ranking
4. **Internal Operations**: Case status tracking and workflow management
5. **Complete Analysis**: Full picture including sensitive business metrics

### Company Wide Version Use Cases
1. **Quality Team Collaboration**: Share quality issues without financial exposure
2. **Cross-Functional Visibility**: Product development, listing managers can see issues
3. **Vendor Communication**: Can be shared with suppliers for quality improvement
4. **Training**: Use as examples without revealing sensitive business data
5. **Departmental Reporting**: Quality metrics visible to relevant teams

---

## Recommendations for Manual Entry Form

### Required Fields (Cannot be empty)
- Product name
- SKU
- Return rate Amazon (or Return Rate B2B)
- Top Issue(s)

### Optional but Important
- ASIN (if Amazon product)
- Fulfilled by (FBA/FBM)
- NCX rate, NCX orders
- Star Rating Amazon

### Leadership-Only Fields (Show only for authorized users)
- Priority
- Total orders (t30)
- Flag Source 1
- Cost of Refunds (Annualized)
- 12m Savings Captured
- Case Status

### Auto-populated Fields
- Date Action Taken (current date when action entered)
- Result Check Date 1, Result 2 Date (when results entered)
- Top Issue(s) Change Date (when issue description updated)
- Follow Up Date (can be calculated or manual)

### Validation Rules
- Return rates should be decimal between 0 and 1 (0-100%)
- Star ratings should be between 0 and 5
- Dates should be in proper format
- Financial fields should be currency format
- Boolean fields (Notified?) should be Yes/No or True/False

---

## Export Functionality Specifications

### Two Export Modes Required

#### 1. Leadership Export
- **File Name**: `Tracker_ Priority List (Leadership).xlsx`
- **Columns**: All 31 columns
- **Access Control**: Leadership/Admin only
- **Includes**: Financial data, priority, case status

#### 2. Company Wide Export
- **File Name**: `Company Wide Quality Tracker.xlsx`
- **Columns**: 25 columns (excludes 6 sensitive columns)
- **Access Control**: All authorized users
- **Excludes**: Priority, Total orders, Flag Source 1, Financial columns, Case Status

### Export Features to Implement
1. **Column Order Preservation**: Maintain exact column order as shown
2. **Sheet Structure**: Include both "Tracker" and "Comments" sheets
3. **Data Formatting**:
   - Currency: $XX,XXX.XX format
   - Decimals: 4 decimal places for rates
   - Dates: YYYY-MM-DD or MM/DD/YYYY
4. **Header Row**: Bold, possibly colored background
5. **Column Widths**: Auto-fit or predefined widths
6. **Filter Row**: Enable Excel auto-filter on header row

---

## Technical Notes

### Current File Status
- Both files currently contain only header rows (no data)
- Both files have 2 sheets: main tracker + comments sheet
- Files use standard .xlsx format (Office Open XML)

### Data Relationship
- Company Wide is a true subset of Leadership
- No columns exist in Company Wide that aren't in Leadership
- Column names are identical between files (for shared columns)
- Data should be consistent between both versions when exported

### Security Considerations
- Leadership version contains sensitive financial data
- Total order volume could reveal business performance
- Priority reveals strategic focus
- Case Status shows internal workflow
- Access controls must prevent unauthorized export of Leadership version
