# PowerShell script to show Excel structure with sample data format
$ErrorActionPreference = "Stop"

$file1 = "c:\Users\Alex\code\testenv\Tracker_ Priority List (Leadership).xlsx"
$file2 = "c:\Users\Alex\code\testenv\Company Wide Quality Tracker.xlsx"

Write-Host "==================================================="
Write-Host "EXCEL FILE STRUCTURE ANALYSIS"
Write-Host "==================================================="
Write-Host ""

# Leadership Sample Data
Write-Host "==================================================="
Write-Host "1. LEADERSHIP VERSION"
Write-Host "   File: Tracker_ Priority List (Leadership).xlsx"
Write-Host "==================================================="
Write-Host ""
Write-Host "Sheet 1: Tracker_ Priority List (Leaders)"
Write-Host "  - 31 columns total"
Write-Host "  - Currently 1 row (headers only, no data)"
Write-Host ""
Write-Host "Column Headers and Sample Data Format:"
Write-Host ""

$leadershipColumns = @(
    @{Name="Priority"; Sample="1"; Description="Ranking/priority level"},
    @{Name="Product name"; Sample="Vive Mobility Walker"; Description="Product name"},
    @{Name="Main Sales Channel (by Volume)"; Sample="Amazon"; Description="Primary sales channel"},
    @{Name="ASIN"; Sample="B07XAMPLE1"; Description="Amazon product identifier"},
    @{Name="SKU"; Sample="VMW-001"; Description="Stock keeping unit"},
    @{Name="Fulfilled by"; Sample="FBA"; Description="Fulfillment method (FBA/FBM)"},
    @{Name="NCX rate"; Sample="0.0234"; Description="Non-conformance rate"},
    @{Name="NCX orders"; Sample="45"; Description="Number of non-conforming orders"},
    @{Name="Total orders (t30)"; Sample="1923"; Description="[LEADERSHIP ONLY] Total orders in last 30 days"},
    @{Name="Star Rating Amazon"; Sample="4.5"; Description="Amazon star rating"},
    @{Name="Return rate Amazon"; Sample="0.0812"; Description="Amazon return rate"},
    @{Name="Return Rate B2B"; Sample="0.0345"; Description="B2B return rate"},
    @{Name="Flag Source 1"; Sample="High Return Rate"; Description="[LEADERSHIP ONLY] Primary flag reason"},
    @{Name="Return Badge Displayed Amazon"; Sample="Yes"; Description="Whether return badge is shown"},
    @{Name="Notification/Notes"; Sample="Customer reports handle issue"; Description="Notes about the issue"},
    @{Name="Top Issue(s)"; Sample="Handle durability"; Description="Main quality issues"},
    @{Name="Cost of Refunds (Annualized)"; Sample="$25,340"; Description="[LEADERSHIP ONLY] [FINANCIAL] Annual refund costs"},
    @{Name="12m Savings Captured (based on rr% reduction)"; Sample="$8,250"; Description="[LEADERSHIP ONLY] [FINANCIAL] Savings from improvements"},
    @{Name="Action Taken"; Sample="Redesigned handle mechanism"; Description="Actions taken to address"},
    @{Name="Date Action Taken"; Sample="2024-11-15"; Description="When action was taken"},
    @{Name="Listing Manager Notified?"; Sample="Yes"; Description="Whether listing manager was notified"},
    @{Name="Product Dev Notified?"; Sample="Yes"; Description="Whether product dev was notified"},
    @{Name="Flag Source"; Sample="Analytics"; Description="How issue was flagged"},
    @{Name="Follow Up Date"; Sample="2025-01-15"; Description="Scheduled follow-up date"},
    @{Name="Result 1 (rr%)"; Sample="0.0612"; Description="Return rate after first check"},
    @{Name="Result Check Date 1"; Sample="2024-12-15"; Description="First result check date"},
    @{Name="Result 2 (rr%)"; Sample="0.0498"; Description="Return rate after second check"},
    @{Name="Result 2 Date"; Sample="2025-01-10"; Description="Second result check date"},
    @{Name="Top Issue(s) Change"; Sample="Reduced handle complaints"; Description="Changes in reported issues"},
    @{Name="Top Issue(s) Change Date"; Sample="2025-01-10"; Description="When issue pattern changed"},
    @{Name="Case Status"; Sample="Monitoring"; Description="[LEADERSHIP ONLY] Current case status"}
)

foreach ($col in $leadershipColumns) {
    $indicator = ""
    if ($col.Description -match "LEADERSHIP ONLY") {
        $indicator = " ***"
    }
    Write-Host "  $($col.Name)$indicator"
    Write-Host "    Sample: $($col.Sample)"
    Write-Host "    Type: $($col.Description)"
    Write-Host ""
}

Write-Host "Sheet 2: Comments"
Write-Host "  - 1 column"
Write-Host "  - Used for additional notes/comments"
Write-Host ""

# Company Wide Sample Data
Write-Host "==================================================="
Write-Host "2. COMPANY WIDE VERSION"
Write-Host "   File: Company Wide Quality Tracker.xlsx"
Write-Host "==================================================="
Write-Host ""
Write-Host "Sheet 1: Company Wide Quality Tracker"
Write-Host "  - 25 columns total"
Write-Host "  - Currently 1 row (headers only, no data)"
Write-Host "  - This is a SUBSET of Leadership version"
Write-Host "  - EXCLUDES all financial and strategic data"
Write-Host ""
Write-Host "Excluded Columns (not in Company Wide):"
Write-Host "  1. Priority - [STRATEGIC]"
Write-Host "  2. Total orders (t30) - [STRATEGIC METRICS]"
Write-Host "  3. Flag Source 1 - [INTERNAL]"
Write-Host "  4. Cost of Refunds (Annualized) - [FINANCIAL]"
Write-Host "  5. 12m Savings Captured (based on rr% reduction) - [FINANCIAL]"
Write-Host "  6. Case Status - [INTERNAL TRACKING]"
Write-Host ""
Write-Host "All other columns match Leadership version exactly."
Write-Host ""

Write-Host "Sheet 2: Comments"
Write-Host "  - 1 column"
Write-Host "  - Used for additional notes/comments"
Write-Host ""

Write-Host "==================================================="
Write-Host "KEY DIFFERENCES SUMMARY"
Write-Host "==================================================="
Write-Host ""
Write-Host "Leadership version contains 6 additional columns:"
Write-Host ""
Write-Host "FINANCIAL DATA (sensitive):"
Write-Host "  - Cost of Refunds (Annualized)"
Write-Host "  - 12m Savings Captured (based on rr% reduction)"
Write-Host ""
Write-Host "STRATEGIC METRICS (internal):"
Write-Host "  - Priority"
Write-Host "  - Total orders (t30)"
Write-Host ""
Write-Host "INTERNAL TRACKING:"
Write-Host "  - Flag Source 1"
Write-Host "  - Case Status"
Write-Host ""
Write-Host "The Company Wide version is designed for broader"
Write-Host "distribution without exposing financial or strategic"
Write-Host "information to all staff members."
Write-Host ""
Write-Host "==================================================="

Write-Host ""
Write-Host "Analysis complete!"
