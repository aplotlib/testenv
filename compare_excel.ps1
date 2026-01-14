# PowerShell script to compare two Excel files
$ErrorActionPreference = "Stop"

$file1 = "c:\Users\Alex\code\testenv\Tracker_ Priority List (Leadership).xlsx"
$file2 = "c:\Users\Alex\code\testenv\Company Wide Quality Tracker.xlsx"

try {
    # Create Excel COM object
    $excel = New-Object -ComObject Excel.Application
    $excel.Visible = $false
    $excel.DisplayAlerts = $false

    # Open both workbooks
    $wb1 = $excel.Workbooks.Open($file1)
    $wb2 = $excel.Workbooks.Open($file2)

    # Get first sheets
    $sheet1 = $wb1.Worksheets.Item(1)
    $sheet2 = $wb2.Worksheets.Item(1)

    # Get column counts
    $cols1 = $sheet1.UsedRange.Columns.Count
    $cols2 = $sheet2.UsedRange.Columns.Count

    Write-Host "==================================================="
    Write-Host "COMPARISON: Leadership vs Company Wide"
    Write-Host "==================================================="
    Write-Host ""
    Write-Host "Leadership File Columns: $cols1"
    Write-Host "Company Wide File Columns: $cols2"
    Write-Host "Difference: $($cols1 - $cols2) additional columns in Leadership"
    Write-Host ""

    # Create arrays to store headers
    $headers1 = @()
    $headers2 = @()

    for ($col = 1; $col -le $cols1; $col++) {
        $headers1 += $sheet1.Cells.Item(1, $col).Text
    }

    for ($col = 1; $col -le $cols2; $col++) {
        $headers2 += $sheet2.Cells.Item(1, $col).Text
    }

    # Find columns in Leadership but NOT in Company Wide
    Write-Host "==================================================="
    Write-Host "COLUMNS ONLY IN LEADERSHIP VERSION:"
    Write-Host "==================================================="
    $leadershipOnly = @()
    foreach ($header in $headers1) {
        if ($header -and $header -notin $headers2) {
            $leadershipOnly += $header
            Write-Host "  - $header"
        }
    }
    Write-Host ""

    # Find columns in Company Wide but NOT in Leadership
    Write-Host "==================================================="
    Write-Host "COLUMNS ONLY IN COMPANY WIDE VERSION:"
    Write-Host "==================================================="
    $companyWideOnly = @()
    foreach ($header in $headers2) {
        if ($header -and $header -notin $headers1) {
            $companyWideOnly += $header
            Write-Host "  - $header"
        }
    }
    if ($companyWideOnly.Count -eq 0) {
        Write-Host "  (None - Company Wide is a subset)"
    }
    Write-Host ""

    # Find common columns
    Write-Host "==================================================="
    Write-Host "COMMON COLUMNS (in both files):"
    Write-Host "==================================================="
    $commonCols = @()
    foreach ($header in $headers1) {
        if ($header -and $header -in $headers2) {
            $commonCols += $header
            Write-Host "  - $header"
        }
    }
    Write-Host ""

    # Analyze Leadership-only columns for financial/private info
    Write-Host "==================================================="
    Write-Host "ANALYSIS OF LEADERSHIP-ONLY COLUMNS:"
    Write-Host "==================================================="
    foreach ($header in $leadershipOnly) {
        $type = "General"

        if ($header -match "Cost|Savings|Refund|\$|Financial|Budget|Revenue|Profit") {
            $type = "FINANCIAL"
        }
        elseif ($header -match "Priority|Total orders") {
            $type = "STRATEGIC/METRICS"
        }
        elseif ($header -match "Status") {
            $type = "STATUS/TRACKING"
        }

        Write-Host "  - $header [$type]"
    }
    Write-Host ""

    # Show column order comparison
    Write-Host "==================================================="
    Write-Host "COLUMN ORDER COMPARISON:"
    Write-Host "==================================================="
    Write-Host "Leadership Columns (in order):"
    for ($i = 0; $i -lt $headers1.Count; $i++) {
        $marker = ""
        if ($headers1[$i] -notin $headers2) {
            $marker = " [LEADERSHIP ONLY]"
        }
        Write-Host "  $($i+1). $($headers1[$i])$marker"
    }
    Write-Host ""
    Write-Host "Company Wide Columns (in order):"
    for ($i = 0; $i -lt $headers2.Count; $i++) {
        Write-Host "  $($i+1). $($headers2[$i])"
    }

    # Close workbooks
    $wb1.Close($false)
    $wb2.Close($false)
    $excel.Quit()

    # Release COM objects
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($wb1) | Out-Null
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($wb2) | Out-Null
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($excel) | Out-Null
    [System.GC]::Collect()
    [System.GC]::WaitForPendingFinalizers()

    Write-Host ""
    Write-Host "Comparison complete!"

} catch {
    Write-Host "Error: $_"
    if ($excel) {
        $excel.Quit()
        [System.Runtime.Interopservices.Marshal]::ReleaseComObject($excel) | Out-Null
    }
    exit 1
}
