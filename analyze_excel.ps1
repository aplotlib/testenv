# PowerShell script to analyze Excel files
param(
    [string]$FilePath
)

$ErrorActionPreference = "Stop"

try {
    # Create Excel COM object
    $excel = New-Object -ComObject Excel.Application
    $excel.Visible = $false
    $excel.DisplayAlerts = $false

    # Open workbook
    $workbook = $excel.Workbooks.Open($FilePath)

    Write-Host "File: $FilePath"
    Write-Host "Number of sheets: $($workbook.Worksheets.Count)"
    Write-Host ""

    # Iterate through each worksheet
    foreach ($worksheet in $workbook.Worksheets) {
        Write-Host "================================"
        Write-Host "Sheet: $($worksheet.Name)"
        Write-Host "================================"

        # Get used range
        $usedRange = $worksheet.UsedRange
        $rowCount = $usedRange.Rows.Count
        $colCount = $usedRange.Columns.Count

        Write-Host "Rows: $rowCount"
        Write-Host "Columns: $colCount"
        Write-Host ""

        # Get column headers (first row)
        Write-Host "Column Headers:"
        for ($col = 1; $col -le $colCount; $col++) {
            $header = $worksheet.Cells.Item(1, $col).Text
            Write-Host "  Column $col : $header"
        }
        Write-Host ""

        # Get sample data (rows 2-4)
        $sampleRows = [Math]::Min(4, $rowCount)
        if ($sampleRows -gt 1) {
            Write-Host "Sample Data (rows 2-$sampleRows):"
            for ($row = 2; $row -le $sampleRows; $row++) {
                Write-Host "  Row $row :"
                for ($col = 1; $col -le $colCount; $col++) {
                    $header = $worksheet.Cells.Item(1, $col).Text
                    $value = $worksheet.Cells.Item($row, $col).Text
                    Write-Host "    $header : $value"
                }
                Write-Host ""
            }
        }
        Write-Host ""
    }

    # Close workbook
    $workbook.Close($false)
    $excel.Quit()

    # Release COM objects
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($workbook) | Out-Null
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($excel) | Out-Null
    [System.GC]::Collect()
    [System.GC]::WaitForPendingFinalizers()

    Write-Host "Analysis complete!"

} catch {
    Write-Host "Error: $_"
    if ($excel) {
        $excel.Quit()
        [System.Runtime.Interopservices.Marshal]::ReleaseComObject($excel) | Out-Null
    }
    exit 1
}
