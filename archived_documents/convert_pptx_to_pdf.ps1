# PowerShell script to convert PowerPoint to PDF
# Uses Microsoft PowerPoint COM automation

param(
    [Parameter(Mandatory=$true)]
    [string]$InputFile
)

# Get absolute path
$InputPath = Resolve-Path $InputFile
$OutputPath = [System.IO.Path]::ChangeExtension($InputPath, ".pdf")

Write-Host "Converting PowerPoint to PDF..."
Write-Host "Input:  $InputPath"
Write-Host "Output: $OutputPath"

try {
    # Create PowerPoint Application object
    $PowerPoint = New-Object -ComObject PowerPoint.Application
    $PowerPoint.Visible = [Microsoft.Office.Core.MsoTriState]::msoTrue

    # Open the presentation
    $Presentation = $PowerPoint.Presentations.Open($InputPath, $true, $true, $false)

    # Save as PDF (ppSaveAsPDF = 32)
    $Presentation.SaveAs($OutputPath, 32)

    # Close the presentation
    $Presentation.Close()

    # Quit PowerPoint
    $PowerPoint.Quit()

    # Release COM objects
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($Presentation) | Out-Null
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($PowerPoint) | Out-Null
    [System.GC]::Collect()
    [System.GC]::WaitForPendingFinalizers()

    Write-Host "SUCCESS: PDF created at $OutputPath" -ForegroundColor Green
    exit 0
}
catch {
    Write-Host "ERROR: Failed to convert PowerPoint to PDF" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red

    # Cleanup on error
    if ($PowerPoint) {
        $PowerPoint.Quit()
        [System.Runtime.Interopservices.Marshal]::ReleaseComObject($PowerPoint) | Out-Null
    }

    exit 1
}
