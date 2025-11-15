# PowerShell script to run all HW03 programs
# CS 4820/5820 - Homework 3: Logical Agents
# Author: Josh Manchester

Write-Host "=================================" -ForegroundColor Green
Write-Host "HW03: Logical Agents - Run All" -ForegroundColor Green
Write-Host "=================================" -ForegroundColor Green
Write-Host ""

# Output file
$outputFile = "HW03_runlog.txt"

Write-Host "Running all programs and saving output to $outputFile..." -ForegroundColor Cyan
Write-Host ""

# Run the main experiment runner and capture output
python run_experiments.py | Tee-Object -FilePath $outputFile

Write-Host ""
Write-Host "=================================" -ForegroundColor Green
Write-Host "Execution Complete!" -ForegroundColor Green
Write-Host "=================================" -ForegroundColor Green
Write-Host ""
Write-Host "Output saved to: $outputFile" -ForegroundColor Yellow
Write-Host ""
Write-Host "You can also run individual programs:" -ForegroundColor Cyan
Write-Host "  python propositional_logic.py    (Part A)" -ForegroundColor White
Write-Host "  python horn_inference.py         (Part B)" -ForegroundColor White
Write-Host "  python wumpus_agent.py           (Part C)" -ForegroundColor White
Write-Host "  python resolution.py             (Part D)" -ForegroundColor White
Write-Host "  python test_all.py               (Test suite)" -ForegroundColor White
Write-Host ""
