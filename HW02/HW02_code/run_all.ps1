# run_all.ps1
# Runs all CS 4820/5820 Homework 2 programs and saves output to log file

# Set working directory to HW02_code
Set-Location "C:\Users\manch\OneDrive\Desktop\CS4820\HW02\HW02_code"

# Pick python interpreter (venv if present, else system)
$venvPy = Join-Path $PWD ".venv\Scripts\python.exe"
if (Test-Path $venvPy) { $py = $venvPy } else { $py = "python" }

# Log file
$log = "HW02_runlog.txt"

# Set console and output encoding to UTF-8
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Initialize log file with UTF-8 BOM
$utf8NoBom = New-Object System.Text.UTF8Encoding $false
[System.IO.File]::WriteAllText($log, "", $utf8NoBom)

# Write header
"==== CS 4820/5820 HW02 Full Run ====" | Out-File $log -Append -Encoding UTF8
"Timestamp: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $log -Append -Encoding UTF8
"Python: $py" | Out-File $log -Append -Encoding UTF8
"" | Out-File $log -Append -Encoding UTF8

function Run-One($title, $file) {
  "---- $title ($file) ----" | Out-File $log -Append -Encoding UTF8
  "" | Out-File $log -Append -Encoding UTF8

  Write-Host "Running: $title..." -ForegroundColor Cyan

  try {
    # Set PYTHONIOENCODING to utf-8 for this script run
    $env:PYTHONIOENCODING = "utf-8"

    # Capture output and append to log
    & $py $file 2>&1 | Out-File $log -Append -Encoding UTF8

    Write-Host "  Completed: $title" -ForegroundColor Green

  } catch {
    "ERROR running $file : $($_.Exception.Message)" | Out-File $log -Append -Encoding UTF8
    Write-Host "  ERROR: $title - $($_.Exception.Message)" -ForegroundColor Red
  }
  "" | Out-File $log -Append -Encoding UTF8
}

# Run all programs in order

# Part A: Sudoku CSP
Write-Host "`n===== PART A: Sudoku CSP =====" -ForegroundColor Yellow
Run-One "Sudoku CSP (All Variants)" "sudoku_csp.py"

# Part B: n-Queens Minimum Conflicts
Write-Host "`n===== PART B: n-Queens Minimum Conflicts =====" -ForegroundColor Yellow
Run-One "n-Queens Minimum Conflicts" "nqueens_minconflicts.py"

# Part C1: PSO for Benchmarks
Write-Host "`n===== PART C1: PSO Benchmark Optimization =====" -ForegroundColor Yellow
Run-One "PSO on Rastrigin and Rosenbrock" "pso_benchmark.py"

# Part C2: PSO for Sudoku
Write-Host "`n===== PART C2: PSO for Sudoku =====" -ForegroundColor Yellow
Run-One "PSO for Sudoku" "pso_sudoku.py"

# Optional: Run full experiments
Write-Host "`n===== FULL EXPERIMENTAL RESULTS =====" -ForegroundColor Yellow
Write-Host "Do you want to run full experiments (generates report data)? [Y/N]" -ForegroundColor Yellow
$response = Read-Host
if ($response -eq 'Y' -or $response -eq 'y') {
    Run-One "Full Experiments" "run_experiments.py"
}

# Optional: Run test suite
Write-Host "`n===== TEST SUITE =====" -ForegroundColor Yellow
Write-Host "Do you want to run the test suite? [Y/N]" -ForegroundColor Yellow
$response = Read-Host
if ($response -eq 'Y' -or $response -eq 'y') {
    Run-One "Comprehensive Test Suite" "test_all.py"
}

"==== END ====" | Out-File $log -Append -Encoding UTF8
Write-Host "`nAll programs completed!" -ForegroundColor Green
Write-Host "Output saved to: $(Join-Path $PWD $log)" -ForegroundColor Green
Write-Host "`nTo view the log file:" -ForegroundColor Cyan
Write-Host "  type $log" -ForegroundColor Cyan
