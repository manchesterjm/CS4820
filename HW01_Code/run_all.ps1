# run_all.ps1
# Fixed version with proper UTF-8 encoding handling

# go to your HW folder
Set-Location "C:\Users\manch\OneDrive\Desktop\CS4820\HW01_Code"

# pick python (venv if present, else system)
$venvPy = Join-Path $PWD ".venv\Scripts\python.exe"
if (Test-Path $venvPy) { $py = $venvPy } else { $py = "python" }

# log file
$log = "HW01_runlog.txt"

# Set console and output encoding to UTF-8
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Initialize log file with UTF-8 BOM
$utf8NoBom = New-Object System.Text.UTF8Encoding $false
[System.IO.File]::WriteAllText($log, "", $utf8NoBom)

# Write header
"==== CS 4820/5820 HW01 Full Run ====" | Out-File $log -Append -Encoding UTF8
"Timestamp: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $log -Append -Encoding UTF8
"Python: $py" | Out-File $log -Append -Encoding UTF8
"" | Out-File $log -Append -Encoding UTF8

function Run-One($title, $file) {
  "---- $title ($file) ----" | Out-File $log -Append -Encoding UTF8
  "" | Out-File $log -Append -Encoding UTF8
  
  try {
    # Set PYTHONIOENCODING to utf-8 for this script run
    $env:PYTHONIOENCODING = "utf-8"
    
    # Capture output and append to log
    & $py $file 2>&1 | Out-File $log -Append -Encoding UTF8
    
  } catch {
    "ERROR running $file : $($_.Exception.Message)" | Out-File $log -Append -Encoding UTF8
  }
  "" | Out-File $log -Append -Encoding UTF8
}

Run-One "BFS" "n_puzzle_BFS.py"
Run-One "DFS" "n_puzzle_Depth_Limited_DFS.py"
Run-One "IDS" "n_puzzle_IDS.py"
Run-One "BDS" "n_puzzle_BDS.py"
Run-One "N-Queens SA" "n_queens_SA.py"
Run-One "N-Queens GA" "n_queens_GA.py"
Run-One "A* (misplaced vs manhattan)" "n_puzzle_ASTAR.py"

"==== END ====" | Out-File $log -Append -Encoding UTF8
Write-Host "Saved to: $(Join-Path $PWD $log)" -ForegroundColor Green