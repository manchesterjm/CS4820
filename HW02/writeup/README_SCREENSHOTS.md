# Adding Screenshots to the Writeup

This folder contains your AAAI24 formatted LaTeX writeup with placeholder figures for screenshots.

## What You Need to Do

### 1. Take Screenshots

Run the programs and capture screenshots of the output:

**Option A: From PowerShell Log File**
```powershell
cd ..\HW02_code
.\run_all.ps1
```
Then open `HW02_runlog.txt` and take screenshots of relevant sections.

**Option B: Run Programs Individually**
```bash
python sudoku_csp.py
python nqueens_minconflicts.py
python pso_benchmark.py
python pso_sudoku.py
```

### 2. Screenshot Recommendations

Based on the figure placeholders in `assignment_writeup.tex`:

**Figure 1 - Sudoku CSP (fig:sudoku-csp):**
- Capture output showing all 4 algorithm variants
- Should include: starting puzzle, algorithm names, times, solved puzzle
- Look for the section in sudoku_csp.py output with all variants

**Figure 2 - n-Queens n=8 (fig:nqueens-8):**
- Capture n=8 output from nqueens_minconflicts.py
- Should show: starting board with conflicts, solved board, steps, time

**Figure 3 - n-Queens n=25 (fig:nqueens-25):**
- Capture n=25 output from nqueens_minconflicts.py
- Should show: steps taken (~90), time (~22ms), verification

**Figure 4 - PSO Rastrigin (fig:pso-rastrigin):**
- Capture Rastrigin section from pso_benchmark.py
- Should show: all 3 configurations with best scores

**Figure 5 - PSO Rosenbrock (fig:pso-rosenbrock):**
- Capture Rosenbrock section from pso_benchmark.py
- Should show: convergence history showing improvement

**Figure 6 - PSO Sudoku (fig:pso-sudoku):**
- Capture output from pso_sudoku.py
- Should show: starting puzzle, iteration progress, final violations

### 3. Save Screenshots

Save your screenshots in this folder with descriptive names:
```
writeup/
├── fig1_sudoku_csp.png
├── fig2_nqueens_8.png
├── fig3_nqueens_25.png
├── fig4_pso_rastrigin.png
├── fig5_pso_rosenbrock.png
└── fig6_pso_sudoku.png
```

### 4. Add to LaTeX

Replace the TODO comments in `assignment_writeup.tex` with:

```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{fig1_sudoku_csp.png}
\caption{Sudoku CSP: All four algorithm variants...}
\label{fig:sudoku-csp}
\end{figure}
```

Do this for each of the 6 figures.

### 5. Upload to Overleaf

Upload these files to Overleaf:
1. `assignment_writeup.tex`
2. `aaai24.sty`
3. `references.bib`
4. All 6 screenshot images (`.png` files)

## LaTeX Compilation

The document uses the `graphicx` package which is already included in the preamble:
```latex
\usepackage{graphicx}
```

When you upload to Overleaf, it should compile automatically. If images don't show:
- Make sure image files are in the same folder as the .tex file
- Check that filenames match exactly (case-sensitive)
- Try recompiling

## Current Status

✅ LaTeX template created with all real experimental data
✅ Figure placeholders added with descriptive captions
✅ AI Disclosure section included
⚠️ **TODO:** Take screenshots and insert into figures
⚠️ **TODO:** Upload to Overleaf and compile PDF

## Quick Commands

From the `writeup/` folder:

```bash
# View the tex file
cat assignment_writeup.tex

# Run the programs to get fresh output
cd ../HW02_code
python sudoku_csp.py
python nqueens_minconflicts.py
python pso_benchmark.py
python pso_sudoku.py
```

Or use the PowerShell script:
```powershell
cd ../HW02_code
.\run_all.ps1
```

All output is saved to `HW02_runlog.txt` for easy screenshot capture!
