# Writeup Folder - Ready for Overleaf!

This folder contains your complete AAAI24 formatted LaTeX writeup with all code output examples.

## ✅ FIGURES COMPLETE - NO SCREENSHOTS NEEDED!

All figures have been created using LaTeX `verbatim` blocks (text-based code listings) instead of screenshots. This looks more professional in academic papers and has several advantages:
- Text is searchable and copyable
- Scales perfectly at any zoom level
- No image quality issues
- More professional appearance

## What You Need to Do

### 1. Upload to Overleaf

Upload only these 3 files to Overleaf:
1. **`assignment_writeup.tex`** - Main document with all content and figures
2. **`aaai24.sty`** - AAAI24 style file
3. **`references.bib`** - Bibliography

**NO IMAGE FILES NEEDED** - All figures are text-based verbatim blocks!

### 2. Compile on Overleaf

The document should compile automatically. It uses standard packages:
- `aaai24` - Conference style
- `times`, `helvet`, `courier` - Fonts
- `algorithm`, `algorithmic` - Algorithm formatting
- `amsmath`, `booktabs` - Math and tables

## Current Status

✅ LaTeX writeup complete with all real experimental data
✅ All 6 figures created using verbatim blocks (no images needed!)
✅ AI Disclosure section included
✅ Bibliography file created
✅ All contractions removed from text
✅ **READY TO UPLOAD TO OVERLEAF!**

## What the Figures Show

**Figure 1 - Sudoku CSP:** All 4 algorithm variants with times showing AC-3 is 3× faster

**Figure 2 - n-Queens n=8:** 5 trials, 100% success, average 16.6 steps

**Figure 3 - n-Queens n=25:** 5 trials, 100% success, average 60.6 steps (linear scaling!)

**Figure 4 - PSO Rastrigin:** 3 configurations, Config 2 best at 59.91 average

**Figure 5 - PSO Rosenbrock:** 3 configurations, Config 2 best at 645.59 average

**Figure 6 - PSO Sudoku:** 3 trials, best 6 violations, average 10.67 (did not solve)

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
