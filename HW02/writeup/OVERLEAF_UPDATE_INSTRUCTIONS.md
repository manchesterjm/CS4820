# Overleaf Update Instructions - CONVERGENCE PLOTS ADDED

## What Was Missing

The assignment requires: **"plot convergence curves, and discuss their effects"**

Your previous submission was missing PSO convergence plots showing how fitness improves over iterations.

## What's Been Added

✅ **2 new convergence plots** (PDF format, publication quality)
✅ **Convergence Analysis subsection** in Part C1
✅ **Analysis of convergence behavior** and parameter effects

---

## Files to Upload to Overleaf

Upload these **6 files total** (replace existing files if prompted):

1. **`assignment_writeup.tex`** ← UPDATED with convergence section
2. **`aaai24.sty`** (no changes)
3. **`aaai24.bst`** (no changes)
4. **`references.bib`** (no changes)
5. **`rastrigin_convergence.pdf`** ← NEW PLOT
6. **`rosenbrock_convergence.pdf`** ← NEW PLOT

---

## What Changed in assignment_writeup.tex

**Added new subsection after Part C1 Analysis (around line 333):**

```latex
\subsection{Convergence Analysis}

[Text analyzing convergence behavior]

\begin{figure}[h]
\centering
\includegraphics[width=0.9\columnwidth]{rastrigin_convergence.pdf}
\caption{PSO convergence on Rastrigin function...}
\label{fig:rastrigin-convergence}
\end{figure}

\begin{figure}[h]
\centering
\includegraphics[width=0.9\columnwidth]{rosenbrock_convergence.pdf}
\caption{PSO convergence on Rosenbrock function...}
\label{fig:rosenbrock-convergence}
\end{figure}
```

**This adds:**
- Convergence Analysis discussion
- Figure 7: Rastrigin convergence plot
- Figure 8: Rosenbrock convergence plot

---

## Steps to Update on Overleaf

### 1. Upload Files

In your Overleaf project:
- Click "Upload" button (top left)
- Select all 6 files listed above
- Confirm replacement if prompted

### 2. Compile PDF

**IMPORTANT:** Compile 2-3 times for bibliography!

- Click "Recompile" button **3 times**
- Or: Menu → "Recompile from scratch"

### 3. Verify Results

Check that the new PDF has:
- ✅ Page count increased from 7 to **8-9 pages** (2 new figures)
- ✅ New "Convergence Analysis" subsection in Part C1
- ✅ Two new plots showing convergence curves
- ✅ All citations still working (page 8-9 for References)

---

## What the Plots Show

### Rastrigin Convergence Plot
- **X-axis:** Iteration number (0-10)
- **Y-axis:** Best fitness value (log scale)
- **Lines:** 3 configurations (different colors)
- **Red dotted line:** Global minimum (0)

**Key insights:**
- Config 2 (swarm=50, w=0.5) consistently best
- Rapid initial convergence
- All configs converge within ~5 iterations

### Rosenbrock Convergence Plot
- Same format as Rastrigin
- Shows Config 2 superiority
- Demonstrates parameter tuning effects

---

## Expected Final PDF

**Page count:** 8-9 pages (was 7 pages)

**Page breakdown:**
- Pages 1-4: Parts A, B, C1 (with new convergence plots)
- Pages 5-6: Part C2, Conclusion, AI Disclosure, Code Examples start
- Pages 7-8: Code Examples (Figures 1-6)
- Page 8-9: References

**New content location:**
- Part C1 Convergence Analysis: End of page 3 / start of page 4
- Figure 7 (Rastrigin plot): Page 3 or 4
- Figure 8 (Rosenbrock plot): Page 4

---

## Troubleshooting

### "File not found: rastrigin_convergence.pdf"
→ Make sure you uploaded both PDF plot files to Overleaf

### "Undefined control sequence"
→ Recompile 2-3 times (LaTeX needs multiple passes)

### "Citation undefined"
→ Recompile 2-3 times to process bibliography

### Plots look blurry
→ They shouldn't - PDFs are vector format. If blurry, try re-uploading

### Page count didn't change
→ Check that you uploaded the NEW assignment_writeup.tex (not old one)

---

## After Compilation

Once you've successfully compiled the new PDF on Overleaf:

1. **Download the final PDF**
2. **Rename it:** `Manchester_Josh_CS4820_HW02_Writeup_FINAL.pdf`
3. I'll update the submission package with the new PDF

---

## Why This Was Critical

**Assignment requirement explicitly states:**
> "Run at least 3 trials, explore multiple parameter settings, **plot convergence curves**, and discuss their effects."

Without convergence plots, the submission would be **incomplete** and could lose points on Part C1 (3 points total).

**With plots:** ✅ Full credit potential for Part C (5.5 pts)
**Without plots:** ❌ Missing required deliverable

---

**Ready to upload to Overleaf now!**
