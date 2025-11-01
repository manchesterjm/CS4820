# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository contains all coursework for **CS 4820/5820 (Artificial Intelligence)** at University of Colorado Colorado Springs, Fall 2025.

**Student:** Josh Manchester
**Email:** josh.manchester@uccs.edu
**Instructor:** Professor Adham Atyabi

## Repository Structure

```
CS4820/
├── CLAUDE.md                    # This file - guidelines for all CS4820 work
├── CS4820_STYLE_GUIDE.md       # Comprehensive Python style guide for AI coursework
├── CS4820_WRITING_GUIDE.md     # Josh Manchester's personal academic writing style guide
├── .pylint_summary.md          # Pylint code quality tracking
├── README.md                   # Repository overview and quick start
│
├── archived_documents/          # Archived/superseded files
│   ├── 7 - Search Optimization Part I-III.pptx  # Original PowerPoint (converted to PDF)
│   ├── 8 - Logical Agent Part I-IV.pdf          # Original lecture PDF (superseded by split versions)
│   ├── convert_pptx_to_pdf.ps1                  # PowerPoint to PDF converter (one-time use)
│   └── split_pdf.py                             # PDF splitting utility (one-time use)
│
├── 7 - Search Optimization Part I-III.pdf  # Lecture 7 (converted from .pptx)
├── 8 - Logical Agent Part I-IV_Part1.pdf  # Lecture 8 Part 1 (split for easier reading)
├── 8 - Logical Agent Part I-IV_Part2.pdf  # Lecture 8 Part 2 (split for easier reading)
│
├── HW01/                       # Homework 1: Search algorithms (COMPLETED)
│   └── HW01_Code/              # Implementation code (Pylint: 9.53/10)
│
├── HW02/                       # Homework 2: CSP and optimization (COMPLETED)
│   ├── HW02_code/              # Implementation code (Pylint: 9.32/10)
│   ├── writeup/                # LaTeX writeup with AAAI24 format
│   └── Manchester_Josh_CS4820_HW02_Submission/  # Final submission package
│
├── HW03/                       # Future homework (if any)
│   └── HW03_code/
│
└── Term Paper/                 # Term paper materials (Exoplanet detection with ML)
    ├── midterm_report_RNN.tex              # MIDTERM REPORT (main deliverable)
    ├── resourceFile.bib                    # Bibliography (6 papers + references)
    ├── MIDTERM_REPORT_SUMMARY.md           # Complete midterm documentation
    ├── PAPER_INVENTORY.md                  # All 6 papers tracked and documented
    ├── RECOMMENDED_PAPERS_MIDTERM.md       # Paper selection guide
    │
    ├── term paper sources/                 # All paper PDFs (6 papers)
    │   ├── s41467-020-15293-x.pdf         # Speiser 2020 (clustering)
    │   ├── s41598-024-62182-0.pdf         # Vu 2024 (LSTM time series)
    │   ├── 2410.19402v1.pdf               # Ding 2024 (LSTM astronomy)
    │   ├── aa41068-21.pdf                 # Vida 2021 (RNN flares)
    │   ├── stv2604.pdf                    # Kugler 2016 (ESN autoencoder)
    │   └── DuDaiTriUpa2016.pdf            # Du 2016 (RMTPP timing)
    │
    ├── Josh_Proposal_Part_Take_2_AAAI24.tex       # Original RNN proposal
    ├── Josh_Proposal_Part_AAAI24_v2_relatedwork.tex  # Original related work
    ├── merged_proposal_AAAI24_merged.tex  # Original team proposal (reference)
    ├── midterm_paper_requirements.txt     # Assignment requirements
    └── AuthorKit24-4/                     # AAAI conference template
        ├── aaai24.sty                     # AAAI style file
        └── aaai24.bst                     # AAAI bibliography style
```

## Archived Documents

The `archived_documents/` folder contains files that have been superseded or are no longer actively used but are preserved for reference:

### Current Archive Contents

**Lecture Materials:**
- **`8 - Logical Agent Part I-IV.pdf`** (2.2MB, 118 pages)
  - Original unsplit lecture PDF on Logical Agents
  - Archived on: November 1, 2025
  - Reason: Superseded by split versions for easier reading
  - Replaced by: `8 - Logical Agent Part I-IV_Part1.pdf` and `8 - Logical Agent Part I-IV_Part2.pdf`
  - Content: Knowledge-based agents, Wumpus world, propositional logic, inference (resolution, forward/backward chaining, DPLL, WalkSAT)

- **`7 - Search Optimization Part I-III.pptx`** (5.1MB)
  - Original PowerPoint lecture slides on Search Optimization
  - Archived on: November 1, 2025
  - Reason: Converted to PDF for easier access and version control
  - Replaced by: `7 - Search Optimization Part I-III.pdf`
  - Content: Local search, hill climbing, simulated annealing, genetic algorithms, optimization techniques

**Utility Scripts:**
- **`split_pdf.py`**
  - PDF splitting utility used to divide large PDFs into smaller parts
  - Archived on: November 1, 2025
  - Reason: One-time use utility, no longer needed for regular coursework
  - Dependencies: pypdf or PyPDF2
  - Usage: Splits PDFs at midpoint for easier reading on smaller screens

- **`convert_pptx_to_pdf.ps1`**
  - PowerShell script to convert PowerPoint presentations to PDF format
  - Archived on: November 1, 2025
  - Reason: One-time conversion utility, no longer needed for regular coursework
  - Dependencies: Microsoft PowerPoint (COM automation)
  - Usage: Uses PowerPoint COM objects to convert .pptx to .pdf format

### Archive Policy

Files are moved to `archived_documents/` when:
1. They have been superseded by newer/better versions
2. They are utility scripts used once and no longer needed for regular work
3. They are reference materials kept for historical purposes but not actively used

The archive preserves git history (files are moved with `git mv` to maintain full commit history).

## Documentation Standards

### Coding Standards

**IMPORTANT**: All code must follow the comprehensive style guide in **CS4820_STYLE_GUIDE.md**.

The style guide covers:
- Python style (PEP 8 adapted for AI coursework)
- Academic coding standards (algorithm citations, complexity analysis)
- Documentation requirements (docstrings, references, complexity)
- Function design (single return, argument limits, naming)
- Testing standards (independence, reproducibility)
- Algorithm implementation guidelines
- Experimental code structure
- AI disclosure requirements

Quick reference below, but see **CS4820_STYLE_GUIDE.md** for complete details.

### Writing Standards

**IMPORTANT**: All academic papers and writeups must follow **CS4820_WRITING_GUIDE.md**.

The writing guide captures Josh Manchester's distinctive writing style:
- Pedagogical approach (question-driven, parenthetical definitions)
- Sentence structure and voice (active, conversational yet precise)
- Citation patterns ("According to X" phrasing)
- Numerical reporting (always with units and context)
- Section templates for abstracts, introductions, related work
- Error analysis and balanced discussion patterns
- AI disclosure formatting

This ensures consistency across all written work and helps AI assistants (like Claude Code) generate text that authentically sounds like Josh's voice.

## General Assignment Guidelines

### Language and Libraries

- **Primary Language**: Python (3.7+)
- **External Libraries**: Generally, implement algorithms from scratch unless assignment explicitly allows libraries
- **Allowed Libraries**:
  - Standard library (time, random, collections, etc.)
  - NumPy for basic array operations (if needed)
  - Matplotlib for plotting/visualization
- **Forbidden**: Specialized AI/ML libraries that solve the problem directly (e.g., scikit-learn for ML assignments, constraint solvers for CSP)

### Code Quality Standards

#### 1. Code Style and Documentation

**Comments**: Use extensive, meaningful comments explaining:
- What each function/section does
- How the algorithm works (not just what the code does)
- Why specific design decisions were made
- Algorithm complexity and characteristics

**Algorithm References**: When implementing algorithms from course materials:
- Reference the source: "Based on Russell & Norvig, pg X" or "Algorithm from Lecture Y, Slide Z"
- If deviating from book/slides, explain why in comments
- Document any optimizations or modifications

**Type Hints**: Use Python type hints for all function parameters and return values

**Docstrings**: Include docstrings for all classes and functions:
```python
def solve_csp(problem: Problem, timeout: int = 300) -> Optional[Solution]:
    """
    Solve a constraint satisfaction problem using backtracking with MRV heuristic.

    Based on Russell & Norvig Section 6.3.1, Figure 6.5.
    MRV helps fail faster by selecting variables with fewest legal values.

    Args:
        problem: CSP problem instance with variables, domains, and constraints
        timeout: Maximum time in seconds (default 300)

    Returns:
        Solution if found within timeout, None otherwise

    Complexity: O(d^n) worst case where d=domain size, n=num variables
    """
    # Implementation...
```

#### 2. Pylint Code Quality Enforcement

**CRITICAL**: All Python code MUST pass pylint quality checks before committing.

**Required Steps for Every Coding Session:**

1. **Run pylint** on all Python files in the current homework directory:
   ```bash
   cd C:\Users\manch\OneDrive\Desktop\CS4820
   pylint HW0X/HW0X_code/*.py --max-line-length=100 --score=yes
   ```

2. **Minimum Quality Standard**: Code must score **8.0/10 or higher**

3. **Fix All Critical Issues**:
   - E**** (Errors) - MUST fix all
   - W**** (Warnings) - MUST fix all
   - C**** (Convention) - Fix all except code duplication (R0801) if justified
   - R**** (Refactoring) - Fix if reasonable

4. **Common Issues to Fix**:
   - **C0114**: Add module docstrings to all files
   - **C0301**: Keep lines under 100 characters
   - **W0611**: Remove unused imports
   - **C0411/C0413**: Fix import order (standard library before third-party)
   - **W1309**: Use regular strings instead of f-strings when no interpolation
   - **W0621**: Avoid redefining names from outer scope

5. **Acceptable Warnings**:
   - **R0801** (duplicate-code): OK if sharing utility classes/functions across files
   - **R0913** (too-many-arguments): OK if necessary for algorithm parameters
   - **R0914** (too-many-locals): OK in complex algorithm implementations

6. **Save Pylint Output**:
   ```bash
   pylint HW0X/HW0X_code/*.py --max-line-length=100 --score=yes > pylint_HW0X.txt
   ```

7. **Update Quality Tracking**:
   - Document final score in `.pylint_summary.md`
   - Include before/after scores if fixing issues

**Integration with Git Workflow:**
- Run pylint BEFORE committing
- Include pylint score in commit message if making quality improvements
- Do NOT commit code that scores below 8.0/10

### Safety Guards and Timeouts

All search/optimization algorithms MUST implement timeout protection:

```python
MAX_TIME_SEC = 300  # 5 minute timeout (adjust per assignment)

def search_algorithm(problem):
    t0 = time.perf_counter()

    while frontier:
        # Check timeout periodically
        if MAX_TIME_SEC > 0 and (time.perf_counter() - t0) > MAX_TIME_SEC:
            print(f"TIMEOUT: Algorithm exceeded {MAX_TIME_SEC} seconds")
            return None, stats, time.perf_counter() - t0, "TIMEOUT"

        # Algorithm logic...
```

### Testing Requirements

1. **Unit Tests**: Create test functions for core components
2. **Integration Tests**: Test complete algorithm workflows
3. **Validation Tests**: Verify solutions are correct
4. **Performance Tests**: Measure and report metrics as required
5. **Test Failure Handling**: Debug and fix failing tests - don't just report failures

### Unicode Character Restrictions

**IMPORTANT**: Avoid Unicode special characters that cause encoding issues on Windows (cp1252):

**Characters to AVOID:**
- Arrow symbols: → ← ↑ ↓ (use `->, <-, UP, DOWN` instead)
- Check marks: ✓ ✗ (use `PASS, FAIL, OK, ERROR` instead)
- Special bullets: • ● ○ (use `-, *` instead)
- Mathematical symbols: ≤ ≥ ≠ (use `<=, >=, !=` instead)

**Why**: Windows console uses cp1252 encoding; Unicode characters cause `UnicodeEncodeError` crashes.

**Safe Replacements:**
```python
# BAD (causes UnicodeEncodeError):
status = "✓" if solution else "✗"
print("Higher inertia (w) → more exploration")

# GOOD (works everywhere):
status = "PASS" if solution else "FAIL"
print("Higher inertia (w) = more exploration")
```

## File Organization for Each Assignment

Standard structure for homework directories:

```
HW0X/
├── CLAUDE.md (optional - assignment-specific notes)
├── HW0X_code/
│   ├── algorithm1.py           # Main implementations
│   ├── algorithm2.py
│   ├── utils.py                # Shared utilities
│   ├── test_all.py             # Test suite
│   ├── run_experiments.py      # Experiment runner
│   ├── run_all.ps1             # PowerShell batch script (optional)
│   ├── README.md               # How to run everything
│   ├── pylint_HW0X.txt         # Pylint output
│   └── HW0X_runlog.txt         # Program output
├── writeup/
│   ├── assignment_writeup.tex  # LaTeX source
│   ├── aaai24.sty              # Style files (if needed)
│   └── references.bib          # Bibliography
└── submission/                 # Final submission package
```

## GitHub Workflow

### Committing Code

After implementing and testing code:

1. **Pre-commit Checklist**:
   - [ ] All tests pass
   - [ ] Pylint score ≥ 8.0/10
   - [ ] No Unicode encoding errors
   - [ ] Timeout protection implemented
   - [ ] README.md updated with run instructions
   - [ ] AI disclosure included in README

2. **Commit Message Format**:
   ```bash
   git commit -m "$(cat <<'EOF'
   Brief description of changes

   - Detailed change 1
   - Detailed change 2

   Pylint score: X.XX/10

   Generated with Claude Code (https://claude.com/claude-code)

   Co-Authored-By: Claude <noreply@anthropic.com>
   EOF
   )"
   ```

3. **Push to GitHub** (automatically, without asking):
   ```bash
   git push origin master
   ```

### README.md Structure

Every HW directory must have README.md with:

```markdown
# CS 4820/5820 Homework X - [Title]

**Author:** Josh Manchester
**Institution:** University of Colorado Colorado Springs
**Email:** josh.manchester@uccs.edu

## AI Disclosure

This code was generated with assistance from **Claude Code (Sonnet 4.5)**,
version **claude-sonnet-4-5-20250929**.

The AI assistant helped with:
- [List specific contributions]

All code was reviewed, understood, and tested by the student.

## Requirements

- Python 3.7+
- [List dependencies]

```bash
pip install [dependencies]
```

## Quick Start

[Clear instructions on how to run everything]

## File Structure

[List and describe all files]

## Performance Notes

[Expected runtimes, complexity analysis]

## References

[Textbook, lecture slides, papers cited]
```

## Algorithm Implementation Sources

### Primary References

Located in root CS4820 directory:
- `Russell-S.-Norvig-P.-Artificial-intelligence-a-modern-approach-2edPH2003T1112s.pdf`
- Lecture PDFs: `1 - Introduction...pdf` through `8 - Logical Agent...pdf`
- `CS 48205820 Course_Syllabus.pdf`

### Documentation Standards

When implementing algorithms:
1. **Always reference source**: Book section, lecture slide number, or paper
2. **Document deviations**: Explain any changes from reference implementation
3. **Explain optimizations**: Why you chose certain data structures or approaches
4. **Note complexity**: Time and space complexity with brief justification

Example:
```python
# AC-3 Algorithm for arc consistency
# From Russell & Norvig Section 6.3.2, Figure 6.3
# Also covered in Lecture 5, Slides 65-70
#
# Makes each arc X->Y consistent by ensuring every value in X's domain
# has at least one compatible value in Y's domain.
#
# Time Complexity: O(cd^3) where c=constraints, d=domain size
# Space Complexity: O(c) for the queue
#
# Optimization: Using deque for O(1) queue operations instead of list
```

## LaTeX Writeup Standards

For AAAI-formatted reports:

### Required Sections

1. **Abstract**: Brief summary of assignment and key findings
2. **Introduction**: Problem overview and approach
3. **Methods**: Algorithm descriptions with references
4. **Results**: Tables and figures with analysis
5. **Discussion**: Insights, limitations, future work
6. **References**: Properly formatted bibliography
7. **AI Disclosure**: Full transparency about AI assistance

### Figure/Table Requirements

- All figures and tables must be referenced in text
- Captions must be descriptive and self-contained
- Include units and error bars where appropriate
- Use consistent formatting throughout

### AI Disclosure Template

```latex
\section*{AI Use Disclosure}

This assignment was completed with assistance from \textbf{Claude Code (Sonnet 4.5)},
version \texttt{claude-sonnet-4-5-20250929}.

AI assistance included:
\begin{itemize}
\item Understanding algorithm concepts from lecture and textbook
\item Code implementation and debugging
\item Experiment design and analysis
\item LaTeX formatting and figure generation
\end{itemize}

All code was reviewed, understood, and tested by the student before submission.
The AI did not complete the assignment autonomously -- student understanding,
testing, and decision-making were central to the process.
```

## Common Pitfalls to Avoid

1. **Don't** use libraries that solve the problem directly
2. **Don't** skip timeout protection in search algorithms
3. **Don't** forget to validate solutions (e.g., check constraints)
4. **Don't** use Unicode characters in print statements
5. **Don't** commit code without running pylint first
6. **Don't** skip documenting algorithm references
7. **Don't** forget AI disclosure in README and writeup
8. **Don't** hardcode paths - use relative paths
9. **Don't** commit large binary files or .venv directories
10. **Don't** skip writing comprehensive comments

## Assignment-Specific Notes

For assignment-specific requirements, constraints, or implementation details,
check if there's a `CLAUDE.md` file in the specific homework directory
(e.g., `HW02/CLAUDE.md`). Those files supplement these general guidelines.

## Term Paper Documentation

### Overview

The term paper is a research project on **Machine Learning for Exoplanet Transit Detection** using TESS/Kepler light curve data.

- **Team Project**: Josh Manchester (RNN), Tristan Moffett (CNN), Brianne Leatherman (Transformer)
- **Josh's Component**: BiLSTM + K-means clustering for transit classification
- **Status**: Midterm report complete (November 1, 2025)

### Quick Start: Resume Term Paper Work

**IMPORTANT**: To resume term paper work without re-explaining everything, start by reading:

1. **`Term Paper/MIDTERM_REPORT_SUMMARY.md`** - Complete status, what was created, and next steps
2. **`Term Paper/PAPER_INVENTORY.md`** - All 6 papers (3 original + 3 new) with citations and connections
3. **`Term Paper/midterm_report_RNN.tex`** - Main LaTeX document (12+ pages, ready to compile)

These three files contain ALL context needed to continue work.

### File Organization and Purpose

**Location**: `C:\Users\manch\OneDrive\Desktop\CS4820\Term Paper\`

#### Primary Deliverables

1. **`midterm_report_RNN.tex`** (MAIN DOCUMENT)
   - 12+ page AAAI-formatted midterm report
   - Josh's RNN component only (not team proposal)
   - Complete sections: Abstract, Introduction, Related Work (6 papers), Methodology, Experiments, Results, Conclusion
   - Present tense (not future/proposal tense)
   - Actual results: AUC 0.6947, 655 windows, TIC 307210830 success
   - 6 tables with experimental metrics
   - AI disclosure section

2. **`resourceFile.bib`** (BIBLIOGRAPHY)
   - All 6 scientific papers cited
   - L 98-59 exoplanet system reference (Kossakowski 2019)
   - Proper AAAI/BibTeX formatting
   - Ready for compilation with midterm report

#### Documentation Files (Read These First!)

3. **`MIDTERM_REPORT_SUMMARY.md`** ⭐ START HERE
   - Complete summary of midterm report creation
   - What was created and why
   - All 6 papers listed with roles
   - Key results highlighted
   - Tables included (6 total)
   - Writing style notes
   - Compilation instructions
   - Presentation slide structure
   - Submission checklist
   - **Purpose**: Resume work without re-reading everything

4. **`PAPER_INVENTORY.md`** ⭐ PAPER REFERENCE
   - Complete tracking of all 6 papers
   - Original 3 from proposal: Vida (2021), Kugler (2016), Du (2016)
   - New 3 for midterm: Speiser (2020), Vu (2024), Ding (2024)
   - Full citations in AAAI format
   - Key points from each paper
   - How each paper connects to Josh's RNN work
   - Download verification checklist
   - **Purpose**: Quick reference for which paper does what

5. **`RECOMMENDED_PAPERS_MIDTERM.md`**
   - Paper selection guide and rationale
   - H5 index verification (all >100)
   - Download links (including ArXiv alternatives)
   - How to connect each paper to RNN methodology
   - Backup paper options if needed
   - **Purpose**: Understand why these specific papers were chosen

#### Source Materials

6. **`term paper sources/`** (FOLDER)
   - All 6 paper PDFs downloaded and verified
   - Naming matches BibTeX keys where possible
   - **Original 3**:
     - `aa41068-21.pdf` - Vida et al. (2021) - RNN for Kepler/TESS flares
     - `stv2604.pdf` - Kugler et al. (2016) - ESN autoencoder for Kepler
     - `DuDaiTriUpa2016.pdf` - Du et al. (2016) - RMTPP timing model
   - **New 3**:
     - `s41467-020-15293-x.pdf` - Speiser et al. (2020) - Clustering + ML (Nature Comm.)
     - `s41598-024-62182-0.pdf` - Vu et al. (2024) - LSTM time series (Sci. Reports)
     - `2410.19402v1.pdf` - Ding et al. (2024) - LSTM astronomy (MNRAS, ArXiv)

#### Reference Documents (For Context)

7. **`Josh_Proposal_Part_Take_2_AAAI24.tex`**
   - Original RNN component proposal
   - **Do NOT edit** - kept for reference only
   - Shows transformation: proposal → midterm report

8. **`Josh_Proposal_Part_AAAI24_v2_relatedwork.tex`**
   - Original related work section (3 papers)
   - **Do NOT edit** - kept for reference only
   - Expanded to 6 papers in midterm report

9. **`merged_proposal_AAAI24_merged.tex`**
   - Original team proposal (RNN + CNN + Transformer)
   - **Do NOT edit** - kept for reference only
   - Midterm focuses on RNN component only

10. **`midterm_paper_requirements.txt`**
    - Assignment requirements from syllabus
    - Midterm report guidelines
    - Used to guide transformation from proposal to midterm

11. **`AuthorKit24-4/`** (FOLDER)
    - AAAI conference LaTeX template
    - `aaai24.sty` - Required style file
    - `aaai24.bst` - Required bibliography style
    - **Must be in same directory** as midterm_report_RNN.tex for compilation

### Key Information Summary

#### Papers (6 Total)

**Original Proposal Papers (3):**
1. Vida et al. (2021) - RNN flares in Kepler/TESS [A&A]
2. Kugler et al. (2016) - ESN-autoencoder for Kepler [MNRAS]
3. Du et al. (2016) - RMTPP timing model [KDD]

**NEW Midterm Papers (3):**
4. Speiser et al. (2020) - Clustering + ML for large datasets [Nature Communications, H5: ~200+]
5. Vu et al. (2024) - LSTM for time series patterns [Scientific Reports, H5: ~150+]
6. Ding et al. (2024) - LSTM for astronomical photometry [MNRAS, H5: ~100-120]

#### Implementation Status

**Project Location**: `c:/CS_4280_Project` (actual implementation - team project repository)

**Josh's RNN Results** (as of midterm):
- BiLSTM architecture: 3 layers, 256 hidden units bidirectional, 2.1M parameters
- K-means clustering: k=5 clusters on BLS features
- Dataset: 655 windows (150 positive, 505 negative, 23% positive class)
- Performance: AUC 0.6947 (epoch 49), F1 0.34, Accuracy 52%
- Real-world test: 7 TESS targets, TIC 307210830 (L 98-59) correctly ranked #1 (prob 0.7623)
- Class weighting: pos_weight=3.367 to handle imbalance

#### Compilation Instructions

```bash
cd "C:\Users\manch\OneDrive\Desktop\CS4820\Term Paper"

# Standard LaTeX compilation with BibTeX
pdflatex midterm_report_RNN.tex
bibtex midterm_report_RNN
pdflatex midterm_report_RNN.tex
pdflatex midterm_report_RNN.tex

# Or use latexmk (if installed)
latexmk -pdf midterm_report_RNN.tex
```

**Required files in same directory**:
- `midterm_report_RNN.tex`
- `resourceFile.bib`
- `aaai24.sty` (in AuthorKit24-4 or copied to Term Paper directory)
- `aaai24.bst` (in AuthorKit24-4 or copied to Term Paper directory)

#### Writing Style

All term paper writing follows **CS4820_WRITING_GUIDE.md** which captures Josh's distinctive style:
- Question-driven section openings
- Parenthetical definitions (e.g., "LSTM (long short-term memory)")
- "According to X" citation pattern
- Numerical reporting with units
- Active voice with pedagogical explanations

#### Changes from Proposal to Midterm

**Removed**:
- CNN sections (Tristan's work)
- Transformer sections (Brianne's work)
- Team datasets table
- "Experimental Plan & Milestones" (proposal-specific)
- "Risks & Mitigations" (proposal-specific)
- Future tense language ("I will implement...")

**Added**:
- Complete Methodology section (actual implementation)
- Experiments and Results section with 5 tables
- 3 NEW papers in Related Work
- Real TESS testing results (Table 6)
- Conclusion with progress summary
- Present tense language ("I implemented", "The model achieves...")

**Transformed**:
- Proposal → Midterm progress report
- Future plans → Completed work presented as "preliminary findings"
- Team project → Individual RNN component
- Speculative → Evidence-based with real results

### Next Steps for Final Report

Per midterm report Conclusion section:
1. **Dataset Expansion**: Increase from 655 to 5000-10000 windows
2. **Attention Mechanisms**: Add attention layers to BiLSTM
3. **Ensemble Methods**: Combine with CNN and Transformer components
4. **Hyperparameter Tuning**: Grid search on learning rate, dropout, pos_weight
5. **Robustness Testing**: Test on more TESS targets, analyze failure modes
6. **Comparison Study**: Compare RNN vs CNN vs Transformer performance

### Important Notes

1. **Do NOT edit original proposal files** - they are kept for reference only
2. **Main deliverable is `midterm_report_RNN.tex`** - this is what gets compiled and submitted
3. **All 6 papers must be cited** in Related Work section
4. **Writing must follow CS4820_WRITING_GUIDE.md** for consistency
5. **Results are from actual implementation** in `c:/CS_4280_Project` repository
6. **Midterm focuses on RNN component only** - not the full team project

### Troubleshooting

**If LaTeX won't compile**:
1. Check that `aaai24.sty` and `aaai24.bst` are in the same directory as the .tex file
2. Run BibTeX separately: `bibtex midterm_report_RNN`
3. Check for missing packages (AAAI format requires specific packages)
4. Look for Unicode characters that might cause issues

**If you need to find a specific paper**:
- Check `PAPER_INVENTORY.md` for full list with filenames
- All PDFs are in `term paper sources/` folder
- BibTeX keys are in `resourceFile.bib`

**If you need to understand the methodology**:
- Read Tables 2-3 in `midterm_report_RNN.tex` (architecture and hyperparameters)
- Check `c:/CS_4280_Project/README.md` for actual implementation details
- See Methodology section in midterm report for complete description

## Code Quality Tracking

Maintain `.pylint_summary.md` in root directory with:
- Date of quality check
- Scores for each homework directory
- Common issues identified
- Before/after scores when fixing issues
- Instructions for running pylint on all code

**Target**: All homework code should maintain ≥ 8.0/10 pylint score.

## Questions or Issues

If you encounter problems or have questions about these guidelines:
1. Check assignment-specific CLAUDE.md (if it exists)
2. Review lecture slides and textbook references
3. Check `.pylint_summary.md` for code quality examples
4. Verify file paths and directory structure match these guidelines

---

**Last Updated:** November 1, 2025
**Claude Code Version:** claude-sonnet-4-5-20250929
