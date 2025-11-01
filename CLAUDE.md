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
├── .pylint_summary.md          # Pylint code quality tracking
├── HW01/                       # Homework 1: Search algorithms
│   └── HW01_Code/              # Implementation code
├── HW02/                       # Homework 2: CSP and optimization
│   └── HW02_code/              # Implementation code
├── HW03/                       # Future homework (if any)
│   └── HW03_code/
└── Term Paper/                 # Term paper materials
```

## Coding Standards

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
