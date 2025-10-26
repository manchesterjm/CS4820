# CS 4820/5820 Homework 2 - Submission Package

**Student:** Josh Manchester
**Email:** josh.manchester@uccs.edu
**Course:** CS 4820/5820 - Artificial Intelligence
**Institution:** University of Colorado Colorado Springs
**Date:** October 26, 2025

---

## Submission Contents

This submission package contains all required materials for Homework 2: Constraint Satisfaction and Metaheuristic Optimization.

### 1. Writeup (PDF)
- **File:** `Manchester_Josh_CS4820_HW02_Writeup.pdf`
- **Format:** AAAI 2024 conference format
- **Pages:** 8-9 pages including references and convergence plots
- **Contents:**
  - Abstract and Introduction
  - Part A: Sudoku CSP with 4 algorithm variants
  - Part B: n-Queens with Minimum Conflicts
  - Part C1: PSO for benchmark optimization **with convergence plots**
  - Part C2: PSO applied to Sudoku
  - Conclusion and AI Use Disclosure
  - Code Output Examples (6 figures)
  - **Convergence Plots (2 figures - Rastrigin & Rosenbrock)**
  - References section

### 2. Source Code
All Python implementations in `HW02_code/` directory:

- **`sudoku_csp.py`** - Part A: Sudoku CSP solver
  - Basic Backtracking
  - Backtracking + MRV + LCV
  - Backtracking + Forward Checking
  - Backtracking + AC-3

- **`nqueens_minconflicts.py`** - Part B: n-Queens solver
  - Minimum Conflicts local search
  - Tested on n=8, 16, 25

- **`pso_benchmark.py`** - Part C1: PSO optimization
  - Rastrigin function (10D)
  - Rosenbrock function (10D)
  - 3 parameter configurations

- **`pso_sudoku.py`** - Part C2: PSO for Sudoku
  - Discrete PSO adaptation
  - Constraint violation minimization

- **`sudoku_puzzles.py`** - Test puzzle collection
  - Easy, medium, hard difficulty levels
  - Validated solvable puzzles

- **`test_all.py`** - Comprehensive test suite
  - Tests all implementations
  - 100% pass rate achieved

- **`run_experiments.py`** - Experimental results generator
  - Runs all algorithm variants
  - Generates tables for writeup

- **`run_all.ps1`** - PowerShell batch execution script
  - Runs all 4 main programs
  - Saves output to `HW02_runlog.txt`

- **`generate_convergence_plots.py`** - Convergence plot generator
  - Runs PSO with all 3 configurations
  - Tracks iteration-by-iteration convergence
  - Generates publication-quality plots (matplotlib)
  - Outputs: `rastrigin_convergence.pdf`, `rosenbrock_convergence.pdf`

### 3. Documentation
- **`README.md`** - Complete implementation guide
  - Algorithm descriptions
  - Installation instructions
  - Run instructions
  - Performance notes
  - AI disclosure

### 4. Experimental Results
- **`HW02_runlog.txt`** - Complete program output from all runs
- All results documented in writeup tables

---

## Running the Code

### Requirements
```bash
Python 3.7+
numpy (for PSO implementations only)
```

### Quick Start
```bash
# Install dependencies
pip install numpy

# Option 1: Run all programs (Windows PowerShell)
cd HW02_code
.\run_all.ps1

# Option 2: Run full experiments
python run_experiments.py

# Option 3: Run test suite
python test_all.py

# Option 4: Run individual programs
python sudoku_csp.py
python nqueens_minconflicts.py
python pso_benchmark.py
python pso_sudoku.py
```

---

## Key Results Summary

### Part A: Sudoku CSP
| Algorithm | Easy (30 given) | Medium (23 given) | Hard (17 given) |
|-----------|-----------------|-------------------|-----------------|
| Basic Backtracking | 0.058s | 12.578s | 300.000s* |
| +MRV+LCV | 0.023s | 1.813s | 10.569s |
| +Forward Checking | 0.021s | 0.177s | 8.828s |
| **+AC-3** | **0.019s** | **0.080s** | **3.974s** |

*Timeout exceeded

**Finding:** AC-3 is 2.7× faster than MRV+LCV and solved hard puzzle in <4 seconds

### Part B: n-Queens Minimum Conflicts
| n | Success Rate | Avg Steps | Avg Time |
|---|--------------|-----------|----------|
| 8 | 100% (5/5) | 45.8 | 0.0011s |
| 16 | 100% (5/5) | 70.4 | 0.0059s |
| 25 | 100% (5/5) | 90.2 | 0.0222s |

**Finding:** Empirical O(n) scaling - solves n=25 in 22ms

### Part C1: PSO Benchmark Optimization
**Rastrigin (10D):**
- Best configuration: Config 2 (swarm=50, w=0.5)
- Average score: 70.89 (global minimum = 0)

**Rosenbrock (10D):**
- Best configuration: Config 2 (swarm=50, w=0.5)
- Average score: 4316.03 (global minimum = 0)

### Part C2: PSO for Sudoku
- Success rate: 0/3 (0%)
- Best result: 6 violations
- Average violations: 10.0
- Conclusion: PSO not suitable for discrete CSPs

---

## Implementation Quality

✅ **All algorithms implemented from scratch**
- No specialized CSP libraries
- No optimization framework dependencies
- Only NumPy for basic array operations in PSO

✅ **100% test success rate**
- Comprehensive test suite
- All algorithms verified correct

✅ **Safety features**
- 5-minute timeout protection
- Iteration limits
- Progress reporting

✅ **Extensive documentation**
- Line-by-line code comments
- References to textbook and lectures
- Complexity analysis

---

## AI Use Disclosure

This assignment was completed with assistance from **Claude Code (Sonnet 4.5)**, version **claude-sonnet-4-5-20250929**.

AI assistance included:
- Understanding algorithm concepts
- Code implementation and debugging
- Feature development (timeouts, testing, output formatting)
- PowerShell script creation
- Experiment runner development
- LaTeX writeup formatting and analysis

All code was reviewed, understood, and tested by the student before submission. AI did not complete the assignment autonomously - student understanding, testing, and decision-making were central to the process.

Full disclosure included in writeup PDF (page 5).

---

## References

1. **Russell, S. J., & Norvig, P. (2020).** *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson.

2. **Atyabi, A. (2025a).** Constraint Satisfaction Problems. Lecture 5, CS 4820/5820: Artificial Intelligence, University of Colorado Colorado Springs. Fall 2025.

3. **Atyabi, A. (2025b).** Search Optimization Part I-III. Lecture 7, CS 4820/5820: Artificial Intelligence, University of Colorado Colorado Springs. Fall 2025.

---

## File Structure

```
HW02/
├── Manchester_Josh_CS4820_HW02_Writeup.pdf    # Final writeup
├── SUBMISSION_README.md                        # This file
├── HW02_code/
│   ├── sudoku_csp.py                          # Part A implementation
│   ├── nqueens_minconflicts.py                # Part B implementation
│   ├── pso_benchmark.py                       # Part C1 implementation
│   ├── pso_sudoku.py                          # Part C2 implementation
│   ├── sudoku_puzzles.py                      # Test puzzles
│   ├── test_all.py                            # Test suite
│   ├── run_experiments.py                     # Experiment runner
│   ├── run_all.ps1                            # Batch execution script
│   ├── README.md                              # Code documentation
│   └── HW02_runlog.txt                        # Program output
└── writeup/
    ├── assignment_writeup.tex                  # LaTeX source
    ├── aaai24.sty                             # AAAI style file
    ├── aaai24.bst                             # AAAI bibliography style
    ├── references.bib                          # Bibliography database
    └── archive/                                # Old PDF versions

```

---

## Verification Checklist

✅ All code files included and functional
✅ PDF writeup complete with all sections
✅ References section present and properly formatted
✅ AI disclosure included
✅ README with run instructions
✅ Test suite passes 100%
✅ All experimental results documented
✅ Code comments reference textbook/lectures

---

## Contact

For questions about this submission:
- **Josh Manchester**
- josh.manchester@uccs.edu

---

**Submission prepared:** October 26, 2025
**Assignment:** CS 4820/5820 Homework 2
**Instructor:** Professor Adham Atyabi
