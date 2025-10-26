# CS 4820/5820 Homework 2 - Constraint Satisfaction & Metaheuristic Optimization

**Author:** Josh Manchester
**Institution:** University of Colorado Colorado Springs
**Email:** josh.manchester@uccs.edu
**Course:** CS 4820/5820 - Artificial Intelligence

## AI Disclosure

This code was generated with assistance from **Claude Code (Sonnet 4.5)**, version **claude-sonnet-4-5-20250929**.

The AI assistant helped with:
- Implementation of CSP algorithms (Backtracking, MRV, LCV, Forward Checking, AC-3)
- Implementation of Minimum Conflicts local search
- Implementation of Particle Swarm Optimization (PSO)
- Code structure, documentation, and comments
- Test suite development

All algorithms were implemented according to specifications from:
- Russell & Norvig, "Artificial Intelligence: A Modern Approach"
- CS 4820/5820 Lecture 5: Constraint Satisfaction Problems
- CS 4820/5820 Lecture 7: Search Optimization Part III

## Requirements

- Python 3.7 or higher
- NumPy (for PSO implementations)
- No specialized CSP or optimization libraries (all algorithms implemented from scratch)

Install NumPy if needed:
```bash
pip install numpy
```

## File Structure

```
HW02_code/
├── sudoku_csp.py              # Part A: Sudoku CSP solver (all variants)
├── nqueens_minconflicts.py    # Part B: n-Queens with Minimum Conflicts
├── pso_benchmark.py           # Part C1: PSO for benchmark functions
├── pso_sudoku.py              # Part C2: PSO applied to Sudoku
├── sudoku_puzzles.py          # Collection of test Sudoku puzzles
├── test_all.py                # Comprehensive test suite
├── run_experiments.py         # Main experiment runner (generates results)
└── README.md                  # This file
```

## Quick Start

### Option 1: Run All Programs with PowerShell Script (Windows)

The easiest way to run everything and save output for screenshots:

```powershell
.\run_all.ps1
```

This PowerShell script will:
- Run all four main programs (Parts A, B, C1, C2)
- Display output on screen in real-time
- Save all output to `HW02_runlog.txt` for screenshots
- Optionally run full experiments and test suite

The script uses `Tee-Object` to simultaneously display and save output, making it easy to capture screenshots for the report.

### Option 2: Run Full Experiments (Recommended)

Run all experiments and generate results for the report:

```bash
python run_experiments.py
```

This will:
- Test all Sudoku CSP solver variants (Part A)
- Run n-Queens for n=8, 16, 25 (Part B)
- Benchmark PSO on Rastrigin and Rosenbrock functions (Part C1)
- Apply PSO to Sudoku (Part C2)
- Output formatted tables ready for inclusion in report

**Runtime:** Approximately 5-8 minutes depending on system

### Option 3: Run Test Suite

Verify all implementations work correctly:

```bash
python test_all.py
```

This runs automated tests on all components and reports pass/fail status.

### Option 4: Run Individual Programs

You can run each part independently:

```bash
# Part A: Sudoku CSP (tests all 4 algorithm variants)
python sudoku_csp.py

# Part B: n-Queens with Minimum Conflicts
python nqueens_minconflicts.py

# Part C1: PSO on benchmark functions
python pso_benchmark.py

# Part C2: PSO for Sudoku
python pso_sudoku.py

# View available test puzzles
python sudoku_puzzles.py
```

## Assignment Parts

### Part A: Sudoku as a CSP (4 points)

**File:** `sudoku_csp.py`

Implements Sudoku as a Constraint Satisfaction Problem with four solver variants:

1. **Basic Backtracking**
   - Naive depth-first search with constraint checking
   - Baseline for comparison

2. **Backtracking + MRV + LCV**
   - MRV (Minimum Remaining Values): Choose most constrained variable first
   - LCV (Least Constraining Value): Try least constraining values first
   - Dramatic performance improvement over basic backtracking

3. **Backtracking + Forward Checking**
   - Maintains arc consistency with future variables
   - Detects failures earlier by tracking legal values
   - Prunes search space more aggressively

4. **Backtracking + AC-3**
   - Full constraint propagation using AC-3 algorithm
   - Most powerful pruning technique
   - Often solves Sudoku without backtracking

**Key algorithms referenced:**
- Backtracking: Russell & Norvig Figure 6.5, Lecture 5 Slide 19
- MRV: Lecture 5 Slides 24, 27
- LCV: Lecture 5 Slide 30
- Forward Checking: Lecture 5 Slide 36
- AC-3: Russell & Norvig Figure 6.3, Lecture 5 Slide 70

### Part B: n-Queens with Minimum Conflicts (3 points)

**File:** `nqueens_minconflicts.py`

Implements the Minimum Conflicts local search heuristic for n-Queens problem.

**Tested on:** n = 8, 16, 25

**Algorithm:** Lecture 5 Slide 55, Russell & Norvig Figure 6.8

**Key features:**
- Starts with random complete assignment
- Iteratively selects conflicted variable
- Assigns value that minimizes conflicts
- Very efficient: typically solves in O(n) steps
- Includes random restart for robustness

**Why it works:**
- n-Queens has high solution density
- Local minima are rare
- Much faster than backtracking for large n

### Part C1: PSO for Benchmark Optimization (3 points)

**File:** `pso_benchmark.py`

Implements Particle Swarm Optimization and tests on standard benchmark functions.

**Benchmark functions:**
1. **Rastrigin:** f(x) = 10n + Σ[xi² - 10cos(2πxi)]
   - Highly multimodal (many local minima)
   - Global minimum: f(0,...,0) = 0

2. **Rosenbrock:** f(x) = Σ[100(xi+1 - xi²)² + (xi - 1)²]
   - Narrow parabolic valley
   - Global minimum: f(1,...,1) = 0

**Tests multiple parameter configurations:**
- Swarm size: 30-50 particles
- Inertia weight (w): 0.5-0.9
- Cognitive/social coefficients (c1, c2): 1.2-2.0
- Runs 3 trials per configuration

**Key features:**
- Social + cognitive learning
- Velocity and position updates
- Convergence tracking
- Boundary handling

### Part C2: PSO for Sudoku (2.5 points)

**File:** `pso_sudoku.py`

Applies PSO to solve Sudoku as an optimization problem.

**Approach:**
- Minimize constraint violations instead of hard CSP
- Each particle = complete Sudoku board
- Fixed cells locked in place
- Row permutations maintained (0 row violations)
- Optimize column and box constraints

**Discrete PSO adaptations:**
- Swap operations instead of continuous velocity
- Probabilistic movement toward personal/global best
- Handles combinatorial structure

**Note:** PSO may not always find perfect solution (0 violations). For guaranteed solutions, use CSP methods from Part A. This demonstrates metaheuristic approach to constraint optimization.

## Understanding the Output

### Sudoku CSP Output
```
Backtracking + AC-3
Solved in 0.0234 seconds

Solution:
5 3 4 | 6 7 8 | 9 1 2
6 7 2 | 1 9 5 | 3 4 8
...
```

### n-Queens Output
```
Trial 1:
  Status: ok
  Steps: 42
  Attempts: 1
  Time: 0.0018s
  Solution verified: 0 conflicts
```

### PSO Benchmark Output
```
Configuration 1 (Standard):
  Trial 1: score=2.45e-02, iters=1000, time=1.2345s, status=converged

  Summary:
    Best score: 1.23e-02
    Avg score: 3.45e-02 ± 1.23e-02
```

### PSO Sudoku Output
```
Trial 1:
  Final score: 5 violations
  Iterations: 3000
  Time: 12.3456s
  Status: max_iterations
```

## Modifying Parameters

### Sudoku CSP
Edit timeout in `sudoku_csp.py`:
```python
MAX_TIME_SEC = 300  # 5 minute timeout
```

### n-Queens Minimum Conflicts
Edit limits in `nqueens_minconflicts.py`:
```python
MAX_STEPS = 1000000  # Max iterations per attempt
```

### PSO Parameters
In `pso_benchmark.py` or `pso_sudoku.py`:
```python
swarm_size = 30      # Number of particles (20-100 typical)
w = 0.7              # Inertia weight (0.4-0.9)
c1 = 1.5             # Cognitive coefficient (1.2-2.0)
c2 = 1.5             # Social coefficient (1.2-2.0)
max_iterations = 1000 # Max iterations
```

### Test Puzzles
Add new puzzles in `sudoku_puzzles.py`:
```python
CUSTOM_PUZZLE = [
    [0, 0, 0, ...],
    ...
]
PUZZLES["custom"] = [CUSTOM_PUZZLE]
```

## Performance Notes

### Expected Runtimes

**Sudoku CSP (easy puzzles):**
- Basic Backtracking: 0.01-0.1s
- +MRV+LCV: 0.001-0.01s
- +Forward Checking: 0.001-0.005s
- +AC-3: <0.001s (often solves without search)

**n-Queens Minimum Conflicts:**
- n=8: <0.01s, typically 20-50 steps
- n=16: <0.01s, typically 30-80 steps
- n=25: <0.02s, typically 40-120 steps

**PSO Benchmark (1000 iterations):**
- Rastrigin (10D): 1-3s
- Rosenbrock (10D): 1-3s

**PSO Sudoku (3000 iterations):**
- Easy puzzle: 10-30s
- May or may not reach 0 violations

### Algorithm Complexity

**Sudoku CSP:**
- Basic Backtracking: O(d^n) time, O(n) space
- +MRV+LCV: Same worst-case, much better average-case
- +Forward Checking: O(d²) overhead per node
- +AC-3: O(cd³) overhead per node (c=constraints, d=domain size)

**n-Queens Minimum Conflicts:**
- Time per step: O(n)
- Expected steps: O(n) empirically
- Space: O(n)

**PSO:**
- Time per iteration: O(swarm_size × dimensions)
- Total: O(iterations × swarm_size × dimensions)
- Space: O(swarm_size × dimensions)

## Safety Features

All implementations include:

1. **Timeout Protection**
   - MAX_TIME_SEC = 300 (5 minutes)
   - Prevents infinite loops
   - Returns partial results if timeout

2. **Iteration Limits**
   - Prevents excessive computation
   - Configurable per algorithm

3. **Progress Reporting**
   - Regular status updates for long runs
   - Convergence tracking

## Troubleshooting

**"ModuleNotFoundError: No module named 'numpy'"**
```bash
pip install numpy
```

**Programs running slowly:**
- Reduce scramble steps for puzzles
- Reduce max_iterations for PSO
- Use easier test puzzles
- All programs have timeout safeguards

**PSO not solving Sudoku:**
- This is normal - PSO is not guaranteed to solve
- Try increasing swarm_size (100-200)
- Try increasing max_iterations (5000+)
- For guaranteed solutions, use CSP methods

**Tests failing:**
- Check numpy is installed
- Check Python version (need 3.7+)
- Some PSO tests may occasionally fail due to randomness
- Re-run test suite if marginal failures occur

## Code Documentation

All Python files include:
- Detailed algorithm explanations
- References to lecture slides and textbook
- Line-by-line comments
- Complexity analysis
- Design decisions and tradeoffs

Read the source code to understand:
- How each algorithm works
- Why it works
- When to use each approach
- Implementation details

## Assignment Deliverables

Based on this code, the assignment requires:

1. **Code submission:** All .py files in this directory
2. **Report (AAAI format):** LaTeX writeup with:
   - Part A: Sudoku CSP comparison table
   - Part B: n-Queens results for n=8,16,25
   - Part C1: PSO benchmark results with parameter analysis
   - Part C2: PSO Sudoku results from 3 trials
   - Figures: Convergence plots for PSO
   - Discussion: Algorithm comparisons and insights

3. **Submission format:** One PDF + all source code

## References

- Russell, S. & Norvig, P. "Artificial Intelligence: A Modern Approach" (4th Edition)
  - Chapter 6: Constraint Satisfaction Problems
  - Chapter 4: Local Search Algorithms

- CS 4820/5820 Lecture Slides:
  - Lecture 5: Constraint Satisfaction Problems (Slides 19, 24, 27, 30, 36, 55, 70)
  - Lecture 7: Search Optimization Part III (PSO)

- Kennedy, J. & Eberhart, R. "Particle Swarm Optimization," 1995

- Jamil, M. & Yang, X. "A Literature Survey of Benchmark Functions for Global Optimization," 2013

## Contact

For questions about this implementation:
- Josh Manchester
- josh.manchester@uccs.edu

## License

This code is for educational purposes as part of CS 4820/5820 coursework.
Individual work only - do not share or copy.
