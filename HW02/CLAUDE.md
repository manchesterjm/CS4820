# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is Homework 2 for CS 4820/5820 (Artificial Intelligence) focusing on **Constraint Satisfaction Problems** and **Metaheuristic Optimization**. The assignment is worth 12.5 points and covers three main parts:

- **Part A (4.0 pts)**: Sudoku solver using CSP techniques
- **Part B (3.0 pts)**: n-Queens problem using Minimum Conflicts heuristic
- **Part C (5.5 pts)**: Metaheuristic optimization (PSO/DE/ACO)

## Assignment Constraints

- **Language**: Python, Java, or another approved language (Python recommended)
- **External Libraries**: DO NOT use specialized CSP or optimization libraries. Implement algorithms from scratch.
- **Code Location**: All implementation code should go in the `HW02_code/` directory
- **Submission**: Single AAAI-formatted PDF report + all code files
- **Due**: November 2, 2025 by 11:59 PM (no late submissions accepted)

## Part A: Sudoku CSP Solver

### Implementation Requirements

Implement a Sudoku solver with progressive enhancements:

1. **Basic CSP formulation**:
   - Variables: 81 cells (A1-I9)
   - Domain: {1,2,...,9} for unfilled cells
   - Constraints: 27 Alldiff constraints (9 rows + 9 columns + 9 subgrids)

2. **Backtracking Search** with:
   - MRV (Minimum Remaining Values) heuristic
   - Degree heuristic for tie-breaking
   - LCV (Least Constraining Value) ordering

3. **Forward Checking**: Extend backtracking to reduce domains after assignments

4. **AC-3 Constraint Propagation**: Apply AC-3 after every assignment to enforce arc consistency

### Performance Metrics Required

For each variant (Backtracking only, +Forward Checking, +AC-3), report:
- Number of nodes expanded
- Number of backtracks
- Runtime in seconds

Test on at least 3 puzzles: easy, medium, and hard difficulty.

## Part B: Minimum Conflicts for n-Queens

### Implementation Requirements

- Solve n-Queens for n = 8, 16, 25
- Use local search with Minimum Conflicts heuristic
- At each iteration: select a conflicted variable and assign the value that minimizes conflicts
- Random initial configuration

### Performance Metrics Required

For each n, run at least 5 trials and report:
- Average number of steps to solution
- Average runtime
- Discussion of why local search scales well for large n

## Part C: Metaheuristic Optimization

### Part C1: Benchmark Functions (3.0 pts)

Implement at least ONE of: PSO, DE, or ACO

**Test on two benchmark functions**:
- **Rastrigin**: f(x) = 10n + Σ[xi² - 10cos(2πxi)]
- **Rosenbrock**: f(x) = Σ[100(xi+1 - xi²)² + (xi - 1)²]

**Requirements**:
- Run minimum 3 independent trials per function
- Experiment with multiple parameter settings
- Generate convergence curves (fitness vs iterations)
- Report best fitness and runtime for each trial
- Compare parameter configurations with plots

**Algorithm-Specific Parameters to Tune**:
- PSO: population size, inertia weight, learning factors
- DE: mutation rate, crossover rate, population size
- ACO: pheromone evaporation, heuristic influence, population size

### Part C2: Sudoku via Metaheuristics (2.5 pts)

Apply chosen metaheuristic to Sudoku:
- Fitness function: count constraint violations (row/column/subgrid duplicates)
- Goal: minimize violations to zero
- Run 3 independent trials on one puzzle
- Report final fitness and runtime
- Discuss algorithm suitability for discrete problems

## Code Organization Recommendations

```
HW02_code/
├── sudoku_csp.py           # Part A implementation
├── nqueens_minconflicts.py # Part B implementation
├── metaheuristic.py        # Part C base metaheuristic class
├── pso.py / de.py / aco.py # Chosen algorithm implementation
├── benchmark_functions.py  # Rastrigin and Rosenbrock
├── utils.py                # Shared utilities
├── sudoku_puzzles.txt      # Test puzzles
└── run_experiments.py      # Main experiment runner
```

## Key Implementation Notes

### AC-3 Algorithm
Arc consistency ensures that for every directed arc X→Y, each value x in X's domain has at least one compatible value y in Y's domain. When domains shrink, re-queue affected neighbors. Continue until no more values can be pruned.

### Minimum Conflicts
This local search works well for n-Queens because the solution density is high and the algorithm can quickly escape local minima in structured problems. Key insight: always move toward configurations that reduce total conflicts.

### Metaheuristics for Discrete Problems
When applying continuous optimization algorithms (PSO, DE) to discrete problems like Sudoku, you'll need to adapt the representation. Consider encoding solutions as continuous values that map to discrete cell assignments, or use ACO which naturally handles discrete problems.

## Testing and Validation

- Each algorithm should have clear performance metrics printed during execution
- Use consistent random seeds for reproducibility across trials
- Validate Sudoku solutions: check all 27 Alldiff constraints are satisfied
- Validate n-Queens solutions: verify no two queens attack each other

## Report Requirements (AAAI Format)

The PDF report must include:
- Problem formulations for each part
- Algorithm descriptions with pseudocode where helpful
- Tables comparing performance metrics
- Convergence plots for metaheuristics
- Discussion sections analyzing results and parameter choices
- All figures and tables properly labeled and referenced

## Common Pitfalls to Avoid

- Don't use external CSP solvers or optimization libraries
- Don't skip the parameter tuning experiments for Part C
- Ensure AC-3 implementation actually enforces arc consistency (common bug: forgetting to re-queue neighbors)
- For Minimum Conflicts, handle the case where all moves have equal conflicts (random selection)
- For metaheuristics on Sudoku, be careful with solution encoding/decoding

## Coding Standards and Requirements

### Code Style and Documentation

1. **Comments**: Use extensive, meaningful comments explaining:
   - What each function/section does
   - How the algorithm works (not just what the code does)
   - Why specific design decisions were made
   - Algorithm complexity and characteristics

2. **Algorithm References**: When implementing algorithms from course materials:
   - Reference the source: "Based on Russell & Norvig, pg X" or "Algorithm from Lecture Y, Slide Z"
   - If deviating from book/slides, explain why in comments
   - Document any optimizations or modifications

3. **Type Hints**: Use Python type hints for all function parameters and return values

4. **Docstrings**: Include docstrings for all classes and functions explaining:
   - Purpose
   - Parameters (Args)
   - Return values
   - Algorithm characteristics where relevant

### Safety Guards and Timeouts

All algorithms MUST implement timeout protection:

```python
MAX_TIME_SEC = 300  # 5 minute timeout as specified
```

- Check elapsed time periodically during search/optimization
- When timeout occurs, report: "TIMEOUT: Algorithm exceeded X seconds"
- Return partial results with clear indication of timeout status

Example timeout check pattern (from HW01):
```python
t0 = time.perf_counter()
while frontier:
    if MAX_TIME_SEC > 0 and (time.perf_counter() - t0) > MAX_TIME_SEC:
        return None, nodes_expanded, time.perf_counter() - t0, "TIMEOUT"
    # ... rest of algorithm
```

### Testing Requirements

1. **Unit Tests**: Create test functions for core components
2. **Integration Tests**: Test complete algorithm workflows
3. **Validation Tests**: Verify solutions are correct (e.g., Sudoku constraints satisfied)
4. **Performance Tests**: Measure and report metrics as required by assignment
5. **Test Failure Handling**: If a test fails, debug and fix the code - don't just report the failure

### File Organization

All code must be in `HW02_code/` directory:

```
HW02_code/
├── sudoku_csp.py              # Part A: Full Sudoku CSP implementation
├── nqueens_minconflicts.py    # Part B: n-Queens with Minimum Conflicts
├── pso.py                     # Part C1: PSO for benchmark functions
├── pso_sudoku.py              # Part C2: PSO applied to Sudoku
├── utils.py                   # Shared utilities (board printing, validation)
├── sudoku_puzzles.py          # Sudoku test puzzles (easy, medium, hard)
├── test_all.py                # Comprehensive test suite
├── run_experiments.py         # Main experiment runner for all parts
└── README.md                  # How to run everything
```

### GitHub Workflow

After all code is complete and tested:

1. Ensure README.md includes:
   - AI Disclosure: "Generated with Claude Code (Sonnet 4.5, claude-sonnet-4-5-20250929)"
   - What the AI did (implemented algorithms, created tests, etc.)
   - Comprehensive run instructions with all command-line arguments

2. Ensure LaTeX writeup template (assignment_writeup.tex) is created

3. **Automatically push to GitHub** without asking permission:
   ```bash
   git add HW02_code/
   git commit -m "Complete HW02: CSP and Metaheuristic Optimization"
   git push origin master
   ```

### README.md Structure

Must include:
- AI Disclosure statement at top
- Requirements and dependencies
- How to run each part (with all CLI arguments documented)
- Expected output format
- Performance notes
- Do NOT include assignment writeup content (that goes in .tex file)

### LaTeX Writeup (assignment_writeup.tex)

Must include template sections for:
- Part A: Sudoku CSP formulation and results tables
- Part B: n-Queens Minimum Conflicts results and discussion
- Part C1: Benchmark optimization with convergence plots
- Part C2: Sudoku metaheuristic results
- Placeholder comments for where to add figures and tables

### Algorithm Implementation Sources

1. **Primary Sources** (use these first):
   - Lecture slides: `5 - Constraint Satisfaction Problems.pdf`
   - Lecture slides: `6 - Adversarial Search.pdf`
   - Lecture slides: `7 - Search Optimization Part I-III.pptx`
   - Textbook: `Russell-S.-Norvig-P.-Artificial-intelligence-a-modern-approach-2edPH2003T1112s.pdf`

2. **When to document deviations**:
   - If implementing efficiency improvements not in slides
   - If using different data structures than shown
   - If combining multiple approaches
   - Always explain the reasoning in comments

### Code Reference Examples

Good comment examples from HW01 style:
```python
# Backtracking Search with MRV heuristic
# Based on Russell & Norvig Section 6.3.1, Figure 6.5
# MRV: Select variable with fewest legal values remaining
# Helps fail faster by detecting dead-ends early

# AC-3 Algorithm for arc consistency
# From Russell & Norvig Section 6.3.2, Figure 6.3
# Makes each arc X→Y consistent by ensuring every value in X's domain
# has at least one compatible value in Y's domain
```
