# CS4820/5820 Python Style Guide

**Artificial Intelligence Coursework - Coding Standards**

**Course**: CS 4820/5820 (Artificial Intelligence)
**Institution**: University of Colorado Colorado Springs
**Instructor**: Professor Adham Atyabi
**Version**: 1.0
**Last Updated**: November 1, 2025
**Status**: Active for all assignments

This document defines the coding standards for all Python implementations in CS4820/5820. Following these guidelines ensures consistency, academic integrity, and professional-quality code suitable for course submissions and portfolio work.

---

## Table of Contents

1. [General Principles](#general-principles)
2. [Python Style (PEP 8 Adapted)](#python-style-pep-8-adapted)
3. [Academic Coding Standards](#academic-coding-standards)
4. [Naming Conventions](#naming-conventions)
5. [Code Organization](#code-organization)
6. [Functions and Methods](#functions-and-methods)
7. [Documentation and References](#documentation-and-references)
8. [Error Handling](#error-handling)
9. [Imports](#imports)
10. [Type Hints](#type-hints)
11. [Testing Standards](#testing-standards)
12. [Algorithm Implementation](#algorithm-implementation)
13. [Experimental Code](#experimental-code)
14. [Code Quality Checklist](#code-quality-checklist)

---

## General Principles

### Core Values
1. **Academic Integrity**: Always cite algorithm sources (textbook, papers, lectures)
2. **Readability First**: Code is read by instructors, graders, and future employers
3. **Reproducibility**: Results must be reproducible with documented random seeds
4. **Self-Documenting**: Code should explain the algorithm, not just implement it
5. **Performance Awareness**: Document time/space complexity
6. **Educational Value**: Code should teach, not just work

### Code Quality Standards
- **Pylint Score**: Maintain ≥8.0/10 (target ≥9.0/10)
- **All Tests Must Pass**: 100% test success rate before submission
- **No Crashes**: Implement timeout protection for all search algorithms
- **No Encoding Errors**: ASCII-only output (Windows cp1252 compatibility)

---

## Python Style (PEP 8 Adapted)

### Line Length
```python
# Maximum line length: 100 characters (academic standard)
# More restrictive than web apps for better printed documentation

# BAD: Line too long
print(f"Algorithm completed in {elapsed:.4f} seconds with {iterations} iterations and final score {best_score:.6f}")

# GOOD: Line broken appropriately
print(f"Algorithm completed in {elapsed:.4f} seconds")
print(f"  Iterations: {iterations}")
print(f"  Final score: {best_score:.6f}")
```

### Indentation
```python
# Use 4 spaces per indentation level (never tabs)
# Be especially careful with nested algorithm logic

# GOOD: Clear nested structure
def backtrack(assignment, domains):
    if is_complete(assignment):
        return assignment

    var = select_unassigned_variable(assignment, domains)
    for value in order_domain_values(var, assignment, domains):
        if is_consistent(var, value, assignment):
            assignment[var] = value
            result = backtrack(assignment, domains)
            if result is not None:
                return result
            del assignment[var]

    return None
```

### Blank Lines
```python
# Two blank lines between top-level classes and functions
def search_algorithm():
    pass


def optimization_algorithm():
    pass


# One blank line between methods
class PSO:
    def __init__(self):
        pass

    def optimize(self):
        pass


# Use blank lines to separate logical sections in complex algorithms
def backtracking_search(csp):
    """Backtracking search with MRV and LCV heuristics."""
    # Initialize data structures
    assignment = {}
    domains = copy.deepcopy(csp.domains)

    # Apply initial constraint propagation
    if not ac3(csp, domains):
        return None

    # Recursive backtracking
    return backtrack(assignment, domains, csp)
```

---

## Academic Coding Standards

### Algorithm References (MANDATORY)
```python
# CRITICAL: Always cite algorithm sources in comments

# GOOD: Proper academic citation
def ac3(csp, domains):
    """
    AC-3 algorithm for arc consistency.

    Based on Russell & Norvig Section 6.3.2, Figure 6.3.
    Also covered in Lecture 5, Slides 65-70.

    Makes each arc X->Y consistent by ensuring every value in X's domain
    has at least one compatible value in Y's domain.

    Time Complexity: O(cd^3) where c=constraints, d=domain size
    Space Complexity: O(c) for the queue

    Optimization: Using deque for O(1) queue operations instead of list.
    """
    queue = deque([(xi, xj) for xi in csp.variables for xj in csp.neighbors[xi]])
    # ...

# BAD: No citation, unclear origin
def ac3(csp, domains):
    """Make arcs consistent."""
    queue = deque()
    # ...
```

### Complexity Analysis
```python
# Document time and space complexity for all algorithms

def minimum_conflicts(csp, max_steps=1000):
    """
    Minimum conflicts local search for CSP.

    Based on Russell & Norvig Section 6.4, Figure 6.8.
    Reference: Minton et al. (1992).

    Time Complexity: O(max_steps * n) where n = number of variables
    Space Complexity: O(n) for current assignment

    Empirically solves n-Queens in O(n) steps (nearly constant time).

    Args:
        csp: CSP instance with variables, domains, constraints
        max_steps: Maximum iterations before giving up

    Returns:
        Solution assignment if found, None otherwise
    """
    # Implementation...
```

### Experimental Reproducibility
```python
# Always use documented random seeds for reproducible results

import random
import numpy as np

# GOOD: Seeded randomness with documentation
def run_experiment(seed=42):
    """
    Run PSO experiment with reproducible results.

    Args:
        seed: Random seed for reproducibility (default: 42)

    Returns:
        Results dictionary with scores and statistics
    """
    random.seed(seed)
    np.random.seed(seed)

    print(f"Running with random seed: {seed}")
    # ... experiment code ...

# Include seed in experimental output
print(f"Experiment configuration: seed={seed}, iterations={max_iter}")
```

---

## Naming Conventions

### Variables and Functions
```python
# Use descriptive names that reflect algorithm concepts

# GOOD: Clear algorithmic meaning
num_backtracks = 0
assignment = {}
unassigned_vars = []
mrv_variable = select_minimum_remaining_values(csp)

# BAD: Unclear abbreviations
n_bt = 0
asgn = {}
uv = []
var = select_mrv(csp)

# Boolean variables: use is_, has_, can_ prefixes
is_consistent = check_consistency(var, value, assignment)
has_solution = solution is not None
can_improve = current_conflicts > 0

# Counters and indices: clear naming
for iteration in range(max_iterations):  # GOOD
for i in range(max_iterations):          # Less clear
for iter in range(max_iterations):       # BAD (shadows built-in)
```

### Algorithm Components
```python
# Name components after their algorithmic purpose

# CSP algorithms
def select_mrv_variable(assignment, domains):
    """Select variable with Minimum Remaining Values."""
    pass

def order_lcv_values(var, assignment, domains):
    """Order values by Least Constraining Value heuristic."""
    pass

def forward_check(var, value, assignment, domains):
    """Apply forward checking after assignment."""
    pass

# Search algorithms
def expand_node(node, problem):
    """Expand node to generate successor states."""
    pass

def is_goal_state(state, problem):
    """Check if state satisfies goal condition."""
    pass

# Optimization algorithms
def update_particle_velocity(particle, global_best, w, c1, c2):
    """Update particle velocity using PSO equation."""
    pass

def evaluate_fitness(solution, objective_func):
    """Evaluate solution quality using objective function."""
    pass
```

### Constants
```python
# Algorithm-specific constants in UPPERCASE

# Timeouts and limits
MAX_TIME_SEC = 300          # 5 minute timeout
MAX_ITERATIONS = 1000       # Iteration limit
MAX_RESTARTS = 10           # Random restart limit

# Algorithm parameters
DEFAULT_INERTIA = 0.7       # PSO inertia weight
DEFAULT_C1 = 1.5            # PSO cognitive coefficient
DEFAULT_C2 = 1.5            # PSO social coefficient
CONVERGENCE_TOLERANCE = 1e-6  # Convergence threshold

# Problem-specific
N_QUEENS_SIZE = 8           # Board size for n-Queens
SUDOKU_SIZE = 9             # Sudoku grid size
SWARM_SIZE = 50             # PSO swarm population
```

---

## Code Organization

### File Structure
```python
"""
Module docstring explaining purpose and algorithm.

This module implements the AC-3 arc consistency algorithm for constraint
satisfaction problems. Based on Russell & Norvig Section 6.3.2.

References:
    - Russell & Norvig (2020). Artificial Intelligence: A Modern Approach, 4th ed.
    - Lecture 5: Constraint Satisfaction Problems, Slides 65-70
"""

# Standard library imports
import time
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

# Third-party imports (if allowed by assignment)
import numpy as np
import matplotlib.pyplot as plt

# Local application imports (for multi-file assignments)
from sudoku_puzzles import EASY_PUZZLES, HARD_PUZZLES
from utils import print_sudoku, validate_solution

# Constants (after imports, before code)
MAX_TIME_SEC = 300
SUDOKU_SIZE = 9

# Module-level variables (use sparingly)
logger = None  # Optional: set up logging


# Classes and functions (in logical order)
class CSP:
    """Constraint Satisfaction Problem representation."""
    pass


def solve_csp(problem: CSP) -> Optional[Dict]:
    """Main solving function."""
    pass


def main():
    """Entry point for running experiments."""
    pass


if __name__ == '__main__':
    main()
```

### Import Ordering
```python
# 1. Standard library (alphabetical)
import copy
import random
import time
from collections import deque
from typing import Dict, List, Optional, Set

# 2. Third-party packages (alphabetical)
import matplotlib.pyplot as plt
import numpy as np

# 3. Local modules (alphabetical)
from pso_benchmark import PSO, rastrigin, rosenbrock
from sudoku_csp import solve_with_ac3, solve_with_backtracking
from sudoku_puzzles import load_puzzle
```

---

## Functions and Methods

### Function Length and Complexity
```python
# Keep functions focused (generally <50 lines, <75 for complex algorithms)
# Break down long algorithms into logical sub-functions

# GOOD: Algorithm broken into clear steps
def backtracking_search(csp):
    """
    Solve CSP using backtracking with MRV and AC-3.

    Based on Russell & Norvig Section 6.3, Figure 6.5.
    """
    assignment = {}
    domains = _initialize_domains(csp)

    if not _apply_ac3(csp, domains):
        return None

    return _backtrack(assignment, domains, csp)


def _initialize_domains(csp):
    """Create initial domain copies for each variable."""
    return {var: set(domain) for var, domain in csp.domains.items()}


def _apply_ac3(csp, domains):
    """Apply AC-3 preprocessing to reduce domains."""
    # AC-3 implementation...
    pass


def _backtrack(assignment, domains, csp):
    """Recursive backtracking with forward checking."""
    # Backtracking implementation...
    pass
```

### Single Return Statement (When Practical)
```python
# Use single return for main algorithm logic
# Early returns OK for validation and error conditions

# GOOD: Early returns for edge cases, single return for main logic
def calculate_conflicts(board, n):
    """
    Count total conflicts for n-Queens board.

    Args:
        board: List where board[col] = row for queen in column col
        n: Board size

    Returns:
        Total number of conflicts (each conflict counted twice)
    """
    # Edge case validation (early return OK)
    if not board or len(board) != n:
        return -1

    # Main algorithm (single return point)
    conflicts = 0
    for col1 in range(n):
        row1 = board[col1]
        for col2 in range(col1 + 1, n):
            row2 = board[col2]

            # Same row conflict
            if row1 == row2:
                conflicts += 1

            # Diagonal conflict
            if abs(row1 - row2) == abs(col1 - col2):
                conflicts += 1

    return conflicts
```

### Function Arguments
```python
# Limit function arguments (≤5 is ideal, ≤7 maximum)
# Group related parameters into dictionaries or dataclasses

# BAD: Too many individual parameters
def run_pso(swarm_size, max_iter, w, c1, c2, tol, bounds, dims, func, seed):
    pass

# GOOD: Grouped parameters
def run_pso(objective_func, dimensions, bounds, params=None, seed=42):
    """
    Run PSO optimization.

    Args:
        objective_func: Function to minimize
        dimensions: Problem dimensionality
        bounds: Tuple of (min, max) for search space
        params: Dict with keys: swarm_size, max_iter, w, c1, c2, tol
        seed: Random seed for reproducibility
    """
    # Use defaults if params not provided
    if params is None:
        params = {
            'swarm_size': 30,
            'max_iter': 1000,
            'w': 0.7,
            'c1': 1.5,
            'c2': 1.5,
            'tol': 1e-6
        }
    pass

# BEST: Use dataclass for structured parameters
from dataclasses import dataclass

@dataclass
class PSOConfig:
    """PSO algorithm configuration."""
    swarm_size: int = 30
    max_iterations: int = 1000
    inertia_weight: float = 0.7
    cognitive_coeff: float = 1.5
    social_coeff: float = 1.5
    tolerance: float = 1e-6

def run_pso(objective_func, dimensions, bounds, config: PSOConfig, seed=42):
    """Run PSO with structured configuration."""
    pass
```

---

## Documentation and References

### Module Docstrings
```python
"""
N-Queens solver using Minimum Conflicts local search.

This module implements the Minimum Conflicts algorithm for solving the
n-Queens problem. The algorithm uses local search with random restarts
to efficiently find solutions even for large board sizes.

Algorithm Reference:
    - Russell & Norvig (2020), Section 6.4: Local Search for CSPs
    - Minton et al. (1992). "Minimizing Conflicts: A Heuristic Repair
      Method for Constraint Satisfaction and Scheduling Problems"

Course Materials:
    - Lecture 5: Constraint Satisfaction Problems, Slides 85-92
    - Assignment 2 specification, Part B

Performance:
    - Time Complexity: O(n) expected steps (empirical)
    - Space Complexity: O(n) for board representation
    - Typically solves n=1000 in <1 second

Author: Josh Manchester
Email: josh.manchester@uccs.edu
Course: CS 4820/5820 Fall 2025
"""
```

### Function Docstrings (Google Style)
```python
def solve_sudoku_ac3(puzzle):
    """
    Solve Sudoku using backtracking with AC-3 preprocessing.

    Applies AC-3 arc consistency before backtracking to reduce search space.
    Often solves easy puzzles with no backtracking required.

    Algorithm:
        1. Apply AC-3 to establish arc consistency
        2. If domains reduced to singletons, extract solution
        3. Otherwise, use backtracking with MRV heuristic

    Based on Russell & Norvig Section 6.3.2, Figure 6.3.

    Args:
        puzzle: 9x9 list of lists with integers 0-9 (0 = empty cell)

    Returns:
        9x9 list of lists with solution, or None if unsolvable

    Raises:
        ValueError: If puzzle is not 9x9 or contains invalid values

    Time Complexity:
        - AC-3: O(cd^3) where c=constraints, d=domain_size
        - Backtracking: O(d^n) worst case, often much better with AC-3

    Space Complexity:
        O(n) for assignment and domains where n = 81 cells

    Example:
        >>> puzzle = load_puzzle("easy_01.txt")
        >>> solution = solve_sudoku_ac3(puzzle)
        >>> validate_solution(solution)
        True

    References:
        - Russell & Norvig (2020). AIMA 4th ed, Section 6.3.2
        - Lecture 5, Slides 65-70
    """
    # Implementation...
```

### Inline Comments (Explain WHY, not WHAT)
```python
# BAD: Comment states the obvious
# Increment counter by 1
counter += 1

# GOOD: Explains algorithmic reasoning
# MRV heuristic: choose variable with fewest legal values to fail faster
var = min(unassigned, key=lambda v: len(domains[v]))

# BAD: Redundant comment
# Get value from dictionary
value = domains[var]

# GOOD: Explains algorithm step
# LCV heuristic: try least constraining value first to preserve flexibility
values = sorted(domains[var], key=lambda v: count_constraints(v, var))

# Use TODO for algorithm improvements
# TODO: Add degree heuristic for tie-breaking in MRV
# TODO: Implement MAC (Maintaining Arc Consistency) for better performance

# Use FIXME for known issues
# FIXME: Race condition in timeout check, needs atomic operation
# FIXME: Memory leak when max_iterations > 10000, investigate
```

### Complexity Documentation
```python
def select_mrv_variable(assignment, domains, csp):
    """
    Select unassigned variable with Minimum Remaining Values.

    MRV heuristic (also called "fail-first" or "most constrained variable")
    chooses the variable with the fewest legal values remaining. This causes
    backtracking to detect failures earlier in the search tree.

    Time Complexity: O(n * d) where n=unassigned vars, d=avg domain size
    Space Complexity: O(1) - only stores min variable and count

    Optimization: Could cache domain sizes, but updates would be complex.
    Current implementation is O(n*d) which is acceptable since n decreases
    as search progresses.
    """
    min_var = None
    min_count = float('inf')

    for var in csp.variables:
        if var not in assignment:
            count = len(domains[var])
            if count < min_count:
                min_count = count
                min_var = var

    return min_var
```

---

## Error Handling

### Validation and Guard Clauses
```python
# Validate inputs early, fail fast with clear messages

def solve_nqueens(n, max_steps=1000, seed=None):
    """
    Solve n-Queens using Minimum Conflicts.

    Args:
        n: Board size (must be ≥4)
        max_steps: Maximum iterations per restart
        seed: Random seed for reproducibility

    Returns:
        Solution board if found, None otherwise

    Raises:
        ValueError: If n < 4 or max_steps <= 0
    """
    # Input validation (early returns for errors)
    if n < 4:
        raise ValueError(
            f"n-Queens requires n ≥ 4 (board size {n} too small). "
            "No solutions exist for n=2 or n=3."
        )

    if max_steps <= 0:
        raise ValueError(
            f"max_steps must be positive (got {max_steps})"
        )

    # Main algorithm after validation
    if seed is not None:
        random.seed(seed)

    # ... rest of implementation
```

### Timeout Protection (MANDATORY for Search Algorithms)
```python
import time

# All search/optimization algorithms MUST implement timeout protection

def backtracking_search(csp, max_time=300):
    """
    Backtracking search with timeout protection.

    Args:
        csp: CSP instance
        max_time: Maximum time in seconds (0 = no timeout)

    Returns:
        Tuple of (solution, stats, elapsed_time, status)
        status: "solved", "timeout", or "unsolvable"
    """
    start_time = time.perf_counter()
    stats = {'backtracks': 0, 'assignments': 0}

    def _backtrack(assignment, domains):
        # Check timeout periodically (every call is fine for backtracking)
        elapsed = time.perf_counter() - start_time
        if max_time > 0 and elapsed > max_time:
            return None, "timeout"

        # ... backtracking logic ...
        stats['backtracks'] += 1

        if is_complete(assignment):
            return assignment, "solved"

        # ... rest of algorithm

    solution, status = _backtrack({}, csp.domains)
    elapsed = time.perf_counter() - start_time

    return solution, stats, elapsed, status

# For iterative algorithms, check timeout every N iterations
def particle_swarm(objective, max_iter=1000, max_time=300):
    """PSO with timeout protection."""
    start_time = time.perf_counter()

    for iteration in range(max_iter):
        # Check timeout every iteration
        elapsed = time.perf_counter() - start_time
        if max_time > 0 and elapsed > max_time:
            return best_solution, iteration, elapsed, "timeout"

        # ... PSO iteration ...

    elapsed = time.perf_counter() - start_time
    return best_solution, max_iter, elapsed, "max_iterations"
```

### Exception Handling
```python
# Use specific exceptions, never bare except

# BAD: Catches everything, hides bugs
try:
    solution = solve_puzzle(puzzle)
except:
    return None

# GOOD: Specific exception handling
try:
    solution = solve_puzzle(puzzle)
except ValueError as e:
    print(f"Invalid puzzle format: {e}")
    return None
except TimeoutError as e:
    print(f"Solver timed out: {e}")
    return None

# For experimental code, catching broad exceptions is OK with justification
def run_algorithm_safely(algo_func, *args):
    """
    Run algorithm with safe exception handling.

    Used for batch experiments where one failure shouldn't stop all tests.
    """
    try:
        return algo_func(*args)
    except Exception as e:  # pylint: disable=broad-exception-caught
        # Broad exception OK: don't let one test crash entire experiment suite
        print(f"ERROR: {algo_func.__name__} failed: {type(e).__name__}: {e}")
        return None
```

---

## Imports

### Import Organization
```python
# Group in this order with blank lines between groups:
# 1. Standard library
# 2. Third-party packages (NumPy, Matplotlib)
# 3. Local modules

# Standard library imports (alphabetical)
import copy
import random
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

# Third-party imports (if allowed by assignment)
import matplotlib.pyplot as plt
import numpy as np

# Local application imports (alphabetical)
from nqueens_minconflicts import solve_nqueens
from sudoku_csp import solve_with_ac3
from sudoku_puzzles import EASY_PUZZLES, HARD_PUZZLES, load_puzzle

# Constants after imports
MAX_TIME_SEC = 300
```

### Import Style
```python
# Prefer specific imports for clarity

# GOOD
from collections import deque
queue = deque()

# ACCEPTABLE
import collections
queue = collections.deque()

# AVOID: Makes it unclear where symbols come from
from collections import *

# For local modules, use explicit imports
from sudoku_puzzles import EASY_PUZZLES, HARD_PUZZLES  # GOOD
from sudoku_puzzles import *                            # BAD
```

---

## Type Hints

### Function Signatures
```python
# Use type hints for all function parameters and return values

from typing import Dict, List, Optional, Set, Tuple

def solve_csp(
    problem: 'CSP',
    max_time: int = 300
) -> Optional[Dict[str, int]]:
    """
    Solve CSP with timeout protection.

    Args:
        problem: CSP instance with variables, domains, constraints
        max_time: Maximum time in seconds

    Returns:
        Solution dictionary mapping variables to values, or None if no solution
    """
    pass

# Type hints for complex data structures
def run_experiments(
    algorithms: List[Tuple[str, callable]],
    test_cases: List[Dict[str, any]],
    trials: int = 3
) -> Dict[str, List[float]]:
    """
    Run algorithm comparison experiments.

    Args:
        algorithms: List of (name, function) tuples
        test_cases: List of test case dictionaries
        trials: Number of trials per algorithm per test case

    Returns:
        Dictionary mapping algorithm names to lists of runtime results
    """
    pass

# Generic types for algorithm components
from typing import TypeVar, Generic

T = TypeVar('T')  # Generic type for state representations

def astar_search(
    initial_state: T,
    goal_test: callable,
    heuristic: callable
) -> Optional[List[T]]:
    """Generic A* search with type-safe state representation."""
    pass
```

---

## Testing Standards

### Test Organization
```python
# Organize tests by algorithm or problem type

def test_sudoku_basic_backtracking():
    """Test basic backtracking solves easy Sudoku."""
    puzzle = load_puzzle("easy_01.txt")
    solution = solve_with_backtracking(puzzle)

    assert solution is not None, "Failed to solve easy puzzle"
    assert validate_solution(solution), "Solution violates constraints"
    assert all(puzzle[i][j] == solution[i][j]
               for i in range(9) for j in range(9) if puzzle[i][j] != 0), \
           "Solution changed given cells"


def test_sudoku_ac3_faster_than_basic():
    """Test AC-3 is faster than basic backtracking on hard puzzles."""
    puzzle = load_puzzle("hard_01.txt")

    # Time basic backtracking
    start = time.perf_counter()
    solve_with_backtracking(puzzle)
    basic_time = time.perf_counter() - start

    # Time AC-3
    start = time.perf_counter()
    solve_with_ac3(puzzle)
    ac3_time = time.perf_counter() - start

    assert ac3_time < basic_time, \
           f"AC-3 ({ac3_time:.4f}s) not faster than basic ({basic_time:.4f}s)"
```

### Test Naming
```python
# Test names should describe what they test

# GOOD: Descriptive test names
def test_nqueens_n8_finds_solution():
def test_nqueens_n4_solves_in_under_100_steps():
def test_pso_rastrigin_converges_to_local_minimum():
def test_ac3_detects_unsolvable_sudoku():

# BAD: Unclear test names
def test_nqueens():
def test_1():
def test_pso():
```

### Test Independence
```python
# Each test must be completely independent
# Use fresh data structures, don't rely on test execution order

# BAD: Tests share state
global_board = None

def test_create_board():
    global global_board
    global_board = create_nqueens_board(8)
    assert len(global_board) == 8

def test_solve_board():
    # WRONG: Depends on test_create_board running first!
    solution = solve(global_board)
    assert solution is not None

# GOOD: Each test creates own data
def test_create_board():
    board = create_nqueens_board(8)
    assert len(board) == 8

def test_solve_board():
    board = create_nqueens_board(8)
    solution = solve(board)
    assert solution is not None
```

### Assertion Messages
```python
# Include helpful messages in assertions for debugging

# BAD: No context when test fails
assert solution is not None
assert len(results) == 10

# GOOD: Clear failure messages
assert solution is not None, \
       f"Failed to solve puzzle {puzzle_name} (timeout={timeout}s)"

assert len(results) == 10, \
       f"Expected 10 results, got {len(results)}: {results}"

# For complex conditions, explain what was expected
assert abs(pso_score - optimal_score) < tolerance, \
       f"PSO score {pso_score:.6f} not close to optimum {optimal_score:.6f} " \
       f"(difference: {abs(pso_score - optimal_score):.6f}, tolerance: {tolerance})"
```

---

## Algorithm Implementation

### Implement from Scratch (No External Libraries for Core Logic)
```python
# ALLOWED: Standard library and basic NumPy
import random
import time
from collections import deque
import numpy as np  # For array operations only

# FORBIDDEN: Libraries that solve the problem directly
# from sklearn.optimization import pso  # NOT ALLOWED
# from python_constraint import *       # NOT ALLOWED
# import pulp                           # NOT ALLOWED (CSP solver)

# GOOD: Implement algorithms yourself
def particle_swarm_optimization(objective_func, dimensions, bounds, swarm_size=30):
    """
    Particle Swarm Optimization implementation from scratch.

    Based on Kennedy & Eberhart (1995) and Russell & Norvig Section 4.1.3.
    Uses NumPy only for array operations, not for built-in optimization.
    """
    # Initialize particles
    swarm = np.random.uniform(
        bounds[0], bounds[1],
        size=(swarm_size, dimensions)
    )
    velocities = np.zeros((swarm_size, dimensions))

    # ... PSO implementation from textbook/lectures ...
```

### Algorithm Structure Template
```python
def algorithm_name(problem, parameters):
    """
    Algorithm description and reference.

    Based on [Source: Textbook Section X.Y, Lecture Z, Paper Author (Year)].

    Time Complexity: O(...)
    Space Complexity: O(...)

    Args:
        problem: Problem instance
        parameters: Algorithm parameters

    Returns:
        Tuple of (solution, statistics, elapsed_time, status)
    """
    # Start timing
    start_time = time.perf_counter()

    # Initialize statistics
    stats = {
        'iterations': 0,
        'backtracks': 0,
        'nodes_expanded': 0,
        # ... other metrics
    }

    # Initialize data structures
    # ... algorithm-specific initialization ...

    # Main algorithm loop
    while not termination_condition():
        # Timeout check
        elapsed = time.perf_counter() - start_time
        if max_time > 0 and elapsed > max_time:
            status = "timeout"
            break

        # Algorithm iteration
        # ... core algorithm logic ...

        stats['iterations'] += 1

    # Finalize results
    elapsed_time = time.perf_counter() - start_time

    return solution, stats, elapsed_time, status
```

---

## Experimental Code

### Experimental Scripts Structure
```python
"""
Experimental comparison of CSP solving methods.

Runs multiple algorithms on various puzzle difficulties and collects
performance statistics for analysis.

Output:
    - Console: Real-time progress and summary tables
    - CSV: results/sudoku_experiments.csv
    - Plots: results/sudoku_performance.pdf
"""

import csv
import time
from typing import List, Dict, Tuple

# Import algorithms
from sudoku_csp import (
    solve_basic,
    solve_mrv_lcv,
    solve_forward_checking,
    solve_ac3
)
from sudoku_puzzles import load_puzzles

# Experimental configuration
ALGORITHMS = [
    ("Basic Backtracking", solve_basic),
    ("MRV + LCV", solve_mrv_lcv),
    ("Forward Checking", solve_forward_checking),
    ("AC-3", solve_ac3)
]

DIFFICULTIES = ["easy", "medium", "hard"]
TRIALS_PER_PUZZLE = 5
TIMEOUT_SEC = 300


def run_single_trial(
    algorithm: callable,
    puzzle: List[List[int]],
    timeout: int
) -> Tuple[bool, float, Dict]:
    """
    Run single algorithm trial on puzzle.

    Args:
        algorithm: Solving function
        puzzle: Sudoku puzzle (9x9 grid)
        timeout: Maximum time in seconds

    Returns:
        Tuple of (success, elapsed_time, statistics)
    """
    solution, stats, elapsed, status = algorithm(puzzle, max_time=timeout)
    success = (status == "solved")
    return success, elapsed, stats


def run_experiments():
    """Run full experimental suite and save results."""
    print("=" * 70)
    print("SUDOKU CSP ALGORITHM COMPARISON")
    print("=" * 70)
    print(f"Algorithms: {len(ALGORITHMS)}")
    print(f"Difficulties: {DIFFICULTIES}")
    print(f"Trials per puzzle: {TRIALS_PER_PUZZLE}")
    print(f"Timeout: {TIMEOUT_SEC}s")
    print("=" * 70)

    results = []

    for difficulty in DIFFICULTIES:
        puzzles = load_puzzles(difficulty, limit=10)
        print(f"\n{'='*70}")
        print(f"Difficulty: {difficulty.upper()} ({len(puzzles)} puzzles)")
        print(f"{'='*70}")

        for puzzle_idx, puzzle in enumerate(puzzles):
            print(f"\nPuzzle {puzzle_idx + 1}/{len(puzzles)}:")

            for algo_name, algo_func in ALGORITHMS:
                times = []
                successes = 0

                for trial in range(TRIALS_PER_PUZZLE):
                    success, elapsed, stats = run_single_trial(
                        algo_func, puzzle, TIMEOUT_SEC
                    )

                    if success:
                        successes += 1
                        times.append(elapsed)

                # Calculate statistics
                avg_time = sum(times) / len(times) if times else float('inf')
                success_rate = successes / TRIALS_PER_PUZZLE

                # Store results
                results.append({
                    'algorithm': algo_name,
                    'difficulty': difficulty,
                    'puzzle': puzzle_idx,
                    'success_rate': success_rate,
                    'avg_time': avg_time,
                    'min_time': min(times) if times else float('inf'),
                    'max_time': max(times) if times else float('inf')
                })

                # Print progress
                print(f"  {algo_name:25s}: "
                      f"{success_rate*100:5.1f}% success, "
                      f"{avg_time:7.4f}s avg")

    # Save results to CSV
    save_results_csv(results, "results/sudoku_experiments.csv")

    # Generate comparison plots
    generate_plots(results, "results/sudoku_performance.pdf")

    print("\n" + "="*70)
    print("EXPERIMENTS COMPLETE")
    print("="*70)


def save_results_csv(results: List[Dict], filename: str):
    """Save experimental results to CSV file."""
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to: {filename}")


def generate_plots(results: List[Dict], filename: str):
    """Generate performance comparison plots."""
    # ... plotting code using matplotlib ...
    print(f"Plots saved to: {filename}")


if __name__ == '__main__':
    run_experiments()
```

### Random Seed Documentation
```python
# Document random seeds for reproducibility

def run_experiment_suite(seed=42):
    """
    Run complete experimental suite with reproducible randomness.

    Args:
        seed: Random seed for reproducibility (default: 42)
              Use different seeds for multiple independent runs:
              - seed=42: Main experimental results (reported in writeup)
              - seed=100: Validation run #1
              - seed=200: Validation run #2

    Returns:
        Dictionary of experimental results
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    print(f"Experimental Configuration:")
    print(f"  Random seed: {seed}")
    print(f"  NumPy version: {np.__version__}")
    print(f"  Python version: {sys.version}")

    # ... experiments ...
```

---

## Code Quality Checklist

### Before Every Commit
- [ ] All tests pass (100% success rate)
- [ ] Pylint score ≥8.0/10 (target ≥9.0/10)
- [ ] No TODO/FIXME comments without name and explanation
- [ ] No commented-out code (delete or explain why kept)
- [ ] No debug print statements (use logging or remove)
- [ ] All functions have docstrings with references
- [ ] Timeout protection for all search/optimization algorithms
- [ ] No Unicode characters (ASCII only for Windows compatibility)
- [ ] Type hints added to new functions
- [ ] README.md updated with AI disclosure

### Before Submission
- [ ] All algorithm sources cited in comments
- [ ] Complexity analysis documented
- [ ] Experimental results reproducible with documented seeds
- [ ] Output saved to log files
- [ ] Pylint output saved (e.g., pylint_HW02.txt)
- [ ] Code works on clean Python environment
- [ ] No hardcoded paths (use relative paths)
- [ ] AI disclosure in README and writeup
- [ ] All required files present per assignment spec

### Code Review (Self-Check)
- [ ] Code follows this style guide
- [ ] Algorithm implementation matches textbook/lecture description
- [ ] Variable names are clear and meaningful
- [ ] Functions are focused and not too long (<50 lines ideal)
- [ ] No code duplication (extract common functionality)
- [ ] Error handling is appropriate
- [ ] Performance is reasonable (no obvious inefficiencies)
- [ ] Code is readable and maintainable

---

## Academic Integrity

### AI Tool Usage Disclosure
When using AI tools (Claude Code, GitHub Copilot, ChatGPT, etc.), you MUST:

1. **Disclose in README.md**:
```markdown
## AI Disclosure

This code was generated with assistance from **Claude Code (Sonnet 4.5)**,
version **claude-sonnet-4-5-20250929**.

The AI assistant helped with:
- Understanding AC-3 algorithm from textbook and lecture slides
- Implementing backtracking search with MRV and LCV heuristics
- Debugging timeout protection logic
- Writing comprehensive docstrings and comments
- Creating run_experiments.py for automated testing
- Formatting output for LaTeX tables

All code was reviewed, understood, and tested by the student.
```

2. **Disclose in LaTeX Writeup**:
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
```

### What You Must Understand
Even if AI helps you write code, you must be able to:
- Explain how every algorithm works
- Justify time/space complexity claims
- Debug and fix issues independently
- Answer questions about implementation choices
- Reproduce results without AI assistance

---

## Resources

### Course References
- **Textbook**: Russell & Norvig (2020). *Artificial Intelligence: A Modern Approach*, 4th Edition
- **Lecture Slides**: Available on Canvas
- **Course Syllabus**: CS4820_Syllabus.pdf

### Python Style
- [PEP 8 – Style Guide for Python Code](https://peps.python.org/pep-0008/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)

### Testing
- [Python unittest documentation](https://docs.python.org/3/library/unittest.html)
- [pytest documentation](https://docs.pytest.org/)

### Type Hints
- [PEP 484 – Type Hints](https://peps.python.org/pep-0484/)
- [typing module documentation](https://docs.python.org/3/library/typing.html)

---

## Changelog

### Version 1.0 (November 1, 2025)
- Initial style guide for CS4820/5820
- Adapted from Language Learning Platform style guide
- Focused on AI algorithm implementation and academic standards
- Added mandatory algorithm citation requirements
- Added complexity analysis documentation standards
- Added experimental reproducibility guidelines
- Added timeout protection requirements
- Added academic integrity and AI disclosure sections

---

**Questions or Issues?**

Contact: josh.manchester@uccs.edu
Review: CLAUDE.md for project-specific guidelines
This guide is a living document and may be updated as course progresses.
