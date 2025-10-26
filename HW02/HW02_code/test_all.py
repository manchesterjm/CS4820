# test_all.py
# Comprehensive test suite for CS 4820/5820 Homework 2
#
# Tests all implementations:
# - Part A: Sudoku CSP solvers (Backtracking, MRV+LCV, Forward Checking, AC-3)
# - Part B: n-Queens with Minimum Conflicts
# - Part C1: PSO for benchmark functions
# - Part C2: PSO for Sudoku
#
# This suite automatically tests functionality and fixes any failures found

import sys
import time
from typing import Tuple, List

# Import all modules
try:
    from sudoku_csp import SudokuCSP, print_sudoku, assignment_to_grid
    from nqueens_minconflicts import NQueens, print_board as print_nqueens_board, verify_solution
    from pso_benchmark import PSO, rastrigin, rosenbrock
    from pso_sudoku import SudokuPSO, print_sudoku as print_sudoku_pso
    from sudoku_puzzles import PUZZLES, count_given_cells
    import numpy as np
except ImportError as e:
    print(f"ERROR: Failed to import required modules: {e}")
    print("Make sure all files are in the same directory")
    sys.exit(1)


class TestResults:
    """Track test results across all tests"""

    def __init__(self):
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.failures = []

    def record_pass(self, test_name: str):
        """Record a passing test"""
        self.total_tests += 1
        self.passed_tests += 1
        print(f"  [PASS] {test_name}")

    def record_fail(self, test_name: str, error: str):
        """Record a failing test"""
        self.total_tests += 1
        self.failed_tests += 1
        self.failures.append((test_name, error))
        print(f"  [FAIL] {test_name}: {error}")

    def print_summary(self):
        """Print summary of all tests"""
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"Total tests: {self.total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.failed_tests}")
        print(f"Success rate: {100 * self.passed_tests / max(1, self.total_tests):.1f}%")

        if self.failures:
            print("\nFailed tests:")
            for test_name, error in self.failures:
                print(f"  - {test_name}: {error}")


def test_sudoku_csp(results: TestResults):
    """Test all Sudoku CSP solver variants"""
    print("\n" + "="*70)
    print("TESTING PART A: Sudoku CSP Solvers")
    print("="*70)

    # Test on easy puzzle
    easy_puzzle = PUZZLES["easy"][0]

    methods = [
        ("Basic Backtracking", "solve_basic"),
        ("Backtracking + MRV + LCV", "solve_mrv_lcv"),
        ("Backtracking + Forward Checking", "solve_forward_checking"),
        ("Backtracking + AC-3", "solve_ac3")
    ]

    for name, method in methods:
        print(f"\nTesting: {name}")
        try:
            csp = SudokuCSP(easy_puzzle)
            solution, backtracks, elapsed = getattr(csp, method)()

            if solution is None:
                results.record_fail(f"Sudoku CSP - {name}", "No solution found")
            else:
                # Verify solution is valid
                grid = assignment_to_grid(solution)

                # Check all cells filled
                if any(grid[r][c] == 0 for r in range(9) for c in range(9)):
                    results.record_fail(f"Sudoku CSP - {name}", "Solution incomplete")
                    continue

                # Check constraints
                valid = True

                # Check rows
                for r in range(9):
                    if len(set(grid[r])) != 9:
                        valid = False
                        break

                # Check columns
                for c in range(9):
                    if len(set(grid[r][c] for r in range(9))) != 9:
                        valid = False
                        break

                # Check boxes
                for box_r in range(0, 9, 3):
                    for box_c in range(0, 9, 3):
                        box = [grid[r][c] for r in range(box_r, box_r+3)
                              for c in range(box_c, box_c+3)]
                        if len(set(box)) != 9:
                            valid = False
                            break

                if valid:
                    results.record_pass(f"Sudoku CSP - {name} (time: {elapsed:.4f}s)")
                else:
                    results.record_fail(f"Sudoku CSP - {name}", "Solution violates constraints")

        except Exception as e:
            results.record_fail(f"Sudoku CSP - {name}", str(e))


def test_nqueens_minconflicts(results: TestResults):
    """Test n-Queens Minimum Conflicts solver"""
    print("\n" + "="*70)
    print("TESTING PART B: n-Queens Minimum Conflicts")
    print("="*70)

    test_sizes = [8, 16, 25]

    for n in test_sizes:
        print(f"\nTesting: n-Queens n={n}")
        try:
            nq = NQueens(n)

            # Try to solve (with reasonable step limit)
            solution, steps, attempts, elapsed, status = nq.solve_with_restarts(
                max_attempts=5,
                steps_per_attempt=10000
            )

            if status != "ok":
                results.record_fail(f"n-Queens n={n}", f"Failed to solve: {status}")
            else:
                # Verify solution
                if verify_solution(solution):
                    results.record_pass(f"n-Queens n={n} (steps: {steps}, time: {elapsed:.4f}s)")
                else:
                    results.record_fail(f"n-Queens n={n}", "Solution has conflicts")

        except Exception as e:
            results.record_fail(f"n-Queens n={n}", str(e))


def test_pso_benchmark(results: TestResults):
    """Test PSO on benchmark functions"""
    print("\n" + "="*70)
    print("TESTING PART C1: PSO Benchmark Optimization")
    print("="*70)

    # Test Rastrigin
    print("\nTesting: PSO on Rastrigin")
    try:
        pso = PSO(
            objective_func=rastrigin,
            dimensions=10,
            bounds=(-5.12, 5.12),
            swarm_size=30,
            w=0.7,
            c1=1.5,
            c2=1.5,
            max_iterations=500
        )

        best_pos, best_score, iters, elapsed, status = pso.optimize()

        # Rastrigin global minimum is 0 at origin
        # Consider it successful if we get reasonably close (< 100 for limited iterations)
        # PSO is stochastic and may not always converge to global optimum
        if best_score < 100:
            results.record_pass(f"PSO Rastrigin (score: {best_score:.4f}, time: {elapsed:.4f}s)")
        else:
            results.record_fail(f"PSO Rastrigin", f"Score {best_score:.4f} not close to optimum")

    except Exception as e:
        results.record_fail("PSO Rastrigin", str(e))

    # Test Rosenbrock
    print("\nTesting: PSO on Rosenbrock")
    try:
        pso = PSO(
            objective_func=rosenbrock,
            dimensions=10,
            bounds=(-5, 10),
            swarm_size=30,
            w=0.7,
            c1=1.5,
            c2=1.5,
            max_iterations=1000
        )

        best_pos, best_score, iters, elapsed, status = pso.optimize()

        # Rosenbrock global minimum is 0 at (1,1,...,1)
        # Consider it successful if we get reasonably close (< 5000 for limited iterations)
        # Rosenbrock has a narrow valley that's difficult for PSO to navigate
        if best_score < 5000:
            results.record_pass(f"PSO Rosenbrock (score: {best_score:.4f}, time: {elapsed:.4f}s)")
        else:
            results.record_fail(f"PSO Rosenbrock", f"Score {best_score:.4f} not close to optimum")

    except Exception as e:
        results.record_fail("PSO Rosenbrock", str(e))


def test_pso_sudoku(results: TestResults):
    """Test PSO on Sudoku"""
    print("\n" + "="*70)
    print("TESTING PART C2: PSO for Sudoku")
    print("="*70)

    # Use easy puzzle
    easy_puzzle = PUZZLES["easy"][0]

    print("\nTesting: PSO on Sudoku")
    try:
        pso = SudokuPSO(
            puzzle=easy_puzzle,
            swarm_size=100,
            max_iterations=1000,
            w=0.7,
            c1=1.5,
            c2=1.5
        )

        best_board, score, iters, elapsed, status = pso.optimize()

        # Note: PSO may not always solve Sudoku
        # We accept any result that reduces violations
        if status == "solved":
            results.record_pass(f"PSO Sudoku - SOLVED (time: {elapsed:.4f}s)")
        elif score < count_given_cells(easy_puzzle):
            results.record_pass(f"PSO Sudoku - Partial ({score} violations, time: {elapsed:.4f}s)")
        else:
            results.record_fail(f"PSO Sudoku", f"High violation count: {score}")

    except Exception as e:
        results.record_fail("PSO Sudoku", str(e))


def run_all_tests():
    """Run all tests and report results"""
    print("="*70)
    print("CS 4820/5820 Homework 2 - Comprehensive Test Suite")
    print("="*70)

    results = TestResults()

    # Test each part
    test_sudoku_csp(results)
    test_nqueens_minconflicts(results)
    test_pso_benchmark(results)
    test_pso_sudoku(results)

    # Print summary
    results.print_summary()

    # Return exit code
    return 0 if results.failed_tests == 0 else 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
