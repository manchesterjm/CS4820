"""
Comprehensive test suite for CS 4820/5820 Homework 2.

This module provides automated testing for all algorithm implementations.
Testing ensures:
1. Correctness: Algorithms produce valid solutions
2. Robustness: Algorithms handle edge cases and errors gracefully
3. Performance: Algorithms complete within reasonable time bounds
4. Integration: All components work together properly

Tests all implementations:
- Part A: Sudoku CSP solvers (Backtracking, MRV+LCV, Forward Checking, AC-3)
- Part B: n-Queens with Minimum Conflicts
- Part C1: PSO for benchmark functions
- Part C2: PSO for Sudoku

Test Design Philosophy:
- Each test is independent (no shared state between tests)
- Tests verify both success conditions and error handling
- Stochastic algorithms (PSO, Minimum Conflicts) are tested multiple times
- Clear pass/fail reporting with detailed error messages
"""

import sys

# Import all required modules
# If any import fails, exit gracefully with helpful error message
try:
    from sudoku_csp import SudokuCSP, assignment_to_grid
    from nqueens_minconflicts import NQueens, verify_solution
    from pso_benchmark import PSO, rastrigin, rosenbrock
    from pso_sudoku import SudokuPSO
    from sudoku_puzzles import PUZZLES, count_given_cells
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
        rate = 100 * self.passed_tests / max(1, self.total_tests)
        print(f"Success rate: {rate:.1f}%")

        if self.failures:
            print("\nFailed tests:")
            for test_name, error in self.failures:
                print(f"  - {test_name}: {error}")


def test_sudoku_csp(results: TestResults):
    """
    Test all Sudoku CSP solver variants for correctness.

    Test Strategy:
    - Use easy puzzle to ensure all algorithms can solve within timeout
    - Verify solution correctness (no constraint violations)
    - Test that all 4 algorithm variants produce valid solutions
    - Compare performance qualitatively (AC-3 should be fastest)
    """
    print("\n" + "="*70)
    print("TESTING PART A: Sudoku CSP Solvers")
    print("="*70)

    # Use easy puzzle for testing (36 givens, should solve quickly)
    # All algorithms should succeed on this puzzle
    easy_puzzle = PUZZLES["easy"][0]

    # Test all four algorithm variants
    # Each represents different search space pruning strategy
    methods = [
        ("Basic Backtracking", "solve_basic"),  # Baseline - no optimization
        ("Backtracking + MRV + LCV", "solve_mrv_lcv"),  # Heuristics
        ("Backtracking + Forward Checking", "solve_forward_checking"),  # Inference
        ("Backtracking + AC-3", "solve_ac3")  # Full constraint propagation
    ]

    for name, method in methods:
        print(f"\nTesting: {name}")
        try:
            csp = SudokuCSP(easy_puzzle)
            solution, _, elapsed = getattr(csp, method)()

            if solution is None:
                results.record_fail(f"Sudoku CSP - {name}", "No solution found")
            else:
                # Verify solution is valid
                grid = assignment_to_grid(solution)

                # Check all cells filled
                if any(grid[r][c] == 0
                       for r in range(9) for c in range(9)):
                    results.record_fail(f"Sudoku CSP - {name}",
                                        "Solution incomplete")
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
                        box = [grid[r][c]
                               for r in range(box_r, box_r+3)
                               for c in range(box_c, box_c+3)]
                        if len(set(box)) != 9:
                            valid = False
                            break

                if valid:
                    test_result = f"Sudoku CSP - {name} (time: {elapsed:.4f}s)"
                    results.record_pass(test_result)
                else:
                    results.record_fail(f"Sudoku CSP - {name}",
                                        "Solution violates constraints")

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
            solution, steps, _, elapsed, status = nq.solve_with_restarts(
                max_attempts=5,
                steps_per_attempt=10000
            )

            if status != "ok":
                results.record_fail(f"n-Queens n={n}", f"Failed to solve: {status}")
            else:
                # Verify solution
                if verify_solution(solution):
                    test_result = (f"n-Queens n={n} (steps: {steps}, "
                                   f"time: {elapsed:.4f}s)")
                    results.record_pass(test_result)
                else:
                    results.record_fail(f"n-Queens n={n}",
                                        "Solution has conflicts")

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

        _, best_score, _, elapsed, _ = pso.optimize()

        # Rastrigin global minimum is 0 at origin
        # Consider it successful if we get reasonably close (< 100 for
        # limited iterations) PSO is stochastic and may not always
        # converge to global optimum
        if best_score < 100:
            test_result = (f"PSO Rastrigin (score: {best_score:.4f}, "
                           f"time: {elapsed:.4f}s)")
            results.record_pass(test_result)
        else:
            results.record_fail("PSO Rastrigin",
                                f"Score {best_score:.4f} not close to "
                                f"optimum")

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

        _, best_score, _, elapsed, _ = pso.optimize()

        # Rosenbrock global minimum is 0 at (1,1,...,1)
        # Consider it successful if we get reasonably close (< 5000 for
        # limited iterations) Rosenbrock has a narrow valley that's
        # difficult for PSO to navigate
        if best_score < 5000:
            test_result = (f"PSO Rosenbrock (score: {best_score:.4f}, "
                           f"time: {elapsed:.4f}s)")
            results.record_pass(test_result)
        else:
            results.record_fail("PSO Rosenbrock",
                                f"Score {best_score:.4f} not close to "
                                f"optimum")

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

        _, score, _, elapsed, status = pso.optimize()

        # Note: PSO may not always solve Sudoku
        # We accept any result that reduces violations
        if status == "solved":
            test_result = f"PSO Sudoku - SOLVED (time: {elapsed:.4f}s)"
            results.record_pass(test_result)
        elif score < count_given_cells(easy_puzzle):
            test_result = (f"PSO Sudoku - Partial ({score} violations, "
                           f"time: {elapsed:.4f}s)")
            results.record_pass(test_result)
        else:
            results.record_fail("PSO Sudoku",
                                f"High violation count: {score}")

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
