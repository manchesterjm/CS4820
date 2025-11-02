"""
Main experiment runner for CS 4820/5820 Homework 2.

This module serves as the centralized experimental framework for all homework parts.
It runs standardized tests across all algorithms, collects performance metrics,
and generates formatted tables ready for inclusion in an AAAI format report.

Experimental Structure:
    Part A: Sudoku CSP Comparison
        - Tests 4 algorithms: Basic Backtracking, +MRV+LCV, +Forward Checking, +AC-3
        - Runs on easy, medium, and hard puzzles
        - Measures: runtime, success rate
        - Purpose: Compare search space pruning effectiveness

    Part B: n-Queens Local Search
        - Algorithm: Minimum Conflicts with random restarts
        - Tests on: n=8, 16, 25 (5 trials each)
        - Measures: steps to solution, runtime, success rate
        - Purpose: Demonstrate local search efficiency

    Part C1: PSO Benchmark Optimization
        - Functions: Rastrigin (multimodal), Rosenbrock (narrow valley)
        - Tests 3 parameter configurations (3 trials each)
        - Measures: best/average fitness, convergence time
        - Purpose: PSO parameter sensitivity analysis

    Part C2: PSO for Sudoku
        - Treats Sudoku as optimization (minimize violations)
        - Runs 3 trials on easy puzzle
        - Measures: final violations, iterations, runtime
        - Purpose: Demonstrate metaheuristic for constraint satisfaction

Usage:
    python run_experiments.py

    Runtime: Approximately 5-8 minutes depending on system
    Output: Formatted tables with statistics for report inclusion
"""

import time
import numpy as np

# Import CSP implementations
from sudoku_csp import SudokuCSP, print_sudoku
from nqueens_minconflicts import NQueens, verify_solution
# Import PSO implementations
from pso_benchmark import PSO, rastrigin, rosenbrock
from pso_sudoku import SudokuPSO
# Import test puzzles
from sudoku_puzzles import PUZZLES, count_given_cells


def run_part_a_experiments():
    """
    Part A: Compare Sudoku CSP solver variants on different difficulty levels.

    This experiment demonstrates how different search space pruning techniques
    affect CSP solving performance. We test 4 algorithm variants:
    1. Basic Backtracking - Baseline with no optimization
    2. +MRV+LCV - Add variable/value ordering heuristics
    3. +Forward Checking - Add inference to maintain arc consistency with future
    4. +AC-3 - Full constraint propagation before and during search

    Experimental Design:
        - Test each algorithm on easy, medium, and hard puzzles
        - Use first puzzle from each difficulty category
        - Record runtime and success status
        - Expected outcome: AC-3 fastest, basic slowest

    Metrics:
        - Runtime (seconds): Wall-clock time to solve
        - Given cells: Number of initial clues (difficulty indicator)
        - Status: PASS (solved) or FAIL (timeout/unsolvable)
    """
    print("\n" + "="*80)
    print("PART A: Sudoku as a CSP")
    print("="*80)

    # Define algorithm methods to test
    # Format: (Display Name, Method Name)
    # Method names correspond to SudokuCSP class methods
    methods = [
        ("Basic Backtracking", "solve_basic"),  # Baseline - no optimization
        ("Backtracking + MRV + LCV", "solve_mrv_lcv"),  # Add heuristics
        ("Backtracking + Forward Checking", "solve_forward_checking"),  # Add inference
        ("Backtracking + AC-3", "solve_ac3")  # Full constraint propagation
    ]

    # Test on three difficulty levels to show scaling behavior
    difficulties = ["easy", "medium", "hard"]

    # Print formatted table header
    print("\n{:<30} {:<10} {:<15} {:>12}".format(
        "Algorithm", "Difficulty", "Given Cells", "Time (s)"
    ))
    print("-" * 80)

    # Run experiments: For each difficulty, test all algorithms
    for diff in difficulties:
        # Use first puzzle from each difficulty level
        puzzle = PUZZLES[diff][0]
        # Count given cells as proxy for difficulty
        given = count_given_cells(puzzle)

        # Test each algorithm on this puzzle
        for name, method in methods:
            # Create fresh CSP instance for each trial
            csp = SudokuCSP(puzzle)
            # Call the specified solver method using getattr for dynamic dispatch
            # Returns: (solution grid or None, stats dict, elapsed time)
            solution, _, elapsed = getattr(csp, method)()

            # Determine success status
            status = "PASS" if solution else "FAIL"
            # Print formatted result row
            print("{:<30} {:<10} {:>15} {:>12.6f} {}".format(
                name, diff, given, elapsed, status
            ))

        # Add blank line between difficulty levels for readability
        print()

    # Explain results for report discussion
    print("\nInterpretation:")
    print("- Basic Backtracking: Naive DFS with no heuristics")
    print("- +MRV+LCV: Variable and value ordering heuristics")
    print("- +Forward Checking: Inference after each assignment")
    print("- +AC-3: Full constraint propagation")
    print("\nExpected: AC-3 should be fastest due to aggressive pruning")


def run_part_b_experiments():
    """
    Part B: n-Queens with Minimum Conflicts local search.

    This experiment demonstrates that local search with random restarts can
    efficiently solve n-Queens, often in O(n) steps. Unlike backtracking CSP,
    local search starts with a complete (but conflicted) assignment and
    iteratively improves it by minimizing conflicts.

    Why Minimum Conflicts works well for n-Queens:
        - High solution density (many solutions exist)
        - Local minima are rare
        - Simple conflict metric guides search effectively

    Experimental Design:
        - Test on n=8, 16, 25 to show linear scaling
        - Run 5 trials per size (handle randomness)
        - Allow up to 10 restart attempts with 100,000 steps each
        - Report aggregated statistics

    Metrics:
        - Success rate: Fraction of trials that found valid solution
        - Average steps: Mean steps to solution (among successes)
        - Average time: Mean wall-clock time (among successes)
    """
    print("\n" + "="*80)
    print("PART B: n-Queens with Minimum Conflicts")
    print("="*80)

    # Test board sizes - chosen to show scaling behavior
    test_sizes = [8, 16, 25]  # Standard, moderate, large
    trials_per_size = 5  # Multiple trials to account for randomness

    # Print formatted table header
    print("\n{:<10} {:<10} {:<15} {:<15} {:<15}".format(
        "n", "Trials", "Success Rate", "Avg Steps", "Avg Time (s)"
    ))
    print("-" * 80)

    # Run experiments for each board size
    for n in test_sizes:
        # Create n-Queens instance
        nq = NQueens(n)

        # Track statistics across trials
        successful = 0  # Count of successful solutions
        total_steps = 0  # Sum of steps for successful trials
        total_time = 0.0  # Sum of time for successful trials

        # Run multiple trials to handle randomness
        for _ in range(trials_per_size):
            # Solve with random restarts allowed
            # max_attempts=10: Try up to 10 random initializations
            # steps_per_attempt=100000: Max steps per initialization
            solution, steps, _, elapsed, status = nq.solve_with_restarts(
                max_attempts=10,
                steps_per_attempt=100000
            )

            # Check if trial succeeded
            if status == "ok" and verify_solution(solution):
                successful += 1
                total_steps += steps  # Accumulate for averaging
                total_time += elapsed

        # Calculate statistics (only over successful trials)
        success_rate = f"{successful}/{trials_per_size}"
        # Use max(1, successful) to avoid division by zero if all trials failed
        avg_steps = total_steps / max(1, successful)
        avg_time = total_time / max(1, successful)

        # Print formatted result row
        print("{:<10} {:<10} {:<15} {:<15.1f} {:<15.6f}".format(
            n, trials_per_size, success_rate, avg_steps, avg_time
        ))

    # Explain results for report discussion
    print("\nInterpretation:")
    print("- Minimum Conflicts is a local search heuristic")
    print("- Typically solves in O(n) steps regardless of board size")
    print("- Much faster than backtracking for large n")
    print("- Success rate should be very high (close to 100%)")


def run_part_c1_experiments():
    """
    Part C1: PSO parameter sensitivity analysis on benchmark functions.

    This experiment tests how PSO hyperparameters affect optimization performance
    on two standard benchmarks with different characteristics:
        - Rastrigin: Highly multimodal (many local minima), tests exploration
        - Rosenbrock: Narrow parabolic valley, tests exploitation/precision

    PSO Parameters Being Tested:
        - swarm_size: Number of particles (larger = more exploration, slower)
        - w (inertia): Balance exploration vs exploitation (0.4-0.9 typical)
        - c1 (cognitive): Attraction to personal best (1.2-2.0 typical)
        - c2 (social): Attraction to global best (1.2-2.0 typical)

    Experimental Design:
        - 3 parameter configurations with different tradeoffs
        - 3 trials per config (handle stochastic nature of PSO)
        - Report best and average fitness across trials
        - Compare convergence times

    Insights:
        - High w (e.g., 0.9): More exploration, better for multimodal
        - Low w (e.g., 0.5): More exploitation, better for unimodal
        - High c1/c2 (e.g., 2.0): Faster convergence, risk premature convergence
        - Larger swarm: Better coverage, but slower per iteration
    """
    print("\n" + "="*80)
    print("PART C1: PSO for Benchmark Optimization")
    print("="*80)

    # Define parameter configurations to test
    # Each config tests different exploration/exploitation balance
    configs = [
        {"name": "Config 1 (Standard)", "swarm_size": 30, "w": 0.7,
         "c1": 1.5, "c2": 1.5, "max_iterations": 1000},  # Balanced parameters
        {"name": "Config 2 (Large Swarm)", "swarm_size": 50, "w": 0.5,
         "c1": 2.0, "c2": 2.0, "max_iterations": 1000},  # More exploitation
        {"name": "Config 3 (High Inertia)", "swarm_size": 40, "w": 0.9,
         "c1": 1.2, "c2": 1.2, "max_iterations": 1500},  # More exploration
    ]

    trials = 3  # Run multiple trials to account for randomness

    # =========================================================================
    # TEST 1: Rastrigin Function (Highly Multimodal)
    # =========================================================================
    # Rastrigin has many local minima in a regular pattern
    # Formula: f(x) = 10n + sum(x_i^2 - 10*cos(2*pi*x_i))
    # Global minimum: f(0,...,0) = 0
    # Challenge: Escape local minima through exploration
    print("\n" + "-"*80)
    print("Benchmark: Rastrigin Function (10D)")
    print("Global minimum: f(0,...,0) = 0")
    print("-"*80)

    # Print table header
    print("\n{:<25} {:<15} {:<15} {:<15}".format(
        "Configuration", "Best Score", "Avg Score", "Avg Time (s)"
    ))
    print("-" * 80)

    # Test each parameter configuration
    for config in configs:
        # Extract name for display (will be restored after)
        name = config.pop("name")
        scores = []  # Track fitness values across trials
        times = []  # Track runtimes across trials

        # Run multiple trials for statistical reliability
        for _ in range(trials):
            # Create PSO instance with Rastrigin objective
            # dimensions=10: 10-dimensional search space
            # bounds=(-5.12, 5.12): Standard Rastrigin search bounds
            pso = PSO(
                objective_func=rastrigin,
                dimensions=10,
                bounds=(-5.12, 5.12),
                **config  # Unpack swarm_size, w, c1, c2, max_iterations
            )
            # Run optimization
            # Returns: (best_position, best_score, iterations, elapsed, status)
            _, best_score, _, elapsed, _ = pso.optimize()
            scores.append(best_score)  # Lower is better (minimization)
            times.append(elapsed)

        config["name"] = name  # Restore name for next iteration

        # Report aggregated statistics
        print("{:<25} {:<15.6f} {:<15.6f} {:<15.6f}".format(
            name,
            min(scores),  # Best (lowest) fitness found
            np.mean(scores),  # Average fitness across trials
            np.mean(times)  # Average runtime
        ))

    # =========================================================================
    # TEST 2: Rosenbrock Function (Narrow Valley)
    # =========================================================================
    # Rosenbrock forms a narrow parabolic valley leading to the global minimum
    # Formula: f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (x_i - 1)^2)
    # Global minimum: f(1,...,1) = 0
    # Challenge: Navigate narrow valley without overshooting (needs precision)
    print("\n" + "-"*80)
    print("Benchmark: Rosenbrock Function (10D)")
    print("Global minimum: f(1,...,1) = 0")
    print("-"*80)

    # Print table header
    print("\n{:<25} {:<15} {:<15} {:<15}".format(
        "Configuration", "Best Score", "Avg Score", "Avg Time (s)"
    ))
    print("-" * 80)

    # Test each parameter configuration
    for config in configs:
        # Extract name for display
        name = config.pop("name")
        scores = []  # Track fitness values across trials
        times = []  # Track runtimes across trials

        # Run multiple trials for statistical reliability
        for _ in range(trials):
            # Create PSO instance with Rosenbrock objective
            # dimensions=10: 10-dimensional search space
            # bounds=(-5, 10): Asymmetric bounds (global min at x=1)
            pso = PSO(
                objective_func=rosenbrock,
                dimensions=10,
                bounds=(-5, 10),
                **config  # Unpack swarm_size, w, c1, c2, max_iterations
            )
            # Run optimization
            _, best_score, _, elapsed, _ = pso.optimize()
            scores.append(best_score)  # Lower is better
            times.append(elapsed)

        config["name"] = name  # Restore name for next iteration

        # Report aggregated statistics
        print("{:<25} {:<15.6f} {:<15.6f} {:<15.6f}".format(
            name,
            min(scores),  # Best (lowest) fitness found
            np.mean(scores),  # Average fitness across trials
            np.mean(times)  # Average runtime
        ))

    print("\nInterpretation:")
    print("- Rastrigin: Highly multimodal, tests ability to escape "
          "local minima")
    print("- Rosenbrock: Narrow valley, tests convergence precision")
    print("- Higher inertia (w) = more exploration, may escape local "
          "minima")
    print("- Higher c1/c2 = stronger attraction to bests, faster "
          "convergence")


def run_part_c2_experiments():
    """
    Part C2: PSO applied to Sudoku as a constraint optimization problem.

    This experiment demonstrates applying a population-based metaheuristic (PSO)
    to a traditionally CSP domain. Instead of hard constraints, we treat Sudoku
    as an optimization problem: minimize the number of constraint violations.

    PSO Adaptation for Sudoku:
        - Each particle = complete 9x9 board (not partial assignment)
        - Fixed cells locked in place (given clues)
        - Row constraints satisfied by design (permutations)
        - Fitness = count of column + box violations
        - Discrete PSO: Use swap operations instead of continuous velocity

    Why This Matters:
        - Shows metaheuristics can approximate solutions to hard problems
        - Useful when exact solution not required or CSP too constrained
        - Illustrates tradeoffs: CSP guarantees solution, PSO may not converge
        - Educational: Contrasts systematic search (backtracking) vs stochastic

    Experimental Design:
        - Test on easy puzzle (36 givens)
        - Run 3 trials (PSO is stochastic)
        - Larger swarm (150) and more iterations (3000) due to discrete nature
        - Report: violations (0 = solved), iterations used, runtime

    Expected Outcome:
        - PSO may or may not reach 0 violations
        - Demonstrates that metaheuristics are not guaranteed solvers
        - For reliable Sudoku solving, use Part A CSP methods
    """
    print("\n" + "="*80)
    print("PART C2: PSO for Sudoku Optimization")
    print("="*80)

    # Use easy puzzle for testing PSO
    puzzle = PUZZLES["easy"][0]
    trials = 3  # Multiple trials since PSO is stochastic

    # Display test puzzle
    print("\nTest Puzzle:")
    print_sudoku(puzzle)
    print(f"Given cells: {count_given_cells(puzzle)}")

    # Print table header
    print("\n{:<10} {:<20} {:<15} {:<15}".format(
        "Trial", "Final Violations", "Iterations", "Time (s)"
    ))
    print("-" * 80)

    # Track statistics across trials
    scores = []  # Violation counts
    times = []  # Runtimes
    iterations = []  # Iterations used

    # Run multiple trials
    for trial in range(1, trials + 1):
        # Create Sudoku PSO instance
        # swarm_size=150: Larger swarm for discrete optimization
        # max_iterations=3000: More iterations than continuous PSO
        # w, c1, c2: Standard PSO parameters
        pso = SudokuPSO(
            puzzle=puzzle,
            swarm_size=150,
            max_iterations=3000,
            w=0.7,  # Moderate inertia
            c1=1.5,  # Cognitive attraction
            c2=1.5  # Social attraction
        )

        # Run optimization
        # Returns: (best_board, violation_count, iterations, elapsed, status)
        _, score, iters, elapsed, status = pso.optimize()

        # Record metrics
        scores.append(score)  # Lower is better (0 = perfect solution)
        times.append(elapsed)
        iterations.append(iters)

        # Format status string
        status_str = "SOLVED" if status == "solved" else f"{status}"
        # Print trial results
        print("{:<10} {:<20} {:<15} {:<15.6f}  {}".format(
            trial, score, iters, elapsed, status_str
        ))

    # Print summary statistics
    print("-" * 80)
    print("{:<10} {:<20.2f} {:<15.1f} {:<15.6f}".format(
        "Average", np.mean(scores), np.mean(iterations), np.mean(times)
    ))

    # Explain results for report discussion
    print("\nInterpretation:")
    print("- PSO treats Sudoku as an optimization problem "
          "(minimize violations)")
    print("- May not always find perfect solution (0 violations)")
    print("- Demonstrates metaheuristic approach to constraint problems")
    print("- For guaranteed solutions, use CSP methods (Part A)")


def main():
    """
    Main experimental orchestrator - runs all homework parts sequentially.

    This function coordinates the entire experimental workflow:
    1. Part A: CSP algorithm comparison (4 algorithms x 3 difficulties)
    2. Part B: n-Queens local search (3 board sizes x 5 trials)
    3. Part C1: PSO benchmarks (2 functions x 3 configs x 3 trials)
    4. Part C2: PSO on Sudoku (3 trials)

    Total runtime: Approximately 5-8 minutes depending on system performance

    Output:
        - Formatted tables printed to console
        - Statistics ready for copy-paste into AAAI report
        - Comparison metrics for algorithm analysis
    """
    # Print experiment header
    print("="*80)
    print("CS 4820/5820 Homework 2 - Full Experimental Results")
    print("="*80)

    # Start overall timer
    start_time = time.perf_counter()

    # Run all experimental parts sequentially
    # Each function is self-contained: sets up, runs, reports results
    run_part_a_experiments()  # CSP comparison
    run_part_b_experiments()  # n-Queens local search
    run_part_c1_experiments()  # PSO parameter analysis
    run_part_c2_experiments()  # PSO on Sudoku

    # Calculate total elapsed time
    total_time = time.perf_counter() - start_time

    # Print experiment footer
    print("\n" + "="*80)
    print(f"All experiments completed in {total_time:.2f} seconds")
    print("="*80)

    # Remind user about next steps
    print("\nResults ready for inclusion in AAAI format report")
    print("Copy relevant tables and statistics to your LaTeX writeup")


if __name__ == "__main__":
    # =========================================================================
    # EXPERIMENT ENTRY POINT
    # =========================================================================
    # Run this script directly to execute all experiments:
    #     python run_experiments.py
    #
    # Optional: Set random seeds for reproducible results
    # Uncomment these lines if you need deterministic behavior:
    #     import random
    #     random.seed(42)
    #     np.random.seed(42)
    # Note: Reproducibility useful for debugging, but assignment likely expects
    # results averaged over multiple random trials, so leaving unseeded is fine.

    main()  # Execute all experiments
