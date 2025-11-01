# run_experiments.py
# Main experiment runner for CS 4820/5820 Homework 2
#
# Runs all required experiments and generates results for report:
# - Part A: Sudoku CSP with different algorithms
# - Part B: n-Queens with Minimum Conflicts (n=8, 16, 25)
# - Part C1: PSO on benchmark functions (multiple parameter configs)
# - Part C2: PSO on Sudoku (3 trials)
#
# Outputs tables and statistics suitable for inclusion in AAAI format report

import time
import numpy as np
from typing import List, Dict, Tuple

from sudoku_csp import SudokuCSP, assignment_to_grid, print_sudoku
from nqueens_minconflicts import NQueens, verify_solution, print_board
from pso_benchmark import PSO, rastrigin, rosenbrock
from pso_sudoku import SudokuPSO
from sudoku_puzzles import PUZZLES, count_given_cells


def run_part_a_experiments():
    """
    Part A: Compare Sudoku CSP solver variants

    Run each algorithm on easy, medium, and hard puzzles
    Report: runtime, nodes expanded (if tracked), backtracks
    """
    print("\n" + "="*80)
    print("PART A: Sudoku as a CSP")
    print("="*80)

    methods = [
        ("Basic Backtracking", "solve_basic"),
        ("Backtracking + MRV + LCV", "solve_mrv_lcv"),
        ("Backtracking + Forward Checking", "solve_forward_checking"),
        ("Backtracking + AC-3", "solve_ac3")
    ]

    difficulties = ["easy", "medium", "hard"]

    # Header
    print("\n{:<30} {:<10} {:<15} {:>12}".format(
        "Algorithm", "Difficulty", "Given Cells", "Time (s)"
    ))
    print("-" * 80)

    for diff in difficulties:
        # Use first puzzle from each difficulty
        puzzle = PUZZLES[diff][0]
        given = count_given_cells(puzzle)

        for name, method in methods:
            csp = SudokuCSP(puzzle)
            solution, backtracks, elapsed = getattr(csp, method)()

            status = "PASS" if solution else "FAIL"
            print("{:<30} {:<10} {:>15} {:>12.6f} {}".format(
                name, diff, given, elapsed, status
            ))

        print()

    print("\nInterpretation:")
    print("- Basic Backtracking: Naive DFS with no heuristics")
    print("- +MRV+LCV: Variable and value ordering heuristics")
    print("- +Forward Checking: Inference after each assignment")
    print("- +AC-3: Full constraint propagation")
    print("\nExpected: AC-3 should be fastest due to aggressive pruning")


def run_part_b_experiments():
    """
    Part B: n-Queens with Minimum Conflicts

    Run 5 trials each for n=8, 16, 25
    Report: average steps, average runtime, success rate
    """
    print("\n" + "="*80)
    print("PART B: n-Queens with Minimum Conflicts")
    print("="*80)

    test_sizes = [8, 16, 25]
    trials_per_size = 5

    # Header
    print("\n{:<10} {:<10} {:<15} {:<15} {:<15}".format(
        "n", "Trials", "Success Rate", "Avg Steps", "Avg Time (s)"
    ))
    print("-" * 80)

    for n in test_sizes:
        nq = NQueens(n)
        successful = 0
        total_steps = 0
        total_time = 0.0

        for trial in range(trials_per_size):
            solution, steps, attempts, elapsed, status = nq.solve_with_restarts(
                max_attempts=10,
                steps_per_attempt=100000
            )

            if status == "ok" and verify_solution(solution):
                successful += 1
                total_steps += steps
                total_time += elapsed

        success_rate = f"{successful}/{trials_per_size}"
        avg_steps = total_steps / max(1, successful)
        avg_time = total_time / max(1, successful)

        print("{:<10} {:<10} {:<15} {:<15.1f} {:<15.6f}".format(
            n, trials_per_size, success_rate, avg_steps, avg_time
        ))

    print("\nInterpretation:")
    print("- Minimum Conflicts is a local search heuristic")
    print("- Typically solves in O(n) steps regardless of board size")
    print("- Much faster than backtracking for large n")
    print("- Success rate should be very high (close to 100%)")


def run_part_c1_experiments():
    """
    Part C1: PSO on benchmark functions

    Test multiple parameter configurations
    Report: best fitness, average fitness, convergence speed
    """
    print("\n" + "="*80)
    print("PART C1: PSO for Benchmark Optimization")
    print("="*80)

    # Parameter configurations to test
    configs = [
        {"name": "Config 1 (Standard)", "swarm_size": 30, "w": 0.7, "c1": 1.5, "c2": 1.5, "max_iterations": 1000},
        {"name": "Config 2 (Large Swarm)", "swarm_size": 50, "w": 0.5, "c1": 2.0, "c2": 2.0, "max_iterations": 1000},
        {"name": "Config 3 (High Inertia)", "swarm_size": 40, "w": 0.9, "c1": 1.2, "c2": 1.2, "max_iterations": 1500},
    ]

    trials = 3

    # Test Rastrigin
    print("\n" + "-"*80)
    print("Benchmark: Rastrigin Function (10D)")
    print("Global minimum: f(0,...,0) = 0")
    print("-"*80)

    print("\n{:<25} {:<15} {:<15} {:<15}".format(
        "Configuration", "Best Score", "Avg Score", "Avg Time (s)"
    ))
    print("-" * 80)

    for config in configs:
        name = config.pop("name")
        scores = []
        times = []

        for _ in range(trials):
            pso = PSO(
                objective_func=rastrigin,
                dimensions=10,
                bounds=(-5.12, 5.12),
                **config
            )
            best_pos, best_score, iters, elapsed, status = pso.optimize()
            scores.append(best_score)
            times.append(elapsed)

        config["name"] = name  # Restore name

        print("{:<25} {:<15.6f} {:<15.6f} {:<15.6f}".format(
            name,
            min(scores),
            np.mean(scores),
            np.mean(times)
        ))

    # Test Rosenbrock
    print("\n" + "-"*80)
    print("Benchmark: Rosenbrock Function (10D)")
    print("Global minimum: f(1,...,1) = 0")
    print("-"*80)

    print("\n{:<25} {:<15} {:<15} {:<15}".format(
        "Configuration", "Best Score", "Avg Score", "Avg Time (s)"
    ))
    print("-" * 80)

    for config in configs:
        name = config.pop("name")
        scores = []
        times = []

        for _ in range(trials):
            pso = PSO(
                objective_func=rosenbrock,
                dimensions=10,
                bounds=(-5, 10),
                **config
            )
            best_pos, best_score, iters, elapsed, status = pso.optimize()
            scores.append(best_score)
            times.append(elapsed)

        config["name"] = name  # Restore name

        print("{:<25} {:<15.6f} {:<15.6f} {:<15.6f}".format(
            name,
            min(scores),
            np.mean(scores),
            np.mean(times)
        ))

    print("\nInterpretation:")
    print("- Rastrigin: Highly multimodal, tests ability to escape local minima")
    print("- Rosenbrock: Narrow valley, tests convergence precision")
    print("- Higher inertia (w) = more exploration, may escape local minima")
    print("- Higher c1/c2 = stronger attraction to bests, faster convergence")


def run_part_c2_experiments():
    """
    Part C2: PSO applied to Sudoku

    Run 3 trials on a test puzzle
    Report: final fitness (violations), runtime
    """
    print("\n" + "="*80)
    print("PART C2: PSO for Sudoku Optimization")
    print("="*80)

    puzzle = PUZZLES["easy"][0]
    trials = 3

    print("\nTest Puzzle:")
    print_sudoku(puzzle)
    print(f"Given cells: {count_given_cells(puzzle)}")

    print("\n{:<10} {:<20} {:<15} {:<15}".format(
        "Trial", "Final Violations", "Iterations", "Time (s)"
    ))
    print("-" * 80)

    scores = []
    times = []
    iterations = []

    for trial in range(1, trials + 1):
        pso = SudokuPSO(
            puzzle=puzzle,
            swarm_size=150,
            max_iterations=3000,
            w=0.7,
            c1=1.5,
            c2=1.5
        )

        best_board, score, iters, elapsed, status = pso.optimize()

        scores.append(score)
        times.append(elapsed)
        iterations.append(iters)

        status_str = "SOLVED" if status == "solved" else f"{status}"
        print("{:<10} {:<20} {:<15} {:<15.6f}  {}".format(
            trial, score, iters, elapsed, status_str
        ))

    print("-" * 80)
    print("{:<10} {:<20.2f} {:<15.1f} {:<15.6f}".format(
        "Average", np.mean(scores), np.mean(iterations), np.mean(times)
    ))

    print("\nInterpretation:")
    print("- PSO treats Sudoku as an optimization problem (minimize violations)")
    print("- May not always find perfect solution (0 violations)")
    print("- Demonstrates metaheuristic approach to constraint problems")
    print("- For guaranteed solutions, use CSP methods (Part A)")


def main():
    """Run all experiments"""
    print("="*80)
    print("CS 4820/5820 Homework 2 - Full Experimental Results")
    print("="*80)

    start_time = time.perf_counter()

    # Run all parts
    run_part_a_experiments()
    run_part_b_experiments()
    run_part_c1_experiments()
    run_part_c2_experiments()

    total_time = time.perf_counter() - start_time

    print("\n" + "="*80)
    print(f"All experiments completed in {total_time:.2f} seconds")
    print("="*80)

    print("\nResults ready for inclusion in AAAI format report")
    print("Copy relevant tables and statistics to your LaTeX writeup")


if __name__ == "__main__":
    # Uncomment for reproducible results:
    # import random
    # random.seed(42)
    # np.random.seed(42)

    main()
