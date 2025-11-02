"""
Particle Swarm Optimization (PSO) for solving Sudoku puzzles.

This module applies PSO to Sudoku as an optimization problem rather than
a CSP with hard constraints. The approach treats Sudoku as a minimization
problem where the objective is to minimize the number of constraint
violations.

Sudoku as an Optimization Problem:
- Each "particle" represents a complete Sudoku board
- Global minimum (0 violations) equals a valid solution
- Fixed cells (given in puzzle) never change
- Only empty cells are decision variables
- Particle position equals values for empty cells
- Domain: {1, 2, 3, 4, 5, 6, 7, 8, 9} for each cell

Fitness Function:
- Count total violations of row, column, and box constraints
- Lower fitness equals better solution
- Fitness equals 0 means puzzle is solved

Algorithm References:
- Lecture 7: Search Optimization Part III (PSO)
- Moraglio & Togelius, "Geometric Particle Swarm Optimization for
  Combinatorial Problems"
- PSO adapted for discrete/permutation problems
"""

from typing import List, Tuple
import time
import copy
import random
import numpy as np

# Safety limit to prevent excessive computation
MAX_TIME_SEC = 300  # 5 minute timeout

class SudokuPSO:
    """
    Particle Swarm Optimization for Sudoku

    Challenges of applying PSO to Sudoku:
    1. Discrete domain (not continuous like standard PSO)
    2. Permutation structure (values 1-9 must appear in each
       row/col/box)
    3. Hard constraints (given cells can't change)

    Solutions:
    1. Use discrete velocity updates (swap operations)
    2. Initialize particles as permutations to satisfy row constraints
    3. Lock given cells in place
    4. Focus on reducing column and box violations
    """

    def __init__(self,
                 puzzle: List[List[int]],
                 swarm_size: int = 100,
                 max_iterations: int = 5000,
                 w: float = 0.7,
                 c1: float = 1.5,
                 c2: float = 1.5):
        """
        Initialize PSO for Sudoku

        Args:
            puzzle: 9x9 Sudoku grid (0 for empty cells)
            swarm_size: Number of particles
            max_iterations: Maximum iterations
            w: Inertia weight
            c1: Cognitive coefficient
            c2: Social coefficient
        """
        self.puzzle = [row[:] for row in puzzle]
        self.swarm_size = swarm_size
        self.max_iterations = max_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2

        # Identify fixed cells (given in puzzle)
        self.fixed_cells = set()
        for r in range(9):
            for c in range(9):
                if puzzle[r][c] != 0:
                    self.fixed_cells.add((r, c))

        # Initialize swarm
        self.particles = []
        self.velocities = []  # For discrete PSO: list of swap operations
        self.personal_bests = []
        self.personal_best_scores = []
        self.global_best = None
        self.global_best_score = float('inf')

        # Track convergence
        self.convergence_history = []

    def count_violations(self, board: List[List[int]]) -> int:
        """
        Count total constraint violations in Sudoku board

        Violations occur when same number appears multiple times in:
        - Same row (but we'll ensure rows are permutations, so 0 violations)
        - Same column
        - Same 3x3 box

        Args:
            board: 9x9 Sudoku grid

        Returns:
            Total number of violations
        """
        violations = 0

        # Count column violations
        for c in range(9):
            column = [board[r][c] for r in range(9)]
            # Violations = duplicates = 9 - unique count
            violations += 9 - len(set(column))

        # Count 3x3 box violations
        for box_r in range(0, 9, 3):
            for box_c in range(0, 9, 3):
                box = []
                for r in range(box_r, box_r + 3):
                    for c in range(box_c, box_c + 3):
                        box.append(board[r][c])
                # Violations = duplicates = 9 - unique count
                violations += 9 - len(set(box))

        return violations

    def create_initial_particle(self) -> List[List[int]]:
        """
        Create initial particle with valid row permutations

        Strategy:
        - For each row, ensure values 1-9 appear exactly once
        - Keep given cells fixed
        - Fill empty cells with remaining values randomly

        This ensures 0 row violations, focusing optimization on columns
        and boxes

        Returns:
            9x9 Sudoku board as initial particle
        """
        board = [[0] * 9 for _ in range(9)]

        for r in range(9):
            # Get values already fixed in this row
            fixed_values = []
            fixed_positions = []
            for c in range(9):
                if (r, c) in self.fixed_cells:
                    fixed_values.append(self.puzzle[r][c])
                    fixed_positions.append(c)
                    board[r][c] = self.puzzle[r][c]

            # Get remaining values to fill
            remaining_values = [v for v in range(1, 10)
                                if v not in fixed_values]
            random.shuffle(remaining_values)

            # Fill empty cells
            value_idx = 0
            for c in range(9):
                if (r, c) not in self.fixed_cells:
                    board[r][c] = remaining_values[value_idx]
                    value_idx += 1

        return board

    def initialize_swarm(self):
        """
        Initialize swarm with random particles

        Each particle is a complete Sudoku board with:
        - Valid row permutations (0 row violations)
        - Random assignments for columns and boxes
        """
        for _ in range(self.swarm_size):
            # Create random particle
            particle = self.create_initial_particle()
            self.particles.append(particle)

            # Initialize velocity as empty (discrete PSO)
            self.velocities.append([])

            # Set personal best to initial position
            score = self.count_violations(particle)
            self.personal_bests.append(copy.deepcopy(particle))
            self.personal_best_scores.append(score)

            # Update global best
            if score < self.global_best_score:
                self.global_best_score = score
                self.global_best = copy.deepcopy(particle)

    def apply_swap(self, board: List[List[int]], row: int, col1: int,
                   col2: int):
        """
        Apply swap operation to board (in place)

        Only swap if both cells are not fixed

        Args:
            board: Board to modify
            row: Row to swap within
            col1, col2: Columns to swap
        """
        if ((row, col1) not in self.fixed_cells and
                (row, col2) not in self.fixed_cells):
            temp = board[row][col1]
            board[row][col1] = board[row][col2]
            board[row][col2] = temp

    def generate_random_swaps(self, num_swaps: int
                              ) -> List[Tuple[int, int, int]]:
        """
        Generate random swap operations

        Each swap is (row, col1, col2) representing swapping two cells
        in a row

        Args:
            num_swaps: Number of swaps to generate

        Returns:
            List of swap operations
        """
        swaps = []
        for _ in range(num_swaps):
            row = random.randint(0, 8)
            col1, col2 = random.sample(range(9), 2)
            swaps.append((row, col1, col2))
        return swaps

    def update_particle(self, particle_idx: int):
        """
        Update particle position using discrete PSO

        Discrete PSO adaptation:
        - Instead of continuous velocity, use sequence of swap
          operations
        - Swaps move particle toward personal best and global best
        - Probabilistically apply swaps based on w, c1, c2

        Algorithm:
        1. With probability w: keep some random swaps (inertia)
        2. With probability c1: apply swaps that move toward personal
           best
        3. With probability c2: apply swaps that move toward global best

        Args:
            particle_idx: Index of particle to update
        """
        # Get references to current particle, its personal best, and swarm's global best
        # These represent the three knowledge sources in PSO
        particle = self.particles[particle_idx]  # Current particle position (Sudoku board)
        personal_best = self.personal_bests[particle_idx]  # Best position this particle has found
        global_best = self.global_best  # Best position any particle has found

        # Create deep copy to modify (don't modify original until we commit changes)
        # This prevents corrupting the particle state if updates fail
        new_particle = copy.deepcopy(particle)

        # Inertia component: random exploration swaps
        # With probability w, apply random swaps to explore new regions
        # Higher w = more exploration (more random changes)
        # Lower w = more exploitation (stick closer to current position)
        if random.random() < self.w:
            # Number of swaps scales with inertia weight
            # More inertia = more random changes = more exploration
            num_swaps = max(1, int(5 * self.w))  # At least 1 swap, up to ~5 swaps
            swaps = self.generate_random_swaps(num_swaps)  # Generate random swap operations
            for row, col1, col2 in swaps:
                # Apply each swap to the new particle
                # This moves particle in random direction (exploration)
                self.apply_swap(new_particle, row, col1, col2)

        # Cognitive component: move toward personal best
        # With probability c1, apply swaps that make particle look more like its personal best
        # This is the "individual learning" component - particle remembers what worked for it
        if random.random() < self.c1:
            # Find differences between current and personal best
            # For each cell that doesn't match, attempt swap to fix it
            # This gradually moves particle toward its best known configuration
            for r in range(9):  # For each row
                for c in range(9):  # For each column
                    if (r, c) not in self.fixed_cells:  # Only modify non-fixed cells
                        target_value = personal_best[r][c]  # Value we want at this cell
                        current_value = new_particle[r][c]  # Value currently at this cell

                        if target_value != current_value:  # If they don't match
                            # Find where target_value currently is in this row
                            # We need to swap it to position (r, c)
                            for c2 in range(9):
                                if (new_particle[r][c2] == target_value and
                                        (r, c2) not in self.fixed_cells):
                                    # Found the target value at position (r, c2)
                                    # Swap probabilistically to add stochasticity
                                    # Don't always swap or we might oscillate
                                    if random.random() < 0.5:
                                        self.apply_swap(new_particle, r, c, c2)
                                    break  # Found and (maybe) swapped, move to next cell

        # Social component: move toward global best
        # With probability c2, apply swaps that make particle look more like global best
        # This is the "social learning" component - particle learns from swarm's success
        # Higher c2 = faster convergence to swarm's best solution
        if random.random() < self.c2:
            # Find differences between current and global best
            # For each cell that doesn't match, attempt swap to fix it
            # This pulls all particles toward the best solution found by anyone
            for r in range(9):  # For each row
                for c in range(9):  # For each column
                    if (r, c) not in self.fixed_cells:
                        target_value = global_best[r][c]
                        current_value = new_particle[r][c]

                        if target_value != current_value:
                            # Find where target_value is in this row
                            for c2 in range(9):
                                if (new_particle[r][c2] == target_value and
                                        (r, c2) not in self.fixed_cells):
                                    # Swap to match global best
                                    # Probabilistic
                                    if random.random() < 0.5:
                                        self.apply_swap(new_particle, r, c,
                                                        c2)
                                    break

        # Update particle
        self.particles[particle_idx] = new_particle

        # Evaluate new fitness
        score = self.count_violations(new_particle)

        # Update personal best
        if score < self.personal_best_scores[particle_idx]:
            self.personal_best_scores[particle_idx] = score
            self.personal_bests[particle_idx] = copy.deepcopy(new_particle)

            # Update global best
            if score < self.global_best_score:
                self.global_best_score = score
                self.global_best = copy.deepcopy(new_particle)

    def optimize(self) -> Tuple[List[List[int]], int, int, float, str]:
        """
        Run PSO optimization on Sudoku

        Returns:
            Tuple of (best_board, best_score, iterations, time, status)
        """
        start_time = time.perf_counter()

        # Initialize swarm
        self.initialize_swarm()
        self.convergence_history = [self.global_best_score]

        # Main optimization loop
        for iteration in range(self.max_iterations):
            # Check timeout
            elapsed = time.perf_counter() - start_time
            if 0 < MAX_TIME_SEC < elapsed:
                return (self.global_best,
                       self.global_best_score,
                       iteration,
                       time.perf_counter() - start_time,
                       "timeout")

            # Update all particles
            for i in range(self.swarm_size):
                self.update_particle(i)

            # Track convergence
            self.convergence_history.append(self.global_best_score)

            # Check if solved
            if self.global_best_score == 0:
                return (self.global_best,
                       0,
                       iteration + 1,
                       time.perf_counter() - start_time,
                       "solved")

            # Progress reporting every 100 iterations
            if (iteration + 1) % 100 == 0:
                score = self.global_best_score
                print(f"  Iteration {iteration + 1}: best score = {score}")

        # Reached max iterations
        return (self.global_best,
               self.global_best_score,
               self.max_iterations,
               time.perf_counter() - start_time,
               "max_iterations")


def print_sudoku(board: List[List[int]]):
    """Display Sudoku board in readable format"""
    for i, row in enumerate(board):
        if i > 0 and i % 3 == 0:
            print("-" * 21)
        row_str = ""
        for j, val in enumerate(row):
            if j > 0 and j % 3 == 0:
                row_str += "| "
            row_str += str(val) + " "
        print(row_str)


def run_sudoku_pso_trial(puzzle: List[List[int]], trial_num: int,
                         swarm_size: int = 100, max_iterations: int = 5000):
    """
    Run single PSO trial on Sudoku puzzle

    Args:
        puzzle: 9x9 Sudoku grid
        trial_num: Trial number for display
        swarm_size: Number of particles
        max_iterations: Max iterations

    Returns:
        Tuple of (best_board, score, iterations, time, status)
    """
    print(f"\nTrial {trial_num}:")
    print(f"  Swarm size: {swarm_size}, Max iterations: {max_iterations}")

    pso = SudokuPSO(
        puzzle=puzzle,
        swarm_size=swarm_size,
        max_iterations=max_iterations,
        w=0.7,
        c1=1.5,
        c2=1.5
    )

    best_board, score, iters, elapsed, status = pso.optimize()

    print(f"  Final score: {score} violations")
    print(f"  Iterations: {iters}")
    print(f"  Time: {elapsed:.4f}s")
    print(f"  Status: {status}")

    if status == "solved":
        print("  ✓ PUZZLE SOLVED!")

    return best_board, score, iters, elapsed, status


if __name__ == "__main__":
    # Set random seed for reproducibility during testing
    # Uncomment for deterministic results:
    # random.seed(42)

    print("="*70)
    print("CS 4820/5820 Homework 2 - Part C2: PSO for Sudoku")
    print("="*70)

    # Test puzzle (same as used in sudoku_csp.py)
    test_puzzle = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ]

    print("\nStarting Puzzle:")
    print_sudoku(test_puzzle)

    # Count given cells
    given_cells = sum(1 for row in test_puzzle for cell in row if cell != 0)
    print(f"\nGiven cells: {given_cells}")
    print(f"Empty cells: {81 - given_cells}")

    print("\nPSO Configuration:")
    print("  Swarm size: 150 particles")
    print("  Max iterations: 3000")
    print("  Inertia weight (w): 0.7")
    print("  Cognitive coeff (c1): 1.5")
    print("  Social coeff (c2): 1.5")
    print()

    # Run 3 trials as required by assignment
    num_trials = 3
    results = []

    for trial in range(1, num_trials + 1):
        result = run_sudoku_pso_trial(
            puzzle=test_puzzle,
            trial_num=trial,
            swarm_size=150,  # Larger swarm for harder optimization
            max_iterations=3000
        )
        results.append(result)

    # Summary statistics
    print("\n" + "="*70)
    print("Summary of PSO for Sudoku")
    print("="*70)

    scores = [r[1] for r in results]
    times = [r[3] for r in results]
    solved = sum(1 for r in results if r[4] == "solved")

    print(f"\nTrials run: {num_trials}")
    print(f"Solved: {solved}/{num_trials}")
    print(f"Best score: {min(scores)} violations")
    print(f"Worst score: {max(scores)} violations")
    print(f"Avg score: {np.mean(scores):.2f} violations")
    print(f"Avg time: {np.mean(times):.4f}s")
    print(f"Total runtime: {sum(times):.4f}s")

    # Show best result
    best_idx = np.argmin(scores)
    best_board, best_score, best_iters, best_time, best_status = results[best_idx]

    print(f"\n" + "="*70)
    print(f"Best Result (Trial {best_idx + 1})")
    print("="*70)
    print(f"Status: {best_status}")
    print(f"Final violations: {best_score}")
    print(f"Iterations: {best_iters}")
    print(f"Time: {best_time:.4f}s")

    if best_score == 0:
        print("\nSOLVED BOARD:")
        print_sudoku(best_board)
    else:
        print(f"\nBest board found ({best_score} violations remaining):")
        print_sudoku(best_board)

        # Show where violations are
        col_violations = 0
        box_violations = 0

        # Count column violations
        for c in range(9):
            column = [best_board[r][c] for r in range(9)]
            col_violations += 9 - len(set(column))

        # Count box violations
        for box_r in range(0, 9, 3):
            for box_c in range(0, 9, 3):
                box = [best_board[r][c]
                       for r in range(box_r, box_r+3)
                       for c in range(box_c, box_c+3)]
                box_violations += 9 - len(set(box))

        print("\nViolation breakdown:")
        print(f"  Column violations: {col_violations}")
        print(f"  Box violations: {box_violations}")
        print(f"  Total: {best_score}")

    print("\n" + "="*70)
    print("Analysis")
    print("="*70)
    print("PSO demonstrates metaheuristic approach to constraint "
          "optimization.")
    print("Unlike CSP methods (Part A) which guarantee solutions, PSO "
          "provides")
    print("approximate optimization. CSP methods solve this puzzle in "
          "<0.02s")
    avg_time = np.mean(times)
    print(f"with certainty, while PSO takes ~{avg_time:.1f}s and may "
          f"not fully solve.")
    print("\nPSO is better suited for continuous optimization and "
          "problems where")
    print("approximate solutions are acceptable.")
    print("="*70)
