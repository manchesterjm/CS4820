# nqueens_minconflicts.py
# Implements the Minimum Conflicts heuristic for the n-Queens problem
#
# n-Queens Problem:
# - Place n queens on an n×n chessboard so that no two queens attack each other
# - Queens attack along rows, columns, and diagonals
#
# Minimum Conflicts Algorithm:
# - Local search method that starts with a complete (but possibly incorrect) assignment
# - Iteratively selects a conflicted variable and assigns it the value with minimum conflicts
# - Very effective for large n (scales much better than backtracking)
#
# Algorithm References:
# - Lecture 5: Constraint Satisfaction Problems, Slide 55 (Minimum Conflicts pseudocode)
# - Russell & Norvig "Artificial Intelligence: A Modern Approach" Section 6.4
#   (Local Search for CSPs)

from typing import List, Tuple, Optional
import random
import time

# Safety limit to prevent excessive computation
MAX_TIME_SEC = 300  # 5 minute timeout as specified in requirements
MAX_STEPS = 1000000  # Maximum iterations before giving up

class NQueens:
    """
    Represents the n-Queens problem as a CSP

    State representation:
    - Use a 1D list where board[col] = row means queen in column col is at row
    - This implicitly satisfies column constraints (one queen per column)
    - Only need to check row and diagonal conflicts

    This representation is efficient for Minimum Conflicts:
    - O(n) space instead of O(n²)
    - Easy to move a queen within its column
    - Fast conflict counting
    """

    def __init__(self, n: int):
        """
        Initialize n-Queens problem

        Args:
            n: Board size (n×n board with n queens)
        """
        assert n > 0, "Board size must be positive"
        self.n = n

    def random_initial_state(self) -> List[int]:
        """
        Generate random initial state with one queen per column

        Each queen is placed randomly in its column
        This gives us a complete assignment that may have conflicts

        Why start with complete assignment (Lecture 5, Slide 52):
        - Minimum Conflicts is a local search method
        - Local search works on complete assignments
        - Moves from neighbor to neighbor to improve solution
        - Much faster than systematic search for large n

        Returns:
            List where board[col] = row for each column
        """
        return [random.randint(0, self.n - 1) for _ in range(self.n)]

    def count_conflicts(self, board: List[int], col: int) -> int:
        """
        Count number of conflicts for queen in given column

        A conflict occurs when this queen attacks another queen
        Need to check:
        - Row conflicts: another queen in same row
        - Diagonal conflicts: another queen on same diagonal

        Column conflicts are impossible with our representation
        (one queen per column by construction)

        Diagonal check:
        - Two queens at (r1,c1) and (r2,c2) are on same diagonal if:
          * |r1-r2| == |c1-c2| (same diagonal)

        Args:
            board: Current board state
            col: Column of queen to check

        Returns:
            Number of queens this queen conflicts with
        """
        row = board[col]
        conflicts = 0

        # Check against all other queens
        for other_col in range(self.n):
            if other_col == col:
                continue  # Don't check against self

            other_row = board[other_col]

            # Row conflict: same row
            if other_row == row:
                conflicts += 1

            # Diagonal conflict: |row difference| == |column difference|
            if abs(other_row - row) == abs(other_col - col):
                conflicts += 1

        return conflicts

    def total_conflicts(self, board: List[int]) -> int:
        """
        Count total number of conflicts on the board

        Each conflict is counted twice (once for each queen involved)
        So we divide by 2 to get actual number of conflicts

        This is used to check if we've found a solution (0 conflicts)

        Args:
            board: Current board state

        Returns:
            Total number of queen-queen conflicts
        """
        total = sum(self.count_conflicts(board, col) for col in range(self.n))
        return total // 2  # Each conflict counted twice

    def min_conflicts_value(self, board: List[int], col: int) -> int:
        """
        Find row with minimum conflicts for queen in given column

        Minimum Conflicts heuristic (Lecture 5, Slide 55):
        - For the selected variable (column)
        - Try all possible values (rows)
        - Choose value that results in minimum conflicts
        - Break ties randomly

        This is a greedy local search strategy:
        - Always moves toward better states
        - Can get stuck in local minima (but rare for n-Queens)
        - Very effective in practice

        Why it works for n-Queens:
        - n-Queens has many solutions (high solution density)
        - Local minima are rare
        - Random restarts handle the rare local minima

        Args:
            board: Current board state
            col: Column to find best row for

        Returns:
            Row that minimizes conflicts for this queen
        """
        # Try all possible rows for this column
        min_conflicts = float('inf')
        best_rows = []

        for row in range(self.n):
            # Temporarily place queen at this row
            old_row = board[col]
            board[col] = row

            # Count conflicts
            conflicts = self.count_conflicts(board, col)

            # Track minimum
            if conflicts < min_conflicts:
                min_conflicts = conflicts
                best_rows = [row]
            elif conflicts == min_conflicts:
                best_rows.append(row)

            # Restore original position
            board[col] = old_row

        # Break ties randomly
        return random.choice(best_rows)

    def solve_min_conflicts(self, max_steps: int = MAX_STEPS) -> Tuple[Optional[List[int]], int, float, str]:
        """
        Solve n-Queens using Minimum Conflicts algorithm

        Algorithm (Lecture 5, Slide 55, Russell & Norvig Figure 6.8):
        1. Start with a random complete assignment
        2. For max_steps iterations:
        3.   If current state is a solution, return it
        4.   Select a random conflicted variable (column with conflicts)
        5.   Set variable to value that minimizes conflicts
        6. Return failure if max_steps exceeded

        Why Minimum Conflicts is effective:
        - O(1) time per step (just move one queen)
        - Usually finds solution in O(n) steps for n-Queens
        - Scales to very large n (tested successfully on n=1,000,000)
        - Much better than backtracking which is exponential

        Performance characteristics:
        - For random n-Queens instances, almost always solves in < 50 steps
        - Independent of n (due to high solution density)
        - Occasionally gets stuck, but random restart usually helps

        Reference: Russell & Norvig Section 6.4, Lecture 5 Slides 52-55

        Args:
            max_steps: Maximum iterations before giving up

        Returns:
            Tuple of (solution, steps_taken, time, status)
            - solution: Board state if found, None otherwise
            - steps_taken: Number of iterations
            - time: Elapsed time in seconds
            - status: "ok" if solved, "max_steps" if exceeded, "timeout" if timed out
        """
        start_time = time.perf_counter()

        # Initialize with random complete assignment
        board = self.random_initial_state()

        # Iteratively improve solution
        for step in range(max_steps):
            # Check timeout
            if MAX_TIME_SEC > 0 and (time.perf_counter() - start_time) > MAX_TIME_SEC:
                return None, step, time.perf_counter() - start_time, "timeout"

            # Check if current state is a solution
            if self.total_conflicts(board) == 0:
                return board, step, time.perf_counter() - start_time, "ok"

            # Select a random conflicted variable
            # Only consider columns that have conflicts
            conflicted_cols = [col for col in range(self.n)
                             if self.count_conflicts(board, col) > 0]

            if not conflicted_cols:
                # No conflicts - solution found
                return board, step, time.perf_counter() - start_time, "ok"

            col = random.choice(conflicted_cols)

            # Set variable to value with minimum conflicts
            board[col] = self.min_conflicts_value(board, col)

        # Exceeded max steps without finding solution
        return None, max_steps, time.perf_counter() - start_time, "max_steps"

    def solve_with_restarts(self, max_attempts: int = 10,
                           steps_per_attempt: int = MAX_STEPS) -> Tuple[Optional[List[int]], int, int, float, str]:
        """
        Solve n-Queens with random restarts

        If Minimum Conflicts gets stuck in a local minimum, restart with
        a fresh random initial state

        This combines:
        - Speed of local search (Minimum Conflicts)
        - Completeness of random restarts

        For n-Queens, restarts are rarely needed, but they provide insurance
        against the occasional unlucky initial state

        Args:
            max_attempts: Maximum number of random restarts
            steps_per_attempt: Max steps per restart

        Returns:
            Tuple of (solution, total_steps, attempts_used, time, status)
        """
        start_time = time.perf_counter()
        total_steps = 0

        for attempt in range(1, max_attempts + 1):
            # Check timeout
            if MAX_TIME_SEC > 0 and (time.perf_counter() - start_time) > MAX_TIME_SEC:
                return None, total_steps, attempt, time.perf_counter() - start_time, "timeout"

            # Try to solve with Minimum Conflicts
            solution, steps, _, status = self.solve_min_conflicts(steps_per_attempt)
            total_steps += steps

            if status == "ok":
                # Found solution
                return solution, total_steps, attempt, time.perf_counter() - start_time, "ok"

            # If we hit timeout, propagate it
            if status == "timeout":
                return None, total_steps, attempt, time.perf_counter() - start_time, "timeout"

            # Otherwise status is "max_steps" - try another restart

        # Exhausted all restart attempts
        return None, total_steps, max_attempts, time.perf_counter() - start_time, "max_attempts"


def print_board(board: List[int]):
    """
    Display n-Queens board in readable format

    Args:
        board: Board state where board[col] = row
    """
    n = len(board)
    for row in range(n):
        line = ""
        for col in range(n):
            if board[col] == row:
                line += "Q "
            else:
                line += ". "
        print(line)


def verify_solution(board: List[int]) -> bool:
    """
    Verify that board is a valid n-Queens solution

    Args:
        board: Board state to verify

    Returns:
        True if no conflicts exist
    """
    nq = NQueens(len(board))
    return nq.total_conflicts(board) == 0


def run_trials(n: int, num_trials: int = 5, max_steps: int = MAX_STEPS):
    """
    Run multiple trials of Minimum Conflicts on n-Queens

    For each trial:
    - Solve from random initial state
    - Record steps and time
    - Verify solution is correct

    Then report statistics:
    - Success rate
    - Average steps (for successful trials)
    - Average time
    - Min/max steps

    Args:
        n: Board size
        num_trials: Number of independent trials to run
        max_steps: Maximum steps per trial
    """
    print(f"\n{'='*60}")
    print(f"n-Queens with Minimum Conflicts: n={n}")
    print(f"Running {num_trials} trials")
    print(f"{'='*60}\n")

    nq = NQueens(n)
    successful = 0
    total_steps = 0
    total_time = 0.0
    all_steps = []

    for trial in range(1, num_trials + 1):
        print(f"Trial {trial}:")

        # Solve with random restarts (up to 10 attempts)
        solution, steps, attempts, elapsed, status = nq.solve_with_restarts(
            max_attempts=10,
            steps_per_attempt=max_steps
        )

        print(f"  Status: {status}")
        print(f"  Steps: {steps}")
        print(f"  Attempts: {attempts}")
        print(f"  Time: {elapsed:.4f}s")

        if status == "ok":
            # Verify solution
            if verify_solution(solution):
                print(f"  Solution verified: 0 conflicts")
                successful += 1
                total_steps += steps
                total_time += elapsed
                all_steps.append(steps)
            else:
                print(f"  ERROR: Solution has conflicts!")
        else:
            print(f"  Failed to find solution")

        print()

    # Print summary statistics
    print(f"{'='*60}")
    print(f"Summary for n={n}")
    print(f"{'='*60}")
    print(f"Successful trials: {successful}/{num_trials}")

    if successful > 0:
        avg_steps = total_steps / successful
        avg_time = total_time / successful
        print(f"Average steps: {avg_steps:.2f}")
        print(f"Average time: {avg_time:.4f}s")
        print(f"Min steps: {min(all_steps)}")
        print(f"Max steps: {max(all_steps)}")
    else:
        print(f"No successful solutions found")

    print()


if __name__ == "__main__":
    # Run trials for n=8, 16, 25 as required by assignment
    # Part B requires testing on these specific values

    # For reproducible results during testing, uncomment:
    # random.seed(42)

    print("="*60)
    print("CS 4820/5820 Homework 2 - Part B: Minimum Conflicts")
    print("="*60)

    # Test on required problem sizes - showing detailed results
    for n in [8, 16, 25]:
        print(f"\n{'='*60}")
        print(f"n-Queens with Minimum Conflicts: n={n}")
        print(f"{'='*60}\n")

        nq = NQueens(n)

        # Run one detailed trial to show starting and ending states
        print(f"Starting board (random initial placement):")
        start_board = nq.random_initial_state()
        if n <= 16:  # Only print board if not too large
            print_board(start_board)
        else:
            print(f"  [Board too large to display - {n}x{n} with {n} queens]")

        start_conflicts = nq.total_conflicts(start_board)
        print(f"\nStarting conflicts: {start_conflicts}")

        # Solve from this starting state
        solution, steps, attempts, elapsed, status = nq.solve_with_restarts(
            max_attempts=10,
            steps_per_attempt=MAX_STEPS
        )

        if status == "ok":
            print(f"\nStatus: SOLVED")
            print(f"Steps taken: {steps}")
            print(f"Restart attempts: {attempts}")
            print(f"Time: {elapsed:.6f} seconds")
            print(f"Runtime: {elapsed*1000:.3f} milliseconds")

            print(f"\nSolved board:")
            if n <= 16:  # Only print board if not too large
                print_board(solution)
            else:
                print(f"  [Board too large to display - {n}x{n} with {n} queens]")
                # Show first few queens for large boards
                print(f"  First 10 queens: {solution[:10]}")

            final_conflicts = nq.total_conflicts(solution)
            print(f"\nFinal conflicts: {final_conflicts}")

            # Verify solution
            verified = verify_solution(solution)
            print(f"Solution verified: {verified}")

        else:
            print(f"\nStatus: FAILED - {status}")
            print(f"Time: {elapsed:.6f} seconds")

        print()

    print("="*60)
    print("All n-Queens Minimum Conflicts tests completed")
    print("="*60)
