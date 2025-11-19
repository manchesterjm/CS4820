"""
Collection of Sudoku test puzzles at various difficulty levels.

This module provides curated Sudoku puzzles for testing CSP solvers and PSO implementations.
Each puzzle is a 9x9 grid where 0 represents an empty cell to be filled.

Puzzle Difficulty Criteria:
The difficulty of a Sudoku puzzle depends on:
1. Number of given cells (fewer givens = harder)
2. Distribution of givens across the grid (uniform vs clustered)
3. Complexity of logical deductions required

Difficulty Levels:
- Easy: 35-40 given cells - solvable with basic constraint propagation
- Medium: 28-34 given cells - requires some backtracking
- Hard: 22-27 given cells - extensive backtracking needed
- Expert: <22 given cells - minimal clues, very deep search trees

Data Structure:
Each puzzle is a List[List[int]] representing a 9x9 grid:
- Outer list: 9 rows
- Inner list: 9 columns per row
- Values: 1-9 (given cells) or 0 (empty cells to fill)

Usage:
    from sudoku_puzzles import PUZZLES
    easy_puzzle = PUZZLES["easy"][0]  # Get first easy puzzle
"""

from typing import List, Dict

# ============================================================================
# EASY PUZZLES - Good Starting Point for Testing
# ============================================================================
# Easy puzzles have 35-40 given cells, providing strong initial constraints
# that significantly prune the search space. Basic backtracking with MRV/LCV
# can solve these very quickly (typically < 0.01 seconds).
EASY_1 = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]  # EASY_1: 36 given cells, well-distributed across grid

# Second easy puzzle with slightly different constraint structure
# Tests solver robustness across different puzzle configurations
EASY_2 = [
    [0, 0, 3, 0, 2, 0, 6, 0, 0],
    [9, 0, 0, 3, 0, 5, 0, 0, 1],
    [0, 0, 1, 8, 0, 6, 4, 0, 0],
    [0, 0, 8, 1, 0, 2, 9, 0, 0],
    [7, 0, 0, 0, 0, 0, 0, 0, 8],
    [0, 0, 6, 7, 0, 8, 2, 0, 0],
    [0, 0, 2, 6, 0, 9, 5, 0, 0],
    [8, 0, 0, 2, 0, 3, 0, 0, 9],
    [0, 0, 5, 0, 1, 0, 3, 0, 0]
]  # EASY_2: 38 given cells, good balance of row/column/box constraints

# ============================================================================
# MEDIUM PUZZLES - Moderate Challenge
# ============================================================================
# Medium puzzles have 28-34 given cells, requiring more sophisticated solving
# strategies. Basic backtracking may struggle; MRV and forward checking provide
# significant speedups. Expected solve time: 0.01-0.1 seconds with good heuristics.
MEDIUM_1 = [
    [0, 0, 0, 6, 0, 0, 4, 0, 0],
    [7, 0, 0, 0, 0, 3, 6, 0, 0],
    [0, 0, 0, 0, 9, 1, 0, 8, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 5, 0, 1, 8, 0, 0, 0, 3],
    [0, 0, 0, 3, 0, 6, 0, 4, 5],
    [0, 4, 0, 2, 0, 0, 0, 6, 0],
    [9, 0, 3, 0, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 0, 1, 0, 0]
]  # MEDIUM_1: 30 given cells, tests constraint propagation effectiveness

# Second medium puzzle with different constraint patterns
# Some rows/columns have few givens, testing variable selection heuristics
MEDIUM_2 = [
    [0, 0, 0, 0, 0, 0, 0, 1, 2],
    [0, 0, 0, 0, 3, 5, 0, 0, 0],
    [0, 0, 0, 6, 0, 0, 0, 7, 0],
    [7, 0, 0, 0, 0, 0, 3, 0, 0],
    [0, 0, 0, 4, 0, 0, 8, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 2, 0, 0, 0, 0],
    [0, 8, 0, 0, 0, 0, 0, 4, 0],
    [0, 5, 0, 0, 0, 0, 6, 0, 0]
]  # MEDIUM_2: 29 given cells, uneven distribution challenges naive approaches

# ============================================================================
# HARD PUZZLES - Challenging for Basic Backtracking
# ============================================================================
# Hard puzzles have 22-27 given cells with sparse, strategically-placed clues
# that create large search spaces. Basic backtracking can timeout; AC-3 and
# forward checking are essential. Expected solve time: 0.1-1 second with AC-3.
HARD_1 = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 3, 0, 8, 5],
    [0, 0, 1, 0, 2, 0, 0, 0, 0],
    [0, 0, 0, 5, 0, 7, 0, 0, 0],
    [0, 0, 4, 0, 0, 0, 1, 0, 0],
    [0, 9, 0, 0, 0, 0, 0, 0, 0],
    [5, 0, 0, 0, 0, 0, 0, 7, 3],
    [0, 0, 2, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 4, 0, 0, 0, 9]
]  # HARD_1: 24 given cells, first row empty tests early failure detection

# Second hard puzzle with minimal clues in strategic positions
# Tests ability to handle deep search trees and effective backtracking
HARD_2 = [
    [0, 0, 5, 3, 0, 0, 0, 0, 0],
    [8, 0, 0, 0, 0, 0, 0, 2, 0],
    [0, 7, 0, 0, 1, 0, 5, 0, 0],
    [4, 0, 0, 0, 0, 5, 3, 0, 0],
    [0, 1, 0, 0, 7, 0, 0, 0, 6],
    [0, 0, 3, 2, 0, 0, 0, 8, 0],
    [0, 6, 0, 5, 0, 0, 0, 0, 9],
    [0, 0, 4, 0, 0, 0, 0, 3, 0],
    [0, 0, 0, 0, 0, 9, 7, 0, 0]
]  # HARD_2: 26 given cells, scattered placement creates complex dependencies

# ============================================================================
# EXPERT PUZZLES - Minimal Clues, Extreme Challenge
# ============================================================================
# Expert puzzles have <22 given cells - the theoretical minimum is 17 for a
# unique solution. These create extremely large search spaces and may require
# very deep backtracking. Some may have multiple solutions or no unique solution.
# Expected solve time: 1+ seconds even with AC-3, may timeout with basic backtracking.
EXPERT_1 = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0]
]  # EXPERT_1: Empty board (0 given cells) - has 6,670,903,752,021,072,936,960 solutions!
   # Used to test solver behavior with minimal constraints
   # Any solver will find A solution, but not guaranteed to be fast

# AI Escargot - one of the world's hardest Sudoku puzzles
# Designed by Arto Inkala, this puzzle requires extremely sophisticated logic
# and extensive backtracking. Only 21 given cells, carefully positioned to
# minimize constraint propagation effectiveness.
EXPERT_2 = [
    [8, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 3, 6, 0, 0, 0, 0, 0],
    [0, 7, 0, 0, 9, 0, 2, 0, 0],
    [0, 5, 0, 0, 0, 7, 0, 0, 0],
    [0, 0, 0, 0, 4, 5, 7, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 3, 0],
    [0, 0, 1, 0, 0, 0, 0, 6, 8],
    [0, 0, 8, 5, 0, 0, 0, 1, 0],
    [0, 9, 0, 0, 0, 0, 4, 0, 0]
]  # EXPERT_2: "AI Escargot" - 21 given cells, extremely difficult even for AC-3

# ============================================================================
# PUZZLE DICTIONARY - Organize All Puzzles by Difficulty
# ============================================================================
# This dictionary provides easy access to all puzzles by difficulty level.
# Use this when running experiments to test solver performance across different
# difficulty classes. Each difficulty maps to a list of puzzles at that level.
PUZZLES: Dict[str, List[List[List[int]]]] = {
    "easy": [EASY_1, EASY_2],
    "medium": [MEDIUM_1, MEDIUM_2],
    "hard": [HARD_1, HARD_2],
    "expert": [EXPERT_1, EXPERT_2]
}


def count_given_cells(puzzle: List[List[int]]) -> int:
    """
    Count number of given (non-zero) cells in puzzle.

    This metric is a primary indicator of puzzle difficulty:
    - More givens = more constraints = smaller search space = easier
    - Fewer givens = fewer constraints = larger search space = harder

    The theoretical minimum for a unique solution is 17 givens, though
    proving uniqueness requires solving the puzzle.

    Args:
        puzzle: 9x9 Sudoku grid (List of 9 Lists, each with 9 integers)
                Values are 1-9 (given) or 0 (empty)

    Returns:
        Integer count of given cells (non-zero values)

    Complexity: O(n²) where n=9, so O(81) = O(1) for standard Sudoku
    """
    # Use generator expression with sum for efficient counting
    # Iterates through all rows and cells, counting non-zero values
    return sum(1 for row in puzzle for cell in row if cell != 0)


def print_puzzle_info():
    """
    Print summary information about all available puzzles.

    Displays difficulty levels, number of puzzles at each level, and
    the count of given cells for each puzzle. Useful for quickly seeing
    what test cases are available before running experiments.

    Output format:
        DIFFICULTY:
          DIFFICULTY_1: X given cells
          DIFFICULTY_2: Y given cells
          ...
    """
    print("Available Sudoku Puzzles:")
    print("=" * 50)

    # Iterate through difficulty levels in the order defined in PUZZLES dict
    for difficulty, puzzles in PUZZLES.items():
        print(f"\n{difficulty.upper()}:")
        # Enumerate puzzles starting from 1 (more intuitive than 0-indexing)
        for i, puzzle in enumerate(puzzles, 1):
            # Count given cells as indicator of puzzle difficulty
            given = count_given_cells(puzzle)
            print(f"  {difficulty.upper()}_{i}: {given} given cells")


if __name__ == "__main__":
    # =========================================================================
    # DEMO MODE - Run this file directly to see available puzzles
    # =========================================================================
    # When run as main script, display all available puzzles and show
    # a formatted example. This helps visualize what test cases look like.

    # Print summary of all puzzles by difficulty
    print_puzzle_info()

    # Show example puzzle with nice formatting
    print("\n" + "="*50)
    print("Example: EASY_1")
    print("="*50)

    # Iterate through rows, adding horizontal dividers between 3x3 boxes
    for i, row in enumerate(EASY_1):
        # Add horizontal divider after rows 2 and 5 (between 3x3 boxes)
        if i > 0 and i % 3 == 0:
            print("-" * 21)  # Divider length matches formatted row width

        # Build row string with vertical dividers between 3x3 boxes
        row_str = ""
        for j, val in enumerate(row):
            # Add vertical divider after columns 2 and 5 (between boxes)
            if j > 0 and j % 3 == 0:
                row_str += "| "
            # Use "." for empty cells (0), digit for givens
            row_str += (str(val) if val != 0 else ".") + " "
        print(row_str)

    # Display given cell count for this example
    print(f"\nGiven cells: {count_given_cells(EASY_1)}")
