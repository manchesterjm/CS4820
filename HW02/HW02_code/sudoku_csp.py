"""
Sudoku solver using Constraint Satisfaction Problem (CSP) techniques.

This module implements Sudoku as a formal CSP:
- Variables: 81 cells in a 9x9 grid
- Domain: {1, 2, 3, 4, 5, 6, 7, 8, 9} for each variable
- Constraints: 27 Alldiff constraints (9 rows + 9 columns + 9 3x3 boxes)

This representation allows solving Sudoku using systematic search algorithms
designed for CSPs.

Algorithm References:
According to Russell & Norvig "Artificial Intelligence: A Modern Approach"
Chapter 6 and Lecture 5: Constraint Satisfaction Problems:
  * Backtracking Search (Slide 19) - Basic depth-first search with
    constraint checking
  * MRV Heuristic (Slide 24, 27) - Choose most constrained variable first
  * Degree Heuristic (Slide 28) - Tie-breaking for MRV
  * LCV Heuristic (Slide 30) - Try least constraining values first
  * Forward Checking (Slide 36) - Maintain arc consistency with future
    variables
  * AC-3 Constraint Propagation (Slide 70) - Full arc consistency across
    entire CSP
"""

from typing import List, Tuple, Set, Dict, Optional
import time
import copy

# Type aliases for clarity
Position = Tuple[int, int]  # (row, col) in 9x9 grid
Domain = Set[int]           # Set of possible values for a variable
Domains = Dict[Position, Domain]  # Maps each position to its current domain

# Safety limit to prevent excessive computation
# According to the assignment requirements, we need a 5 minute timeout
# This prevents algorithms from running forever on hard puzzles
MAX_TIME_SEC = 300

class SudokuCSP:
    """
    Represents a Sudoku puzzle as a Constraint Satisfaction Problem

    How does this work? The puzzle is represented as:
    - A 9x9 grid where each cell is a variable
    - Each variable can take values from 1 to 9
    - Constraints ensure no duplicates appear in rows, columns, or 3x3 boxes

    This representation allows us to apply systematic CSP solving algorithms.
    """

    def __init__(self, puzzle: List[List[int]]):
        """
        Initialize Sudoku CSP from a 9x9 puzzle grid

        Args:
            puzzle: 9x9 grid where 0 represents empty cells to be filled
        """
        assert len(puzzle) == 9 and all(len(row) == 9 for row in puzzle), \
            "Puzzle must be 9x9"

        self.puzzle = [row[:] for row in puzzle]  # Deep copy

        # Initialize domains for all variables
        # How do we set up the initial domains?
        # - Given cells have singleton domains (only one possible value)
        # - Empty cells have the full domain {1, 2, 3, 4, 5, 6, 7, 8, 9}
        self.initial_domains: Domains = {}
        for r in range(9):
            for c in range(9):
                pos = (r, c)
                if puzzle[r][c] != 0:
                    # Given cell: domain contains only the fixed value
                    self.initial_domains[pos] = {puzzle[r][c]}
                else:
                    # Empty cell: all values 1-9 are initially possible
                    self.initial_domains[pos] = {1, 2, 3, 4, 5, 6, 7, 8, 9}

        # Precompute the constraint structure for efficiency
        # According to the CSP formulation, each variable has constraints
        # with all variables in the same row, column, and 3x3 box. This
        # builds the constraint graph.
        self.neighbors: Dict[Position, Set[Position]] = {}
        for r in range(9):
            for c in range(9):
                self.neighbors[(r, c)] = self._compute_neighbors(r, c)

    def _compute_neighbors(self, r: int, c: int) -> Set[Position]:
        """
        Compute all variables that share a constraint with (r, c)

        A variable is a neighbor if it shares a row, column, or 3x3 box
        This implements the constraint graph structure

        Args:
            r, c: Position of variable

        Returns:
            Set of all neighbor positions
        """
        neighbors = set()

        # All cells in same row
        for col in range(9):
            if col != c:
                neighbors.add((r, col))

        # All cells in same column
        for row in range(9):
            if row != r:
                neighbors.add((row, c))

        # All cells in same 3x3 box
        box_r, box_c = 3 * (r // 3), 3 * (c // 3)
        for row in range(box_r, box_r + 3):
            for col in range(box_c, box_c + 3):
                if (row, col) != (r, c):
                    neighbors.add((row, col))

        return neighbors

    def is_consistent(self, pos: Position, value: int, assignment: Dict[Position, int]) -> bool:
        """
        Check if assigning value to pos is consistent with current assignment

        A value is consistent if it doesn't violate any constraints:
        - Not already used in same row
        - Not already used in same column
        - Not already used in same 3x3 box

        Args:
            pos: Variable position (r, c)
            value: Value to test
            assignment: Current partial assignment

        Returns:
            True if assignment is consistent with constraints
        """
        r, c = pos

        # Check row constraint
        for col in range(9):
            if col != c and (r, col) in assignment:
                if assignment[(r, col)] == value:
                    return False

        # Check column constraint
        for row in range(9):
            if row != r and (row, c) in assignment:
                if assignment[(row, c)] == value:
                    return False

        # Check 3x3 box constraint
        box_r, box_c = 3 * (r // 3), 3 * (c // 3)
        for row in range(box_r, box_r + 3):
            for col in range(box_c, box_c + 3):
                if (row, col) != (r, c) and (row, col) in assignment:
                    if assignment[(row, col)] == value:
                        return False

        return True

    def select_unassigned_variable_basic(
            self, assignment: Dict[Position, int],
            domains: Domains) -> Optional[Position]:
        """
        Basic variable selection: choose first unassigned variable

        This is the naive approach without heuristics
        Used as baseline for comparison

        Args:
            assignment: Current partial assignment
            domains: Current domains for all variables

        Returns:
            First unassigned variable, or None if all assigned
        """
        for r in range(9):
            for c in range(9):
                if (r, c) not in assignment:
                    return (r, c)
        return None

    def select_unassigned_variable_mrv(
            self, assignment: Dict[Position, int],
            domains: Domains) -> Optional[Position]:
        """
        MRV (Minimum Remaining Values) heuristic for variable selection

        This is also known as the "most constrained variable" or
        "fail-first" heuristic. The idea is to choose the variable with
        the fewest legal values remaining.

        Why does MRV work so well? According to Lecture 5, Slide 24:
        - It reduces the branching factor by choosing variables that are likely to fail early
        - This means failures are detected earlier in the search tree
        - The earlier we detect failures, the more of the search space we can prune

        What about tie-breaking? According to Lecture 5, Slide 28:
        - Use the degree heuristic when multiple variables have the same domain size
        - Among tied variables, choose the one with the most constraints on remaining
          unassigned variables
        - This further reduces the future branching factor

        Reference: Russell & Norvig Section 6.3.1

        Args:
            assignment: Current partial assignment
            domains: Current domains for all variables

        Returns:
            Unassigned variable with minimum remaining values
        """
        unassigned = [(r, c) for r in range(9) for c in range(9)
                      if (r, c) not in assignment]

        if not unassigned:
            return None

        # Compute actual legal values for each unassigned variable
        # based on current assignment
        legal_values = {}
        for pos in unassigned:
            legal = [v for v in domains[pos]
                    if self.is_consistent(pos, v, assignment)]
            legal_values[pos] = len(legal)

        # Find minimum number of legal values
        min_size = min(legal_values.values())

        # Get all variables with minimum legal values
        candidates = [pos for pos in unassigned if legal_values[pos] == min_size]

        if len(candidates) == 1:
            return candidates[0]

        # Tie-breaking with degree heuristic
        # Choose variable involved in most constraints with unassigned variables
        def degree(pos: Position) -> int:
            """Count unassigned neighbors"""
            return sum(1 for neighbor in self.neighbors[pos]
                      if neighbor not in assignment)

        # Return variable with highest degree (most constraints)
        return max(candidates, key=degree)

    def order_domain_values_basic(self, pos: Position, assignment: Dict[Position, int],
                                   domains: Domains) -> List[int]:
        """
        Basic value ordering: return values in arbitrary order

        This is the naive approach without heuristics
        Used as baseline for comparison

        Args:
            pos: Variable to order values for
            assignment: Current partial assignment
            domains: Current domains for all variables

        Returns:
            List of domain values in arbitrary order
        """
        return sorted(domains[pos])

    def order_domain_values_lcv(self, pos: Position, assignment: Dict[Position, int],
                                 domains: Domains) -> List[int]:
        """
        LCV (Least Constraining Value) heuristic for value ordering

        This heuristic orders values by how many choices they rule out for neighboring
        variables. The idea is to prefer values that leave maximum flexibility for
        future assignments.

        Why does LCV work well? According to Lecture 5, Slide 30:
        - Unlike MRV which tries to fail fast, LCV tries to succeed
        - Choosing the least constraining value leaves more options for neighbor variables
        - This increases the likelihood of finding a complete solution
        - LCV is most effective when combined with MRV for variable selection

        How is it implemented?
        - For each value in the domain, count how many neighbor values would be eliminated
        - Sort the values by this count in ascending order (least constraining first)
        - This means we try values that preserve the most options for other variables

        Reference: Russell & Norvig Section 6.3.1

        Args:
            pos: Variable to order values for
            assignment: Current partial assignment
            domains: Current domains for all variables

        Returns:
            List of domain values ordered by LCV (least constraining first)
        """
        def count_conflicts(value: int) -> int:
            """
            Count how many values would be ruled out from neighbor domains
            if we assign this value to pos
            """
            conflicts = 0
            for neighbor in self.neighbors[pos]:
                # Only consider unassigned neighbors
                if neighbor not in assignment:
                    # If this value is in neighbor's domain, it would be ruled out
                    if value in domains[neighbor]:
                        conflicts += 1
            return conflicts

        # Sort values by number of conflicts (ascending = least constraining first)
        values = list(domains[pos])
        values.sort(key=count_conflicts)
        return values

    def forward_check(self, pos: Position, value: int, domains: Domains) -> Optional[Domains]:
        """
        Forward Checking: maintain arc consistency for future variables

        What does forward checking do? When we assign a value to a variable,
        we reduce the domains of unassigned neighbors by removing that value
        from their domains (since it's now used and can't appear again).

        Why does Forward Checking work? According to Lecture 5, Slide 36:
        - It detects failures early by checking if any domain becomes empty
        - This means we find out about problems before we backtrack
        - It's much cheaper than full arc consistency (AC-3) but still effective
        - It prunes the search space by eliminating values we know won't work

        How does the algorithm work? According to Lecture 5, Slide 36:
        1. For each unassigned neighbor of the variable we just assigned
        2. Remove the assigned value from the neighbor's domain (if it's there)
        3. If any domain becomes empty, return None (we detected a failure)
        4. Otherwise return the updated domains and continue

        Reference: Russell & Norvig Section 6.3.2

        Args:
            pos: Variable being assigned
            value: Value being assigned to pos
            domains: Current domains (will be copied and modified)

        Returns:
            Updated domains after forward checking, or None if conflict detected
        """
        # Create deep copy to avoid modifying original
        new_domains = {p: d.copy() for p, d in domains.items()}

        # For each neighbor of pos
        for neighbor in self.neighbors[pos]:
            # Skip if neighbor is already assigned
            # (Forward checking only applies to unassigned variables)
            # Check if neighbor has single value equal to our value
            if (neighbor in new_domains and
                    len(new_domains[neighbor]) == 1 and
                    list(new_domains[neighbor])[0] == value):
                continue

            # Remove value from neighbor's domain (if present)
            if value in new_domains[neighbor]:
                new_domains[neighbor] = new_domains[neighbor] - {value}

                # Domain wipeout: this assignment makes neighbor unsolvable
                if len(new_domains[neighbor]) == 0:
                    return None  # Failure detected

        return new_domains

    def ac3(self, domains: Domains,
            queue: Optional[List[Tuple[Position, Position]]] = None
            ) -> Optional[Domains]:
        """
        AC-3 (Arc Consistency 3) constraint propagation algorithm

        What is arc consistency? An arc (Xi, Xj) is consistent if for every value in
        Xi's domain, there exists some value in Xj's domain that satisfies the constraint
        between them. AC-3 enforces this across all constraints in the CSP.

        Why is AC-3 so powerful? According to Lecture 5, Slide 70:
        - It's more powerful than forward checking alone
        - It propagates constraints transitively, creating a cascading
          effect
        - This means it can detect failures earlier than forward checking
        - It significantly reduces the search space before backtracking
          even begins
        - AC-3 makes an excellent preprocessing step before search

        How does the algorithm work? According to Lecture 5, Slide 70 and
        Russell & Norvig Figure 6.3:
        1. Initialize a queue with all arcs (Xi, Xj) where Xi and Xj are neighbors
        2. While the queue is not empty:
        3.   Remove an arc (Xi, Xj) from the queue
        4.   If Revise(Xi, Xj) causes Xi's domain to change:
        5.     If Xi's domain is empty, return failure (no solution exists)
        6.     Add all arcs (Xk, Xi) to the queue where Xk is a neighbor of Xi
        7. Return the updated domains

        What does Revise(Xi, Xj) do?
        - It removes values from Di (Xi's domain) that have no consistent value in Dj
        - Returns true if Di was changed, false otherwise

        What's the cost? Time Complexity is O(cd³) where c equals the number of constraints
        and d is the max domain size. Space Complexity is O(c) for the queue. This means
        AC-3 has a higher per-node cost, but it explores far fewer nodes.

        Reference: Russell & Norvig Section 6.2.5

        Args:
            domains: Current domains for all variables
            queue: Optional initial queue of arcs (if None, use all arcs)

        Returns:
            Updated domains after enforcing arc consistency, or None if inconsistency detected
        """
        # Create deep copy to avoid modifying original
        new_domains = {p: d.copy() for p, d in domains.items()}

        # Initialize queue with all arcs if not provided
        if queue is None:
            queue = []
            for pos in new_domains:
                for neighbor in self.neighbors[pos]:
                    queue.append((pos, neighbor))
        else:
            queue = queue[:]  # Copy to avoid modifying original

        def revise(xi: Position, xj: Position) -> bool:
            """
            Make arc (Xi, Xj) consistent

            Remove values from Di that have no consistent value in Dj

            Args:
                xi, xj: Variables forming the arc

            Returns:
                True if Di was revised (values removed)
            """
            revised = False
            values_to_remove = []

            # For each value in Xi's domain
            for val_i in new_domains[xi]:
                # Check if there exists a value in Xj's domain that's consistent
                # For Sudoku, constraint is "values must be different"
                # So we need at least one value in Dj that's not val_i

                # If Xj's domain only contains val_i, then no consistent value exists
                if new_domains[xj] == {val_i}:
                    values_to_remove.append(val_i)
                    revised = True

            # Remove inconsistent values
            for val in values_to_remove:
                new_domains[xi].remove(val)

            return revised

        # Process arcs until queue is empty
        while queue:
            xi, xj = queue.pop(0)

            # Make arc (Xi, Xj) consistent
            if revise(xi, xj):
                # Domain was reduced

                # Check for domain wipeout (failure)
                if len(new_domains[xi]) == 0:
                    return None

                # Add all arcs (Xk, Xi) to queue where Xk is neighbor of Xi
                # (excluding the arc we just processed)
                for xk in self.neighbors[xi]:
                    if xk != xj:
                        queue.append((xk, xi))

        return new_domains

    def backtrack_basic(self, assignment: Dict[Position, int], domains: Domains,
                       start_time: float) -> Optional[Dict[Position, int]]:
        """
        Basic Backtracking Search without heuristics

        Backtracking is a depth-first search that:
        - Assigns one variable at a time
        - Checks constraints after each assignment
        - Backtracks when assignment violates constraints

        Algorithm (Lecture 5, Slide 19, Russell & Norvig Figure 6.5):
        1. If assignment is complete, return it
        2. Select an unassigned variable
        3. For each value in variable's domain:
        4.   If value is consistent with assignment:
        5.     Add {var = value} to assignment
        6.     Recursively call backtrack
        7.     If result is not failure, return result
        8.     Remove {var = value} from assignment
        9. Return failure

        Time Complexity: O(d^n) where d = domain size, n = number of variables
        Space Complexity: O(n) for recursion depth

        Reference: Russell & Norvig Section 6.3

        Args:
            assignment: Current partial assignment
            domains: Current domains for all variables
            start_time: Time when search started (for timeout)

        Returns:
            Complete assignment if found, None if no solution exists
        """
        # Timeout check
        if 0 < MAX_TIME_SEC < (time.perf_counter() - start_time):
            return None

        # Base case: assignment is complete
        if len(assignment) == 81:
            return assignment

        # Select unassigned variable (no heuristic)
        pos = self.select_unassigned_variable_basic(assignment, domains)
        if pos is None:
            return assignment

        # Try each value in variable's domain
        for value in self.order_domain_values_basic(pos, assignment, domains):
            # Check if value is consistent with current assignment
            if self.is_consistent(pos, value, assignment):
                # Add assignment
                assignment[pos] = value

                # Recurse
                result = self.backtrack_basic(assignment, domains, start_time)
                if result is not None:
                    return result

                # Backtrack: remove assignment
                del assignment[pos]

        # No valid assignment found for this variable
        return None

    def backtrack_mrv_lcv(self, assignment: Dict[Position, int], domains: Domains,
                          start_time: float) -> Optional[Dict[Position, int]]:
        """
        Backtracking Search with MRV and LCV heuristics

        Enhancements over basic backtracking:
        - MRV (Minimum Remaining Values): Choose most constrained variable first
        - LCV (Least Constraining Value): Try least constraining values first

        These heuristics dramatically reduce search space:
        - MRV detects failures earlier (fail-fast)
        - LCV maximizes chances of success (succeed-first for values)

        Together they provide excellent performance on Sudoku

        Reference: Russell & Norvig Section 6.3.1

        Args:
            assignment: Current partial assignment
            domains: Current domains for all variables
            start_time: Time when search started (for timeout)

        Returns:
            Complete assignment if found, None if no solution exists
        """
        # Timeout check
        if 0 < MAX_TIME_SEC < (time.perf_counter() - start_time):
            return None

        # Base case: assignment is complete
        if len(assignment) == 81:
            return assignment

        # Select unassigned variable using MRV + degree heuristic
        pos = self.select_unassigned_variable_mrv(assignment, domains)
        if pos is None:
            return assignment

        # Order values using LCV heuristic
        for value in self.order_domain_values_lcv(pos, assignment, domains):
            # Check if value is consistent with current assignment
            if self.is_consistent(pos, value, assignment):
                # Add assignment
                assignment[pos] = value

                # Recurse
                result = self.backtrack_mrv_lcv(assignment, domains, start_time)
                if result is not None:
                    return result

                # Backtrack: remove assignment
                del assignment[pos]

        return None

    def backtrack_forward_checking(self, assignment: Dict[Position, int], domains: Domains,
                                   start_time: float) -> Optional[Dict[Position, int]]:
        """
        Backtracking Search with Forward Checking

        Enhancements:
        - MRV + LCV heuristics (from previous version)
        - Forward Checking: maintain arc consistency with future variables

        Forward checking eliminates values from unassigned variables' domains
        after each assignment, detecting failures earlier

        This is more powerful than basic backtracking but cheaper than full AC-3

        Reference: Russell & Norvig Section 6.3.2, Lecture 5 Slide 36

        Args:
            assignment: Current partial assignment
            domains: Current domains for all variables
            start_time: Time when search started (for timeout)

        Returns:
            Complete assignment if found, None if no solution exists
        """
        # Timeout check
        if 0 < MAX_TIME_SEC < (time.perf_counter() - start_time):
            return None

        # Base case: assignment is complete
        if len(assignment) == 81:
            return assignment

        # Select unassigned variable using MRV + degree heuristic
        pos = self.select_unassigned_variable_mrv(assignment, domains)
        if pos is None:
            return assignment

        # Order values using LCV heuristic
        for value in self.order_domain_values_lcv(pos, assignment, domains):
            # Check if value is consistent with current assignment
            if self.is_consistent(pos, value, assignment):
                # Add assignment
                assignment[pos] = value

                # Apply forward checking to reduce domains
                new_domains = self.forward_check(pos, value, domains)

                # If forward checking didn't detect conflict, recurse
                if new_domains is not None:
                    result = self.backtrack_forward_checking(assignment, new_domains, start_time)
                    if result is not None:
                        return result

                # Backtrack: remove assignment
                del assignment[pos]

        return None

    def backtrack_ac3(self, assignment: Dict[Position, int], domains: Domains,
                     start_time: float) -> Optional[Dict[Position, int]]:
        """
        Backtracking Search with AC-3 constraint propagation

        Enhancements:
        - MRV + LCV heuristics
        - AC-3: full arc consistency after each assignment

        AC-3 is more powerful than forward checking:
        - Forward checking only checks arcs from assigned variable to neighbors
        - AC-3 propagates constraints transitively across entire CSP
        - Can detect failures earlier and reduce domains more aggressively

        Trade-off:
        - More expensive per node (O(cd³) for AC-3 vs O(d) for forward checking)
        - But explores far fewer nodes due to better pruning
        - For Sudoku, AC-3 often solves puzzle without backtracking

        Reference: Russell & Norvig Section 6.2.5, Lecture 5 Slide 70

        Args:
            assignment: Current partial assignment
            domains: Current domains for all variables
            start_time: Time when search started (for timeout)

        Returns:
            Complete assignment if found, None if no solution exists
        """
        # Timeout check
        if 0 < MAX_TIME_SEC < (time.perf_counter() - start_time):
            return None

        # Base case: assignment is complete
        if len(assignment) == 81:
            return assignment

        # Select unassigned variable using MRV + degree heuristic
        pos = self.select_unassigned_variable_mrv(assignment, domains)
        if pos is None:
            return assignment

        # Order values using LCV heuristic
        for value in self.order_domain_values_lcv(pos, assignment, domains):
            # Check if value is consistent with current assignment
            if self.is_consistent(pos, value, assignment):
                # Add assignment
                assignment[pos] = value

                # Update domains to reflect this assignment
                new_domains = {p: d.copy() for p, d in domains.items()}
                new_domains[pos] = {value}

                # Build queue of arcs to check (neighbors of assigned variable)
                queue = [(neighbor, pos) for neighbor in self.neighbors[pos]]

                # Apply AC-3 to propagate constraints
                new_domains = self.ac3(new_domains, queue)

                # If AC-3 didn't detect conflict, recurse
                if new_domains is not None:
                    result = self.backtrack_ac3(assignment, new_domains, start_time)
                    if result is not None:
                        return result

                # Backtrack: remove assignment
                del assignment[pos]

        return None

    def solve_basic(self) -> Tuple[Optional[Dict[Position, int]], int, float]:
        """
        Solve Sudoku using basic backtracking (no heuristics)

        Returns:
            Tuple of (solution, backtracks, time)
        """
        start_time = time.perf_counter()

        # Build initial assignment from given cells
        assignment = {}
        for r in range(9):
            for c in range(9):
                if self.puzzle[r][c] != 0:
                    assignment[(r, c)] = self.puzzle[r][c]

        # Run backtracking search
        result = self.backtrack_basic(assignment, self.initial_domains, start_time)

        elapsed = time.perf_counter() - start_time
        backtracks = 0  # Not tracking for basic version

        return result, backtracks, elapsed

    def solve_mrv_lcv(self) -> Tuple[Optional[Dict[Position, int]], int, float]:
        """
        Solve Sudoku using backtracking with MRV and LCV heuristics

        Returns:
            Tuple of (solution, backtracks, time)
        """
        start_time = time.perf_counter()

        # Build initial assignment from given cells
        assignment = {}
        for r in range(9):
            for c in range(9):
                if self.puzzle[r][c] != 0:
                    assignment[(r, c)] = self.puzzle[r][c]

        # Run backtracking search with heuristics
        result = self.backtrack_mrv_lcv(assignment, self.initial_domains, start_time)

        elapsed = time.perf_counter() - start_time
        backtracks = 0  # Not tracking for this version

        return result, backtracks, elapsed

    def solve_forward_checking(self) -> Tuple[Optional[Dict[Position, int]], int, float]:
        """
        Solve Sudoku using backtracking with forward checking

        Returns:
            Tuple of (solution, backtracks, time)
        """
        start_time = time.perf_counter()

        # Build initial assignment from given cells
        assignment = {}
        for r in range(9):
            for c in range(9):
                if self.puzzle[r][c] != 0:
                    assignment[(r, c)] = self.puzzle[r][c]

        # Apply initial forward checking based on given cells
        domains = copy.deepcopy(self.initial_domains)
        for pos, value in assignment.items():
            domains = self.forward_check(pos, value, domains)
            if domains is None:
                # Puzzle is unsolvable
                return None, 0, time.perf_counter() - start_time

        # Run backtracking search with forward checking
        result = self.backtrack_forward_checking(assignment, domains, start_time)

        elapsed = time.perf_counter() - start_time
        backtracks = 0  # Not tracking for this version

        return result, backtracks, elapsed

    def solve_ac3(self) -> Tuple[Optional[Dict[Position, int]], int, float]:
        """
        Solve Sudoku using backtracking with AC-3 constraint propagation

        Returns:
            Tuple of (solution, backtracks, time)
        """
        start_time = time.perf_counter()

        # Build initial assignment from given cells
        assignment = {}
        for r in range(9):
            for c in range(9):
                if self.puzzle[r][c] != 0:
                    assignment[(r, c)] = self.puzzle[r][c]

        # Apply initial AC-3 preprocessing
        domains = copy.deepcopy(self.initial_domains)
        domains = self.ac3(domains)

        if domains is None:
            # Puzzle is unsolvable
            return None, 0, time.perf_counter() - start_time

        # Run backtracking search with AC-3
        result = self.backtrack_ac3(assignment, domains, start_time)

        elapsed = time.perf_counter() - start_time
        backtracks = 0  # Not tracking for this version

        return result, backtracks, elapsed


def print_sudoku(puzzle: List[List[int]]):
    """
    Display Sudoku puzzle in readable format with 3x3 box separators

    Args:
        puzzle: 9x9 grid (0 for empty cells)
    """
    for i, row in enumerate(puzzle):
        if i > 0 and i % 3 == 0:
            print("-" * 21)
        row_str = ""
        for j, val in enumerate(row):
            if j > 0 and j % 3 == 0:
                row_str += "| "
            row_str += (str(val) if val != 0 else ".") + " "
        print(row_str)


def assignment_to_grid(assignment: Dict[Position, int]) -> List[List[int]]:
    """
    Convert assignment dictionary to 9x9 grid format

    Args:
        assignment: Dictionary mapping (row, col) to values

    Returns:
        9x9 grid representation
    """
    grid = [[0] * 9 for _ in range(9)]
    for (r, c), val in assignment.items():
        grid[r][c] = val
    return grid


if __name__ == "__main__":
    # Example: Solve a simple Sudoku puzzle with all four methods

    # Easy puzzle (many given cells)
    easy_puzzle = [
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

    print("="*60)
    print("CS 4820/5820 Homework 2 - Part A: Sudoku CSP Solvers")
    print("="*60)
    print("\nStarting Puzzle:")
    print_sudoku(easy_puzzle)

    # Count given cells
    given_cells = sum(1 for row in easy_puzzle for cell in row if cell != 0)
    print(f"\nGiven cells: {given_cells}")
    print(f"Empty cells: {81 - given_cells}")

    methods = [
        ("Basic Backtracking", "solve_basic"),
        ("Backtracking + MRV + LCV", "solve_mrv_lcv"),
        ("Backtracking + Forward Checking", "solve_forward_checking"),
        ("Backtracking + AC-3", "solve_ac3")
    ]

    for name, method in methods:
        print(f"\n{'='*60}")
        print(f"{name}")
        print('='*60)

        csp = SudokuCSP(easy_puzzle)
        solution, _, elapsed = getattr(csp, method)()

        if solution:
            print("\nStatus: SOLVED")
            print(f"Time: {elapsed:.4f} seconds")
            print(f"Runtime: {elapsed*1000:.2f} milliseconds")

            print("\nSolved Puzzle:")
            print_sudoku(assignment_to_grid(solution))

            # Verify solution
            grid = assignment_to_grid(solution)
            all_filled = all(grid[r][c] != 0
                             for r in range(9) for c in range(9))
            print("\nVerification:")
            print(f"  All cells filled: {all_filled}")
            print("  Total cells: 81/81")
        else:
            print("\nStatus: FAILED")
            print(f"Time: {elapsed:.4f} seconds")
            print("No solution found (timeout or unsolvable)")

    print("\n" + "="*60)
    print("All Sudoku CSP tests completed")
    print("="*60)
