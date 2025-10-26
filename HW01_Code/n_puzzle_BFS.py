# n_puzzle_BFS.py
# Implements Breadth-First Search for the n-puzzle problem
# BFS explores shallowest nodes first using a FIFO (First-In-First-Out) queue
# Guarantees finding the OPTIMAL (shortest) solution for unit-cost problems

from collections import deque  # Efficient FIFO queue implementation
from typing import Tuple, List, Dict, Optional, Set
import random, time

# Type alias: Position represents a puzzle state as a flattened tuple
# Example for 8-puzzle: (1, 2, 3, 4, 5, 6, 7, 8, 0) where 0 is the blank tile
Position = Tuple[int, ...]

class NPuzzle:
    """
    Represents an n-puzzle problem where n = m^2 - 1
    For m=3: 8-puzzle (3x3 board with 8 tiles + 1 blank)
    For m=4: 15-puzzle (4x4 board with 15 tiles + 1 blank)
    """
    def __init__(self, m: int = 3):
        assert m > 1, "Board size must be at least 2x2"
        self.m = m  # Board dimension (m x m)
        # Goal state: tiles in order 1,2,3,...,n with blank (0) at end
        # Example for 8-puzzle: (1, 2, 3, 4, 5, 6, 7, 8, 0)
        self.goal: Position = tuple([i for i in range(1, m*m)] + [0])

    def neighbors(self, s: Position) -> List[Position]:
        """
        Generate all valid successor states by sliding tiles into the blank space
        The blank can move up, down, left, or right (if not at board edge)
        
        Args:
            s: Current puzzle state
            
        Returns:
            List of all reachable states from current state (2-4 neighbors typically)
        """
        m = self.m
        z = s.index(0)  # Find index of blank tile (0)
        r, c = divmod(z, m)  # Convert 1D index to 2D row,col coordinates

        def swap(i: int, j: int) -> Position:
            """Helper: swap tiles at positions i and j, return new state"""
            a = list(s)
            a[i], a[j] = a[j], a[i]
            return tuple(a)

        out: List[Position] = []
        # Generate neighbors by sliding blank in each valid direction
        if r > 0:           # Can slide tile from above down (blank moves up)
            out.append(swap(z, z - m))
        if r < m-1:         # Can slide tile from below up (blank moves down)
            out.append(swap(z, z + m))
        if c > 0:           # Can slide tile from left (blank moves left)
            out.append(swap(z, z - 1))
        if c < m-1:         # Can slide tile from right (blank moves right)
            out.append(swap(z, z + 1))
        return out

    def is_goal(self, s: Position) -> bool:
        """
        Test if state s is the goal state
        
        Args:
            s: State to test
            
        Returns:
            True if s matches goal configuration
        """
        return s == self.goal

    def random_start(self, steps: int) -> Position:
        """
        Generate a random solvable starting state by making random moves from goal
        This ensures the resulting state is always solvable (maintains puzzle parity)
        
        Args:
            steps: Number of random moves to make from goal state
            
        Returns:
            A randomized but solvable starting state
        """
        s = self.goal
        for _ in range(steps):
            # Take a random legal move - this maintains solvability
            s = random.choice(self.neighbors(s))
        return s

def reconstruct(parent: Dict[Position, Optional[Position]], s: Position) -> List[Position]:
    """
    Reconstruct solution path by following parent pointers backward from goal to start
    
    Args:
        parent: Dictionary mapping each state to its parent state in the search tree
        s: Goal state to start reconstruction from
        
    Returns:
        List of states from start to goal (inclusive)
    """
    path: List[Position] = [s]
    # Follow parent pointers backward until we reach start (parent is None)
    while parent[s] is not None:
        s = parent[s]
        path.append(s)
    path.reverse()  # Reverse to get start-to-goal order
    return path

def breadth_first_search(puz: NPuzzle, start: Position):
    """
    Breadth-First Search: Explore shallowest nodes first using a FIFO queue
    
    BFS Characteristics:
    - Uses a queue (FIFO) as frontier
    - Explores all nodes at depth d before any at depth d+1
    - OPTIMAL: Guaranteed to find shortest path for unit-cost problems
    - Complete: Will find solution if one exists
    - Space complexity: O(b^d) where b=branching factor, d=solution depth
    - Time complexity: O(b^d)
    
    Why BFS is good for n-puzzle:
    - Guarantees shortest solution (fewest moves)
    - Works well for 8-puzzle with moderate scrambling
    - Can handle some 15-puzzle instances (with lighter scrambling)
    
    Args:
        puz: NPuzzle instance defining the problem
        start: Initial puzzle state
        
    Returns:
        Tuple of (solution_path, nodes_expanded, time_elapsed)
    """
    t0 = time.perf_counter()  # Start timing
    
    # Initialize frontier with start state
    # Queue (FIFO): states are explored in the order they were discovered
    # This ensures we explore by layers (all depth d before depth d+1)
    frontier = deque([start])
    
    # Track all states we've seen to avoid revisiting
    # In BFS, first visit to a state is always via shortest path
    explored: Set[Position] = {start}
    
    # Track parent pointers for solution reconstruction
    # Maps each state to the state it was reached from
    parent: Dict[Position, Optional[Position]] = {start: None}
    
    nodes_expanded = 0  # Count nodes we've examined (for performance metrics)

    while frontier:  # Continue until frontier is empty
        # Dequeue: get oldest state in frontier (FIFO = breadth-first)
        # This ensures we process all states at depth d before depth d+1
        s = frontier.popleft()
        
        # Goal test: check if we've reached the solution
        # Since BFS explores by depth, this is the SHORTEST path to goal
        if puz.is_goal(s):
            return reconstruct(parent, s), nodes_expanded, time.perf_counter() - t0
        
        # Expand this node: generate all successor states
        for t in puz.neighbors(s):
            # Only add new states (not yet explored)
            # Since we're doing graph search, we skip revisiting states
            if t not in explored:
                explored.add(t)         # Mark as seen
                parent[t] = s           # Record how we reached this state
                frontier.append(t)      # Add to back of queue (FIFO)
        
        # Count this node as expanded after generating its children
        nodes_expanded += 1

    # Frontier exhausted without finding goal: no solution exists
    # This shouldn't happen for solvable puzzle instances
    return None, nodes_expanded, time.perf_counter() - t0

def print_board(s: Position, m: int):
    """
    Display puzzle state as a human-readable m x m grid
    
    Args:
        s: Puzzle state to display
        m: Board dimension
    """
    for i in range(0, m*m, m):
        row = s[i:i+m]
        print(" ".join(str(x) for x in row))

def run_once(m: int, steps: int):
    """
    Run BFS once on a random puzzle instance
    
    Args:
        m: Board dimension (3 for 8-puzzle, 4 for 15-puzzle)
        steps: Number of random moves to scramble the puzzle
        
    Returns:
        Tuple of (solution_length, nodes_expanded, time)
    """
    puz = NPuzzle(m=m)
    start = puz.random_start(steps=steps)
    path, expanded, dt = breadth_first_search(puz, start)
    
    # Display results
    print(f"\nBFS on {m}x{m}")
    print("start:")
    print_board(start, m)
    print("moves:", (len(path) - 1) if path else None)  # Path length minus start state
    print("expanded:", expanded)
    print("time_s:", round(dt, 4))
    
    return (len(path) - 1) if path else None, expanded, dt

def run_trials_auto():
    """
    Automated test harness: run BFS on both 8-puzzle and 15-puzzle
    
    For each puzzle size, run exactly 3 trials and compute averages
    
    BFS performance characteristics:
    - 8-puzzle (3x3): Handles deep scrambles well (40+ moves)
    - 15-puzzle (4x4): More limited - use lighter scrambling (12 steps)
      BFS explores exponentially: b^d nodes where b≈2-4, d=solution depth
      15-puzzle solutions often 50-80 moves deep → huge search space
    
    Scramble configuration:
    - 3x3 with 40 steps: Creates challenges requiring ~30-40 moves to solve
    - 4x4 with 12 steps: Creates solvable instances BFS can handle in reasonable time
    """
    trials = 3  # Number of runs per puzzle size
    # Configuration: (board_size, scramble_steps)
    configs = [(3, 40), (4, 40)]
    
    for m, steps in configs:
        total_moves = 0
        total_expanded = 0
        total_time = 0.0
        solved = 0  # Count how many trials found solutions
        
        print("\n" + "="*32)
        print(f"Running {trials} trial(s) on {m}x{m} with steps={steps}")
        print("="*32)
        
        # Run exactly 'trials' number of independent tests
        for i in range(1, trials+1):
            print(f"\ntrial {i}:")
            moves, expanded, dt = run_once(m, steps)
            
            # Accumulate statistics
            if moves is not None:
                solved += 1
                total_moves += moves
            total_expanded += expanded
            total_time += dt
        
        # Compute and display averages
        avg_moves = (total_moves/solved) if solved else None
        avg_expanded = total_expanded/trials
        avg_time = total_time/trials
        
        print("\n--- summary ---")
        print("solved:", f"{solved}/{trials}")
        print("avg_moves:", round(avg_moves, 2) if avg_moves is not None else None)
        print("avg_expanded:", round(avg_expanded, 2))
        print("avg_time_s:", round(avg_time, 4))

if __name__ == "__main__":
    # Run automated trials on both 8-puzzle and 15-puzzle (3 trials each)
    run_trials_auto()