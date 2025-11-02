# n_puzzle_DFS.py
# Implements Depth-First Search for the n-puzzle problem
# DFS explores deepest nodes first using a LIFO (Last-In-First-Out) stack

from typing import Tuple, List, Dict, Optional, Set
import random, time

# Type alias: Position represents a puzzle state as a flattened tuple
# Example for 8-puzzle: (1, 2, 3, 4, 5, 6, 7, 8, 0) where 0 is the blank tile
Position = Tuple[int, ...]

# Safety limits to prevent infinite loops or excessive memory usage
MAX_NODES = 10000000  # Stop after expanding this many nodes
MAX_TIME_SEC = 60     # Stop after this many seconds (0 to disable)

# Trial configuration for automated testing
TRIALS_REQUIRED = 3         # Number of successful runs needed per puzzle size
MAX_ATTEMPTS_PER_SIZE = 20  # Maximum attempts before giving up

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

def depth_first_search(puz: NPuzzle, start: Position):
    """
    Depth-First Search: Explore deepest nodes first using a LIFO stack
    
    DFS Characteristics:
    - Uses a stack (LIFO) as frontier
    - Explores one branch completely before backtracking
    - NOT optimal: may find a long solution even if short one exists
    - Space efficient: O(bd) where b=branching factor, d=depth
    - Time complexity: O(b^m) where m=maximum depth (can be huge!)
    
    WARNING: DFS is impractical for 15-puzzle due to exploring very deep paths
    
    Args:
        puz: NPuzzle instance defining the problem
        start: Initial puzzle state
        
    Returns:
        Tuple of (solution_path, nodes_expanded, time_elapsed, status_string)
    """
    t0 = time.perf_counter()  # Start timing
    
    # Initialize frontier with start state
    # Stack (LIFO): most recently added states are explored first
    stack: List[Position] = [start]
    
    # Track parent pointers for solution reconstruction
    # Maps each state to the state it was reached from
    parent: Dict[Position, Optional[Position]] = {start: None}
    
    # Track explored states to avoid cycles (graph search)
    # Without this, DFS could loop forever in the state graph
    explored: Set[Position] = set()
    
    nodes_expanded = 0  # Count nodes we've examined (for performance metrics)

    while stack:  # Continue until stack is empty (all reachable states explored)
        # Check time limit to prevent hanging on hard problems
        if MAX_TIME_SEC and (time.perf_counter() - t0) >= MAX_TIME_SEC:
            return None, nodes_expanded, time.perf_counter() - t0, "timeout"
        
        # Pop from stack: get most recently added state (LIFO = depth-first)
        s = stack.pop()
        
        # Skip if already explored (can happen due to multiple paths to same state)
        if s in explored:
            continue
        
        # Goal test: check if we've reached the solution
        if puz.is_goal(s):
            return reconstruct(parent, s), nodes_expanded, time.perf_counter() - t0, "ok"
        
        # Mark state as explored so we don't revisit it
        explored.add(s)
        nodes_expanded += 1
        
        # Check node expansion limit to prevent excessive memory/time usage
        if nodes_expanded >= MAX_NODES:
            return None, nodes_expanded, time.perf_counter() - t0, "cap"
        
        # Generate and add all successor states to frontier
        for t in puz.neighbors(s):
            # Only consider states we haven't explored yet
            if t not in explored:
                # Only add to stack if we haven't seen this state before
                # This prevents duplicate entries in the stack
                if t not in parent:
                    parent[t] = s       # Record how we reached this state
                    stack.append(t)     # Add to frontier (will be explored deeply first)
    
    # Stack emptied without finding goal: no solution exists (shouldn't happen for solvable puzzles)
    return None, nodes_expanded, time.perf_counter() - t0, "failure"

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
    Run DFS once on a random puzzle instance
    
    Args:
        m: Board dimension (3 for 8-puzzle, 4 for 15-puzzle)
        steps: Number of random moves to scramble the puzzle
        
    Returns:
        Tuple of (solution_length, nodes_expanded, time, status)
    """
    puz = NPuzzle(m=m)
    start = puz.random_start(steps=steps)
    path, expanded, dt, status = depth_first_search(puz, start)
    
    # Display results
    print(f"\nDFS on {m}x{m}")
    print("start:")
    print_board(start, m)
    print("moves:", (len(path) - 1) if path else None)  # Path length minus start state
    print("expanded:", expanded)
    print("time_s:", round(dt, 4))
    print("status:", status)
    
    return (len(path) - 1) if path else None, expanded, dt, status

def run_trials_auto():
    """
    Automated test harness: run DFS on both 8-puzzle and 15-puzzle
    
    For each puzzle size, attempt to get TRIALS_REQUIRED successful solutions
    Retry on failure (timeout/cap) up to MAX_ATTEMPTS_PER_SIZE times
    
    NOTE: DFS performance characteristics:
    - 8-puzzle (3x3): Usually works with moderate scrambles
    - 15-puzzle (4x4): IMPRACTICAL - even minimal scrambles often timeout
      DFS explores depth-first, can wander into very deep non-solution paths
      This is a known limitation, not a bug
    """
    # Configuration: (board_size, scramble_steps)
    # 15-puzzle uses only 3 steps to have any hope of completing
    configs = [(3, 60), (4, 3)]
    
    for m, steps in configs:
        successes = 0
        attempts = 0
        total_moves = 0
        total_expanded = 0
        total_time = 0.0
        
        print("\n" + "="*36)
        print(f"Target: {TRIALS_REQUIRED} successful trial(s) on {m}x{m} (steps={steps})")
        if m == 4:
            print("NOTE: DFS is impractical for 15-puzzle; using minimal scramble")
            print("      Even with light scrambling, DFS may timeout frequently")
        print("="*36)
        
        # Keep trying until we get enough successful runs
        while successes < TRIALS_REQUIRED and attempts < MAX_ATTEMPTS_PER_SIZE:
            attempts += 1
            print(f"\nattempt {attempts}:")
            moves, expanded, dt, status = run_once(m, steps)
            
            # Count as success only if we found a solution
            if status == "ok" and moves is not None:
                successes += 1
                total_moves += moves
                total_expanded += expanded
                total_time += dt
            else:
                print("retrying due to", status)
        
        # Print summary statistics
        print("\n--- summary ---")
        print("attempts:", attempts)
        print("successful:", f"{successes}/{TRIALS_REQUIRED}")
        if successes > 0:
            print("avg_moves:", round(total_moves/successes, 2))
            print("avg_expanded:", round(total_expanded/successes, 2))
            print("avg_time_s:", round(total_time/successes, 4))
        else:
            print("avg_moves:", None)
            print("avg_expanded:", None)
            print("avg_time_s:", None)

if __name__ == "__main__":
    # Uncomment next line for reproducible results during testing/debugging
    # random.seed(0)
    
    # Run automated trials on both 8-puzzle and 15-puzzle
    run_trials_auto()