# n_puzzle_IDS.py
# Implements Iterative Deepening Search for the n-puzzle problem
# IDS combines the space efficiency of DFS with the optimality of BFS
# Works by running depth-limited DFS with increasing depth limits

from typing import Tuple, List, Dict, Optional, Set
import random, time

# Type alias: Position represents a puzzle state as a flattened tuple
# Example for 8-puzzle: (1, 2, 3, 4, 5, 6, 7, 8, 0) where 0 is the blank tile
Position = Tuple[int, ...]

# Safety limits to prevent excessive computation
MAX_NODES_TOTAL = 1500000  # Total node expansions allowed across ALL depth iterations
MAX_TIME_SEC = 60          # Time limit per IDS run (0 to disable)
MAX_DEPTH_CAP = 80         # Maximum depth limit to try (prevents infinite loops)

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
        self.goal: Position = tuple([i for i in range(1, m*m)] + [0])

    def neighbors(self, s: Position) -> List[Position]:
        """
        Generate all valid successor states by sliding tiles into the blank space
        
        Args:
            s: Current puzzle state
            
        Returns:
            List of all reachable states from current state (2-4 neighbors)
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
        if r > 0: out.append(swap(z, z - m))      # up
        if r < m-1: out.append(swap(z, z + m))    # down
        if c > 0: out.append(swap(z, z - 1))      # left
        if c < m-1: out.append(swap(z, z + 1))    # right
        return out

    def is_goal(self, s: Position) -> bool:
        """Test if state s is the goal state"""
        return s == self.goal

    def random_start(self, steps: int) -> Position:
        """
        Generate a random solvable starting state by random walk from goal
        This ensures the resulting state is always solvable
        
        Args:
            steps: Number of random moves to make from goal state
            
        Returns:
            A randomized but solvable starting state
        """
        s = self.goal
        for _ in range(steps):
            s = random.choice(self.neighbors(s))
        return s

def reconstruct(parent: Dict[Position, Optional[Position]], s: Position) -> List[Position]:
    """
    Reconstruct solution path by following parent pointers backward from goal to start
    
    Args:
        parent: Dictionary mapping each state to its parent state
        s: Goal state to start reconstruction from
        
    Returns:
        List of states from start to goal (inclusive)
    """
    path: List[Position] = [s]
    while parent[s] is not None:
        s = parent[s]
        path.append(s)
    path.reverse()
    return path

def depth_limited_search(puz: NPuzzle, start: Position, limit: int,
                         t0: float, max_time_sec: int,
                         node_budget: int) -> tuple:
    """
    Depth-Limited Search: DFS that stops at a specified depth limit
    
    This is a helper for IDS. It performs DFS but cuts off at depth=limit.
    Uses recursion with path checking to avoid cycles.
    
    Returns different status codes:
    - "ok": Found goal within depth limit
    - "cutoff": Reached depth limit, solution may exist deeper
    - "failure": Exhausted all paths at this depth, no solution exists
    - "timeout": Exceeded time limit
    - "cap": Exceeded node budget
    
    Args:
        puz: NPuzzle instance
        start: Initial state
        limit: Maximum depth to explore
        t0: Start time (for timeout checking)
        max_time_sec: Time limit in seconds
        node_budget: Maximum nodes allowed for this iteration
        
    Returns:
        Tuple of (status, goal_state_or_None, nodes_used, parent_dict)
    """
    # Initialize parent tracking for path reconstruction
    parent: Dict[Position, Optional[Position]] = {start: None}
    
    # Track current path to detect cycles (path checking)
    # More memory efficient than tracking all explored for DLS
    on_path: Set[Position] = set()
    
    nodes_used = 0  # Count nodes expanded in this DLS iteration

    def rec(s: Position, depth: int) -> tuple:
        """
        Recursive DFS with depth limit
        
        Args:
            s: Current state
            depth: Current depth from start
            
        Returns:
            Tuple of (status_string, goal_state_or_None)
        """
        nonlocal nodes_used
        
        # Check resource limits
        if max_time_sec and (time.perf_counter() - t0) >= max_time_sec:
            return "timeout", None
        if nodes_used >= node_budget:
            return "cap", None
        
        # Goal test: found solution!
        if puz.is_goal(s):
            return "ok", s
        
        # Depth limit reached: cutoff (solution might exist deeper)
        if depth == limit:
            return "cutoff", None
        
        # Mark state as on current path (cycle detection)
        on_path.add(s)
        nodes_used += 1
        
        cutoff_seen = False  # Track if any child hit cutoff
        
        # Explore all neighbors
        for t in puz.neighbors(s):
            # Skip if already on current path (would create cycle)
            if t in on_path:
                continue
            
            # Record parent if first time seeing this state
            if t not in parent:
                parent[t] = s
            
            # Recursively explore this neighbor
            status, goal = rec(t, depth+1)
            
            # Found goal: propagate success upward
            if status == "ok":
                return "ok", goal
            
            # Track if we hit cutoff (means solution might exist deeper)
            if status == "cutoff":
                cutoff_seen = True
            
            # Propagate timeout/cap immediately
            if status in ("timeout", "cap"):
                return status, None
        
        # Remove from path (backtrack)
        on_path.remove(s)
        
        # Return cutoff if any child was cutoff, else failure
        # Cutoff means "try deeper", failure means "no solution this way"
        return ("cutoff" if cutoff_seen else "failure"), None

    # Start recursive search from depth 0
    status, goal = rec(start, 0)
    return status, (goal if status == "ok" else None), nodes_used, parent

def iterative_deepening_search(puz: NPuzzle, start: Position,
                               max_depth_cap: int,
                               max_nodes_total: int,
                               max_time_sec: int):
    """
    Iterative Deepening Search: Run DLS with increasing depth limits
    
    IDS Characteristics:
    - Combines DFS's space efficiency with BFS's optimality
    - Runs DLS with limit=0, then 1, then 2, etc. until solution found
    - OPTIMAL: Finds shortest path (like BFS)
    - Space efficient: O(bd) like DFS, not O(b^d) like BFS
    - Time complexity: O(b^d) - seems wasteful but only ~b/(b-1) overhead
    
    Why IDS is good for n-puzzle:
    - Guaranteed shortest solution (optimal)
    - Much less memory than BFS
    - Better than DFS for finding solutions at unknown depths
    - Good balance for 15-puzzle where BFS uses too much memory
    
    How it works:
    1. Try DLS with limit=0 (only check start state)
    2. Try DLS with limit=1 (one move deep)
    3. Try DLS with limit=2 (two moves deep)
    4. Continue until solution found or max depth reached
    
    Yes, we re-expand states, but the exponential growth means most work
    is at the deepest level anyway!
    
    Example of why overhead is acceptable:
    - At depth d, we expand b^d nodes
    - Total over all iterations: b^0 + b^1 + ... + b^d = (b^(d+1)-1)/(b-1)
    - This is only about b/(b-1) times more than just b^d
    - For b=4: only 1.33x overhead!
    
    Args:
        puz: NPuzzle instance
        start: Initial state
        max_depth_cap: Maximum depth to try
        max_nodes_total: Total node budget across all iterations
        max_time_sec: Time limit
        
    Returns:
        Tuple of (path, total_nodes, time, status, final_depth_limit)
    """
    t0 = time.perf_counter()  # Start timing
    total_nodes = 0  # Track total nodes across all DLS iterations
    
    # Try increasing depth limits: 0, 1, 2, 3, ...
    for L in range(0, max_depth_cap + 1):
        # Calculate remaining node budget for this iteration
        remaining_nodes = max(0, max_nodes_total - total_nodes)
        if remaining_nodes == 0:
            # Used up entire node budget without finding solution
            return None, total_nodes, time.perf_counter() - t0, "cap", None
        
        # Run depth-limited search with current limit L
        status, goal, used, parent = depth_limited_search(
            puz, start, L, t0, max_time_sec, remaining_nodes
        )
        
        # Accumulate nodes expanded in this iteration
        total_nodes += used
        
        # Check status of this DLS iteration
        if status == "ok":
            # Found solution at depth L!
            path = reconstruct(parent, goal)
            return path, total_nodes, time.perf_counter() - t0, "ok", L
        
        if status in ("timeout", "cap"):
            # Hit resource limit - give up
            return None, total_nodes, time.perf_counter() - t0, status, L
        
        # Status is "cutoff" or "failure"
        # Cutoff: Solution might exist deeper, try next depth
        # Failure: No solution at this depth, but might exist deeper
        # Either way, continue to next depth limit
    
    # Reached max depth without finding solution
    return None, total_nodes, time.perf_counter() - t0, "cutoff", max_depth_cap

def print_board(s: Position, m: int):
    """Display puzzle state as a human-readable m x m grid"""
    for i in range(0, m*m, m):
        row = s[i:i+m]
        print(" ".join(str(x) for x in row))

def run_once(m: int, steps: int):
    """
    Run IDS once on a random puzzle instance
    
    Args:
        m: Board dimension (3 for 8-puzzle, 4 for 15-puzzle)
        steps: Number of random moves to scramble the puzzle
        
    Returns:
        Tuple of (solution_length, nodes_expanded, time, status)
    """
    puz = NPuzzle(m=m)
    start = puz.random_start(steps=steps)
    path, expanded, dt, status, L = iterative_deepening_search(
        puz, start, MAX_DEPTH_CAP, MAX_NODES_TOTAL, MAX_TIME_SEC
    )
    
    # Display results
    print(f"\nIDS on {m}x{m}")
    print("start:")
    print_board(start, m)
    print("moves:", (len(path) - 1) if path else None)
    print("expanded:", expanded)
    print("time_s:", round(dt, 4))
    print("status:", status, "limit:", L)
    
    return (len(path) - 1) if path else None, expanded, dt, status

def run_trials_auto():
    """
    Automated test harness: run IDS on both 8-puzzle and 15-puzzle
    
    For each puzzle size, attempt to get TRIALS_REQUIRED successful solutions
    Retry on failure up to MAX_ATTEMPTS_PER_SIZE times
    
    IDS performance characteristics:
    - 8-puzzle (3x3): Handles moderate scrambles well (30 steps)
    - 15-puzzle (4x4): Limited but better than BFS on memory (8 steps)
      IDS re-expands states but uses much less memory than BFS
      Good choice when memory is limited but need optimal solution
    
    Adaptive retry strategy:
    - If hitting limits repeatedly, reduce scramble depth slightly
    - This helps ensure we get required number of successful runs
    """
    # Configuration: (board_size, scramble_steps)
    configs = [(3, 30), (4, 30)]
    
    for m, steps in configs:
        successes = 0
        attempts = 0
        total_moves = 0
        total_expanded = 0
        total_time = 0.0
        
        print("\n" + "="*36)
        print(f"Target: {TRIALS_REQUIRED} successful trial(s) on {m}x{m} (steps={steps})")
        print("="*36)
        
        # Keep trying until we get enough successful runs
        while successes < TRIALS_REQUIRED and attempts < MAX_ATTEMPTS_PER_SIZE:
            attempts += 1
            print(f"\nattempt {attempts}:")
            moves, expanded, dt, status = run_once(m, steps)
            
            # Count as success if we found a solution
            if status == "ok" and moves is not None:
                successes += 1
                total_moves += moves
                total_expanded += expanded
                total_time += dt
            else:
                # Retry with fresh random start
                print("retrying due to", status)
                # Gentle backoff: if repeatedly hitting limits, reduce scrambling
                if status in ("timeout", "cap") and steps > 2:
                    steps -= 1
                    print(f"  (reducing scramble to {steps} steps)")
        
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