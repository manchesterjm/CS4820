# n_puzzle_BDS.py
# Implements Bidirectional Search for the n-puzzle problem
# BDS runs two simultaneous BFS searches: one forward from start, one backward from goal
# Meets in the middle for O(b^(d/2)) efficiency instead of O(b^d)

from collections import deque  # Two BFS frontiers (forward and backward)
from typing import Tuple, List, Dict, Optional, Set
import random, time

# Type alias: Position represents a puzzle state as a flattened tuple
# Example for 8-puzzle: (1, 2, 3, 4, 5, 6, 7, 8, 0) where 0 is the blank tile
Position = Tuple[int, ...]

# Safety limits to prevent excessive computation
MAX_NODES = 1500000  # Cap total expansions per search
MAX_TIME_SEC = 60    # Cap wall time per search (0 to disable)

# Trial configuration for automated testing
TRIALS_REQUIRED = 3
MAX_ATTEMPTS_PER_SIZE = 20

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

def _reconstruct(parent: Dict[Position, Optional[Position]], meet: Position) -> List[Position]:
    """
    Reconstruct path from one endpoint to meeting point by following parent pointers
    
    Args:
        parent: Dictionary mapping states to their parents in search tree
        meet: Meeting point state to reconstruct path to
        
    Returns:
        List of states from endpoint to meeting point
    """
    path = [meet]
    while parent[meet] is not None:
        meet = parent[meet]
        path.append(meet)
    path.reverse()
    return path

def bidirectional_bfs(puz: NPuzzle, start: Position):
    """
    Bidirectional Search: Run two BFS searches simultaneously
    
    BDS Characteristics:
    - Runs forward BFS from start AND backward BFS from goal
    - Searches meet in the middle
    - OPTIMAL: Finds shortest path (like BFS)
    - More efficient: O(b^(d/2)) instead of O(b^d)
    - Why? 2*b^(d/2) << b^d. Example: if b=4, d=10: 2*1024 vs 1,048,576!
    
    How it works:
    1. Maintain two frontiers: one growing from start, one from goal
    2. Expand the smaller frontier at each step (balanced growth)
    3. Stop when frontiers intersect (found shortest path!)
    4. Reconstruct: start→meeting point + meeting point→goal
    
    Why BDS is good for n-puzzle:
    - Much more efficient than regular BFS for larger puzzles
    - Still guarantees optimal solution
    - Can handle deeper 15-puzzle instances than BFS
    - Works because goal is known (required for backward search)
    
    Args:
        puz: NPuzzle instance
        start: Initial puzzle state
        
    Returns:
        Tuple of (path, nodes_expanded, time, status)
    """
    t0 = time.perf_counter()  # Start timing
    
    # Trivial case: already at goal
    if start == puz.goal:
        return [start], 0, 0.0, "ok"

    # Initialize forward search (from start)
    f_front = deque([start])  # Forward frontier (FIFO queue)
    f_seen: Set[Position] = {start}  # States seen by forward search
    f_par: Dict[Position, Optional[Position]] = {start: None}  # Parent pointers for forward
    
    # Initialize backward search (from goal)
    b_front = deque([puz.goal])  # Backward frontier (FIFO queue)
    b_seen: Set[Position] = {puz.goal}  # States seen by backward search
    b_par: Dict[Position, Optional[Position]] = {puz.goal: None}  # Parent pointers for backward
    
    nodes_expanded = 0  # Total nodes expanded across both searches

    # Continue until both frontiers are exhausted or we meet
    while f_front and b_front:
        # Check resource limits
        if MAX_TIME_SEC and (time.perf_counter() - t0) >= MAX_TIME_SEC:
            return None, nodes_expanded, time.perf_counter() - t0, "timeout"
        if nodes_expanded >= MAX_NODES:
            return None, nodes_expanded, time.perf_counter() - t0, "cap"

        # Optimization: expand the smaller frontier first
        # This keeps the searches balanced, minimizing total nodes expanded
        # If one frontier is much larger, we're wasting work on that side
        if len(f_front) <= len(b_front):
            # Expand forward search
            meet, exps = _expand_layer(puz, f_front, f_seen, f_par, b_seen)
        else:
            # Expand backward search
            meet, exps = _expand_layer(puz, b_front, b_seen, b_par, f_seen)
        
        # Accumulate node expansions from this layer
        nodes_expanded += exps

        # Check if searches met in the middle
        if meet is not None:
            # Reconstruct complete path: start → meet → goal
            f_path = _reconstruct(f_par, meet)  # start → meeting point
            b_path = _reconstruct(b_par, meet)  # goal → meeting point
            
            # Combine paths: forward + reverse(backward - meeting point)
            b_path = list(reversed(b_path))[1:]  # Reverse and skip duplicate meeting state
            path = f_path + b_path
            
            return path, nodes_expanded, time.perf_counter() - t0, "ok"

    # Both frontiers exhausted without meeting: no solution exists
    # This shouldn't happen for solvable puzzle instances
    return None, nodes_expanded, time.perf_counter() - t0, "failure"

def _expand_layer(puz: NPuzzle, frontier: deque, seen: Set[Position],
                  parent: Dict[Position, Optional[Position]], 
                  other_seen: Set[Position]) -> Tuple[Optional[Position], int]:
    """
    Expand one complete BFS layer from a frontier
    
    This processes all states currently in the frontier (one depth level)
    and generates their successors. Checks if any successor is in the
    other search's seen set (indicating the searches have met).
    
    Args:
        puz: NPuzzle instance
        frontier: Current frontier (FIFO queue) to expand
        seen: Set of states this search has seen
        parent: Parent pointers for this search
        other_seen: Set of states the OTHER search has seen
        
    Returns:
        Tuple of (meeting_state_or_None, nodes_expanded_count)
        - If searches meet: returns the meeting state
        - Otherwise: returns None
        - Always returns count of nodes expanded in this layer
    """
    if not frontier:
        return None, 0
    
    # Process exactly one BFS layer (all states at current depth)
    expansions = 0
    layer_size = len(frontier)  # Snapshot size (new states added go to next layer)
    
    for _ in range(layer_size):
        s = frontier.popleft()  # Get next state from this layer
        expansions += 1  # Count this as an expansion
        
        # Generate all successor states
        for t in puz.neighbors(s):
            # Skip if we've already seen this state in THIS search
            if t in seen:
                continue
            
            # Mark as seen and record parent
            seen.add(t)
            parent[t] = s
            
            # Check if OTHER search has seen this state
            # If so, the searches have met in the middle!
            if t in other_seen:
                return t, expansions
            
            # Add to frontier for next layer
            frontier.append(t)
    
    # Completed layer without meeting
    return None, expansions

def print_board(s: Position, m: int):
    """Display puzzle state as a human-readable m x m grid"""
    for i in range(0, m*m, m):
        row = s[i:i+m]
        print(" ".join(str(x) for x in row))

def run_once(m: int, steps: int):
    """
    Run BDS once on a random puzzle instance
    
    Args:
        m: Board dimension (3 for 8-puzzle, 4 for 15-puzzle)
        steps: Number of random moves to scramble the puzzle
        
    Returns:
        Tuple of (solution_length, nodes_expanded, time, status)
    """
    puz = NPuzzle(m=m)
    start = puz.random_start(steps=steps)
    path, expanded, dt, status = bidirectional_bfs(puz, start)
    
    # Display results
    print(f"\nBDS on {m}x{m}")
    print("start:")
    print_board(start, m)
    print("moves:", (len(path) - 1) if path else None)
    print("expanded:", expanded)
    print("time_s:", round(dt, 4))
    print("status:", status)
    
    return (len(path) - 1) if path else None, expanded, dt, status

def run_trials_auto():
    """
    Automated test harness: run BDS on both 8-puzzle and 15-puzzle
    
    For each puzzle size, attempt to get TRIALS_REQUIRED successful solutions
    Retry on failure up to MAX_ATTEMPTS_PER_SIZE times
    
    BDS performance characteristics:
    - 8-puzzle (3x3): Handles deeper scrambles than regular BFS (30 steps)
    - 15-puzzle (4x4): Better than BFS but still limited (10 steps)
      BDS explores ~2*b^(d/2) nodes vs b^d for BFS
      Big improvement but 15-puzzle solutions still 50+ moves deep
    
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