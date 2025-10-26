# n_puzzle_Depth_Limited_DFS.py
# Implements Depth-Limited Search for the n-puzzle problem
# DFS with a maximum depth limit to prevent infinite exploration

from typing import Tuple, List, Dict, Optional, Set
import random, time

Position = Tuple[int, ...]

MAX_NODES = 60000000
MAX_TIME_SEC = 120

TRIALS_REQUIRED = 3
MAX_ATTEMPTS_PER_SIZE = 5

# Depth limits based on puzzle size and scramble steps
# For 15-puzzle with typical scrambles, we'll use 3x the scramble distance
DEPTH_LIMIT_MULTIPLIER = 1

class NPuzzle:
    """
    Represents an n-puzzle problem where n = m^2 - 1
    """
    def __init__(self, m: int = 3):
        assert m > 1, "Board size must be at least 2x2"
        self.m = m
        self.goal: Position = tuple([i for i in range(1, m*m)] + [0])

    def neighbors(self, s: Position) -> List[Position]:
        """Generate all valid successor states"""
        m = self.m
        z = s.index(0)
        r, c = divmod(z, m)

        def swap(i: int, j: int) -> Position:
            a = list(s)
            a[i], a[j] = a[j], a[i]
            return tuple(a)

        out: List[Position] = []
        if r > 0:
            out.append(swap(z, z - m))
        if r < m-1:
            out.append(swap(z, z + m))
        if c > 0:
            out.append(swap(z, z - 1))
        if c < m-1:
            out.append(swap(z, z + 1))
        return out

    def is_goal(self, s: Position) -> bool:
        """Test if state s is the goal state"""
        return s == self.goal

    def random_start(self, steps: int) -> Position:
        """Generate a random solvable starting state"""
        s = self.goal
        for _ in range(steps):
            s = random.choice(self.neighbors(s))
        return s

def reconstruct(parent: Dict[Position, Optional[Position]], s: Position) -> List[Position]:
    """Reconstruct solution path from parent pointers"""
    path: List[Position] = [s]
    while parent[s] is not None:
        s = parent[s]
        path.append(s)
    path.reverse()
    return path

def depth_limited_search(puz: NPuzzle, start: Position, depth_limit: int):
    """
    Depth-Limited DFS: Explore branches only up to a maximum depth
    
    This prevents DFS from wandering down infinitely deep paths.
    If no solution is found within the depth limit, it returns failure.
    
    Args:
        puz: NPuzzle instance
        start: Initial puzzle state
        depth_limit: Maximum depth to explore before cutting off a branch
        
    Returns:
        Tuple of (solution_path, nodes_expanded, time_elapsed, status_string)
    """
    t0 = time.perf_counter()
    
    # Stack now stores (state, depth) tuples to track how deep we are
    stack: List[Tuple[Position, int]] = [(start, 0)]
    
    parent: Dict[Position, Optional[Position]] = {start: None}
    
    # Track depth at which each state was first reached
    depth_reached: Dict[Position, int] = {start: 0}
    
    nodes_expanded = 0

    while stack:
        if MAX_TIME_SEC and (time.perf_counter() - t0) >= MAX_TIME_SEC:
            return None, nodes_expanded, time.perf_counter() - t0, "timeout"
        
        # Pop state and its current depth
        s, current_depth = stack.pop()
        
        # Skip if we've already explored this state at a shallower depth
        if current_depth > depth_reached.get(s, float('inf')):
            continue
        
        # Goal test
        if puz.is_goal(s):
            return reconstruct(parent, s), nodes_expanded, time.perf_counter() - t0, "ok"
        
        nodes_expanded += 1
        
        if nodes_expanded >= MAX_NODES:
            return None, nodes_expanded, time.perf_counter() - t0, "cap"
        
        # Only expand children if we haven't reached depth limit
        if current_depth < depth_limit:
            for t in puz.neighbors(s):
                # Add successor if we haven't seen it, or if we're reaching it at a shallower depth
                if t not in depth_reached or current_depth + 1 < depth_reached[t]:
                    depth_reached[t] = current_depth + 1
                    parent[t] = s
                    stack.append((t, current_depth + 1))
    
    # Exhausted search space within depth limit without finding solution
    return None, nodes_expanded, time.perf_counter() - t0, "depth_limit_reached"

def print_board(s: Position, m: int):
    """Display puzzle state as a grid"""
    for i in range(0, m*m, m):
        row = s[i:i+m]
        print(" ".join(str(x) for x in row))

def run_once(m: int, steps: int):
    """Run depth-limited DFS once"""
    # Set depth limit to 3x the scramble distance
    depth_limit = steps * DEPTH_LIMIT_MULTIPLIER
    
    puz = NPuzzle(m=m)
    start = puz.random_start(steps=steps)
    path, expanded, dt, status = depth_limited_search(puz, start, depth_limit)
    
    print(f"\nDepth-Limited DFS on {m}x{m} (depth_limit={depth_limit})")
    print("start:")
    print_board(start, m)
    print("moves:", (len(path) - 1) if path else None)
    print("expanded:", expanded)
    print("time_s:", round(dt, 4))
    print("status:", status)
    
    return (len(path) - 1) if path else None, expanded, dt, status

def run_trials_auto():
    """
    Automated test harness with depth-limited DFS
    
    Depth limits prevent infinite exploration, making 15-puzzle more tractable.
    However, solutions may not be found if they exceed the depth limit.
    """
    # Using more reasonable scrambles now that we have depth limiting
    configs = [(3, 60), (4, 30)]
    
    for m, steps in configs:
        successes = 0
        attempts = 0
        total_moves = 0
        total_expanded = 0
        total_time = 0.0
        
        depth_limit = steps * DEPTH_LIMIT_MULTIPLIER
        
        print("\n" + "="*50)
        print(f"Target: {TRIALS_REQUIRED} successful trial(s) on {m}x{m}")
        print(f"Scramble steps: {steps}, Depth limit: {depth_limit}")
        print("="*50)
        
        while successes < TRIALS_REQUIRED and attempts < MAX_ATTEMPTS_PER_SIZE:
            attempts += 1
            print(f"\nattempt {attempts}:")
            moves, expanded, dt, status = run_once(m, steps)
            
            if status == "ok" and moves is not None:
                successes += 1
                total_moves += moves
                total_expanded += expanded
                total_time += dt
            else:
                print(f"retrying due to {status}")
        
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
    # random.seed(0)  # Uncomment for reproducible results
    run_trials_auto()