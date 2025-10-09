# n_puzzle_ASTAR.py
# Implements A* Search for the n-puzzle problem with two heuristics
# A* is an informed search algorithm that uses f(n) = g(n) + h(n)
# where g(n) is cost to reach n, h(n) is estimated cost from n to goal

from typing import Tuple, List, Dict, Optional, Set, Callable
from collections import deque
import heapq, random, time

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
        self.goal: Position = tuple([i for i in range(1, m*m)] + [0])
        
        # Pre-compute goal positions for efficient Manhattan distance calculation
        # Maps each tile number to its (row, col) in the goal state
        # Example for 8-puzzle: {1: (0,0), 2: (0,1), ..., 8: (2,2), 0: (2,2)}
        self.goal_pos = {tile: divmod(idx, m) for idx, tile in enumerate(self.goal)}

    def neighbors(self, s: Position) -> List[Position]:
        """
        Generate all valid successor states by sliding tiles into the blank space
        
        Args:
            s: Current puzzle state
            
        Returns:
            List of all reachable states (2-4 neighbors typically)
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

    def random_start(self, steps: int = 40) -> Position:
        """
        Generate a random solvable starting state by random walk from goal
        
        Args:
            steps: Number of random moves to make from goal state (default 40)
            
        Returns:
            A randomized but solvable starting state
        """
        s = self.goal
        for _ in range(steps):
            s = random.choice(self.neighbors(s))
        return s

# ============================================================================
# HEURISTIC FUNCTIONS
# ============================================================================
# A* requires a heuristic h(n) that estimates cost from state n to goal
# Good heuristics should be:
# 1. ADMISSIBLE: Never overestimate true cost (h(n) ≤ h*(n))
# 2. CONSISTENT: h(n) ≤ cost(n,n') + h(n') for all neighbors n'
# Both properties guarantee A* finds optimal solution

def h_misplaced(puz: NPuzzle, s: Position) -> int:
    """
    Heuristic 1: Count of misplaced tiles (excluding blank)
    
    This counts how many tiles are not in their goal position.
    
    Properties:
    - ADMISSIBLE: Each misplaced tile needs at least 1 move to fix
    - CONSISTENT: Moving a tile can only change misplaced count by ±1
    - Simple and fast to compute
    - Underestimates significantly (weak heuristic)
    
    Example for 8-puzzle:
        Current: [2,1,3,4,5,6,7,8,0]  Goal: [1,2,3,4,5,6,7,8,0]
        Tiles 1 and 2 are misplaced → h = 2
    
    Args:
        puz: NPuzzle instance
        s: Current state
        
    Returns:
        Number of misplaced tiles (excluding blank)
    """
    # Count tiles that are not in their goal position
    # Skip blank (0) since its position doesn't matter for counting
    return sum(1 for i, t in enumerate(s) if t != 0 and t != puz.goal[i])

def h_manhattan(puz: NPuzzle, s: Position) -> int:
    """
    Heuristic 2: Sum of Manhattan distances of all tiles to their goals
    
    Manhattan distance = |x1-x2| + |y1-y2| (taxicab distance)
    This is the minimum number of moves a tile needs if nothing blocks it.
    
    Properties:
    - ADMISSIBLE: Manhattan is minimum moves without considering obstacles
    - CONSISTENT: Moving a tile changes its Manhattan by ±1
    - DOMINATES h_misplaced: h_manhattan ≥ h_misplaced always
    - Much stronger heuristic: expands far fewer nodes
    
    Example for 8-puzzle:
        Tile 7 at position (1,1) should be at (2,0)
        Manhattan = |1-2| + |1-0| = 1 + 1 = 2
    
    Why it dominates misplaced:
    - If a tile is misplaced: h_misplaced contributes 1
    - Same tile: h_manhattan contributes ≥1 (its actual distance)
    - Sum over all tiles: h_manhattan ≥ h_misplaced
    
    Args:
        puz: NPuzzle instance (has pre-computed goal_pos)
        s: Current state
        
    Returns:
        Sum of Manhattan distances for all tiles (excluding blank)
    """
    m = puz.m
    dist = 0
    
    for idx, t in enumerate(s):
        if t == 0:  # Skip blank tile
            continue
        
        # Current position of this tile
        r, c = divmod(idx, m)
        
        # Goal position of this tile (pre-computed)
        rg, cg = puz.goal_pos[t]
        
        # Add Manhattan distance: |row_diff| + |col_diff|
        dist += abs(r - rg) + abs(c - cg)
    
    return dist

def reconstruct(parent: Dict[Position, Optional[Position]], s: Position) -> List[Position]:
    """
    Reconstruct solution path by following parent pointers from goal to start
    
    Args:
        parent: Dictionary mapping states to their parents
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

def a_star(puz: NPuzzle, start: Position, h: Callable[[NPuzzle, Position], int]):
    """
    A* Search: Best-first search using f(n) = g(n) + h(n)
    
    A* Characteristics:
    - Uses priority queue ordered by f(n) = g(n) + h(n)
      - g(n): actual cost from start to n (path length so far)
      - h(n): heuristic estimate from n to goal
    - OPTIMAL: Guaranteed shortest path if h is admissible
    - COMPLETE: Will find solution if one exists
    - Efficiency depends on heuristic quality
    
    Why A* is excellent for n-puzzle:
    - Informed search: uses heuristic to guide toward goal
    - Much more efficient than uninformed search (BFS, DFS, IDS)
    - With Manhattan distance: solves 8-puzzle easily, 15-puzzle possible
    - Balances exploration (g) and exploitation (h)
    
    How it works:
    1. Start with initial state in priority queue
    2. Pop state with lowest f(n) = g(n) + h(n)
    3. If goal: done!
    4. Otherwise: expand neighbors, add to queue with their f values
    5. Track closed set to avoid re-expanding states
    
    Args:
        puz: NPuzzle instance
        start: Initial puzzle state
        h: Heuristic function h(puz, state) → estimated_cost
        
    Returns:
        Tuple of (solution_path, nodes_expanded, time_elapsed)
    """
    t0 = time.perf_counter()  # Start timing
    
    # Track actual cost g(n) from start to each state
    # g[state] = shortest known path length from start to state
    g: Dict[Position, int] = {start: 0}
    
    # Track parent pointers for solution reconstruction
    parent: Dict[Position, Optional[Position]] = {start: None}
    
    # Priority queue: stores (f_value, tiebreaker, state)
    # Python's heapq is a min-heap, so lowest f comes out first
    # Tiebreaker ensures deterministic ordering when f values equal
    pq: List[Tuple[int, int, Position]] = []
    f_start = h(puz, start)  # f(start) = g(start) + h(start) = 0 + h(start)
    heapq.heappush(pq, (f_start, 0, start))
    
    # Closed set: states we've already expanded (explored)
    # Once expanded with optimal g, never need to expand again
    closed: Set[Position] = set()
    
    nodes_expanded = 0  # Count expansions for performance metrics
    tie = 0  # Tiebreaker counter for stable heap ordering

    while pq:
        # Get state with lowest f(n) value
        f, _, s = heapq.heappop(pq)
        
        # Skip if already expanded (can happen due to multiple paths)
        if s in closed:
            continue
        
        # Goal test: found optimal solution!
        if puz.is_goal(s):
            path = reconstruct(parent, s)
            dt = time.perf_counter() - t0
            return path, nodes_expanded, dt
        
        # Mark as expanded
        closed.add(s)
        nodes_expanded += 1
        
        # Current cost to reach this state
        gs = g[s]
        
        # Expand: generate and evaluate all neighbors
        for t in puz.neighbors(s):
            # Skip if already fully explored
            if t in closed:
                continue
            
            # Cost to reach neighbor: current cost + 1 (unit step cost)
            gt = gs + 1
            
            # Only update if this is a better path to t
            # (or first path to t)
            if t not in g or gt < g[t]:
                g[t] = gt  # Update best known cost
                parent[t] = s  # Update parent pointer
                
                # Calculate f(t) = g(t) + h(t)
                ft = gt + h(puz, t)
                
                # Add to priority queue with f value
                tie += 1  # Increment tiebreaker
                heapq.heappush(pq, (ft, tie, t))
    
    # Priority queue exhausted without finding goal
    # This shouldn't happen for solvable puzzles
    dt = time.perf_counter() - t0
    return None, nodes_expanded, dt

def print_board(s: Position, m: int):
    """Display puzzle state as a human-readable m x m grid"""
    for i in range(0, m*m, m):
        row = s[i:i+m]
        print(" ".join(str(x) for x in row))

def run_one_trial(puz: NPuzzle, steps: int):
    """
    Run A* with both heuristics on the SAME random puzzle instance
    This allows fair comparison of heuristic performance
    
    Args:
        puz: NPuzzle instance
        steps: Number of random moves to scramble the puzzle
        
    Returns:
        Tuple of ((moves1, expanded1, time1), (moves2, expanded2, time2))
        First tuple is h_misplaced results, second is h_manhattan results
    """
    # Generate one random start state
    start = puz.random_start(steps=steps)
    
    print("\nstart:")
    print_board(start, puz.m)

    # Run A* with h_misplaced
    path1, exp1, dt1 = a_star(puz, start, h_misplaced)
    d1 = (len(path1) - 1) if path1 else None
    print("\nA* with h_misplaced")
    print("moves:", d1)
    print("expanded:", exp1)
    print("time_s:", round(dt1, 4))

    # Run A* with h_manhattan on SAME start state
    path2, exp2, dt2 = a_star(puz, start, h_manhattan)
    d2 = (len(path2) - 1) if path2 else None
    print("\nA* with h_manhattan")
    print("moves:", d2)
    print("expanded:", exp2)
    print("time_s:", round(dt2, 4))

    return (d1, exp1, dt1), (d2, exp2, dt2)

def run_trials_auto():
    """
    Automated test harness for A* heuristic comparison
    
    Runs 3 trials on 8-puzzle (3x3), each trial uses the SAME initial
    configuration for both heuristics to ensure fair comparison.
    
    Expected results:
    - Both find optimal solution (same move count)
    - h_manhattan expands FAR fewer nodes (dominates h_misplaced)
    - h_manhattan typically faster (fewer expansions outweighs computation)
    
    Why compare on same start:
    - Eliminates random variation in difficulty
    - Direct comparison of heuristic effectiveness
    - Shows how better heuristic reduces search effort
    """
    puz = NPuzzle(m=3)  # 8-puzzle
    trials = 3
    steps = 40  # Moderate scrambling for interesting comparison
    
    # Accumulators for averages
    sum1 = [0.0, 0.0, 0.0]  # [moves, expanded, time] for h_misplaced
    sum2 = [0.0, 0.0, 0.0]  # [moves, expanded, time] for h_manhattan
    
    for i in range(1, trials+1):
        print("\n" + "="*28)
        print(f"trial {i} on 3x3 (steps={steps})")
        print("="*28)
        
        # Run both heuristics on same puzzle instance
        (d1, e1, t1), (d2, e2, t2) = run_one_trial(puz, steps)
        
        # Accumulate results
        sum1[0] += d1; sum1[1] += e1; sum1[2] += t1
        sum2[0] += d2; sum2[1] += e2; sum2[2] += t2

    # Print comparison summary
    print("\n--- summary (averages over same-start trials) ---")
    print("h_misplaced avg_moves:", round(sum1[0]/trials, 2), 
          "avg_expanded:", round(sum1[1]/trials, 2), 
          "avg_time_s:", round(sum1[2]/trials, 4))
    print("h_manhattan avg_moves:", round(sum2[0]/trials, 2), 
          "avg_expanded:", round(sum2[1]/trials, 2), 
          "avg_time_s:", round(sum2[2]/trials, 4))

    # Educational notes about heuristic properties
    print("\nnotes:")
    print("h_misplaced:")
    print("  - Admissible: Never overestimates (each misplaced tile needs ≥1 move)")
    print("  - Consistent: Moving a tile changes misplaced count by at most 1")
    print("  - Weak heuristic: Underestimates significantly, explores many nodes")
    print("\nh_manhattan:")
    print("  - Admissible: Manhattan is minimum moves ignoring obstacles")
    print("  - Consistent: Moving a tile changes its Manhattan distance by ±1")
    print("  - Dominates h_misplaced: h_manhattan(n) ≥ h_misplaced(n) for all n")
    print("  - Strong heuristic: Better estimates → far fewer nodes expanded")
    print("  - Result: Usually finds solution faster despite more complex computation")

if __name__ == "__main__":
    # Uncomment next line for reproducible results during testing/debugging
    # random.seed(0)
    
    # Run A* comparison on 8-puzzle with both heuristics
    run_trials_auto()