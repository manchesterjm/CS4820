# n_queens_SA.py
# Implements Simulated Annealing for the n-queens problem
# SA is a probabilistic local search that accepts worse moves to escape local optima
# Inspired by metallurgical annealing process (heating and slow cooling)

from typing import List, Tuple
import random, math, time

# ============================================================================
# STATE REPRESENTATION AND EVALUATION
# ============================================================================
# State encoding: one queen per column; state[c] = row of queen in column c
# Example for 4-queens: [1, 3, 0, 2] means:
#   Column 0: queen at row 1
#   Column 1: queen at row 3
#   Column 2: queen at row 0
#   Column 3: queen at row 2
# This encoding guarantees exactly one queen per column (simplifies problem)

def conflicts(state: List[int]) -> int:
    """
    Count the number of attacking queen pairs
    
    Two queens attack each other if they share:
    - Same row: state[c1] == state[c2]
    - Same diagonal: |state[c1] - state[c2]| == |c1 - c2|
    
    This is our objective function to minimize.
    Goal state has conflicts = 0 (no attacks).
    
    Args:
        state: List where state[col] = row of queen in that column
        
    Returns:
        Number of attacking pairs (0 = solution)
    """
    n = len(state)
    cnt = 0
    
    # Check all pairs of queens
    for c1 in range(n):
        r1 = state[c1]  # Row of queen in column c1
        
        for c2 in range(c1+1, n):  # Only check pairs once (c1 < c2)
            r2 = state[c2]  # Row of queen in column c2
            
            # Check if queens attack each other
            if r1 == r2:  # Same row
                cnt += 1
            elif abs(r1 - r2) == abs(c1 - c2):  # Same diagonal
                cnt += 1
    
    return cnt

def random_neighbor(state: List[int]) -> List[int]:
    """
    Generate a random neighbor by moving one queen within its column
    
    Local move: Pick a random column, move its queen to a different row.
    This maintains the "one queen per column" constraint.
    
    Args:
        state: Current state
        
    Returns:
        New state with one queen moved to a different row
    """
    n = len(state)
    c = random.randrange(n)  # Pick random column to modify
    r_new = random.randrange(n)  # Pick random new row
    
    # Ensure we actually move the queen (different row)
    while r_new == state[c]:
        r_new = random.randrange(n)
    
    # Create new state with modified queen position
    nxt = state[:]
    nxt[c] = r_new
    return nxt

def simulated_annealing(n: int,
                        T0: float = 1.0,
                        cooling: float = 0.995,
                        max_iters: int = 200000):
    """
    Simulated Annealing: Probabilistic local search with temperature schedule
    
    SA Characteristics:
    - Starts with high temperature: accepts many worse moves (exploration)
    - Gradually cools: becomes more selective (exploitation)
    - Temperature controls acceptance probability: P(accept) = e^(ΔE/T)
    - Complete: Will eventually converge (though not guaranteed optimal)
    - Good for hard optimization problems with many local optima
    
    Why SA works for n-queens:
    - Many local optima in the search space
    - Pure hill climbing gets stuck
    - Random restarts help but wasteful
    - SA escapes local optima probabilistically
    - Annealing schedule balances exploration vs exploitation
    
    Temperature schedule:
    - T starts at T0 (initial temperature)
    - T *= cooling each iteration (geometric cooling)
    - Higher T → accept worse moves more often
    - Lower T → behave more like hill climbing
    - As T → 0: becomes greedy (only accept improvements)
    
    Acceptance rule:
    - If ΔE > 0 (improvement): always accept
    - If ΔE < 0 (worse): accept with probability e^(ΔE/T)
    - Example: ΔE=-1, T=1.0 → P=0.368 (36.8% chance)
    - Example: ΔE=-1, T=0.1 → P=0.000045 (rare)
    
    Args:
        n: Board size (n x n board, n queens)
        T0: Initial temperature (higher = more exploration)
        cooling: Cooling rate (multiply T by this each iteration)
                 Closer to 1.0 = slower cooling = more thorough search
        max_iters: Maximum iterations before giving up
        
    Returns:
        Tuple of (solution_state, expansions, time, success_bool)
    """
    # Initialize with random state: random row for each column
    current = [random.randrange(n) for _ in range(n)]
    
    # Evaluate initial state
    # Use negative conflicts as "value" (maximize value = minimize conflicts)
    cur_h = -conflicts(current)
    
    # Initialize temperature
    T = T0
    
    expanded = 0  # Count neighbor evaluations (for performance metrics)
    t0 = time.perf_counter()  # Start timing
    
    # Main SA loop
    for it in range(1, max_iters+1):
        # Check if temperature has cooled to near-zero (stopping criterion)
        if T <= 1e-12:
            break
        
        # Generate random neighbor (local move)
        nxt = random_neighbor(current)
        nxt_h = -conflicts(nxt)
        
        # Calculate change in value: ΔE = value(next) - value(current)
        # Positive ΔE means improvement (fewer conflicts)
        dE = nxt_h - cur_h
        
        # Acceptance decision
        if dE > 0:
            # Improvement: always accept (move to better state)
            accept = True
        else:
            # Worse move: accept probabilistically based on temperature
            # P(accept) = e^(ΔE/T)
            # - Large negative ΔE (much worse) → low probability
            # - High T → higher probability (more exploration)
            # - Low T → lower probability (more exploitation)
            accept = random.random() < math.exp(dE / T)
        
        # Apply acceptance decision
        if accept:
            current = nxt
            cur_h = nxt_h
        
        # Check if we found a solution (no conflicts)
        if -cur_h == 0:
            dt = time.perf_counter() - t0
            return current, expanded, dt, True
        
        # Cool the temperature (geometric schedule)
        T *= cooling
        expanded += 1
    
    # Max iterations reached without finding solution
    dt = time.perf_counter() - t0
    return current, expanded, dt, (-cur_h == 0)

def print_board(state: List[int]) -> None:
    """
    Display n-queens board as a grid
    Q = queen, . = empty square
    
    Args:
        state: Queen positions (state[col] = row)
    """
    n = len(state)
    for r in range(n):
        row = []
        for c in range(n):
            # Check if there's a queen at this position
            row.append('Q' if state[c] == r else '.')
        print(" ".join(row))

# ============================================================================
# TRIAL HARNESS
# ============================================================================
# Trial configuration
TRIALS_REQUIRED = 3   # Need 3 successful solutions per problem size
MAX_ATTEMPTS_PER_SIZE = 30  # Maximum attempts before giving up

def run_once(n: int, T0: float, cooling: float, max_iters: int):
    """
    Run SA once on n-queens problem
    
    Args:
        n: Board size
        T0: Initial temperature
        cooling: Cooling rate
        max_iters: Maximum iterations
        
    Returns:
        Tuple of (success, expansions, time, conflicts)
    """
    sol, expanded, dt, ok = simulated_annealing(n, T0, cooling, max_iters)
    
    print(f"\nSA on n={n}")
    print("board:")
    print_board(sol)
    print("conflicts:", conflicts(sol))
    print("expanded:", expanded)
    print("time_s:", round(dt, 4))
    print("status:", "ok" if ok else "incomplete")
    
    return ok, expanded, dt, conflicts(sol)

def run_trials_auto():
    """
    Automated test harness: run SA on n=4 and n=8
    
    For each problem size, attempt to get TRIALS_REQUIRED successful solutions
    Retry on failure up to MAX_ATTEMPTS_PER_SIZE times
    
    SA parameter tuning:
    - n=4 (small): Lower temperature, faster cooling (converges quickly)
    - n=8 (larger): Higher temperature, slower cooling (needs more exploration)
    
    Temperature guidelines:
    - T0 ≈ average ΔE for typical bad move (problem-dependent)
    - Cooling rate: 0.95-0.999 (slower = more thorough but more time)
    - For n-queens: conflicts change by small amounts, so T0=1-2 works well
    
    Adaptive retry strategy:
    - If repeatedly failing: slow down cooling (more exploration)
    - Helps ensure we achieve required number of successes
    """
    # Configuration: (n, T0, cooling, max_iters)
    configs = [
        (4, 1.0, 0.995, 50000),    # 4-queens: quick convergence
        (8, 2.0, 0.998, 200000),   # 8-queens: needs more exploration
    ]
    
    for n, T0, cooling, max_iters in configs:
        successes = 0
        attempts = 0
        total_expanded = 0
        total_time = 0.0
        
        print("\n" + "="*36)
        print(f"Target: {TRIALS_REQUIRED} successful trial(s) on n={n}")
        print(f"Parameters: T0={T0}, cooling={cooling}, max_iters={max_iters}")
        print("="*36)
        
        # Keep trying until we get enough successful runs
        while successes < TRIALS_REQUIRED and attempts < MAX_ATTEMPTS_PER_SIZE:
            attempts += 1
            print(f"\nattempt {attempts}:")
            ok, expanded, dt, conf = run_once(n, T0, cooling, max_iters)
            
            # Count as success if we found a solution (0 conflicts)
            if ok:
                successes += 1
                total_expanded += expanded
                total_time += dt
            else:
                # Adaptive strategy: slow down cooling to improve success rate
                # Slower cooling = more iterations at each temperature
                # = more thorough exploration
                cooling = min(0.9995, cooling + 0.0005)
                print(f"  (adjusting cooling to {cooling} for better exploration)")
        
        # Print summary statistics
        print("\n--- summary ---")
        print("attempts:", attempts)
        print("successful:", f"{successes}/{TRIALS_REQUIRED}")
        if successes > 0:
            print("avg_expanded:", round(total_expanded/successes, 2))
            print("avg_time_s:", round(total_time/successes, 4))
        else:
            print("avg_expanded:", None)
            print("avg_time_s:", None)

if __name__ == "__main__":
    # Uncomment next line for reproducible results during testing/debugging
    # random.seed(0)
    
    # Run automated trials on both 4-queens and 8-queens
    run_trials_auto()