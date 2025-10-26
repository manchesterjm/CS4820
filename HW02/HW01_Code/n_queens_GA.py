# n_queens_GA.py
# Implements Genetic Algorithm for the n-queens problem
# GA is inspired by biological evolution: selection, crossover, mutation
# Maintains a population of solutions that evolve over generations

from typing import List, Tuple
import random, time, math

# ============================================================================
# STATE REPRESENTATION AND EVALUATION
# ============================================================================
# State encoding: one queen per column; state[c] = row of queen in column c
# Example for 4-queens: [1, 3, 0, 2] means queens at (1,0), (3,1), (0,2), (2,3)
# This encoding guarantees exactly one queen per column

def conflicts(state: List[int]) -> int:
    """
    Count the number of attacking queen pairs
    
    Two queens attack if they share:
    - Same row: state[c1] == state[c2]
    - Same diagonal: |state[c1] - state[c2]| == |c1 - c2|
    
    Lower is better. Goal: conflicts = 0
    
    Args:
        state: List where state[col] = row of queen in that column
        
    Returns:
        Number of attacking pairs (0 = solution)
    """
    n = len(state)
    cnt = 0
    for c1 in range(n):
        r1 = state[c1]
        for c2 in range(c1+1, n):
            r2 = state[c2]
            # Check for row conflict or diagonal conflict
            if r1 == r2 or abs(r1 - r2) == abs(c1 - c2):
                cnt += 1
    return cnt

def fitness(state: List[int]) -> int:
    """
    Calculate fitness: higher values are better
    
    Fitness = max_possible_pairs - conflicts
    For n queens: max pairs = n*(n-1)/2 (all possible queen pairs)
    
    Why this fitness function?
    - Converts minimization (conflicts) to maximization (fitness)
    - Fitness=0 when all queens attack (worst)
    - Fitness=max when no attacks (solution)
    - Proportional to solution quality
    
    Args:
        state: Queen positions
        
    Returns:
        Fitness value (higher = better)
    """
    n = len(state)
    max_pairs = n * (n - 1) // 2  # Total possible pairs
    return max_pairs - conflicts(state)

def random_state(n: int) -> List[int]:
    """
    Generate random state: random row for each column
    
    Args:
        n: Board size
        
    Returns:
        Random state (likely has conflicts)
    """
    return [random.randrange(n) for _ in range(n)]

def init_population(n: int, size: int) -> List[List[int]]:
    """
    Create initial population of random individuals
    
    Args:
        n: Board size
        size: Population size
        
    Returns:
        List of random states
    """
    return [random_state(n) for _ in range(size)]

# ============================================================================
# GENETIC OPERATORS
# ============================================================================

def select_parent(pop: List[List[int]], fit: List[int]) -> List[int]:
    """
    Select parent using fitness-proportionate (roulette wheel) selection
    
    Selection probability ∝ fitness
    - High fitness individuals more likely to be selected
    - But low fitness still has chance (maintains diversity)
    
    How roulette wheel works:
    1. Sum all fitness values → total
    2. Pick random number r in [0, total]
    3. Walk through population accumulating fitness
    4. When accumulator ≥ r, select that individual
    
    Why fitness-proportionate?
    - Balances exploitation (favor good solutions) and exploration (keep diversity)
    - Better than pure elitism (too greedy) or random (no guidance)
    
    Args:
        pop: Population of individuals
        fit: Fitness values (parallel to pop)
        
    Returns:
        Selected parent individual
    """
    total = sum(fit)
    
    # Edge case: all fitness values are 0
    if total == 0:
        return random.choice(pop)
    
    # Pick random point on "roulette wheel"
    r = random.uniform(0, total)
    
    # Find which individual this point lands on
    acc = 0.0
    for ind, f in zip(pop, fit):
        acc += f
        if acc >= r:
            return ind
    
    # Floating point edge case: return last individual
    return pop[-1]

def crossover(p1: List[int], p2: List[int], cx_rate: float) -> Tuple[List[int], List[int]]:
    """
    Perform crossover (recombination) between two parents
    
    Crossover combines genetic material from two parents to create offspring.
    Uses uniform crossover: for each column, randomly pick from p1 or p2.
    
    Why uniform crossover?
    - For n-queens, each column is independent
    - Mixing columns from both parents creates new combinations
    - More thorough mixing than single-point crossover
    
    Crossover probability:
    - With probability cx_rate: perform crossover
    - Otherwise: return copies of parents (no mixing)
    - Typical values: 0.6-0.9 (most offspring are crossed)
    
    Args:
        p1: First parent
        p2: Second parent
        cx_rate: Crossover probability
        
    Returns:
        Tuple of two offspring (c1, c2)
    """
    n = len(p1)
    
    # Decide whether to perform crossover
    if random.random() > cx_rate:
        # No crossover: return clones of parents
        return p1[:], p2[:]
    
    # Uniform crossover: swap each column with 50% probability
    c1 = p1[:]
    c2 = p2[:]
    for i in range(n):
        if random.random() < 0.5:
            # Swap columns between offspring
            c1[i], c2[i] = p2[i], p1[i]
    
    return c1, c2

def mutate(ind: List[int], mut_rate: float) -> None:
    """
    Perform mutation on an individual (in-place)
    
    Mutation introduces random changes to maintain diversity.
    For each column, with probability mut_rate, move queen to random row.
    
    Why mutation?
    - Prevents premature convergence to local optimum
    - Introduces genetic material not present in initial population
    - Acts as "insurance" against losing good genes
    
    Mutation rate:
    - Too high: becomes random search (evolution doesn't work)
    - Too low: population becomes homogeneous (stuck in local optimum)
    - Typical: 0.01-0.1 (1-10% of genes mutated)
    - For n-queens: ~0.05 works well (mutate a few columns per individual)
    
    Args:
        ind: Individual to mutate (modified in-place)
        mut_rate: Per-gene mutation probability
    """
    n = len(ind)
    for i in range(n):
        # Mutate this column with probability mut_rate
        if random.random() < mut_rate:
            # Move queen to a different random row
            r = random.randrange(n)
            while r == ind[i]:  # Ensure we actually change it
                r = random.randrange(n)
            ind[i] = r

def is_solution(state: List[int]) -> bool:
    """
    Check if state is a solution (no conflicts)
    
    Args:
        state: Queen positions
        
    Returns:
        True if no queens attack each other
    """
    return conflicts(state) == 0

# ============================================================================
# GENETIC ALGORITHM MAIN LOOP
# ============================================================================

def ga_run(n: int, pop_size: int, cx_rate: float, mut_rate: float, 
           max_gens: int, elite_keep: int = 2):
    """
    Run Genetic Algorithm for n-queens
    
    GA Process:
    1. Initialize random population
    2. Loop for max_gens generations:
       a. Evaluate fitness of all individuals
       b. Check if solution found
       c. Select parents (fitness-proportionate)
       d. Create offspring (crossover + mutation)
       e. Replace old population with new (keep elite)
    
    GA Components:
    - Population: Set of candidate solutions
    - Selection: Choose parents based on fitness
    - Crossover: Combine parents to create offspring
    - Mutation: Random changes for diversity
    - Elitism: Keep best individuals across generations
    
    Why GA works for n-queens:
    - Large search space (n^n possible states)
    - Crossover combines good partial solutions
    - Population maintains diversity
    - Fitness guides search toward solution
    
    Parameters explained:
    - pop_size: Larger = more diversity but slower
    - cx_rate: Probability of crossover (0.7-0.9 typical)
    - mut_rate: Probability of mutation per gene (0.01-0.1 typical)
    - max_gens: Maximum generations before giving up
    - elite_keep: Number of best individuals to preserve
    
    Args:
        n: Board size (n x n board, n queens)
        pop_size: Population size (number of individuals)
        cx_rate: Crossover rate (probability of recombination)
        mut_rate: Mutation rate (probability per gene)
        max_gens: Maximum number of generations
        elite_keep: Number of elite individuals to preserve
        
    Returns:
        Tuple of (best_solution, generations, evaluations, time, success)
    """
    # Initialize population with random individuals
    pop = init_population(n, pop_size)
    
    t0 = time.perf_counter()  # Start timing
    generations = 0
    evaluations = 0  # Count fitness evaluations (for performance metrics)
    
    # Main evolutionary loop
    while generations < max_gens:
        # Evaluate fitness of entire population
        fit = [fitness(x) for x in pop]
        evaluations += len(pop)
        
        # Find best individual in current generation
        best_idx = max(range(len(pop)), key=lambda i: fit[i])
        best = pop[best_idx]
        
        # Check if we found a solution
        if is_solution(best):
            dt = time.perf_counter() - t0
            return best, generations, evaluations, dt, True
        
        # Create new population for next generation
        new_pop: List[List[int]] = []
        
        # Elitism: preserve the best individuals unchanged
        # This ensures we never lose the best solution found so far
        for _ in range(elite_keep):
            new_pop.append(best[:])  # Add copies of best
        
        # Fill rest of population with offspring from selection + crossover + mutation
        while len(new_pop) < pop_size:
            # Select two parents (fitness-proportionate)
            p1 = select_parent(pop, fit)
            p2 = select_parent(pop, fit)
            
            # Create two offspring via crossover
            c1, c2 = crossover(p1, p2, cx_rate)
            
            # Mutate first offspring and add to new population
            mutate(c1, mut_rate)
            if len(new_pop) < pop_size:
                new_pop.append(c1)
            
            # Mutate second offspring and add to new population
            mutate(c2, mut_rate)
            if len(new_pop) < pop_size:
                new_pop.append(c2)
        
        # Replace old population with new population
        pop = new_pop
        generations += 1
    
    # Reached max generations without finding solution
    dt = time.perf_counter() - t0
    
    # Final evaluation to return best individual found
    fit = [fitness(x) for x in pop]
    evaluations += len(pop)
    best_idx = max(range(len(pop)), key=lambda i: fit[i])
    best = pop[best_idx]
    
    return best, generations, evaluations, dt, is_solution(best)

def print_board(state: List[int]) -> None:
    """
    Display n-queens board as a grid
    Q = queen, . = empty square
    """
    n = len(state)
    for r in range(n):
        row = []
        for c in range(n):
            row.append('Q' if state[c] == r else '.')
        print(" ".join(row))

# ============================================================================
# TRIAL HARNESS AND PARAMETER TUNING
# ============================================================================

def run_trials(n: int, cfg: Tuple[int, float, float, int], 
               trials_required: int = 3, max_attempts: int = 30):
    """
    Run GA trials with specific configuration until we get required successes
    
    Args:
        n: Board size
        cfg: Configuration tuple (pop_size, cx_rate, mut_rate, max_gens)
        trials_required: Number of successful runs needed
        max_attempts: Maximum attempts before giving up
        
    Returns:
        Tuple of (success_bool, (avg_gens, avg_evals, avg_time))
    """
    pop_size, cx_rate, mut_rate, max_gens = cfg
    
    successes = 0
    attempts = 0
    total_gens = 0
    total_evals = 0
    total_time = 0.0
    
    # Keep trying until we get enough successful runs
    while successes < trials_required and attempts < max_attempts:
        attempts += 1
        sol, gens, evals, dt, ok = ga_run(n, pop_size, cx_rate, mut_rate, max_gens)
        
        print(f"\nGA on n={n} attempt {attempts}")
        print_board(sol)
        print("conflicts:", conflicts(sol))
        print("gens:", gens)
        print("evals:", evals)
        print("time_s:", round(dt, 4))
        print("status:", "ok" if ok else "incomplete")
        
        # Count as success if we found a solution
        if ok:
            successes += 1
            total_gens += gens
            total_evals += evals
            total_time += dt
        else:
            # Adaptive strategy: reduce mutation rate slightly
            # Less mutation = more exploitation of good solutions
            mut_rate = max(0.01, mut_rate * 0.9)
    
    # Print summary statistics
    print("\n--- summary ---")
    print("config:", {"pop": pop_size, "cx": cx_rate, "mut": mut_rate, "gens": max_gens})
    print("attempts:", attempts)
    print("successful:", f"{successes}/{trials_required}")
    
    if successes > 0:
        avg_gens = total_gens / successes
        avg_evals = total_evals / successes
        avg_time = total_time / successes
        print("avg_gens:", round(avg_gens, 2))
        print("avg_evals:", round(avg_evals, 2))
        print("avg_time_s:", round(avg_time, 4))
        return True, (avg_gens, avg_evals, avg_time)
    else:
        print("avg_gens:", None)
        print("avg_evals:", None)
        print("avg_time_s:", None)
        return False, (math.inf, math.inf, math.inf)

def main():
    """
    Main experiment: Test 3 GA configurations on 4-queens, pick best, use for 8-queens
    
    Configuration parameters to tune:
    1. Population size: Larger = more diversity but slower per generation
    2. Crossover rate: Higher = more mixing of solutions
    3. Mutation rate: Higher = more exploration vs exploitation
    4. Max generations: More = better chance but longer runtime
    
    Strategy:
    - Test 3 different configurations on 4-queens
    - Pick configuration with best performance (lowest time, then fewest generations)
    - Use that configuration on 8-queens
    
    Configurations tested:
    A. Small population, high crossover, moderate mutation
    B. Medium population, high crossover, low mutation
    C. Medium population, lower crossover, higher mutation
    """
    # Three configurations to test: (pop_size, cx_rate, mut_rate, max_gens)
    configs = [
        (40, 0.9, 0.10, 200),  # Config A: Small pop, high cx, moderate mut
        (80, 0.9, 0.05, 300),  # Config B: Medium pop, high cx, low mut
        (60, 0.8, 0.15, 300),  # Config C: Medium pop, lower cx, higher mut
    ]
    
    print("="*40)
    print("PHASE 1: Testing 3 configurations on n=4")
    print("="*40)
    
    cfg_results = []
    
    # Test each configuration on 4-queens
    for i, cfg in enumerate(configs, 1):
        print(f"\n{'='*40}")
        print(f"Configuration {chr(64+i)}: pop={cfg[0]}, cx={cfg[1]}, mut={cfg[2]}, gens={cfg[3]}")
        print(f"{'='*40}")
        
        ok, metrics = run_trials(4, cfg, trials_required=3, max_attempts=30)
        cfg_results.append((cfg, metrics))
    
    # Select best configuration based on performance
    # Primary: lowest average time
    # Secondary: lowest average generations (as tiebreaker)
    def score(metrics):
        avg_gens, avg_evals, avg_time = metrics
        return (avg_time, avg_gens)  # Tuple comparison: time first, then gens
    
    best_cfg, best_metrics = min(cfg_results, key=lambda x: score(x[1]))
    
    print("\n" + "="*40)
    print("BEST CONFIGURATION SELECTED")
    print("="*40)
    print(f"Config: pop={best_cfg[0]}, cx={best_cfg[1]}, mut={best_cfg[2]}, gens={best_cfg[3]}")
    print(f"Performance on n=4:")
    print(f"  avg_gens: {round(best_metrics[0], 2)}")
    print(f"  avg_evals: {round(best_metrics[1], 2)}")
    print(f"  avg_time_s: {round(best_metrics[2], 4)}")
    
    # Run best configuration on 8-queens
    print("\n" + "="*40)
    print("PHASE 2: Running best config on n=8")
    print("="*40)
    run_trials(8, best_cfg, trials_required=3, max_attempts=30)
    
    print("\n" + "="*40)
    print("EXPERIMENT COMPLETE")
    print("="*40)
    print("\nKey insights:")
    print("- GA performance depends heavily on parameter tuning")
    print("- Population size balances diversity vs computational cost")
    print("- Crossover rate controls exploitation of good solutions")
    print("- Mutation rate controls exploration of new solutions")
    print("- Elitism ensures we never lose best solutions found")
    print("- Fitness-proportionate selection balances pressure and diversity")

if __name__ == "__main__":
    # Uncomment next line for reproducible results during testing/debugging
    # random.seed(0)
    
    # Run main experiment: test configs on n=4, use best for n=8
    main()