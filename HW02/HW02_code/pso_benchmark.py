"""
Particle Swarm Optimization (PSO) for benchmark function optimization.

This module implements PSO, a population-based metaheuristic inspired by the
social behavior of bird flocking or fish schooling. Each particle in the swarm
has a position (candidate solution) and a velocity. Particles move through the
search space influenced by three factors:
- Their own best position found so far (cognitive component)
- The global best position found by the entire swarm (social component)
- Inertia from their current velocity (exploration vs exploitation)

Benchmark functions included:
1. Rastrigin: f(x) = 10n + Σ[xi² - 10cos(2πxi)]
   - Highly multimodal with approximately 10^n local minima
   - Global minimum: f(0,...,0) = 0
   - Tests the algorithm's ability to escape local minima

2. Rosenbrock: f(x) = Σ[100(xi+1 - xi²)² + (xi - 1)²]
   - Has a narrow parabolic valley leading to the global minimum
   - Global minimum: f(1,...,1) = 0
   - Tests the algorithm's ability to navigate a narrow valley

Algorithm References:
According to Lecture 7: Search Optimization Part III (PSO slides),
Kennedy & Eberhart, "Particle Swarm Optimization," 1995,
and benchmark functions from Jamil & Yang, "A Literature Survey of
Benchmark Functions"
"""

from typing import Callable, Tuple, List, Dict
import time
import numpy as np

# Safety limit to prevent excessive computation
# According to the assignment requirements, we need a 5 minute timeout
# This prevents PSO from running forever on hard problems
MAX_TIME_SEC = 300


class PSO:
    """
    Particle Swarm Optimization for continuous function minimization

    How does the PSO algorithm work?
    1. Initialize the swarm with random positions and velocities
    2. Evaluate the fitness of each particle (how good is this solution?)
    3. Update each particle's personal best and the global best
    4. Update velocities based on inertia, cognitive, and social
       components
    5. Update positions based on the new velocities
    6. Repeat until convergence or we reach the maximum iterations

    What are the key equations?
    Velocity update equation:
    v[i] = w*v[i] + c1*r1*(pbest[i] - x[i]) + c2*r2*(gbest - x[i])

    where:
    - w: inertia weight (controls exploration vs exploitation tradeoff)
    - c1: cognitive coefficient (how much the particle trusts its own
      experience)
    - c2: social coefficient (how much the particle trusts the swarm's
      experience)
    - r1, r2: random values in [0,1] (adds stochasticity to the search)

    Position update equation:
    x[i] = x[i] + v[i]

    This means the particle moves by adding its velocity to its current
    position.
    """

    def __init__(self,
                 objective_func: Callable[[np.ndarray], float],
                 dimensions: int,
                 bounds: Tuple[float, float],
                 swarm_size: int = 30,
                 w: float = 0.7,
                 c1: float = 1.5,
                 c2: float = 1.5,
                 max_iterations: int = 1000,
                 tolerance: float = 1e-6):
        """
        Initialize PSO optimizer

        Args:
            objective_func: Function to minimize (takes ndarray, returns float)
            dimensions: Number of dimensions in search space
            bounds: (min, max) bounds for each dimension
            swarm_size: Number of particles in swarm
            w: Inertia weight (0.4-0.9 typical)
            c1: Cognitive coefficient (1.5-2.0 typical)
            c2: Social coefficient (1.5-2.0 typical)
            max_iterations: Maximum iterations
            tolerance: Stop if improvement < tolerance
        """
        self.objective_func = objective_func
        self.dimensions = dimensions
        self.bounds = bounds
        self.swarm_size = swarm_size
        self.w = w  # Inertia weight
        self.c1 = c1  # Cognitive coefficient
        self.c2 = c2  # Social coefficient
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        # Initialize swarm
        self.positions = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = float('inf')

        # Track convergence history
        self.convergence_history = []

    def initialize_swarm(self):
        """
        Initialize particle positions and velocities randomly

        Positions: uniformly distributed within bounds
        Velocities: small random values (usually fraction of position
        range)

        This provides good coverage of search space initially
        """
        min_bound, max_bound = self.bounds

        # Initialize positions uniformly in search space
        self.positions = np.random.uniform(
            min_bound,
            max_bound,
            (self.swarm_size, self.dimensions)
        )

        # Initialize velocities to small random values
        # Velocity range typically 10-20% of position range
        velocity_range = (max_bound - min_bound) * 0.1
        self.velocities = np.random.uniform(
            -velocity_range,
            velocity_range,
            (self.swarm_size, self.dimensions)
        )

        # Initialize personal bests to current positions
        self.personal_best_positions = self.positions.copy()
        self.personal_best_scores = np.array([
            self.objective_func(pos) for pos in self.positions
        ])

        # Find initial global best
        best_idx = np.argmin(self.personal_best_scores)
        self.global_best_position = self.personal_best_positions[best_idx].copy()
        self.global_best_score = self.personal_best_scores[best_idx]

    def update_velocities(self):
        """
        Update particle velocities using PSO velocity equation

        Velocity update has three components:
        1. Inertia: w * v[i]
           - Maintains current direction
           - High w: more exploration
           - Low w: more exploitation

        2. Cognitive: c1 * r1 * (pbest[i] - x[i])
           - Attraction to particle's own best position
           - Encourages exploitation of good regions

        3. Social: c2 * r2 * (gbest - x[i])
           - Attraction to swarm's best position
           - Encourages convergence to global optimum

        Random components r1, r2 add stochasticity for exploration
        """
        # Generate random matrices for cognitive and social components
        # r1, r2 are uniform random values in [0,1]
        # Shape: (swarm_size x dimensions) so each particle-dimension gets unique random value
        # This adds stochasticity to prevent premature convergence
        r1 = np.random.random((self.swarm_size, self.dimensions))
        r2 = np.random.random((self.swarm_size, self.dimensions))

        # Inertia component: w * v[i]
        # Maintains momentum in current direction
        # Higher w (e.g., 0.9) = more exploration (particle keeps going in same direction)
        # Lower w (e.g., 0.4) = more exploitation (particle slows down, focuses on best positions)
        inertia = self.w * self.velocities

        # Cognitive component: c1 * r1 * (pbest[i] - x[i])
        # Pulls particle toward its own personal best position
        # (pbest[i] - x[i]) is the direction vector pointing from current position to personal best
        # Multiplying by random r1 adds stochasticity (sometimes pull strongly, sometimes weakly)
        # c1 controls overall strength of this attraction (typical value: 1.5-2.0)
        cognitive = (self.c1 * r1 *
                     (self.personal_best_positions - self.positions))

        # Social component: c2 * r2 * (gbest - x[i])
        # Pulls particle toward the global best position found by entire swarm
        # (gbest - x[i]) is direction vector from current position to global best
        # Broadcasting: gbest is 1D array, gets broadcast to match positions shape
        # c2 controls strength of social influence (typical value: 1.5-2.0)
        # Higher c2 = particles converge faster to swarm's best solution
        social = (self.c2 * r2 *
                  (self.global_best_position - self.positions))

        # Combine all three components to get new velocity
        # New velocity = where particle was going + where it wants to go personally + where swarm is going
        # This creates a balance between exploration and exploitation
        self.velocities = inertia + cognitive + social

        # Velocity clamping to prevent particles from moving too fast
        # Without clamping, velocities can explode and particles fly out of search space
        # Limit velocity to 20% of search space range (common heuristic)
        # This prevents overshooting while still allowing reasonable movement
        min_bound, max_bound = self.bounds
        v_max = (max_bound - min_bound) * 0.2  # Maximum velocity magnitude
        self.velocities = np.clip(self.velocities, -v_max, v_max)  # Clamp to [-v_max, +v_max]

    def update_positions(self):
        """
        Update particle positions based on velocities

        Position update: x[i] = x[i] + v[i]

        Boundary handling:
        - If particle moves outside bounds, clamp to boundary
        - Reset velocity component that caused violation
        """
        # Update positions: x_new = x_old + velocity
        # Each particle moves in the direction and magnitude specified by its velocity
        # This is simple Eulerian integration: position += velocity * dt (where dt=1)
        self.positions = self.positions + self.velocities

        # Boundary handling: clamp positions and reset velocities
        # Without boundary handling, particles can fly outside the search space
        # and never come back, wasting computational resources
        min_bound, max_bound = self.bounds

        # Find violations: create boolean masks for out-of-bounds positions
        # below_min[i,j] = True if particle i's dimension j went below minimum
        # above_max[i,j] = True if particle i's dimension j went above maximum
        below_min = self.positions < min_bound
        above_max = self.positions > max_bound

        # Clamp positions to valid search space
        # Any position < min_bound gets set to min_bound
        # Any position > max_bound gets set to max_bound
        # This prevents particles from leaving the search space
        self.positions = np.clip(self.positions, min_bound, max_bound)

        # Reset velocity components that hit boundaries (bounce back with damping)
        # When a particle hits a boundary, reverse its velocity in that dimension
        # Multiply by -0.5 instead of -1.0 to add damping (particle loses energy on bounce)
        # This prevents particles from bouncing back and forth endlessly at boundaries
        self.velocities[below_min] *= -0.5  # Reverse and dampen velocity for positions that went too low
        self.velocities[above_max] *= -0.5  # Reverse and dampen velocity for positions that went too high

    def evaluate_and_update_bests(self):
        """
        Evaluate fitness and update personal and global bests

        For each particle:
        - Evaluate objective function at current position
        - Update personal best if improved
        - Update global best if any particle improved it
        """
        # Iterate through all particles in the swarm
        for i in range(self.swarm_size):
            # Evaluate fitness: call objective function on current position
            # Lower fitness = better solution (we're minimizing)
            # For Rastrigin/Rosenbrock, this computes the function value at this point
            fitness = self.objective_func(self.positions[i])

            # Update personal best if this particle found a better position
            # Personal best = best position this individual particle has ever seen
            # This is the "cognitive" memory component of PSO
            if fitness < self.personal_best_scores[i]:
                self.personal_best_scores[i] = fitness  # Update personal best score
                self.personal_best_positions[i] = self.positions[i].copy()  # Save position (must copy!)

                # Update global best if this is best position found by ANY particle
                # Global best = best position found by entire swarm
                # This is the "social" knowledge component of PSO
                # All particles will be attracted to this position in next velocity update
                if fitness < self.global_best_score:
                    self.global_best_score = fitness  # Update global best score
                    self.global_best_position = self.positions[i].copy()  # Save position (must copy!)

    def optimize(self) -> Tuple[np.ndarray, float, int, float, str]:
        """
        Run PSO optimization

        Returns:
            Tuple of (best_position, best_score, iterations, time, status)
        """
        # Record start time for timeout checking and performance measurement
        start_time = time.perf_counter()

        # Initialize swarm: create particles with random positions/velocities
        # This also evaluates initial fitness and sets initial personal/global bests
        self.initialize_swarm()

        # Initialize convergence history with the initial best score
        # We'll track global best score at each iteration to detect convergence
        self.convergence_history = [self.global_best_score]

        # Main optimization loop: iterate until convergence or max iterations
        # Each iteration is one complete PSO update cycle:
        # 1. Update velocities (based on inertia, cognitive, social)
        # 2. Update positions (move particles)
        # 3. Evaluate fitness (see if we found better solutions)
        for iteration in range(self.max_iterations):
            # Check timeout to prevent running forever on hard problems
            # According to assignment requirements, we need 5 minute timeout
            elapsed = time.perf_counter() - start_time
            if 0 < MAX_TIME_SEC < elapsed:
                # Timeout exceeded: return best found so far
                return (self.global_best_position,
                       self.global_best_score,
                       iteration,  # How many iterations we completed
                       time.perf_counter() - start_time,
                       "timeout")

            # PSO Algorithm Step 1: Update velocities
            # Apply velocity update equation with inertia, cognitive, and social components
            # This determines how particles will move in next step
            self.update_velocities()

            # PSO Algorithm Step 2: Update positions
            # Move particles based on their velocities
            # Also handles boundary conditions (particles that leave search space)
            self.update_positions()

            # PSO Algorithm Step 3: Evaluate and update bests
            # Compute fitness at new positions
            # Update personal bests (individual particle memories)
            # Update global best (swarm's collective knowledge)
            self.evaluate_and_update_bests()

            # Track convergence: record best score found so far
            # This history lets us plot convergence curves and detect stagnation
            self.convergence_history.append(self.global_best_score)

            # Check for convergence (algorithm has stopped improving)
            # If improvement is less than tolerance, we've converged
            if iteration > 0:
                prev_score = self.convergence_history[-2]  # Score from previous iteration
                curr_score = self.convergence_history[-1]  # Score from current iteration
                improvement = abs(prev_score - curr_score)  # How much did we improve?
                if improvement < self.tolerance:
                    # Converged: no significant improvement, stop early
                    return (self.global_best_position,
                           self.global_best_score,
                           iteration + 1,  # Total iterations completed
                           time.perf_counter() - start_time,
                           "converged")

        # Reached max iterations without timeout or convergence
        # Return best solution found so far
        return (self.global_best_position,
               self.global_best_score,
               self.max_iterations,
               time.perf_counter() - start_time,
               "max_iterations")


# Benchmark Functions

def rastrigin(x: np.ndarray) -> float:
    """
    Rastrigin function: highly multimodal benchmark

    f(x) = 10n + Σ[xi² - 10cos(2πxi)]

    Properties:
    - Domain: typically [-5.12, 5.12]^n
    - Global minimum: f(0,...,0) = 0
    - Many local minima (10^n for n dimensions)
    - Tests ability to escape local minima

    Reference: Jamil & Yang 2013, Rastrigin's Function

    Args:
        x: Point to evaluate (n-dimensional array)

    Returns:
        Function value at x
    """
    n = len(x)
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


def rosenbrock(x: np.ndarray) -> float:
    """
    Rosenbrock function: narrow valley benchmark

    f(x) = Σ[100(xi+1 - xi²)² + (xi - 1)²]

    Properties:
    - Domain: typically [-5, 10]^n
    - Global minimum: f(1,...,1) = 0
    - Narrow parabolic valley leading to minimum
    - Easy to find valley, hard to converge to minimum
    - Tests ability to navigate narrow valleys

    Reference: Jamil & Yang 2013, Rosenbrock Function

    Args:
        x: Point to evaluate (n-dimensional array)

    Returns:
        Function value at x
    """
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)


def run_pso_benchmark(func_name: str,
                     func: Callable[[np.ndarray], float],
                     dimensions: int,
                     bounds: Tuple[float, float],
                     num_trials: int = 3,
                     configs: List[Dict] = None):
    """
    Run PSO on benchmark function with multiple parameter configurations

    For each configuration:
    - Run multiple trials
    - Record best fitness, convergence speed
    - Report statistics

    Args:
        func_name: Name of function for display
        func: Objective function to minimize
        dimensions: Problem dimensionality
        bounds: Search space bounds
        num_trials: Number of independent trials per configuration
        configs: List of PSO parameter dictionaries
    """
    if configs is None:
        # Default configurations to test
        configs = [
            {"swarm_size": 30, "w": 0.7, "c1": 1.5, "c2": 1.5,
             "max_iterations": 1000},
            {"swarm_size": 50, "w": 0.5, "c1": 2.0, "c2": 2.0,
             "max_iterations": 1000},
            {"swarm_size": 40, "w": 0.9, "c1": 1.2, "c2": 1.2,
             "max_iterations": 1500},
        ]

    print(f"\n{'='*70}")
    print(f"PSO Benchmark: {func_name}")
    print(f"Dimensions: {dimensions}, Bounds: {bounds}")
    print(f"{'='*70}\n")

    for config_idx, config in enumerate(configs, 1):
        print(f"Configuration {config_idx}: {config}")
        print(f"{'-'*70}")

        best_scores = []
        times = []
        iterations = []

        for trial in range(1, num_trials + 1):
            # Create PSO optimizer with this configuration
            pso = PSO(
                objective_func=func,
                dimensions=dimensions,
                bounds=bounds,
                **config
            )

            # Run optimization
            _, best_score, iters, elapsed, status = pso.optimize()

            print(f"  Trial {trial}: score={best_score:.6e}, "
                  f"iters={iters}, time={elapsed:.4f}s, "
                  f"status={status}")

            best_scores.append(best_score)
            times.append(elapsed)
            iterations.append(iters)

        # Report statistics for this configuration
        avg_score = np.mean(best_scores)
        std_score = np.std(best_scores)
        min_score = np.min(best_scores)
        avg_time = np.mean(times)
        avg_iters = np.mean(iterations)

        print("\n  Summary:")
        print(f"    Best score: {min_score:.6e}")
        print(f"    Avg score: {avg_score:.6e} +/- {std_score:.6e}")
        print(f"    Avg iterations: {avg_iters:.1f}")
        print(f"    Avg time: {avg_time:.4f}s")
        print()


if __name__ == "__main__":
    # Set random seed for reproducibility during testing
    # Uncomment for deterministic results:
    # np.random.seed(42)
    # random.seed(42)

    print("="*70)
    print("CS 4820/5820 Homework 2 - Part C1: PSO Benchmark Optimization")
    print("="*70)

    # Test on Rastrigin function
    # 10-dimensional problem with many local minima
    print("\n" + "="*70)
    print("Rastrigin Function")
    print("="*70)
    print("Properties:")
    print("  - Highly multimodal with many local minima")
    print("  - Global minimum: f(0,0,...,0) = 0")
    print("  - Domain: [-5.12, 5.12]^10")
    print()

    run_pso_benchmark(
        func_name="Rastrigin",
        func=rastrigin,
        dimensions=10,
        bounds=(-5.12, 5.12),
        num_trials=3
    )

    # Test on Rosenbrock function
    # 10-dimensional problem with narrow valley
    print("\n" + "="*70)
    print("Rosenbrock Function")
    print("="*70)
    print("Properties:")
    print("  - Narrow parabolic valley")
    print("  - Global minimum: f(1,1,...,1) = 0")
    print("  - Domain: [-5, 10]^10")
    print()

    run_pso_benchmark(
        func_name="Rosenbrock",
        func=rosenbrock,
        dimensions=10,
        bounds=(-5, 10),
        num_trials=3
    )

    # Example: Show convergence curve for single run
    print("\n" + "="*70)
    print("Detailed Example: Rastrigin Convergence")
    print("="*70)
    print("\nRunning PSO with:")
    print("  Swarm size: 30")
    print("  Inertia weight (w): 0.7")
    print("  Cognitive coeff (c1): 1.5")
    print("  Social coeff (c2): 1.5")
    print("  Max iterations: 500")
    print()

    pso = PSO(
        objective_func=rastrigin,
        dimensions=10,
        bounds=(-5.12, 5.12),
        swarm_size=30,
        w=0.7,
        c1=1.5,
        c2=1.5,
        max_iterations=500
    )

    best_pos, best_score, iters, elapsed, status = pso.optimize()

    print(f"Status: {status}")
    print(f"Final best score: {best_score:.6e}")
    print(f"Iterations completed: {iters}")
    print(f"Time: {elapsed:.4f} seconds")
    print(f"Runtime: {elapsed*1000:.2f} milliseconds")
    print("\nBest position found:")
    print(f"  {best_pos}")
    print("\nConvergence history (every 50 iterations):")
    for i in range(0, len(pso.convergence_history), 50):
        improvement = ""
        if i > 0:
            prev = pso.convergence_history[i-50]
            curr = pso.convergence_history[i]
            delta = prev - curr
            improvement = f" (improved by {delta:.6e})"
        score = pso.convergence_history[i]
        print(f"  Iteration {i:3d}: {score:12.6e}{improvement}")

    print("\n" + "="*70)
    print("All PSO benchmark tests completed")
    print("="*70)
