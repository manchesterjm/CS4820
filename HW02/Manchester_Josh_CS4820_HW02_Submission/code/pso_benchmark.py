# pso_benchmark.py
# Implements Particle Swarm Optimization (PSO) for benchmark function optimization
#
# PSO Overview:
# - Population-based metaheuristic inspired by social behavior of bird flocking
# - Each particle has position (candidate solution) and velocity
# - Particles move through search space influenced by:
#   * Their own best position (cognitive component)
#   * Global best position (social component)
#   * Inertia from current velocity
#
# Benchmark Functions:
# 1. Rastrigin: f(x) = 10n + Σ[xi² - 10cos(2πxi)]
#    - Highly multimodal (many local minima)
#    - Global minimum: f(0,...,0) = 0
#
# 2. Rosenbrock: f(x) = Σ[100(xi+1 - xi²)² + (xi - 1)²]
#    - Narrow valley leading to global minimum
#    - Global minimum: f(1,...,1) = 0
#
# Algorithm References:
# - Lecture 7: Search Optimization Part III (PSO slides)
# - Kennedy & Eberhart, "Particle Swarm Optimization," 1995
# - Benchmark functions from Jamil & Yang, "A Literature Survey of Benchmark Functions"

from typing import Callable, Tuple, List, Dict
import numpy as np
import random
import time
import math

# Safety limit to prevent excessive computation
MAX_TIME_SEC = 300  # 5 minute timeout as specified in requirements


class PSO:
    """
    Particle Swarm Optimization for continuous function minimization

    PSO Algorithm:
    1. Initialize swarm with random positions and velocities
    2. Evaluate fitness of each particle
    3. Update personal best and global best
    4. Update velocities based on inertia, cognitive, and social components
    5. Update positions
    6. Repeat until convergence or max iterations

    Velocity update equation:
    v[i] = w*v[i] + c1*r1*(pbest[i] - x[i]) + c2*r2*(gbest - x[i])

    where:
    - w: inertia weight (controls exploration vs exploitation)
    - c1: cognitive coefficient (attraction to personal best)
    - c2: social coefficient (attraction to global best)
    - r1, r2: random values in [0,1]

    Position update equation:
    x[i] = x[i] + v[i]
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
        Velocities: small random values (usually fraction of position range)

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

        Random components r1, r2 add stochasticity
        """
        # Generate random matrices for cognitive and social components
        r1 = np.random.random((self.swarm_size, self.dimensions))
        r2 = np.random.random((self.swarm_size, self.dimensions))

        # Inertia component: maintain current velocity
        inertia = self.w * self.velocities

        # Cognitive component: attraction to personal best
        cognitive = self.c1 * r1 * (self.personal_best_positions - self.positions)

        # Social component: attraction to global best
        social = self.c2 * r2 * (self.global_best_position - self.positions)

        # Update velocities
        self.velocities = inertia + cognitive + social

        # Velocity clamping to prevent explosion
        # Limit velocity to fraction of search space
        min_bound, max_bound = self.bounds
        v_max = (max_bound - min_bound) * 0.2
        self.velocities = np.clip(self.velocities, -v_max, v_max)

    def update_positions(self):
        """
        Update particle positions based on velocities

        Position update: x[i] = x[i] + v[i]

        Boundary handling:
        - If particle moves outside bounds, clamp to boundary
        - Reset velocity component that caused violation
        """
        # Update positions
        self.positions = self.positions + self.velocities

        # Boundary handling: clamp positions and reset velocities
        min_bound, max_bound = self.bounds

        # Find violations
        below_min = self.positions < min_bound
        above_max = self.positions > max_bound

        # Clamp positions
        self.positions = np.clip(self.positions, min_bound, max_bound)

        # Reset velocity components that hit boundaries
        self.velocities[below_min] *= -0.5  # Bounce back with damping
        self.velocities[above_max] *= -0.5

    def evaluate_and_update_bests(self):
        """
        Evaluate fitness and update personal and global bests

        For each particle:
        - Evaluate objective function at current position
        - Update personal best if improved
        - Update global best if any particle improved it
        """
        for i in range(self.swarm_size):
            # Evaluate fitness
            fitness = self.objective_func(self.positions[i])

            # Update personal best if improved
            if fitness < self.personal_best_scores[i]:
                self.personal_best_scores[i] = fitness
                self.personal_best_positions[i] = self.positions[i].copy()

                # Update global best if improved
                if fitness < self.global_best_score:
                    self.global_best_score = fitness
                    self.global_best_position = self.positions[i].copy()

    def optimize(self) -> Tuple[np.ndarray, float, int, float, str]:
        """
        Run PSO optimization

        Returns:
            Tuple of (best_position, best_score, iterations, time, status)
        """
        start_time = time.perf_counter()

        # Initialize swarm
        self.initialize_swarm()
        self.convergence_history = [self.global_best_score]

        # Main optimization loop
        for iteration in range(self.max_iterations):
            # Check timeout
            if MAX_TIME_SEC > 0 and (time.perf_counter() - start_time) > MAX_TIME_SEC:
                return (self.global_best_position,
                       self.global_best_score,
                       iteration,
                       time.perf_counter() - start_time,
                       "timeout")

            # Update velocities
            self.update_velocities()

            # Update positions
            self.update_positions()

            # Evaluate and update bests
            self.evaluate_and_update_bests()

            # Track convergence
            self.convergence_history.append(self.global_best_score)

            # Check for convergence (no improvement)
            if iteration > 0:
                improvement = abs(self.convergence_history[-2] - self.convergence_history[-1])
                if improvement < self.tolerance:
                    return (self.global_best_position,
                           self.global_best_score,
                           iteration + 1,
                           time.perf_counter() - start_time,
                           "converged")

        # Reached max iterations
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
            {"swarm_size": 30, "w": 0.7, "c1": 1.5, "c2": 1.5, "max_iterations": 1000},
            {"swarm_size": 50, "w": 0.5, "c1": 2.0, "c2": 2.0, "max_iterations": 1000},
            {"swarm_size": 40, "w": 0.9, "c1": 1.2, "c2": 1.2, "max_iterations": 1500},
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
            best_pos, best_score, iters, elapsed, status = pso.optimize()

            print(f"  Trial {trial}: score={best_score:.6e}, "
                  f"iters={iters}, time={elapsed:.4f}s, status={status}")

            best_scores.append(best_score)
            times.append(elapsed)
            iterations.append(iters)

        # Report statistics for this configuration
        avg_score = np.mean(best_scores)
        std_score = np.std(best_scores)
        min_score = np.min(best_scores)
        avg_time = np.mean(times)
        avg_iters = np.mean(iterations)

        print(f"\n  Summary:")
        print(f"    Best score: {min_score:.6e}")
        print(f"    Avg score: {avg_score:.6e} ± {std_score:.6e}")
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
    print(f"\nBest position found:")
    print(f"  {best_pos}")
    print(f"\nConvergence history (every 50 iterations):")
    for i in range(0, len(pso.convergence_history), 50):
        improvement = ""
        if i > 0:
            delta = pso.convergence_history[i-50] - pso.convergence_history[i]
            improvement = f" (improved by {delta:.6e})"
        print(f"  Iteration {i:3d}: {pso.convergence_history[i]:12.6e}{improvement}")

    print("\n" + "="*70)
    print("All PSO benchmark tests completed")
    print("="*70)
