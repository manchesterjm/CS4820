# generate_convergence_plots.py
# Generate convergence plots for PSO benchmark optimization
#
# Creates publication-quality plots showing:
# - Convergence behavior across iterations
# - Comparison of different parameter configurations
# - Best fitness over time for Rastrigin and Rosenbrock functions
#
# Outputs:
# - rastrigin_convergence.pdf: Convergence plot for Rastrigin function
# - rosenbrock_convergence.pdf: Convergence plot for Rosenbrock function

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for generating files
import matplotlib.pyplot as plt
from pso_benchmark import PSO, rastrigin, rosenbrock

# Configure matplotlib for publication quality
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['lines.linewidth'] = 1.5


def run_pso_with_convergence(objective_func, func_name, bounds, dimensions=10, trials=3):
    """
    Run PSO with multiple configurations and collect convergence data

    Args:
        objective_func: Function to optimize
        func_name: Name for display
        bounds: (min, max) search space bounds
        dimensions: Problem dimensionality
        trials: Number of trials per configuration

    Returns:
        Dictionary mapping config names to convergence histories
    """
    # PSO parameter configurations (same as in main experiments)
    configs = {
        'Config 1 (swarm=30, w=0.7)': {
            'swarm_size': 30,
            'w': 0.7,
            'c1': 1.5,
            'c2': 1.5,
            'max_iterations': 1000
        },
        'Config 2 (swarm=50, w=0.5)': {
            'swarm_size': 50,
            'w': 0.5,
            'c1': 2.0,
            'c2': 2.0,
            'max_iterations': 1000
        },
        'Config 3 (swarm=40, w=0.9)': {
            'swarm_size': 40,
            'w': 0.9,
            'c1': 1.2,
            'c2': 1.2,
            'max_iterations': 1500
        }
    }

    results = {}

    print(f"\n{'='*70}")
    print(f"Generating convergence data for {func_name}")
    print(f"{'='*70}\n")

    for config_name, params in configs.items():
        print(f"Running {config_name}...")
        convergence_histories = []

        for trial in range(trials):
            # Create PSO optimizer
            pso = PSO(
                objective_func=objective_func,
                dimensions=dimensions,
                bounds=bounds,
                swarm_size=params['swarm_size'],
                w=params['w'],
                c1=params['c1'],
                c2=params['c2'],
                max_iterations=params['max_iterations'],
                tolerance=1e-6
            )

            # Run optimization
            best_pos, best_score, iters, elapsed, status = pso.optimize()

            # Store convergence history
            convergence_histories.append(pso.convergence_history)

            print(f"  Trial {trial + 1}: Final score = {best_score:.4f}, " +
                  f"Iterations = {iters}, Status = {status}")

        # Average convergence across trials
        # Pad shorter histories to same length with their final value
        max_len = max(len(h) for h in convergence_histories)
        padded_histories = []
        for history in convergence_histories:
            padded = list(history) + [history[-1]] * (max_len - len(history))
            padded_histories.append(padded)

        avg_convergence = np.mean(padded_histories, axis=0)
        results[config_name] = avg_convergence

        print(f"  Average final score: {avg_convergence[-1]:.4f}\n")

    return results


def plot_convergence(convergence_data, func_name, global_min, output_file):
    """
    Create convergence plot showing all configurations

    Args:
        convergence_data: Dict mapping config names to convergence arrays
        func_name: Function name for title
        global_min: Global minimum value for reference line
        output_file: Output PDF filename
    """
    plt.figure(figsize=(8, 5))

    # Color scheme for different configs
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    linestyles = ['-', '--', '-.']

    # Plot each configuration
    for (config_name, convergence), color, linestyle in zip(
            convergence_data.items(), colors, linestyles):
        iterations = np.arange(len(convergence))
        plt.plot(iterations, convergence, label=config_name,
                color=color, linestyle=linestyle, linewidth=2)

    # Add horizontal line for global minimum
    plt.axhline(y=global_min, color='red', linestyle=':', linewidth=1.5,
                label=f'Global Min = {global_min}', alpha=0.7)

    # Labels and title
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness Value (log scale)')
    plt.title(f'PSO Convergence: {func_name} Function (10D, averaged over 3 trials)')
    plt.yscale('log')  # Log scale for better visualization
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', framealpha=0.9)

    # Tight layout to prevent label cutoff
    plt.tight_layout()

    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nConvergence plot saved to: {output_file}")
    plt.close()


def main():
    """
    Generate all convergence plots for PSO benchmark experiments
    """
    print("\n" + "="*70)
    print("PSO CONVERGENCE PLOT GENERATION")
    print("="*70)
    print("\nThis script generates convergence plots for:")
    print("  - Rastrigin function (highly multimodal)")
    print("  - Rosenbrock function (narrow valley)")
    print("\nEach plot shows 3 parameter configurations averaged over 3 trials\n")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate Rastrigin convergence plot
    print("\n" + "-"*70)
    print("1. RASTRIGIN FUNCTION")
    print("-"*70)

    rastrigin_convergence = run_pso_with_convergence(
        objective_func=rastrigin,
        func_name='Rastrigin',
        bounds=(-5.12, 5.12),
        dimensions=10,
        trials=3
    )

    plot_convergence(
        convergence_data=rastrigin_convergence,
        func_name='Rastrigin',
        global_min=0,
        output_file='rastrigin_convergence.pdf'
    )

    # Generate Rosenbrock convergence plot
    print("\n" + "-"*70)
    print("2. ROSENBROCK FUNCTION")
    print("-"*70)

    rosenbrock_convergence = run_pso_with_convergence(
        objective_func=rosenbrock,
        func_name='Rosenbrock',
        bounds=(-5, 10),
        dimensions=10,
        trials=3
    )

    plot_convergence(
        convergence_data=rosenbrock_convergence,
        func_name='Rosenbrock',
        global_min=0,
        output_file='rosenbrock_convergence.pdf'
    )

    print("\n" + "="*70)
    print("CONVERGENCE PLOT GENERATION COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  - rastrigin_convergence.pdf")
    print("  - rosenbrock_convergence.pdf")
    print("\nThese plots show how PSO converges over iterations for different")
    print("parameter configurations, demonstrating the effect of swarm size,")
    print("inertia weight, and cognitive/social coefficients.\n")


if __name__ == '__main__':
    main()
