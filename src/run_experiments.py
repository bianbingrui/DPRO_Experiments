"""Run numerical experiments for DPRO and MEU analysis."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import List, Dict, Tuple, Any
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel

from configs.large_config import (
    X_SPACE_SIZE, OMEGA_SPACE_SIZE, S_SPACE_SIZE,
    NUM_BASIC_ACTS, BASIC_AA_ACTS, COMPARISON_AA_ACTS,
    RISK_PROFILES, TRUE_DISTRIBUTION, MAX_SAMPLES,
    NUM_TRIALS, RANDOM_SEED, EPSILON
)
from src.utils import (
    generate_risk_parameters,
    calculate_meu_gamma,
    calculate_utility,
    convert_act_to_matrix
)
from src.optimization import solve_dpro, solve_meu, solve_nr
from src.experiment import DPROExperiment

console = Console()

def display_optimal_solution(dpro_z: np.ndarray, meu_z: np.ndarray, nr_z: np.ndarray, 
                        dpro_value: float, meu_value: float, nr_value: float):
    """Display the optimal solution in a formatted table with detailed visualization.
    
    Args:
        dpro_z: DPRO solution weights
        meu_z: MEU solution weights
        nr_z: Non-robust solution weights
        dpro_value: DPRO optimal value
        meu_value: MEU optimal value
        nr_value: NR optimal value
    """
    # Create main comparison table
    table = Table(title="[bold]Optimal Solution Comparison[/bold]")
    
    # Add columns with improved styling
    table.add_column("Method", style="cyan", justify="center")
    table.add_column("Optimal Value", style="magenta", justify="right")
    
    # Create detailed weight tables for each method
    dpro_table = Table(show_header=True, header_style="bold cyan", show_lines=True)
    dpro_table.add_column("Act", style="dim")
    dpro_table.add_column("Weight", justify="right")
    dpro_table.add_column("Percentage", justify="right")
    
    meu_table = Table(show_header=True, header_style="bold red", show_lines=True)
    meu_table.add_column("Act", style="dim")
    meu_table.add_column("Weight", justify="right")
    meu_table.add_column("Percentage", justify="right")
    
    nr_table = Table(show_header=True, header_style="bold green", show_lines=True)
    nr_table.add_column("Act", style="dim")
    nr_table.add_column("Weight", justify="right")
    nr_table.add_column("Percentage", justify="right")
    
    # Add rows to DPRO table
    for i, weight in enumerate(dpro_z):
        percentage = weight * 100
        dpro_table.add_row(
            f"r_{i+1}",
            f"{weight:.6f}",
            f"{percentage:.2f}%"
        )
    
    # Add rows to MEU table
    for i, weight in enumerate(meu_z):
        percentage = weight * 100
        meu_table.add_row(
            f"r_{i+1}",
            f"{weight:.6f}",
            f"{percentage:.2f}%"
        )
    
    # Add rows to NR table
    for i, weight in enumerate(nr_z):
        percentage = weight * 100
        nr_table.add_row(
            f"r_{i+1}",
            f"{weight:.6f}",
            f"{percentage:.2f}%"
        )
    
    # Add rows to main table
    table.add_row(
        "DPRO",
        f"{dpro_value:.6f}",
    )
    table.add_row(
        "MEU",
        f"{meu_value:.6f}",
    )
    table.add_row(
        "NR",
        f"{nr_value:.6f}",
    )
    
    # Display the tables with proper spacing and formatting
    console.print("\n")
    console.print(Panel(table, title="[bold yellow]Optimization Results Summary[/bold yellow]", 
                       border_style="yellow"))
    
    # Display detailed weight distributions
    console.print("\n[bold cyan]DPRO Optimal Solution (z*_DPRO)[/bold cyan]")
    console.print(dpro_table)
    
    console.print("\n[bold red]MEU Optimal Solution (z*_MEU)[/bold red]")
    console.print(meu_table)
    
    console.print("\n[bold green]Non-Robust Optimal Solution (z*_NR)[/bold green]")
    console.print(nr_table)
    
    # Add a visual representation of weight distributions
    console.print("\n[bold]Weight Distribution Visualization:[/bold]")
    max_bar_width = 40  # Maximum width for the bar chart
    
    console.print("\n[cyan]DPRO Weights:[/cyan]")
    for i, weight in enumerate(dpro_z):
        bar_width = int(weight * max_bar_width)
        bar = "█" * bar_width
        console.print(f"r_{i+1:<2}: {bar} {weight:.4f}")
    
    console.print("\n[red]MEU Weights:[/red]")
    for i, weight in enumerate(meu_z):
        bar_width = int(weight * max_bar_width)
        bar = "█" * bar_width
        console.print(f"r_{i+1:<2}: {bar} {weight:.4f}")
    
    console.print("\n[green]NR Weights:[/green]")
    for i, weight in enumerate(nr_z):
        bar_width = int(weight * max_bar_width)
        bar = "█" * bar_width
        console.print(f"r_{i+1:<2}: {bar} {weight:.4f}")

def run_risk_aversion_experiment(
    n_experiments: int = 100,
    seed: int = RANDOM_SEED
) -> Dict[str, np.ndarray]:
    """Run experiments with varying risk aversion parameters.
    
    Args:
        n_experiments: Number of experiments to run
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary containing experiment results
    """
    np.random.seed(seed)
    
    # Convert acts to matrix format
    acts_matrix = [convert_act_to_matrix(act, X_SPACE_SIZE, S_SPACE_SIZE) 
                  for act in BASIC_AA_ACTS.values()]
    
    # Initialize results storage
    results = {
        'gamma_omega': [],  # Risk aversion parameters
        'gamma_meu': [],    # MEU risk aversion
        'dpro_weights': [], # DPRO solution weights
        'meu_weights': [],  # MEU solution weights
        'nr_weights': [],   # NR solution weights
        'dpro_values': [],  # DPRO optimal values
        'meu_values': [],   # MEU optimal values
        'nr_values': []     # NR optimal values
    }
    
    # Create progress bar with rich
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Running risk aversion experiments...", total=n_experiments)
        
        best_dpro_value = float('-inf')
        best_meu_value = float('-inf')
        best_nr_value = float('-inf')
        best_dpro_weights = None
        best_meu_weights = None
        best_nr_weights = None
        
        for _ in range(n_experiments):
            # Generate random risk parameters
            gamma_omega = generate_risk_parameters(OMEGA_SPACE_SIZE, seed + _)
            
            # Use true distribution with small perturbations
            phi_hat = TRUE_DISTRIBUTION + np.random.normal(0, 0.05, OMEGA_SPACE_SIZE)
            phi_hat = np.clip(phi_hat, 0.1, None)  # Ensure no probability is too small
            phi_hat = phi_hat / np.sum(phi_hat)    # Normalize
            
            psi_hat = TRUE_DISTRIBUTION + np.random.normal(0, 0.05, S_SPACE_SIZE)
            psi_hat = np.clip(psi_hat, 0.1, None)
            psi_hat = psi_hat / np.sum(psi_hat)
            
            # Calculate MEU gamma
            gamma_meu = calculate_meu_gamma(gamma_omega, phi_hat)
            
            # Solve DPRO, MEU, and NR with matrix format acts
            dpro_z, dpro_value = solve_dpro(phi_hat, psi_hat, gamma_omega, acts_matrix)
            meu_z, meu_value = solve_meu(phi_hat, psi_hat, gamma_meu, acts_matrix)
            nr_z, nr_value = solve_nr(phi_hat, psi_hat, gamma_omega, acts_matrix)
            
            # Update best solutions
            if dpro_value > best_dpro_value:
                best_dpro_value = dpro_value
                best_dpro_weights = dpro_z
            if meu_value > best_meu_value:
                best_meu_value = meu_value
                best_meu_weights = meu_z
            if nr_value > best_nr_value:
                best_nr_value = nr_value
                best_nr_weights = nr_z
            
            # Store results
            results['gamma_omega'].append(gamma_omega)
            results['gamma_meu'].append(gamma_meu)
            results['dpro_weights'].append(dpro_z)
            results['meu_weights'].append(meu_z)
            results['nr_weights'].append(nr_z)
            results['dpro_values'].append(dpro_value)
            results['meu_values'].append(meu_value)
            results['nr_values'].append(nr_value)
            
            # Update progress
            progress.update(task, advance=1)
        
        # Display the best solution found
        console.print("\n[bold yellow]Best Solution Found:[/bold yellow]")
        display_optimal_solution(
            best_dpro_weights, best_meu_weights, best_nr_weights,
            best_dpro_value, best_meu_value, best_nr_value
        )
    
    # Convert lists to numpy arrays
    for key in results:
        results[key] = np.array(results[key])
    
    return results

def run_realization_experiment(
    n_samples: int = MAX_SAMPLES,
    n_evaluations: int = 100,
    n_people: int = 100,
    epsilon: float = EPSILON,
    seed: int = RANDOM_SEED
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Run realization experiments with proper evaluation process."""
    np.random.seed(seed)
    
    # Initialize results storage
    results = {
        'dpro_values': [],     # DPRO welfare values
        'meu_values': [],      # MEU welfare values
        'nr_values': []        # Non-robust welfare values
    }
    
    # Step 1: Generate empirical distributions using n_samples
    console.print(f"\n[cyan]Generating empirical distributions using {n_samples} samples...[/cyan]")
    phi_star = TRUE_DISTRIBUTION
    psi_star = np.array([0.3, 0.4, 0.2, 0.1])
    
    phi_samples = np.random.choice(OMEGA_SPACE_SIZE, size=n_samples, p=phi_star)
    psi_samples = np.random.choice(S_SPACE_SIZE, size=n_samples, p=psi_star)
    
    phi_hat = np.bincount(phi_samples, minlength=OMEGA_SPACE_SIZE) / n_samples
    psi_hat = np.bincount(psi_samples, minlength=S_SPACE_SIZE) / n_samples
    
    # Step 2: Solve for optimal policies using empirical distributions
    console.print("[cyan]Solving for optimal DPRO, MEU, and NR policies...[/cyan]")
    base_gamma = np.array(list(RISK_PROFILES.values()))
    gamma_meu = calculate_meu_gamma(base_gamma, phi_hat)
    
    # Convert acts to matrix format
    acts_matrix = [convert_act_to_matrix(act, X_SPACE_SIZE, S_SPACE_SIZE) 
                  for act in BASIC_AA_ACTS.values()]
    
    dpro_z, dpro_value = solve_dpro(phi_hat, psi_hat, base_gamma, acts_matrix)
    meu_z, meu_value = solve_meu(phi_hat, psi_hat, gamma_meu, acts_matrix)
    nr_z, nr_value = solve_nr(phi_hat, psi_hat, base_gamma, acts_matrix)
    
    # Display optimal solutions
    display_optimal_solution(dpro_z, meu_z, nr_z, dpro_value, meu_value, nr_value)
    
    # Step 3: Evaluate policies n_evaluations times
    console.print(f"[cyan]Evaluating policies over {n_evaluations} test realizations...[/cyan]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Evaluating welfare...", total=n_evaluations)
        
        for _ in range(n_evaluations):
            # Sample n_people risk attitudes from true phi*
            mental_states = np.random.choice(OMEGA_SPACE_SIZE, size=n_people, p=phi_star)
            
            # Sample one material state from true psi*
            material_state = np.random.choice(S_SPACE_SIZE, p=psi_star)
            
            # Calculate welfare for all three policies
            dpro_welfare = 0
            meu_welfare = 0
            nr_welfare = 0
            
            for person in range(n_people):
                mental_state = mental_states[person]
                gamma = base_gamma[mental_state]
                
                # Calculate utilities for this person under all policies
                for i, act in enumerate(acts_matrix):
                    outcome_probs = act[:, material_state]
                    utilities = np.array([calculate_utility(x+1, gamma) 
                                       for x in range(X_SPACE_SIZE)])
                    dpro_welfare += dpro_z[i] * np.sum(utilities * outcome_probs)
                    meu_welfare += meu_z[i] * np.sum(utilities * outcome_probs)
                    nr_welfare += nr_z[i] * np.sum(utilities * outcome_probs)
            
            # Average welfare over number of people
            dpro_welfare /= n_people
            meu_welfare /= n_people
            nr_welfare /= n_people
            
            # Store results
            results['dpro_values'].append(dpro_welfare)
            results['meu_values'].append(meu_welfare)
            results['nr_values'].append(nr_welfare)
            
            progress.update(task, advance=1)
    
    # Convert lists to numpy arrays
    for key in results:
        results[key] = np.array(results[key])
    
    # Create parameters dictionary
    params = {
        "n_samples": n_samples,
        "n_evaluations": n_evaluations,
        "n_people": n_people,
        "epsilon": epsilon,
        "seed": seed
    }
    
    return results, params

def ensure_plot_directory(subdir: str = "") -> str:
    """Ensure the plot directory exists.
    
    Args:
        subdir: Optional subdirectory name within plots directory
        
    Returns:
        Path to the plot directory
    """
    # Create base plots directory
    base_dir = "plots"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # If subdirectory is specified, create it
    if subdir:
        plot_dir = os.path.join(base_dir, subdir)
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        return plot_dir
    return base_dir

def save_out_of_sample_plot(realization_results: Dict[str, np.ndarray], params: Dict[str, Any]):
    """Save just the out-of-sample performance comparison plot with parameters in filename."""
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot data
    evaluation_indices = np.arange(len(realization_results['dpro_values']))
    
    # Plot individual points
    ax.scatter(evaluation_indices, realization_results['dpro_values'],
              marker='*', color='blue', s=50, alpha=0.3, label='DPRO')
    ax.scatter(evaluation_indices, realization_results['meu_values'],
              marker='^', color='red', s=50, alpha=0.3, label='MEU')
    ax.scatter(evaluation_indices, realization_results['nr_values'],
              marker='o', color='green', s=50, alpha=0.3, label='NR')
    
    # Plot moving averages
    window = 10
    dpro_ma = np.convolve(realization_results['dpro_values'], 
                         np.ones(window)/window, mode='valid')
    meu_ma = np.convolve(realization_results['meu_values'], 
                        np.ones(window)/window, mode='valid')
    nr_ma = np.convolve(realization_results['nr_values'], 
                       np.ones(window)/window, mode='valid')
    ma_indices = evaluation_indices[window-1:]
    
    ax.plot(ma_indices, dpro_ma, 'b-', alpha=0.8, label='DPRO (MA)')
    ax.plot(ma_indices, meu_ma, 'r-', alpha=0.8, label='MEU (MA)')
    ax.plot(ma_indices, nr_ma, 'g-', alpha=0.8, label='NR (MA)')
    
    # Labels and title
    ax.set_xlabel('Test Realization Index')
    ax.set_ylabel('Social Welfare')
    title = 'Out-of-sample Performance Comparison\n'
    title += f'ε={params["epsilon"]}, n_samples={params["n_samples"]}, n_people={params["n_people"]}'
    ax.set_title(title)
    ax.legend()
    
    # Add summary statistics
    dpro_mean = np.mean(realization_results['dpro_values'])
    meu_mean = np.mean(realization_results['meu_values'])
    nr_mean = np.mean(realization_results['nr_values'])
    
    dpro_std = np.std(realization_results['dpro_values'])
    meu_std = np.std(realization_results['meu_values'])
    nr_std = np.std(realization_results['nr_values'])
    
    stats_text = f'Mean Values:\n'
    stats_text += f'DPRO: {dpro_mean:.4f} ± {dpro_std:.4f}\n'
    stats_text += f'MEU:  {meu_mean:.4f} ± {meu_std:.4f}\n'
    stats_text += f'NR:   {nr_mean:.4f} ± {nr_std:.4f}'
    
    ax.text(0.95, 0.95, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Create filename with parameters and ensure plot directory exists
    plot_dir = ensure_plot_directory("out_of_sample")
    filename = f'comparison_eps{params["epsilon"]}_samples{params["n_samples"]}_npeople{params["n_people"]}.png'
    filepath = os.path.join(plot_dir, filename)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    console.print(f"[green]Saved out-of-sample plot to: {filepath}[/green]")

def plot_results(results: Dict[str, np.ndarray], realization_results: Dict[str, np.ndarray], params: Dict[str, Any]):
    """Plot experiment results.
    
    Args:
        results: Dictionary containing risk aversion experiment results
        realization_results: Dictionary containing realization experiment results
        params: Dictionary of parameters used in the experiment
    """
    # Debug prints
    print("Array shapes:")
    print(f"dpro_weights shape: {results['dpro_weights'].shape}")
    print(f"meu_weights shape: {results['meu_weights'].shape}")
    print(f"gamma_omega shape: {results['gamma_omega'].shape}")
    
    # Set up the figure with adjusted size for more acts
    fig, axes = plt.subplots(3, 2, figsize=(16, 20))
    fig.suptitle(f'DPRO vs MEU Analysis (ε={params["epsilon"]}, trials={params["n_samples"]}, n_people={params["n_people"]})', 
                fontsize=16)
    
    # Plot 1: Risk aversion vs solution weights for safe acts (first half)
    ax = axes[0, 0]
    n_safe_acts = NUM_BASIC_ACTS // 2
    safe_acts_weight_dpro = np.sum(results['dpro_weights'][:, :n_safe_acts], axis=1)
    safe_acts_weight_meu = np.sum(results['meu_weights'][:, :n_safe_acts], axis=1)
    
    ax.scatter(results['gamma_omega'][:, 0], safe_acts_weight_dpro,
              marker='*', color='blue', s=100, label='DPRO (Safe Acts)', alpha=0.6)
    ax.scatter(results['gamma_omega'][:, 0], safe_acts_weight_meu,
              marker='^', color='red', s=100, label='MEU (Safe Acts)', alpha=0.6)
    ax.set_xlabel('Risk Aversion (γ)')
    ax.set_ylabel('Total Weight on Safe Acts')
    ax.set_title('Risk Aversion vs Safe Act Weights')
    ax.legend()
    
    # Plot 2: DPRO vs MEU optimal values
    ax = axes[0, 1]
    ax.scatter(results['dpro_values'], results['meu_values'], 
              marker='*', color='blue', s=100, alpha=0.6)
    ax.plot([min(results['dpro_values']), max(results['dpro_values'])],
            [min(results['dpro_values']), max(results['dpro_values'])],
            'r--', label='y=x')
    ax.set_xlabel('DPRO Optimal Value')
    ax.set_ylabel('MEU Optimal Value')
    ax.set_title('DPRO vs MEU Optimal Values')
    ax.legend()
    
    # Plot 3: Weight distributions with adjusted bar width
    ax = axes[1, 0]
    x = np.arange(NUM_BASIC_ACTS)
    width = 0.3  # Smaller width for more bars
    ax.bar(x - width/2, np.mean(results['dpro_weights'], axis=0), width,
           color='blue', label='DPRO', alpha=0.6)
    ax.bar(x + width/2, np.mean(results['meu_weights'], axis=0), width,
           color='red', label='MEU', alpha=0.6)
    ax.set_xlabel('Basic AA Act Index')
    ax.set_ylabel('Average Weight')
    ax.set_title('Average Weight Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels([f'r_{i+1}' for i in range(NUM_BASIC_ACTS)], rotation=45)
    ax.legend()
    
    # Plot 4: Risk aversion distribution
    ax = axes[1, 1]
    colors = ['blue', 'red', 'green', 'purple']
    for i in range(OMEGA_SPACE_SIZE):
        sns.kdeplot(results['gamma_omega'][:, i], ax=ax, 
                   color=colors[i], label=f'Mental State {i+1}')
    ax.set_xlabel('Risk Aversion (γ)')
    ax.set_ylabel('Density')
    ax.set_title('Risk Aversion Distribution by Mental State')
    ax.legend()
    
    # Plot 5: Out-of-sample Performance Comparison
    ax = axes[2, 0]
    evaluation_indices = np.arange(len(realization_results['dpro_values']))
    
    # Plot individual points
    ax.scatter(evaluation_indices, realization_results['dpro_values'],
              marker='*', color='blue', s=50, alpha=0.3, label='DPRO')
    ax.scatter(evaluation_indices, realization_results['meu_values'],
              marker='^', color='red', s=50, alpha=0.3, label='MEU')
    
    # Plot moving averages
    window = 10
    dpro_ma = np.convolve(realization_results['dpro_values'], 
                         np.ones(window)/window, mode='valid')
    meu_ma = np.convolve(realization_results['meu_values'], 
                        np.ones(window)/window, mode='valid')
    ma_indices = evaluation_indices[window-1:]
    
    ax.plot(ma_indices, dpro_ma, 'b-', alpha=0.8, label='DPRO (MA)')
    ax.plot(ma_indices, meu_ma, 'r-', alpha=0.8, label='MEU (MA)')
    
    ax.set_xlabel('Evaluation Index')
    ax.set_ylabel('Social Welfare')
    ax.set_title('Out-of-sample Performance Comparison')
    ax.legend()
    
    # Plot 6: Performance difference histogram
    ax = axes[2, 1]
    performance_diff = realization_results['dpro_values'] - realization_results['meu_values']
    sns.histplot(performance_diff, kde=True, ax=ax, color='purple')
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.6)
    ax.set_xlabel('DPRO - MEU Performance')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Performance Difference')
    
    # Add summary statistics
    mean_diff = np.mean(performance_diff)
    std_diff = np.std(performance_diff)
    win_rate = np.mean(performance_diff > 0) * 100
    
    stats_text = f'Mean Diff: {mean_diff:.4f}\n'
    stats_text += f'Std Dev: {std_diff:.4f}\n'
    stats_text += f'DPRO Win Rate: {win_rate:.1f}%'
    
    ax.text(0.95, 0.95, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Create filename with parameters and ensure plot directory exists
    plot_dir = ensure_plot_directory("full_results")
    filename = f'analysis_eps{params["epsilon"]}_trials{params["n_samples"]}_npeople{params["n_people"]}.png'
    filepath = os.path.join(plot_dir, filename)
    
    # Save figure
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    console.print(f"[green]Saved full results plot to: {filepath}[/green]")

def main():
    """Run experiments and plot results."""
    console.print("[bold blue]Starting DPRO vs MEU Analysis[/bold blue]")
    
    # Ensure plot directories exist
    ensure_plot_directory("out_of_sample")
    ensure_plot_directory("full_results")
    
    # Run risk aversion experiments
    console.print("\n[bold]Running Risk Aversion Experiments[/bold]")
    results = run_risk_aversion_experiment()
    
    # Run realization experiments with different parameters
    console.print("\n[bold]Running Realization Experiments[/bold]")
    
    # List of parameters to try
    epsilons = [0.01, 0.1, 1.0, 10.0]
    n_samples_list = [5, 10, 20, 50, 100, 150]  # Added smaller sample sizes
    n_people_list = [50, 100, 150]   # For evaluating policies
    
    for eps in epsilons:
        for n_samples in n_samples_list:
            for n_people in n_people_list:
                console.print(f"\nRunning with ε={eps}, n_samples={n_samples}, n_people={n_people}")
                realization_results, params = run_realization_experiment(
                    n_samples=n_samples,
                    n_people=n_people,
                    epsilon=eps
                )
                
                # Plot results
                console.print(f"\n[bold]Generating Plots for ε={eps}, n_samples={n_samples}, n_people={n_people}[/bold]")
                plot_results(results, realization_results, params)
                save_out_of_sample_plot(realization_results, params)
    
    console.print("[green]All experiments completed successfully![/green]")
    console.print("\n[bold]Plot directories:[/bold]")
    console.print(f"Full results: {os.path.abspath('plots/full_results')}")
    console.print(f"Out-of-sample comparisons: {os.path.abspath('plots/out_of_sample')}")

if __name__ == '__main__':
    main() 