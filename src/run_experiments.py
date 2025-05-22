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
import scipy.stats as stats

from configs.large_config import (
    X_SPACE_SIZE, OMEGA_SPACE_SIZE, S_SPACE_SIZE,
    NUM_BASIC_ACTS, BASIC_AA_ACTS, COMPARISON_AA_ACTS,
    RISK_PROFILES, TRUE_DISTRIBUTION_MENTAL, TRUE_DISTRIBUTION_MATERIAL, MAX_SAMPLES,
    NUM_TRIALS, RANDOM_SEED
)
from src.utils import (
    generate_risk_parameters,
    calculate_meu_gamma,
    calculate_utility,
    convert_act_to_matrix,
    compute_c_omega_m
)
from src.optimization_funcs import solve_dpro, solve_meu, solve_nr

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
        nr_value: SEU optimal value
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
    
    # Add rows to SEU table
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
        "SEU",
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
    
    console.print("\n[green]SEU Weights:[/green]")
    for i, weight in enumerate(nr_z):
        bar_width = int(weight * max_bar_width)
        bar = "█" * bar_width
        console.print(f"r_{i+1:<2}: {bar} {weight:.4f}")

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
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot data
    evaluation_indices = np.arange(len(realization_results['dpro_values']))
    
    # Calculate means and confidence intervals (95%)
    dpro_mean = np.mean(realization_results['dpro_values'])
    meu_mean = np.mean(realization_results['meu_values'])
    nr_mean = np.mean(realization_results['nr_values'])
    
    dpro_std = np.std(realization_results['dpro_values'])
    meu_std = np.std(realization_results['meu_values'])
    nr_std = np.std(realization_results['nr_values'])
    
    # 95% confidence interval
    confidence_level = 0.95
    z_score = 1.96  # for 95% confidence
    n = len(evaluation_indices)
    
    dpro_ci = z_score * dpro_std / np.sqrt(n)
    meu_ci = z_score * meu_std / np.sqrt(n)
    nr_ci = z_score * nr_std / np.sqrt(n)
    
    # Plot individual points with transparency
    ax.scatter(evaluation_indices, realization_results['dpro_values'],
              marker='*', color='blue', s=50, alpha=0.2, label='DPRO samples')
    ax.scatter(evaluation_indices, realization_results['meu_values'],
              marker='^', color='red', s=50, alpha=0.2, label='MEU samples')
    ax.scatter(evaluation_indices, realization_results['nr_values'],
              marker='o', color='green', s=50, alpha=0.2, label='SEU samples')
    
    # Plot mean lines
    ax.axhline(y=dpro_mean, color='blue', linestyle='-', alpha=0.8, 
               label=f'DPRO mean: {dpro_mean:.4f} ± {dpro_ci:.4f}')
    ax.axhline(y=meu_mean, color='red', linestyle='-', alpha=0.8,
               label=f'MEU mean: {meu_mean:.4f} ± {meu_ci:.4f}')
    ax.axhline(y=nr_mean, color='green', linestyle='-', alpha=0.8,
               label=f'SEU mean: {nr_mean:.4f} ± {nr_ci:.4f}')
    
    # Add confidence intervals as shaded regions
    ax.fill_between(evaluation_indices, 
                   dpro_mean - dpro_ci, dpro_mean + dpro_ci,
                   color='blue', alpha=0.2)
    ax.fill_between(evaluation_indices,
                   meu_mean - meu_ci, meu_mean + meu_ci,
                   color='red', alpha=0.2)
    ax.fill_between(evaluation_indices,
                   nr_mean - nr_ci, nr_mean + nr_ci,
                   color='green', alpha=0.2)
    
    # Labels and title
    ax.set_xlabel('Test Realization Index')
    ax.set_ylabel('Social Welfare')
    title = 'Out-of-sample Performance Comparison\n'
    title += f'ε={params["epsilon"]}, n_samples={params["n_samples"]}, n_people={params["n_people"]}'
    ax.set_title(title)
    
    # Adjust legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add statistical significance tests
    stats_text = f'\nStatistical Tests ({confidence_level*100}% CI):\n'
    
    # T-test between DPRO and MEU
    t_stat_dpro_meu, p_val_dpro_meu = stats.ttest_ind(
        realization_results['dpro_values'],
        realization_results['meu_values']
    )
    stats_text += f'DPRO vs MEU: p={p_val_dpro_meu:.4e}\n'
    
    # T-test between DPRO and SEU
    t_stat_dpro_nr, p_val_dpro_nr = stats.ttest_ind(
        realization_results['dpro_values'],
        realization_results['nr_values']
    )
    stats_text += f'DPRO vs SEU: p={p_val_dpro_nr:.4e}\n'
    
    # Add performance improvement percentages
    dpro_vs_meu = ((dpro_mean - meu_mean) / abs(meu_mean)) * 100
    dpro_vs_nr = ((dpro_mean - nr_mean) / abs(nr_mean)) * 100
    
    stats_text += f'\nPerformance Improvement:\n'
    stats_text += f'DPRO vs MEU: {dpro_vs_meu:+.2f}%\n'
    stats_text += f'DPRO vs SEU: {dpro_vs_nr:+.2f}%'
    
    # Add stats text to plot
    ax.text(1.05, 0.5, stats_text,
            transform=ax.transAxes,
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Create filename with parameters and ensure plot directory exists
    plot_dir = ensure_plot_directory("out_of_sample")
    filename = f'comparison_eps{params["epsilon"]}_samples{params["n_samples"]}_npeople{params["n_people"]}.png'
    filepath = os.path.join(plot_dir, filename)
    
    # Save figure with adjusted layout
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    console.print(f"[green]Saved out-of-sample plot to: {filepath}[/green]")

def plot_results(results: Dict[str, np.ndarray], realization_results: Dict[str, np.ndarray], params: Dict[str, Any]):
    """Plot experiment results with confidence intervals."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 20))
    fig.suptitle(f'DPRO vs MEU Analysis (ε={params["epsilon"]}, trials={params["n_samples"]}, n_people={params["n_people"]})', fontsize=16)

    # Plot 1: Risk aversion vs solution weights for safe acts (first half)
    ax = axes[0, 0]
    n_safe_acts = NUM_BASIC_ACTS // 2
    # Handle 1D arrays (single experiment)
    safe_acts_weight_dpro = np.sum(results['dpro_weights'][:n_safe_acts])
    safe_acts_weight_meu = np.sum(results['meu_weights'][:n_safe_acts])
    # For single experiment, just plot a single point
    ax.scatter([results['gamma_omega'][0]], [safe_acts_weight_dpro], marker='*', color='blue', s=100, label='DPRO (Safe Acts)', alpha=0.6)
    ax.scatter([results['gamma_omega'][0]], [safe_acts_weight_meu], marker='^', color='red', s=100, label='MEU (Safe Acts)', alpha=0.6)
    ax.set_xlabel('Risk Aversion (γ)')
    ax.set_ylabel('Total Weight on Safe Acts')
    ax.set_title('Risk Aversion vs Safe Act Weights')
    ax.legend()

    # Plot 2: DPRO vs MEU optimal values
    ax = axes[0, 1]
    ax.scatter([results['dpro_value']], [results['meu_value']], marker='*', color='blue', s=100, alpha=0.6)
    ax.plot([min(results['dpro_value'], results['meu_value'])], [min(results['dpro_value'], results['meu_value'])], 'r--', label='y=x')
    ax.set_xlabel('DPRO Optimal Value')
    ax.set_ylabel('MEU Optimal Value')
    ax.set_title('DPRO vs MEU Optimal Values')
    ax.legend()

    # Plot 3: Weight distributions with adjusted bar width
    ax = axes[1, 0]
    x = np.arange(NUM_BASIC_ACTS)
    width = 0.3
    ax.bar(x - width/2, results['dpro_weights'], width, color='blue', label='DPRO', alpha=0.6)
    ax.bar(x + width/2, results['meu_weights'], width, color='red', label='MEU', alpha=0.6)
    ax.set_xlabel('Basic AA Act Index')
    ax.set_ylabel('Weight')
    ax.set_title('Weight Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels([f'r_{i+1}' for i in range(NUM_BASIC_ACTS)], rotation=45)
    ax.legend()

    # Plot 4: Risk aversion distribution (single point)
    ax = axes[1, 1]
    colors = ['blue', 'red', 'green', 'purple']
    for i in range(OMEGA_SPACE_SIZE):
        ax.scatter([results['gamma_omega'][i]], [1], color=colors[i], label=f'Mental State {i+1}')
    ax.set_xlabel('Risk Aversion (γ)')
    ax.set_ylabel('Density (single point)')
    ax.set_title('Risk Aversion (Single Experiment)')
    ax.legend()

    # Plot 5: Out-of-sample Performance Comparison with Confidence Intervals
    ax = axes[2, 0]
    evaluation_indices = np.arange(len(realization_results['dpro_values']))
    dpro_mean = np.mean(realization_results['dpro_values'])
    meu_mean = np.mean(realization_results['meu_values'])
    nr_mean = np.mean(realization_results['nr_values'])
    dpro_std = np.std(realization_results['dpro_values'])
    meu_std = np.std(realization_results['meu_values'])
    nr_std = np.std(realization_results['nr_values'])
    z_score = 1.96
    n = len(evaluation_indices)
    dpro_ci = z_score * dpro_std / np.sqrt(n)
    meu_ci = z_score * meu_std / np.sqrt(n)
    nr_ci = z_score * nr_std / np.sqrt(n)
    ax.scatter(evaluation_indices, realization_results['dpro_values'], marker='*', color='blue', s=50, alpha=0.2)
    ax.scatter(evaluation_indices, realization_results['meu_values'], marker='^', color='red', s=50, alpha=0.2)
    ax.scatter(evaluation_indices, realization_results['nr_values'], marker='o', color='green', s=50, alpha=0.2)
    ax.axhline(y=dpro_mean, color='blue', linestyle='-', alpha=0.8, label=f'DPRO: {dpro_mean:.4f} ± {dpro_ci:.4f}')
    ax.axhline(y=meu_mean, color='red', linestyle='-', alpha=0.8, label=f'MEU: {meu_mean:.4f} ± {meu_ci:.4f}')
    ax.axhline(y=nr_mean, color='green', linestyle='-', alpha=0.8, label=f'SEU: {nr_mean:.4f} ± {nr_ci:.4f}')
    ax.fill_between(evaluation_indices, dpro_mean - dpro_ci, dpro_mean + dpro_ci, color='blue', alpha=0.2)
    ax.fill_between(evaluation_indices, meu_mean - meu_ci, meu_mean + meu_ci, color='red', alpha=0.2)
    ax.fill_between(evaluation_indices, nr_mean - nr_ci, nr_mean + nr_ci, color='green', alpha=0.2)
    ax.set_xlabel('Evaluation Index')
    ax.set_ylabel('Social Welfare')
    ax.set_title('Out-of-sample Performance Comparison')
    ax.legend()

    # Plot 6: Performance difference histogram with confidence intervals
    ax = axes[2, 1]
    dpro_meu_diff = realization_results['dpro_values'] - realization_results['meu_values']
    dpro_nr_diff = realization_results['dpro_values'] - realization_results['nr_values']
    sns.histplot(data=dpro_meu_diff, color='red', alpha=0.5, label='DPRO-MEU', kde=True, ax=ax)
    sns.histplot(data=dpro_nr_diff, color='green', alpha=0.5, label='DPRO-SEU', kde=True, ax=ax)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.6)
    diff_mean_dpro_meu = np.mean(dpro_meu_diff)
    diff_mean_dpro_nr = np.mean(dpro_nr_diff)
    diff_ci_dpro_meu = z_score * np.std(dpro_meu_diff) / np.sqrt(n)
    diff_ci_dpro_nr = z_score * np.std(dpro_nr_diff) / np.sqrt(n)
    stats_text = f'Mean Differences (95% CI):\n'
    stats_text += f'DPRO-MEU: {diff_mean_dpro_meu:.4f} ± {diff_ci_dpro_meu:.4f}\n'
    stats_text += f'DPRO-SEU: {diff_mean_dpro_nr:.4f} ± {diff_ci_dpro_nr:.4f}\n'
    t_stat_meu, p_val_meu = stats.ttest_ind(realization_results['dpro_values'], realization_results['meu_values'])
    t_stat_nr, p_val_nr = stats.ttest_ind(realization_results['dpro_values'], realization_results['nr_values'])
    stats_text += f'\np-values:\n'
    stats_text += f'DPRO vs MEU: {p_val_meu:.4e}\n'
    stats_text += f'DPRO vs SEU: {p_val_nr:.4e}'
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_xlabel('Performance Difference')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Performance Differences')
    ax.legend()

    plot_dir = ensure_plot_directory("full_results")
    filename = f'analysis_eps{params["epsilon"]}_trials{params["n_samples"]}_npeople{params["n_people"]}.png'
    filepath = os.path.join(plot_dir, filename)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    console.print(f"[green]Saved full results plot to: {filepath}[/green]")

def sample_distribution(true_dist, n_samples):
    samples = np.random.choice(len(true_dist), size=n_samples, p=true_dist)
    freq = np.bincount(samples, minlength=len(true_dist)) / n_samples
    return freq

def solve_stage(n_mental_realization, n_material_realization, epsilon, seed=42):
    np.random.seed(seed)
    # 1. Sample phi_hat, psi_hat
    phi_hat = sample_distribution(TRUE_DISTRIBUTION_MENTAL, n_mental_realization)
    psi_hat = sample_distribution(TRUE_DISTRIBUTION_MATERIAL, n_material_realization)
    # 2. Prepare risk parameters
    gamma_omega = np.array(list(RISK_PROFILES.values()))
    gamma_meu = calculate_meu_gamma(gamma_omega, phi_hat)
    acts_matrix = [convert_act_to_matrix(act, X_SPACE_SIZE, S_SPACE_SIZE) for act in BASIC_AA_ACTS.values()]
    comparison_acts_matrix = [convert_act_to_matrix(act, X_SPACE_SIZE, S_SPACE_SIZE) for act in COMPARISON_AA_ACTS.values()]
    # 3. Solve for optimal solutions
    c_omega_m = np.zeros((OMEGA_SPACE_SIZE, len(comparison_acts_matrix)))
    dpro_z, dpro_value = solve_dpro(phi_hat, psi_hat, gamma_omega, acts_matrix, comparison_acts_matrix=comparison_acts_matrix, c_omega_m=c_omega_m, epsilon=epsilon)
    print("z*_DPRO:", dpro_z)
    meu_z, meu_value = solve_meu(phi_hat, psi_hat, gamma_meu, acts_matrix, epsilon=epsilon)
    print("z*_MEU:", meu_z)
    nr_z, nr_value = solve_nr(phi_hat, psi_hat, gamma_omega, acts_matrix)
    print("z*_NR:", nr_z)
    display_optimal_solution(dpro_z, meu_z, nr_z, dpro_value, meu_value, nr_value)
    return {
        'phi_hat': phi_hat,
        'psi_hat': psi_hat,
        'gamma_omega': gamma_omega,
        'gamma_meu': gamma_meu,
        'dpro_z': dpro_z,
        'meu_z': meu_z,
        'nr_z': nr_z,
        'dpro_value': dpro_value,
        'meu_value': meu_value,
        'nr_value': nr_value,
        'epsilon': epsilon
    }

def evaluate_stage(opt_solutions, n_trial=100, population=50, seed=42):
    np.random.seed(seed)
    phi_star = TRUE_DISTRIBUTION_MENTAL
    psi_star = TRUE_DISTRIBUTION_MATERIAL
    dpro_z, meu_z, nr_z = opt_solutions['dpro_z'], opt_solutions['meu_z'], opt_solutions['nr_z']
    gamma_omega = opt_solutions['gamma_omega']
    acts_matrix = [convert_act_to_matrix(act, X_SPACE_SIZE, S_SPACE_SIZE) for act in BASIC_AA_ACTS.values()]
    dpro_values, meu_values, nr_values = [], [], []
    for _ in range(n_trial):
        mental_states = np.random.choice(len(phi_star), size=population, p=phi_star)
        material_state = np.random.choice(len(psi_star), p=psi_star)
        dpro_welfare = 0
        meu_welfare = 0
        nr_welfare = 0
        for person in range(population):
            mental_state = mental_states[person]
            gamma = gamma_omega[mental_state]
            for i, act in enumerate(acts_matrix):
                outcome_probs = act[:, material_state]
                utilities = np.array([calculate_utility(x+1, gamma) for x in range(X_SPACE_SIZE)])
                dpro_welfare += dpro_z[i] * np.sum(utilities * outcome_probs)
                meu_welfare += meu_z[i] * np.sum(utilities * outcome_probs)
                nr_welfare += nr_z[i] * np.sum(utilities * outcome_probs)
        dpro_values.append(dpro_welfare/population)
        meu_values.append(meu_welfare/population)
        nr_values.append(nr_welfare/population)
    return {
        'dpro_values': np.array(dpro_values),
        'meu_values': np.array(meu_values),
        'nr_values': np.array(nr_values)
    }

def plot_trials_with_cummean_and_ci(results, opt_solutions, params):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import sem, t
    n_trial = len(results['dpro_values'])
    x = np.arange(1, n_trial+1)
    fig, ax = plt.subplots(figsize=(12, 7))
    # Plot all points
    ax.scatter(x, results['dpro_values'], color='blue', alpha=0.3, label='DPRO')
    ax.scatter(x, results['meu_values'], color='red', alpha=0.3, label='MEU')
    ax.scatter(x, results['nr_values'], color='green', alpha=0.3, label='SEU')
    
    # Plot polylines connecting each trial's result for each method
    ax.plot(x, results['dpro_values'], color='blue', alpha=0.5, linestyle='--', linewidth=1, label='DPRO Polyline')
    ax.plot(x, results['meu_values'], color='red', alpha=0.5, linestyle='dashdot', linewidth=1, label='MEU Polyline')
    ax.plot(x, results['nr_values'], color='green', alpha=0.5, linestyle='-.', linewidth=1, label='SEU Polyline')
    
    # Cumulative mean and CI
    for key, color, label in [('dpro_values','blue','DPRO'),('meu_values','red','MEU'),('nr_values','green','SEU')]:
        vals = results[key]
        cummean = np.cumsum(vals) / (np.arange(n_trial)+1)
        # 95% CI for mean at each point
        ci = np.array([t.interval(0.95, i, loc=cummean[i], scale=sem(vals[:i+1])) if i>0 else (cummean[i],cummean[i]) for i in range(n_trial)])
        lower = ci[:,0]
        upper = ci[:,1]
        ax.plot(x, cummean, color=color, label=f'{label} Mean', linewidth=2)
        ax.fill_between(x, lower, upper, color=color, alpha=0.15)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Sample Total Social Welfare')
    ax.set_title(f"Out-of-sample Performance (n_mental_samples={params['n_mental_realization']}, n_material_samples={params['n_material_realization']}, ε={params['epsilon']}, n_trial={params['n_trial']}, population={params['population']})")
    ax.legend()
    plt.tight_layout()
    plot_dir = ensure_plot_directory('out_of_sample')
    filename = f'out_of_sample_nmental{params["n_mental_realization"]}_nmaterial{params["n_material_realization"]}_eps{params["epsilon"]}_ntrial{params["n_trial"]}_pop{params["population"]}.png'
    plt.savefig(os.path.join(plot_dir, filename), dpi=300)
    plt.close()
    console.print(f"[green]Saved out-of-sample trial plot to: {os.path.join(plot_dir, filename)}[/green]")

def main():
    console.print("[bold blue]Starting DPRO/MEU/SEU Experiment (Refactored)[/bold blue]")
    ensure_plot_directory("out_of_sample")
    # Example parameter grid
    n_mental_list = [5, 50, 100]
    n_material_list = [5, 50, 100]
    epsilons = [0.01, 0.1, 1.0, 10.0]
    population = [50, 100]
    n_trial = [50, 100]
    for n_mental in n_mental_list:
        for n_material in n_material_list:
            for eps in epsilons:
                for trial in n_trial:
                    for pop in population:
                        console.print(f"\n[bold]Solving for n_mental={n_mental}, n_material={n_material}, ε={eps}[/bold]")
                        opt_solutions = solve_stage(n_mental, n_material, eps)
                        results = evaluate_stage(opt_solutions, n_trial=trial, population=pop)
                        params = {
                            'n_mental_realization': n_mental,
                            'n_material_realization': n_material,
                            'epsilon': eps,
                            'n_trial': trial,
                            'population': pop
                        }
                        plot_trials_with_cummean_and_ci(results, opt_solutions, params)
    console.print("[green]All experiments completed successfully![/green]")

if __name__ == '__main__':
    main() 