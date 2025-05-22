"""Run numerical experiments with multiple trials for DPRO, MEU, and SEU analysis."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
from rich.console import Console
from rich.progress import Progress
import pandas as pd
from typing import Dict, Any
from scipy import stats
import os

from configs.large_config import (
    X_SPACE_SIZE, OMEGA_SPACE_SIZE, S_SPACE_SIZE,
    NUM_BASIC_ACTS, BASIC_AA_ACTS, COMPARISON_AA_ACTS,
    RISK_PROFILES, TRUE_DISTRIBUTION_MENTAL, TRUE_DISTRIBUTION_MATERIAL
)
from src.utils import (
    calculate_utility,
    convert_act_to_matrix,
    compute_c_omega_m,
    calculate_meu_gamma
)
from src.optimization_funcs import solve_dpro, solve_meu, solve_nr

console = Console()

raw_probs = np.random.rand(OMEGA_SPACE_SIZE)
probs = raw_probs / raw_probs.sum()
TRUE_DISTRIBUTION_MENTAL = probs

raw_probs_material = np.random.rand(S_SPACE_SIZE)
probs_material = raw_probs_material / raw_probs_material.sum()
TRUE_DISTRIBUTION_MATERIAL = probs_material

def sample_distribution(true_dist, n_samples):
    """Sample from a true distribution to get empirical distribution."""
    samples = np.random.choice(len(true_dist), size=n_samples, p=true_dist)
    freq = np.bincount(samples, minlength=len(true_dist)) / n_samples
    return freq

def run_single_trial(n_mental_realization: int, n_material_realization: int, 
                    epsilon: float, seed: int = None) -> Dict[str, float]:
    """Run a single trial of the experiment.
    Returns:
        Dictionary containing welfare values, objective values, and solution vectors for each method
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Sample empirical distributions
    phi_hat = sample_distribution(TRUE_DISTRIBUTION_MENTAL, n_mental_realization)
    psi_hat = sample_distribution(TRUE_DISTRIBUTION_MATERIAL, n_material_realization)
    
    # Prepare parameters
    gamma_omega = np.array(list(RISK_PROFILES.values()))
    #gamma_meu = calculate_meu_gamma(gamma_omega, phi_hat)
    gamma_meu = np.random.uniform(0, 1)
    
    # Convert acts to matrix form
    acts_matrix = [convert_act_to_matrix(act, X_SPACE_SIZE, S_SPACE_SIZE) 
                  for act in BASIC_AA_ACTS.values()]
    comparison_acts_matrix = [convert_act_to_matrix(act, X_SPACE_SIZE, S_SPACE_SIZE) 
                            for act in COMPARISON_AA_ACTS.values()]
    
    # Compute reference utilities
    c_omega_m = compute_c_omega_m(gamma_omega, comparison_acts_matrix)
    
    # Solve optimization problems
    dpro_z, dpro_obj = solve_dpro(phi_hat, psi_hat, gamma_omega, acts_matrix, 
                          comparison_acts_matrix=comparison_acts_matrix, 
                          c_omega_m=c_omega_m, epsilon=epsilon)
    meu_z, meu_obj = solve_meu(phi_hat, psi_hat, gamma_meu, acts_matrix, epsilon=epsilon)
    nr_z, nr_obj = solve_nr(phi_hat, psi_hat, gamma_omega, acts_matrix)
    # Evaluate solutions using true distribution
    dpro_welfare = evaluate_solution(dpro_z, gamma_omega, acts_matrix)
    meu_welfare = evaluate_solution(meu_z, gamma_omega, acts_matrix)
    nr_welfare = evaluate_solution(nr_z, gamma_omega, acts_matrix)
    
    return {
        'dpro_welfare': dpro_welfare,
        'meu_welfare': meu_welfare,
        'seu_welfare': nr_welfare,
        'dpro_obj': dpro_obj,
        'meu_obj': meu_obj,
        'nr_obj': nr_obj,
        'dpro_z': dpro_z,
        'meu_z': meu_z,
        'nr_z': nr_z
    }

def evaluate_solution(z: np.ndarray, gamma_omega: np.ndarray, 
                     acts_matrix: List[np.ndarray]) -> float:
    """Evaluate a solution using the true distribution.
    
    Args:
        z: Solution weights
        gamma_omega: Risk aversion parameters
        acts_matrix: List of act matrices
    
    Returns:
        Expected welfare under true distribution
    """
    n_omega = len(gamma_omega)
    n_s = acts_matrix[0].shape[1]
    total_welfare = 0
    
    for omega in range(n_omega):
        for s in range(n_s):
            # Calculate utility for each act
            act_utilities = []
            for act in acts_matrix:
                outcome_probs = act[:, s]
                utilities = np.array([calculate_utility(x+1, gamma_omega[omega]) 
                                    for x in range(X_SPACE_SIZE)])
                act_utilities.append(np.sum(utilities * outcome_probs))
            
            # Weighted sum of utilities
            welfare = np.sum(z * np.array(act_utilities))
            total_welfare += TRUE_DISTRIBUTION_MENTAL[omega] * TRUE_DISTRIBUTION_MATERIAL[s] * welfare
    
    return total_welfare * 100

def run_experiments(n_trials: int = 100, n_mental_realization: int = 50,
                   n_material_realization: int = 50, epsilon: float = 1,
                   seed: int = 42) -> Dict[str, np.ndarray]:
    """Run multiple trials of the experiment.
    Returns:
        Dictionary containing welfare values, objective values, and solution vectors for each method across all trials
    """
    np.random.seed(seed)
    results = {
        'dpro_welfare': [],
        'meu_welfare': [],
        'seu_welfare': [],
        'dpro_obj': [],
        'meu_obj': [],
        'nr_obj': [],
        'dpro_z': [],
        'meu_z': [],
        'nr_z': []
    }
    
    with Progress() as progress:
        task = progress.add_task("[cyan]Running trials...", total=n_trials)
        
        for i in range(n_trials):
            trial_results = run_single_trial(
                n_mental_realization, n_material_realization, epsilon, seed=i
            )
            for method in results:
                results[method].append(trial_results[method])
            progress.update(task, advance=1)
    
    # Convert lists to numpy arrays
    for method in results:
        results[method] = np.array(results[method])
    
    return results

def plot_results(results: Dict[str, np.ndarray],
                 params: Dict[str, Any],
                 save_path: Optional[str] = None):
    """
    Plot custom box‐plots with 95% CI clamped inside the sample range:
      - Box  = 95% CI of the mean, but limited to [min(x), max(x)]
      - Whiskers = true sample min/max
      - Median line drawn at the mean
      - No scatter points
      - Boxes centered exactly over their x‐tick
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats

    # 1) Theme
    sns.set_theme(style="ticks", context="talk")

    # 2) Methods and labels
    methods = ['dpro_welfare', 'meu_welfare', 'seu_welfare']
    labels  = [m.replace('_', ' ').upper() for m in methods]
    palette = sns.color_palette("pastel", n_colors=len(methods))

    # 3) Build stats_list for bxp, clamping CI inside [min, max]
    stats_list = []
    for m, lab in zip(methods, labels):
        vals     = np.array(results[m])
        mean_val = vals.mean()
        sem      = stats.sem(vals)
        ci_low, ci_high = stats.norm.interval(0.95, loc=mean_val, scale=sem)
        # clamp to sample range
        ci_low  = max(ci_low,  vals.min())
        ci_high = min(ci_high, vals.max())
        stats_list.append({
            'mean':   mean_val,
            'med':    mean_val,
            'q1':     ci_low,
            'q3':     ci_high,
            'whislo': vals.min(),
            'whishi': vals.max(),
            'label':  lab,
            'fliers': []    # suppress default outliers
        })

    # 4) Draw boxes
    fig, ax = plt.subplots(figsize=(8, 6))
    positions = list(range(len(methods)))
    bxp = ax.bxp(
        stats_list,
        showfliers=False,
        patch_artist=True,
        widths=0.6,
        positions=positions
    )

    # 5) Style boxes/whiskers/caps/medians
    for patch, color in zip(bxp['boxes'], palette):
        patch.set_facecolor(color)
        patch.set_edgecolor("gray")
        patch.set_alpha(0.7)
    for whisker in bxp['whiskers']:
        whisker.set_color("gray")
    for cap in bxp['caps']:
        cap.set_color("gray")
        cap.set_linewidth(1.5)
    for median in bxp['medians']:
        median.set_color("black")
        median.set_linewidth(2)

    # 6) Axis ticks & labels
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_xlim(-0.5, len(methods) - 0.5)
    ax.set_xlabel("Method", fontsize=14)
    ax.set_ylabel("Expected Welfare", fontsize=14)

    # 7) Title & grid
    ax.set_title(
        f"Comparison of Methods\n"
        f"(n_trials={params['n_trials']}, "
        f"n_mental={params['n_mental_realization']}, "
        f"n_material={params['n_material_realization']}, "
        f"ε={params['epsilon']})",
        fontsize=16, pad=12
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.xaxis.grid(False)
    sns.despine(trim=True)

    # # 8) p‐value annotation
    # p1 = stats.ttest_ind(results['dpro_welfare'], results['meu_welfare']).pvalue
    # p2 = stats.ttest_ind(results['dpro_welfare'], results['seu_welfare']).pvalue
    # ax.text(
    #     0.02, 0.98,
    #     f"DPRO vs MEU: p={p1:.2e}\nDPRO vs SEU: p={p2:.2e}",
    #     transform=ax.transAxes, va="top",
    #     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    #     fontsize=12
    # )

    # 9) Save & close
    plt.tight_layout()
    out = save_path or "method_comparison.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

def main():
    """Main function to run the experiments over parameter grid and save plots in 'out_of_smaple'."""
    # Parameter grids
    n_trials_list = [10, 50, 100]
    n_mental_realization_list = [5, 10, 50]
    n_material_realization_list = [20]
    epsilon_list = [0.01, 0.1, 1]
    seed = 42

    # Create output directory if it doesn't exist
    output_dir = 'out_of_smaple'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    import itertools
    for n_trials in n_trials_list:
        for n_mental_realization in n_mental_realization_list:
            for n_material_realization in n_material_realization_list:
                for epsilon in epsilon_list:
                    params = {
                        'n_trials': n_trials,
                        'n_mental_realization': n_mental_realization,
                        'n_material_realization': n_material_realization,
                        'epsilon': epsilon,
                        'seed': seed
                    }
                    console.print(f"\n[bold]Running: n_trials={n_trials}, n_mental={n_mental_realization}, n_material={n_material_realization}, epsilon={epsilon}[/bold]")
                    results = run_experiments(**params)
                    # Print summary statistics
                    console.print("\n[bold]Summary Statistics:[/bold]")
                    for method, values in results.items():
                        if 'welfare' in method:
                            mean = np.mean(values)
                            std = np.std(values)
                            ci = 1.96 * std / np.sqrt(len(values))
                            console.print(f"{method.upper().replace('_', ' ')}:")
                            console.print(f"  Mean: {mean:.4f}")
                            console.print(f"  Std:  {std:.4f}")
                            console.print(f"  95% CI: [{mean-ci:.4f}, {mean+ci:.4f}]")
                    # Save plot with parameterized filename
                    plot_filename = f"boxplot_trials{n_trials}_nmental{n_mental_realization}_nmaterial{n_material_realization}_eps{epsilon}.png"
                    plot_path = os.path.join(output_dir, plot_filename)
                    plot_results(results, params, save_path=plot_path)
    console.print("\n[green]All experiments completed! Plots saved in 'out_of_smaple' folder.[/green]")

if __name__ == '__main__':
    main() 
