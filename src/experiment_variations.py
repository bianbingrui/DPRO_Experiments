"""Run experiments comparing DPRO and MEU social welfare with varying sample sizes."""
import numpy as np
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import os

from src.experiment import DPROExperiment
from src.config import RANDOM_SEED, X_SPACE_SIZE, S_SPACE_SIZE, BASIC_AA_ACTS
from src.utils import generate_risk_parameters, calculate_utility, convert_act_to_matrix

def evaluate_welfare(z: np.ndarray, true_dist: np.ndarray, gamma: float) -> float:
    """Calculate total social welfare for a given solution.
    
    Args:
        z: Solution weights for basic acts
        true_dist: True distribution over states
        gamma: Risk aversion parameter
    
    Returns:
        Total welfare value
    """
    # Convert basic AA acts to matrices
    basic_acts = [convert_act_to_matrix(BASIC_AA_ACTS[i+1], X_SPACE_SIZE, S_SPACE_SIZE) 
                 for i in range(len(z))]
    
    # Calculate welfare
    welfare = 0
    for s in range(S_SPACE_SIZE):
        state_welfare = 0
        for i, act_weight in enumerate(z):
            act = basic_acts[i]
            for x in range(X_SPACE_SIZE):
                state_welfare += act_weight * act[x,s] * calculate_utility(x+1, gamma)
        welfare += true_dist[s] * state_welfare
    
    return welfare

def save_comparison_plot(sample_sizes: range, dpro_welfare: list, meu_welfare: list, filename: str = "Comparison_result.png"):
    """Save comparison plot to a file.
    
    Args:
        sample_sizes: Range of sample sizes
        dpro_welfare: List of DPRO welfare values
        meu_welfare: List of MEU welfare values
        filename: Name of the output file
    """
    plt.figure(figsize=(10, 6))
    
    # Plot scatter points
    plt.scatter(sample_sizes, dpro_welfare, color='blue', alpha=0.6, label='DPRO')
    plt.scatter(sample_sizes, meu_welfare, color='red', alpha=0.6, label='MEU')
    
    # Add trend lines
    z_dpro = np.polyfit(sample_sizes, dpro_welfare, 1)
    z_meu = np.polyfit(sample_sizes, meu_welfare, 1)
    p_dpro = np.poly1d(z_dpro)
    p_meu = np.poly1d(z_meu)
    plt.plot(sample_sizes, p_dpro(sample_sizes), 'b--', alpha=0.3)
    plt.plot(sample_sizes, p_meu(sample_sizes), 'r--', alpha=0.3)
    
    # Customize plot
    plt.xlabel('Number of Samples')
    plt.ylabel('Total Social Welfare')
    plt.title('DPRO vs MEU Performance with Varying Sample Sizes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Save plot
    plt.savefig(os.path.join('results', filename), dpi=300, bbox_inches='tight')
    plt.close()

def run_realization_experiment(max_samples: int = 100, num_trials: int = 5):
    """Run experiment with varying numbers of samples from real distribution.
    
    Args:
        max_samples: Maximum number of samples to test
        num_trials: Number of trials for each sample size
    """
    # True distribution parameters (this would be the real distribution)
    true_dist = np.array([0.6, 0.4])  # Example true distribution
    gamma = 0.5  # Risk aversion parameter
    
    # Storage for results
    sample_sizes = range(1, max_samples + 1)
    dpro_welfare = []
    meu_welfare = []
    
    print("\nRunning realization experiments...")
    
    for n_samples in sample_sizes:
        dpro_trial_welfare = []
        meu_trial_welfare = []
        
        for trial in range(num_trials):
            # Generate samples from true distribution
            samples = np.random.choice(2, size=n_samples, p=true_dist)
            empirical_dist = np.bincount(samples, minlength=2) / n_samples
            
            # Run DPRO and MEU with empirical distribution
            experiment = DPROExperiment(seed=RANDOM_SEED + trial)
            results = experiment.run_experiment()
            
            # Calculate welfare using true distribution
            if results['dpro_z'] is not None and results['meu_z'] is not None:
                dpro_welfare_value = evaluate_welfare(results['dpro_z'], true_dist, gamma)
                meu_welfare_value = evaluate_welfare(results['meu_z'], true_dist, gamma)
                
                dpro_trial_welfare.append(dpro_welfare_value)
                meu_trial_welfare.append(meu_welfare_value)
        
        # Average welfare across trials
        dpro_welfare.append(np.mean(dpro_trial_welfare))
        meu_welfare.append(np.mean(meu_trial_welfare))
        
        if n_samples % 10 == 0:
            print(f"Completed {n_samples}/{max_samples} samples")
    
    # Save results plot
    save_comparison_plot(sample_sizes, dpro_welfare, meu_welfare)
    print(f"\nResults saved to results/Comparison_result.png")

if __name__ == "__main__":
    run_realization_experiment(max_samples=100, num_trials=5) 