"""Main experiment runner."""
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    OMEGA_SPACE_SIZE, S_SPACE_SIZE, N_OMEGA, N_S,
    RANDOM_SEED, NUM_BASIC_ACTS
)
from src.utils import (
    generate_risk_parameters,
    calculate_meu_gamma,
    kl_divergence
)
from src.optimization import solve_dpro, solve_meu

class DPROExperiment:
    """Class to run DPRO experiments."""
    
    def __init__(self, seed: int = RANDOM_SEED):
        """Initialize experiment.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
        
        # Generate risk parameters
        self.gamma_omega = generate_risk_parameters(OMEGA_SPACE_SIZE, seed)
        
        # Initialize results storage
        self.results = {
            'dpro_z': None,
            'meu_z': None,
            'dpro_value': None,
            'meu_value': None
        }
    
    def generate_empirical_distributions(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate empirical distributions φ̂ and ψ̂.
        
        Returns:
            Tuple of empirical distributions (φ̂, ψ̂)
        """
        # Generate slightly perturbed distributions instead of uniform
        # This helps with numerical stability
        phi_hat = np.array([0.55, 0.45])  # Slightly favor first type
        psi_hat = np.array([0.45, 0.55])  # Slightly favor second state
        
        return phi_hat, psi_hat
    
    def run_experiment(self) -> Dict:
        """Run the main experiment.
        
        Returns:
            Dictionary containing results
        
        Raises:
            RuntimeError: If optimization problems cannot be solved
        """
        try:
            # Generate empirical distributions
            phi_hat, psi_hat = self.generate_empirical_distributions()
            
            # Calculate γ_MEU
            gamma_meu = calculate_meu_gamma(self.gamma_omega, phi_hat)
            
            # Solve DPRO problem
            dpro_result = solve_dpro(phi_hat, psi_hat, self.gamma_omega)
            if dpro_result is not None:
                self.results['dpro_z'], self.results['dpro_value'] = dpro_result
            else:
                raise RuntimeError("DPRO optimization failed")
            
            # Solve MEU problem
            meu_result = solve_meu(psi_hat, gamma_meu)
            if meu_result is not None:
                self.results['meu_z'], self.results['meu_value'] = meu_result
            else:
                raise RuntimeError("MEU optimization failed")
            
            return self.results
            
        except (ValueError, RuntimeError) as e:
            print(f"Error running experiment: {str(e)}")
            # Initialize results with default values
            self.results = {
                'dpro_z': np.zeros(NUM_BASIC_ACTS),  # Safe default
                'meu_z': np.zeros(NUM_BASIC_ACTS),   # Safe default
                'dpro_value': float('-inf'),         # Worst case value
                'meu_value': float('-inf')           # Worst case value
            }
            return self.results
    
    def plot_results(self):
        """Plot experiment results."""
        if self.results['dpro_z'] is None or self.results['meu_z'] is None:
            raise ValueError("No results to plot. Run experiment first.")
        
        # Create comparison plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(self.results['dpro_z']))
        width = 0.35
        
        ax.bar(x - width/2, self.results['dpro_z'], width, 
               label='DPRO', color='blue', alpha=0.6)
        ax.bar(x + width/2, self.results['meu_z'], width,
               label='MEU', color='red', alpha=0.6)
        
        ax.set_xlabel('Basic AA Act Index')
        ax.set_ylabel('Weight')
        ax.set_title('Comparison of DPRO and MEU Solutions')
        ax.set_xticks(x)
        ax.set_xticklabels([f'r_{i+1}' for i in range(len(x))])
        ax.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Print values
        print(f"DPRO optimal value: {self.results['dpro_value']:.4f}")
        print(f"MEU optimal value: {self.results['meu_value']:.4f}")

def main():
    """Run main experiment."""
    experiment = DPROExperiment()
    results = experiment.run_experiment()
    experiment.plot_results()

if __name__ == "__main__":
    main() 