"""Utility functions for the DPRO experiments."""
import numpy as np
from typing import Dict, List, Tuple
from configs.large_config import X_SPACE_SIZE

def calculate_utility(x: float, gamma: float) -> float:
    """Calculate utility for a given outcome and risk aversion parameter.
    
    Args:
        x: Outcome value
        gamma: Risk aversion parameter
    
    Returns:
        Utility value
    """
    # Scale down x to prevent extreme values
    #x_scaled = x / X_SPACE_SIZE
    
    # if gamma == 1:
    #     return 0.1 * np.log(x_scaled)  # Scale log utility
    
    # # Base utility with proper scaling
    #base_utility = (x_scaled ** (1 - gamma)) / (1 - gamma)
    #base_utility = 1 - np.exp(-gamma * x)
    base_utility = 1 - np.exp(-gamma * x) 
    return base_utility 

def generate_risk_parameters(n_agents: int, seed: int = None) -> np.ndarray:
    """Generate random risk aversion parameters.
    
    Args:
        n_agents: Number of agents (size of Ω)
        seed: Random seed for reproducibility
    
    Returns:
        Array of risk aversion parameters
    """
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(0, 1, size=n_agents)

def calculate_meu_gamma(gamma_omega: np.ndarray, phi_hat: np.ndarray) -> float:
    """Calculate γ_MEU as weighted average of γ_ω values.
    
    Args:
        gamma_omega: Array of risk aversion parameters
        phi_hat: Empirical distribution over Ω
    
    Returns:
        MEU risk aversion parameter
    """
    return np.sum(phi_hat * gamma_omega)

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Calculate KL divergence between two distributions.
    
    Args:
        p: First distribution
        q: Second distribution
    
    Returns:
        KL divergence value
    """
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    p = p + eps
    q = q + eps
    p = p / np.sum(p)
    q = q / np.sum(q)
    return np.sum(p * np.log(p / q))

def convert_act_to_matrix(act: Dict[int, Dict[int, float]], 
                         n_x: int, n_s: int) -> np.ndarray:
    """Convert AA act from dictionary format to matrix format.
    
    Args:
        act: AA act in dictionary format
        n_x: Size of outcome space
        n_s: Size of material space
    
    Returns:
        Matrix representation of AA act
    """
    matrix = np.zeros((n_x, n_s))
    for s in range(1, n_s + 1):
        for x in range(1, n_x + 1):
            matrix[x-1, s-1] = act[s][x]
    return matrix

def verify_distribution_constraints(act_matrix: np.ndarray) -> bool:
    """Verify that AA act satisfies distribution constraints.
    
    Args:
        act_matrix: Matrix representation of AA act
    
    Returns:
        True if constraints are satisfied, False otherwise
    """
    # Check if probabilities sum to 1 for each state
    return np.allclose(np.sum(act_matrix, axis=0), 1.0) and \
           np.all(act_matrix >= 0) and \
           np.all(act_matrix <= 1) 

def compute_c_omega_m(gamma_omega, comparison_acts_matrix):
    """
    Compute the reference utility c_omega_m for each mental state and comparison act.

    Args:
        gamma_omega: Array of risk aversion parameters for each mental state (n_omega,)
        comparison_acts_matrix: List of comparison act matrices (each of shape (n_x, n_s))

    Returns:
        c_omega_m: 2D numpy array of shape (n_omega, m), where each entry is the mean utility
                   of comparison act m under mental state omega (averaged over all material states)
    """
    n_omega = len(gamma_omega)
    m = len(comparison_acts_matrix)
    c_omega_m = np.zeros((n_omega, m))
    for omega in range(n_omega):
        for m_idx, act in enumerate(comparison_acts_matrix):
            # Compute the mean utility of act m under mental state omega
            utilities = []
            for s in range(act.shape[1]):
                outcome_probs = act[:, s]
                util = np.sum([calculate_utility(x+1, gamma_omega[omega]) * outcome_probs[x] for x in range(act.shape[0])])
                utilities.append(util)
            c_omega_m[omega, m_idx] = np.min(utilities)
    return c_omega_m