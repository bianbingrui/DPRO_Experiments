import numpy as np
from typing import Dict, Tuple

def create_reference_distribution(num_states: int, distribution_type: str = 'uniform') -> np.ndarray:
    """
    Create a reference distribution over states.
    
    Args:
        num_states: Number of states
        distribution_type: Type of distribution ('uniform' or 'normal')
        
    Returns:
        Reference distribution array of shape (num_states,)
    """
    if distribution_type == 'uniform':
        return np.ones(num_states) / num_states
    elif distribution_type == 'normal':
        x = np.linspace(-2, 2, num_states)
        pdf = np.exp(-x**2/2)
        return pdf / pdf.sum()
    else:
        raise ValueError(f"Unknown distribution type: {distribution_type}")

def create_endowments(num_states: int, num_outcomes: int, endowment_type: str = 'linear') -> np.ndarray:
    """
    Create endowments for each state-outcome pair.
    
    Args:
        num_states: Number of states
        num_outcomes: Number of outcomes
        endowment_type: Type of endowment structure ('linear' or 'quadratic')
        
    Returns:
        Endowment array of shape (num_states, num_outcomes)
    """
    if endowment_type == 'linear':
        # Linear endowments increasing with state and outcome
        return np.outer(np.linspace(1, 2, num_states), np.linspace(1, 2, num_outcomes))
    elif endowment_type == 'quadratic':
        # Quadratic endowments
        x = np.linspace(-1, 1, num_states)
        y = np.linspace(-1, 1, num_outcomes)
        X, Y = np.meshgrid(x, y)
        return 1 + X**2 + Y**2
    else:
        raise ValueError(f"Unknown endowment type: {endowment_type}")

def create_baseline_policies(num_states: int, num_outcomes: int, policy_type: str = 'deterministic') -> np.ndarray:
    """
    Create baseline policies (AA acts).
    
    Args:
        num_states: Number of states
        num_outcomes: Number of outcomes
        policy_type: Type of policies ('deterministic', 'uniform', or 'mixed')
        
    Returns:
        Array of baseline policies of shape (num_policies, num_states, num_outcomes)
    """
    if policy_type == 'deterministic':
        # Create deterministic policies mapping each state to different outcomes
        num_policies = num_outcomes
        policies = np.zeros((num_policies, num_states, num_outcomes))
        for i in range(num_policies):
            policies[i, :, i] = 1.0
        return policies
    
    elif policy_type == 'uniform':
        # Create a single uniform policy
        return np.ones((1, num_states, num_outcomes)) / num_outcomes
    
    elif policy_type == 'mixed':
        # Create a mix of deterministic and uniform policies
        det_policies = create_baseline_policies(num_states, num_outcomes, 'deterministic')
        unif_policy = create_baseline_policies(num_states, num_outcomes, 'uniform')
        return np.concatenate([det_policies, unif_policy], axis=0)
    
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")

def create_ambiguity_set_params(
    reference_distribution: np.ndarray,
    epsilon: float = 0.1,
    ambiguity_type: str = 'kl_divergence'
) -> Dict:
    """
    Create parameters for ambiguity set.
    
    Args:
        reference_distribution: Reference distribution over states
        epsilon: Size parameter for ambiguity set
        ambiguity_type: Type of ambiguity set ('kl_divergence' or 'wasserstein')
        
    Returns:
        Dictionary of ambiguity set parameters
    """
    return {
        'type': ambiguity_type,
        'epsilon': epsilon,
        'reference_distribution': reference_distribution
    }

def create_confidence_bounds(
    num_states: int,
    num_agents: int,
    confidence_level: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create confidence bounds for the ambiguity set.
    
    Args:
        num_states: Number of material states
        num_agents: Number of agents
        confidence_level: Confidence level for the bounds
        
    Returns:
        Tuple of (lower_bounds, upper_bounds) arrays of shape (num_states, num_agents)
    """
    # For uniform distribution, bounds are symmetric
    ref_dist = create_reference_distribution(num_states, 'uniform')
    margin = (1 - confidence_level) / 2
    lower_bounds = np.maximum(ref_dist - margin, 0)
    upper_bounds = np.minimum(ref_dist + margin, 1)
    return lower_bounds, upper_bounds

def create_kl_divergence_set(
    reference_distribution: np.ndarray,
    epsilon: float
) -> Dict:
    """
    Create parameters for a KL divergence ambiguity set.
    
    Args:
        reference_distribution: Reference distribution
        epsilon: Size parameter for the ambiguity set
        
    Returns:
        Dictionary of parameters for the ambiguity set
    """
    return {
        'type': 'kl_divergence',
        'epsilon': epsilon,
        'reference_distribution': reference_distribution
    }

def create_wasserstein_set(
    reference_distribution: np.ndarray,
    radius: float
) -> Dict:
    """
    Create parameters for a Wasserstein distance ambiguity set.
    
    Args:
        reference_distribution: Reference distribution
        radius: Radius of the Wasserstein ball
        
    Returns:
        Dictionary of parameters for the ambiguity set
    """
    return {
        'type': 'wasserstein',
        'radius': radius,
        'reference_distribution': reference_distribution
    } 