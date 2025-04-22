import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import cvxpy as cp

from dpro_experiments.models.dpro import DPROModel
from dpro_experiments.models.meu import MEUModel
from dpro_experiments.utils.utilities import (
    create_utility_functions,
    create_reference_distribution,
    create_endowments
)

def create_baseline_policies(num_states: int, num_outcomes: int) -> np.ndarray:
    """
    Create baseline policies for the social welfare problem.
    Each policy is a probability distribution over outcomes for each state.
    """
    num_policies = 3  # Number of baseline policies
    policies = np.zeros((num_policies, num_states, num_outcomes))
    
    # Policy 1: Equal distribution
    policies[0] = np.ones((num_states, num_outcomes)) / num_outcomes
    
    # Policy 2: Favor higher outcomes
    for s in range(num_states):
        policies[1, s] = np.linspace(0.1, 0.3, num_outcomes)
        policies[1, s] /= np.sum(policies[1, s])
    
    # Policy 3: Favor lower outcomes
    for s in range(num_states):
        policies[2, s] = np.linspace(0.3, 0.1, num_outcomes)
        policies[2, s] /= np.sum(policies[2, s])
    
    return policies

def create_reference_distribution(num_states: int) -> np.ndarray:
    """Create a reference distribution over states."""
    return np.ones(num_states) / num_states

def create_endowments(num_states: int, num_outcomes: int) -> np.ndarray:
    """Create endowments for each state-outcome pair."""
    endowments = np.zeros((num_states, num_outcomes))
    for s in range(num_states):
        endowments[s] = np.linspace(1, 5, num_outcomes)  # Outcomes from 1 to 5
    return endowments

def evaluate_social_welfare(
    policy_weights: np.ndarray,
    utility_functions: List[callable],
    state_distribution: np.ndarray,
    endowments: np.ndarray
) -> float:
    """
    Evaluate total social welfare for a given policy and state distribution.
    
    Args:
        policy_weights: Weights for policy outcomes
        utility_functions: List of utility functions for each agent type
        state_distribution: Distribution over material states
        endowments: Endowments for each state-outcome pair
        
    Returns:
        Total social welfare
    """
    num_agents = len(utility_functions)
    num_states, num_outcomes = endowments.shape
    
    # Compute utilities for each agent, state, and outcome
    agent_utilities = np.zeros((num_agents, num_states))
    for i, utility_fn in enumerate(utility_functions):
        for s in range(num_states):
            expected_utility = 0
            for o in range(num_outcomes):
                utility = utility_fn(endowments[s, o])
                expected_utility += policy_weights[o] * utility
            agent_utilities[i, s] = expected_utility
    
    # Compute expected utility over states for each agent
    agent_expected_utilities = np.sum(
        agent_utilities * state_distribution[np.newaxis, :],
        axis=1
    )
    
    # Return total social welfare (sum of utilities)
    return np.sum(agent_expected_utilities)

def run_comparison_experiment(
    num_agent_types: int = 3,
    num_material_states: int = 5,
    num_outcomes: int = 4,
    risk_aversion_range: Tuple[float, float] = (0.2, 0.8),
    epsilon: float = 0.1,
    num_samples: int = 1000
) -> Dict:
    """
    Run comparison experiment between DPRO and MEU models.
    
    Args:
        num_agent_types: Number of agent types
        num_material_states: Number of material states
        num_outcomes: Number of outcomes
        risk_aversion_range: Range of risk aversion parameters
        epsilon: Size of ambiguity set
        num_samples: Number of sampled distributions to evaluate
        
    Returns:
        Dictionary containing experiment results
    """
    # Create utility functions with different risk aversion parameters
    utility_functions = create_utility_functions(num_agent_types, risk_aversion_range)
    
    # Create problem components
    reference_distribution = create_reference_distribution(num_agent_types, num_material_states)
    endowments = create_endowments(num_material_states, num_outcomes)
    
    # Set up ambiguity set parameters
    ambiguity_set_params = {
        'type': 'kl_divergence',
        'epsilon': epsilon
    }
    
    # Solve DPRO model
    dpro_model = DPROModel(
        num_agent_types=num_agent_types,
        num_material_states=num_material_states,
        num_outcomes=num_outcomes,
        utility_functions=utility_functions,
        reference_distribution=reference_distribution,
        endowments=endowments,
        ambiguity_set_params=ambiguity_set_params
    )
    dpro_result = dpro_model.solve()
    
    # Solve MEU model (using average risk aversion)
    avg_risk_aversion = np.mean(np.linspace(risk_aversion_range[0], risk_aversion_range[1], num_agent_types))
    meu_utility = create_utility_functions(1, (avg_risk_aversion, avg_risk_aversion))[0]
    
    meu_model = MEUModel(
        num_material_states=num_material_states,
        num_outcomes=num_outcomes,
        utility_function=meu_utility,
        reference_distribution=reference_distribution.mean(axis=0),  # Average over agent types
        endowments=endowments,
        ambiguity_set_params=ambiguity_set_params
    )
    meu_result = meu_model.solve()
    
    # Generate sample distributions and evaluate utilities
    dpro_welfares = []
    meu_welfares = []
    
    for _ in range(num_samples):
        # Generate random distribution over states
        state_dist = np.random.dirichlet(np.ones(num_material_states))
        
        # Evaluate DPRO policy
        dpro_welfare = evaluate_social_welfare(
            dpro_result['policy_weights'],
            utility_functions,
            state_dist,
            endowments
        )
        dpro_welfares.append(dpro_welfare)
        
        # Evaluate MEU policy
        meu_welfare = evaluate_social_welfare(
            meu_result['policy_weights'],
            utility_functions,
            state_dist,
            endowments
        )
        meu_welfares.append(meu_welfare)
    
    return {
        'dpro': {
            'result': dpro_result,
            'welfares': dpro_welfares,
            'mean_welfare': np.mean(dpro_welfares),
            'std_welfare': np.std(dpro_welfares)
        },
        'meu': {
            'result': meu_result,
            'welfares': meu_welfares,
            'mean_welfare': np.mean(meu_welfares),
            'std_welfare': np.std(meu_welfares)
        },
        'parameters': {
            'num_agent_types': num_agent_types,
            'num_material_states': num_material_states,
            'num_outcomes': num_outcomes,
            'risk_aversion_range': risk_aversion_range,
            'epsilon': epsilon,
            'num_samples': num_samples
        }
    }

def plot_results(results: Dict):
    """Plot comparison of DPRO and MEU welfare distributions."""
    dpro_welfares = results['dpro']['welfares']
    meu_welfares = results['meu']['welfares']
    
    plt.figure(figsize=(10, 6))
    plt.hist(dpro_welfares, bins=50, alpha=0.5, label='DPRO')
    plt.hist(meu_welfares, bins=50, alpha=0.5, label='MEU')
    plt.axvline(results['dpro']['mean_welfare'], color='blue', linestyle='--', label='DPRO Mean')
    plt.axvline(results['meu']['mean_welfare'], color='orange', linestyle='--', label='MEU Mean')
    plt.xlabel('Total Social Welfare')
    plt.ylabel('Frequency')
    plt.title('Distribution of Social Welfare: DPRO vs MEU')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Run experiment with default parameters
    results = run_comparison_experiment()
    
    # Print summary statistics
    print("\nDPRO Results:")
    print(f"Mean Welfare: {results['dpro']['mean_welfare']:.4f}")
    print(f"Std Welfare: {results['dpro']['std_welfare']:.4f}")
    
    print("\nMEU Results:")
    print(f"Mean Welfare: {results['meu']['mean_welfare']:.4f}")
    print(f"Std Welfare: {results['meu']['std_welfare']:.4f}")
    
    # Plot results
    plot_results(results) 