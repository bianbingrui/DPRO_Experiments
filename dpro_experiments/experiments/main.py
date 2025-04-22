import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import cvxpy as cp

from dpro_experiments.models.dpro import DPROModel
from dpro_experiments.models.meu import MEUModel
from dpro_experiments.utils.utilities import create_power_utility
from dpro_experiments.utils.ambiguity_sets import (
    create_reference_distribution,
    create_endowments,
    create_baseline_policies,
    create_ambiguity_set_params
)

def evaluate_group_utility(
    policy_weights: np.ndarray,
    baseline_policies: np.ndarray,
    utility_functions: List[callable],
    state_distribution: np.ndarray,
    endowments: np.ndarray
) -> float:
    """
    Evaluate expected utility for the whole group.
    
    Args:
        policy_weights: Weights for baseline policies
        baseline_policies: Array of baseline policies (AA acts)
        utility_functions: List of utility functions for agents
        state_distribution: Distribution over material states
        endowments: Endowments for each state-outcome pair
        
    Returns:
        Expected utility for the group
    """
    num_states, num_outcomes = endowments.shape
    num_agents = len(utility_functions)
    
    # Compute policy-induced outcome distribution for each state
    outcome_distributions = np.sum(
        policy_weights[:, np.newaxis, np.newaxis] * baseline_policies,
        axis=0
    )
    
    # Compute utilities for each agent, state, and outcome
    agent_utilities = np.zeros((num_agents, num_states))
    for i, utility_fn in enumerate(utility_functions):
        for s in range(num_states):
            expected_utility = 0
            for o in range(num_outcomes):
                utility = utility_fn(endowments[s, o])
                expected_utility += outcome_distributions[s, o] * utility
            agent_utilities[i, s] = expected_utility
    
    # Compute expected utility over states for each agent
    agent_expected_utilities = np.sum(
        agent_utilities * state_distribution[np.newaxis, :],
        axis=1
    )
    
    # Return average utility across agents
    return np.mean(agent_expected_utilities)

def run_comparison_experiment(
    num_agents: int = 3,
    num_states: int = 5,
    num_outcomes: int = 4,
    risk_aversion: float = 0.5,
    epsilon: float = 0.1,
    num_samples: int = 1000
) -> Dict:
    """
    Run comparison experiment between DPRO and MEU models.
    
    Args:
        num_agents: Number of agents
        num_states: Number of states
        num_outcomes: Number of outcomes
        risk_aversion: Risk aversion parameter
        epsilon: Size of ambiguity set
        num_samples: Number of sampled distributions to evaluate
        
    Returns:
        Dictionary containing experiment results
    """
    # Create utility functions (power utility with different risk aversions)
    utility_functions = [
        create_power_utility(risk_aversion + 0.1 * i)
        for i in range(num_agents)
    ]
    
    # Create problem components
    reference_distribution = create_reference_distribution(num_states)
    endowments = create_endowments(num_states, num_outcomes, 'linear')
    baseline_policies = create_baseline_policies(num_states, num_outcomes, 'mixed')
    ambiguity_set_params = create_ambiguity_set_params(
        reference_distribution,
        epsilon,
        'kl_divergence'
    )
    
    # Solve DPRO model
    dpro_model = DPROModel(
        num_states=num_states,
        num_outcomes=num_outcomes,
        baseline_policies=baseline_policies,
        utility_function=utility_functions[0],  # Use first agent's utility
        reference_distribution=reference_distribution,
        endowments=endowments,
        ambiguity_set_params=ambiguity_set_params
    )
    dpro_result = dpro_model.solve()
    
    # Solve MEU model
    meu_model = MEUModel(
        num_states=num_states,
        num_outcomes=num_outcomes,
        baseline_policies=baseline_policies,
        utility_function=utility_functions[0],  # Use first agent's utility
        reference_distribution=reference_distribution,
        endowments=endowments,
        ambiguity_set_params=ambiguity_set_params
    )
    meu_result = meu_model.solve()
    
    # Generate sample distributions and evaluate utilities
    dpro_utilities = []
    meu_utilities = []
    
    for _ in range(num_samples):
        # Generate random distribution over states
        state_dist = np.random.dirichlet(np.ones(num_states))
        
        # Evaluate DPRO policy
        dpro_utility = evaluate_group_utility(
            dpro_result['policy_weights'],
            baseline_policies,
            utility_functions,
            state_dist,
            endowments
        )
        dpro_utilities.append(dpro_utility)
        
        # Evaluate MEU policy
        meu_utility = evaluate_group_utility(
            meu_result['policy_weights'],
            baseline_policies,
            utility_functions,
            state_dist,
            endowments
        )
        meu_utilities.append(meu_utility)
    
    return {
        'dpro': {
            'result': dpro_result,
            'utilities': dpro_utilities,
            'mean_utility': np.mean(dpro_utilities),
            'std_utility': np.std(dpro_utilities)
        },
        'meu': {
            'result': meu_result,
            'utilities': meu_utilities,
            'mean_utility': np.mean(meu_utilities),
            'std_utility': np.std(meu_utilities)
        },
        'parameters': {
            'num_agents': num_agents,
            'num_states': num_states,
            'num_outcomes': num_outcomes,
            'risk_aversion': risk_aversion,
            'epsilon': epsilon,
            'num_samples': num_samples
        }
    }

def plot_utility_distributions(results: Dict):
    """
    Plot distributions of utilities for DPRO and MEU policies.
    
    Args:
        results: Results from comparison experiment
    """
    plt.figure(figsize=(12, 6))
    
    # Plot histograms
    plt.hist(
        results['dpro']['utilities'],
        bins=30,
        alpha=0.5,
        label=f"DPRO (mean={results['dpro']['mean_utility']:.3f})",
        color='blue'
    )
    plt.hist(
        results['meu']['utilities'],
        bins=30,
        alpha=0.5,
        label=f"MEU (mean={results['meu']['mean_utility']:.3f})",
        color='red'
    )
    
    plt.xlabel('Group Utility')
    plt.ylabel('Frequency')
    plt.title('Distribution of Group Utilities under Different Policies')
    plt.legend()
    plt.grid(True)
    plt.show()

def generate_test_distributions(num_states: int, num_distributions: int = 5) -> np.ndarray:
    """
    Generate systematic test distributions over states.
    
    Args:
        num_states: Number of states
        num_distributions: Number of test distributions to generate
        
    Returns:
        Array of distributions of shape (num_distributions, num_states)
    """
    distributions = []
    
    # Uniform distribution
    distributions.append(np.ones(num_states) / num_states)
    
    # Peaked distributions at different states
    for peak_state in range(min(num_states, num_distributions - 1)):
        dist = np.ones(num_states) * 0.1 / (num_states - 1)
        dist[peak_state] = 0.9
        distributions.append(dist)
    
    # Fill remaining with random distributions if needed
    while len(distributions) < num_distributions:
        distributions.append(np.random.dirichlet(np.ones(num_states)))
    
    return np.array(distributions)

def parameter_study(
    num_agents_list: List[int] = [2, 3, 5],
    num_states_list: List[int] = [3, 5, 7],
    risk_aversion: float = 0.5,
    epsilon: float = 0.1
) -> Dict:
    """
    Study how performance varies with number of agents and states.
    
    Args:
        num_agents_list: List of number of agents to test
        num_states_list: List of number of states to test
        risk_aversion: Base risk aversion parameter
        epsilon: Size of ambiguity set
        
    Returns:
        Dictionary containing results for each parameter combination
    """
    results = {}
    
    for L in num_agents_list:
        for K in num_states_list:
            print(f"\nRunning experiment with {L} agents and {K} states...")
            
            # Run comparison experiment
            exp_results = run_comparison_experiment(
                num_agents=L,
                num_states=K,
                num_outcomes=4,  # Fixed as per problem setup
                risk_aversion=risk_aversion,
                epsilon=epsilon,
                num_samples=5  # Use systematic distributions
            )
            
            # Store results
            results[(L, K)] = exp_results
    
    return results

def plot_parameter_study(results: Dict):
    """
    Plot results from parameter study.
    
    Args:
        results: Results from parameter study
    """
    num_agents_list = sorted(list(set(L for L, _ in results.keys())))
    num_states_list = sorted(list(set(K for _, K in results.keys())))
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot effect of number of agents
    for K in num_states_list:
        dpro_means = [results[(L, K)]['dpro']['mean_utility'] for L in num_agents_list]
        meu_means = [results[(L, K)]['meu']['mean_utility'] for L in num_agents_list]
        
        axes[0].plot(num_agents_list, dpro_means, '-o', label=f'DPRO (K={K})')
        axes[0].plot(num_agents_list, meu_means, '--o', label=f'MEU (K={K})')
    
    axes[0].set_xlabel('Number of Agents (L)')
    axes[0].set_ylabel('Mean Group Utility')
    axes[0].set_title('Effect of Number of Agents')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot effect of number of states
    for L in num_agents_list:
        dpro_means = [results[(L, K)]['dpro']['mean_utility'] for K in num_states_list]
        meu_means = [results[(L, K)]['meu']['mean_utility'] for K in num_states_list]
        
        axes[1].plot(num_states_list, dpro_means, '-o', label=f'DPRO (L={L})')
        axes[1].plot(num_states_list, meu_means, '--o', label=f'MEU (L={L})')
    
    axes[1].set_xlabel('Number of States (K)')
    axes[1].set_ylabel('Mean Group Utility')
    axes[1].set_title('Effect of Number of States')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run parameter study
    study_results = parameter_study(
        num_agents_list=[2, 3, 5],
        num_states_list=[3, 5, 7],
        risk_aversion=0.5,
        epsilon=0.1
    )
    
    # Plot parameter study results
    plot_parameter_study(study_results)
    
    # Run detailed comparison for specific case
    detailed_results = run_comparison_experiment(
        num_agents=3,
        num_states=5,
        num_outcomes=4,
        risk_aversion=0.5,
        epsilon=0.1,
        num_samples=1000
    )
    
    # Print summary statistics
    print("\nDetailed Results for L=3, K=5:")
    print("\nDPRO Results:")
    print(f"Mean Utility: {detailed_results['dpro']['mean_utility']:.3f}")
    print(f"Std Utility: {detailed_results['dpro']['std_utility']:.3f}")
    
    print("\nMEU Results:")
    print(f"Mean Utility: {detailed_results['meu']['mean_utility']:.3f}")
    print(f"Std Utility: {detailed_results['meu']['std_utility']:.3f}")
    
    # Plot utility distributions for specific case
    plot_utility_distributions(detailed_results) 