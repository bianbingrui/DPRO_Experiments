import numpy as np
from typing import List, Callable
import cvxpy as cp

def create_utility_functions(
    num_agents: int,
    risk_aversion: float = 0.5,
    utility_type: str = 'power'
) -> List[callable]:
    """
    Create utility functions for each agent.
    
    Args:
        num_agents: Number of agents
        risk_aversion: Risk aversion parameter for power utility
        utility_type: Type of utility function ('power' or 'piecewise')
        
    Returns:
        List of utility functions, one for each agent
    """
    if utility_type == 'power':
        # Power utility: u(x) = x^(1-γ) / (1-γ) for γ ≠ 1, log(x) for γ = 1
        def power_utility(x):
            if risk_aversion == 1:
                return cp.log(x)
            else:
                return (x ** (1 - risk_aversion)) / (1 - risk_aversion)
        return [power_utility] * num_agents
    
    elif utility_type == 'piecewise':
        # Piecewise linear utility with kink at x=1
        def piecewise_utility(x):
            return cp.minimum(x, 1) + 0.5 * cp.maximum(x - 1, 0)
        return [piecewise_utility] * num_agents
    
    else:
        raise ValueError(f"Unknown utility type: {utility_type}")

def create_power_utility(risk_aversion: float) -> Callable:
    """
    Create a power utility function.
    
    Args:
        risk_aversion: Risk aversion parameter (γ)
        
    Returns:
        Power utility function: u(x) = x^(1-γ)/(1-γ) for γ ≠ 1, log(x) for γ = 1
    """
    def power_utility(x):
        if risk_aversion == 1:
            return np.log(x)
        else:
            return (x ** (1 - risk_aversion)) / (1 - risk_aversion)
    return power_utility

def create_exponential_utility(risk_aversion: float = 1.0) -> Callable:
    """
    Create an exponential utility function.
    
    Args:
        risk_aversion: Risk aversion parameter (a)
        
    Returns:
        Exponential utility function: u(x) = -exp(-ax)/a
    """
    def exponential_utility(x):
        return -cp.exp(-risk_aversion * x) / risk_aversion
    return exponential_utility

def create_piecewise_linear_utility(breakpoints: list, slopes: list) -> Callable:
    """
    Create a piecewise linear utility function.
    
    Args:
        breakpoints: List of x-coordinates where slope changes
        slopes: List of slopes for each piece
        
    Returns:
        Piecewise linear utility function
    """
    def piecewise_utility(x):
        result = slopes[0] * x
        for i in range(len(breakpoints)):
            result += slopes[i+1] * cp.maximum(x - breakpoints[i], 0)
        return result
    return piecewise_utility

def create_utility_functions(
    num_agents: int,
    risk_aversion_range: tuple = (0.2, 0.8)
) -> List[Callable]:
    """
    Create utility functions for each agent type with different risk aversion parameters.
    
    Args:
        num_agents: Number of agent types
        risk_aversion_range: Range of risk aversion parameters (min, max)
        
    Returns:
        List of utility functions, one for each agent type
    """
    risk_aversions = np.linspace(risk_aversion_range[0], risk_aversion_range[1], num_agents)
    return [create_power_utility(γ) for γ in risk_aversions]

def create_reference_distribution(
    num_agent_types: int = 1,
    num_material_states: int = None
) -> np.ndarray:
    """
    Create a reference distribution over agent types and material states.
    
    Args:
        num_agent_types: Number of agent types (L)
        num_material_states: Number of material states (K). If None, creates a distribution only over agent types.
        
    Returns:
        Reference distribution. If num_material_states is None, shape is (num_agent_types,).
        Otherwise, shape is (num_agent_types, num_material_states).
    """
    if num_material_states is None:
        return np.ones(num_agent_types) / num_agent_types
    else:
        return np.ones((num_agent_types, num_material_states)) / (num_agent_types * num_material_states)

def create_endowments(
    num_material_states: int,
    num_outcomes: int,
    min_value: float = 1.0,
    max_value: float = 5.0
) -> np.ndarray:
    """
    Create endowments for each state-outcome pair.
    
    Args:
        num_material_states: Number of material states (K)
        num_outcomes: Number of outcomes (N)
        min_value: Minimum endowment value
        max_value: Maximum endowment value
        
    Returns:
        Endowments array of shape (K, N)
    """
    endowments = np.zeros((num_material_states, num_outcomes))
    for k in range(num_material_states):
        endowments[k] = np.linspace(min_value, max_value, num_outcomes)
    return endowments 