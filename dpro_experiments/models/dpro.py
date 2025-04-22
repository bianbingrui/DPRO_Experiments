import numpy as np
import cvxpy as cp
from typing import Dict, List, Callable

class DPROModel:
    """
    Distributionally Robust Optimization (DPRO) model for social welfare allocation.
    This model considers ambiguity in both agent types and material states.
    """
    
    def __init__(
        self,
        num_agent_types: int,
        num_material_states: int,
        num_outcomes: int,
        utility_functions: List[Callable],
        reference_distribution: np.ndarray,
        endowments: np.ndarray,
        ambiguity_set_params: Dict
    ):
        """
        Initialize the DPRO model.
        
        Args:
            num_agent_types: Number of different agent types (L)
            num_material_states: Number of material states (K)
            num_outcomes: Number of possible outcomes (N)
            utility_functions: List of utility functions for each agent type
            reference_distribution: Reference distribution over states and agent types
            endowments: Endowments for each state-outcome pair
            ambiguity_set_params: Parameters for constructing ambiguity sets
        """
        self.L = num_agent_types
        self.K = num_material_states
        self.N = num_outcomes
        self.utility_functions = utility_functions
        self.reference_distribution = reference_distribution
        self.endowments = endowments
        self.epsilon = ambiguity_set_params.get('epsilon', 0.1)
        
    def solve(self) -> Dict:
        """
        Solve the DPRO optimization problem.
        
        Returns:
            Dictionary containing:
            - objective_value: Optimal objective value
            - policy_weights: Optimal weights for baseline policies
            - worst_case_distribution: Worst-case distribution over states and agent types
        """
        # Define variables
        z = cp.Variable(self.N, nonneg=True)  # Policy weights
        beta = cp.Variable()  # Dual variable for probability simplex
        gamma = cp.Variable()  # Dual variable for KL divergence
        lambda1 = cp.Variable((self.L, self.K))  # Dual variables for exponential cone
        lambda2 = cp.Variable((self.L, self.K))
        lambda3 = cp.Variable((self.L, self.K))
        
        # Define constraints
        constraints = [
            cp.sum(z) == 1,  # Policy weights sum to 1
            gamma + lambda3 == 0  # Constraint from dual formulation
        ]
        
        # Add utility constraints for each agent type and state
        for l in range(self.L):
            for k in range(self.K):
                # Compute utility for this agent type and state
                utility = self.utility_functions[l](self.endowments[k])
                
                # Add constraint from dual formulation
                constraints.append(
                    utility + beta - lambda2[l,k] <= 0
                )
                
                # Add exponential cone constraint
                constraints.append(
                    cp.ExpCone(lambda1[l,k], lambda2[l,k], lambda3[l,k])
                )
        
        # Define objective
        objective = cp.Maximize(
            -beta - self.epsilon * gamma - 
            cp.sum(cp.multiply(self.reference_distribution, lambda1))
        )
        
        # Form and solve problem
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS)
        
        if prob.status != 'optimal':
            raise ValueError(f"Problem status: {prob.status}")
        
        # Recover primal solution (worst-case distribution)
        theta = lambda1.value / gamma.value
        
        return {
            'objective_value': prob.value,
            'policy_weights': z.value,
            'worst_case_distribution': theta
        } 