import numpy as np
import cvxpy as cp
from typing import Dict, Callable

class MEUModel:
    """
    Maxmin Expected Utility (MEU) model for social welfare allocation.
    This model uses a single utility function with average risk aversion.
    """
    
    def __init__(
        self,
        num_material_states: int,
        num_outcomes: int,
        utility_function: Callable,
        reference_distribution: np.ndarray,
        endowments: np.ndarray,
        ambiguity_set_params: Dict
    ):
        """
        Initialize the MEU model.
        
        Args:
            num_material_states: Number of material states (K)
            num_outcomes: Number of possible outcomes (N)
            utility_function: Utility function with average risk aversion
            reference_distribution: Reference distribution over states
            endowments: Endowments for each state-outcome pair
            ambiguity_set_params: Parameters for constructing ambiguity sets
        """
        self.K = num_material_states
        self.N = num_outcomes
        self.utility_function = utility_function
        self.reference_distribution = reference_distribution
        self.endowments = endowments
        self.epsilon = ambiguity_set_params.get('epsilon', 0.1)
        
    def solve(self) -> Dict:
        """
        Solve the MEU optimization problem.
        
        Returns:
            Dictionary containing:
            - objective_value: Optimal objective value
            - policy_weights: Optimal weights for baseline policies
            - worst_case_distribution: Worst-case distribution over states
        """
        # Define variables
        z = cp.Variable(self.N, nonneg=True)  # Policy weights
        beta = cp.Variable()  # Dual variable for probability simplex
        gamma = cp.Variable()  # Dual variable for KL divergence
        lambda1 = cp.Variable(self.K)  # Dual variables for exponential cone
        lambda2 = cp.Variable(self.K)
        lambda3 = cp.Variable(self.K)
        
        # Define constraints
        constraints = [
            cp.sum(z) == 1,  # Policy weights sum to 1
            gamma + lambda3 == 0  # Constraint from dual formulation
        ]
        
        # Add utility constraints for each state
        for k in range(self.K):
            # Compute utility for this state
            utility = self.utility_function(self.endowments[k])
            
            # Add constraint from dual formulation
            constraints.append(
                utility + beta - lambda2[k] <= 0
            )
            
            # Add exponential cone constraint
            constraints.append(
                cp.ExpCone(lambda1[k], lambda2[k], lambda3[k])
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
    
    def _compute_utilities(self,
        baseline_policies: np.ndarray,
        z: cp.Variable) -> np.ndarray:
        """
        Compute utilities for each state.
        """
        utilities = np.zeros(self.K)
        for k in range(self.K):
            policy = np.sum(baseline_policies * z, axis=1)
            utilities[k] = np.mean([u(policy[k]) for u in self.utility_functions])
        return utilities 