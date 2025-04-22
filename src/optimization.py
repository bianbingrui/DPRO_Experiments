"""Optimization solvers for DPRO and MEU problems."""
import cvxpy as cp
import numpy as np
from typing import Dict, Tuple, List

from configs.large_config import (
    X_SPACE_SIZE, S_SPACE_SIZE, NUM_BASIC_ACTS,
    BASIC_AA_ACTS, COMPARISON_AA_ACTS, EPSILON
)
from src.utils import convert_act_to_matrix, calculate_utility

def solve_dpro(phi_hat: np.ndarray, psi_hat: np.ndarray, gamma_omega: np.ndarray,
               acts: List[np.ndarray], verbose: bool = False) -> Tuple[np.ndarray, float]:
    """Solve the DPRO optimization problem.
    
    Args:
        phi_hat: Empirical distribution over Ω
        psi_hat: Empirical distribution over S
        gamma_omega: Array of risk aversion parameters
        acts: List of AA acts in matrix form
        verbose: Whether to print solver output
    
    Returns:
        Tuple of optimal z and optimal value
    """
    n_acts = len(acts)
    n_omega = len(phi_hat)
    n_s = acts[0].shape[1]
    n_x = acts[0].shape[0]
    
    # Decision variables
    z = cp.Variable(n_acts)
    alpha = cp.Variable(n_omega)
    beta = cp.Variable(n_omega)
    t = cp.Variable()
    
    # Dual variables for each omega
    lambda_1 = cp.Variable(n_omega)
    lambda_2 = cp.Variable(n_omega)
    lambda_3 = cp.Variable(n_omega)
    
    # Calculate utilities with proper scaling
    utilities = {}
    for omega in range(n_omega):
        utilities[omega] = {}
        for s in range(n_s):
            utilities[omega][s] = np.array([calculate_utility(x+1, gamma_omega[omega]) 
                                          for x in range(n_x)])
    
    # Objective: maximize expected utility minus regularization
    objective = cp.sum(phi_hat @ alpha) - EPSILON * t - 1e-3 * cp.sum_squares(z)
    
    constraints = []
    
    # Basic constraints on z
    constraints.append(cp.sum(z) == 1)
    constraints.append(z >= 0)
    
    # Bounds on variables for numerical stability
    constraints.append(beta >= -1)
    constraints.append(beta <= 1)
    constraints.append(t >= -1)
    constraints.append(t <= 1)
    
    # Dual exponential cone constraints for each omega
    for omega in range(n_omega):
        # Scale lambda variables for numerical stability
        constraints.append(cp.ExpCone(-0.1 * lambda_2[omega], 
                                    -0.1 * lambda_1[omega], 
                                    0.1 * lambda_3[omega]))
        constraints.append(lambda_1[omega] + lambda_2[omega] <= t)
        
        # Bounds on dual variables
        constraints.append(lambda_1[omega] >= -1)
        constraints.append(lambda_1[omega] <= 1)
        constraints.append(lambda_2[omega] >= -1)
        constraints.append(lambda_2[omega] <= 1)
        constraints.append(lambda_3[omega] >= -1)
        constraints.append(lambda_3[omega] <= 1)
        
        # Alpha-beta constraints for each state
        for s in range(n_s):
            lhs = alpha[omega]
            for k in range(n_acts):
                act_k = acts[k]
                lhs -= z[k] * np.sum(utilities[omega][s] * act_k[:, s])
            constraints.append(lhs <= beta[omega])
    
    # Define and solve the problem
    prob = cp.Problem(cp.Maximize(objective), constraints)
    
    # Try different solvers with better numerical parameters
    solvers = [
        (cp.CLARABEL, {'tol': 1e-6, 'max_iter': 10000}),
        (cp.SCS, {'eps': 1e-6, 'max_iters': 20000, 'alpha': 1.5, 'scale': 0.5}),
        (cp.ECOS, {'abstol': 1e-6, 'reltol': 1e-6, 'max_iters': 10000, 'feastol': 1e-6})
    ]
    
    for solver, params in solvers:
        try:
            prob.solve(solver=solver, verbose=verbose, **params)
            if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                # Normalize z to ensure it sums to 1 and is non-negative
                z_value = np.maximum(z.value, 0)
                z_normalized = z_value / np.sum(z_value)
                return z_normalized, prob.value
        except Exception as e:
            if verbose:
                print(f"Solver {solver} failed with error: {e}")
            continue
    
    raise ValueError("No solver could solve the DPRO problem. Try adjusting the problem parameters.")

def solve_meu(phi_hat: np.ndarray, psi_hat: np.ndarray, gamma_meu: float,
              acts: List[np.ndarray], verbose: bool = False) -> Tuple[np.ndarray, float]:
    """Solve the MEU optimization problem.
    
    Args:
        phi_hat: Empirical distribution over Ω
        psi_hat: Empirical distribution over S
        gamma_meu: MEU risk aversion parameter
        acts: List of AA acts in matrix form
        verbose: Whether to print solver output
    
    Returns:
        Tuple of optimal z and optimal value
    """
    n_acts = len(acts)
    n_s = acts[0].shape[1]
    n_x = acts[0].shape[0]
    
    # Decision variable
    z = cp.Variable(n_acts)
    
    # Calculate expected utilities for each act
    utilities = []
    for k in range(n_acts):
        act_k = acts[k]
        act_utilities = np.zeros(n_s)
        for s in range(n_s):
            outcomes = np.arange(1, n_x + 1)
            act_utilities[s] = np.sum([calculate_utility(x, gamma_meu) * act_k[x-1, s] for x in outcomes])
        utilities.append(act_utilities)
    utilities = np.array(utilities)
    
    # Objective: maximize expected utility with regularization
    objective = cp.sum(z @ utilities @ psi_hat) - 1e-2 * cp.sum_squares(z)
    
    # Constraints
    constraints = [
        cp.sum(z) == 1,
        z >= 0
    ]
    
    # Define and solve the problem
    prob = cp.Problem(cp.Maximize(objective), constraints)
    
    # Try different solvers with looser tolerances
    solvers = [
        (cp.CLARABEL, {'tol': 1e-5, 'max_iter': 5000}),
        (cp.SCS, {'eps': 1e-5, 'max_iters': 20000, 'alpha': 1.8, 'scale': 0.1}),
        (cp.ECOS, {'abstol': 1e-5, 'reltol': 1e-5, 'max_iters': 5000, 'feastol': 1e-5})
    ]
    
    for solver, params in solvers:
        try:
            prob.solve(solver=solver, verbose=verbose, **params)
            if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                # Normalize z to ensure it sums to 1 and is non-negative
                z_value = np.maximum(z.value, 0)  # Ensure non-negativity
                z_normalized = z_value / np.sum(z_value)
                return z_normalized, prob.value
        except Exception as e:
            if verbose:
                print(f"Solver {solver} failed with error: {e}")
            continue
    
    raise ValueError("No solver could solve the MEU problem. Try adjusting the problem parameters.")

def solve_nr(phi_hat: np.ndarray, psi_hat: np.ndarray, gamma: np.ndarray, acts: List[np.ndarray]) -> Tuple[np.ndarray, float]:
    """Solve the non-robust optimization problem.
    
    This solves the simple expected utility maximization:
    max_z sum_{ω,s} φ(ω)ψ(s)u(z,ω,s)
    s.t. sum_i z_i = 1
         z ≥ 0
    
    Args:
        phi_hat: Empirical distribution over mental states (|Ω|,)
        psi_hat: Empirical distribution over material states (|S|,)
        gamma: Risk aversion parameters for each mental state (|Ω|,)
        acts: List of act matrices, each of shape (|X|, |S|)
    
    Returns:
        Tuple of:
        - Optimal solution z* (n_acts,)
        - Optimal value
    """
    n_acts = len(acts)
    n_omega = len(phi_hat)
    n_s = len(psi_hat)
    
    # Decision variable
    z = cp.Variable(n_acts)
    
    # Calculate expected utilities for each (omega, s) pair
    expected_utility = 0
    for omega in range(n_omega):
        for s in range(n_s):
            # Get probability of this (omega, s) pair
            prob = phi_hat[omega] * psi_hat[s]
            
            # Calculate utility for each act under this (omega, s)
            for i, act in enumerate(acts):
                outcome_probs = act[:, s]  # Probabilities over outcomes for this material state
                utilities = np.array([calculate_utility(x+1, gamma[omega]) 
                                   for x in range(len(outcome_probs))])
                expected_utility += prob * z[i] * np.sum(utilities * outcome_probs)
    
    # Objective: maximize expected utility
    objective = cp.Maximize(expected_utility)
    
    # Constraints
    constraints = [
        cp.sum(z) == 1,  # Probability simplex
        z >= 0           # Non-negative weights
    ]
    
    # Solve the problem
    problem = cp.Problem(objective, constraints)
    
    # Try different solvers
    solvers = ['CLARABEL', 'SCS', 'ECOS']
    solution_found = False
    
    for solver in solvers:
        try:
            if solver == 'CLARABEL':
                problem.solve(solver=solver, max_iter=10000)
            elif solver == 'SCS':
                problem.solve(solver=solver, max_iters=10000, eps=1e-8)
            else:  # ECOS
                problem.solve(solver=solver, max_iters=1000, abstol=1e-8, reltol=1e-8)
            
            if problem.status == 'optimal':
                solution_found = True
                break
        except:
            continue
    
    if not solution_found:
        raise RuntimeError("Could not solve NR problem with any available solver")
    
    # Get solution
    z_nr = z.value
    optimal_value = problem.value
    
    # Ensure solution is normalized and feasible
    z_nr = np.clip(z_nr, 0, None)  # Ensure non-negativity
    z_nr = z_nr / np.sum(z_nr)     # Ensure normalization
    
    return z_nr, optimal_value

acts_matrix = [convert_act_to_matrix(act, X_SPACE_SIZE, S_SPACE_SIZE) 
               for act in BASIC_AA_ACTS.values()] 