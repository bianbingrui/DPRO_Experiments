"""Optimization solvers for DPRO and MEU problems."""
import cvxpy as cp
import numpy as np
from typing import Dict, Tuple, List

from configs.large_config import (
    X_SPACE_SIZE, S_SPACE_SIZE, NUM_BASIC_ACTS,
    BASIC_AA_ACTS, COMPARISON_AA_ACTS
)
from src.utils import convert_act_to_matrix, calculate_utility

def calculate_act_utility(act: np.ndarray, gamma: float, s:int) -> np.ndarray:
    """Calculate the utility of an act for a given material state and risk aversion parameter.
    
    Args:
        act: The act matrix
        gamma: The risk aversion parameter
        s: The material state index

    Returns:
        Utility value
    """
    return np.sum(act[:, s] * calculate_utility(np.arange(1, X_SPACE_SIZE + 1), gamma))

def solve_dpro(phi_hat: np.ndarray, psi_hat: np.ndarray, gamma_omega: np.ndarray,
               acts: List[np.ndarray], comparison_acts_matrix: List[np.ndarray] = None, c_omega_m=None, verbose: bool = False, epsilon: float = 0.0) -> Tuple[np.ndarray, float]:
    """Solve the DPRO optimization problem.
    
    Args:
        phi_hat: Empirical distribution over Ω
        psi_hat: Empirical distribution over S
        gamma_omega: Array of risk aversion parameters
        acts: List of AA acts in matrix form
        comparison_acts_matrix: List of comparison acts in matrix form
        c_omega_m: 2D numpy array of shape (n_omega, m), reference utility for each (omega, m)
        verbose: Whether to print solver output
    
    Returns:
        Tuple of optimal z and optimal value
    """
    assert c_omega_m is not None, "c_omega_m must be provided!"
    if comparison_acts_matrix is None:
        comparison_acts_matrix = acts

    n_acts = len(acts)
    m = len(comparison_acts_matrix)
    n_omega = len(phi_hat)
    n_s = acts[0].shape[1]
    n_x = acts[0].shape[0]
    vartheta_hat = np.outer(phi_hat,psi_hat)
    
    # Decision variables
    z = cp.Variable(n_acts)
    alpha = cp.Variable((m, n_omega))
    beta = cp.Variable()
    gamma = cp.Variable()
    
    # Dual variables for each omega
    lambda1 = cp.Variable((n_omega, n_s))
    lambda2 = cp.Variable((n_omega, n_s))
    lambda3 = cp.Variable((n_omega, n_s))
    
    # Calculate utilities
    utilities = np.zeros((n_acts, n_omega, n_s))
    for i in range(n_acts):
        for omega in range(n_omega):
            for s in range(n_s):
                # acts[i] has shape (X_SPACE_SIZE, S_SPACE_SIZE)
                # calculate_utility returns shape (X_SPACE_SIZE,)
                # This computes the expected utility for act i, mental state omega, and material state s
                utilities[i, omega, s] = np.sum(
                    acts[i][:, s] * calculate_utility(np.arange(1, X_SPACE_SIZE + 1), gamma_omega[omega])
                )

    comparison_utilities = [[] for _ in range(m)]
    for i in range(m):
        comparison_utilities[i] = np.zeros((n_omega, n_s))
        for omega in range(n_omega):
            for s in range(n_s):
                comparison_utilities[i][omega, s] = calculate_act_utility(comparison_acts_matrix[i], gamma_omega[omega], s)

    # Objective
    expr = []
    for omega in range(n_omega):
        for s in range(n_s):
            expr.append(cp.sum(cp.multiply(z, utilities[:, omega, s])))
    objective = - cp.sum(cp.multiply(lambda1, vartheta_hat))  -beta - epsilon * gamma 
    
    constraints = []
    
    # Basic constraints on z
    constraints.append(cp.sum(z) == 1)
    constraints.append(z >= 0)
    
    # Bounds on variables for numerical stability
    constraints.append(beta >= -1)
    constraints.append(beta <= 1)
    constraints.append(gamma >= -1)
    constraints.append(gamma <= 1)
    
    # Dual exponential cone constraints for each omega
    for omega in range(n_omega):
        for s in range(n_s):
            # Scale lambda variables for numerical stability
            constraints.append(cp.ExpCone(-0.1 * lambda2[omega, s], 
                                        -0.1 * lambda1[omega, s], 
                                        0.1 * lambda3[omega, s]))
            constraints.append(lambda3[omega, s] + gamma == 0)
        
            # Bounds on dual variables
            constraints.append(lambda1[omega, s] >= -1)
            constraints.append(lambda1[omega, s] <= 1)
            constraints.append(lambda2[omega, s] >= -1)
            constraints.append(lambda2[omega, s] <= 1)
            constraints.append(lambda3[omega, s] >= -1)
            constraints.append(lambda3[omega, s] <= 1)

    utilities_constraints_value = np.zeros((m,n_omega,n_s))
    # Alpha-beta constraints for each state
    for omega in range(n_omega):
        for s in range(n_s):
            for m_idx in range(m):
                utilities_constraints_value[m_idx, omega, s] =  c_omega_m[omega, m_idx] - comparison_utilities[m_idx][omega, s] 
    
    for omega in range(n_omega):
        for s in range(n_s):
            if m > 0:
                constraints.append(
                    expr[omega * n_omega + s ] + cp.sum(cp.multiply(alpha[:, omega], utilities_constraints_value[:,omega,s])) + beta - lambda2[omega, s] <= 0
                )

        for m_idx in range(m):
            for omega in range(n_omega):
                constraints.append(alpha[m_idx, omega] >= 0)
    
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
              acts: List[np.ndarray], verbose: bool = False, epsilon: float = 0.0) -> Tuple[np.ndarray, float]:
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
    n_omega = len(phi_hat)
    n_s = acts[0].shape[1]
    n_x = acts[0].shape[0]
    vartheta_hat = np.outer(phi_hat,psi_hat)
    
    # Decision variable
    z = cp.Variable(n_acts)
    beta = cp.Variable()
    gamma = cp.Variable()

    # Dual variables for each omega
    lambda1 = cp.Variable(n_s)
    lambda2 = cp.Variable(n_s)
    lambda3 = cp.Variable(n_s)

    # Calculate expected utilities for each act
    utilities = np.zeros((n_acts, n_s))
    for i in range(n_acts):
        for s in range(n_s):
            utilities[i, s] = np.sum(acts[i][:, s] * calculate_utility(np.arange(1, X_SPACE_SIZE + 1), gamma_meu))
    
    # Objective: maximize expected utility with regularization
    objective = - beta - epsilon * gamma - cp.sum(cp.multiply(psi_hat, lambda1))
    
    expr = []
    for s in range(n_s):
        expr.append(cp.sum(cp.multiply(z, utilities[:, s])))

    # Constraints
    constraints = [
        cp.sum(z) == 1,
        z >= 0,
        gamma >= 0
    ]

    constraints.append(beta >= -1)
    constraints.append(beta <= 1)

    for s in range(n_s):
        constraints.append(expr[s] + beta - lambda2[s] <= 0)
        constraints.append(lambda1[s] + gamma == 0)
        constraints.append(lambda1[s] >= -1)
        constraints.append(lambda1[s] <= 1)
        constraints.append(lambda2[s] >= -1)
        constraints.append(lambda2[s] <= 1)
        constraints.append(lambda3[s] >= -1)
        constraints.append(lambda3[s] <= 1)
    
    # Dual exponential cone constraints for each omega
    for s in range(n_s):
        # Scale lambda variables for numerical stability
        constraints.append(cp.ExpCone(-0.1 * lambda2[s], 
                                    -0.1 * lambda1[s], 
                                    0.1 * lambda3[s]))

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

def solve_nr(phi_hat: np.ndarray, psi_hat: np.ndarray, gamma_omega: np.ndarray, acts: List[np.ndarray]) -> Tuple[np.ndarray, float]:
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
    vartheta_hat = np.outer(phi_hat,psi_hat)
    
    # Decision variable
    z = cp.Variable(n_acts)
    
    # Calculate utilities
    utilities = np.zeros((n_acts, n_omega, n_s))
    for i in range(n_acts):
        for omega in range(n_omega):
            for s in range(n_s):
                # acts[i] has shape (X_SPACE_SIZE, S_SPACE_SIZE)
                # calculate_utility returns shape (X_SPACE_SIZE,)
                # This computes the expected utility for act i, mental state omega, and material state s
                utilities[i, omega, s] = np.sum(
                    acts[i][:, s] * calculate_utility(np.arange(1, X_SPACE_SIZE + 1), gamma_omega[omega])
                )

    # Calculate expected utilities for each (omega, s) pair
    expr = []
    for omega in range(n_omega):
        for s in range(n_s):
            expr.append(cp.sum(cp.multiply(z, utilities[:, omega, s])))
    
    # Objective: maximize expected utility
    total_expected_utility = 0
    for omega in range(n_omega):
        for s in range(n_s):
            total_expected_utility += vartheta_hat[omega, s] * expr[omega * n_omega + s]
    objective = cp.Maximize(total_expected_utility)
    
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


