"""Test cases for DPRO experiments."""
import numpy as np
import pytest
from typing import Dict, Tuple

from src.config import (
    X_SPACE_SIZE, S_SPACE_SIZE, OMEGA_SPACE_SIZE,
    BASIC_AA_ACTS, COMPARISON_AA_ACTS, EPSILON
)
from src.utils import (
    calculate_utility,
    generate_risk_parameters,
    calculate_meu_gamma,
    kl_divergence,
    convert_act_to_matrix,
    verify_distribution_constraints
)
from src.optimization import solve_dpro, solve_meu
from src.experiment import DPROExperiment

def test_utility_function():
    """Test utility function calculation."""
    # Test with γ = 0.5
    x = 2.0
    gamma = 0.5
    expected = 2 * np.sqrt(2)  # (2^0.5) / 0.5
    utility = calculate_utility(x, gamma)
    assert np.isclose(utility, expected), f"Expected {expected}, got {utility}"
    
    # Test with γ = 1 (log utility)
    gamma = 1.0
    expected = np.log(x)
    utility = calculate_utility(x, gamma)
    assert np.isclose(utility, expected), f"Expected {expected}, got {utility}"

def test_risk_parameters():
    """Test risk parameter generation."""
    n_agents = 10
    seed = 42
    params = generate_risk_parameters(n_agents, seed)
    
    assert len(params) == n_agents
    assert np.all(params >= 0) and np.all(params < 1)
    
    # Test reproducibility
    params2 = generate_risk_parameters(n_agents, seed)
    assert np.allclose(params, params2)

def test_meu_gamma():
    """Test MEU gamma calculation."""
    gamma_omega = np.array([0.3, 0.7])
    phi_hat = np.array([0.6, 0.4])
    expected = 0.3 * 0.6 + 0.7 * 0.4  # 0.46
    
    result = calculate_meu_gamma(gamma_omega, phi_hat)
    assert np.isclose(result, expected)

def test_kl_divergence():
    """Test KL divergence calculation."""
    p = np.array([0.3, 0.7])
    q = np.array([0.5, 0.5])
    
    kl = kl_divergence(p, q)
    assert kl > 0  # KL divergence should be positive
    assert np.isclose(kl_divergence(p, p), 0)  # KL(P||P) = 0

def test_act_conversion():
    """Test AA act conversion to matrix format."""
    act = BASIC_AA_ACTS[1]  # Use first basic act
    matrix = convert_act_to_matrix(act, X_SPACE_SIZE, S_SPACE_SIZE)
    
    # Check dimensions
    assert matrix.shape == (X_SPACE_SIZE, S_SPACE_SIZE)
    
    # Check if it's a valid distribution
    assert verify_distribution_constraints(matrix)

def test_full_experiment():
    """Test full experiment workflow."""
    # Initialize experiment
    experiment = DPROExperiment(seed=42)
    
    # Generate distributions
    phi_hat, psi_hat = experiment.generate_empirical_distributions()
    assert len(phi_hat) == OMEGA_SPACE_SIZE
    assert len(psi_hat) == S_SPACE_SIZE
    
    # Run experiment
    results = experiment.run_experiment()
    
    # Check results structure
    assert 'dpro_z' in results
    assert 'meu_z' in results
    assert 'dpro_value' in results
    assert 'meu_value' in results
    
    # Check solution validity
    dpro_z = results['dpro_z']
    meu_z = results['meu_z']
    
    # Check if solutions are valid probability distributions
    assert np.isclose(np.sum(dpro_z), 1)
    assert np.all(dpro_z >= 0)
    assert np.isclose(np.sum(meu_z), 1)
    assert np.all(meu_z >= 0)

def test_specific_case():
    """Test with specific, hand-calculated case."""
    # Set up a simple test case
    phi_hat = np.array([0.5, 0.5])  # Equal weights for agent types
    psi_hat = np.array([0.5, 0.5])  # Equal weights for states
    gamma_omega = np.array([0.3, 0.7])  # Different risk aversions
    
    # Solve DPRO
    dpro_z, dpro_value = solve_dpro(phi_hat, psi_hat, gamma_omega)
    
    # Solve MEU
    gamma_meu = calculate_meu_gamma(gamma_omega, phi_hat)
    meu_z, meu_value = solve_meu(psi_hat, gamma_meu)
    
    # Basic sanity checks
    assert len(dpro_z) == len(meu_z) == 5  # Number of basic acts
    assert np.isclose(np.sum(dpro_z), 1)
    assert np.isclose(np.sum(meu_z), 1)
    
    # Check that solutions are reasonable
    # Both solutions should put more weight on safer acts than very risky acts
    assert dpro_z[0] + dpro_z[1] > dpro_z[4]  # DPRO prefers safer over very risky
    assert meu_z[0] + meu_z[1] > meu_z[4]     # MEU prefers safer over very risky
    
    # The difference between DPRO and MEU weights should be small
    # since the distributions are uniform and risk parameters are moderate
    assert np.abs((dpro_z[0] + dpro_z[1]) - (meu_z[0] + meu_z[1])) < 0.1

if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__]) 