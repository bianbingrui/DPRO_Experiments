"""Base configuration for DPRO experiments."""
import numpy as np

# Space dimensions
X_SPACE_SIZE = 3
OMEGA_SPACE_SIZE = 2
S_SPACE_SIZE = 2

# Number of acts
NUM_BASIC_ACTS = 5
NUM_COMPARISON_ACTS = 3

# Basic AA acts definitions (r_i)
BASIC_AA_ACTS = {
    # r₁ (Very Safe)
    1: {
        1: {1: 0.3, 2: 0.4, 3: 0.3},  # s=1
        2: {1: 0.3, 2: 0.4, 3: 0.3}   # s=2
    },
    # r₂ (Moderately Safe)
    2: {
        1: {1: 0.25, 2: 0.5, 3: 0.25},
        2: {1: 0.3, 2: 0.4, 3: 0.3}
    },
    # r₃ (Balanced)
    3: {
        1: {1: 0.2, 2: 0.6, 3: 0.2},
        2: {1: 0.2, 2: 0.6, 3: 0.2}
    },
    # r₄ (Moderately Risky)
    4: {
        1: {1: 0.2, 2: 0.3, 3: 0.5},
        2: {1: 0.5, 2: 0.3, 3: 0.2}
    },
    # r₅ (Very Risky)
    5: {
        1: {1: 0.1, 2: 0.2, 3: 0.7},
        2: {1: 0.7, 2: 0.2, 3: 0.1}
    }
}

# Comparison AA acts (c_m)
COMPARISON_AA_ACTS = {
    # c₁ (Conservative)
    1: {
        1: {1: 0.4, 2: 0.4, 3: 0.2},
        2: {1: 0.4, 2: 0.4, 3: 0.2}
    },
    # c₂ (Moderate)
    2: {
        1: {1: 0.3, 2: 0.4, 3: 0.3},
        2: {1: 0.3, 2: 0.4, 3: 0.3}
    },
    # c₃ (Aggressive)
    3: {
        1: {1: 0.2, 2: 0.3, 3: 0.5},
        2: {1: 0.5, 2: 0.3, 3: 0.2}
    }
}

# Risk profiles for mental states
RISK_PROFILES = {
    "RISK_AVERSE": 0.7,    # ω₁: Risk-averse agent
    "RISK_SEEKING": 0.3    # ω₂: Risk-seeking agent
}

# Experiment parameters
TRUE_DISTRIBUTION = np.array([0.6, 0.4])
MAX_SAMPLES = 100
NUM_TRIALS = 5
RANDOM_SEED = 42

# KL-divergence parameter
EPSILON = 10.0 