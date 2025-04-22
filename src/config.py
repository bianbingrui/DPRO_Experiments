"""Configuration file containing all experimental settings."""
import numpy as np

# Space dimensions
X_SPACE_SIZE = 3  # |X| = 3 (outcomes)
OMEGA_SPACE_SIZE = 2  # |Ω| = 2 (mental states)
S_SPACE_SIZE = 2  # |S| = 2 (material states)

# Number of basic AA acts
NUM_BASIC_ACTS = 5  # d = 5

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
NUM_COMPARISON_ACTS = 3  # |M| = 3
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

# Sample sizes
N_OMEGA = 100  # N^Ω
N_S = 100  # N^S

# KL-divergence parameter
EPSILON = 10.0  # Increased from 1.0 for better numerical stability

# Random seed for reproducibility
RANDOM_SEED = 42 