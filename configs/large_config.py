"""Large configuration for DPRO experiments with expanded state spaces."""
import numpy as np

# Space dimensions
X_SPACE_SIZE = 5
OMEGA_SPACE_SIZE = 4
S_SPACE_SIZE = 4

# Number of acts
NUM_BASIC_ACTS = 8
NUM_COMPARISON_ACTS = 5

# Basic AA acts definitions (r_i)
BASIC_AA_ACTS = {
    # r₁ (Very Safe) - Almost uniform distribution
    1: {s: {x: 0.2 for x in range(1, 6)} for s in range(1, 5)},
    
    # r₂ (Safe) - Slight bias to middle outcomes
    2: {
        1: {1: 0.15, 2: 0.2, 3: 0.3, 4: 0.2, 5: 0.15},
        2: {1: 0.15, 2: 0.2, 3: 0.3, 4: 0.2, 5: 0.15},
        3: {1: 0.15, 2: 0.2, 3: 0.3, 4: 0.2, 5: 0.15},
        4: {1: 0.15, 2: 0.2, 3: 0.3, 4: 0.2, 5: 0.15}
    },
    
    # r₃ (Moderately Safe) - Stronger bias to middle outcomes
    3: {
        1: {1: 0.1, 2: 0.2, 3: 0.4, 4: 0.2, 5: 0.1},
        2: {1: 0.1, 2: 0.2, 3: 0.4, 4: 0.2, 5: 0.1},
        3: {1: 0.1, 2: 0.2, 3: 0.4, 4: 0.2, 5: 0.1},
        4: {1: 0.1, 2: 0.2, 3: 0.4, 4: 0.2, 5: 0.1}
    },
    
    # r₄ (Balanced with Low Risk) - Bias towards lower outcomes
    4: {
        1: {1: 0.3, 2: 0.3, 3: 0.2, 4: 0.1, 5: 0.1},
        2: {1: 0.1, 2: 0.1, 3: 0.2, 4: 0.3, 5: 0.3},
        3: {1: 0.3, 2: 0.3, 3: 0.2, 4: 0.1, 5: 0.1},
        4: {1: 0.1, 2: 0.1, 3: 0.2, 4: 0.3, 5: 0.3}
    },
    
    # r₅ (Balanced with High Risk) - Bias towards higher outcomes
    5: {
        1: {1: 0.1, 2: 0.1, 3: 0.2, 4: 0.3, 5: 0.3},
        2: {1: 0.3, 2: 0.3, 3: 0.2, 4: 0.1, 5: 0.1},
        3: {1: 0.1, 2: 0.1, 3: 0.2, 4: 0.3, 5: 0.3},
        4: {1: 0.3, 2: 0.3, 3: 0.2, 4: 0.1, 5: 0.1}
    },
    
    # r₆ (Moderately Risky) - Extreme outcomes with moderate probability
    6: {
        1: {1: 0.3, 2: 0.1, 3: 0.2, 4: 0.1, 5: 0.3},
        2: {1: 0.3, 2: 0.1, 3: 0.2, 4: 0.1, 5: 0.3},
        3: {1: 0.3, 2: 0.1, 3: 0.2, 4: 0.1, 5: 0.3},
        4: {1: 0.3, 2: 0.1, 3: 0.2, 4: 0.1, 5: 0.3}
    },
    
    # r₇ (Risky) - High variance
    7: {
        1: {1: 0.4, 2: 0.1, 3: 0.0, 4: 0.1, 5: 0.4},
        2: {1: 0.4, 2: 0.1, 3: 0.0, 4: 0.1, 5: 0.4},
        3: {1: 0.4, 2: 0.1, 3: 0.0, 4: 0.1, 5: 0.4},
        4: {1: 0.4, 2: 0.1, 3: 0.0, 4: 0.1, 5: 0.4}
    },
    
    # r₈ (Very Risky) - Extreme bias to highest/lowest outcomes
    8: {
        1: {1: 0.05, 2: 0.1, 3: 0.15, 4: 0.2, 5: 0.5},
        2: {1: 0.5, 2: 0.2, 3: 0.15, 4: 0.1, 5: 0.05},
        3: {1: 0.05, 2: 0.5, 3: 0.15, 4: 0.2, 5: 0.1},
        4: {1: 0.1, 2: 0.2, 3: 0.15, 4: 0.5, 5: 0.05}
    }
}

# Comparison AA acts (c_m)
COMPARISON_AA_ACTS = {
    # c₁ (Very Conservative)
    1: {s: {1: 0.3, 2: 0.3, 3: 0.2, 4: 0.1, 5: 0.1} for s in range(1, 5)},
    
    # c₂ (Conservative)
    2: {s: {1: 0.2, 2: 0.3, 3: 0.3, 4: 0.1, 5: 0.1} for s in range(1, 5)},
    
    # c₃ (Moderate)
    3: {s: {1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2} for s in range(1, 5)},
    
    # c₄ (Aggressive)
    4: {s: {1: 0.1, 2: 0.1, 3: 0.3, 4: 0.3, 5: 0.2} for s in range(1, 5)},
    
    # c₅ (Very Aggressive)
    5: {s: {1: 0.1, 2: 0.1, 3: 0.2, 4: 0.3, 5: 0.3} for s in range(1, 5)}
}

# Risk profiles for mental states
RISK_PROFILES = {
    "VERY_RISK_AVERSE": 0.8,     # ω₁: Very risk-averse agent
    "RISK_AVERSE": 0.6,          # ω₂: Moderately risk-averse agent
    "RISK_SEEKING": 0.4,         # ω₃: Moderately risk-seeking agent
    "VERY_RISK_SEEKING": 0.2     # ω₄: Very risk-seeking agent
}

# Experiment parameters
TRUE_DISTRIBUTION = np.array([0.3, 0.3, 0.2, 0.2])  # More balanced distribution for 4 states
MAX_SAMPLES = 100
NUM_TRIALS = 5
RANDOM_SEED = 42

# KL-divergence parameter
EPSILON = 1  # Reduced from 10000 to get more reasonable values