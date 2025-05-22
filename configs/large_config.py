"""Large configuration for DPRO experiments with expanded state spaces."""
import numpy as np
# Space dimensions
X_SPACE_SIZE = 5
OMEGA_SPACE_SIZE = 5
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
        1: {1: 0.1, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2},
        2: {1: 0.32, 2: 0.28, 3: 0.1, 4: 0.2, 5: 0.1},
        3: {1: 0.2, 2: 0.11, 3: 0.19, 4: 0.2, 5: 0.3},
        4: {1: 0.02, 2: 0.02, 3: 0.76, 4: 0.2, 5: 0}
    },
    
    # r₃ (Moderately Safe) - Stronger bias to middle outcomes
    3: {
        1: {1: 1, 2: 0, 3: 0, 4: 0, 5: 0},
        2: {1: 0.8, 2: 0.2, 3: 0, 4: 0, 5: 0},
        3: {1: 0.62, 2: 0.08, 3: 0.2, 4: 0, 5: 0.1},
        4: {1: 0, 2: 0.2, 3: 0.4, 4: 0.2, 5: 0.2}
    },
    
    # r₄ (Balanced with Low Risk) - Bias towards lower outcomes
    4: {
        1: {1: 0.05, 2: 0.22, 3: 0.08, 4: 0.35, 5: 0.3},
        2: {1: 0, 2: 0, 3: 0, 4: 0.1, 5: 0.9},
        3: {1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2},
        4: {1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.5}
    },
    
    # r₅ (Balanced with High Risk) - Bias towards higher outcomes
    5: {
        1: {1: 0.01, 2: 0.01, 3: 0.95, 4: 0.01, 5: 0.01},
        2: {1: 0.99, 2: 0, 3: 0, 4: 0, 5: 0.01},
        3: {1: 0.1, 2: 0.1, 3: 0.2, 4: 0.3, 5: 0.3},
        4: {1: 0, 2: 0, 3: 0, 4: 0, 5: 1}
    },
    
    # r₆ (Moderately Risky) - Extreme outcomes with moderate probability
    6: {
        1: {1: 0.4, 2: 0, 3: 0, 4: 0.1, 5: 0.5},
        2: {1: 0.8, 2: 0, 3: 0, 4: 0, 5: 0.2},
        3: {1: 0, 2: 0, 3: 0, 4: 0, 5: 1},
        4: {1: 0, 2: 0, 3: 0.6, 4: 0, 5: 0.4}
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
    1: {s: {1: 1, 2: 0, 3: 0, 4: 0, 5: 0} for s in range(1, 5)},
    
    # c₂ (Conservative)
    2: {s: {1: 0, 2: 1, 3: 0, 4: 0, 5: 0} for s in range(1, 5)},
    
    # c₃ (Moderate)
    3: {s: {1: 0, 2: 0, 3: 1, 4: 0, 5: 0} for s in range(1, 5)},
    
    # c₄ (Aggressive)
    4: {s: {1: 0, 2: 0, 3: 0, 4: 1, 5: 0} for s in range(1, 5)},
    
    # c₅ (Very Aggressive)
    5: {s: {1: 0, 2: 0, 3: 0, 4: 0, 5: 1} for s in range(1, 5)}
}

# Risk profiles for mental states
RISK_PROFILES = {
    "Extremely Risk-Averse": 0.99,     # ω₁: Very risk-averse agent
    "VERY_RISK_AVERSE": 0.67,     # ω₁: Very risk-averse agent
    "RISK_AVERSE": 0.35,          # ω₂: Moderately risk-averse agent
    "RISK_SEEKING": 0.2,         # ω₃: Moderately risk-seeking agent
    "VERY_RISK_SEEKING": 0.05,     # ω₄: Very risk-seeking agent
}

# Experiment parameters
TRUE_DISTRIBUTION_MENTAL = np.array([0.6, 0.2, 0.1, 0.05, 0.05])  # More balanced distribution for 4 states
TRUE_DISTRIBUTION_MATERIAL = np.array([0.13, 0.7, 0.16, 0.01])
MAX_SAMPLES = 100
NUM_TRIALS = 5
RANDOM_SEED = 42



