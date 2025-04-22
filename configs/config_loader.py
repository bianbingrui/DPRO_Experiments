"""Configuration loader for DPRO experiments."""
import argparse
from dataclasses import dataclass
from enum import Enum
import importlib

class ConfigType(Enum):
    BASE = "base"
    LARGE = "large"

@dataclass
class ExperimentConfig:
    """Configuration class for DPRO experiments."""
    # Space dimensions
    X_SPACE_SIZE: int
    OMEGA_SPACE_SIZE: int
    S_SPACE_SIZE: int
    
    # Acts
    NUM_BASIC_ACTS: int
    NUM_COMPARISON_ACTS: int
    BASIC_AA_ACTS: dict
    COMPARISON_AA_ACTS: dict
    
    # Risk profiles
    RISK_PROFILES: dict
    
    # Parameters
    TRUE_DISTRIBUTION: list
    MAX_SAMPLES: int
    NUM_TRIALS: int
    RANDOM_SEED: int
    EPSILON: float

    @classmethod
    def load(cls, config_type: ConfigType) -> 'ExperimentConfig':
        """Load configuration based on type.
        
        Args:
            config_type: Type of configuration to load
            
        Returns:
            Loaded configuration
        """
        # Import appropriate configuration module
        module_name = f"configs.{config_type.value}_config"
        config_module = importlib.import_module(module_name)
        
        # Create configuration instance
        return cls(
            X_SPACE_SIZE=config_module.X_SPACE_SIZE,
            OMEGA_SPACE_SIZE=config_module.OMEGA_SPACE_SIZE,
            S_SPACE_SIZE=config_module.S_SPACE_SIZE,
            NUM_BASIC_ACTS=config_module.NUM_BASIC_ACTS,
            NUM_COMPARISON_ACTS=config_module.NUM_COMPARISON_ACTS,
            BASIC_AA_ACTS=config_module.BASIC_AA_ACTS,
            COMPARISON_AA_ACTS=config_module.COMPARISON_AA_ACTS,
            RISK_PROFILES=config_module.RISK_PROFILES,
            TRUE_DISTRIBUTION=config_module.TRUE_DISTRIBUTION,
            MAX_SAMPLES=config_module.MAX_SAMPLES,
            NUM_TRIALS=config_module.NUM_TRIALS,
            RANDOM_SEED=config_module.RANDOM_SEED,
            EPSILON=config_module.EPSILON
        )

def get_config_from_args() -> ExperimentConfig:
    """Get configuration from command-line arguments.
    
    Returns:
        Loaded configuration based on command-line arguments
    """
    parser = argparse.ArgumentParser(description='DPRO Experiment Configuration')
    parser.add_argument(
        '--config',
        type=str,
        choices=['base', 'large'],
        default='base',
        help='Configuration type to use (base or large)'
    )
    args = parser.parse_args()
    
    config_type = ConfigType(args.config)
    return ExperimentConfig.load(config_type)

if __name__ == "__main__":
    # Example usage
    config = get_config_from_args()
    print(f"Loaded {config.X_SPACE_SIZE}x{config.OMEGA_SPACE_SIZE}x{config.S_SPACE_SIZE} configuration") 