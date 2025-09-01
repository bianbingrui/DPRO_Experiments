# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-12-01

### Added
- Initial release of DPRO Optimization framework
- Distributionally Robust Preference Optimization (DPRO) implementation
- Maximum Expected Utility (MEU) comparison
- Non-robust (NR) baseline implementation
- Random acts matrix generation for enhanced experimental diversity
- Comprehensive parameter grid experiments
- Automated result visualization and plotting
- Support for mental state ambiguity modeling
- Risk aversion parameter framework
- Utility function implementations

### Features
- **Core Optimization**: DPRO, MEU, and NR solvers using CVXPY
- **Random Acts Generation**: Stochastic generation of AA acts matrices
- **Parameter Grid Experiments**: Systematic exploration of parameter space
- **Visualization**: Automated generation of comparison plots
- **Configuration**: Flexible parameter configuration system
- **Progress Tracking**: Rich console interface for experiment monitoring

### Technical Details
- Python 3.8+ compatibility
- CVXPY-based optimization solvers
- NumPy and SciPy for numerical computations
- Matplotlib and Seaborn for visualization
- Rich library for enhanced console output
- Modular architecture for easy extension

## [0.2.0] - 2024-11-15

### Added
- Random acts matrix generation functionality
- Enhanced experimental framework
- Improved visualization tools
- Better parameter configuration

### Changed
- Refactored code structure for better maintainability
- Updated experiment runner with random acts generation
- Improved documentation and code comments
- Consolidated experiment runners into single `run_trials.py` file

### Removed
- Redundant optimization modules
- Duplicate experiment runners
- Unused virtual environment files

## [0.1.0] - 2024-10-01

### Added
- Basic DPRO implementation
- MEU comparison framework
- Initial experiment structure
- Basic utility functions

---

## Version History

- **1.0.0**: First stable release with complete DPRO framework
- **0.2.0**: Code cleanup and random acts generation, consolidated into single experiment runner
- **0.1.0**: Initial development version

## Future Plans

### Planned for v1.1.0
- Additional optimization algorithms
- Enhanced visualization options
- Performance benchmarking tools
- Extended parameter validation

### Planned for v1.2.0
- Machine learning integration
- Real-world data examples
- Performance optimization
- Additional utility functions

---

For detailed information about each release, please refer to the [GitHub releases page](https://github.com/yourusername/DPRO_Experiments/releases).
