# Distributionally Robust Policy Optimization (DPRO)

This repository contains the implementation and experimental analysis of Distributionally Robust Policy Optimization (DPRO), a novel approach to decision-making under uncertainty that combines robust optimization with behavioral economics insights.

## Overview

DPRO is designed to make optimal decisions while accounting for:
- Uncertainty in probability distributions
- Heterogeneous risk preferences
- Limited sample availability
- Out-of-sample performance guarantees

The implementation compares DPRO against traditional approaches:
- Maximum Expected Utility (MEU)
- Non-robust optimization (NR)

## Project Structure

```
.
├── configs/
│   └── large_config.py    # Configuration parameters and constants
├── src/
│   ├── optimization.py    # DPRO, MEU, and NR optimization implementations
│   ├── utils.py          # Utility functions and helpers
│   ├── experiment.py     # Experiment class definition
│   └── run_experiments.py # Main experiment runner
├── plots/                # Generated experiment plots
└── results/             # Experimental results data
```

## Requirements

- Python 3.8+
- NumPy
- CVXPY
- Matplotlib
- Seaborn
- Rich (for console output)

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run experiments:
```bash
PYTHONPATH=$PYTHONPATH:. python src/run_experiments.py
```

2. The script will:
   - Run risk aversion experiments
   - Generate empirical distributions
   - Compare DPRO, MEU, and NR solutions
   - Create visualization plots
   - Save results in the `plots/` directory

## Experiment Parameters

Key parameters that can be modified in `configs/large_config.py`:
- `EPSILON`: Robustness parameter
- `MAX_SAMPLES`: Maximum number of samples for empirical distribution
- `NUM_TRIALS`: Number of experiment trials
- `RISK_PROFILES`: Risk aversion parameters for different mental states

## Results

The experiments generate several plots:
1. Risk aversion vs solution weights
2. DPRO vs MEU optimal values
3. Average weight distribution
4. Risk aversion distribution by mental state
5. Out-of-sample performance comparison
6. Performance difference distribution

Results are saved in two directories:
- `plots/full_results/`: Complete analysis plots
- `plots/out_of_sample/`: Out-of-sample performance comparisons

## Citation

If you use this code in your research, please cite:
```bibtex
@article{dpro2024,
  title={Distributionally Robust Policy Optimization with Heterogeneous Risk Preferences},
  author={Bingrui Bian, William B, Haskell},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or feedback, please open an issue or contact [your-email@domain.com]. 