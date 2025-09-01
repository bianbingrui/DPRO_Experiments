# DPRO Optimization: Distributionally Robust Preference Optimization

This repository contains the numerical experiments and implementation for the paper "Distributionally Robust Preference Optimization with Mental States" (working title).

## 📖 **Paper Abstract**

This work introduces a novel approach to distributionally robust optimization that accounts for both material uncertainty and mental state ambiguity. We propose the Distributionally Robust Preference Optimization (DPRO) model, which extends traditional robust optimization by incorporating mental states that influence risk preferences without affecting objective probabilities.

## 🎯 **Key Features**

- **DPRO Model**: Distributionally robust optimization with mental state ambiguity
- **MEU Comparison**: Comparison with Maximum Expected Utility approach
- **Random Acts Generation**: Stochastic generation of AA acts for enhanced experimental diversity
- **Comprehensive Analysis**: Welfare comparison across different parameter settings
- **Reproducible Results**: Complete experimental setup with configurable parameters

## 🏗️ **Repository Structure**

```
DPRO_Experiments/
├── run_trials.py              # Main experiment runner (random acts generation)
├── src/
│   ├── optimization_funcs.py # Core optimization solvers (DPRO, MEU, NR)
│   └── utils.py             # Utility functions and helpers
├── configs/
│   └── large_config.py      # Configuration parameters
├── out_of_smaple_v2/        # Experimental results and plots
├── plots/                   # Additional visualization outputs
├── Plots_for_paper/         # Paper-related figures
├── Plots_for_paper2/        # Additional paper figures
├── experiment_settings.txt  # Detailed experiment configuration
├── requirements.txt         # Python dependencies
└── main.tex               # LaTeX paper draft
```

## 🚀 **Quick Start**

### **Installation**

1. Clone the repository:
```bash
git clone https://github.com/yourusername/DPRO_Experiments.git
cd DPRO_Experiments
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### **Running Experiments**

Run the main numerical experiments:
```bash
python run_trials.py
```

This will execute a comprehensive parameter grid experiment:
- **Trials**: 10, 50, 100
- **Mental states**: 5, 10, 50 samples
- **Material states**: 5, 10, 50 samples
- **Epsilon values**: 0.01, 0.1, 1.0

### **Custom Experiments**

For custom experiments, you can import and use the functions directly:

```python
from run_trials import run_experiments, plot_results

# Run custom experiment
results = run_experiments(
    n_trials=100,
    n_mental_realization=20,
    n_material_realization=30,
    epsilon=0.1,
    seed=42
)

# Generate plots
plot_results(results, params, save_path="custom_experiment.png")
```

## 🔬 **Methodology**

### **DPRO Model**
The Distributionally Robust Preference Optimization model solves:

$$\min_{z \in \Delta} \max_{(\phi,\psi) \in \mathcal{C}} \mathbb{E}_{(\phi,\psi)}[u(z,\omega,s)]$$

where:
- $z$ are policy weights
- $(\phi,\psi)$ are distributions over mental and material states
- $\mathcal{C}$ is the ambiguity set
- $u(z,\omega,s)$ is the utility function

### **Mental States**
- Mental states influence risk preferences without affecting objective probabilities
- Each mental state has a unique risk aversion parameter $\gamma_\omega$
- Utility function: $u(x,\omega) = \frac{x^{1-\gamma_\omega}}{1-\gamma_\omega}$

### **AA Acts**
- Acts are represented as probability matrices $r_i(x,s)$
- Each act satisfies $\sum_x r_i(x,s) = 1$ for all states $s$
- Random generation ensures experimental diversity

## 📊 **Results**

The experiments compare three approaches:
1. **DPRO**: Distributionally robust optimization
2. **MEU**: Maximum expected utility
3. **NR**: Non-robust (baseline)

Results are automatically saved in the `out_of_smaple_v2/` directory with comprehensive visualizations.

## 📋 **Dependencies**

- **Core**: numpy, scipy, matplotlib, seaborn
- **Optimization**: cvxpy, clarabel, scs, ecos
- **Progress**: rich
- **Documentation**: pandas

## 📚 **Citation**

If you use this code in your research, please cite:

```bibtex
@article{yourname2024dpro,
  title={Distributionally Robust Preference Optimization with Mental States},
  author={Your Name},
  journal={Working Paper},
  year={2024}
}
```

## 🤝 **Contributing**

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 **Contact**

For questions or collaboration, please contact: [your.email@university.edu]

## 🙏 **Acknowledgments**

We thank the research community for valuable feedback and discussions on this work.

---

**Note**: This is a working paper. Results and methodology may be updated as the research progresses. 