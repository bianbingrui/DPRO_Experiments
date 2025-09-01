from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="dpro-optimization",
    version="1.0.0",
    author="Bingrui Bian, William B. Haskell",
    author_email="bbian@purdue.edu",
    description="Distributionally Robust Preference Optimization with Mental States",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/DPRO_Experiments",
    packages=find_packages(),
    py_modules=["run_trials"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    keywords="optimization, distributionally robust, mental states, preferences, uncertainty",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/DPRO_Experiments/issues",
        "Source": "https://github.com/yourusername/DPRO_Experiments",
        "Documentation": "https://github.com/yourusername/DPRO_Experiments#readme",
    },
    entry_points={
        "console_scripts": [
            "dpro-experiments=run_trials:main",
        ],
    },
)