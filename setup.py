from setuptools import setup, find_packages

setup(
    name="quantammsim",
    packages=find_packages(),
    install_requires=[
        "jax>=0.4.0",
        "jaxlib",  # Required for JAX to work
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "flask",
        "flask-jwt-extended",
        "scipy",
        "seaborn",
        "cvxpy",
        "matplotlib",
        "tqdm",
        "optuna",
        "pyarrow",
        "plotly",
        "bidask",
        "Historic_Crypto",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black",
            "flake8",
            "pytest-cov",
            "hypothesis",
        ],
        "docs": [
            "sphinx",
            "sphinx-automodapi",
            "sphinx-rtd-theme",
        ],
    },
    python_requires=">=3.8",
)
