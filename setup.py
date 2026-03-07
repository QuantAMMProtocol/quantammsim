from setuptools import setup, find_packages

setup(
    name="quantammsim",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "jax>=0.4.27",
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
        "Historic-Crypto",
        "gdown",
        "binance_historical_data",
        "dask",
        "jsonpickle",
        "flask_cors",
        "optax",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "pytest-xdist>=3.0",
            "pytest-timeout>=2.0",
            "black",
            "flake8",
            "hypothesis",
        ],
        "docs": [
            "sphinx",
            "sphinx-automodapi",
            "sphinx-rtd-theme",
        ],
        "calibration": [
            "numpyro>=0.15.0",
            "arviz>=0.15.0",
        ],
    },
    python_requires=">=3.9",
)
