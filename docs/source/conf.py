import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "quantammsim"
copyright = "2024, quantamm.fi"
author = "QuantAMM.fi team"
version = "0.1"
release = "0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


# Extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.coverage",
    "sphinx_automodapi.automodapi",
]

# Mock imports for packages that might be problematic on RTD
autodoc_mock_imports = [
    "jax",
    "jaxlib",
    "flask_jwt_extended",
    "numpy",
    "scipy",
    "scipy.stats",
    "scipy.spatial",
    "scipy.sparse",
    "seaborn",
    "cvxpy",
    "pandas",
    "pandas._libs",
    "pandas.compat",
    "pandas.util",
    "matplotlib",
    "tqdm",
]

# Theme
html_theme = "sphinx_rtd_theme"

# Autodoc settings
autodoc_member_order = "bysource"
autodoc_typehints = "description"
automodapi_toctreedirnm = "api/generated"
automodsumm_writereprocessed = False

# templates_path = ['_templates']
# exclude_patterns = []


# # -- Options for HTML output -------------------------------------------------
# # https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
# html_static_path = ['_static']

# Add to existing conf.py
html_theme_options = {
    "navigation_depth": 4,
    "titles_only": False,
    "logo_only": False,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "style_nav_header_background": "#2980B9",
}

# Better autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Add intersphinx mapping
extensions.append("sphinx.ext.intersphinx")
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
}

# Add to existing configuration
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "inherited-members": False,  # Add this line
}

# Add this setting
autodoc_inherit_docstrings = False

exclude_patterns = ["quantamm.rst", "quantamm.*.rst", "quantammsim.rst", "quantammsim.*.rst"]
