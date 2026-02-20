import os
import sys
from typing import ForwardRef

# Path setup
sys.path.insert(0, os.path.abspath("../.."))


# Workaround for type aliases (from JAX)
def _do_not_evaluate_in_jax(
    self,
    globalns,
    *args,
    _evaluate=ForwardRef._evaluate,
):
    if globalns.get("__name__", "").startswith("quantammsim"):
        return self
    return _evaluate(self, globalns, *args)


ForwardRef._evaluate = _do_not_evaluate_in_jax

# Project information
project = "quantammsim"
copyright = "2025-2026, QuantAMM.fi"
author = "QuantAMM.fi team"
version = "0.1"
release = "0.1"

# Extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
]

# Theme settings
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 4,
    "titles_only": False,
    "logo_only": False,
    "prev_next_buttons_location": "bottom",
    "style_nav_header_background": "#2980B9",
}

# Autodoc settings
autodoc_default_options = {
    "members": None,
    "undoc-members": False,
    "show-inheritance": True,
    "special-members": "__init__",
    "private-members": False,
    # "noindex": True
}

# Add these settings
autodoc_member_order = 'bysource'
autosummary_generate = True
autosummary_imported_members = False
primary_domain = "py"  

# Type hints
autodoc_typehints = "description"
autodoc_typehints_description_target = "all"
autodoc_inherit_docstrings = False

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

# Autosummary
autosummary_generate_overwrite = False
templates_path = ['_templates']
add_module_names = False

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
}

# Basic exclude patterns
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

# Add explicit module exclusions
exclude_modules = [
    'quantammsim.internal.*',
    'quantammsim.*.tests.*',
]

# Output settings

html_build_dir = "_build/html"
html_static_path = ["_static"]

# Build output settings
output_dir = "build"

# Add this setting
autodoc_preserve_defaults = True
