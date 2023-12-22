"""
Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import os

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "GVAL"
copyright = "2023, NOAA-OWP"
author = "Fernando Aristizabal, Gregory Petrochenkov"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_rtd_theme",
    "nbsphinx",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- Setting Pandoc's path ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-pandoc_path

# Set docs path: `gval/docs/`
docs_dir_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

# Set and make pandoc path
pandoc_dir_path = os.path.join(docs_dir_path, "pandoc")

# Set and make pandoc executable path
pandoc_executable_path = os.path.join(pandoc_dir_path, "pandoc")

# Set pandoc executable path to environment variable
os.environ.setdefault("PANDOC", pandoc_executable_path)
