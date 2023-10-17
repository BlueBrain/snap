# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

import importlib

# -- Project information -----------------------------------------------------

project = "Blue Brain SNAP"
author = "Blue Brain Project, EPFL"

release = importlib.metadata.distribution("bluepysnap").version
version = release

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx-bluebrain-theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

# ensure a useful title is used
html_title = "Blue Brain SNAP"

# hide source links
html_show_sourcelink = False

# set the theme settings
html_theme_options = {
    "repo_url": "https://github.com/BlueBrain/snap/",
    "repo_name": "BlueBrain/snap",
}

# autodoc settings
autodoc_default_options = {
    "members": True,
}

autoclass_content = "both"

autodoc_mock_imports = ["libsonata"]

# autosummary settings
autosummary_generate = True

suppress_warnings = [
    "autosectionlabel.*",
]
