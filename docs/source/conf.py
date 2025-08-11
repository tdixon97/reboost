# Configuration file for the Sphinx documentation builder.
from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path

from sphinx.ext.napoleon.docstring import GoogleDocstring, NumpyDocstring

sys.path.insert(0, Path(__file__).parents[2].resolve().as_posix())

project = "reboost"
copyright = "The LEGEND Collaboration"
version = importlib.metadata.version("reboost")

extensions = [
    "sphinx.ext.githubpages",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "myst_parser",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"
language = "python"

# Furo theme
html_theme = "furo"
html_theme_options = {
    "source_repository": "https://github.com/legend-exp/reboost",
    "source_branch": "main",
    "source_directory": "docs/source",
}
html_title = f"{project} {version}"

autodoc_default_options = {"ignore-module-all": True}

myst_enable_extensions = ["colon_fence", "substitution", "dollarmath"]

# sphinx-napoleon
# enforce consistent usage of NumPy-style docstrings
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_ivar = True

# fix napoleon "Returns" section to not need an actual type.
NumpyDocstring._consume_returns_section = GoogleDocstring._consume_returns_section

# intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pygama": ("https://pygama.readthedocs.io/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "awkward": ("https://awkward-array.org/doc/stable", None),
    "pint": ("https://pint.readthedocs.io/en/stable", None),
    "pyg4ometry": ("https://pyg4ometry.readthedocs.io/en/stable", None),
    "legendmeta": ("https://pylegendmeta.readthedocs.io/en/stable/", None),
    "lgdo": ("https://legend-pydataobj.readthedocs.io/en/stable/", None),
    "dbetto": ("https://dbetto.readthedocs.io/en/stable/", None),
}  # add new intersphinx mappings here

# sphinx-autodoc
# Include __init__() docstring in class docstring
autoclass_content = "both"
autodoc_typehints = "description"
autodoc_typehints_description_target = "all"
autodoc_typehints_format = "short"
