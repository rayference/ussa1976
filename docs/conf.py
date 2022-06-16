"""Sphinx configuration."""
from datetime import datetime


project = "USSA1976"
author = "Yvan Nollet"
copyright = f"{datetime.now().year}, {author}"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "sphinxcontrib.bibtex",
]
autodoc_typehints = "description"
bibtex_bibfiles = ["bibliography.bib"]
html_theme = "furo"
intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("http://docs.scipy.org/doc/scipy/reference/", None),
    "xarray": ("http://xarray.pydata.org/en/stable/", None),
}

napoleon_preprocess_types = True
napoleon_type_aliases = {
    "callable": ":py:func:`callable`",
    "array": "~numpy.ndarray",
    "DataArray": "~xarray.DataArray",
    "Dataset": "~xarray.Dataset",
}
