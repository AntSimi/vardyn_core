# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'VarDyn'
copyright = '2026, Florian Le Guillou, Valentin Bellemin-Laponnaz, Sammy Metref, Clément Ubelmann, Emmanuel Cosme, Antoine Delepoulle'
author = 'Florian Le Guillou, Valentin Bellemin-Laponnaz, Sammy Metref, Clément Ubelmann, Emmanuel Cosme, Antoine Delepoulle'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_gallery.gen_gallery",
]

# autodoc conf
autoclass_content = "both"

# Active autosummary
autosummary_generate = True

templates_path = ["_templates"]

source_suffix = ".rst"

# The master toctree document.
master_doc = "index"
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

sphinx_gallery_conf = {
    "examples_dirs": "../../examples",  # path to your example scripts
    "gallery_dirs": "auto_examples",  # path to where to save gallery generated output
    "capture_repr": ("_repr_html_", "__repr__"),
    "backreferences_dir": "gen_modules/backreferences",
    "doc_module": ("vardyn_core",),
    "reference_url": {
        "vardyn_core": None,
    },
    "line_numbers": False,
    "filename_pattern": "/vardyn_core_",
    "matplotlib_animations": True,
    "within_subsection_order": "FileNameSortKey",
}