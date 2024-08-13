# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from importlib.metadata import version as get_version

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "fusiondls"
copyright = "2024, Cyd Cowley, Mike Kryjak"
author = "Cyd Cowley, Mike Kryjak"

# The full version, including alpha/beta/rc tags
release = get_version(project)
# Major.minor version
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.coverage",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx_autodoc_typehints",
    "myst_nb",
]

autosummary_generate = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Numpy-doc config
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_attr_annotations = True
napoleon_preprocess_types = True

# The default role for text marked up `like this`
default_role = "any"

# Tell sphinx what the primary language being documented is.
primary_domain = "py"

# Tell sphinx what the pygments highlight language should be.
highlight_language = "python"

# Include "todo" directives in output
todo_include_todos = True

# Don't execute notebook examples
nb_execution_mode = "off"

# Parse LaTeX in notebooks
myst_enable_extensions = [
    "amsmath",
    "dollarmath",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"

html_static_path = ["_static"]

html_theme_options = {
    "repository_url": "https://github.com/cydcowley/DLS-model",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_edit_page_button": True,
    "use_repository_button": True,
    "use_issues_button": True,
    "home_page_in_toc": False,
}

pygments_style = "sphinx"

# -- Options for intersphinx extension ---------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html#configuration

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}
