# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

# Allow conf.py to find the xdem module
sys.path.append(os.path.abspath("../.."))
sys.path.append(os.path.abspath("../../xdem/"))
sys.path.append(os.path.abspath(".."))

import xdem.version

# -- Project information -----------------------------------------------------

project = 'xdem'
copyright = '2021, Erik Mannerfelt, Romain Hugonnet, Amaury Dehecq and others'

author = 'Erik Mannerfelt, Romain Hugonnet, Amaury Dehecq and others'

# The full version, including alpha/beta/rc tags
release = xdem.version.version


os.environ["PYTHON"] = sys.executable


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",  # Create the API documentation automatically
    "sphinx.ext.viewcode",  # Create the "[source]" button in the API to show the source code.
    'matplotlib.sphinxext.plot_directive',  # Render matplotlib figures from code.
    "sphinx.ext.autosummary",  # Create API doc summary texts from the docstrings.
    "sphinx.ext.inheritance_diagram",  # For class inheritance diagrams (see coregistration.rst).
    "sphinx_autodoc_typehints",  # Include type hints in the API documentation.
    "sphinxcontrib.programoutput",
    "sphinx_gallery.gen_gallery",  # Examples gallery
    "sphinx.ext.intersphinx",
]

#autosummary_generate = True

sphinx_gallery_conf = {
     "examples_dirs": os.path.join(os.path.dirname(__file__), "../", "../", "examples"),   # path to your example scripts
     "gallery_dirs": "auto_examples",  # path to where to save gallery generated output
     "inspect_global_variables": True,  # Make links to the class/function definitions.
     "reference_url": {
         # The module you locally document uses None
        "xdem": None,
    },
     # directory where function/class granular galleries are stored
    "backreferences_dir"  : "gen_modules/backreferences",
    "doc_module": ("xdem", "geoutils")
}

# Add any paths that contain templates here, relative to this directory.
templates_path = [os.path.join(os.path.dirname(__file__), '_templates')]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
#exclude_patterns = [
#    "_templates"
#]
#exclude_patterns = [
#    "api/modules.rst"  # This is not really needed, but is created automatically by autodoc
#]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']  # Commented out as we have no custom static data


'''
def run_apidoc(_):
    """
    Make sure readthedocs finds the module.

    Maybe this is not needed?
    """
    from sphinx.ext.apidoc import main
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    module = os.path.join(cur_dir, "../../", "xdem")
    output_path = os.path.join(cur_dir, 'gen_modules/')
    main(['-e', '-o', output_path, module, os.path.join(module, "version.py"), "--force"])


def setup(app):
    app.connect('builder-inited', run_apidoc)
'''
