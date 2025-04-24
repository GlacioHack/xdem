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
sys.path.insert(0, os.path.dirname(__file__))

from sphinx_gallery.sorting import ExplicitOrder

import xdem

# -- Project information -----------------------------------------------------

project = "xDEM"
copyright = "2024, xDEM developers"
author = "Romain Hugonnet, Erik Mannerfelt, Amaury Dehecq and others"

# The full version, including alpha/beta/rc tags
release = xdem.__version__

os.environ["PYTHON"] = sys.executable


# -- General configuration ---------------------------------------------------

master_doc = "index"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",  # Create the API documentation automatically
    "sphinx.ext.viewcode",  # Create the "[source]" button in the API to show the source code.
    "matplotlib.sphinxext.plot_directive",  # Render matplotlib figures from code.
    "sphinx.ext.autosummary",  # Create API doc summary texts from the docstrings.
    "sphinx.ext.inheritance_diagram",  # For class inheritance diagrams (see coregistration.rst).
    "sphinx.ext.graphviz",  # To render graphviz diagrams.
    "sphinx_design",  # To render nice blocks
    "sphinx_autodoc_typehints",  # Include type hints in the API documentation.
    "sphinxcontrib.programoutput",
    "sphinx_gallery.gen_gallery",  # Examples gallery
    "sphinx.ext.intersphinx",
    # "myst_parser",  !! Not needed with myst_nb !! # Form of Markdown that works with sphinx, used a lot by the Sphinx Book Theme
    "myst_nb",  # MySt for rendering Jupyter notebook in documentation
]

# For sphinx design to work properly
myst_enable_extensions = ["colon_fence", "dollarmath"]

# For myst-nb to find the Jupyter kernel (=environment) to run from
nb_kernel_rgx_aliases = {".*xdem.*": "python3"}
nb_execution_raise_on_error = True  # To fail documentation build on notebook execution error
nb_execution_show_tb = True  # To show full traceback on notebook execution error
nb_output_stderr = "warn"  # To warn if an error is raised in a notebook cell (if intended, override to "show" in cell)
nb_execution_mode = "cache"

intersphinx_mapping = {
    "python": ("https://docs.python.org/", None),
    "geoutils": ("https://geoutils.readthedocs.io/en/stable", None),
    "rasterio": ("https://rasterio.readthedocs.io/en/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "pyproj": ("https://pyproj4.github.io/pyproj/stable", None),
    "geopandas": ("https://geopandas.org/en/stable", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "rioxarray": ("https://corteva.github.io/rioxarray/stable/", None),
}

sphinx_gallery_conf = {
    "examples_dirs": [
        os.path.join(os.path.dirname(__file__), "../", "../", "examples/basic"),
        os.path.join(os.path.dirname(__file__), "../", "../", "examples/advanced"),
    ],  # path to your example scripts
    "gallery_dirs": ["basic_examples", "advanced_examples"],  # path to where to save gallery generated output
    "inspect_global_variables": True,  # Make links to the class/function definitions.
    "reference_url": {
        # The module you locally document uses None
        "xdem": None,
    },
    # directory where function/class granular galleries are stored
    "backreferences_dir": "gen_modules/backreferences",
    "doc_module": ("xdem",),  # which function/class levels are used to create galleries
    # 'subsection_order': ExplicitOrder([os.path.join(os.path.dirname(__file__), "../", "../", "examples", "basic"),
    #                                    os.path.join(os.path.dirname(__file__), "../", "../", "examples", "advanced")])
    "remove_config_comments": True,
    # To remove comments such as sphinx-gallery-thumbnail-number (only works in code, not in text)
    "reset_modules": (
        "matplotlib",
        "sphinxext.reset_mpl",
    ),
    # To reset matplotlib for each gallery (and run custom function that fixes the default DPI)
}

extlinks = {
    "issue": ("https://github.com/GlacioHack/xdem/issues/%s", "GH"),
    "pull": ("https://github.com/GlacioHack/xdem/pull/%s", "PR"),
}

# For matplotlib figures generate with sphinx plot: (suffix, dpi)
plot_formats = [(".png", 600)]

# To avoid long path names in inheritance diagrams
inheritance_alias = {
    "geoutils.georaster.raster.Raster": "geoutils.Raster",
    "geoutils.georaster.raster.Mask": "geoutils.Mask",
    "geoutils.geovector.Vector": "geoutils.Vector",
    "xdem.dem.DEM": "xdem.DEM",
    "xdem.coreg.base.Coreg": "xdem.Coreg",
    "xdem.coreg.affine.AffineCoreg": "xdem.AffineCoreg",
    "xdem.coreg.biascorr.BiasCorr": "xdem.BiasCorr",
}

# To have an edge color that works in both dark and light mode
inheritance_edge_attrs = {"color": "dodgerblue1"}

# To avoid fuzzy PNGs
graphviz_output_format = "svg"

# Add any paths that contain templates here, relative to this directory.
templates_path = [os.path.join(os.path.dirname(__file__), "_templates")]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_templates"]


# To ignore warnings due to having myst-nb reading the .ipynb created by sphinx-gallery
# Should eventually be fixed, see: https://github.com/executablebooks/MyST-NB/issues/363
def setup(app):
    # Ignore .ipynb files
    app.registry.source_suffix.pop(".ipynb", None)


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"
html_favicon = "_static/xdem_logo_only.svg"
html_logo = "_static/xdem_logo.svg"
html_title = "xDEM"

html_theme_options = {
    "path_to_docs": "doc/source",
    "use_sidenotes": True,
    "repository_url": "https://github.com/GlacioHack/xdem",
    "repository_branch": "main",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "use_source_button": True,
    "use_issues_button": True,
    "use_download_button": True,
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org/",
        "notebook_interface": "jupyterlab",
        # For launching Binder in Jupyterlab to open MD files as notebook (downloads them otherwise)
    },
    "show_toc_level": 3,  # To show more levels on the right sidebar TOC
    "logo": {
        "image_dark": "_static/xdem_logo_dark.svg",
    },
    "announcement": (
        "⚠️ Our 0.1 release refactored several early-development functions for long-term stability, "
        "to update your code see the release notes. ⚠️"
    ),
}

# For dark mode
html_context = {
    # ...
    "default_mode": "auto"
}


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["imgs", "_static"]  # Commented out as we have no custom static data

html_css_files = [
    "css/custom.css",
]
