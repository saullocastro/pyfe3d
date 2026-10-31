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
#import os
#import sys
#sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------

# The full version, including alpha/beta/rc tags
import os
import runpy

from pyfe3d.version import __version__, __year__
release = __version__

project = 'pyfe3d'
author = 'Saullo G. P. Castro'
copyright = '2021-%s, %s' % (__year__, author)


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'numpydoc',
    'sphinx.ext.mathjax',
    'matplotlib.sphinxext.plot_directive',
    'nbsphinx',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

default_role = 'math'

# NOTE the lists of methods of the classes are not tables of contents, whose
#      entries would require a page for each method
numpydoc_class_members_toctree = False

# NOTE figures of the plot directive, cross_sectional_analysis.rst
plot_include_source = True
plot_html_show_source_link = False
plot_html_show_formats = False
plot_formats = [('png', 150)]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#html_theme = 'default'
html_theme = 'pydata_sphinx_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
#html_static_path = ['_static']


def _write_generated(app):
    # NOTE the interactive 3D figure of cross_sectional_analysis.rst is
    #      written by a script into _generated, which is not tracked
    runpy.run_path(os.path.join(app.srcdir, '_scripts',
                                'vabs_warping_3d.py'))


def setup(app):
    app.connect('builder-inited', _write_generated)
