# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'HyperOpt Examples'
copyright = '2022, Ather Abbas'
author = 'Ather Abbas'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
'sphinx.ext.todo',
'sphinx.ext.viewcode',
'sphinx.ext.autodoc',
'sphinx.ext.autosummary',
'sphinx.ext.doctest',
'sphinx.ext.intersphinx',
'sphinx.ext.imgconverter',
'sphinx_issues',
'sphinx.ext.mathjax',
'sphinx.ext.napoleon',
'sphinx.ext.githubpages',
"sphinx-prompt",
"sphinx_gallery.gen_gallery",
'sphinx.ext.ifconfig',
]

toggleprompt_offset_right  = 30

# specify the master doc, otherwise the build at read the docs fails
master_doc = 'index'

templates_path = ['_templates']
exclude_patterns = ["utils.py"]


sphinx_gallery_conf = {
    'backreferences_dir': 'gen_modules/backreferences',
    #'doc_module': ('sphinx_gallery', 'numpy'),
    'reference_url': {
        'sphinx_gallery': None,
    },
    'examples_dirs': os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'examples'),
    'gallery_dirs': 'auto_examples',
    'compress_images': ('images', 'thumbnails'),
    'filename_pattern': 'example',
    'ignore_pattern': 'utils.py',

    'binder': {'org': 'sphinx-gallery',
               'repo': 'sphinx-gallery.github.io',
               'branch': 'master',
               'binderhub_url': 'https://mybinder.org',
               'dependencies': os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.binder', 'requirements.txt'),
               'notebooks_dir': 'notebooks',
               'use_jupyter_lab': True,
               },
    #'show_memory': True,
    #'junit': os.path.join('sphinx-gallery', 'junit-results.xml'),
    # capture raw HTML or, if not present, __repr__ of last expression in
    # each code block
    'capture_repr': ('_repr_html_', '__repr__'),
    'matplotlib_animations': True,
    'image_srcset': ["2x"]
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
