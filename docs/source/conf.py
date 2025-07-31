# Configuration file for the Sphinx documentation builder.
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# -- Project information -----------------------------------------------------
project = 'Graph Hypernetwork Forge'
copyright = '2024, Daniel Schmidt'
author = 'Daniel Schmidt'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.napoleon',
    'myst_parser',
    'sphinx_rtd_theme',
]

# MyST configuration for Markdown support
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

# Source file suffixes
source_suffix = {
    '.rst': None,
    '.md': None,
}

# The master toctree document
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2980b9',
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

html_static_path = ['_static']
html_css_files = ['custom.css']

# -- Options for LaTeX output ------------------------------------------------
latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'preamble': '',
    'fncychap': '',
    'printindex': '',
}

latex_documents = [
    (master_doc, 'GraphHypernetworkForge.tex', 'Graph Hypernetwork Forge Documentation',
     'Daniel Schmidt', 'manual'),
]

# -- Extension configuration -------------------------------------------------

# autodoc configuration
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'member-order': 'bysource',
}

autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'
autodoc_class_signature = 'mixed'

# autosummary configuration
autosummary_generate = True
autosummary_imported_members = True

# napoleon configuration for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'torch_geometric': ('https://pytorch-geometric.readthedocs.io/en/latest/', None),
    'transformers': ('https://huggingface.co/docs/transformers/master/en/', None),
}

# todo configuration
todo_include_todos = True

# viewcode configuration
viewcode_follow_imported_members = True

# mathjax configuration
mathjax3_config = {
    'tex': {
        'inlineMath': [['$', '$'], ['\\(', '\\)']],
        'displayMath': [['$$', '$$'], ['\\[', '\\]']],
    },
}

# Custom configuration for ML documentation
def setup(app):
    """Custom Sphinx setup."""
    app.add_css_file('custom.css')
    
    # Add custom directives for ML documentation
    from docutils.parsers.rst import directives
    from docutils import nodes
    from docutils.parsers.rst.directives import unchanged
    
    class ModelCard(directives.Directive):
        """Custom directive for model documentation cards."""
        has_content = True
        required_arguments = 1
        optional_arguments = 0
        option_spec = {
            'architecture': unchanged,
            'parameters': unchanged,
            'datasets': unchanged,
            'metrics': unchanged,
        }
        
        def run(self):
            model_name = self.arguments[0]
            content = f"**Model:** {model_name}\n\n"
            
            for option, value in self.options.items():
                content += f"**{option.title()}:** {value}\n\n"
            
            content += "\n".join(self.content)
            
            # Create a container for the model card
            container = nodes.container()
            container['classes'].append('model-card')
            
            # Parse the content as reStructuredText
            self.state.nested_parse(
                self.content,
                self.content_offset,
                container
            )
            
            return [container]
    
    app.add_directive('model-card', ModelCard)