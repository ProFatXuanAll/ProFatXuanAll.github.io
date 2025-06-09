"""Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""  # noqa: INP001

from typing import Any

author: str = 'ProFatXuanAll'
copyright: str = '2024, ProFatXuanAll'  # noqa: A001
project: str = "ProFatXuanAll's blog"
release: str = ''
version: str = ''

########################################################################################################################
# General configuration.
# See https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration for details.
########################################################################################################################
add_function_parentheses: bool = False
add_module_names: bool = True
default_role: str | None = None
exclude_patterns: list[str] = [
  '.venv/',
  'venv/',
]
extensions: list[str] = [
  # Built-in extensions.
  'sphinx.ext.autodoc',
  'sphinx.ext.autosectionlabel',
  'sphinx.ext.duration',
  'sphinx.ext.extlinks',
  'sphinx.ext.githubpages',
  'sphinx.ext.intersphinx',
  'sphinx.ext.mathjax',
  'sphinx.ext.napoleon',
  'sphinx.ext.todo',
  # 3rd party extensions.
  'sphinx_copybutton',
  'sphinx_design',
  'sphinxcontrib.bibtex',
  'sphinxcontrib.mermaid',
]
keep_warnings: bool = True
nitpicky: bool = True
nitpick_ignore: list[tuple[str, str]] = []
nitpick_ignore_regex: list[tuple[str, str]] = []
numfig: bool = False
primary_domain: str | None = None
root_doc: str = 'index'
source_encoding: str = 'utf-8-sig'
source_suffix: dict[str, str] = {
  '.rst': 'restructuredtext',
}
templates_path: list[str] = ['_templates']
today_fmt: str = '%Y-%m-%d'

#######################################################################################################################
# Options for i18n.
# See https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-internationalization for details.
#######################################################################################################################
language: str = 'zh_TW'

#######################################################################################################################
# Options for math.
# See https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-math for details.
#######################################################################################################################
math_eqref_format: str = 'eq.{number}'
math_number_all: bool = False
math_numfig: bool = False

########################################################################################################################
# Options for HTML output.
# See https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output for details.
########################################################################################################################
html_favicon: str = 'assets/images/favicon.png'
html_last_updated_fmt: str | None = '%Y-%m-%d'
html_logo: str = ''
html_math_renderer: str = 'mathjax'
html_search_language: str = 'zh'
html_static_path: list[str] = [
  'assets',
]
html_theme: str = 'furo'
html_title: str = "ProFatXuanAll's blog"

########################################################################################################################
# Configuration for `sphinx.ext.autosectionlabel`.
# See https://www.sphinx-doc.org/en/master/usage/extensions/autosectionlabel.html for details.
########################################################################################################################
autosectionlabel_prefix_document: bool = True
autosectionlabel_maxdepth: int | None = 1

########################################################################################################################
# Configuration for `sphinx.ext.extlinks`.
# See https://www.sphinx-doc.org/en/master/usage/extensions/extlinks.html for details.
########################################################################################################################
extlinks_detect_hardcoded_links: bool = True

########################################################################################################################
# Configuration for `sphinx.ext.intersphinx`.
# See https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html for details.
########################################################################################################################
intersphinx_mapping: dict[str, tuple[str, str | None]] = {
  'numpy': ('http://docs.scipy.org/doc/numpy/', None),
  'pydantic': ('https://docs.pydantic.dev/latest/', None),
  'python': ('https://docs.python.org/3/', None),
  'torch': ('https://pytorch.org/docs/main/', None),
}

########################################################################################################################
# Configuration for `sphinx.ext.mathjax`.
# See https://www.sphinx-doc.org/en/master/usage/extensions/math.html#sphinx-ext-mathjax-render-math-via-javascript
# and https://docs.mathjax.org/en/latest/index.html for details.
########################################################################################################################
mathjax3_config: dict[str, Any] = {
  # Explicitly loads components.
  # See https://docs.mathjax.org/en/latest/input/tex/extensions.html#configuring-tex-extensions for details.
  # Note that we use `tex-mml-chtml` component script, which means we do not need to load the following extensions:
  # `ams`, `autoload`, `configmacros`, `newcommand`, `require`, `noundefined`.
  # See https://docs.mathjax.org/en/latest/web/components/combined.html#tex-mml-chtml-component for details.
  'loader': {
    'load': [
      '[tex]/braket',
      '[tex]/cancel',
      '[tex]/centernot',
      '[tex]/color',
      '[tex]/mathtools',
      '[tex]/physics',
    ],
  },
  'tex': {
    # Customize mathjax macros.
    'macros': {
      # Sets and fields.
      'C': r'{\mathbb{C}}',  # Set of complex numbers.
      'N': r'{\mathbb{N}}',  # Set of natural number.
      'Q': r'{\mathbb{Q}}',  # Set of rationals.
      'R': r'{\mathbb{R}}',  # Set of real numbers.
      'Z': r'{\mathbb{Z}}',  # Set of integers.
      # Common vectors.
      'zv': r'{\mathbf{0}}',  # Zero vector.
      # Operators.
      'Sim': r'\operatorname{sim}',
      # Algorithm.
      'algoCmt': [r'{\color{ForestGreen} \texttt{// #1}}', 1],
      'algoElse': r'\textbf{else do}',
      'algoElseIf': [r'\textbf{else if } #1 \textbf{ do}', 1],
      'algoEndFor': r'\textbf{end for}',
      'algoEndIf': r'\textbf{end if}',
      'algoEndProc': r'\textbf{end procedure}',
      'algoEq': r'\leftarrow',
      'algoFalse': r'\textbf{ False }',
      'algoFor': [r'\textbf{for } #1 \textbf{ do}', 1],
      'algoIf': [r'\textbf{if } #1 \textbf{ do}', 1],
      'algoIs': r'\textbf{ is }',
      'algoProc': [r'\textbf{procedure} #1', 1],
      'algoReturn': r'\textbf{return }',
      'algoTrue': r'\textbf{ True }',
      'indent': [r'\hspace{#1em}', 1],
      'cat': [r'\operatorname{concate}\pa{#1}', 1],
      'drop': [r'\operatorname{dropout}\pa{#1, #2}', 2],
      'loss': r'\operatorname{Loss}',
      'msk': r'\operatorname{mask}',
      'pos': r'\operatorname{pos}',
      'sof': [r'\operatorname{softmax}\pa{#1}', 1],
      'sz': [r'\operatorname{size}\pa{#1}', 1],
    },
    # Explicitly includes extensions.
    # See https://docs.mathjax.org/en/latest/input/tex/extensions.html#configuring-tex-extensions for details.
    # Note that we use `tex-mml-chtml` component script, which means we do not need to load the following extensions:
    # `ams`, `autoload`, `configmacros`, `newcommand`, `require`, `noundefined`.
    # See https://docs.mathjax.org/en/latest/web/components/combined.html#tex-mml-chtml-component for details.
    'packages': {
      '[+]': [
        'braket',
        'cancel',
        'centernot',
        'color',
        'mathtools',
        'physics',
      ],
    },
    # Use AMS numbering rule.
    # See https://docs.mathjax.org/en/latest/input/tex/eqnumbers.html for details.
    'tags': 'ams',
  },
}
# Don't change this line unless you want to upgrade `mathjax` dependency.
mathjax_path: str = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'

#######################################################################################################################
# Configuration for `sphinx.ext.napoleon`.
# See https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html for details.
#######################################################################################################################
napoleon_google_docstring: bool = False
napoleon_numpy_docstring: bool = True

#######################################################################################################################
# Configuration for `sphinx.ext.todo`.
# See https://www.sphinx-doc.org/en/master/usage/extensions/todo.html for details.
#######################################################################################################################
todo_include_todos: bool = True

#######################################################################################################################
# Configuration for `sphinx_copybutton`.
# See https://sphinx-copybutton.readthedocs.io/en/latest/ for details.
#######################################################################################################################
copybutton_exclude: str = '.linenos, .go, .gp'

########################################################################################################################
# Configuration for `sphinxcontrib.bibtex`.
# See https://sphinxcontrib-bibtex.readthedocs.io/en/latest/ for details.
########################################################################################################################
bibtex_bibfiles: list[str] = [
  'bibtex/ACL.bib',
  'bibtex/Cognitive-Science.bib',
  'bibtex/CVPR.bib',
  'bibtex/EMNLP.bib',
  'bibtex/ICLR.bib',
  'bibtex/JMLR.bib',
  'bibtex/NAACL.bib',
  'bibtex/Nature.bib',
  'bibtex/Neural-Computation.bib',
  'bibtex/Neural-Networks.bib',
  'bibtex/NIPS.bib',
  'bibtex/misc.bib',
]

########################################################################################################################
# Configuration for `furo` theme.
# See https://pradyunsg.me/furo/ for details.
########################################################################################################################
html_css_files: list[str] = [
  'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/fontawesome.min.css',
  'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/solid.min.css',
  'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/brands.min.css',
]
html_theme_options: dict[str, Any] = {
  'footer_icons': [
    {
      'name': 'GitHub',
      'url': 'https://github.com/ProFatXuanAll/ProFatXuanAll.github.io',
      'html': '',
      'class': 'fa-brands fa-solid fa-github fa-2x',
    },
  ],
}
