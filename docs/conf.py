# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html
#
# Part of the standardized "xeos stack" shared across hdrake's packages:
# Sphinx + Furo + myst-nb, recursive-autosummary API reference, notebooks
# rendered from committed outputs (never executed at build time).

from importlib.metadata import version as get_version
from pathlib import Path
import shutil

# -- Project information -----------------------------------------------------
project = "xbudget"
author = "Henri F. Drake"
copyright = "2026, Henri F. Drake"

release = get_version(project)
version = ".".join(release.split(".")[:2])

master_doc = "index"

# -- General configuration ---------------------------------------------------
extensions = [
    # myst-nb bundles the MyST Markdown parser and adds notebook support, so it
    # both renders docs/*.md and displays docs/examples/*.ipynb. Do NOT also list
    # myst_parser — myst-nb registers it and a second registration errors.
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
# NB: do NOT exclude "generated" — that's the recursive-autosummary stub dir
# (docs/generated/, git-ignored) and Sphinx must read it into the doctree.
exclude_patterns = ["_build", "build", "Thumbs.db", ".DS_Store"]

# The example notebooks are authored with jumps in heading level (e.g. an H1
# title followed by H3 section headers); that's fine for readers but trips
# myst-nb's structural linter. Suppress it rather than edit committed notebooks.
suppress_warnings = ["myst.header"]

# -- Notebooks (myst-nb) -----------------------------------------------------
# Do NOT execute notebooks at build time: the developer runs them locally and
# commits them with their outputs (some need a ~600 MB Zenodo dataset). Whatever
# outputs are committed are exactly what readers see. See docs/contributing.md.
nb_execution_mode = "off"

# MyST Markdown niceties: $...$ / $$...$$ math and ::: colon fences.
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence"]

# -- API reference (recursive autosummary) -----------------------------------
# One page per object: the autosummary stub templates (docs/_templates/
# autosummary/) drive :members: on the per-class/function pages, so we do NOT
# also set "members" as a global autodoc default — doing both documents every
# member twice (once inline on the module page, once on its own page) and trips
# -W with "duplicate object description".
autosummary_generate = True
# Don't pull `from .x import *` re-exports into the top-level xbudget page; each
# object is documented once under its defining submodule instead (default False,
# set explicitly for clarity).
autosummary_imported_members = False
autodoc_default_options = {
    "show-inheritance": True,
}
autodoc_typehints = "description"
napoleon_numpy_docstring = True
napoleon_google_docstring = False
# xbudget's runtime deps (numpy, xarray, xgcm) all pip-install cleanly, so
# nothing needs mocking. (regionate will set this to ["cartopy"].)
autodoc_mock_imports = []

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

# -- Intersphinx: the package "family" web -----------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "xgcm": ("https://xgcm.readthedocs.io/en/stable/", None),
    "xeos": ("https://xeos.readthedocs.io/en/latest/", None),
    "sectionate": ("https://sectionate.readthedocs.io/en/latest/", None),
    "regionate": ("https://regionate.readthedocs.io/en/latest/", None),
    "xwmt": ("https://xwmt.readthedocs.io/en/latest/", None),
    "xwmb": ("https://xwmb.readthedocs.io/en/latest/", None),
}

# -- Copy example notebooks from repo-root examples/ into docs/examples/ ------
# The notebooks live at the repo root so they're runnable in place; the build
# copies the ones the docs reference into docs/examples/ (git-ignored).
HERE = Path(__file__).resolve()
DOCS_DIR = HERE.parent  # docs/
REPO_ROOT = HERE.parents[1]  # repo root
EXAMPLES_SRC = REPO_ROOT / "examples"
EXAMPLES_DST = DOCS_DIR / "examples"

# Only the notebooks referenced from the docs toctree (keeps the build light and
# avoids copying scratch notebooks). Names are relative to examples/.
DOC_NOTEBOOKS = [
    "handling_missing_diagnostics.ipynb",
    "MOM6_budget_examples_mass_heat_salt.ipynb",
    "eccov4r4_budget_examples_mass_heat_salt.ipynb",
    "eccov4r4_heat_budget_decomposition.ipynb",
]


def _sync_examples():
    if EXAMPLES_DST.exists():
        shutil.rmtree(EXAMPLES_DST)  # fresh copy so removed notebooks don't linger
    EXAMPLES_DST.mkdir(parents=True, exist_ok=True)
    for name in DOC_NOTEBOOKS:
        src = EXAMPLES_SRC / name
        if src.exists():
            shutil.copy2(src, EXAMPLES_DST / name)


_sync_examples()

# -- HTML output (Furo) ------------------------------------------------------
html_theme = "furo"
html_title = f"{project} {release}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]  # styles the sidebar "View on GitHub" button

html_theme_options = {
    # `source_repository` is the repo URL our sidebar "View on GitHub" button
    # (docs/_templates/sidebar/brand.html) links to. We disable Furo's own
    # top-right view/edit icons ("top_of_page_buttons": []): they point at the
    # current page's *source*, which isn't the "back to the repo" link we want.
    "source_repository": "https://github.com/hdrake/xbudget/",
    "source_branch": "main",
    "source_directory": "docs/",
    "top_of_page_buttons": [],
    # Per-package accent color (xbudget = blue).
    # NB: Furo prepends "--" to these keys itself — do NOT include it here, or
    # you get "----color-brand-primary" and the accent silently no-ops.
    "light_css_variables": {
        "color-brand-primary": "#1a7fb5",
        "color-brand-content": "#1a7fb5",
    },
    "dark_css_variables": {
        "color-brand-primary": "#8ec6e6",
        "color-brand-content": "#8ec6e6",
    },
    # A second GitHub link, pointing at the repo home, in the footer.
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/hdrake/xbudget",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0"
                     viewBox="0 0 16 16" width="1em" height="1em">
                  <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
}
