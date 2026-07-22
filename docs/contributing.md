# Contributor guide

Contributions are welcome — bug reports, recipes for new models, docs, and code.
Please [open an issue or pull request](https://github.com/hdrake/xbudget/issues).

## Development environment

Clone the repository and create the development environment:

```bash
git clone https://github.com/hdrake/xbudget.git
cd xbudget
conda env create -f docs/environment.yml
conda activate docs_env_xbudget
pip install -e .
```

Run the test suite with `pytest`. The end-to-end tests that need the ~600 MB
example MOM6/ECCO datasets (fetched from Zenodo) skip automatically when the data
is absent.

## Example notebooks

Example notebooks live at the repository root in `examples/` so they run in
place, and are copied into `docs/examples/` at build time.

```{important}
Run notebooks locally and commit them **with their outputs**. The documentation
build does **not** execute notebooks (`nb_execution_mode = "off"` in
`docs/conf.py`), so whatever outputs you commit are exactly what readers see.
Re-run and re-commit a notebook whenever its code or results change.
```

This is deliberate: several notebooks depend on large datasets downloaded from
Zenodo, so executing them in CI or on Read the Docs would be slow and fragile.

## Building the docs

The documentation uses Sphinx + [Furo](https://pradyunsg.me/furo/) +
[myst-nb](https://myst-nb.readthedocs.io/). Build it locally exactly as CI and
Read the Docs do (warnings are errors):

```bash
pip install -r docs/requirements.txt
python -m sphinx -b html -W --keep-going docs docs/_build/html
```

Then open `docs/_build/html/index.html`.
