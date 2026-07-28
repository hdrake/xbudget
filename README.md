xbudget: easy handling of budgets diagnosed from General Circulation Models with xarray
=========================

[![PyPI](https://badge.fury.io/py/xbudget.svg)](https://badge.fury.io/py/xbudget)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/xbudget)](https://anaconda.org/conda-forge/xbudget)
[![Docs](https://readthedocs.org/projects/xbudget/badge/?version=latest)](https://xbudget.readthedocs.io/en/latest/)
[![License](https://img.shields.io/github/license/hdrake/xbudget)](https://github.com/hdrake/xbudget)

## Quick Start Guide
**For users: minimal installation within an existing environment**
```bash
conda install -c conda-forge xbudget
```

**For developers: installing from scratch using `conda`**
```bash
git clone git@github.com:hdrake/xbudget.git
cd xbudget
conda env create -f docs/environment.yml
conda activate docs_env_xbudget
pip install -e .
python -m ipykernel install --user --name docs_env_xbudget --display-name "docs_env_xbudget"
jupyter-lab
```

## Releasing

**The git tag is the version.** `xbudget` has no version string checked into the
source tree: `hatch-vcs` derives it from the tag at build time and writes
`xbudget/_version.py` (gitignored, but shipped inside the sdist and wheel). To
cut a release you tag; there is no file to bump and nothing to keep in sync.

1. Make sure `main` is green and has everything you want in the release, and
   move the `## Unreleased` entries in [CHANGELOG.md](CHANGELOG.md) under the
   new version heading.
2. **Publish a GitHub Release** whose tag is `vX.Y.Z`, targeting the commit you
   want to ship:
   ```bash
   gh release create vX.Y.Z --target "$(git rev-parse origin/main)" \
     --title vX.Y.Z --generate-notes
   ```
   Publishing it (not merely pushing a tag) is what fires the workflow. Target
   the commit you actually want: a tag placed *before* the commit you meant to
   release builds the previous version, and PyPI rejects it as a duplicate.
3. The **Publish to PyPI** workflow builds from that tag and uploads. It checks
   out with `fetch-depth: 0` so the tag is visible to `hatch-vcs`, and asserts
   that the built version matches the tag before publishing.
4. Verify: <https://pypi.org/project/xbudget/>.
5. conda-forge builds from the PyPI sdist and lags by design; the autotick bot
   opens the version-bump PR. The feedstock recipe must list `hatch-vcs`
   alongside `hatchling` in its `host` requirements, since it builds with
   `--no-build-isolation`.
