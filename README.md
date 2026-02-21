xbudget: easy handling of budgets diagnosed from General Circulation Models with xarray
=========================

[![PyPI](https://badge.fury.io/py/xbudget.svg)](https://badge.fury.io/py/xbudget)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/xbudget)](https://anaconda.org/conda-forge/xbudget)
[![Docs](https://readthedocs.org/projects/xbudget/badge/?version=latest)](https://xbudget.readthedocs.io/en/latest/)
[![License](https://img.shields.io/github/license/hdrake/xbudget)](https://github.com/hdrake/xbudget)

## Quick Start Guide
**For users: minimal installation within an existing environment**
```bash
pip install xbudget
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