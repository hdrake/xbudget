# xbudget
Helper functions and meta-data conventions for wrangling finite-volume ocean model budgets.

Quick Start Guide
-----------------

**For users: minimal installation within an existing environment**
```bash
pip install git+https://github.com/hdrake/xbudget.git@main
```

**For developers: installing from scratch using `conda`**
```bash
git clone git@github.com:hdrake/xbudget.git
cd xbudget
conda env create -f ci/environment.yml
conda activate test_env_xbudget
pip install -e .
python -m ipykernel install --user --name test_env_xbudget --display-name "test_env_xbudget"
jupyter-lab
```