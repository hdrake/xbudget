xbudget: easy handling of budgets diagnosed from General Circulation Models with xarray
=========================

|pypi| |conda forge| |conda-forge| |Build Status| |docs| |license|

Quick Start Guide
-----------------

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

.. |conda forge| image:: https://img.shields.io/conda/vn/conda-forge/xbudget
   :target: https://anaconda.org/conda-forge/xbudget
.. |Build Status| image:: https://img.shields.io/github/workflow/status/hdrake/xbudget/CI?logo=github
   :target: https://github.com/hdrake/xbudget/actions
   :alt: GitHub Workflow CI Status
.. |pypi| image:: https://badge.fury.io/py/xbudget.svg
   :target: https://badge.fury.io/py/xbudget
   :alt: pypi package
.. |docs| image:: http://readthedocs.org/projects/xbudget/badge/?version=latest
   :target: http://xgcm.readthedocs.org/en/latest/?badge=latest
   :alt: documentation status
.. |license| image:: https://img.shields.io/github/license/mashape/apistatus.svg
   :target: https://github.com/hdrake/xbudget
   :alt: license
.. |conda-forge| image:: https://img.shields.io/conda/dn/conda-forge/xbudget?label=conda-forge
   :target: https://anaconda.org/conda-forge/xbudget