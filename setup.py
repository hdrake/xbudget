from setuptools import setup, find_namespace_packages
setup(
    name="xbudget",
    version="0.0.1",
    author="Henri Drake",
    author_email="hfdrake@uci.edu",
    description=("Helper functions and meta-data conventions for wrangling finite-volume ocean model budgets."),
    license="GPLv3",
    keywords="",
    url="https://github.com/hdrake/xbudget",
    packages=find_namespace_packages(where="src"),
    package_dir={"": "src"},
    package_data={"xbudget.conventions": ["*.yaml"]}
)
