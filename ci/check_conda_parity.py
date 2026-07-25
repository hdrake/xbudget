"""Check that the conda-forge recipe's run requirements match pyproject.toml.

The conda-forge autotick bot updates the version and sha256 of
recipe/meta.yaml when a new xbudget is published to PyPI, but it never
touches the dependency lists. This script compares the feedstock's run
requirements against the dependencies declared in pyproject.toml so that a
missing or stale pin shows up in CI instead of in a user's environment.

Note that the feedstock necessarily lags a release: it only changes after
the bot's version PR is merged. A failure right after tagging usually just
means that PR is still open.

Usage:
    python ci/check_conda_parity.py [--recipe PATH_OR_URL]
"""

import argparse
import re
import sys
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

import yaml

FEEDSTOCK_RECIPE = (
    "https://raw.githubusercontent.com/conda-forge/xbudget-feedstock/main/recipe/meta.yaml"
)

# PyPI project name -> conda-forge package name, for the cases where they differ.
CONDA_NAMES = {}

# Handled separately against requires-python rather than as a normal dependency.
PYTHON = "python"


def read_recipe(source):
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source, timeout=30) as response:
            return response.read().decode("utf-8")
    return Path(source).read_text()


def render_jinja(text):
    """Resolve the small subset of jinja that conda recipes use.

    Substitutes `{% set foo = "bar" %}` variables, then drops any remaining
    `{{ ... }}` expressions so the result is loadable as plain YAML.
    """
    variables = dict(re.findall(r'{%\s*set\s+(\w+)\s*=\s*"([^"]*)"\s*%}', text))
    text = re.sub(r"{%.*?%}", "", text)

    def substitute(match):
        return variables.get(match.group(1).strip(), "")

    text = re.sub(r"{{\s*([\w|.\s]+?)\s*}}", substitute, text)
    return re.sub(r"{{.*?}}", "", text)


def split_requirement(requirement):
    """Split "xgcm >=0.10.0" into ("xgcm", ">=0.10.0")."""
    name, _, constraint = requirement.strip().partition(" ")
    return name.strip().lower(), constraint.replace(" ", "")


def parse_pyproject(path):
    with open(path, "rb") as f:
        project = tomllib.load(f)["project"]
    dependencies = {}
    for dependency in project["dependencies"]:
        name, constraint = split_requirement(dependency)
        dependencies[CONDA_NAMES.get(name, name)] = constraint
    return dependencies, project.get("requires-python", "").replace(" ", "")


def parse_recipe(text):
    recipe = yaml.safe_load(render_jinja(text))
    run = recipe.get("requirements", {}).get("run") or []
    return dict(split_requirement(requirement) for requirement in run)


def compare(pyproject_deps, requires_python, recipe_deps):
    problems = []

    recipe_python = recipe_deps.pop(PYTHON, None)
    if recipe_python is None:
        problems.append("recipe run requirements do not list python")
    elif requires_python and recipe_python != requires_python:
        problems.append(
            f"python: pyproject requires-python is {requires_python!r}, "
            f"recipe pins {recipe_python!r}"
        )

    for name, constraint in sorted(pyproject_deps.items()):
        if name not in recipe_deps:
            problems.append(f"{name}: in pyproject.toml but missing from the recipe")
        elif recipe_deps[name] != constraint:
            problems.append(
                f"{name}: pyproject pins {constraint or '(unpinned)'}, "
                f"recipe pins {recipe_deps[name] or '(unpinned)'}"
            )

    for name in sorted(set(recipe_deps) - set(pyproject_deps)):
        problems.append(f"{name}: in the recipe but missing from pyproject.toml")

    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recipe",
        default=FEEDSTOCK_RECIPE,
        help="path or URL of the conda recipe (default: the conda-forge feedstock)",
    )
    args = parser.parse_args()

    try:
        recipe_text = read_recipe(args.recipe)
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        print(f"WARNING: could not read {args.recipe}: {error}")
        print("Skipping the parity check rather than failing on a transient error.")
        return 0

    root = Path(__file__).resolve().parent.parent
    pyproject_deps, requires_python = parse_pyproject(root / "pyproject.toml")
    recipe_deps = parse_recipe(recipe_text)

    problems = compare(pyproject_deps, requires_python, recipe_deps)
    if not problems:
        print(f"Recipe run requirements match pyproject.toml ({len(pyproject_deps)} deps).")
        return 0

    print(f"Recipe run requirements disagree with pyproject.toml ({args.recipe}):")
    for problem in problems:
        print(f"  - {problem}")
    print(
        "\nUpdate recipe/meta.yaml in the feedstock to match. If a version PR is "
        "still open there, merging it may be all that is needed."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
