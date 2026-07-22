"""Load and save xbudget recipes.

A recipe is an ordinary nested dict, so it can be written by hand in YAML or
built in Python. These helpers move it between the two, and validate it on the
way out — see :doc:`the recipes guide </recipes>` for the schema.
"""
import yaml
from os import path

from .parse import parse_budgets, _resolve_recipe

__all__ = ["load_preset_budget", "load_yaml", "save_yaml"]


def load_preset_budget(model="MOM6"):
    """Loads preset xbudget dictionary from yaml file for supported models.

    Parameters
    ----------
    model : str (default "MOM6")
      Name of any recipe shipped in ``xbudget/recipes/`` (without the
      ``.yaml`` suffix). Currently: "MOM6", "MOM6_3Donly", "MOM6_drift",
      "MOM6_surface", "ECCOV4r4_native".
      Please open an Issue if you would like to contribute an xbudget yaml
      file for a new model–see /recipes/ for examples.

    Returns
    -------
    Python dictionary

    See also
    --------
    load_yaml, save_yaml, xbudget.parse_budgets
    """
    return load_yaml(f"{path.dirname(__file__)}/recipes/{model}.yaml")


def load_yaml(filepath):
    """Loads a yaml file as a Python dictionary.

    This is a thin reader: it does not check that the result is a valid xbudget
    recipe, because the shipped recipes contain placeholder terms that
    the parser deliberately warns about and skips, and re-reading a file should
    not re-raise those warnings. To validate, call
    :func:`xbudget.parse_budgets` on the result (or use :func:`save_yaml`, which
    validates before writing).

    Parameters
    ----------
    filepath : path to yaml file, as str

    Returns
    -------
    Python dictionary

    Raises
    ------
    yaml.YAMLError
        If the file is not valid YAML.

    See also
    --------
    save_yaml, xbudget.parse_budgets
    """
    with open(filepath, "r") as stream:
        return yaml.safe_load(stream)


def save_yaml(recipe=None, filepath=None, *, xbudget_dict=None):
    """Write a recipe to a yaml file, after checking that it is valid.

    Validating here means a malformed recipe cannot reach disk: you get a
    :class:`xbudget.BudgetParseError` naming the offending path while you are
    authoring it, rather than a confusing failure the next time it is loaded.

    Key order is preserved rather than sorted, because it is meaningful: the
    operands of a ``sum``/``product`` are reported in recipe order by
    :meth:`xbudget.BudgetQuery.get_vars`.

    Parameters
    ----------
    recipe : dict
        A recipe in xbudget format.
    filepath : str
        Path to write to.
    xbudget_dict : dict, optional
        Deprecated alias for ``recipe``; removed in xbudget v1.0.

    Raises
    ------
    xbudget.BudgetParseError
        If the recipe does not match the schema. Nothing is written.

    Examples
    --------
    >>> recipe = {"heat": {"rhs": {"var": None, "sum": {
    ...     "var": None, "forcing": {"var": "surface_heat_flux"}}}}}
    >>> xbudget.save_yaml(recipe, "my_model.yaml")
    >>> xbudget.load_yaml("my_model.yaml") == recipe
    True

    See also
    --------
    load_yaml, xbudget.parse_budgets
    """
    recipe = _resolve_recipe(recipe, xbudget_dict, "save_yaml")
    if filepath is None:
        raise TypeError("save_yaml() missing required argument: 'filepath'")
    parse_budgets(recipe)  # raises BudgetParseError before we write
    with open(filepath, "w") as stream:
        yaml.safe_dump(
            recipe, stream, sort_keys=False, default_flow_style=False
        )
