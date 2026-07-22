"""`var: null` placeholders are optional.

A recipe only needs ``var`` when it is *naming* a diagnostic
(``var: "thetao"``). The ``var: null`` placeholders that mark a derived
quantity carry no information — an absent key already means the same thing —
so a recipe may omit them entirely.

These tests pin that equivalence so the terse form stays supported rather than
working by accident.
"""
import copy
import warnings

import numpy as np
import pytest
import xarray as xr
import yaml

import xbudget
from xbudget.parse import parse_budgets
from xbudget.query import BudgetQuery

# The same budget written both ways: with the placeholders, and without.
VERBOSE = {
    "heat": {
        "lambda": "thetao",
        "lhs": {"var": None, "sum": {"var": None, "tendency": {"var": "tend"}}},
        "rhs": {
            "var": None,
            "sum": {
                "var": None,
                "advection": {"var": "adv"},
                "forcing": {
                    "var": None,
                    "product": {"var": None, "sign": -1.0, "flux": "flx", "area": "area"},
                },
            },
        },
    }
}

TERSE = {
    "heat": {
        "lambda": "thetao",
        "lhs": {"sum": {"tendency": {"var": "tend"}}},
        "rhs": {
            "sum": {
                "advection": {"var": "adv"},
                "forcing": {"product": {"sign": -1.0, "flux": "flx", "area": "area"}},
            }
        },
    }
}


@pytest.fixture
def ds():
    rng = np.random.default_rng(0)
    return xr.Dataset(
        {
            "tend": (("x",), rng.random(4)),
            "adv": (("x",), rng.random(4)),
            "flx": (("x",), rng.random(4)),
            "area": (("x",), rng.random(4) + 1.0),
        },
        coords={"x": [0, 1, 2, 3]},
    )


def _collect(ds, recipe, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xbudget.collect_budgets(ds, copy.deepcopy(recipe), **kw)
    return ds


def test_parse_trees_are_identical():
    """The placeholders carry no information, so both parse to the same tree."""
    assert parse_budgets(copy.deepcopy(VERBOSE)) == parse_budgets(copy.deepcopy(TERSE))


def test_terse_produces_the_same_variables_and_values(ds):
    a = _collect(ds.copy(deep=True), VERBOSE)
    b = _collect(ds.copy(deep=True), TERSE)
    assert set(a.data_vars) == set(b.data_vars)
    derived = [v for v in a.data_vars if "xbudget_path" in a[v].attrs]
    assert derived  # guard against passing vacuously
    for v in derived:
        xr.testing.assert_identical(a[v], b[v])


def test_terse_queries_the_same(ds):
    a, b = ds.copy(deep=True), ds.copy(deep=True)
    _collect(a, VERBOSE)
    _collect(b, TERSE)
    qa = BudgetQuery(a, copy.deepcopy(VERBOSE))
    qb = BudgetQuery(b, copy.deepcopy(TERSE))
    assert qa.aggregate() == qb.aggregate()
    assert qa.var("heat_rhs_forcing") == qb.var("heat_rhs_forcing")
    assert qa.get_vars("heat_rhs") == qb.get_vars("heat_rhs")


@pytest.mark.parametrize(
    "spelling", ["{var: null}", "{}"], ids=["var-null", "empty-dict"]
)
def test_contentless_node_spellings_agree(ds, spelling):
    """A node with no operations produces nothing, however you spell it.

    Deleting the `var: null` line from a node whose *only* key it was leaves an
    empty node. `{}` and `{var: null}` must stay equivalent; note that removing
    the key *and* the braces yields YAML null, which is a different thing (the
    operand is dropped at parse) but has the same effect here: no variable.
    """
    recipe = {"heat": {"rhs": {"sum": yaml.safe_load(f"placeholder: {spelling}")}}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xbudget.collect_budgets(ds, recipe)
    q = BudgetQuery(ds, recipe)
    assert q.aggregate()["heat"]["rhs"] == {}
    assert not [v for v in ds.data_vars if "xbudget_path" in ds[v].attrs]


def test_decompose_of_an_unmaterialized_term(ds):
    """Decomposing a term that never materialized, from a terse recipe.

    The unmaterialized `forcing` term (its diagnostic is absent) must drop out
    of the aggregate rather than leak a partial entry, even with the `var: null`
    placeholders omitted.
    """
    recipe = {
        "heat": {
            "rhs": {
                "sum": {
                    "forcing": {"product": {"d": "not_in_dataset"}},
                    "advection": {"var": "adv"},
                }
            }
        }
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xbudget.collect_budgets(ds, recipe)
        rhs = BudgetQuery(ds, recipe).aggregate(decompose=["forcing"])["heat"]["rhs"]
    # the unmaterialized term drops out; it must not leak the raw node dict
    assert "forcing" not in rhs
    assert rhs == {"advection": "heat_rhs_advection"}


def test_bare_null_operand_is_also_dropped(ds):
    """`placeholder:` with nothing under it is YAML null -> skipped at parse."""
    recipe = {"heat": {"rhs": {"sum": yaml.safe_load("placeholder:")}}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xbudget.collect_budgets(ds, recipe)
    assert BudgetQuery(ds, recipe).aggregate()["heat"]["rhs"] == {}
