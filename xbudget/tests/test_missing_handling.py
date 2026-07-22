"""Missing-diagnostic handling: don't mislead about a term that isn't there.

A recipe is meant to serve datasets with different diagnostics available, so a
term referencing an absent variable is skipped rather than fatal. These tests pin
that the skip is never silent-and-forgotten:

* a partial ``sum`` is flagged incomplete and records what it was built without;
* a ``product`` missing a required factor is dropped, not fabricated as zero (A);
* incompleteness propagates up to every ancestor;
* ``on_missing`` selects warn / raise / ignore;
* ``optional: true`` declares an expected absence, exempt from all of the above.

All synthetic, so they run in CI.
"""
import warnings

import numpy as np
import pytest
import xarray as xr

import xbudget
from xbudget import MissingDiagnosticError
from xbudget.query import BudgetQuery


@pytest.fixture
def ds():
    rng = np.random.default_rng(0)
    return xr.Dataset(
        {
            "adv": (("x",), rng.random(4)),
            "dif": (("x",), rng.random(4)),
            "area": (("x",), rng.random(4) + 1.0),
        },
        coords={"x": [0, 1, 2, 3]},
    )


# A sum with one absent operand ("surface" names a diagnostic not in `ds`).
PARTIAL_SUM = {
    "heat": {
        "rhs": {
            "sum": {
                "advection": {"var": "adv"},
                "diffusion": {"var": "dif"},
                "surface": {"var": "missing_sfc"},
            }
        }
    }
}


def _collect(ds, recipe, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xbudget.collect_budgets(ds, recipe, **kw)
    return ds


# -- A: no fabricated zeros -------------------------------------------------


def test_product_missing_factor_is_dropped(ds):
    recipe = {"heat": {"rhs": {"product": {"flux": "missing_flux", "area": "area"}}}}
    _collect(ds, recipe)
    assert "heat_rhs" not in ds  # not a zero-filled array
    q = BudgetQuery(ds, recipe)
    assert q.var("heat_rhs") is None
    assert q.is_complete("heat_rhs") is False
    assert q.missing("heat_rhs") == ["missing_flux"]


def test_sum_keeps_survivors_but_flags_incomplete(ds):
    _collect(ds, PARTIAL_SUM)
    # The sum still builds from the two present operands...
    assert "heat_rhs" in ds
    np.testing.assert_allclose(ds["heat_rhs"].values, ds["adv"].values + ds["dif"].values)
    # ...and says, durably, that it is not the whole story.
    assert ds["heat_rhs"].attrs["xbudget_incomplete"] == 1
    assert ds["heat_rhs"].attrs["xbudget_missing"] == ["surface"]


def test_complete_budget_carries_no_incompleteness_attrs(ds):
    """The common case stays clean: nothing extra stamped when nothing is missing."""
    recipe = {"heat": {"rhs": {"sum": {"advection": {"var": "adv"}, "diffusion": {"var": "dif"}}}}}
    _collect(ds, recipe)
    for v in ds.data_vars:
        attrs = ds[v].attrs
        assert "xbudget_incomplete" not in attrs
        assert "xbudget_missing" not in attrs


# -- B: incompleteness propagates upward ------------------------------------


def test_incompleteness_propagates_to_ancestors(ds):
    recipe = {
        "heat": {
            "rhs": {
                "sum": {
                    "group": {"sum": {"a": {"var": "adv"}, "b": {"var": "missing_b"}}},
                    "other": {"var": "dif"},
                }
            }
        }
    }
    _collect(ds, recipe)
    q = BudgetQuery(ds, recipe)
    # The inner sum dropped "b"; the outer sum inherits incompleteness even
    # though it dropped nothing of its own.
    assert q.is_complete("heat_rhs_group") is False
    assert q.is_complete("heat_rhs") is False
    assert ds["heat_rhs"].attrs["xbudget_incomplete"] == 1
    assert "xbudget_missing" not in ds["heat_rhs"].attrs  # no *direct* drop
    assert q.missing("heat_rhs") == []            # the culprit is the descendant
    assert q.missing("heat_rhs_group") == ["b"]
    assert set(q.incomplete_terms()) == {"heat_rhs", "heat_rhs_group"}


def test_missing_map_lists_every_gap(ds):
    _collect(ds, PARTIAL_SUM)
    q = BudgetQuery(ds, PARTIAL_SUM)
    m = q.missing()
    assert m[("heat", "rhs")] == ["surface"]                 # the partial sum
    assert m[("heat", "rhs", "surface")] == ["missing_sfc"]  # the absent leaf


# -- D: on_missing policy ---------------------------------------------------


def test_on_missing_warn_emits_one_summary(ds):
    with pytest.warns(UserWarning, match="missing diagnostic") as record:
        xbudget.collect_budgets(ds, PARTIAL_SUM, on_missing="warn")
    assert len(record) == 1  # one summary, not one-per-operand
    msg = str(record[0].message)
    assert "missing_sfc" in msg
    assert "heat_rhs" in msg  # names the now-incomplete term


def test_on_missing_raise(ds):
    with pytest.raises(MissingDiagnosticError) as exc:
        xbudget.collect_budgets(ds, PARTIAL_SUM, on_missing="raise")
    assert "missing_sfc" in str(exc.value)
    # the structured attribute lists every problem, not just the first
    diagnostics = {d for d, _path in exc.value.missing}
    assert "missing_sfc" in diagnostics


def test_on_missing_ignore_is_silent_but_still_stamps(ds):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would fail the test
        xbudget.collect_budgets(ds, PARTIAL_SUM, on_missing="ignore")
    # silent, yet the durable record is still there
    assert ds["heat_rhs"].attrs["xbudget_incomplete"] == 1
    assert ds["heat_rhs"].attrs["xbudget_missing"] == ["surface"]


def test_on_missing_invalid_value(ds):
    with pytest.raises(ValueError, match="on_missing"):
        xbudget.collect_budgets(ds, PARTIAL_SUM, on_missing="explode")


# -- optional: expected absence, declared in the recipe ---------------------

OPTIONAL_SUM = {
    "heat": {
        "rhs": {
            "sum": {
                "advection": {"var": "adv"},
                "surface": {"optional": True, "var": "missing_sfc"},
            }
        }
    }
}


def test_optional_absence_is_silent_and_not_incomplete(ds):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # optional must not warn
        xbudget.collect_budgets(ds, OPTIONAL_SUM, on_missing="warn")
    assert "heat_rhs" in ds
    np.testing.assert_allclose(ds["heat_rhs"].values, ds["adv"].values)
    assert "xbudget_incomplete" not in ds["heat_rhs"].attrs
    q = BudgetQuery(ds, OPTIONAL_SUM)
    assert q.is_complete("heat_rhs") is True
    assert q.incomplete_terms() == []
    assert q.missing() == {}  # the optional gap is excluded


def test_optional_does_not_raise(ds):
    # on_missing="raise" would fire for a required miss; optional is exempt.
    xbudget.collect_budgets(ds, OPTIONAL_SUM, on_missing="raise")
    assert "heat_rhs" in ds


def test_optional_subtree_suppresses_nested_misses(ds):
    """`optional` silences the *whole* subtree, not just its top node."""
    recipe = {
        "heat": {
            "rhs": {
                "sum": {
                    "advection": {"var": "adv"},
                    "eddy": {
                        "optional": True,
                        "product": {"flux": "missing_flux", "area": "area"},
                    },
                }
            }
        }
    }
    xbudget.collect_budgets(ds, recipe, on_missing="raise")  # must not raise
    q = BudgetQuery(ds, recipe)
    assert q.is_complete("heat_rhs") is True
    assert q.missing() == {}


# -- get_vars surfaces the gap ----------------------------------------------


def test_get_vars_reports_missing_operands(ds):
    _collect(ds, PARTIAL_SUM)
    q = BudgetQuery(ds, PARTIAL_SUM)
    out = q.get_vars("heat_rhs")
    assert out["sum"] == ["heat_rhs_advection", "heat_rhs_diffusion"]
    assert out["missing"] == ["surface"]


# -- recipe-only (no dataset): completeness is unknowable -------------------


def test_completeness_unknown_without_data():
    q = BudgetQuery(None, PARTIAL_SUM)
    assert q.is_complete("heat_rhs") is None
    assert q.incomplete_terms() == []
    with pytest.raises(ValueError, match="needs a dataset"):
        q.missing()
