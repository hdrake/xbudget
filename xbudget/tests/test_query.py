"""Tests for the query layer (BudgetQuery).

These all run on the synthetic grid, deliberately: CI has neither example
dataset, so anything gated on real data would never protect this code.
"""
import copy
import glob
import os
import warnings

import pytest
import yaml

import xbudget
from xbudget.query import BudgetQuery

from conftest import SYNTHETIC_PRESET

RECIPES = sorted(
    glob.glob(
        os.path.join(os.path.dirname(__file__), "..", "recipes", "*.yaml")
    )
)


@pytest.fixture
def collected(synthetic_grid, synthetic_preset):
    """A grid with the synthetic budget collected + its query."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # 'not_present' is missing by design
        xbudget.collect_budgets(synthetic_grid, synthetic_preset)
    return synthetic_grid, BudgetQuery(synthetic_grid, synthetic_preset)


# -- resolution -------------------------------------------------------------


def test_var_resolves_primary(collected):
    grid, q = collected
    assert q.var("tracer_rhs_diffusion") == "tracer_rhs_diffusion"
    assert q.var("tracer_rhs_diffusion") in grid._ds


def test_var_by_path_tuple(collected):
    _, q = collected
    assert q.var(("tracer", "rhs", "diffusion")) == q.var("tracer_rhs_diffusion")


def test_var_multi_op_term(collected):
    """A term with two operations: the product is primary, the sum is suffixed."""
    grid, q = collected
    assert q.var("tracer_rhs_boundary") == "tracer_rhs_boundary"
    assert "tracer_rhs_boundary_sum" in grid._ds
    # ...and they are genuinely different arrays, not aliases.
    assert not grid._ds["tracer_rhs_boundary"].equals(grid._ds["tracer_rhs_boundary_sum"])


def test_var_non_primary_op_is_addressable(collected):
    """Every variable in the dataset can be looked up by the name it has.

    The sum of a multi-operation term is emitted under its suffixed name, so
    that name must resolve — otherwise a user who sees the variable in their
    dataset cannot ask the query layer about it.
    """
    grid, q = collected
    assert q.var("tracer_rhs_boundary_sum") == "tracer_rhs_boundary_sum"
    assert q.var("tracer_rhs_boundary_sum") in grid._ds


def test_get_vars_non_primary_op_reports_its_own_operands(collected):
    _, q = collected
    out = q.get_vars("tracer_rhs_boundary_sum")
    assert out["var"] == "tracer_rhs_boundary_sum"
    # the sum's operand, not the sibling product's
    assert out["sum"] == ["tracer_rhs_boundary_convergence"]


def test_every_emitted_variable_is_addressable(collected):
    """No xbudget-created variable is orphaned from the query layer."""
    grid, q = collected
    created = {
        v for v in grid._ds.data_vars if "xbudget_path" in grid._ds[v].attrs
    }
    assert created  # guard against the assertion passing vacuously
    for name in created:
        assert q.var(name) == name, f"{name} in dataset but not addressable"


def test_var_leaf_explicit_var(collected):
    """A leaf naming a diagnostic resolves to the renamed copy, not the raw name."""
    grid, q = collected
    assert q.var("tracer_rhs_direct") == "tracer_rhs_direct"
    assert grid._ds["tracer_rhs_direct"].equals(grid._ds["diag_a"])


def test_var_unmaterialized_is_none(collected):
    """A term whose diagnostic was absent resolves to None, not a bad name."""
    _, q = collected
    assert q.var("tracer_rhs_missing") is None


def test_var_primary_op_skipped_uses_runtime_name(synthetic_grid, synthetic_preset_skips):
    """Names follow the run, not the recipe.

    The `renamed` term lists a product first, so a structural reading predicts
    the product owns the bare name and the sum is "tracer_rhs_renamed_sum". But
    the product's diagnostic is absent, so at run time the sum claims the bare
    name. The query layer must report what was actually emitted.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xbudget.collect_budgets(synthetic_grid, synthetic_preset_skips)
    q = BudgetQuery(synthetic_grid, synthetic_preset_skips)

    assert "tracer_rhs_renamed" in synthetic_grid._ds
    assert "tracer_rhs_renamed_sum" not in synthetic_grid._ds
    assert q.var("tracer_rhs_renamed") == "tracer_rhs_renamed"


# -- get_vars ---------------------------------------------------------------


def test_get_vars_sum_children(collected):
    _, q = collected
    out = q.get_vars("tracer_rhs")
    assert out["var"] == "tracer_rhs"
    # 'missing' is dropped; the rest keep recipe order.
    assert out["sum"] == [
        "tracer_rhs_diffusion",
        "tracer_rhs_boundary",
        "tracer_rhs_direct",
    ]


def test_get_vars_product_operands(collected):
    """Constants and diagnostics are reported inline, in recipe order."""
    _, q = collected
    assert q.get_vars("tracer_rhs_diffusion")["product"] == [-1.0, "diag_a", "area"]


def test_get_vars_raw_diagnostic(collected):
    _, q = collected
    out = q.get_vars("diag_a")
    assert out["var"] == "diag_a"
    assert ("tracer", "rhs", "direct") in out["referenced_by"]


def test_get_vars_batch_list(collected):
    _, q = collected
    out = q.get_vars(["tracer_rhs", "tracer_rhs_diffusion", "diag_a"])
    assert isinstance(out, list) and len(out) == 3
    assert out[1]["var"] == "tracer_rhs_diffusion"


def test_get_vars_tuple_is_one_address_not_a_batch(collected):
    _, q = collected
    out = q.get_vars(("tracer", "rhs", "diffusion"))
    assert isinstance(out, dict)
    assert out["var"] == "tracer_rhs_diffusion"


def test_get_vars_unknown_raises_with_suggestion(collected):
    _, q = collected
    with pytest.raises(KeyError) as exc:
        q.get_vars("tracer_rhs_diffusio")
    assert "tracer_rhs_diffusion" in str(exc.value)


def test_get_vars_operator_infixed_name_raises_with_suggestion(collected):
    """An operator-infixed name (not produced by the engine) is rejected."""
    _, q = collected
    with pytest.raises(KeyError) as exc:
        q.get_vars("tracer_rhs_sum_diffusion")
    # the close-match hint points at the real name
    assert "tracer_rhs_diffusion" in str(exc.value)


# -- aggregate --------------------------------------------------------------


def test_aggregate_flat(collected):
    _, q = collected
    assert q.aggregate()["tracer"]["rhs"] == {
        "diffusion": "tracer_rhs_diffusion",
        "boundary": "tracer_rhs_boundary",
        "direct": "tracer_rhs_direct",
    }


def test_aggregate_decompose_list(collected):
    _, q = collected
    rhs = q.aggregate(decompose=["boundary"])["tracer"]["rhs"]
    assert rhs["boundary_convergence"] == "tracer_rhs_boundary_convergence"
    assert "boundary" not in rhs
    assert rhs["diffusion"] == "tracer_rhs_diffusion"  # untouched


def test_aggregate_decompose_str_equals_single_item_list(collected):
    _, q = collected
    assert q.aggregate(decompose="boundary") == q.aggregate(decompose=["boundary"])


def test_aggregate_decompose_matches_exactly_not_substring(collected):
    """`decompose` matches a term name exactly, not as a substring."""
    _, q = collected
    assert q.aggregate(decompose="boundary_and_more") == q.aggregate()


def test_aggregate_decompose_sum_whose_only_child_shares_its_name(synthetic_grid):
    """A one-child sum must still decompose, even if the child repeats the label.

    `_aggregate_term` returns {term.name: var} for a term with no sum, which is
    shape-identical to a real one-child sum keyed by the parent's own name.
    Telling them apart by shape silently no-ops the decompose.
    """
    preset = {
        "tracer": {
            "rhs": {
                "sum": {
                    "boundary": {"sum": {"boundary": {"product": {"d": "diag_b"}}}},
                    "other": {"product": {"d": "diag_a"}},
                }
            }
        }
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xbudget.collect_budgets(synthetic_grid, preset)
    q = BudgetQuery(synthetic_grid, preset)

    rhs = q.aggregate(decompose=["boundary"])["tracer"]["rhs"]
    assert rhs["boundary_boundary"] == "tracer_rhs_boundary_boundary"
    assert "boundary" not in rhs
    assert rhs["boundary_boundary"] in synthetic_grid._ds


def test_product_with_missing_factor_is_dropped_not_zeroed(synthetic_grid):
    """A product missing a required factor is not materialized at all.

    The evaluator no longer multiplies in 0.0 (which fabricated an identically
    zero variable that read as a real, null contribution). An unknown factor is
    not a zero one, so the whole term is dropped and reported as such.
    """
    preset = {"tracer": {"rhs": {"product": {"d": "not_present", "area": "area"}}}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xbudget.collect_budgets(synthetic_grid, preset)
    q = BudgetQuery(synthetic_grid, preset)

    # No zero-filled variable landed in the dataset.
    assert "tracer_rhs" not in synthetic_grid._ds
    assert q.var("tracer_rhs") is None
    assert q.is_complete("tracer_rhs") is False
    # get_vars lists the surviving factor and names the one that went missing —
    # never a fake 0.0.
    out = q.get_vars("tracer_rhs")
    assert out["var"] is None
    assert out["product"] == ["area"]
    assert out["missing"] == ["not_present"]
    assert q.missing("tracer_rhs") == ["not_present"]


def test_aggregate_preserves_budget_metadata(collected):
    _, q = collected
    assert q.aggregate()["tracer"]["lambda"] == "tracer_concentration"


def test_aggregate_drops_unmaterialized(collected):
    _, q = collected
    assert "missing" not in q.aggregate()["tracer"]["rhs"]


# -- budget metadata --------------------------------------------------------


def test_metadata_single_and_all(collected):
    _, q = collected
    assert q.metadata("tracer") == {"lambda": "tracer_concentration"}
    assert q.metadata() == {"tracer": {"lambda": "tracer_concentration"}}


def test_contains_rejects_unknown_names(collected):
    """`in` must be a real membership test, not always True.

    `_lookup` returns None for an unrecognized string rather than raising, so a
    bare try/except reported every string as present.
    """
    _, q = collected
    assert "totally_bogus_name" not in q
    assert ("no", "such", "path") not in q
    assert 42 not in q
    # ...while everything `var()` accepts is still reported as present
    assert "tracer_rhs_diffusion" in q            # a term by name
    assert ("tracer", "rhs", "diffusion") in q    # a term by path
    assert "diag_a" in q                          # a raw diagnostic
    assert "tracer_rhs_boundary_sum" in q         # an operation-suffixed name


def test_metadata_unknown_budget_raises(collected):
    _, q = collected
    with pytest.raises(KeyError):
        q.metadata("does_not_exist")


def test_metadata_is_a_copy(collected):
    """Mutating the returned dict must not corrupt the query's internal state.

    Both branches copy: `metadata(name)` returns one budget's dict, and
    `metadata()` returns a dict of them -- the inner dicts of the latter must be
    copies too, or mutating one silently rewrites the parsed recipe.
    """
    _, q = collected
    q.metadata("tracer")["lambda"] = "tampered"
    assert q.lambda_var("tracer") == "tracer_concentration"

    q.metadata()["tracer"]["lambda"] = "tampered too"
    assert q.lambda_var("tracer") == "tracer_concentration"


def test_metadata_accessors():
    """thickness/lambda_var/surface_lambda read a budget's metadata keys."""
    preset = {
        "mass": {"lambda": "density", "thickness": "thkcello", "rhs": {}},
        "heat": {"lambda": "thetao", "surface_lambda": "tos", "rhs": {}},
    }
    q = BudgetQuery(None, preset)
    assert q.thickness() == "thkcello"  # defaults to the mass budget
    assert q.thickness("mass") == "thkcello"
    assert q.lambda_var("mass") == "density"
    assert q.lambda_var("heat") == "thetao"
    assert q.surface_lambda("heat") == "tos"
    # Absent metadata keys resolve to None rather than raising.
    assert q.thickness("heat") is None
    assert q.surface_lambda("mass") is None
    # An unknown budget is an error, not None -- including the defaulted "mass".
    with pytest.raises(KeyError):
        q.lambda_var("does_not_exist")
    with pytest.raises(KeyError):
        BudgetQuery(None, {"heat": {"lambda": "thetao", "rhs": {}}}).thickness()


@pytest.mark.parametrize("path", RECIPES, ids=lambda p: os.path.basename(p))
def test_shipped_recipes_declare_expected_metadata(path):
    """Pin the accessors to the key names the shipped recipes actually use.

    The other metadata tests build their own recipes, so they only prove the
    accessors read the keys the *test* wrote. This pins them to the real files:
    renaming `thickness`/`lambda` in the YAML would otherwise break every
    downstream consumer (xwmt reads the mass thickness to build its vertical
    metrics) with a green suite.
    """
    with open(path) as f:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # tolerated placeholders
            q = BudgetQuery(None, yaml.safe_load(f))

    assert "mass" in q.budgets, "every shipped recipe declares a mass budget"
    assert q.thickness() == "thkcello"
    for name in q.budgets:
        assert q.lambda_var(name) is not None, f"{name} declares no `lambda`"


# -- recipes ------------------------------------------------------------


@pytest.mark.parametrize("path", RECIPES, ids=lambda p: os.path.basename(p))
def test_all_shipped_recipes_queryable(path):
    """Every shipped recipe builds a query, with no colliding output names.

    `"_".join(path)` is not injective — ("a", "b_c") and ("a", "b", "c") both
    give "a_b_c" — and a collision would mean one term silently overwriting
    another's variable in the dataset.
    """
    with open(path) as f:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # tolerated placeholders
            q = BudgetQuery(None, yaml.safe_load(f))
    assert q.duplicate_names == {}
    # Every term is addressable, and the aggregate of every budget is non-empty
    # (a recipe whose sides collapse to nothing is a broken recipe).
    terms = q.terms()
    assert terms
    for path_, name in terms.items():
        assert q.var(path_) == name
    agg = q.aggregate()
    for budget, body in agg.items():
        for side in ("lhs", "rhs"):
            if side in body:
                assert body[side], f"{budget}/{side} aggregated to nothing"


def test_query_without_data_reports_planned_names():
    """With no dataset there is nothing to check against, so nothing is dropped."""
    q = BudgetQuery(None, copy.deepcopy(SYNTHETIC_PRESET))
    assert "missing" in q.aggregate()["tracer"]["rhs"]
