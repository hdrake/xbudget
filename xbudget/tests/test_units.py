"""Units: the arithmetic helpers, unit inference, stamping, and the assertion mode.

All CI-safe: the unit algebra is pure, and the inference/assertion tests run on
the synthetic grid (with units attached here rather than in the shared fixture, so
the many tests that use the unit-less synthetic grid are undisturbed).
"""
import glob
import warnings

import pytest
import yaml

import xbudget
from xbudget import units as U
from xbudget.nodes import Constant, VarRef
from xbudget.parse import BudgetParseError, parse_budgets
from xbudget.tests.conftest import build_synthetic_grid


# -- pure unit arithmetic ---------------------------------------------------

def _u(s):
    return U.parse_units(s)


class TestUnitArithmetic:
    def test_parse_unknown_is_none(self):
        assert U.parse_units(None) is None
        assert U.parse_units("") is None
        assert U.parse_units("   ") is None
        assert U.parse_units("not-a-unit") is None

    def test_parse_practical_salinity_is_dimensionless(self):
        # UDUNITS has no `psu`; PSS-78 salinity is dimensionless.
        assert U.convertible(U.parse_units("psu"), U.DIMENSIONLESS)
        assert U.convertible(U.parse_units("PSU"), _u("1"))

    def test_product_multiplies(self):
        got = U.product_units([_u("m s-1"), _u("kg m-3"), _u("m2")])
        assert U.convertible(got, _u("kg s-1"))

    def test_product_with_unknown_is_unknown(self):
        assert U.product_units([_u("m"), None]) is None

    def test_product_dimensionalizes_offset_temperature(self):
        # A heat-content product carries degC; cf-units treats it as kelvin in a
        # product, so theta * c_p * rho * dh/dt * area reconciles to W.
        got = U.product_units(
            [_u("degC"), _u("J kg-1 K-1"), _u("m s-1"), _u("kg m-3"), _u("m2")]
        )
        assert U.convertible(got, _u("W"))

    def test_sum_requires_compatibility(self):
        assert U.convertible(U.sum_units([_u("W"), _u("W")]), _u("W"))
        assert U.sum_units([_u("W"), _u("kg s-1")]) is None
        assert U.sum_units([_u("W"), None]) is None

    def test_sum_compatible_flags_only_real_clashes(self):
        assert U.sum_compatible([_u("W"), _u("J s-1")])      # convertible
        assert U.sum_compatible([_u("W"), None])             # unknown != clash
        assert not U.sum_compatible([_u("W"), _u("kg s-1")])

    def test_reciprocal_and_difference(self):
        assert U.convertible(U.reciprocal_units(_u("s")), _u("s-1"))
        assert U.reciprocal_units(None) is None
        assert U.difference_units(_u("kg s-1")) == _u("kg s-1")
        assert U.difference_units(None) is None

    def test_divergence_of_matching_fluxes(self):
        assert U.convertible(U.divergence_units(_u("kg s-1"), _u("kg s-1")), _u("kg s-1"))
        assert U.divergence_units(_u("kg s-1"), _u("W")) is None


# -- parsing constants and budget units -------------------------------------

class TestParseUnits:
    def test_bare_number_is_dimensionless(self):
        recipe = {"b": {"rhs": {"product": {"sign": -1.0, "d": "x"}}}}
        prod = parse_budgets(recipe)["b"].sides["rhs"].operations[0]
        const = dict(prod.terms)["sign"]
        assert isinstance(const, Constant)
        assert const.value == -1.0
        assert U.convertible(U.parse_units(const.units), U.DIMENSIONLESS)

    def test_dict_form_constant_carries_units(self):
        recipe = {
            "b": {"rhs": {"product": {
                "rho": {"value": 1035.0, "units": "kg m-3"}, "d": "x",
            }}}
        }
        prod = parse_budgets(recipe)["b"].sides["rhs"].operations[0]
        rho = dict(prod.terms)["rho"]
        assert rho == Constant(1035.0, units="kg m-3")

    def test_constant_value_must_be_numeric(self):
        recipe = {"b": {"rhs": {"product": {"c": {"value": "oops"}}}}}
        with pytest.raises(BudgetParseError, match="numeric 'value'"):
            parse_budgets(recipe)

    def test_constant_rejects_extra_keys(self):
        recipe = {"b": {"rhs": {"product": {
            "c": {"value": 1.0, "units": "1", "typo": 3},
        }}}}
        with pytest.raises(BudgetParseError, match="unexpected key"):
            parse_budgets(recipe)

    def test_dict_with_operation_is_a_subterm_not_constant(self):
        # A mapping that also names an operation is a sub-term even if it has a
        # `value`-named child term (only a units-constant has a bare `value`).
        recipe = {"b": {"rhs": {"sum": {
            "t": {"product": {"value": "x"}},   # `value` here is a term label
        }}}}
        sum_op = parse_budgets(recipe)["b"].sides["rhs"].operations[0]
        t = dict(sum_op.terms)["t"]
        assert not isinstance(t, Constant)

    def test_budget_units_metadata(self):
        recipe = {"mass": {"units": "kg s-1", "rhs": {"product": {"d": "x"}}}}
        assert parse_budgets(recipe)["mass"].metadata["units"] == "kg s-1"

    def test_shipped_recipes_declare_budget_units(self):
        for path in glob.glob("xbudget/recipes/*.yaml"):
            budgets = parse_budgets(yaml.safe_load(open(path)))
            for name, budget in budgets.items():
                declared = budget.metadata.get("units")
                assert declared is not None, f"{path}:{name} has no units"
                assert U.parse_units(declared) is not None, f"{path}:{name} {declared!r}"


# -- the shared `constants` table -------------------------------------------

class TestSharedConstants:
    RECIPE = {
        "constants": {
            "seawater_density": {"value": 1035.0, "units": "kg m-3"},
            "half": 0.5,
        },
        "mass": {"units": "kg s-1", "rhs": {"product": {
            "dhdt": "diag_a",
            "density": "seawater_density",
            "scale": "half",
            "area": "area",
        }}},
    }

    def test_reference_resolves_to_the_declared_constant(self):
        prod = parse_budgets(self.RECIPE)["mass"].sides["rhs"].operations[0]
        operands = dict(prod.terms)
        assert operands["density"] == Constant(1035.0, units="kg m-3")
        # A bare-number entry is dimensionless, exactly like an inline one.
        assert operands["scale"] == Constant(0.5, units="1")

    def test_constants_is_not_a_budget(self):
        assert list(parse_budgets(self.RECIPE)) == ["mass"]

    def test_unreferenced_names_stay_diagnostics(self):
        prod = parse_budgets(self.RECIPE)["mass"].sides["rhs"].operations[0]
        assert dict(prod.terms)["dhdt"] == VarRef("diag_a")

    def test_referenced_constant_carries_its_units_through(self):
        grid = build_synthetic_grid()
        grid._ds["diag_a"].attrs["units"] = "m s-1"
        grid._ds["area"].attrs["units"] = "m2"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xbudget.collect_budgets(
                grid, self.RECIPE, assert_unit_consistency=True
            )
        q = xbudget.BudgetQuery(grid, self.RECIPE)
        assert U.convertible(U.parse_units(q.units(("mass", "rhs"))), _u("kg s-1"))

    def test_reference_reaches_nested_operations(self):
        # A constant is substituted wherever an operand is parsed, including
        # inside a `difference`'s sub-term (which threads its own parse path).
        recipe = {
            "constants": {"rho": {"value": 2.0, "units": "kg m-3"}},
            "b": {"rhs": {"difference": {"t": {"product": {
                "rho": "rho", "f": "flux",
            }}}}},
        }
        diff = parse_budgets(recipe)["b"].sides["rhs"].operations[0]
        inner = dict(diff.operand.operations[0].terms)
        assert inner["rho"] == Constant(2.0, units="kg m-3")

    def test_constants_must_be_a_mapping(self):
        with pytest.raises(BudgetParseError, match="must be a dict"):
            parse_budgets({"constants": [1, 2, 3]})

    def test_constant_entry_must_be_numeric_or_mapping(self):
        with pytest.raises(BudgetParseError, match="must be a number"):
            parse_budgets({"constants": {"rho": "1035 kg m-3"}})

    def test_constant_entry_validates_like_an_inline_one(self):
        with pytest.raises(BudgetParseError, match="numeric 'value'"):
            parse_budgets({"constants": {"rho": {"value": "oops"}}})

    def test_shipped_recipes_reference_every_constant_they_declare(self):
        # A constant nothing refers to is dead weight, and (more usefully) this
        # catches a rename that updated the table but not its uses -- the stale
        # reference then degrades quietly into a diagnostic never found.
        for path in glob.glob("xbudget/recipes/*.yaml"):
            raw = yaml.safe_load(open(path))
            declared = set(raw.get("constants", {}))
            referenced = {
                s for s in _string_values(
                    {k: v for k, v in raw.items() if k != "constants"}
                )
            }
            assert declared <= referenced, (
                f"{path}: constants declared but never referenced: "
                f"{sorted(declared - referenced)}"
            )


def _string_values(node):
    """Every string value anywhere in a raw recipe dict (a possible reference)."""
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _string_values(value)
    elif isinstance(node, list):
        for value in node:
            yield from _string_values(value)


# -- static inference over the typed tree -----------------------------------

class TestInferUnits:
    def test_infers_a_product_from_the_dataset(self):
        grid = _units_grid()
        recipe = {"mass": {"rhs": {"product": {
            "d": "diag_a", "area": "area",
        }}}}
        term = parse_budgets(recipe)["mass"].sides["rhs"]
        lookup = lambda n: grid._ds[n].attrs.get("units")  # noqa: E731
        assert U.convertible(U.infer_units(term, lookup), _u("kg s-1"))

    def test_without_a_lookup_only_constants_are_known(self):
        recipe = {"b": {"rhs": {"product": {
            "rho": {"value": 1035.0, "units": "kg m-3"}, "d": "diag_a",
        }}}}
        term = parse_budgets(recipe)["b"].sides["rhs"]
        # The diagnostic's units are unknown, so the product's are too...
        assert U.infer_units(term) is None
        # ...but the constant on its own still reports what it declares.
        rho = dict(term.operations[0].terms)["rho"]
        assert U.convertible(U.infer_units(rho), _u("kg m-3"))

    def test_matches_what_the_evaluator_stamps(self):
        grid = _units_grid()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xbudget.collect_budgets(grid, CONSISTENT)
        budgets = parse_budgets(CONSISTENT)
        lookup = lambda n: grid._ds[n].attrs.get("units")  # noqa: E731
        for term in budgets["tracer"].sides.values():
            stamped = grid._ds["_".join(term.path)].attrs["units"]
            assert U.format_units(U.infer_units(term, lookup)) == stamped

    def test_leaf_var_resolves_to_its_diagnostic(self):
        grid = _units_grid()
        recipe = {"b": {"rhs": {"sum": {"direct": {"var": "flux"}}}}}
        leaf = dict(
            parse_budgets(recipe)["b"].sides["rhs"].operations[0].terms
        )["direct"]
        lookup = lambda n: grid._ds[n].attrs.get("units")  # noqa: E731
        assert U.convertible(U.infer_units(leaf, lookup), _u("kg s-1"))


# -- inference, stamping, and assert_unit_consistency ----------------------

def _units_grid():
    """The synthetic grid with UDUNITS attributes so inference has something."""
    grid = build_synthetic_grid()
    grid._ds["diag_a"].attrs["units"] = "kg m-2 s-1"
    grid._ds["diag_b"].attrs["units"] = "kg m-2 s-1"
    grid._ds["area"].attrs["units"] = "m2"
    grid._ds["flux"].attrs["units"] = "kg s-1"
    return grid


# A units-consistent tracer budget: every rhs summand reconciles to kg s-1.
CONSISTENT = {
    "tracer": {
        "units": "kg s-1",
        "rhs": {"sum": {
            "diffusion": {"product": {"sign": -1.0, "d": "diag_a", "area": "area"}},
            "boundary": {"product": {"d": "diag_b", "area": "area"}},
            "convergence": {"difference": {"transport": "flux"}},
        }},
    }
}


class TestInferenceAndAssertion:
    def test_units_are_stamped(self):
        grid = _units_grid()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xbudget.collect_budgets(grid, CONSISTENT)
        q = xbudget.BudgetQuery(grid, CONSISTENT)
        for addr in [("tracer", "rhs"), ("tracer", "rhs", "diffusion"),
                     ("tracer", "rhs", "convergence")]:
            assert U.convertible(U.parse_units(q.units(addr)), _u("kg s-1"))

    def test_constant_units_flow_into_product(self):
        grid = build_synthetic_grid()
        grid._ds["diag_a"].attrs["units"] = "m s-1"   # a thickness tendency
        grid._ds["area"].attrs["units"] = "m2"
        recipe = {"mass": {"units": "kg s-1", "rhs": {"product": {
            "dhdt": "diag_a",
            "density": {"value": 1035.0, "units": "kg m-3"},
            "area": "area",
        }}}}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xbudget.collect_budgets(grid, recipe, assert_unit_consistency=True)
        q = xbudget.BudgetQuery(grid, recipe)
        assert U.convertible(U.parse_units(q.units(("mass", "rhs"))), _u("kg s-1"))

    def test_assert_passes_when_consistent(self):
        grid = _units_grid()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Should not raise.
            xbudget.collect_budgets(grid, CONSISTENT, assert_unit_consistency=True)

    def test_assert_raises_on_mismatch(self):
        grid = _units_grid()
        mism = {"tracer": {**CONSISTENT["tracer"], "units": "W"}}  # declare wrong units
        with pytest.raises(xbudget.UnitError) as exc:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                xbudget.collect_budgets(grid, mism, assert_unit_consistency=True)
        assert exc.value.issues  # carries the offending issue list

    def test_default_warns_on_mismatch(self):
        grid = _units_grid()
        mism = {"tracer": {**CONSISTENT["tracer"], "units": "W"}}
        with pytest.warns(UserWarning, match="did not fully reconcile"):
            xbudget.collect_budgets(grid, mism, assert_unit_consistency=False)

    def test_assert_raises_on_unknown_units(self):
        grid = build_synthetic_grid()  # no units on any diagnostic
        with pytest.raises(xbudget.UnitError):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                xbudget.collect_budgets(grid, CONSISTENT, assert_unit_consistency=True)

    def test_undeclared_budget_is_not_checked(self):
        grid = _units_grid()
        no_units = {"tracer": {"rhs": CONSISTENT["tracer"]["rhs"]}}
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)  # no unit warning expected
            warnings.filterwarnings("ignore", message=".*mismatched dimensions.*")
            xbudget.collect_budgets(grid, no_units, assert_unit_consistency=True)  # must not raise

    def test_query_units_and_budget_units(self):
        grid = _units_grid()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xbudget.collect_budgets(grid, CONSISTENT)
        q = xbudget.BudgetQuery(grid, CONSISTENT)
        assert q.budget_units("tracer") == "kg s-1"
        assert q.units(("tracer", "rhs")) is not None
        # A query with no dataset knows the declared units but not inferred ones.
        q_planned = xbudget.BudgetQuery(None, CONSISTENT)
        assert q_planned.budget_units("tracer") == "kg s-1"
        assert q_planned.units(("tracer", "rhs")) is None
