"""Unit tests for parsing xbudget recipes into the typed tree."""
import glob
import os

import pytest

from xbudget.nodes import (
    Budget,
    Constant,
    Difference,
    Product,
    Sum,
    Term,
    VarRef,
)
from xbudget.parse import BudgetParseError, parse_budgets

RECIPES = sorted(
    glob.glob(
        os.path.join(os.path.dirname(__file__), "..", "recipes", "*.yaml")
    )
)


def test_parses_simple_sum():
    d = {"heat": {"rhs": {"var": None, "sum": {"var": None, "adv": "advd"}}}}
    budgets = parse_budgets(d)
    assert set(budgets) == {"heat"}
    heat = budgets["heat"]
    assert isinstance(heat, Budget)
    rhs = heat.sides["rhs"]
    assert rhs.path == ("heat", "rhs")
    assert len(rhs.operations) == 1
    (op,) = rhs.operations
    assert isinstance(op, Sum)
    (name, operand) = op.terms[0]
    assert name == "adv"
    # A bare-string operand resolves to a nested term whose single operation is
    # represented by its VarRef value; here "advd" is a leaf variable.
    assert operand == VarRef("advd")


def test_parses_product_with_constant_and_varref():
    d = {
        "mass": {
            "lhs": {
                "var": None,
                "product": {"var": None, "density": 1035.0, "area": "areacello"},
            }
        }
    }
    (op,) = parse_budgets(d)["mass"].sides["lhs"].operations
    assert isinstance(op, Product)
    operands = dict(op.terms)
    assert operands["density"] == Constant(1035.0)
    assert operands["area"] == VarRef("areacello")


def test_parses_multi_operation_term():
    """A term may carry both a product and a sum (two decompositions)."""
    d = {
        "mass": {
            "rhs": {
                "var": None,
                "product": {"var": None, "a": "x"},
                "sum": {"var": None, "b": {"var": None, "difference": {"d": "umo"}}},
            }
        }
    }
    term = parse_budgets(d)["mass"].sides["rhs"]
    kinds = [op.kind for op in term.operations]
    assert kinds == ["product", "sum"]
    sum_op = term.operations[1]
    (_, sub_term) = sum_op.terms[0]
    assert isinstance(sub_term, Term)
    assert sub_term.operations[0] == Difference(operand=VarRef("umo"))


def test_metadata_preserved_and_not_treated_as_side():
    d = {"heat": {"lambda": "thetao", "surface_lambda": "tos", "rhs": {"var": None}}}
    heat = parse_budgets(d)["heat"]
    assert heat.metadata == {"lambda": "thetao", "surface_lambda": "tos"}
    assert set(heat.sides) == {"rhs"}


def test_none_operand_is_skipped():
    d = {"heat": {"rhs": {"var": None, "sum": {"var": None, "ghost": None, "a": "x"}}}}
    (op,) = parse_budgets(d)["heat"].sides["rhs"].operations
    assert [n for n, _ in op.terms] == ["a"]


@pytest.mark.parametrize(
    "bad",
    [
        "not a dict",
        {"heat": "not a dict"},
        {"heat": {"rhs": {"var": None, "sum": "not a dict"}}},
        {"heat": {"rhs": {"var": None, "difference": "not a dict"}}},
    ],
)
def test_structural_errors_raise(bad):
    """Genuinely malformed structure (non-dict where a mapping is required)."""
    with pytest.raises(BudgetParseError):
        parse_budgets(bad)


@pytest.mark.parametrize(
    "tolerated",
    [
        # unknown key on a term (e.g. a scalar left without its enclosing product)
        {"heat": {"rhs": {"var": None, "bogus": 1.0}}},
        # difference / unary op with an unavailable (null) or ambiguous source
        {"heat": {"rhs": {"var": None, "difference": {"a": "x", "b": "y"}}}},
        {"heat": {"rhs": {"var": None, "difference": {"a": None}}}},
    ],
)
def test_tolerated_malformations_warn_and_skip(tolerated):
    """The parser mirrors the legacy engine: it warns and skips placeholders /
    malformed-but-ignorable terms rather than failing, so real recipes that
    contain unavailable-diagnostic placeholders still load."""
    with pytest.warns(UserWarning):
        budgets = parse_budgets(tolerated)
    # the offending operation is skipped, leaving the term with no operations
    rhs = budgets["heat"].sides["rhs"]
    assert rhs.operations == ()


@pytest.mark.parametrize("path", RECIPES)
def test_all_shipped_recipes_parse(path):
    import yaml

    with open(path) as f:
        budgets = parse_budgets(yaml.safe_load(f))
    assert budgets
    for budget in budgets.values():
        assert isinstance(budget, Budget)
        assert set(budget.sides) <= {"lhs", "rhs"}
