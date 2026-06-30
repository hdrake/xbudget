"""Parse an ``xbudget_dict`` convention into the typed expression tree.

This is the single place that knows the YAML/dict schema. It validates as it
builds, so malformed conventions raise a clear ``BudgetParseError`` naming the
offending path instead of surfacing as a deep ``KeyError``/``NameError`` during
evaluation.
"""
import numbers
import warnings

from .nodes import (
    Budget,
    Constant,
    Difference,
    LateralDivergence,
    Reciprocal,
    Term,
    VarRef,
    NARY_OPS,
    SPECIAL_OPS,
    OPERATION_KEYS,
)

SIDES = ("lhs", "rhs")


class BudgetParseError(ValueError):
    """Raised when an xbudget convention does not match the expected schema."""


def _fmt(path):
    return "/".join(str(p) for p in path) or "<root>"


def parse_budgets(xbudget_dict):
    """Parse a convention dict into ``{budget_name: Budget}``.

    Parameters
    ----------
    xbudget_dict : dict
        A convention in xbudget format (e.g. from ``load_preset_budget``).

    Returns
    -------
    dict of str -> Budget
    """
    if not isinstance(xbudget_dict, dict):
        raise BudgetParseError(
            f"Top-level xbudget convention must be a dict, got "
            f"{type(xbudget_dict).__name__}."
        )

    budgets = {}
    for name, body in xbudget_dict.items():
        if not isinstance(body, dict):
            raise BudgetParseError(
                f"Budget '{name}' must be a dict, got {type(body).__name__}."
            )
        metadata = {k: v for k, v in body.items() if k not in SIDES}
        sides = {}
        for side in SIDES:
            if body.get(side) is not None:
                sides[side] = _parse_term(body[side], (name, side), side)
        budgets[name] = Budget(name=name, metadata=metadata, sides=sides)
    return budgets


def _parse_term(node, path, name):
    """Parse a single term node (a quantity defined by its operations)."""
    if not isinstance(node, dict):
        raise BudgetParseError(
            f"Term at {_fmt(path)} must be a dict, got {type(node).__name__}."
        )

    operations = []
    for key, value in node.items():
        if key == "var":
            continue
        if key in NARY_OPS:
            operations.append(_parse_nary(key, value, path))
        elif key == "difference":
            op = _parse_difference(value, path)
            if op is not None:
                operations.append(op)
        elif key == "reciprocal":
            op = _parse_reciprocal(value, path)
            if op is not None:
                operations.append(op)
        elif key in SPECIAL_OPS:
            operations.append(_parse_lateral_divergence(value, path))
        else:
            # The legacy engine silently ignores keys that are neither `var`
            # nor an operation (e.g. a `sign`/`density` scalar left directly on
            # a term because its enclosing `product:` was omitted). Mirror that
            # tolerance, but warn so the malformation is visible.
            warnings.warn(
                f"Ignoring unexpected key '{key}' on term at {_fmt(path)}; "
                f"expected 'var' or one of {sorted(OPERATION_KEYS)}. This term "
                f"may be missing an enclosing operation (e.g. 'product').",
                UserWarning,
            )

    return Term(
        name=name,
        path=path,
        operations=tuple(operations),
        explicit_var=node.get("var"),
    )


def _parse_nary(kind, body, path):
    """Parse a ``sum`` or ``product`` operation."""
    if not isinstance(body, dict):
        raise BudgetParseError(
            f"'{kind}' at {_fmt(path)} must be a dict, got "
            f"{type(body).__name__}."
        )
    terms = []
    for term_name, term_value in body.items():
        if term_name == "var":
            continue
        operand = _parse_operand(term_value, path + (term_name,), term_name)
        if operand is not None:
            terms.append((term_name, operand))
    return NARY_OPS[kind](terms=tuple(terms))


def _parse_operand(value, path, name):
    """Parse one operand of a sum/product: constant, var reference, or term."""
    if value is None:
        # Tolerated: a placeholder operand with no content (legacy behavior
        # silently skipped these).
        return None
    if isinstance(value, dict):
        return _parse_term(value, path, name)
    if isinstance(value, bool):
        raise BudgetParseError(
            f"Operand '{name}' at {_fmt(path)} is a bool; expected a number, "
            f"variable name, or sub-term."
        )
    if isinstance(value, numbers.Number):
        return Constant(float(value))
    if isinstance(value, str):
        return VarRef(value)
    raise BudgetParseError(
        f"Operand '{name}' at {_fmt(path)} has unsupported type "
        f"{type(value).__name__}."
    )


def _single_operand(kind, body, path):
    """Return the single non-``var`` (name, value) of a unary op body, or None.

    Warns and returns ``None`` for the malformed/placeholder cases the legacy
    engine tolerates (zero or several operands).
    """
    if not isinstance(body, dict):
        raise BudgetParseError(
            f"'{kind}' at {_fmt(path)} must be a dict, got "
            f"{type(body).__name__}."
        )
    operands = [(k, v) for k, v in body.items() if k != "var"]
    if len(operands) != 1:
        warnings.warn(
            f"'{kind}' at {_fmt(path)} should reference exactly one operand, "
            f"found {len(operands)}: {[k for k, _ in operands]}; skipping.",
            UserWarning,
        )
        return None
    return operands[0]


def _parse_difference(body, path):
    """Parse a ``difference`` operation.

    The operand is either a raw variable name (``VarRef``) or a nested term that
    is computed first and then differenced (``Term``). Returns ``None`` for an
    unavailable-diagnostic placeholder, matching the legacy engine.
    """
    operand = _single_operand("difference", body, path)
    if operand is None:
        return None
    name, value = operand
    if isinstance(value, str):
        return Difference(operand=VarRef(value))
    if isinstance(value, dict):
        return Difference(operand=_parse_term(value, path + (name,), name))
    warnings.warn(
        f"'difference' at {_fmt(path)} operand '{name}' is "
        f"{type(value).__name__}, not a variable or sub-term; skipping.",
        UserWarning,
    )
    return None


def _parse_reciprocal(body, path):
    """Parse a ``reciprocal`` operation (a single source variable name).

    The operand is a variable name, either bare or wrapped as ``{var: name}``.
    Returns ``None`` for an unavailable-diagnostic placeholder.
    """
    operand = _single_operand("reciprocal", body, path)
    if operand is None:
        return None
    _, value = operand
    source = value.get("var") if isinstance(value, dict) else value
    if not isinstance(source, str):
        warnings.warn(
            f"'reciprocal' at {_fmt(path)} does not reference a variable name "
            f"(got {type(source).__name__}); skipping.",
            UserWarning,
        )
        return None
    return Reciprocal(source=source)


def _parse_lateral_divergence(body, path):
    """Parse a ``lateral_divergence`` operation (an Fx/Fy flux pair)."""
    if not isinstance(body, dict):
        raise BudgetParseError(
            f"'lateral_divergence' at {_fmt(path)} must be a dict, got "
            f"{type(body).__name__}."
        )
    missing = [c for c in ("Fx", "Fy") if c not in body]
    if missing:
        raise BudgetParseError(
            f"'lateral_divergence' at {_fmt(path)} requires 'Fx' and 'Fy' "
            f"flux sub-terms; missing {missing}."
        )
    fx = _parse_term(body["Fx"], path + ("Fx",), "Fx")
    fy = _parse_term(body["Fy"], path + ("Fy",), "Fy")
    return LateralDivergence(fx=fx, fy=fy)
