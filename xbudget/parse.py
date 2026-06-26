"""Parse an ``xbudget_dict`` convention into the typed expression tree.

This is the single place that knows the YAML/dict schema. It validates as it
builds, so malformed conventions raise a clear ``BudgetParseError`` naming the
offending path instead of surfacing as a deep ``KeyError``/``NameError`` during
evaluation.
"""
import numbers

from .nodes import (
    Budget,
    Constant,
    Term,
    VarRef,
    NARY_OPS,
    UNARY_OPS,
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
        elif key in UNARY_OPS:
            operations.append(_parse_unary(key, value, path))
        else:
            raise BudgetParseError(
                f"Unexpected key '{key}' on term at {_fmt(path)}; expected "
                f"'var' or one of {sorted(OPERATION_KEYS)}."
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


def _parse_unary(kind, body, path):
    """Parse a ``difference`` operation (a single source variable)."""
    if not isinstance(body, dict):
        raise BudgetParseError(
            f"'{kind}' at {_fmt(path)} must be a dict, got "
            f"{type(body).__name__}."
        )
    sources = [(k, v) for k, v in body.items() if k != "var"]
    if len(sources) != 1:
        raise BudgetParseError(
            f"'{kind}' at {_fmt(path)} must reference exactly one variable, "
            f"found {len(sources)}: {[k for k, _ in sources]}."
        )
    _, source_value = sources[0]
    # difference references a bare variable name; tolerate a {{var: ...}}
    # sub-dict form as well.
    if isinstance(source_value, dict):
        source = source_value.get("var")
    else:
        source = source_value
    if not isinstance(source, str):
        raise BudgetParseError(
            f"'{kind}' at {_fmt(path)} must reference a variable name (str), "
            f"got {type(source).__name__}."
        )
    return UNARY_OPS[kind](source=source)
