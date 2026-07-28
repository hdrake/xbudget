"""Parse an xbudget ``recipe`` dict into the typed expression tree.

This is the single place that knows the YAML/dict schema. It validates as it
builds, so malformed recipes raise a clear ``BudgetParseError`` naming the
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

#: Top-level recipe key holding the shared constant table (not a budget).
CONSTANTS_KEY = "constants"


class BudgetParseError(ValueError):
    """Raised when an xbudget recipe does not match the expected schema."""


def _fmt(path):
    return "/".join(str(p) for p in path) or "<root>"


def parse_budgets(recipe):
    """Parse a recipe dict into ``{budget_name: Budget}``.

    Every top-level key is a budget except the reserved ``constants`` key, which
    declares scalars shared across the recipe (see :func:`_parse_constants`).
    Constant references are substituted while parsing, so the returned tree holds
    only :class:`~xbudget.nodes.Constant` nodes — nothing downstream needs to know
    the table existed.

    Parameters
    ----------
    recipe : dict
        A recipe in xbudget format (e.g. from ``load_preset_budget``).

    Returns
    -------
    dict of str -> Budget

    Examples
    --------
    >>> recipe = xbudget.load_preset_budget(model="MOM6")
    >>> budgets = xbudget.parse_budgets(recipe)
    >>> list(budgets)
    ['mass', 'heat', 'salt']
    >>> budgets["heat"].metadata
    {'lambda': 'thetao', 'surface_lambda': 'tos'}
    """
    if not isinstance(recipe, dict):
        raise BudgetParseError(
            f"Top-level xbudget recipe must be a dict, got "
            f"{type(recipe).__name__}."
        )

    consts = _parse_constants(recipe.get(CONSTANTS_KEY))
    budgets = {}
    for name, body in recipe.items():
        if name == CONSTANTS_KEY:
            continue
        if not isinstance(body, dict):
            raise BudgetParseError(
                f"Budget '{name}' must be a dict, got {type(body).__name__}."
            )
        metadata = {k: v for k, v in body.items() if k not in SIDES}
        sides = {}
        for side in SIDES:
            if body.get(side) is not None:
                sides[side] = _parse_term(body[side], (name, side), side, consts)
        budgets[name] = Budget(name=name, metadata=metadata, sides=sides)
    return budgets


def _parse_constants(block):
    """Parse the top-level ``constants`` table into ``{name: Constant}``.

    Physical constants (a reference density, a heat capacity) recur throughout a
    recipe, and repeating a ``{value:, units:}`` mapping at every use is both
    tedious and a place for the two copies to drift apart. Declaring them once
    at the root and referring to them by name keeps a single source of truth::

        constants:
          seawater_density: {value: 1035., units: "kg m-3"}
        ...
              density: "seawater_density"     # resolves to the constant above

    Entries take the same two forms an inline constant does: a bare number
    (dimensionless) or a ``{value:, units:}`` mapping. A string operand is looked
    up here first and falls back to being a diagnostic name, so a constant whose
    name collides with a dataset variable shadows it — name them distinctly.
    """
    if block is None:
        return {}
    if not isinstance(block, dict):
        raise BudgetParseError(
            f"Top-level '{CONSTANTS_KEY}' must be a dict of "
            f"name -> number or {{value:, units:}}, got {type(block).__name__}."
        )
    consts = {}
    for name, spec in block.items():
        path = (CONSTANTS_KEY, name)
        if isinstance(spec, dict):
            consts[name] = _parse_constant(spec, path, name)
        elif isinstance(spec, bool) or not isinstance(spec, numbers.Number):
            raise BudgetParseError(
                f"Constant '{name}' at {_fmt(path)} must be a number or a "
                f"{{value:, units:}} mapping, got {type(spec).__name__}."
            )
        else:
            consts[name] = Constant(float(spec))
    return consts


def _parse_term(node, path, name, consts):
    """Parse a single term node (a quantity defined by its operations)."""
    if not isinstance(node, dict):
        raise BudgetParseError(
            f"Term at {_fmt(path)} must be a dict, got {type(node).__name__}."
        )

    operations = []
    for key, value in node.items():
        if key == "var":
            continue
        if key == "optional":
            # A recognized term-level flag, not an operation and not a stray
            # key: read below via node.get("optional"), so skip it here (and do
            # not warn about it as an unexpected key).
            continue
        if key in NARY_OPS:
            operations.append(_parse_nary(key, value, path, consts))
        elif key == "difference":
            op = _parse_difference(value, path, consts)
            if op is not None:
                operations.append(op)
        elif key == "reciprocal":
            op = _parse_reciprocal(value, path)
            if op is not None:
                operations.append(op)
        elif key in SPECIAL_OPS:
            operations.append(_parse_lateral_divergence(value, path, consts))
        else:
            # Tolerate keys that are neither `var` nor an operation (e.g. a
            # `sign`/`density` scalar left directly on a term because its
            # enclosing `product:` was omitted) rather than failing the whole
            # parse, but warn so the malformation is visible.
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
        optional=bool(node.get("optional", False)),
    )


def _parse_nary(kind, body, path, consts):
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
        operand = _parse_operand(term_value, path + (term_name,), term_name, consts)
        if operand is not None:
            terms.append((term_name, operand))
    return NARY_OPS[kind](terms=tuple(terms))


def _parse_operand(value, path, name, consts):
    """Parse one operand of a sum/product: constant, var reference, or term."""
    if value is None:
        # Tolerated: a placeholder operand with no content is skipped.
        return None
    if isinstance(value, dict):
        # A ``{value:, units:}`` mapping is a constant with declared units, not a
        # sub-term. Distinguish by the ``value`` key (and the absence of any
        # operation key): a real sub-term names operations, never ``value``.
        if "value" in value and not (value.keys() & OPERATION_KEYS):
            return _parse_constant(value, path, name)
        return _parse_term(value, path, name, consts)
    if isinstance(value, bool):
        raise BudgetParseError(
            f"Operand '{name}' at {_fmt(path)} is a bool; expected a number, "
            f"variable name, or sub-term."
        )
    if isinstance(value, numbers.Number):
        return Constant(float(value))
    if isinstance(value, str):
        # A string names either a constant declared in the recipe's top-level
        # `constants` table or a raw diagnostic in the dataset.
        if value in consts:
            return consts[value]
        return VarRef(value)
    raise BudgetParseError(
        f"Operand '{name}' at {_fmt(path)} has unsupported type "
        f"{type(value).__name__}."
    )


def _parse_constant(body, path, name):
    """Parse a ``{value:, units:}`` constant (units default to dimensionless)."""
    value = body["value"]
    if isinstance(value, bool) or not isinstance(value, numbers.Number):
        raise BudgetParseError(
            f"Constant '{name}' at {_fmt(path)} needs a numeric 'value', got "
            f"{type(value).__name__}."
        )
    units = body.get("units", "1")
    if not isinstance(units, str):
        raise BudgetParseError(
            f"Constant '{name}' at {_fmt(path)} 'units' must be a UDUNITS string, "
            f"got {type(units).__name__}."
        )
    extra = set(body) - {"value", "units"}
    if extra:
        raise BudgetParseError(
            f"Constant '{name}' at {_fmt(path)} has unexpected key(s) "
            f"{sorted(extra)}; only 'value' and 'units' are allowed."
        )
    return Constant(float(value), units=units)


def _single_operand(kind, body, path):
    """Return the single non-``var`` (name, value) of a unary op body, or None.

    Warns and returns ``None`` for the tolerated malformed/placeholder cases
    (zero or several operands).
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


def _parse_difference(body, path, consts):
    """Parse a ``difference`` operation.

    The operand is either a raw variable name (``VarRef``) or a nested term that
    is computed first and then differenced (``Term``). Returns ``None`` for an
    unavailable-diagnostic placeholder.
    """
    operand = _single_operand("difference", body, path)
    if operand is None:
        return None
    name, value = operand
    if isinstance(value, str):
        return Difference(operand=VarRef(value))
    if isinstance(value, dict):
        return Difference(
            operand=_parse_term(value, path + (name,), name, consts)
        )
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


def _parse_lateral_divergence(body, path, consts):
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
    fx = _parse_term(body["Fx"], path + ("Fx",), "Fx", consts)
    fy = _parse_term(body["Fy"], path + ("Fy",), "Fy", consts)
    return LateralDivergence(fx=fx, fy=fy)
