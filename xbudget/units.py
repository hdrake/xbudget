"""Unit inference for xbudget derived variables, backed by UDUNITS (cf-units).

The evaluator materializes one variable per operation; this module gives each
one an inferred ``units`` attribute by composing the units of its inputs, using
the same arithmetic the physics does:

- ``product``  -> multiply the operand units;
- ``sum``      -> the common unit of the summands (they must be dimensionally
  compatible), else *unknown*;
- ``difference`` -> unchanged (a finite difference preserves units);
- ``reciprocal`` -> ``1 / unit``;
- ``lateral_divergence`` -> the flux unit (Fx and Fy must be compatible).

Units are ``cf_units.Unit`` objects (real UDUNITS parsing/algebra) or ``None``.
``None`` means *unknown* — a missing, blank, or unparseable ``units`` attribute —
and is deliberately distinct from the dimensionless :data:`DIMENSIONLESS`
(``Unit("1")``). Unknown propagates: any operation with an unknown operand yields
unknown rather than a fabricated dimension.

Every function here is pure and total (it never raises and never warns): it just
answers "what are the units?". Whether an unknown or a mismatch is *tolerable*
is policy, and lives in :mod:`xbudget.evaluate` (the ``assert_unit_consistency``
flag) rather than here.

Two conventions worth knowing:

- **Practical salinity.** UDUNITS does not define ``psu``; PSS-78 salinity is
  dimensionless, so :func:`parse_units` aliases ``psu`` (and spellings of it) to
  ``"1"``. A salt term is then ``salinity x mass-flux``, which reconciles to
  ``kg s-1`` (the residual ``1e-3`` from a ``unit_conversion`` factor is a scale,
  not a dimension, and does not affect convertibility).
- **Temperature.** ``degC`` is an offset unit, but cf-units dimensionalizes it to
  kelvin inside a product automatically (``degC * kg -> kg.K``), so a heat-content
  term ``theta x c_p x rho x dh/dt x area`` reconciles to ``W`` with no special
  handling here.
"""
from functools import reduce
from operator import mul

from cf_units import Unit

from .nodes import (
    Constant,
    Difference,
    LateralDivergence,
    Product,
    Reciprocal,
    Sum,
    Term,
    VarRef,
)

__all__ = [
    "DIMENSIONLESS",
    "parse_units",
    "product_units",
    "sum_units",
    "sum_compatible",
    "reciprocal_units",
    "difference_units",
    "divergence_units",
    "convertible",
    "format_units",
    "infer_units",
]

DIMENSIONLESS = Unit("1")

# UDUNITS has no practical-salinity unit; PSS-78 salinity is dimensionless.
_ALIASES = {
    "psu": "1",
    "PSU": "1",
    "practical_salinity_unit": "1",
    "practical_salinity_units": "1",
    "pss-78": "1",
    "PSS-78": "1",
}


def parse_units(spec):
    """Parse a units string (or ``Unit``/attribute) into a ``Unit``, or ``None``.

    ``None`` (unknown) is returned for ``None``, a blank string, an unparseable
    string, or a cf-units ``unknown``/``no-unit`` sentinel — anything that is not
    a real dimension. This is distinct from :data:`DIMENSIONLESS`.
    """
    if spec is None:
        return None
    if isinstance(spec, Unit):
        return spec if _is_real(spec) else None
    text = str(spec).strip()
    if not text:
        return None
    text = _ALIASES.get(text, text)
    try:
        unit = Unit(text)
    except ValueError:
        return None
    return unit if _is_real(unit) else None


def _is_real(unit):
    """True unless ``unit`` is cf-units' ``unknown``/``no-unit`` sentinel."""
    return not (unit.is_unknown() or unit.is_no_unit())


def product_units(units):
    """Units of a product of operands; ``None`` if any operand is unknown."""
    units = list(units)
    if any(u is None for u in units):
        return None
    if not units:
        return DIMENSIONLESS
    # Seed the reduction with the first real unit rather than DIMENSIONLESS so an
    # offset operand (e.g. degC) is never the right operand of ``1 * degC``.
    return reduce(mul, units[1:], units[0])


def reciprocal_units(unit):
    """Units of ``1 / x``; ``None`` if ``x``'s units are unknown."""
    if unit is None:
        return None
    return DIMENSIONLESS / unit


def difference_units(unit):
    """Units of a finite difference: unchanged (may be ``None``)."""
    return unit


def sum_compatible(units):
    """Whether every *known* summand shares a dimension.

    ``True`` when fewer than two units are known (nothing to contradict); ``False``
    only when two known units are dimensionally incompatible. Unknown operands do
    not make a sum *incompatible* — they make its result unknown (see
    :func:`sum_units`) — so this is the signal the evaluator uses to distinguish a
    real unit clash (worth a warning/error) from a merely-absent unit.
    """
    known = [u for u in units if u is not None]
    return all(known[0].is_convertible(u) for u in known[1:])


def sum_units(units):
    """Common units of the summands, or ``None``.

    ``None`` if any summand is unknown or if two summands are dimensionally
    incompatible. Otherwise the units of the first summand (all are compatible).
    """
    units = list(units)
    if not units or any(u is None for u in units):
        return None
    if not sum_compatible(units):
        return None
    return units[0]


def divergence_units(fx, fy):
    """Units of a horizontal flux divergence: the (compatible) flux units."""
    return sum_units([fx, fy])


def convertible(unit, other):
    """Whether ``unit`` is dimensionally convertible to ``other`` (both known)."""
    if unit is None or other is None:
        return False
    return unit.is_convertible(other)


def format_units(unit):
    """A stable string for a ``Unit`` (for stamping as an attribute), or ``None``."""
    return None if unit is None else str(unit)


def infer_units(node, lookup=None):
    """Infer a node's units directly from the typed tree, without evaluating.

    The evaluator infers units as a side-effect of building variables, which is
    the authoritative answer but only exists after a run. This walks the same
    tree with the same arithmetic and answers the question *statically*, so a
    recipe can be shown with its units before (or without) any data — see
    :func:`xbudget.display.show_recipe`.

    ``lookup`` maps a raw diagnostic name to its units spec (typically
    ``lambda name: ds[name].attrs.get("units")``). Without it only constants
    carry units, so most terms come back unknown; with it, everything a
    dimensionally complete recipe describes resolves.

    Returns a :class:`cf_units.Unit` or ``None`` (unknown). Like the rest of this
    module it never raises: an unresolvable node is unknown, not an error.
    """
    if lookup is None:
        def lookup(_name):
            return None
    return _infer(node, lookup)


def _infer(node, lookup):
    if isinstance(node, Constant):
        return parse_units(node.units)
    if isinstance(node, VarRef):
        return parse_units(lookup(node.name))
    if isinstance(node, Term):
        return _infer_term(node, lookup)
    if isinstance(node, Sum):
        return sum_units([_infer(o, lookup) for _n, o in node.terms])
    if isinstance(node, Product):
        return product_units([_infer(o, lookup) for _n, o in node.terms])
    if isinstance(node, Difference):
        return difference_units(_infer(node.operand, lookup))
    if isinstance(node, Reciprocal):
        return reciprocal_units(parse_units(lookup(node.source)))
    if isinstance(node, LateralDivergence):
        return divergence_units(_infer(node.fx, lookup), _infer(node.fy, lookup))
    return None


def _infer_term(term, lookup):
    """Units of a term's *primary* value, mirroring the evaluator's choice.

    A leaf ``var`` claims the term's name outright; otherwise the first
    sum/product operation does, falling back to the first operation of any kind
    (see ``evaluate._Evaluator._eval_term``).
    """
    if isinstance(term.explicit_var, str):
        return parse_units(lookup(term.explicit_var))
    if not term.operations:
        return None
    primary = next(
        (op for op in term.operations if isinstance(op, (Sum, Product))),
        term.operations[0],
    )
    return _infer(primary, lookup)
