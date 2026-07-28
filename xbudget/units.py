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
is policy, and lives in :mod:`xbudget.evaluate` (the ``strict`` flag) rather than
here.

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
