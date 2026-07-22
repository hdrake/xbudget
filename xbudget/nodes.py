"""Typed expression tree for xbudget recipes.

An xbudget recipe (the nested ``recipe`` dict loaded from YAML) describes
how to build each budget term from raw model diagnostics. Historically that
recipe was walked directly as untyped nested dicts, with a node's meaning
inferred from which magic keys it happened to carry. These dataclasses give the
recipe an explicit, immutable shape so the parser, evaluator, and query helpers
can dispatch on node *type* instead of probing dict keys.

The grammar (mirrors the YAML, see ``recipes/*.yaml``)::

    Budget        := name, metadata, {side: Term}          # side in {lhs, rhs}
    Term          := name, path, explicit_var?, [Operation]
    Operation     := Sum | Product | Difference
    Sum/Product   := [(operand_name, Operand)]
    Difference    := source variable name (differenced across a grid axis)
    Operand       := Constant | VarRef | Term

A ``Term`` may carry more than one ``Operation`` (e.g. a bulk ``Product`` and an
equivalent finer ``Sum`` decomposition of the same quantity); each operation
yields its own output variable when evaluated.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Constant:
    """A scalar factor/addend in a sum or product (e.g. a density, a sign)."""
    value: float


@dataclass(frozen=True)
class VarRef:
    """A reference to a raw diagnostic variable expected in the dataset."""
    name: str


@dataclass(frozen=True)
class Sum:
    """Add named operands together."""
    terms: tuple  # tuple[(name: str, operand: Constant | VarRef | "Term")]
    kind = "sum"


@dataclass(frozen=True)
class Product:
    """Multiply named operands together."""
    terms: tuple  # tuple[(name: str, operand: Constant | VarRef | "Term")]
    kind = "product"


@dataclass(frozen=True)
class Difference:
    """Finite-difference a staggered quantity across its grid axis.

    ``operand`` is either a :class:`VarRef` (a raw diagnostic) or a nested
    :class:`Term` (a quantity computed first, then differenced) — both expose a
    ``.name``.
    """
    operand: object
    kind = "difference"


@dataclass(frozen=True)
class Reciprocal:
    """Safe reciprocal (1/x with zeros mapped to infinity) of a variable."""
    source: str   # name of the variable to invert
    kind = "reciprocal"


@dataclass(frozen=True)
class LateralDivergence:
    """Horizontal flux divergence ``div(Fx, Fy)`` of two flux sub-terms.

    Evaluated on cell centers with native xgcm so face-connected (e.g. LLC)
    topologies are handled correctly.
    """
    fx: object  # Term producing the X-face flux
    fy: object  # Term producing the Y-face flux
    kind = "lateral_divergence"


@dataclass(frozen=True)
class Term:
    """A node in a budget tree: a named quantity defined by its operations.

    Attributes
    ----------
    name : str
        This term's key under its parent (the budget side name for a root).
    path : tuple of str
        Full path of term names from the budget root, e.g.
        ``("heat", "rhs", "diffusion", "lateral")``. Used as the canonical
        identity and to derive output variable names.
    operations : tuple of Operation
        One or more of Sum/Product/Difference. The first sum/product operation
        provides the term's primary value.
    explicit_var : str or None
        A pre-named output variable, if the recipe pinned one (rare).
    optional : bool
        Declares this term's diagnostics as *expected* to be absent on some
        datasets. When they are, the term is dropped silently — no warning, no
        ``on_missing="raise"`` error, and no ``xbudget_incomplete`` flag on the
        parent — instead of the loud "unexpectedly missing" treatment. This is
        the honest, self-documenting alternative to deleting the term from the
        recipe to quiet the warning. It suppresses missing-diagnostic alarms for
        the *whole* subtree rooted at this term.
    """
    name: str
    path: tuple
    operations: tuple
    explicit_var: object = None
    optional: bool = False


@dataclass(frozen=True)
class Budget:
    """A named budget (e.g. ``heat``) with lhs/rhs term trees and metadata."""
    name: str
    metadata: dict
    sides: dict  # {"lhs": Term, "rhs": Term} (either may be absent)


# Operations that introduce a single source variable rather than named terms.
UNARY_OPS = {"difference": Difference, "reciprocal": Reciprocal}
NARY_OPS = {"sum": Sum, "product": Product}
# Operations with their own bespoke operand shape (handled specially in parse).
SPECIAL_OPS = {"lateral_divergence": LateralDivergence}
OPERATION_KEYS = set(UNARY_OPS) | set(NARY_OPS) | set(SPECIAL_OPS)
