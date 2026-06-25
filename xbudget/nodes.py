"""Typed expression tree for xbudget recipes.

An xbudget convention (the nested ``xbudget_dict`` loaded from YAML) describes
how to build each budget term from raw model diagnostics. Historically that
recipe was walked directly as untyped nested dicts, with a node's meaning
inferred from which magic keys it happened to carry. These dataclasses give the
recipe an explicit, immutable shape so the parser, evaluator, and query helpers
can dispatch on node *type* instead of probing dict keys.

The grammar (mirrors the YAML, see ``conventions/*.yaml``)::

    Budget        := name, metadata, {side: Term}          # side in {lhs, rhs}
    Term          := name, path, explicit_var?, [Operation]
    Operation     := Sum | Product | Difference | Reciprocal
    Sum/Product   := [(operand_name, Operand)]
    Difference    := source variable name (differenced across a grid axis)
    Reciprocal    := source variable name (safe 1/x)
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
    """Finite-difference a staggered variable across its grid axis."""
    source: str   # name of the variable to difference
    kind = "difference"


@dataclass(frozen=True)
class Reciprocal:
    """Safe reciprocal (1/x with zeros mapped to infinity) of a variable."""
    source: str   # name of the variable to invert
    kind = "reciprocal"


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
        One or more of Sum/Product/Difference/Reciprocal. The first operation
        provides the term's primary value.
    explicit_var : str or None
        A pre-named output variable, if the convention pinned one (rare).
    """
    name: str
    path: tuple
    operations: tuple
    explicit_var: object = None


@dataclass(frozen=True)
class Budget:
    """A named budget (e.g. ``heat``) with lhs/rhs term trees and metadata."""
    name: str
    metadata: dict
    sides: dict  # {"lhs": Term, "rhs": Term} (either may be absent)


# Operations that introduce a single source operand rather than named terms.
UNARY_OPS = {"difference": Difference, "reciprocal": Reciprocal}
NARY_OPS = {"sum": Sum, "product": Product}
OPERATION_KEYS = set(UNARY_OPS) | set(NARY_OPS)
