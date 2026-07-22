"""Evaluate a typed budget tree against a dataset/grid.

The evaluator walks the :mod:`xbudget.nodes` tree and materializes one output
variable per *operation* (a ``sum``/``product``/``difference``), naming it by
its structured path with the operator infixes dropped (see ``_new_name``). A
term with more than one operation emits one variable each; the first
sum/product operation (or a lone operation) claims the bare path name, and any
siblings are suffixed with their operator kind.

It is intentionally side-effecting in one place only: it writes the derived
variables into the dataset. It never mutates the recipe (the typed tree is
immutable). It returns a ``records`` map (new variable name -> its path and
operation) describing what it built.

Missing diagnostics
-------------------
A recipe is meant to serve datasets with different diagnostics available, so a
term that references an absent variable is *skipped* rather than fatal. The
important thing is that a skip is never silent-and-forgotten: every variable the
evaluator emits that was built from fewer inputs than its recipe describes is
stamped with ``xbudget_incomplete`` and ``xbudget_missing`` attributes, which
persist to disk and are queryable through :class:`xbudget.query.BudgetQuery`.
This turns "a term is there when it isn't" into "the term is there and says so."

Three policies (``on_missing``) select how loud a missing *required* diagnostic
is: ``"warn"`` (default) emits one end-of-run summary; ``"raise"`` fails with a
:class:`MissingDiagnosticError`; ``"ignore"`` is silent (but still stamps the
attributes). A term (or subtree) declared ``optional`` in the recipe is exempt
from all three — its absence is expected, so it is dropped with no warning, no
error, and no ``incomplete`` flag on its parent.

A ``product`` with a *missing* required factor is dropped (the term does not
materialize) rather than multiplying in ``0.0`` and emitting an identically-zero
variable. An unknown factor is not the same as a zero one; a fabricated zero
reads as a real (null) contribution to whatever consumes it. A ``sum`` still
drops only the missing operand and builds from the rest, flagged incomplete.
"""
import warnings
from functools import reduce
from operator import mul

import numpy as np
import xarray as xr
import xgcm

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
from .collect import (
    _warn_if_summands_broadcast,
    lateral_divergence,
)

ON_MISSING = ("warn", "raise", "ignore")


class MissingDiagnosticError(ValueError):
    """A required diagnostic was absent under ``on_missing="raise"``.

    Carries the list of ``(diagnostic, term_path)`` pairs that could not be
    resolved, so the caller sees every problem at once rather than the first.
    """

    def __init__(self, message, missing):
        super().__init__(message)
        self.missing = missing


def _new_name(path):
    """Output variable name: the term path joined, without operator infixes."""
    return "_".join(path)


class _Evaluator:
    def __init__(self, data, allow_rechunk=True, on_missing="warn"):
        if on_missing not in ON_MISSING:
            raise ValueError(
                f"Unknown on_missing {on_missing!r}; expected one of {ON_MISSING}."
            )
        if isinstance(data, xgcm.grid.Grid):
            self.grid = data
            self.ds = data._ds
        else:
            self.grid = None
            self.ds = data
        self.allow_rechunk = allow_rechunk
        self.on_missing = on_missing
        # new variable name -> {"path", "op"}
        self.records = {}
        # Missing-diagnostic bookkeeping, drained once at the end of run().
        # Only *required* (non-optional) misses land here.
        self._missing = []        # list of (diagnostic_name, term_path tuple)
        self._incomplete = []     # names of emitted variables flagged incomplete

    def run(self, budgets):
        for budget in budgets.values():
            for term in budget.sides.values():
                self._eval_term(term)
        self._finish()
        return self.records

    # -- missing-diagnostic reporting ---------------------------------------

    def _record_missing(self, diagnostic, path, alarming):
        """Note that ``diagnostic`` (referenced at ``path``) is absent.

        ``alarming`` is ``False`` inside an ``optional`` subtree, where a miss
        is expected: it is not surfaced in the summary, does not trigger
        ``on_missing="raise"``, and does not flag anything incomplete.
        """
        if alarming and diagnostic is not None:
            self._missing.append((diagnostic, tuple(path)))

    def _finish(self):
        """Report the accumulated required-but-missing diagnostics, once."""
        if not self._missing:
            return
        diagnostics = sorted({d for d, _ in self._missing})
        incomplete = sorted(set(self._incomplete))
        detail = (
            f"missing diagnostic(s) {diagnostics}"
            + (
                f"; {len(incomplete)} budget term(s) are now incomplete: "
                f"{incomplete}"
                if incomplete
                else ""
            )
        )
        if self.on_missing == "raise":
            raise MissingDiagnosticError(
                f"collect_budgets(on_missing='raise'): {detail}. Provide the "
                f"diagnostics, mark the term(s) `optional` in the recipe, or use "
                f"on_missing='warn'/'ignore'.",
                missing=list(self._missing),
            )
        if self.on_missing == "warn":
            warnings.warn(
                f"xbudget: {detail}. The affected variables carry "
                f"`xbudget_incomplete`/`xbudget_missing` attrs; query them with "
                f"BudgetQuery.missing(). Mark expected-absent terms `optional` "
                f"in the recipe to silence this, or use on_missing='raise' to "
                f"fail instead.",
                UserWarning,
            )
        # on_missing == "ignore": stay silent (the attrs were still stamped).

    # -- term ---------------------------------------------------------------

    def _eval_term(self, term, optional_ctx=False):
        """Evaluate a term; emit a variable per operation; return primary value.

        ``optional_ctx`` is ``True`` when an ancestor (or this term) is declared
        ``optional``, in which case missing diagnostics anywhere in this subtree
        are expected and silent.
        """
        opt = optional_ctx or term.optional
        base = _new_name(term.path)

        primary = None
        first_emitted = True

        # Leaf term: ``var`` references an existing diagnostic directly (e.g.
        # ``{"var": "advective_tendency"}``). Aliased to the term's name and
        # treated as the term's primary value.
        if isinstance(term.explicit_var, str):
            if term.explicit_var in self.ds:
                out = self.ds[term.explicit_var].rename(base).copy()
                out.attrs["provenance"] = term.explicit_var
                out.attrs["xbudget_path"] = list(term.path)
                out.attrs["xbudget_op"] = "var"
                self.ds[base] = out
                self.records[base] = {"path": list(term.path), "op": "var"}
                primary = out
                first_emitted = False
            else:
                # A leaf naming a diagnostic that is absent: record it here (once)
                # so the summary names the diagnostic, not just the parent term.
                self._record_missing(term.explicit_var, term.path, alarming=not opt)

        evaluated = []  # (op, value, provenance, opmeta) for producing operations
        for op in term.operations:
            value, provenance, opmeta = self._eval_op(op, term, opt)
            if value is not None:
                evaluated.append((op, value, provenance, opmeta))

        # Choose which operation gets the bare path name (the term's primary
        # value): the first sum/product, else the first operation of any kind. A
        # leaf var, if present, already claimed it above (first_emitted is
        # False), so no operation does.
        primary_idx = None
        if first_emitted:
            for i, (op, _v, _p, _m) in enumerate(evaluated):
                if op.kind in ("sum", "product"):
                    primary_idx = i
                    break
            if primary_idx is None and evaluated:
                primary_idx = 0

        for i, (op, value, provenance, opmeta) in enumerate(evaluated):
            is_primary = i == primary_idx
            new_name = base if is_primary else f"{base}_{op.kind}"

            out = value.rename(new_name)
            # Set a fresh attrs dict rather than mutate whatever xarray
            # arithmetic left behind: `sum(op_list)` can carry an operand's
            # attributes through, which would otherwise leak a child's
            # `xbudget_missing`/`xbudget_incomplete` onto this parent.
            out.attrs = {
                "provenance": provenance,
                "xbudget_path": list(term.path),
                "xbudget_op": op.kind,
            }
            self._stamp_incompleteness(out, opmeta, opt)
            self.ds[new_name] = out
            self.records[new_name] = {"path": list(term.path), "op": op.kind}
            if is_primary:
                primary = out

        return primary

    def _stamp_incompleteness(self, out, opmeta, opt):
        """Mark ``out`` incomplete if it was built from fewer inputs than asked.

        ``xbudget_incomplete`` (int ``1``) is the queryable flag; ``xbudget_missing``
        lists the operands that were dropped at this node (a child that dropped
        deeper carries its own detail). Nothing is stamped inside an ``optional``
        subtree, where incompleteness is expected — the whole point of the flag
        is to mark the *un*expected.
        """
        if opt or not opmeta:
            return
        missing = opmeta.get("missing") or []
        incomplete = bool(opmeta.get("incomplete")) or bool(missing)
        if not incomplete:
            return
        out.attrs["xbudget_incomplete"] = 1
        if missing:
            out.attrs["xbudget_missing"] = list(missing)
        self._incomplete.append(out.name)

    # -- operations ---------------------------------------------------------

    def _eval_op(self, op, term, opt):
        if isinstance(op, (Sum, Product)):
            return self._eval_nary(op, term, opt)
        if isinstance(op, Difference):
            return self._eval_difference(op, term, opt)
        if isinstance(op, Reciprocal):
            return self._eval_reciprocal(op, term, opt)
        if isinstance(op, LateralDivergence):
            return self._eval_lateral_divergence(op, term, opt)
        raise TypeError(f"Unknown operation type {type(op).__name__}")

    def _eval_reciprocal(self, op, term, opt):
        ds = self.ds
        if op.source not in ds:
            self._record_missing(op.source, term.path, alarming=not opt)
            return None, None, None
        var = 1.0 / xr.where(ds[op.source] == 0, np.inf, ds[op.source])
        return var, op.source, {"missing": [], "incomplete": False}

    def _eval_lateral_divergence(self, op, term, opt):
        fx = self._eval_term(op.fx, optional_ctx=opt)
        fy = self._eval_term(op.fy, optional_ctx=opt)
        if fx is None or fy is None:
            # The flux sub-terms already recorded whatever diagnostic they were
            # missing; a divergence of an absent flux is simply not built.
            return None, None, None
        var = lateral_divergence(self.grid, fx, fy)
        incomplete = bool(fx.attrs.get("xbudget_incomplete")) or bool(
            fy.attrs.get("xbudget_incomplete")
        )
        return var, [fx.name, fy.name], {"missing": [], "incomplete": incomplete}

    def _eval_nary(self, op, term, opt):
        ds = self.ds
        op_list = []
        missing = []       # labels/names of operands dropped at this node
        incomplete = False
        for name, operand in op.terms:
            value = None
            operand_optional = opt
            if isinstance(operand, Term):
                operand_optional = opt or operand.optional
                child = self._eval_term(operand, optional_ctx=operand_optional)
                if child is not None:
                    value = child
                    if child.attrs.get("xbudget_incomplete"):
                        incomplete = True
                # else: the child already recorded its own missing diagnostic.
                dropped_label = name
            elif isinstance(operand, Constant):
                value = operand.value
            elif isinstance(operand, VarRef):
                if operand.name in ds:
                    value = ds[operand.name]
                else:
                    self._record_missing(operand.name, term.path, alarming=not opt)
                dropped_label = operand.name

            if value is not None:
                op_list.append(value)
                continue

            # This operand did not resolve.
            if op.kind == "product":
                # A product needs every factor; an unknown one is not a zero one.
                # Drop the whole term rather than fabricate an all-zero variable.
                return None, None, None
            # A sum keeps its surviving operands and records the gap.
            if not operand_optional:
                missing.append(dropped_label)
                incomplete = True

        if len(op_list) == 0:
            return None, None, None
        if op.kind == "sum":
            _warn_if_summands_broadcast(op_list, _new_name(term.path))
        var = sum(op_list) if op.kind == "sum" else reduce(mul, op_list, 1)
        if not isinstance(var, xr.DataArray):
            # Reduced to a pure scalar (e.g. all variable operands missing); no
            # variable is emitted in this case.
            return None, None, None
        provenance = [o.name if isinstance(o, xr.DataArray) else o for o in op_list]
        return var, provenance, {"missing": missing, "incomplete": incomplete}

    def _eval_difference(self, op, term, opt):
        if self.grid is None:
            raise ValueError(
                "Input `data` must be an `xgcm.Grid` instance when using "
                "`difference` operations."
            )
        ds = self.ds
        operand = op.operand
        incomplete = False
        if isinstance(operand, VarRef):
            if operand.name not in ds:
                self._record_missing(operand.name, term.path, alarming=not opt)
                return None, None, None
            source = ds[operand.name]
            provenance = operand.name
        else:  # a computed sub-term, differenced after evaluation
            source = self._eval_term(operand, optional_ctx=opt)
            if source is None:
                return None, None, None
            provenance = source.name
            incomplete = bool(source.attrs.get("xbudget_incomplete"))

        staggered_axes = {
            axn: c
            for axn, ax in self.grid.axes.items()
            for pos, c in ax.coords.items()
            if pos != "center"
        }
        candidate_axes = [
            axn for (axn, c) in staggered_axes.items() if c in source.dims
        ]
        if len(candidate_axes) != 1:
            raise ValueError(
                "Finite difference inconsistent with finite volume discretization."
            )
        axis = candidate_axes[0]

        if self.allow_rechunk:
            try:
                original_chunks = dict(ds.chunksizes)
            except Exception:
                warnings.warn(
                    "Dataset chunks are inconsistent; using unify_chunks()",
                    UserWarning,
                )
                original_chunks = dict(ds.unify_chunks().chunksizes)

            axis_dim = [
                d for d in source.dims if d in self.grid.axes[axis].coords.values()
            ]
            if len(axis_dim) != 1:
                raise ValueError(
                    f"Expected to find one dimension for axis '{axis}' in "
                    f"'{provenance}', but found {len(axis_dim)}: {axis_dim}"
                )
            axis_dim = axis_dim[0]

            temporary_chunks = {
                axis_dim: -1,
                **{d: "auto" for d in source.dims if d != axis_dim},
            }
            var = self.grid.diff(
                source.chunk(temporary_chunks).fillna(0.0), axis=axis
            )
            # Restore the original chunking of the dimensions we know; leave any
            # others as `grid.diff` produced them. Only naming the dims we want
            # changed avoids reading `var.chunksizes`, which raises when the
            # result's coords are chunked differently from its data.
            restore = {
                d: original_chunks[d] for d in var.dims if d in original_chunks
            }
            if restore:
                var = var.chunk(restore)
        else:
            var = self.grid.diff(source.fillna(0.0), axis)

        return var, provenance, {"missing": [], "incomplete": incomplete}


def evaluate_budgets(data, budgets, allow_rechunk=True, on_missing="warn"):
    """Evaluate parsed budgets into ``data``; return ``records``.

    Parameters
    ----------
    data : xgcm.Grid or xr.Dataset
        The dataset (or grid) to read diagnostics from and write derived
        variables into. Modified in place.
    budgets : dict of str -> Budget
        Parsed budgets, from :func:`xbudget.parse.parse_budgets`.
    allow_rechunk : bool, default True
        Temporarily rechunk staggered variables for ``difference`` operations.
    on_missing : {"warn", "raise", "ignore"}, default "warn"
        How to handle a *required* diagnostic that is absent from ``data``.
        ``"warn"`` emits one end-of-run summary ``UserWarning``; ``"raise"``
        raises :class:`MissingDiagnosticError`; ``"ignore"`` is silent. In every
        case the affected variables are stamped with ``xbudget_incomplete`` /
        ``xbudget_missing`` attributes, and terms declared ``optional`` in the
        recipe are exempt.

    Returns
    -------
    records : dict
        New variable name -> metadata (``{"path": [...], "op": kind}``).

    Examples
    --------
    Parse a recipe to a typed tree, then evaluate it into a grid (or dataset).
    The derived variables are written into ``data`` in place; ``records`` maps
    each new name to its structural identity:

    >>> budgets = xbudget.parse_budgets(recipe)
    >>> records = xbudget.evaluate_budgets(grid, budgets)
    >>> records["heat_rhs"]
    {'path': ['heat', 'rhs'], 'op': 'sum'}
    >>> grid._ds["heat_rhs"]           # the derived variable now lives in the data
    <xarray.DataArray 'heat_rhs' ...>

    ``collect_budgets`` is the usual one-call entry point (it parses and
    evaluates); call ``evaluate_budgets`` directly when you already hold a parsed
    tree or want to control the two steps separately.
    """
    return _Evaluator(
        data, allow_rechunk=allow_rechunk, on_missing=on_missing
    ).run(budgets)
