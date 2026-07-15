"""Evaluate a typed budget tree against a dataset/grid.

The evaluator walks the :mod:`xbudget.nodes` tree and materializes one output
variable per *operation* (a ``sum``/``product``/``difference``), naming it by
its structured path with the operator infixes dropped (see ``_new_name``). This
collapses the legacy engine's duplicate "copy" variables while preserving the
numerical results.

It is intentionally side-effecting in one place only: it writes the derived
variables into the dataset. It never mutates the recipe (the typed tree is
immutable). Alongside the new variables it records a legacy->new name map so
callers can offer backward-compatible names and a migration table.
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
from .collect import _warn_missing_variable, lateral_divergence


def _new_name(path):
    """Output variable name: the term path joined, without operator infixes."""
    return "_".join(path)


class _Evaluator:
    def __init__(self, data, allow_rechunk=True):
        if isinstance(data, xgcm.grid.Grid):
            self.grid = data
            self.ds = data._ds
        else:
            self.grid = None
            self.ds = data
        self.allow_rechunk = allow_rechunk
        # legacy variable name -> new variable name (covers both the legacy
        # "actual" operator-suffixed names and the plain "copy" names).
        self.alias_map = {}
        # new variable name -> {"path", "op", "legacy_actual", "legacy_copy"}
        self.records = {}

    def run(self, budgets):
        for budget in budgets.values():
            for side, term in budget.sides.items():
                self._eval_term(term, f"{budget.name}_{side}")
        return self.alias_map, self.records

    # -- term ---------------------------------------------------------------

    def _eval_term(self, term, legacy_namepath):
        """Evaluate a term; emit a variable per operation; return primary value.

        ``legacy_namepath`` is the variable name the previous engine would have
        used for this term (operator infixes included via the parents), used to
        build the backward-compatibility alias map.
        """
        base = _new_name(term.path)

        primary = None
        first_emitted = True

        # Leaf term: ``var`` references an existing diagnostic directly (e.g.
        # ``{"var": "advective_tendency"}``). The legacy engine aliased it to the
        # term's name; we do the same and treat it as the term's primary value.
        if isinstance(term.explicit_var, str) and term.explicit_var in self.ds:
            out = self.ds[term.explicit_var].rename(base).copy()
            out.attrs["provenance"] = term.explicit_var
            out.attrs["xbudget_path"] = list(term.path)
            out.attrs["xbudget_op"] = "var"
            self.ds[base] = out
            self.alias_map[legacy_namepath] = base
            self.records[base] = {
                "path": list(term.path),
                "op": "var",
                "legacy_actual": legacy_namepath,
            }
            primary = out
            first_emitted = False

        evaluated = []  # (op, value, provenance) for value-producing operations
        for op in term.operations:
            value, provenance = self._eval_op(op, term, legacy_namepath)
            if value is not None:
                evaluated.append((op, value, provenance))

        # Choose which operation gets the bare path name (the term's primary
        # value). The legacy engine reserved the plain namepath ("copy") for the
        # first sum/product term, so prefer that; a lone difference takes the
        # bare name (with no legacy copy). A leaf var, if present, already claimed
        # it above (first_emitted is False), so no operation does.
        primary_idx = None
        if first_emitted:
            for i, (op, _v, _p) in enumerate(evaluated):
                if op.kind in ("sum", "product"):
                    primary_idx = i
                    break
            if primary_idx is None and evaluated:
                primary_idx = 0

        for i, (op, value, provenance) in enumerate(evaluated):
            is_primary = i == primary_idx
            new_name = base if is_primary else f"{base}_{op.kind}"
            legacy_actual = f"{legacy_namepath}_{op.kind}"

            out = value.rename(new_name)
            out.attrs["provenance"] = provenance
            out.attrs["xbudget_path"] = list(term.path)
            out.attrs["xbudget_op"] = op.kind
            self.ds[new_name] = out

            self.alias_map[legacy_actual] = new_name
            self.records[new_name] = {
                "path": list(term.path),
                "op": op.kind,
                "legacy_actual": legacy_actual,
            }
            if is_primary:
                # The legacy engine also emitted a plain "copy" at the namepath
                # for every operation except a lone difference.
                if op.kind != "difference":
                    self.alias_map[legacy_namepath] = new_name
                    self.records[new_name]["legacy_copy"] = legacy_namepath
                primary = out

        return primary

    # -- operations ---------------------------------------------------------

    def _eval_op(self, op, term, legacy_namepath):
        if isinstance(op, (Sum, Product)):
            return self._eval_nary(op, legacy_namepath)
        if isinstance(op, Difference):
            return self._eval_difference(op, legacy_namepath)
        if isinstance(op, Reciprocal):
            return self._eval_reciprocal(op)
        if isinstance(op, LateralDivergence):
            return self._eval_lateral_divergence(op, legacy_namepath)
        raise TypeError(f"Unknown operation type {type(op).__name__}")

    def _eval_reciprocal(self, op):
        ds = self.ds
        if op.source not in ds:
            _warn_missing_variable(op.source)
            return None, None
        var = 1.0 / xr.where(ds[op.source] == 0, np.inf, ds[op.source])
        return var, op.source

    def _eval_lateral_divergence(self, op, legacy_namepath):
        fx = self._eval_term(op.fx, f"{legacy_namepath}_Fx")
        fy = self._eval_term(op.fy, f"{legacy_namepath}_Fy")
        if fx is None or fy is None:
            warnings.warn(
                f"Could not compute fluxes for {legacy_namepath}, skipping.",
                UserWarning,
            )
            return None, None
        var = lateral_divergence(self.grid, fx, fy)
        return var, [fx.name, fy.name]

    def _eval_nary(self, op, legacy_namepath):
        ds = self.ds
        op_list = []
        for name, operand in op.terms:
            if isinstance(operand, Term):
                child_legacy = f"{legacy_namepath}_{op.kind}_{name}"
                child_value = self._eval_term(operand, child_legacy)
                if child_value is not None:
                    op_list.append(child_value)
                elif operand.explicit_var is not None and operand.explicit_var not in ds:
                    _warn_missing_variable(operand.explicit_var)
            elif isinstance(operand, Constant):
                op_list.append(operand.value)
            elif isinstance(operand, VarRef):
                if operand.name in ds:
                    op_list.append(ds[operand.name])
                else:
                    _warn_missing_variable(operand.name)
                    if op.kind == "product":
                        op_list.append(0.0)

        if len(op_list) == 0:
            return None, None
        var = sum(op_list) if op.kind == "sum" else reduce(mul, op_list, 1)
        if not isinstance(var, xr.DataArray):
            # Reduced to a pure scalar (e.g. all variable operands missing);
            # the legacy engine emitted no variable in this case.
            return None, None
        provenance = [o.name if isinstance(o, xr.DataArray) else o for o in op_list]
        return var, provenance

    def _eval_difference(self, op, legacy_namepath):
        if self.grid is None:
            raise ValueError(
                "Input `data` must be an `xgcm.Grid` instance when using "
                "`difference` operations."
            )
        ds = self.ds
        operand = op.operand
        if isinstance(operand, VarRef):
            if operand.name not in ds:
                _warn_missing_variable(operand.name)
                return None, None
            source = ds[operand.name]
            provenance = operand.name
        else:  # a computed sub-term, differenced after evaluation
            source = self._eval_term(
                operand, f"{legacy_namepath}_difference_{operand.name}"
            )
            if source is None:
                return None, None
            provenance = source.name

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

        return var, provenance


def evaluate_budgets(data, budgets, allow_rechunk=True):
    """Evaluate parsed budgets into ``data``; return ``(alias_map, records)``.

    Parameters
    ----------
    data : xgcm.Grid or xr.Dataset
        The dataset (or grid) to read diagnostics from and write derived
        variables into. Modified in place.
    budgets : dict of str -> Budget
        Parsed budgets, from :func:`xbudget.parse.parse_budgets`.
    allow_rechunk : bool, default True
        Temporarily rechunk staggered variables for ``difference`` operations.

    Returns
    -------
    alias_map : dict
        Legacy variable name -> new variable name.
    records : dict
        New variable name -> metadata ({path, op, legacy_actual, legacy_copy}).
    """
    return _Evaluator(data, allow_rechunk=allow_rechunk).run(budgets)
