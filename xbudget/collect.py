"""The public entry point: parse a recipe and evaluate it into a dataset.

``collect_budgets`` is a thin wrapper over the typed engine
(:func:`xbudget.parse.parse_budgets` then
:func:`xbudget.evaluate.evaluate_budgets`). It never mutates the recipe; it only
adds derived variables to the dataset. Query the result with
:class:`xbudget.query.BudgetQuery`.
"""
import warnings

import xarray as xr

__all__ = [
    "collect_budgets",
    "lateral_divergence",
]


def _warn_if_summands_broadcast(op_list, name):
    """Warn when a ``sum`` mixes operands of differing dimensionality (issue #11).

    In a finite-volume budget every term of a ``sum`` should live on the same
    grid and therefore carry identical dimensions. When they do not, ``xarray``
    silently broadcasts the lower-rank operand across the dimensions it lacks —
    so a 2D surface flux summed with a 3D flux convergence is spread *uniformly*
    over the vertical rather than deposited at the single level that outcrops.
    That is almost never the intended finite-volume broadcast, and the recipe
    does not carry the outcropping level needed to do it correctly, so for now
    we surface the situation loudly instead of returning a wrong budget.

    Only ``xr.DataArray`` operands are compared; scalar constants (``sign``,
    ``density``, …) are ignored.
    """
    dim_sets = [frozenset(o.dims) for o in op_list if isinstance(o, xr.DataArray)]
    if len(dim_sets) < 2:
        return
    common = frozenset.intersection(*dim_sets)
    broadcast_dims = frozenset.union(*dim_sets) - common
    if broadcast_dims:
        warnings.warn(
            f"Summing terms with mismatched dimensions while building "
            f"'{name}': the operands carry dimension sets "
            f"{[tuple(sorted(s)) for s in dim_sets]}. xarray will broadcast the "
            f"lower-dimensional term(s) across {tuple(sorted(broadcast_dims))}, "
            f"e.g. spreading a 2D surface flux uniformly over the vertical of a "
            f"3D flux convergence instead of depositing it at the outcropping "
            f"level. Verify this broadcast is intended; see "
            f"https://github.com/hdrake/xbudget/issues/11.",
            UserWarning,
        )


def lateral_divergence(grid, Fx, Fy):
    """Horizontal flux divergence ``div(Fx, Fy)`` on cell centers, via xgcm.

    Uses ``grid.diff`` with a vector ``other_component`` so face-connected
    topologies (e.g. the LLC tiles of ECCO) are stitched correctly, for any grid
    topology xgcm supports.
    """
    if grid is None:
        raise ValueError(
            "Input `data` must be an `xgcm.Grid` instance when using "
            "`lateral_divergence` operations."
        )
    dFx = grid.diff({"X": Fx}, "X", other_component={"Y": Fy})
    dFy = grid.diff({"Y": Fy}, "Y", other_component={"X": Fx})
    return dFx + dFy


def collect_budgets(data, recipe, allow_rechunk=True, on_missing="warn"):
    """Materialize every budget term described by ``recipe`` into ``data``.

    The recipe dict is parsed into a typed expression tree
    (:mod:`xbudget.nodes`) and evaluated
    (:func:`xbudget.evaluate.evaluate_budgets`). It does **not** mutate
    ``recipe``; it only adds derived variables to the dataset. Each derived
    variable is named by its term path with operator infixes dropped (e.g.
    ``heat_rhs_diffusion_lateral``) and carries ``xbudget_path``, ``xbudget_op``,
    and ``provenance`` attributes. Query the result with
    :class:`xbudget.query.BudgetQuery`.

    Parameters
    ----------
    data : xgcm.Grid or xr.Dataset
        Budget diagnostics to read from and write derived variables into,
        modified in place. A grid is required if the recipe uses
        ``difference`` or ``lateral_divergence`` operations.
    recipe : dict
        A recipe in xbudget format (e.g. from ``load_preset_budget``).
    allow_rechunk : bool (default: True)
        Whether to temporarily rechunk when taking differences along a dimension,
        e.g. to compute flux divergences on `center` from fluxes on `outer` or
        tendencies on `center` from snapshots on `outer`.
    on_missing : {"warn", "raise", "ignore"} (default: "warn")
        How to handle a *required* diagnostic that a term references but that is
        absent from ``data``. ``"warn"`` (default) skips the term and emits one
        end-of-run summary ``UserWarning`` naming the missing diagnostics and the
        now-incomplete terms; ``"raise"`` raises
        :class:`~xbudget.evaluate.MissingDiagnosticError` so a recipe/dataset
        mismatch fails loudly; ``"ignore"`` skips silently. In every case the
        affected variables are stamped with ``xbudget_incomplete`` /
        ``xbudget_missing`` attributes (query them with
        :meth:`xbudget.BudgetQuery.missing`), and a term declared ``optional`` in
        the recipe is exempt — its absence is expected and never alarms.

    Returns
    -------
    data : xgcm.Grid or xr.Dataset
        The same object passed in, for convenience.

    Examples
    --------
    Load a preset recipe, collect its budgets into a grid, and query the result:

    >>> recipe = xbudget.load_preset_budget(model="MOM6")
    >>> xbudget.collect_budgets(grid, recipe)      # adds derived vars in place
    >>> q = xbudget.BudgetQuery(grid, recipe)
    >>> q.var("heat_rhs")                          # the closed RHS heat tendency
    'heat_rhs'
    >>> grid._ds["heat_rhs"].attrs["xbudget_path"]
    ['heat', 'rhs']
    """
    from .parse import parse_budgets
    from .evaluate import evaluate_budgets

    budgets = parse_budgets(recipe)
    evaluate_budgets(
        data, budgets, allow_rechunk=allow_rechunk, on_missing=on_missing
    )
    return data
