from operator import mul
from functools import reduce
import copy
import numpy as np
import numbers
import xarray as xr
import xgcm

import warnings

from .nodes import OPERATION_KEYS

__all__ = [
    "aggregate",
    "disaggregate",
    "deep_search",
    "collect_budgets",
    "budget_fill_dict",
    "get_vars",
    "flatten",
    "flatten_lol",
    "lateral_divergence",
]

def _warn_legacy_helper(name):
    """Warn that a recipe-reading query helper is on its way out.

    ``FutureWarning`` rather than ``DeprecationWarning``: the latter is hidden
    unless the call is in ``__main__``, so a user calling xbudget from their own
    analysis module — most of them — would never see it.
    """
    warnings.warn(
        f"xbudget.{name}() is deprecated and will be removed in xbudget v1.0. "
        f"It reads the `var` fields that only a (deprecated) "
        f"name_scheme='legacy' run fills into `xbudget_dict`. Use "
        f"xbudget.BudgetQuery(data, xbudget_dict) instead, which queries the "
        f"default v1 output; see CHANGELOG.md for the equivalents.",
        FutureWarning,
        stacklevel=3,
    )

def _is_unfilled_recipe(xbudget_dict):
    """True if the recipe has derived terms but none was ever filled in.

    The legacy engine records each derived term's variable name in its ``var``
    field as it goes. If every derived term's ``var`` is still empty, the recipe
    has not been through that engine: either it was never collected, or it was
    collected with the default ``name_scheme="v1"``, which deliberately leaves
    the recipe alone. The dict-reading helpers then have nothing to report and
    would hand back a silently empty answer.

    Only terms that *could* have been filled count. A skeleton convention like
    ``MOM6_drift`` — whose terms are declared but reference no diagnostics — has
    nothing to materialize, so a legacy run legitimately leaves it empty and
    this returns False. Otherwise the check could not tell "never ran the legacy
    engine" from "ran it, but every term was skipped".
    """
    fillable, filled = 0, 0

    def references_a_diagnostic(node):
        """Does this subtree name any dataset variable to build from?"""
        if isinstance(node, str):
            return True
        if isinstance(node, dict):
            return any(references_a_diagnostic(v) for v in node.values())
        return False

    def walk(node):
        nonlocal fillable, filled
        if not isinstance(node, dict):
            return
        ops = {k: v for k, v in node.items() if k in OPERATION_KEYS}
        if ops and any(references_a_diagnostic(v) for v in ops.values()):
            fillable += 1
            if node.get("var") is not None:
                filled += 1
        for k, v in node.items():
            if k != "var":
                walk(v)

    for budget in xbudget_dict.values():
        if isinstance(budget, dict):
            for side in ("lhs", "rhs"):
                walk(budget.get(side))
    return fillable > 0 and filled == 0


def _require_filled_recipe(xbudget_dict, func):
    """Fail loudly rather than return a silently empty result."""
    if not _is_unfilled_recipe(xbudget_dict):
        return
    raise ValueError(
        f"xbudget.{func}() found no filled-in terms in `xbudget_dict`, so it "
        f"has nothing to report. It reads the `var` fields that only the "
        f"deprecated `collect_budgets(..., name_scheme='legacy')` writes into "
        f"the recipe; the default `name_scheme='v1'` leaves the recipe "
        f"untouched (and you may not have called collect_budgets at all). "
        f"Query the v1 output instead:\n\n"
        f"    xbudget.collect_budgets(data, xbudget_dict)\n"
        f"    q = xbudget.BudgetQuery(data, xbudget_dict)\n"
        f"    q.aggregate()          # or q.var(...) / q.get_vars(...)\n\n"
        f"See CHANGELOG.md for the full migration."
    )


def _warn_missing_variable(name):
    """Warn that a requested variable is absent from the dataset and skipped."""
    warnings.warn(
        f"Variable {name} is missing from the dataset `ds`, so it is being "
        f"skipped. To suppress this warning, remove {name} from the "
        f"`xbudget_dict`.",
        UserWarning,
    )


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
    topologies (e.g. the LLC tiles of ECCO) are stitched correctly. This
    reproduces the previously hand-rolled LLC90 flux stitching exactly, for any
    grid topology xgcm supports.
    """
    if grid is None:
        raise ValueError(
            "Input `data` must be an `xgcm.Grid` instance when using "
            "`lateral_divergence` operations."
        )
    dFx = grid.diff({"X": Fx}, "X", other_component={"Y": Fy})
    dFy = grid.diff({"Y": Fy}, "Y", other_component={"X": Fx})
    return dFx + dFy

def aggregate(xbudget_dict, decompose=[]):
    """Aggregate xbudget dictionary into simpler root-level budgets.

    .. deprecated:: 0.7.0
        Removed in v1.0. This reads the ``var`` fields that the engine fills in,
        so it only returns meaningful results after a (also deprecated) legacy
        run, i.e. ``collect_budgets(data, xbudget_dict, name_scheme="legacy")``.
        Use :meth:`xbudget.BudgetQuery.aggregate` instead, which queries the
        default ``name_scheme="v1"`` output::

            q = xbudget.BudgetQuery(grid, xbudget_dict)
            q.aggregate(decompose=["diffusion"])

    Parameters
    ----------
    xbudget_dict : dictionary in xbudget-compatible format
    decompose : str or list (default: [])
        Name of variable type(s) to decompose into the summed parts

    Examples
    --------
    >>> xbudget_dict = {
        "heat": {
            "rhs": {
                "sum": {
                    "advection": {
                        "var":"advective_tendency"
                    },
                    "var": "heat_rhs_sum"
                },
                "var": "heat_rhs",
            }
        }
    }
    >>> xbudget.aggregate(xbudget_dict)
    {'heat': {'rhs': {'advection': 'advective_tendency'}}}

    >>>xbudget_dict = {
        "heat": {
            "rhs": {
                "sum": {
                    "advection": {
                        "var":"advective_tendency",
                        "sum": {
                            "horizontal": {
                                "var":"advective_tendency_h",
                            },
                            "vertical": {
                                "var":"advective_tendency_v"
                            },
                            "var":"heat_rhs_sum_advection_sum"
                        }
                    },
                    "var": "heat_rhs_sum"
                },
                "var": "heat_rhs",
            }
        }
    }
    >>> xbudget.aggregate(xbudget_dict)
    {'heat': {'rhs': {'advection': 'advective_tendency'}}}

    >>> xbudget.aggregate(xbudget_dict, decompose="advection")
    {'heat': {'rhs': {'advection_horizontal': 'advective_tendency_h',
    'advection_vertical': 'advective_tendency_v'}}}

    See also
    --------
    disaggregate, deep_search, _deep_search
    """
    _warn_legacy_helper("aggregate")
    _require_filled_recipe(xbudget_dict, "aggregate")
    new_budgets = copy.deepcopy(xbudget_dict)
    for tr, tr_xbudget_dict in xbudget_dict.items():
        for side,terms in tr_xbudget_dict.items():
            if side in ["lhs", "rhs"]:
                new_budgets[tr][side] = _deep_search(
                    _disaggregate(tr_xbudget_dict[side], decompose=decompose)
                )
    return new_budgets

def disaggregate(b, decompose=[]):
    """Disaggregate variable's provenance dictionary into summed parts

    Parameters
    ----------
    b : xbudget sub-dictionary for a variable
    decompose : str or list (default: [])
        Name of variable type(s) to decompose into the summed parts

    Examples
    --------
    >>> b = {
        "sum": {
            "advection": {
                "var":"advective_tendency",
                "sum": {
                    "horizontal": {
                        "var":"advective_tendency_h",
                    },
                    "vertical": {
                        "var":"advective_tendency_v"
                    },
                    "var":"heat_rhs_sum_advection_sum"
                }
            },
            "var": "heat_rhs_sum"
        },
        "var": "heat_rhs",
    }
    >>> {'advection': 'advective_tendency'}
    {'advection': 'advective_tendency'}

    >>> xbudget.disaggregate(b, decompose="advection")
    {'advection': {'horizontal': 'advective_tendency_h',
    'vertical': 'advective_tendency_v'}}
    
    See also
    --------
    aggregate
    """
    _warn_legacy_helper("disaggregate")
    return _disaggregate(b, decompose=decompose)

def _disaggregate(b, decompose=[]):
    """Recursive body of :func:`disaggregate` (warning-free, so it fires once)."""
    if "sum" in b:
        bsum_novar = {k:v for (k,v) in b["sum"].items() if (k!="var") and (v is not None)}
        # A term node reports its `var`, absent or null alike (both mean "the
        # engine did not fill this in"); a bare constant/variable operand
        # reports itself. Reading it with .get keeps `var: null` optional --
        # subscripting would KeyError, and testing `"var" in v` would fall
        # through and hand the whole node dict back to the caller.
        sum_dict = dict(
            (k, v.get("var")) if isinstance(v, dict) else (k, v)
            for k, v in bsum_novar.items()
        )
        b_recurse = {}
        for (k,v) in sum_dict.items():
            if k not in decompose:
                b_recurse[k] = v
            else:
                v_dict = _disaggregate(b["sum"][k], decompose=decompose)
                if "product" in v_dict.keys():
                    b_recurse[k] = v_dict.get("var")
                else:
                    b_recurse[k] = v_dict
        return b_recurse
    return b

def deep_search(b):
    """Utility function for searching for variables in xbudget dictionary.

    See also
    --------
    aggregate, _deep_search
    """
    _warn_legacy_helper("deep_search")
    return _deep_search(b, new_b=None, k_last=None)

def _deep_search(b, new_b=None, k_last=None):
    """Recursive function for searching for variables in xbudget dictionary.

    See also
    --------
    aggregate, deep_search
    """
    if new_b is None:
        new_b = {}
    if type(b) is str:
        new_b[k_last] = b
    elif type(b) is dict:
        for (k, v) in b.items():
            if k_last is not None:
                k = f"{k_last}_{k}"
            _deep_search(v, new_b=new_b, k_last=k)
        return new_b

def collect_budgets(data, xbudget_dict, allow_rechunk=True, name_scheme="v1"):
    """Materialize every budget term described by ``xbudget_dict`` into ``data``.

    The convention dict is parsed into a typed expression tree
    (:mod:`xbudget.nodes`) and evaluated (:func:`xbudget.evaluate.evaluate_budgets`).
    Unlike the historical engine, this does **not** mutate ``xbudget_dict``; it
    only adds derived variables to the dataset.

    Parameters
    ----------
    data : xgcm.Grid or xr.Dataset
        Budget diagnostics to read from and write derived variables into,
        modified in place. A grid is required if the convention uses
        ``difference`` or ``lateral_divergence`` operations.
    xbudget_dict : dict
        A convention in xbudget format (e.g. from ``load_preset_budget``).
    allow_rechunk : bool (default: True)
        Whether to temporarily rechunk when taking differences along a dimension,
        e.g. to compute flux divergences on `center` from fluxes on `outer` or
        tendencies on `center` from snapshots on `outer`.
    name_scheme : {"v1", "legacy"} (default: "v1")
        ``"v1"`` (recommended) uses the typed engine: each derived variable is
        named by its term path with operator infixes dropped (e.g.
        ``heat_rhs_diffusion_lateral``) and the recipe dict is left untouched.
        ``"legacy"`` reproduces the historical behavior exactly: the
        operator-suffixed names (e.g. ``heat_rhs_sum_diffusion_sum_lateral_product``)
        plus their plain "copy" aliases, **and it mutates ``xbudget_dict`` in
        place** to fill in ``var`` fields (which the legacy ``get_vars`` /
        ``aggregate`` query helpers rely on).

    Returns
    -------
    data : xgcm.Grid or xr.Dataset
        The same object passed in, for convenience.
    """
    if name_scheme == "legacy":
        warnings.warn(
            "name_scheme='legacy' is deprecated and will be removed in xbudget "
            "v1.0, along with the `name_scheme` argument itself (v1 naming will "
            "be the only behavior). Legacy mode also mutates `xbudget_dict` in "
            "place. Migrate to the default name_scheme='v1' and query the result "
            "with xbudget.BudgetQuery(data, xbudget_dict) instead of "
            "get_vars()/aggregate(); see CHANGELOG.md for the legacy->v1 name "
            "mapping.",
            FutureWarning,
            stacklevel=2,
        )
        # Faithful legacy behavior: reuse the reference engine, which emits the
        # historical names and fills the recipe dict in place so that the
        # dict-based query helpers (get_vars/aggregate) keep working.
        for eq, sides in xbudget_dict.items():
            for side in ("lhs", "rhs"):
                if side in sides:
                    budget_fill_dict(
                        data, sides[side], f"{eq}_{side}", allow_rechunk=allow_rechunk
                    )
        return data

    if name_scheme != "v1":
        raise ValueError(
            f"Unknown name_scheme {name_scheme!r}; expected 'v1' or 'legacy'."
        )

    from .parse import parse_budgets
    from .evaluate import evaluate_budgets

    budgets = parse_budgets(xbudget_dict)
    evaluate_budgets(data, budgets, allow_rechunk=allow_rechunk)
    return data

def budget_fill_dict(data, xbudget_dict, namepath, allow_rechunk = True):
    """Recursively fill xbudget dictionary (legacy engine).

    .. deprecated::
        The historical dict-walking engine, retained as the reference
        implementation behind ``name_scheme="legacy"``. Prefer
        :func:`collect_budgets` (typed engine); this mutates both ``data`` and
        ``xbudget_dict`` in place.

    Parameters
    ----------
    data : xgcm.grid or xr.Dataset
    xbudget_dict : dictionary in xbudget-compatible format containing variable in namepath
    namepath : name of variable in dataset (data._ds or data)
    allow_rechunk : bool (default: True)
        Whether to temporarily rechunk when taking differences along a dimension,
        e.g. to compute flux divergences on `center` from fluxes on `outer` or
        tendencies on `center` from snapshots on `outer`.
    """
    if type(data)==xgcm.grid.Grid:
        grid = data
        ds = grid._ds
    else:
        ds = data
        grid = None
    
    var_pref = None

    # `var` is optional: an absent key means the same as `var: null` (a value
    # this engine fills in), so read it with .get rather than subscripting.
    explicit_var = xbudget_dict.get("var")

    if ((explicit_var is not None) and
        (explicit_var in ds)       and
        (namepath not in ds)):
        var_rename = ds[explicit_var].rename(namepath)
        var_rename.attrs['provenance'] = explicit_var
        ds[namepath] = ds[explicit_var]
        var_pref = ds[namepath]

    # Snapshot the items: this loop fills `var` into the recipe as it goes, and
    # with `var` optional that can *add* a key rather than overwrite one, which
    # would otherwise raise "dictionary changed size during iteration".
    for k,v in list(xbudget_dict.items()):
        if k in ['sum', 'product']:
            op_list = []
            for k_term, v_term in v.items():
                if isinstance(v_term, dict): # recursive call to get this variable
                    v_term_recursive = budget_fill_dict(data, v_term, f"{namepath}_{k}_{k_term}", allow_rechunk = allow_rechunk)
                    if v_term_recursive is not None:
                        op_list.append(v_term_recursive)
                    elif v_term.get("var") is not None and v_term.get("var") not in ds:
                        _warn_missing_variable(v_term.get("var"))
                elif isinstance(v_term, numbers.Number):
                    op_list.append(v_term)
                elif isinstance(v_term, str):
                    if v_term in ds:
                        op_list.append(ds[v_term])
                    else:
                        _warn_missing_variable(v_term)
                        if k=="product":
                            op_list.append(0.)

            # Compute variable from sum or product operation
            if (
                (len(op_list) == 0) |
                all([e is None for e in op_list]) |
                any([e is None for e in op_list])
            ):
                return None
            else:
                if k == "sum":
                    _warn_if_summands_broadcast(op_list, f"{namepath}_{k}")
                var = sum(op_list) if k=="sum" else reduce(mul, op_list, 1)
                if not isinstance(var, xr.DataArray):
                    continue

            # Variable metadata
            var_name = f"{namepath}_{k}"
            var = var.rename(var_name)
            var_provenance = [o.name if isinstance(o, xr.DataArray) else o for o in op_list]
            var.attrs["provenance"] = var_provenance
            ds[var_name] = var
            if (xbudget_dict[k].get("var") is None):
                xbudget_dict[k]["var"] = var_name

            if (xbudget_dict.get("var") is None):
                var_copy = var.copy()
                var_copy.attrs["provenance"] = var_name
                xbudget_dict["var"] = namepath
                if namepath not in ds:
                    ds[namepath] = var_copy

            # keep record of the first-listed variable
            if var_pref is None:
                var_pref = var.copy()
        
        if k == "reciprocal":
            v_term = [v_term for k_term, v_term in v.items() if k_term != "var"][0]
            # The source is a variable name, either bare or wrapped as
            # {var: name} (matching parse._parse_reciprocal).
            source = v_term.get("var") if isinstance(v_term, dict) else v_term
            if source not in ds:
                _warn_missing_variable(source)
                continue

            #A safe reciprocal that filters zeros out.
            var = 1.0 / xr.where(ds[source] == 0, np.inf, ds[source])

            var_name = f"{namepath}_reciprocal"
            var = var.rename(var_name)
            var.attrs["provenance"] = source
            ds[var_name] = var
            if v.get('var') is None:
                v['var'] = var_name
            if xbudget_dict.get("var") is None:
                var_copy = var.copy()
                var_copy.attrs["provenance"] = var_name
                xbudget_dict["var"] = namepath
                if namepath not in ds:
                    ds[namepath] = var_copy
            if var_pref is None:
                var_pref = var.copy()

        if k == "lateral_divergence":
            Fx = budget_fill_dict(data, v["Fx"], f"{namepath}_Fx", allow_rechunk=allow_rechunk)
            Fy = budget_fill_dict(data, v["Fy"], f"{namepath}_Fy", allow_rechunk=allow_rechunk)
            if Fx is None or Fy is None:
                warnings.warn(f"Could not compute fluxes for {namepath}, skipping.")
                continue

            var = lateral_divergence(grid, Fx, Fy)
            var_name = f"{namepath}_lateral_divergence"
            var = var.rename(var_name)
            var.attrs["provenance"] = [Fx.name, Fy.name]
            ds[var_name] = var
            if v.get("var") is None:
                v["var"] = var_name

            if xbudget_dict.get("var") is None:
                var_copy = var.copy()
                var_copy.attrs["provenance"] = var_name
                xbudget_dict["var"] = namepath
                if namepath not in ds:
                    ds[namepath] = var_copy
            
            if var_pref is None:
                var_pref = var.copy()

        if k == "difference":
            if grid is None:
                raise ValueError(
                    "Input `data` must be an `xgcm.Grid` instance when using "
                    "`difference` operations."
                )
            staggered_axes = {
                axn:c for axn,ax in grid.axes.items()
                for pos,c in ax.coords.items()
                if pos!="center"
            }
            k_term, v_term = [(k_term, v_term) for k_term, v_term in v.items() if k_term != "var"][0]
            if isinstance(v_term, dict):
                source = budget_fill_dict(data, v_term, f"{namepath}_difference_{k_term}", allow_rechunk = allow_rechunk)
                if source is None:
                    continue
            else:
                if v_term not in ds:
                    _warn_missing_variable(v_term)
                    continue
                source = ds[v_term]

            candidate_axes = [axn for (axn,c) in staggered_axes.items() if c in source.dims]
            if len(candidate_axes) == 1:
                axis = candidate_axes[0]
            else:
                raise ValueError("Finite difference inconsistent with finite volume discretization.")

            if allow_rechunk:
                try: #extract original chunks when possible
                    #not using source since it may not have the non-staggered dimension chunks.
                    original_chunks = dict(ds.chunksizes)
                except Exception:
                    warnings.warn("Dataset chunks are inconsistent; using unify_chunks()", UserWarning)
                    original_chunks = dict(ds.unify_chunks().chunksizes)

                # Find the staggered dimension for the given axis in the DataArray
                axis_dim = [d for d in source.dims if d in grid.axes[axis].coords.values()]
                if len(axis_dim) != 1:
                    raise ValueError(f"Expected to find one dimension for axis '{axis}' in variable '{source.name}', but found {len(axis_dim)}: {axis_dim}")
                axis_dim = axis_dim[0]
            
                # Temporarily rechunk to put the difference dim in a single chunk, all other chunks are auto.
                temporary_chunks = {axis_dim: -1, **{d: "auto" for d in source.dims if d != axis_dim}}
                var = grid.diff(source.chunk(temporary_chunks).fillna(0.0), axis=axis)
                # Restore the original chunking of the dimensions we know; leave
                # any others as `grid.diff` produced them. Only naming the dims we
                # want changed avoids reading `var.chunksizes`, which raises when
                # the result's coords are chunked differently from its data.
                restore = {d: original_chunks[d] for d in var.dims if d in original_chunks}
                if restore:
                    var = var.chunk(restore)
            else:
                var = grid.diff(source.fillna(0.), axis)

            var_name = f"{namepath}_difference"
            var = var.rename(var_name)
            var_provenance = source.name
            var.attrs["provenance"] = var_provenance
            ds[var_name] = var
            if var_pref is None:
                var_pref = var.copy()



    return var_pref

def get_vars(xbudget_dict, terms):
    """Get xbudget sub-dictionaries for specified terms.

    .. deprecated:: 0.7.0
        Removed in v1.0. Reads the ``var`` fields filled in by a (also
        deprecated) legacy run, i.e.
        ``collect_budgets(data, xbudget_dict, name_scheme="legacy")``. Use
        :meth:`xbudget.BudgetQuery.get_vars` / :meth:`xbudget.BudgetQuery.var`
        instead, which query the default ``name_scheme="v1"`` output::

            q = xbudget.BudgetQuery(grid, xbudget_dict)
            q.var("heat_rhs_diffusion")

    Parameters
    ----------
    xbudget_dict : dictionary in xbudget-compatible format
    terms : str or list of str

    Examples
    -------
    >>> xbudget_dict = {
        "heat": {
            "rhs": {
                "sum": {
                    "advection": {
                        "var":"advective_tendency"
                    },
                    "var": "heat_rhs_sum"
                },
                "var": "heat_rhs",
            }
        }
    }
    >>> xbudget.get_vars(xbudget_dict, "heat_rhs_sum")
    {'var': 'heat_rhs_sum', 'sum': ['advective_tendency']}
    """
    _warn_legacy_helper("get_vars")
    result = _get_vars(xbudget_dict, terms)
    if result is None:
        # A miss is usually a typo, but on an unfilled recipe *every* derived
        # term misses, and the caller would get an opaque `None` (and then a
        # TypeError on ["var"]). Explain that case rather than let it surface
        # three lines later.
        _require_filled_recipe(xbudget_dict, "get_vars")
    return result

def _get_vars(b, terms, k_long=""):
    """Recursive version of _get_vars for determining variable provenance tree.
    
    Parameters
    ----------
    b : dictionary
    terms : str or list of str
    k_long : variable name suffix
    
    See also
    --------
    get_vars
    """
    if isinstance(terms, (list, np.ndarray)):
        return [_get_vars(b, term) for term in terms]
    elif type(terms) is str:
        for k,v in b.items():
            if type(v) is str:
                k_short = k_long.replace("_sum", "").replace("_product", "")
                if v==terms:
                    decomps = {"var": v}
                    if len(terms) > len("_sum"):
                        if (terms[-len("_sum"):] == "_sum") and ("sum" in b):
                            ts = {kk:vv for (kk,vv) in b["sum"].items() if kk!="var"}
                            decomps["sum"] = [vv["var"] if type(vv) is dict else vv for (kk,vv) in ts.items()]
                        elif (terms[-len("_sum"):] == "_sum"):
                            ts = {kk:vv for (kk,vv) in b.items() if kk!="var"}
                            decomps["sum"] = [vv["var"] if type(vv) is dict else vv for (kk,vv) in ts.items()]
                    if len(terms) > len("_product"):
                        if (terms[-len("_product"):] == "_product") and ("product" in b):
                            ts = {kk:vv for (kk,vv) in b["product"].items() if kk!="var"}
                            decomps["product"] = [vv["var"] if type(vv) is dict else vv for (kk,vv) in ts.items()]
                        elif (terms[-len("_product"):] == "_product"):
                            ts = {kk:vv for (kk,vv) in b.items() if kk!="var"}
                            decomps["product"] = [vv["var"] if type(vv) is dict else vv for (kk,vv) in ts.items()]
                    return decomps

                if k!="var":
                    k_short+="_"+k
                if k_short==terms:
                    return v
            elif type(v) is dict:
                if k_long=="":
                    new_k = k
                elif len(k_long)>0:
                    new_k = f"{k_long}_{k}"
                var = _get_vars(v, terms, k_long=new_k)
                if var is not None:
                    return var

def flatten(container):
    for i in container:
        if isinstance(i, (list,tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i
            
def flatten_lol(lol):
    """Flatten a list of lists into a single list."""
    return list(flatten(lol))
