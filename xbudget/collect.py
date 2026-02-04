from operator import mul
from functools import reduce
import copy
import numpy as np
import numbers
import xarray as xr
import xgcm

import warnings

def aggregate(xbudget_dict, decompose=[]):
    """Aggregate xbudget dictionary into simpler root-level budgets.

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
    new_budgets = copy.deepcopy(xbudget_dict)
    for tr, tr_xbudget_dict in xbudget_dict.items():
        for side,terms in tr_xbudget_dict.items():
            if side in ["lhs", "rhs"]:
                new_budgets[tr][side] = deep_search(
                    disaggregate(tr_xbudget_dict[side], decompose=decompose)
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
    if "sum" in b:
        bsum_novar = {k:v for (k,v) in b["sum"].items() if (k!="var") and (v is not None)}
        sum_dict = dict((k,v["var"]) if ("var" in v) else (k,v) for k,v in bsum_novar.items())
        b_recurse = {}
        for (k,v) in sum_dict.items():
            if k not in decompose:
                b_recurse[k] = v
            else:
                v_dict = disaggregate(b["sum"][k], decompose=decompose)
                if "product" in v_dict.keys():
                    b_recurse[k] = v_dict["var"]
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
    return _deep_search(b, new_b={}, k_last=None)

def _deep_search(b, new_b={}, k_last=None):
    """Recursive function for searching for variables in xbudget dictionary.
    
    See also
    --------
    aggregate, deep_search
    """
    if type(b) is str:
        new_b[k_last] = b
    elif type(b) is dict:
        for (k, v) in b.items():
            if k_last is not None:
                k = f"{k_last}_{k}"
            _deep_search(v, new_b=new_b, k_last=k)
        return new_b

def collect_budgets(ds, xbudget_dict):
    """Fills xbudget dictionary with all tracer content tendencies

    Parameters
    ----------
    ds : xr.Dataset containing budget diagnostics
    xbudget_dict : dictionary in xbudget-compatible format
        Example format:
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
    """
    for eq, v in xbudget_dict.items():
        for side in ["lhs", "rhs"]:
            if side in v:
                budget_fill_dict(ds, v[side], f"{eq}_{side}")

def budget_fill_dict(data, xbudget_dict, namepath):
    """Recursively fill xbudget dictionary

    Parameters
    ----------
    data : xgcm.grid or xr.Dataset
    xbudget_dict : dictionary in xbudget-compatible format containing variable in namepath
    namepath : name of variable in dataset (data._ds or data)
    """
    if type(data)==xgcm.grid.Grid:
        grid = data
        ds = grid._ds
    else:
        ds = data
        grid = None
    
    var_pref = None

    if ((xbudget_dict["var"] is not None) and
        (xbudget_dict["var"] in ds)       and
        (namepath not in ds)):
        var_rename = ds[xbudget_dict["var"]].rename(namepath)
        var_rename.attrs['provenance'] = xbudget_dict["var"]
        ds[namepath] = ds[xbudget_dict["var"]]
        var_pref = ds[namepath]

    for k,v in xbudget_dict.items():
        if k in ['sum', 'product']:
            op_list = []
            for k_term, v_term in v.items():
                if isinstance(v_term, dict): # recursive call to get this variable
                    v_term_recursive = budget_fill_dict(data, v_term, f"{namepath}_{k}_{k_term}")
                    if v_term_recursive is not None:
                        op_list.append(v_term_recursive)
                    elif v_term.get("var") is not None and v_term.get("var") not in ds:
                        warnings.warn(f"Variable {v_term.get('var')} is missing from the dataset `ds`, so it is being skipped. To suppress this warning, remove {v_term.get('var')} from the `xbudget_dict`.", UserWarning)
                elif isinstance(v_term, numbers.Number):
                    op_list.append(v_term)
                elif isinstance(v_term, str):
                    if v_term in ds:
                        op_list.append(ds[v_term])
                    else:
                        warnings.warn(f"Variable {v_term} is missing from the dataset `ds`, so it is being skipped. To suppress this warning, remove {v_term} from the `xbudget_dict`.", UserWarning)
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
                var = sum(op_list) if k=="sum" else reduce(mul, op_list, 1)
                if not isinstance(var, xr.DataArray):
                    continue

            # Variable metadata
            var_name = f"{namepath}_{k}"
            var = var.rename(var_name)
            var_provenance = [o.name if isinstance(o, xr.DataArray) else o for o in op_list]
            var.attrs["provenance"] = var_provenance
            ds[var_name] = var
            if (xbudget_dict[k]["var"] is None):
                xbudget_dict[k]["var"] = var_name

            if (xbudget_dict["var"] is None):
                var_copy = var.copy()
                var_copy.attrs["provenance"] = var_name
                xbudget_dict["var"] = namepath
                if namepath not in ds:
                    ds[namepath] = var_copy

            # keep record of the first-listed variable
            if var_pref is None:
                var_pref = var.copy()
                
        if k == "difference":
            if grid is not None:
                staggered_axes = {
                    axn:c for axn,ax in grid.axes.items()
                    for pos,c in ax.coords.items()
                    if pos!="center"
                }
                v_term = [v_term for k_term,v_term in v.items() if k_term!="var"][0]
                if v_term not in ds:
                    warnings.warn(f"Variable {v_term} is missing from the dataset `ds`, so it is being skipped. To suppress this warning, remove {v_term} from the `xbudget_dict`.")
                    continue
                candidate_axes = [axn for (axn,c) in staggered_axes.items() if c in ds[v_term].dims]
                if len(candidate_axes) == 1:
                    axis = candidate_axes[0]
                else:
                    raise ValueError("Flux difference inconsistent with finite volume discretization.")
                var = grid.diff(ds[v_term].fillna(0.), axis)
                var_name = f"{namepath}_difference"
                var = var.rename(var_name)
                var_provenance = v_term
                var.attrs["provenance"] = var_provenance
                ds[var_name] = var
                if var_pref is None:
                    var_pref = var.copy()
            else:
                raise ValueError("Input `ds` must be `xgcm.Grid` instance if using `difference` operations.")

    return var_pref

def get_vars(xbudget_dict, terms):
    """Get xbudget sub-dictionaries for specified terms.

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
    return _get_vars(xbudget_dict, terms)

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
