from operator import mul
from functools import reduce
import copy
import numpy as np
import numbers
import xarray as xr

def aggregate(budgets, decompose=[]):
    new_budgets = copy.deepcopy(budgets)
    for tr,budget in budgets.items():
        for part,terms in budget.items():
            if part in ["lhs", "rhs"]:
                new_budgets[tr][part] = deep_search(
                    disaggregate(budget[part], decompose=decompose)
                )
    return new_budgets

def disaggregate(b, decompose=[]):
    if "sum" in b:
        bsum_novar = {k:v for (k,v) in b["sum"].items() if (k!="var") and (v is not None)}
        sum_dict = dict((k,v["var"]) if ("var" in v) else (k,v) for k,v in bsum_novar.items())
        return {k:v if k not in decompose else disaggregate(b["sum"][k], decompose=decompose) for (k,v) in sum_dict.items()}
    return b

def deep_search(b):
    return _deep_search(b, new_b={}, k_last=None)

def _deep_search(b, new_b={}, k_last=None):
    if type(b) is str:
        new_b[k_last] = b
    elif type(b) is dict:
        for (k, v) in b.items():
            if k_last is not None:
                k = f"{k_last}_{k}"
            _deep_search(v, new_b=new_b, k_last=k)
        return new_b

def collect_budgets(ds, budgets_dict):
    for eq, v in budgets_dict.items():
        for part in ["lhs", "rhs"]:
            if part in v:
                budget_fill_dict(ds, v[part], f"{eq}_{part}")

def budget_fill_dict(ds, bdict, namepath):
    var_pref = None

    if ((bdict["var"] is not None) and
        (bdict["var"] in ds)       and
        (namepath not in ds)):
        var_rename = ds[bdict["var"]].rename(namepath)
        var_rename.attrs['provenance'] = bdict["var"]
        ds[namepath] = ds[bdict["var"]]
        var_pref = ds[namepath]

    for k,v in bdict.items():
        if k in ['sum', 'product']:
            op_list = []
            for k_term, v_term in v.items():
                if isinstance(v_term, dict): # recursive call to get this variable
                    op_list.append(budget_fill_dict(ds, v_term, f"{namepath}_{k}_{k_term}"))
                elif isinstance(v_term, numbers.Number):
                    op_list.append(v_term)
                elif isinstance(v_term, str):
                    op_list.append(ds[v_term])

            # Compute variable from sum or product operation
            if (len(op_list) == 0) | all([e is None for e in op_list]):
                return None
            else:
                var = sum(op_list) if k=="sum" else reduce(mul, op_list, 1)

            # Variable metadata
            var_name = f"{namepath}_{k}"
            var = var.rename(var_name)
            var_provenance = [o.name if isinstance(o, xr.DataArray) else o for o in op_list]
            var.attrs["provenance"] = var_provenance
            ds[var_name] = var
            if (bdict[k]["var"] is None):
                bdict[k]["var"] = var_name

            if (bdict["var"] is None):
                var_copy = var.copy()
                var_copy.attrs["provenance"] = var_name
                bdict["var"] = namepath
                if namepath not in ds:
                    ds[namepath] = var_copy

            # keep record of the first-listed variable
            if var_pref is None:
                var_pref = var.copy()
                
        if k == "difference": # PLACEHOLDER–NOT YET SUPPORTED
            if var_pref is None:
                var_pref = None

    return var_pref
        
def get_vars(b, terms, k_long=""):
    if isinstance(terms, (list, np.ndarray)):
        return [get_vars(b, term) for term in terms]
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
                var = get_vars(v, terms, k_long=new_k)
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
    return list(flatten(lol))
