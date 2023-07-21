from operator import mul
from functools import reduce
import copy

def aggregate(budgets, decompose=[]):
    new_budgets = copy.deepcopy(budgets)
    for tr,budget in budgets.items():
        for part,terms in budget.items():
            if part in ["lhs", "rhs"]:
                new_budgets[tr][part] = deep_search(disaggregate(budget[part], decompose=decompose))
    return new_budgets

def disaggregate(b, decompose=[]):
    if "sum" in b:
        bsum_novar = {k:v for (k,v) in b["sum"].items() if k!="var"}
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
                budget_fill(ds, v[part], f"{eq}_{part}", mode="sum_first")
                budget_fill(ds, v[part], f"{eq}_{part}", mode="product_first")
                
def budget_fill(ds, budget, namepath, mode="sum_first"):
    if type(budget) is float:
        return budget
    elif type(budget) is str:
        return ds[budget]
    elif type(budget) is dict:
        if mode=="sum_first":
            if "sum" in budget:
                vname = f"{namepath}_sum"
                sum_list = [budget_fill(ds, v, f"{namepath}_{k}", mode=mode) for k,v in budget["sum"].items() if k!="var"]
                budget["sum"]["var"] = vname
                if budget["var"] is None:
                    budget["var"] = vname
                if vname not in ds:
                    ds[vname] = sum([da for da in sum_list])
                return ds[vname]
            if "var" in budget:
                if budget["var"] is not None:
                    return ds[budget["var"]]
            if "product" in budget:
                vname = f"{namepath}_product"
                mul_list = [budget_fill(ds, v, f"{namepath}", mode=mode) for k,v in budget["product"].items() if k!="var"]
                budget["product"]["var"] = vname
                if budget["var"] is None:
                    budget["var"] = vname
                if vname not in ds:
                    ds[vname] = reduce(mul, [e for e in mul_list], 1)
                return ds[vname]
            if "var" in budget:
                if budget["var"] is not None:
                    return ds[budget["var"]]
        elif mode=="product_first":
            if "product" in budget:
                vname = f"{namepath}_product"
                mul_list = [budget_fill(ds, v, f"{namepath}", mode=mode) for k,v in budget["product"].items() if k!="var"]
                budget["product"]["var"] = vname
                if budget["var"] is None:
                    budget["var"] = vname
                if vname not in ds:
                    ds[vname] = reduce(mul, [e for e in mul_list], 1)
                return ds[vname]
            if "sum" in budget:
                vname = f"{namepath}_sum"
                sum_list = [budget_fill(ds, v, f"{namepath}_{k}", mode=mode) for k,v in budget["sum"].items() if k!="var"]
                budget["sum"]["var"] = vname
                if budget["var"] is None:
                    budget["var"] = vname
                if vname not in ds:
                    ds[vname] = sum([da for da in sum_list])
                return ds[vname]
            if "var" in budget:
                if budget["var"] is not None:
                    return ds[budget["var"]]
    else:
        raise ValueError("Broken.")

def flatten(container):
    for i in container:
        if isinstance(i, (list,tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i
            
def flatten_lol(lol):
    return list(flatten(lol))
