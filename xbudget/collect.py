from operator import mul
from functools import reduce
import copy

def aggregate(budgets):
    new_budgets = copy.deepcopy(budgets)
    for tr,budget in budgets.items():
        for part,terms in budget.items():
            if part in ["lhs", "rhs"]:
                new_budgets[tr][part] = {}
                for k,v in budget[part]["sum"].items():
                    if k=="var": continue
                    if "var" in v:
                        new_budgets[tr][part][k] = v['var']
                    else:
                        new_budgets[tr][part][k] = v
            elif part in ["surface_advective_ocean_flux"]:
                new_budgets[tr][part] = {part: budget[part]["var"]}
                
    return new_budgets

def collect_budgets(ds, budgets_dict):
    for eq, v in budgets_dict.items():
        for part in ["rhs", "lhs", "surface_advective_ocean_flux"]:
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
                ds[vname] = sum([v for v in sum_list])
                return ds[vname]
            elif "product" in budget:
                vname = f"{namepath}_product"
                mul_list = [budget_fill(ds, v, f"{namepath}", mode=mode) for k,v in budget["product"].items() if k!="var"]
                budget["product"]["var"] = vname
                if budget["var"] is None:
                    budget["var"] = vname
                ds[vname] = reduce(mul, [e for e in mul_list], 1)
                return ds[vname]
            elif "var" in budget:
                return ds[budget["var"]]
        elif mode=="product_first":
            if "product" in budget:
                vname = f"{namepath}_product"
                mul_list = [budget_fill(ds, v, f"{namepath}", mode=mode) for k,v in budget["product"].items() if k!="var"]
                budget["product"]["var"] = vname
                if budget["var"] is None:
                    budget["var"] = vname
                ds[vname] = reduce(mul, [e for e in mul_list], 1)
                return ds[vname]
            elif "sum" in budget:
                vname = f"{namepath}_sum"
                sum_list = [budget_fill(ds, v, f"{namepath}_{k}", mode=mode) for k,v in budget["sum"].items() if k!="var"]
                budget["sum"]["var"] = vname
                if budget["var"] is None:
                    budget["var"] = vname
                ds[vname] = sum([v for v in sum_list])
                return ds[vname]
            elif "var" in budget:
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