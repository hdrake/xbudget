from operator import mul
from functools import reduce

def collect_budgets(ds, budgets_dict, mode="shallow"):
    for eq, v in budgets_dict.items():
        for part in ["rhs", "lhs", "surface_advective_ocean_flux"]:
            if part in v:
                budget_search(ds, v[part], f"{eq}_{part}", mode=mode)
                
def budget_search(ds, d, namepath, mode="shallow"):
    if type(d) is float:
        return d
    
    if mode=="shallow":
        if "var" in d:
            if d["var"] is not None:
                return([d["var"]])
        if "product" in d:
            vname = f"{namepath}_product"
            d["product"]["var"] = vname
            if d["var"] is None:
                d["var"] = vname
            l = [budget_search(ds, v, f"{vname}", mode=mode) for k,v in d["product"].items() if k !="var"]
            ds[vname] = reduce(mul, [e if type(e) is float else ds[e] for e in l if e !="var"], 1)
            return([vname])
        if "sum" in d:
            vname = f"{namepath}_sum"
            d["sum"]["var"] = vname
            if d["var"] is None:
                d["var"] = vname
            sum_list = flatten_lol([budget_search(ds, v, f"{namepath}_{k}", mode=mode) for k,v in d["sum"].items() if k !="var"])
            ds[vname] = sum([ds[v] for v in sum_list])
            return(sum_list)
        
    elif mode=="deep":
        if "sum" in d:
            vname = f"{namepath}_sum"
            d["sum"]["var"] = vname
            if d["var"] is None:
                d["var"] = vname
            sum_list = flatten_lol([budget_search(ds, v, f"{namepath}_{k}", mode=mode) for k,v in d["sum"].items() if k !="var"])
            ds[vname] = sum([ds[v] for v in sum_list])
            return(sum_list)
        if "product" in d:
            vname = f"{namepath}_product"
            d["product"]["var"] = vname
            if d["var"] is None:
                d["var"] = vname
            l = [budget_search(ds, v, f"{vname}", mode=mode) for k,v in d["product"].items() if k !="var"]
            ds[vname] = reduce(mul, [e if type(e) is float else ds[e] for e in l if e !="var"], 1)
            return([vname])
        
    if type(d) is str:
        return d
    else:
        raise ValueError("Can not find term.")

def flatten(container):
    for i in container:
        if isinstance(i, (list,tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i
            
def flatten_lol(lol):
    return list(flatten(lol))