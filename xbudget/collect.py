from operator import mul
from functools import reduce

def collect_terms(ds, budget_dict, mode="shallow"):
    return flatten_lol(_budget_search(ds, budget_dict, mode=mode))

def _budget_search(ds, d, mode="shallow"):
    if type(d) is float:
            return d
    if mode=="shallow":
        if "total" in d:
            if d["total"] is not None:
                return([d["total"]])
        if "product" in d:
            l = [_budget_search(ds, v, mode=mode) for k,v in d["product"].items()]
            product_str = "product_"+"_x_".join([str(e) for e in l])
            ds[f"{product_str}"] = reduce(mul, [v if type(v) is float else ds[v] for v in l], 1)
            return([product_str])
        elif "sum" in d:
            return([_budget_search(ds, v, mode=mode) for k,v in d["sum"].items()])

        
    elif mode=="deep":
        if "sum" in d:
            return([_budget_search(ds, v, mode=mode) for k,v in d["sum"].items()])
        elif "product" in d:
            l = [_budget_search(ds, v, mode=mode) for k,v in d["product"].items()]
            product_str = "product_"+"_x_".join([str(e) for e in l])
            ds[f"{product_str}"] = reduce(mul, [v if type(v) is float else ds[v] for v in l], 1)
            return([product_str])
        elif "total" in d:
            if d["total"] is not None:
                return([d["total"]])
            
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