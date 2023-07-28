import yaml
from os import path

def load_preset_budget(model="MOM6"):
    return load_yaml(f"{path.dirname(__file__)}/conventions/{model}.yaml")
    
def load_yaml(filepath):
    with open(filepath, "r") as stream:
        try:
            budgets_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return budgets_dict