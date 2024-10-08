import yaml
from os import path

def load_preset_budget(model="MOM6"):
    """Loads preset xbudget dictionary from yaml file for supported models.

    Note: Does not yet check whether or not the format of the yaml file (or
    the corresponding Python dictionary) is in an xbudget-compatible format!

    Parameters
    ----------
    model : str (default "MOM6")
      Currently supported models: ["MOM6"]
      Please open an Issue if you would like to contribute an xbudget yaml
      file for a new model–see /conventions/ for examples.

    Returns
    -------
    Python dictionary
    """
    return load_yaml(f"{path.dirname(__file__)}/conventions/{model}.yaml")
    
def load_yaml(filepath):
    """Loads a yaml file as a Python dictionary.

    Note: Does not yet check whether or not the format of the yaml file (or
    the corresponding Python dictionary) is in an xbudget-compatible format!

    Parameters
    ----------
    filepath : path to yaml file, as str

    Returns
    -------
    Python dictionary
    """
    with open(filepath, "r") as stream:
        try:
            xbudget_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return xbudget_dict