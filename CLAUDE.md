# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`xbudget` wrangles finite-volume budgets (mass, heat, salt) diagnosed from ocean General Circulation Models — primarily MOM6 — into closed budgets using `xarray` and `xgcm`. The library's job is to take a dataset of raw model diagnostics plus a *convention* describing how those diagnostics combine, and materialize every intermediate and aggregate term as a named variable in the dataset.

## Commands

Tests use `pytest` (no separate build/lint step):

```bash
pytest                                          # full suite
pytest xbudget/tests/test_utilities.py          # one file
pytest xbudget/tests/test_utilities.py::TestCollectBudgets::test_collect_budgets_basic   # one test
```

Dev environment (conda + editable install):

```bash
conda env create -f docs/environment.yml   # or ci/environment.yml for the minimal test env
conda activate docs_env_xbudget
pip install -e .
```

CI runs `pytest` across the Python versions in `.github/workflows/ci.yml`. `pyproject.toml` declares `requires-python >= 3.11`; the version is sourced from `xbudget/version.py`.

## Core architecture

The central abstraction is the **`xbudget_dict`** — a nested provenance tree (loaded from a YAML *convention* file) that describes how to build each budget term from raw diagnostics. Understanding its structure is the key to working in this codebase.

### The xbudget_dict tree

Top-level keys are budgets (`mass`, `heat`, `salt`). Each budget has `lhs` and/or `rhs` sub-trees plus metadata keys (`lambda`, `thickness`, `surface_lambda`) that the engine does not interpret. Within a side, terms nest recursively. Every node carries a `var` key (the name of the variable it resolves to, initially `null` in YAML for derived terms) plus optionally one **operation** key:

- `sum` — add the child terms together
- `product` — multiply child terms (scalar numbers are allowed as factors, e.g. `density: 1035.`, `sign: -1.`)
- `difference` — finite-difference a staggered flux across a grid axis (**requires an `xgcm.Grid`, not a bare Dataset**)
- `reciprocal` — safe `1/x` that maps zeros to infinity to avoid div-by-zero

A node may carry more than one operation (e.g. a bulk `product` and an equivalent finer `sum` decomposition of the same quantity). Leaf string values (e.g. `"areacello"`, `"umo"`) are raw diagnostic variable names expected in the input dataset. Convention files live in `xbudget/conventions/*.yaml` (`MOM6.yaml` is the canonical/largest; also `MOM6_3Donly`, `MOM6_drift`, `MOM6_surface`).

### Two modules

- **`presets.py`** — `load_preset_budget(model="MOM6")` and `load_yaml()` deserialize a convention YAML into the dict. No validation of dict structure.
- **`collect.py`** — all the tree-walking logic:
  - `collect_budgets(ds, xbudget_dict)` → top-level entry point. Walks `lhs`/`rhs` of each budget and calls `budget_fill_dict`.
  - `budget_fill_dict(data, xbudget_dict, namepath)` → the recursive workhorse. **Mutates the dataset and the recipe dict in place**: it adds a new variable per node named by its `namepath` (e.g. `heat_rhs_sum_diffusion`) and back-fills each node's `var` key with that name. Each created variable gets a `provenance` attr recording its inputs. `data` may be an `xgcm.Grid` (its `._ds` is used; required for `difference`) or a plain `xr.Dataset`.
  - `aggregate` / `disaggregate` → collapse a fully-filled tree down to flat root-level budgets; `decompose=[...]` keeps named term types broken out into their parts.
  - `get_vars(xbudget_dict, terms)` → query the provenance subtree(s) for given term name(s); reads the `var` fields that `budget_fill_dict` filled in.
  - `deep_search`, `flatten_lol` → internal helpers.

### Key behaviors to know

- **Input and output are the same dict.** `budget_fill_dict` both reads the recipe and writes results back into it (filling `var` fields), and the query helpers (`get_vars`, `aggregate`) depend on those filled fields. Run `collect_budgets` before querying. Tests `copy.deepcopy` the recipe before filling.
- **Missing diagnostics are skipped with a `UserWarning`, not an error** — a term whose variable is absent from `ds` is dropped, and a `sum`/`product` containing missing inputs collapses accordingly. This lets one convention serve datasets with different available diagnostics.
- **Stringly-typed identity.** Variable names are built by concatenating the key path with underscores (`heat_rhs_sum_diffusion_sum_lateral`), and `get_vars` reverse-engineers structure from those strings.
- `budget_fill_dict` mutates both `ds` and the passed `xbudget_dict`; `deep_search`/`_get_vars` are the recursive helpers behind aggregation and querying.

## Data & examples

- `examples/load_example_model_grid.py` — `load_MOM6_coarsened_diagnostics()` downloads (from Zenodo, cached in `data/`) and builds an `xgcm.Grid` with X/Y center/outer coords and `areacello` metrics. This is the standard fixture for realistic end-to-end use.
- `examples/MOM6_budget_examples_mass_heat_salt.ipynb` — worked tutorial notebook.
- The example `.nc` (~600 MB) is gitignored and fetched on demand; only `data/README.md` is tracked.
