# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`xbudget` wrangles finite-volume budgets (mass, heat, salt) diagnosed from ocean General Circulation Models — primarily MOM6 — into closed budgets using `xarray` and `xgcm`. The library's job is to take a dataset of raw model diagnostics plus a *convention* describing how those diagnostics combine, and materialize every intermediate and aggregate term as a named variable in the dataset.

> This branch refactors the engine internals. The convention/YAML format is unchanged, but the in-memory representation is now a typed expression tree and the default output variable names are simplified. See `CHANGELOG.md` for the migration guide.

## Commands

Tests use `pytest` (no separate build/lint step). The base conda environment may have a NumPy 1.x/2.x mismatch — run tests in the project env (e.g. `docs_env_xbudget`):

```bash
pytest                                          # full suite
pytest xbudget/tests/test_parse.py              # one file
pytest xbudget/tests/test_utilities.py::TestCollectBudgets::test_collect_budgets_basic   # one test
```

The end-to-end characterization and engine-equivalence tests need the ~600 MB example MOM6 dataset (gitignored, fetched from Zenodo); they **skip** when it is absent. Regenerate the characterization golden after an intended change with `XBUDGET_REGEN_CHARN=1 pytest xbudget/tests/test_characterization.py -s`.

Dev environment (conda + editable install):

```bash
conda env create -f docs/environment.yml   # or ci/environment.yml for the minimal test env
conda activate docs_env_xbudget
pip install -e .
```

## Core architecture

The central abstraction is the **`xbudget_dict`** — a nested provenance tree (loaded from a YAML *convention* file) describing how to build each budget term from raw diagnostics. It is the public input format. Internally it is parsed into a typed expression tree and evaluated.

### The xbudget_dict tree (input format — unchanged)

Top-level keys are budgets (`mass`, `heat`, `salt`). Each budget has `lhs` and/or `rhs` sub-trees plus metadata keys (`lambda`, `thickness`, `surface_lambda`) that the engine does not interpret. Within a side, terms nest recursively. Every node carries a `var` key (a variable name, or `null` for derived terms) plus optionally one or more **operation** keys:

- `sum` — add the child terms together
- `product` — multiply child terms (scalar numbers allowed as factors, e.g. `density: 1035.`, `sign: -1.`)
- `difference` — finite-difference across a grid axis (**requires an `xgcm.Grid`**); the operand is a raw variable *or* a computed sub-term
- `reciprocal` — safe `1/x` (zeros → inf) of a variable
- `lateral_divergence` — horizontal flux divergence `div(Fx, Fy)` of two flux sub-terms, via native xgcm (`grid.diff` with `other_component` + `face_connections`); works on face-connected LLC grids

A node may carry more than one operation (e.g. a bulk `product` and an equivalent finer `sum`). Leaf string values (`"areacello"`, `"umo"`) are raw diagnostic names. Conventions live in `xbudget/conventions/*.yaml` — `MOM6.yaml` (canonical; also `MOM6_3Donly`, `MOM6_drift`, `MOM6_surface`) and `ECCOV4r4_native.yaml` (LLC90 native-grid budgets).

### The typed engine (parse → evaluate)

```
xbudget_dict ──parse_budgets──▶ typed tree (nodes.py) ──evaluate_budgets──▶ derived variables + alias map
```

- **`nodes.py`** — immutable dataclasses: `Budget`, `Term`, and the operations `Sum`/`Product`/`Difference`/`Reciprocal`/`LateralDivergence` plus `Constant`/`VarRef`. A `Term` carries its structured `path` (its canonical identity) and may hold multiple operations. The native `lateral_divergence` helper lives in `collect.py` and is shared by both engines.
- **`parse.py`** — `parse_budgets(dict) -> {name: Budget}`. The single source of schema truth; validates and raises `BudgetParseError` naming the offending path on malformed conventions.
- **`evaluate.py`** — `evaluate_budgets(data, budgets)` walks the tree and materializes **one variable per operation**, named by its term path with operator infixes dropped (e.g. `heat_rhs_diffusion_lateral`). It is pure with respect to the recipe (never mutates it); it only writes derived variables into the dataset. Each variable gets `xbudget_path` (structured identity), `xbudget_op` (the operation kind), and `provenance` (immediate inputs) attrs. Returns `(alias_map, records)` — `alias_map` maps every legacy name to its new name; `records` maps each new variable to its metadata. Dispatch is on node type (`Difference` requires an `xgcm.Grid` in its signature, so a grid-less difference fails fast with a clear error).
- **`collect.py`** — the public surface:
  - `collect_budgets(data, xbudget_dict, allow_rechunk=True, name_scheme="v1")` → parses then evaluates. **`v1` (default)** uses the simplified names and does **not** mutate the recipe dict. **`legacy`** reuses `budget_fill_dict` to reproduce the historical operator-suffixed names *and* fill the recipe dict in place.
  - `budget_fill_dict(...)` → the legacy dict-walking engine, retained as a reference implementation (pinned by the equivalence test) and used by `name_scheme="legacy"`. It mutates both the dataset and the recipe dict.
  - `aggregate` / `disaggregate` / `get_vars` → dict-based query helpers. **They read the `var` fields that the legacy engine fills**, so they only work after a `name_scheme="legacy"` run. For `v1`, query via the `records`/`alias_map` from `evaluate_budgets` and the `provenance`/`xbudget_path` attrs.

### Key behaviors to know

- **Naming changed (major-version cleanup).** `v1` emits one variable per node/operation with operator infixes dropped; the legacy engine emitted duplicate "copy" variables (108 → 57 on the MOM6 example). Use `name_scheme="legacy"` or the `alias_map` to bridge. `CHANGELOG.md` has the old→new table.
- **Missing diagnostics are skipped with a `UserWarning`, not an error** — a `sum`/`product` containing missing inputs collapses accordingly, so one convention can serve datasets with different available diagnostics.
- **`difference` rechunking:** `allow_rechunk=True` (default) temporarily rechunks the difference dimension into a single chunk (required by `grid.diff`) then restores chunking.
- **Lenient parser.** `parse.py` mirrors the legacy engine: it **warns and skips** unavailable-diagnostic placeholders (e.g. a `null`-source `difference`) and stray non-operation keys instead of failing, so real conventions with such terms still load. (This tolerance previously masked the malformed `bolus_mass_flux_convergence` term in `ECCOV4r4_native.yaml` — missing its `product:` wrapper, so the eddy bolus transport was silently dropped from the mass budget; that has since been fixed in the convention.)
- **xgcm version:** `lateral_divergence` needs native face-connected differencing, available only in xgcm **after 0.9.0** (currently the dev `main`). Run/test in an env with that xgcm.

### Tests

- `test_parse.py` — parser units + validation; asserts all shipped conventions parse; covers the tolerated-malformation path.
- `test_evaluate_equivalence.py` — proves the typed engine is numerically identical to the legacy `budget_fill_dict`: a synthetic grid (always), the MOM6 grid, and the **ECCO LLC90 grid** (both gated on their data files; the ECCO case exercises reciprocal, difference-of-sub-term, and native `lateral_divergence`).
- `test_characterization.py` (+ `characterization_MOM6.json`) — golden snapshot of the typed engine's absolute MOM6 output.
- `test_utilities.py` — the legacy engine, `aggregate`/`get_vars`/`disaggregate`, and `collect_budgets` behavior.

## Data & examples

- `examples/load_example_model_grid.py` — `load_MOM6_coarsened_diagnostics()` builds a MOM6 `xgcm.Grid` (X/Y center/outer, `areacello` metric). `examples/load_example_ecco_grid.py` — `load_ECCOV4r4_coarsened_diagnostics()` builds the ECCO **LLC90** grid with 13-tile `face_connections`. Both download from Zenodo, cached in `data/` (gitignored; only `data/README.md` tracked).
- Notebooks: `MOM6_budget_examples_mass_heat_salt.ipynb`; `eccov4r4_budget_examples_mass_heat_salt.ipynb` (ECCO closure); `eccov4r4_heat_budget_decomposition.ipynb` (ECCO heat decomposition). The ECCO notebooks and the MOM6 one call `collect_budgets(..., name_scheme="legacy")` because they use `get_vars`/`aggregate`.

## Pull request workflow

When you push a new commit to a branch that already has an open pull request, update the PR description (the top comment / body) so it stays consistent with the latest commit — don't leave it describing only the original state:

- Refresh the summary so it reflects what the branch does now.
- If the description contains a task list / checklist, check off (`- [x]`) the items the new commit completed and add entries for any follow-up work it introduced.
- Reflect scope, naming, or API changes so a reviewer reading only the PR body sees the current truth.

Update it with the GitHub CLI as part of the same push, e.g. `gh pr edit <number> --body-file <path>` (or `--body "..."`), so the description never lags behind the commits.
