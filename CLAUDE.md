# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`xbudget` wrangles finite-volume budgets (mass, heat, salt) diagnosed from ocean General Circulation Models — primarily MOM6 — into closed budgets using `xarray` and `xgcm`. The library's job is to take a dataset of raw model diagnostics plus a *recipe* describing how those diagnostics combine, and materialize every intermediate and aggregate term as a named variable in the dataset.

The engine is a typed expression tree (`parse -> evaluate`). The recipe/YAML format has been stable across that refactor; what changed in 0.7.0 is the in-memory representation, the output variable names, and the fact that the recipe is no longer mutated. 0.7.0 is a **breaking release**: the old dict-walking engine and its recipe-reading helpers were removed outright (not deprecated). `CHANGELOG.md` has the migration guide.

## Commands

Tests use `pytest` (no separate build/lint step). The base conda environment may have a NumPy 1.x/2.x mismatch — run tests in a project env built from `docs/environment.yml` (which now pins xgcm >= 0.10.0; an older env with the pre-0.10 dev xgcm will fail on `padding=`):

```bash
pytest                                          # full suite
pytest xbudget/tests/test_parse.py              # one file
pytest xbudget/tests/test_collect.py::TestCollectBudgets::test_collect_budgets_basic   # one test
```

The end-to-end characterization test needs the ~600 MB example MOM6 dataset (gitignored, fetched from Zenodo); it **skips** when the file is absent. Regenerate the characterization golden after an intended change with `XBUDGET_REGEN_CHARN=1 pytest xbudget/tests/test_characterization.py -s`.

Dev environment (conda + editable install):

```bash
conda env create -f docs/environment.yml   # or ci/environment.yml for the minimal test env
conda activate docs_env_xbudget
pip install -e .
```

## Core architecture

The central abstraction is the **`recipe`** — a nested provenance tree (loaded from a YAML *recipe* file) describing how to build each budget term from raw diagnostics. It is the public input format. Internally it is parsed into a typed expression tree and evaluated.

### The recipe tree (input format — unchanged)

Top-level keys are budgets (`mass`, `heat`, `salt`). Each budget has `lhs` and/or `rhs` sub-trees plus metadata keys (`lambda`, `thickness`, `surface_lambda`) that the engine does not interpret. Within a side, terms nest recursively. A node names a diagnostic with a `var` key (`var: "thetao"`); a derived term omits `var` entirely (the `var: null` placeholders are optional as of 0.7.0) and carries one or more **operation** keys instead:

- `sum` — add the child terms together
- `product` — multiply child terms (scalar numbers allowed as factors, e.g. `density: 1035.`, `sign: -1.`)
- `difference` — finite-difference across a grid axis (**requires an `xgcm.Grid`**); the operand is a raw variable *or* a computed sub-term
- `reciprocal` — safe `1/x` (zeros → inf) of a variable
- `lateral_divergence` — horizontal flux divergence `div(Fx, Fy)` of two flux sub-terms, via native xgcm (`grid.diff` with `other_component` + `face_connections`); works on face-connected LLC grids

A node may carry more than one operation (e.g. a bulk `product` and an equivalent finer `sum`). Leaf string values (`"areacello"`, `"umo"`) are raw diagnostic names. Recipes live in `xbudget/recipes/*.yaml` — `MOM6.yaml` (canonical; also `MOM6_3Donly`, `MOM6_drift`, `MOM6_surface`) and `ECCOV4r4_native.yaml` (LLC90 native-grid budgets).

### The typed engine (parse → evaluate)

```
recipe ──parse_budgets──▶ typed tree (nodes.py) ──evaluate_budgets──▶ derived variables + records
```

- **`nodes.py`** — immutable dataclasses: `Budget`, `Term`, and the operations `Sum`/`Product`/`Difference`/`Reciprocal`/`LateralDivergence` plus `Constant`/`VarRef`. A `Term` carries its structured `path` (its canonical identity) and may hold multiple operations. The native `lateral_divergence` helper lives in `collect.py`.
- **`parse.py`** — `parse_budgets(recipe) -> {name: Budget}`. The single source of schema truth; validates and raises `BudgetParseError` naming the offending path on malformed recipes.
- **`evaluate.py`** — `evaluate_budgets(data, budgets)` walks the tree and materializes **one variable per operation**, named by its term path with operator infixes dropped (e.g. `heat_rhs_diffusion_lateral`). When a term has several operations, the first `sum`/`product` (or a lone op) claims the bare path name; siblings are suffixed with their operator kind. It is pure with respect to the recipe (never mutates it); it only writes derived variables into the dataset. Each variable gets `xbudget_path` (structured identity), `xbudget_op` (the operation kind), and `provenance` (immediate inputs) attrs. Returns `records` (each new variable name → its `{path, op}` metadata). Dispatch is on node type (`Difference` requires an `xgcm.Grid` in its signature, so a grid-less difference fails fast with a clear error).
- **`collect.py`** — the public surface:
  - `collect_budgets(data, recipe, allow_rechunk=True, on_missing="warn")` → parses then evaluates. Uses the simplified names and does **not** mutate the recipe dict. This is the only engine (0.7.0 removed the old dict-walking one).
  - `lateral_divergence(grid, Fx, Fy)` → native-xgcm horizontal flux divergence, used by the evaluator.
- **`query.py`** — `BudgetQuery(data, recipe)`, the query layer: `.var(term)`, `.get_vars(term)`, `.aggregate(decompose=...)`, `.terms()`, plus completeness (`missing`/`is_complete`/`incomplete_terms`) and budget-metadata accessors (`metadata`/`thickness`/`lambda_var`/`surface_lambda`). Built from (data, recipe) rather than a live run, so it works on a reopened dataset. Resolution is a 3-rung rule mirroring `evaluate.py` (primary name → `explicit_var` → operation-suffixed name) and it **checks the dataset** rather than predicting names: which operation owns a term's bare name is a runtime fact (if the primary op is skipped for a missing diagnostic, a sibling claims it), so a recipe-only reading would be wrong, not just optimistic. (Rung 2, `explicit_var`, also resolves a leaf term to its raw diagnostic before `collect_budgets` has run.)
- **`presets.py`** — `load_preset_budget` / `load_yaml` / `save_yaml`. `save_yaml` validates via `parse_budgets` before writing and dumps with `sort_keys=False` (key order drives operand order).

### Key behaviors to know

- **Naming.** One variable per node/operation with operator infixes dropped; no duplicate "copy" variables (108 → 57 on the MOM6 example, 140 → 75 on ECCO). `CHANGELOG.md` has the old→new table for anyone migrating from 0.6.x.
- **Missing diagnostics are skipped, not fatal** — a `sum` drops only the missing operand and builds from the rest (flagged `xbudget_incomplete`); a `product` with a missing *required* factor is dropped entirely (an unknown factor is not a zero one). `collect_budgets(on_missing="warn"|"raise"|"ignore")` sets the policy; `optional: true` on a term exempts its subtree. Query it back with `BudgetQuery.missing()`.
- **`difference` rechunking:** `allow_rechunk=True` (default) temporarily rechunks the difference dimension into a single chunk (required by `grid.diff`) then restores chunking.
- **Lenient parser.** `parse.py` **warns and skips** unavailable-diagnostic placeholders (e.g. a `null`-source `difference`) and stray non-operation keys instead of failing, so real recipes with such terms still load. (This tolerance previously masked the malformed `bolus_mass_flux_convergence` term in `ECCOV4r4_native.yaml` — missing its `product:` wrapper, so the eddy bolus transport was silently dropped from the mass budget; that has since been fixed in the recipe.)
- **xgcm version:** requires **xgcm >= 0.10.0** — `lateral_divergence` needs native face-connected differencing, first shipped there. 0.10.0 also removed `xgcm.Grid`'s `periodic` argument and renamed `boundary` to `padding` (both now raise `ValueError`), so every grid construction in `examples/` and `tests/` uses `padding=`; `periodic=False` translates to `padding="fill"`.

### Tests

- `test_parse.py` — parser units + validation; asserts all shipped recipes parse; covers the tolerated-malformation path.
- `test_characterization.py` (+ `characterization_MOM6.json`) — golden snapshot of the engine's absolute MOM6 output (data-gated; exercises reciprocal/difference-of-sub-term only via the real recipe locally).
- `test_collect.py` — `collect_budgets` behavior (no recipe mutation, lhs/rhs, the grid-guard on `difference`).
- `test_query.py` — the query layer, all on the synthetic grid so it actually runs in CI.
- `test_missing_handling.py` / `test_optional_var.py` / `test_broadcast_warning.py` / `test_display.py` — missing-diagnostic handling, `var: null` optionality, the issue-#11 broadcast guard, and the tree display; all synthetic-grid, CI-safe.
- `conftest.py` — `synthetic_grid` / `synthetic_preset` fixtures, plus `SYNTHETIC_PRESET_SKIPS` (a preset whose structurally-primary op is skipped at run time — the regression test for name resolution being a runtime fact).

**CI has neither example dataset**, so every data-gated test skips there. Only the synthetic-grid tests actually protect the engine in CI; `lateral_divergence` and `reciprocal` on a real face-connected grid are exercised *only* by the data-gated characterization / ECCO notebooks locally.

## Data & examples

- `examples/load_example_model_grid.py` — `load_MOM6_coarsened_diagnostics()` builds a MOM6 `xgcm.Grid` (X/Y center/outer, `areacello` metric). `examples/load_example_ecco_grid.py` — `load_ECCOV4r4_coarsened_diagnostics()` builds the ECCO **LLC90** grid with 13-tile `face_connections`. Both download from Zenodo, cached in `data/` (gitignored; only `data/README.md` tracked).
- Notebooks: `MOM6_budget_examples_mass_heat_salt.ipynb`; `eccov4r4_budget_examples_mass_heat_salt.ipynb` (ECCO closure); `eccov4r4_heat_budget_decomposition.ipynb` (ECCO heat decomposition); `handling_missing_diagnostics.ipynb`. All use `collect_budgets` + `BudgetQuery`.
- Re-executing a notebook needs a kernel whose env has xgcm >= 0.10.0 and the Zenodo data. Their `kernelspec` is `name: python3`, which nbconvert resolves to the *default* python3 kernel — not necessarily the project env — so pass `--ExecutePreprocessor.kernel_name=<your-registered-kernel>` explicitly or it fails on `import xgcm`.
- Docs (`docs/source/`) are Sphinx with `myst_parser` (for `.md`) + `nbsphinx` (notebooks copied in from `examples/` at build time by a hook in `conf.py`). `quickstart.md` and `recipes.md` are hand-written. Build locally with `cd docs && make html`, or exactly as CI does: `python -m sphinx -b html -W --keep-going docs/source docs/build/html`. `-W` mirrors `.readthedocs.yaml`'s `fail_on_warning: true` — a broken cross-reference fails the build.
- Docs are built three ways, deliberately: the `docs` job in `ci.yml` (pre-merge gate, uploads the rendered HTML as a `docs-html` artifact), Read the Docs PR builds (the hosted preview link on the PR), and RTD `latest` on merge. If you change the Sphinx config, keep the CI command and `.readthedocs.yaml` in agreement.

## Pull request workflow

When you push a new commit to a branch that already has an open pull request, update the PR description (the top comment / body) so it stays consistent with the latest commit — don't leave it describing only the original state:

- Refresh the summary so it reflects what the branch does now.
- If the description contains a task list / checklist, check off (`- [x]`) the items the new commit completed and add entries for any follow-up work it introduced.
- Reflect scope, naming, or API changes so a reviewer reading only the PR body sees the current truth.

Update it with the GitHub CLI as part of the same push, e.g. `gh pr edit <number> --body-file <path>` (or `--body "..."`), so the description never lags behind the commits.
