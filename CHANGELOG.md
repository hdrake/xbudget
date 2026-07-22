# Changelog

## 0.7.0 — typed engine + new query API (breaking)

This release replaces the recursive dict-walking engine with a typed expression
tree (parse → evaluate) and a new query layer (`BudgetQuery`). The recipe/YAML
format is **unchanged**; the in-memory representation, the engine, the output
variable names, and the query API are new.

**This is a breaking release.** The old dict-walking engine and the
recipe-reading helpers (`budget_fill_dict`, `aggregate`, `get_vars`,
`disaggregate`, `deep_search`) have been **removed**, not deprecated. There is
one engine and one way to query it. If you are upgrading from 0.6.x, follow the
migration below.

### Migration

Variable names changed, and the helpers that looked them up are replaced by
`BudgetQuery`:

```python
xbudget.collect_budgets(grid, recipe)          # same call, typed engine
q = xbudget.BudgetQuery(grid, recipe)          # new: query what it produced

q.var("heat_lhs_advection")            # was: get_vars(d, "heat_lhs_sum_advection")["var"]
q.get_vars("heat_lhs_advection")       # was: get_vars(d, "heat_lhs_sum_advection_sum")
q.aggregate()                          # was: xbudget.aggregate(d)
q.aggregate(decompose=["diffusion"])   # was: xbudget.aggregate(d, decompose=["diffusion"])
```

| Removed | Replacement |
|---|---|
| `get_vars(d, term)["var"]` | `q.var(term)` |
| `get_vars(d, term)` | `q.get_vars(term)` |
| `aggregate(d, decompose=...)` | `q.aggregate(decompose=...)` |
| `disaggregate`, `deep_search` | `q.aggregate` / `q.get_vars` |
| `budget_fill_dict` | `collect_budgets` |
| `xbudget_dict=` keyword argument | `recipe=` |
| `flatten`, `flatten_lol` | *(removed; no replacement)* |

`collect_budgets`'s second parameter is now named `recipe` (was `xbudget_dict`).

### Breaking changes

1. **Simplified variable names.** Derived variables are named by their term path
   with the `sum`/`product`/`difference` operator infixes dropped, and the
   redundant "copy" duplicates the old engine emitted are gone. One variable is
   produced per operation.

   | Old (0.6.x) name | New name |
   |---|---|
   | `heat_rhs` | `heat_rhs` *(unchanged)* |
   | `heat_rhs_sum` | `heat_rhs` *(the copy/sum collapse into one)* |
   | `heat_rhs_sum_diffusion` | `heat_rhs_diffusion` |
   | `heat_rhs_sum_diffusion_sum_lateral_product` | `heat_rhs_diffusion_lateral` |
   | `mass_rhs_sum_advection_sum_lateral_sum_zonal_convergence_product_zonal_divergence_difference` | `mass_rhs_advection_lateral_zonal_convergence_zonal_divergence` |

   On the example MOM6 grid this reduces 108 variables to 57; the ECCOv4r4 LLC90
   native-grid recipe materializes 73 (Eulerian volume budget — the GM bolus is
   exposed as metadata rather than a budget term; see the ECCO note below). The
   canonical identity of each variable is also stored structurally in its
   `xbudget_path` attribute (a list of term names), so you never need to parse
   the flat name.

2. **`collect_budgets` no longer mutates the recipe dict.** It previously filled
   each node's `var` field in place; it now leaves `recipe` untouched and returns
   the data object. Query the output with `BudgetQuery`. A welcome side effect: a
   recipe now round-trips through `save_yaml`/`load_yaml` unchanged.

3. **`collect_budgets` first parameter is named `data`** (a grid or dataset),
   matching its long-standing behavior of accepting either.

4. **A `product` with a missing required factor is now dropped, not fabricated
   as zero.** The old engine multiplied a missing factor in as `0.0`, emitting an
   identically-zero variable that looked like a real (null) contribution to
   whatever consumed it — the classic "the term is there when it isn't." An
   unknown factor is not a zero one, so the typed engine no longer builds the
   term at all; `BudgetQuery.var()` returns `None` for it and it is reported by
   `missing()`. A `sum` is unaffected — it still drops only the missing operand
   and builds from the rest (now flagged incomplete). This changes results only
   where a product actually referenced an absent diagnostic; a literal `0.`
   *constant* in a recipe (e.g. `lambda_mass: 0.` in the salt budget) is a real
   factor and is unchanged.

### New

- **`var: null` placeholders are no longer needed.** Write `var` only when you
  are naming a diagnostic (`var: "thetao"`); a derived term says nothing at all.
  This was always true of the parser — a missing `var` and an explicit
  `var: null` have always read the same — but every shipped recipe was
  written the verbose way, so it read as mandatory. The placeholders are now
  gone from all five shipped recipes (412 lines; `MOM6.yaml` 349 → 241),
  with the parse trees verified byte-identical before and after.

  ```yaml
  # before                      # now
  forcing:                      forcing:
    var: null                     product:
    product:                        flux: "hfds"
      var: null                     area: "areacello"
      flux: "hfds"
      area: "areacello"
  ```

  Existing recipes keep working untouched — this removes a requirement, it
  does not add one. `var: null` is still meaningful in the two places where it
  is a node's *sole* key, and so is what makes the node a node: a term that
  declares a name but no operations (the `MOM6_drift` skeleton), and an
  otherwise-empty budget side (`lhs:` in `MOM6_surface`).
- **`xbudget.BudgetQuery(data, recipe)`** — find what a run produced:
  `.var(term)`, `.get_vars(term)`, `.aggregate(decompose=...)`, `.terms()`,
  plus the budget-metadata accessors `.metadata(budget=None)`,
  `.thickness(budget="mass")`, `.lambda_var(budget)` and `.surface_lambda(budget)`
  (the recipe's declared state variables — most notably the mass budget's layer
  thickness, which `xwmt.WaterMass` needs). It is built from the data and the
  recipe rather than from a live engine run, so it also works on a dataset
  reopened from disk.

  Terms are addressed by their name (`"heat_rhs_diffusion"`), their path
  (`("heat", "rhs", "diffusion")`), or a raw diagnostic name (which reports the
  terms that reference it). An unknown name raises a `KeyError` with a
  close-match suggestion rather than silently missing.
- **`xbudget.show_recipe(recipe)`** — render a recipe as a collapsible tree: an
  interactive HTML tree in a notebook, an indented ASCII tree at a terminal.
  `BudgetQuery` gains the matching `_repr_html_`, annotated with each term's
  resolved variable name and with unmaterialized terms greyed out.
- **`xbudget.save_yaml(recipe, filepath)`** — write a recipe to YAML.
  It validates first (so a malformed recipe cannot reach disk) and preserves key
  order (which is meaningful: it drives operand order). `load_yaml` now re-raises
  YAML errors instead of failing with an unrelated `UnboundLocalError`.
- `xbudget.parse_budgets(recipe)` → typed tree (`xbudget.nodes.Budget`),
  the single schema-validating entry point; raises `xbudget.BudgetParseError`
  with the offending path on malformed recipes.
- `xbudget.evaluate_budgets(data, budgets)` → pure evaluator; returns a
  `records` map (each new variable name → its `{path, op}` metadata).
- Each derived variable carries `xbudget_path` (structured identity),
  `xbudget_op` (operation kind), and `provenance` (immediate inputs) attributes.
- **Missing diagnostics can no longer mislead you into thinking a term is there.**
  A term built from fewer inputs than its recipe describes now says so, durably
  and queryably:
  - Affected variables are stamped `xbudget_incomplete: 1` and `xbudget_missing:
    [...]` (the inputs that were expected but absent). These persist to disk, so
    the record survives a save/reopen — where a warning would not.
  - `BudgetQuery` reads them back: `is_complete(term)`, `missing(term=None)`
    (what each term was built without), `incomplete_terms()` (every flagged
    variable), and a `"missing"` key on `get_vars`. Incompleteness propagates, so
    an ancestor of a dropped term is flagged too. On a recipe-only query
    (`data=None`) completeness is reported as unknown rather than guessed.
  - `collect_budgets(..., on_missing=...)` selects the policy: `"warn"` (default)
    emits **one** end-of-run summary naming the missing diagnostics and the
    now-incomplete terms (instead of one warning per operand); `"raise"` raises
    the new `xbudget.MissingDiagnosticError` (its `.missing` lists every gap);
    `"ignore"` is silent but still stamps the attributes.
  - A recipe term may declare `optional: true` to mark a diagnostic as *expected*
    to be absent on some datasets. Its whole subtree is then exempt — dropped
    with no warning, no `raise`, and no `incomplete` flag — which documents the
    intent in the recipe instead of deleting the term to quiet the warning.
- **ECCOv4r4 / LLC90 support in the typed engine.** The `reciprocal` and
  `lateral_divergence` operations and a `difference` of a *computed sub-term*
  (not just a raw variable) are all handled, so the native-grid ECCO mass/heat/
  salt budgets evaluate. New `ECCOV4r4_native` recipe and example notebooks
  (`eccov4r4_budget_examples_mass_heat_salt`, `eccov4r4_heat_budget_decomposition`).
  The example loader (`examples/load_example_ecco_grid.py`) assembles the budget
  from the per-collection ECCO V4r4 (2010, LLC90) files on Zenodo
  (DOI 10.5281/zenodo.21479854) — monthly-mean fluxes, month-boundary snapshots,
  grid geometry, and geothermal input — which carry every diagnostic the recipe
  needs. All three budgets close (mass/heat/salt residuals ~3e-4 / 5e-6 / 6e-4 of
  the tendency, pointwise, over the full year).
- **The ECCO volume budget is Eulerian-only, and the GM bolus is exposed as
  metadata for water-mass transformation.** The bolus (GM eddy) velocity is
  non-divergent, so it carries *zero net volume* and does not belong in the
  native volume budget — including it only injects discretization noise (the
  archived `U/V/WVELSTAR` are non-divergent to ~6% under xgcm's operators, and
  that residual accumulates down the column into a spurious convergence larger
  than the true mass tendency). ECCO's own budget tutorials build the volume
  budget from `UVELMASS`/`VVELMASS`/`WVELMASS` alone, and the heat/salt fluxes
  (`ADV*_TH`/`ADV*_SLT`) are already residual-mean (bolus included), so the eddy
  advection is *already* in those budgets. The bolus is not discarded: a new
  `bolus` metadata block on the mass budget names its mass transports
  (`bolus_{x,y,z}_mass_transport`), read via `BudgetQuery.bolus_transports()`,
  because the bolus transport across isopycnals *is* a real transformation term
  in density coordinates (`xwmb` consumes it).
- **`lateral_divergence` now uses native xgcm** (`grid.diff` with
  `other_component` + `face_connections`) instead of a hand-rolled LLC90 flux
  stitcher, so it is correct on any face-connected topology xgcm supports rather
  than LLC90 specifically. Verified bit-identical (zero differing elements) to
  the old stitcher under xgcm 0.10.0, on the full 13-tile ECCO grid, for the
  Eulerian mass-flux fields. The `xbudget/llc90` module is removed.
- **Broadcast guard (issue #11).** When a `sum` mixes a 2D surface flux with a 3D
  flux convergence, xarray silently broadcasts the 2D operand across the vertical.
  The engine now emits a `UserWarning` naming the term and the broadcast
  dimensions ahead of the sum.

### Fixed

- The `difference` operation now raises a clear `ValueError` up front when no
  grid is supplied, instead of an opaque `NameError`.

### Removed

- **The dict-walking engine and its recipe-reading helpers.** `budget_fill_dict`,
  `aggregate`, `get_vars`, `disaggregate`, `deep_search`, and the `flatten` /
  `flatten_lol` utilities are gone. Use `collect_budgets` + `BudgetQuery`.
- **The `xbudget_dict=` keyword argument** of `collect_budgets` — its second
  parameter is now named `recipe`. (The new `parse_budgets`, `save_yaml`, and
  `BudgetQuery` take `recipe` as well.)
- The `xbudget/llc90` module (`diff_2d_flux_llc90`), the hand-rolled LLC90 flux
  stitcher. `lateral_divergence` now uses native xgcm for any face-connected
  grid; the two were verified bit-identical on the ECCO grid before it went.

### Renamed

- **"convention" → "recipe"** throughout. The nested dict describing how a model
  builds its budgets is now called a *recipe* everywhere — prose, docstrings, and
  the public argument name. Concretely:
  - the `xbudget/conventions/` directory of shipped YAMLs → `xbudget/recipes/`
    (`load_preset_budget` is unchanged; it reads from the new path);
  - `collect_budgets`'s recipe argument is renamed `xbudget_dict` → `recipe`
    (the new `parse_budgets`, `save_yaml`, and `BudgetQuery` take `recipe` too).

  ("Convention" in the sense of a sign/ordering choice is untouched.)

### Dependencies

- **xgcm >= 0.10.0 is now required** (was >= 0.9.0). The LLC `lateral_divergence`
  relies on native face-connected differencing (`grid.diff` with
  `other_component`), which 0.10.0 is the first release to ship correctly.
- xgcm 0.10.0 removed the `periodic` argument to `xgcm.Grid` and renamed
  `boundary` to `padding`. This does not affect the xbudget API, but it does
  affect **your** grid construction: replace `boundary=` with `padding=`, and
  `periodic=False` with `padding="fill"` (`periodic=True` with
  `padding="periodic"`). The example grid loaders in `examples/` show the new
  form.

### Parser tolerance

- The parser **warns and skips** unavailable-diagnostic placeholders (e.g. a
  `difference` whose source is `null`) and terms with stray non-operation keys,
  rather than failing, so real recipes with such placeholders still load. (This
  same tolerance is what let the malformed bolus term above pass silently before
  it was fixed — the warning it emitted is what surfaced the bug.)
