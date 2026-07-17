# Changelog

## 0.7.0 — internals refactor (typed engine)

This release replaces the recursive dict-walking engine with a typed expression
tree (parse → evaluate). The convention/YAML format is **unchanged**; the
in-memory representation, the engine, and the default output variable names are
new. Numerical results are identical to the previous engine — verified by
end-to-end equivalence tests on the example MOM6 grid (108 → 57 variables) and
the ECCOv4r4 LLC90 grid (140 → 75 variables, 0 mismatches).

### Quick migration

Variable names changed, and the helpers that looked them up are replaced by
`BudgetQuery`:

```python
xbudget.collect_budgets(grid, xbudget_dict)          # unchanged call
q = xbudget.BudgetQuery(grid, xbudget_dict)          # new

q.var("heat_lhs_advection")            # was: get_vars(d, "heat_lhs_sum_advection")["var"]
q.get_vars("heat_lhs_advection")       # was: get_vars(d, "heat_lhs_sum_advection_sum")
q.aggregate()                          # was: xbudget.aggregate(d)
q.aggregate(decompose=["diffusion"])   # was: xbudget.aggregate(d, decompose=["diffusion"])
```

| Deprecated (removed in v1.0) | Replacement |
|---|---|
| `get_vars(d, term)["var"]` | `q.var(term)` |
| `get_vars(d, term)` | `q.get_vars(term)` |
| `aggregate(d, decompose=...)` | `q.aggregate(decompose=...)` |
| `disaggregate`, `deep_search` | `q.aggregate` / `q.get_vars` |
| `collect_budgets(..., name_scheme="legacy")` | the default `name_scheme="v1"` |
| `budget_fill_dict` | `collect_budgets` |

If you need the old behavior *right now*, `name_scheme="legacy"` still
reproduces it exactly — historical variable names *and* the in-place filling of
the recipe dict that `get_vars`/`aggregate` depend on:

```python
xbudget.collect_budgets(grid, xbudget_dict, name_scheme="legacy")
```

It now emits a `FutureWarning`: both the `"legacy"` value and the `name_scheme`
argument itself are going away in v1.0, along with the helpers above. Adopt the
new scheme at your own pace, but do adopt it.

### Breaking changes

1. **Simplified variable names (default `name_scheme="v1"`).** Derived
   variables are now named by their term path with the `sum`/`product`/
   `difference` operator infixes dropped, and the redundant "copy" duplicates
   the old engine emitted are gone. One variable is produced per operation.

   | Legacy name | New name |
   |---|---|
   | `heat_rhs` | `heat_rhs` *(unchanged)* |
   | `heat_rhs_sum` | `heat_rhs` *(the copy/sum collapse into one)* |
   | `heat_rhs_sum_diffusion` | `heat_rhs_diffusion` |
   | `heat_rhs_sum_diffusion_sum_lateral_product` | `heat_rhs_diffusion_lateral` |
   | `mass_rhs_sum_advection_sum_lateral_sum_zonal_convergence_product_zonal_divergence_difference` | `mass_rhs_advection_lateral_zonal_convergence_zonal_divergence` |

   On the example MOM6 grid this reduces 108 variables to 57. The canonical
   identity of each variable is also stored structurally in its
   `xbudget_path` attribute (a list of term names), so you never need to parse
   the flat name.

2. **`collect_budgets` no longer mutates the recipe dict** (in `v1` mode). It
   previously filled each node's `var` field in place; it now leaves
   `xbudget_dict` untouched and returns the data object. Because the legacy
   `get_vars`/`aggregate` helpers read those filled `var` fields, they only work
   after a `name_scheme="legacy"` run (which still fills the dict). Use
   `BudgetQuery` (below) to query the `v1` output.

   A welcome side effect: a recipe now round-trips through `save_yaml`/`load_yaml`
   unchanged, because running the engine no longer rewrites it.

3. **`collect_budgets` signature** gained a `name_scheme` keyword and its first
   parameter is named `data` (a grid or dataset), matching its long-standing
   behavior of accepting either.

### New

- **`var: null` placeholders are no longer needed.** Write `var` only when you
  are naming a diagnostic (`var: "thetao"`); a derived term says nothing at all.
  This was always true of the parser — a missing `var` and an explicit
  `var: null` have always read the same — but every shipped convention was
  written the verbose way, so it read as mandatory. The placeholders are now
  gone from all five shipped conventions (412 lines; `MOM6.yaml` 352 → 244),
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

  Existing conventions keep working untouched — this removes a requirement, it
  does not add one. The one place the placeholders are still meaningful is a
  term that declares a name but no operations (the `MOM6_drift` skeleton), where
  `var: null` is what makes the node a node.
- **`xbudget.BudgetQuery(data, xbudget_dict)`** — the v1 way to find what a run
  produced: `.var(term)`, `.get_vars(term)`, `.aggregate(decompose=...)`,
  `.terms()`, `.alias_map`. It is built from the data and the recipe rather than
  from a live engine run, so it also works on a dataset reopened from disk, and
  it resolves a legacy-filled recipe correctly too.

  Terms are addressed by their v1 name (`"heat_rhs_diffusion"`), their path
  (`("heat", "rhs", "diffusion")`), or a raw diagnostic name (which reports the
  terms that reference it). A legacy operator-infixed name raises a `KeyError`
  naming its v1 equivalent, rather than silently missing.
- **`xbudget.save_yaml(xbudget_dict, filepath)`** — write a convention to YAML.
  It validates first (so a malformed recipe cannot reach disk) and preserves key
  order (which is meaningful: it drives operand order). `load_yaml` now re-raises
  YAML errors instead of failing with an unrelated `UnboundLocalError`.
- `xbudget.parse_budgets(xbudget_dict)` → typed tree (`xbudget.nodes.Budget`),
  the single schema-validating entry point; raises `xbudget.BudgetParseError`
  with the offending path on malformed conventions.
- `xbudget.evaluate_budgets(data, budgets)` → pure evaluator; returns
  `(alias_map, records)` where `alias_map` maps every legacy name to its new
  name and `records` maps each new variable to its `{path, op, ...}` metadata.
- Each derived variable carries `xbudget_path` (structured identity),
  `xbudget_op` (operation kind), and `provenance` (immediate inputs) attributes.
- **ECCOv4r4 / LLC90 support in the typed engine.** The `reciprocal` and
  `lateral_divergence` operations and a `difference` of a *computed sub-term*
  (not just a raw variable) are all handled, so the native-grid ECCO mass/heat/
  salt budgets evaluate under `name_scheme="v1"`. New `ECCOV4r4_native`
  convention and example notebooks (`eccov4r4_budget_examples_mass_heat_salt`,
  `eccov4r4_heat_budget_decomposition`).
- **`lateral_divergence` now uses native xgcm** (`grid.diff` with
  `other_component` + `face_connections`) instead of a hand-rolled LLC90 flux
  stitcher, so it is correct on any face-connected topology xgcm supports rather
  than LLC90 specifically. Verified bit-identical (zero differing elements) to
  the old stitcher under xgcm 0.10.0, on the full 13-tile ECCO grid, for both the
  Eulerian and bolus mass-flux pairs. The `xbudget/llc90` module is removed.

### Fixed

- **ECCO mass budget: the lateral eddy-bolus transport was silently dropped.**
  The `bolus_mass_flux_convergence` term in `ECCOV4r4_native.yaml` was missing
  its enclosing `product:` wrapper, so its `sign`/`density`/`volume_flux_divergence`
  children sat directly on the term and were ignored — the GM bolus velocity
  (`UVELSTAR`/`VVELSTAR`) contributed nothing to the mass budget. The wrapper is
  now restored, so the bolus convergence is materialized and included. **This
  changes ECCO mass-budget results** (the bolus term is no longer zero).
- **`aggregate`/`get_vars` no longer return a silently empty answer** when the
  recipe was never filled in. `collect_budgets(grid, d)` followed by
  `aggregate(d)` — the default `v1` engine plus the legacy helper — used to hand
  back `{}` for every derived term, with nothing to indicate why. It now raises
  a `ValueError` naming `BudgetQuery`. (A convention with nothing to fill, like
  the `MOM6_drift` skeleton, is unaffected.) **If you use xbudget from xwmt or
  xwmb, this is the error you will hit** — see the migration table above.
- The legacy engine no longer requires `var` keys to be present: it reads them
  with `.get` and adds them as it fills, so it works on placeholder-free
  conventions too. (This also fixes a latent "dictionary changed size during
  iteration" and a `KeyError` in `disaggregate`, both of which only stayed
  hidden because `var: null` was always there to be read.)
- The `difference` operation's grid guard was misattached, so a `difference`
  on a plain `Dataset` raised an opaque `NameError`, and a `difference` term
  evaluated after another operation in the same node raised spuriously even
  with a valid grid. It now raises a clear `ValueError` up front when no grid
  is supplied. (Also fixes a mutable-default-argument footgun in the internal
  search helper.)

### Deprecated

All of the following emit a `FutureWarning` and are **removed in v1.0**. See the
migration table at the top.

- **`name_scheme="legacy"`**, and the `name_scheme` argument itself — v1 naming
  becomes the only behavior.
- **`get_vars`, `aggregate`, `disaggregate`, `deep_search`** — they read the
  `var` fields that only a legacy run fills, so they cannot serve the default
  engine. Replaced by `BudgetQuery`. They are left behaving exactly as before
  (rather than re-pointed at v1) on purpose: a legacy-filled recipe is
  indistinguishable from a clean one, so re-pointing them would have returned v1
  names for a dataset holding legacy names — a silent wrong answer instead of a
  loud rename.
- **`budget_fill_dict`** — the legacy reference engine, still used internally by
  `name_scheme="legacy"` and as the equivalence-test oracle; superseded by
  `collect_budgets` / `evaluate_budgets`.

`BudgetQuery.aggregate` is a faithful replacement for `aggregate`, with six
deliberate differences:

1. A leaf term reports its path name (`heat_rhs_advection`) rather than the raw
   diagnostic (`advective_tendency`). Same array — but now one carrying
   `xbudget_path`/`provenance` attributes.
2. Values have no `_sum`/`_product` infixes (the v1 naming change).
3. Terms that were not materialized (a missing diagnostic) are dropped. Legacy
   kept unmaterialized *leaves*, so `grid._ds[v]` would then raise `KeyError`.
4. `decompose` matches term names exactly. Legacy tested `k not in decompose`
   against a bare string, so `decompose="advection"` also matched a term named
   `"adv"`.
5. Decomposing a leaf no longer emits a spurious `"<name>_var"` key.
6. A side with no `sum` returns `{term: var}` rather than a shredded node dict.

`BudgetQuery.get_vars` likewise reports only operands that were actually used:
an absent diagnostic is omitted from a `sum`'s operand list (it simply drops
out), but appears as `0.0` in a `product`'s — mirroring the evaluator, which
multiplies the missing factor in as zero. The legacy `get_vars` listed the
recipe verbatim, so it could name variables that were never in the dataset.

### Removed

- The `xbudget/llc90` module (`diff_2d_flux_llc90`), the hand-rolled LLC90 flux
  stitcher. `lateral_divergence` now uses native xgcm for any face-connected
  grid; the two were verified bit-identical on the ECCO grid before it went.

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
  mirroring the legacy engine's behavior rather than failing, so real
  conventions with such placeholders still load. (This same tolerance is what
  let the malformed bolus term above pass silently before it was fixed — the
  warning it emitted is what surfaced the bug.)
