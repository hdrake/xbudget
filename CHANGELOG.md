# Changelog

## Unreleased — internals refactor (typed engine)

This release replaces the recursive dict-walking engine with a typed expression
tree (parse → evaluate). The convention/YAML format is **unchanged**; the
in-memory representation, the engine, and the default output variable names are
new. Numerical results are identical to the previous engine (verified by an
end-to-end equivalence test on the example MOM6 grid).

### Quick migration

Add `name_scheme="legacy"` to your `collect_budgets` call to keep the previous
behavior exactly — historical variable names *and* the in-place filling of the
recipe dict that `get_vars`/`aggregate` depend on:

```python
xbudget.collect_budgets(grid, xbudget_dict, name_scheme="legacy")
```

Everything downstream (old variable names, `get_vars`, `aggregate`) then works
unchanged. Adopt the new scheme at your own pace.

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
   after a `name_scheme="legacy"` run (which still fills the dict). To query the
   `v1` output, use the `records`/`alias_map` returned by `evaluate_budgets`
   (below) and the `provenance` / `xbudget_path` attributes on each variable.

3. **`collect_budgets` signature** gained a `name_scheme` keyword and its first
   parameter is named `data` (a grid or dataset), matching its long-standing
   behavior of accepting either.

### New

- `xbudget.parse_budgets(xbudget_dict)` → typed tree (`xbudget.nodes.Budget`),
  the single schema-validating entry point; raises `xbudget.BudgetParseError`
  with the offending path on malformed conventions.
- `xbudget.evaluate_budgets(data, budgets)` → pure evaluator; returns
  `(alias_map, records)` where `alias_map` maps every legacy name to its new
  name and `records` maps each new variable to its `{path, op, ...}` metadata.
- Each derived variable carries `xbudget_path` (structured identity) and
  `provenance` (immediate inputs) attributes.

### Fixed

- The `difference` operation's grid guard was misattached, so a `difference`
  on a plain `Dataset` raised an opaque `NameError`, and a `difference` term
  evaluated after another operation in the same node raised spuriously even
  with a valid grid. It now raises a clear `ValueError` up front when no grid
  is supplied. (Also fixes a mutable-default-argument footgun in the internal
  search helper.)

### Deprecated

- `budget_fill_dict` is retained as the legacy reference engine (still used
  internally by `name_scheme="legacy"`) but is superseded by `collect_budgets`
  / `evaluate_budgets`.
