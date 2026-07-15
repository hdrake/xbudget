# Writing a convention

A *convention* (the `xbudget_dict`) is the recipe that tells xbudget how to build
each term of a budget out of the diagnostics your model actually writes. It is an
ordinary nested dictionary, so you can write it as YAML or build it in Python —
they are the same thing, and this page covers both directions.

Writing one for a new model is a one-time job. Once it exists, closing that
model's budgets is a two-line call.

If you have not closed a budget before, start with the
[Quickstart](quickstart.md); this page is the reference behind it.

## Anatomy

The nesting is always the same:

```
budget            (e.g. "heat")
├── metadata      (lambda, thickness, surface_lambda, ...)
├── lhs           a term
└── rhs           a term
     └── operation   (sum, product, difference, reciprocal, lateral_divergence)
          └── operands: named sub-terms, diagnostic names, or constants
```

A minimal budget, in YAML:

```yaml
heat:
  lhs:
    var: null
    sum:
      var: null
      tendency: {var: "opottemptend"}
  rhs:
    var: null
    sum:
      var: null
      surface_forcing:
        var: null
        product:
          var: null
          flux_per_unit_area: "hfds"
          area: "areacello"
```

Three things are going on:

**`var: null` means "xbudget fills this in".** It marks a quantity that gets
derived. When the value is a *string* instead, as in `tendency: {var:
"opottemptend"}`, it is a leaf: it points straight at a diagnostic in your
dataset.

**Operands are named.** The keys under a `sum` or `product` (`flux_per_unit_area`,
`area`) are labels you choose. They are not looked up anywhere — they name the
term for your own benefit, and they become part of the output variable name, so
pick names you would want to read later.

**Bare numbers are constants.** A `sign: -1.` or `density: 1035.` operand is used
as a scalar, which is how conventions flip a sign or convert units:

```yaml
lateral:
  var: null
  product:
    var: null
    sign: -1.
    tracer_content_tendency_per_unit_area: "T_advection_xy"
    area: "areacello"
```

Budget-level keys that are not `lhs`/`rhs` are metadata, carried through
untouched — `lambda` (the tracer the budget is written in), `thickness`,
`surface_lambda`:

```yaml
heat:
  lambda: "thetao"
  surface_lambda: "tos"
  lhs: ...
```

## The operations

| Operation | Shape | Produces |
|---|---|---|
| `sum` | named operands | their sum |
| `product` | named operands | their product |
| `difference` | one operand | the operand differenced across its grid axis |
| `reciprocal` | one variable | `1/x`, with zeros mapped to infinity (so `1/x` is 0) |
| `lateral_divergence` | `Fx` and `Fy` sub-terms | `div(Fx, Fy)` on cell centres |

`sum` and `product` take any number of operands, each of which may itself be a
sub-term, so budgets nest as deeply as you need.

**`difference`** takes a single operand and differences it across whichever axis
it is staggered on — the axis is inferred from the operand's dimensions, so you
do not name it:

```yaml
zonal_divergence:
  var: null
  difference:
    var: null
    zonal_mass_transport: "umo"
```

The operand can also be a computed sub-term rather than a raw diagnostic, in
which case it is evaluated first and then differenced.

**`reciprocal`** inverts a variable, named bare or as `{var: name}`:

```yaml
dt_inv:
  var: null
  reciprocal: {var: null, dt: {var: "dt"}}
```

**`lateral_divergence`** takes two flux sub-terms and forms their horizontal
divergence:

```yaml
volume_flux_divergence:
  var: null
  lateral_divergence:
    var: null
    Fx:
      var: null
      product: {var: null, u_velocity: "UVELMASS", dyG: "dyG", dz: "drF"}
    Fy:
      var: null
      product: {var: null, v_velocity: "VVELMASS", dxG: "dxG", dz: "drF"}
```

`difference` and `lateral_divergence` are discretization-aware, so they need an
`xgcm.Grid` rather than a plain `Dataset`. `lateral_divergence` uses xgcm's
face-connected differencing, so it is correct on tiled topologies such as the
ECCO LLC90 grid — provide `face_connections` when you build the grid and the
tiles are stitched for you.

## How a recipe becomes variables

```
load_yaml / load_preset_budget    YAML file  -> dict
        |
parse_budgets                     dict       -> typed tree   (validates)
        |
evaluate_budgets                  tree + data -> variables written into your dataset
```

`collect_budgets` runs the last two steps for you; the intermediate pieces are
public if you want them (`xbudget.parse_budgets`, `xbudget.evaluate_budgets`).

**Naming.** Each derived variable is named by its path through the recipe, joined
with underscores. A term at `heat` → `rhs` → `diffusion` → `lateral` becomes:

```
heat_rhs_diffusion_lateral
```

That is the whole rule. A term carrying more than one operation (say a bulk
`product` *and* an equivalent finer `sum` of the same quantity) emits one
variable per operation: the first `sum`/`product` gets the plain path name and
the others get the operation appended, e.g. `heat_rhs_diffusion_lateral_sum`.

**Attributes.** Every derived variable records where it came from:

```python
>>> ds["heat_rhs_surface_forcing"].attrs
{'provenance': ['surface_heat_flux', 'cell_area'],
 'xbudget_path': ['heat', 'rhs', 'surface_forcing'],
 'xbudget_op': 'product'}
```

`xbudget_path` is the structured identity, so you never have to parse the flat
name to work out what a variable is.

You can query all of this without touching attributes at all:

```python
q = xbudget.BudgetQuery(ds, xbudget_dict)
q.var("heat_rhs_diffusion_lateral")   # the variable name
q.get_vars("heat_rhs")                # its operands
q.aggregate()                         # every top-level term, per budget
```

## Authoring in Python, exporting to YAML

Nothing says a recipe has to start life as a file. Build it as a dict, check it
against real data, then write it out:

```python
xbudget_dict = {
    "heat": {
        "lhs": {"var": None, "sum": {"var": None, "tendency": {"var": "heat_tendency"}}},
        "rhs": {
            "var": None,
            "sum": {
                "var": None,
                "advection": {"var": "advective_flux_convergence"},
                "surface_forcing": {
                    "var": None,
                    "product": {"var": None, "flux": "surface_heat_flux", "area": "cell_area"},
                },
            },
        },
    }
}

xbudget.save_yaml(xbudget_dict, "my_model.yaml")
```

which writes:

```yaml
heat:
  lhs:
    var: null
    sum:
      var: null
      tendency:
        var: heat_tendency
  rhs:
    var: null
    sum:
      var: null
      advection:
        var: advective_flux_convergence
      surface_forcing:
        var: null
        product:
          var: null
          flux: surface_heat_flux
          area: cell_area
```

and reads back identically:

```python
>>> xbudget.load_yaml("my_model.yaml") == xbudget_dict
True
```

`save_yaml` validates before it writes, so a malformed recipe raises
`BudgetParseError` while you are authoring it rather than silently landing on
disk. Key order is preserved rather than alphabetized, because order is
meaningful: operands come back from `get_vars` in the order you wrote them.

```{note}
The round-trip is clean because `collect_budgets` does not modify your recipe.
The deprecated `name_scheme="legacy"` does: it fills every `var` field in place
with that run's variable names, so saving a recipe afterwards would pin it to
one particular run. Another reason to leave `name_scheme` alone.
```

## When something is wrong

**Structural errors raise.** `parse_budgets` (and therefore `save_yaml` and
`collect_budgets`) raises `BudgetParseError` naming the offending path:

```python
>>> xbudget.parse_budgets({"heat": {"rhs": {"sum": "not a dict"}}})
BudgetParseError: 'sum' at heat/rhs must be a dict, got str.
```

**Missing diagnostics warn and skip.** If your dataset does not have a variable
a term references, that term is skipped with a `UserWarning` and everything else
still builds. This is deliberate: one convention can serve datasets with
different diagnostics available. It also means a budget can fail to close simply
because a term was silently dropped, so it is worth reading those warnings —
`BudgetQuery.var()` returns `None` for a term that was not materialized, which
is the programmatic way to check.

**Stray keys warn.** A key that is neither `var` nor an operation is ignored with
a warning, usually meaning an enclosing `product:` was left out:

```yaml
# wrong: sign/density sit directly on the term and are ignored
bolus_convergence:
  var: null
  sign: -1.0
  density: 1029.0

# right
bolus_convergence:
  var: null
  product:
    var: null
    sign: -1.0
    density: 1029.0
```

This exact malformation once dropped the eddy bolus transport from the ECCO mass
budget entirely, and the only outward sign was that warning. They are worth
reading rather than filtering.

## Contributing a convention

If you write a convention for a model xbudget does not ship yet, please
[open an issue or a pull request](https://github.com/hdrake/xbudget/issues) —
the existing ones in `xbudget/conventions/` are good references, and
`MOM6.yaml` is the most complete.
