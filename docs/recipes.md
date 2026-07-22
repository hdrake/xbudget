# Writing a recipe

A *recipe* tells xbudget how to build each term of a budget out of the
diagnostics your model actually writes. It is an ordinary nested dictionary, so
you can write it as YAML or build it in Python — they are the same thing, and
this page covers both directions.

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
    sum:
      tendency: {var: "opottemptend"}
  rhs:
    sum:
      surface_forcing:
        product:
          flux_per_unit_area: "hfds"
          area: "areacello"
```

Three things are going on:

**`var` names a diagnostic — and you only write it when you have one to name.**
`tendency: {var: "opottemptend"}` is a leaf: it points straight at a variable in
your dataset. All other variables are inferred and *derived* by `xbudget` — the
absence of `var` is what marks a quantity xbudget computes for you.

```{note}
You may see `var: null` scattered through older recipes (and through
xbudget's own, before v0.7.0). The pre-v0.7.0 engine read `var` by direct
subscript, so it *did* require the key to be present on every derived term —
remove it and you got a `KeyError`. The v0.7.0 parser dropped that requirement:
it reads a missing `var` and an explicit `var: null` identically. Old recipes
keep working untouched, but new ones can leave the placeholders out, which is
roughly a third fewer lines.
```

**Operands are named.** The keys under a `sum` or `product` (e.g.,
`flux_per_unit_area` and `area`) are labels you choose. They are not looked up
anywhere — they name the term for your own benefit, and they become part of the
output variable name inferred by xbudget, so pick names you would want to read
later.

**Bare numbers are constants.** A `sign: -1.` or `density: 1035.` operand is used
as a scalar, which is how recipes flip a sign or convert units:

```yaml
lateral:
  product:
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

Metadata describes budget *state* the engine does not build but downstream tools
depend on. In particular the mass budget's `thickness` (e.g. `"thkcello"`) is its
prognostic layer-thickness variable, which
[`xwmt`](https://github.com/hdrake/xwmt) — the water-mass transformation package
whose needs motivated xbudget in the first place — relies on to build the
vertical metrics of its `xwmt.WaterMass` object. Read metadata through the query
layer rather than
reaching into the raw dict, so it is independent of the naming scheme and works
on a reopened dataset:

```python
q = xbudget.BudgetQuery(ds, recipe)
q.thickness()            # -> "thkcello"   (defaults to the mass budget)
q.lambda_var("heat")     # -> "thetao"
q.surface_lambda("heat") # -> "tos"
q.metadata("mass")       # -> {"lambda": "density", "thickness": "thkcello"}
```

## Seeing a recipe

A real recipe nests deeply, and printed as raw JSON it is hard to read. Instead,
`show_recipe` renders it as a collapsible tree — expand a level by clicking its
arrow, the way an `xarray.Dataset` repr expands:

```python
xbudget.show_recipe(recipe)          # every budget
xbudget.show_recipe(recipe, "heat")  # just one
```

In a Jupyter/VSCode notebook it displays as an interactive HTML tree (operator
badges, diagnostics, and constants distinguished); printed at a terminal it
falls back to an indented ASCII tree. It reads a recipe directly, so you can
inspect one before ever running it.

Once you *have* run a budget, displaying the query does the same but annotated
with each term's resolved variable name — with terms whose diagnostics were
missing greyed out, so the tree doubles as a map of what the run materialized:

```python
xbudget.collect_budgets(grid, recipe)
xbudget.BudgetQuery(grid, recipe)    # same tree, annotated with run state
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
  difference:
    zonal_mass_transport: "umo"
```

The operand can also be a computed sub-term rather than a raw diagnostic, in
which case it is evaluated first and then differenced.

```{warning}
Prefer `lateral_divergence` over `difference` for **horizontal** flux
convergences. A one-axis `difference` differences within a single logical grid,
so it is only correct where that axis is contiguous; it silently gives the wrong
answer across the seams of a tiled topology (e.g. the ECCO LLC90 faces).
`lateral_divergence` forms the full `div(Fx, Fy)` through xgcm's face-connected
differencing, which stitches those seams correctly. Reach for a bare
`difference` only for a genuinely one-dimensional convergence (e.g. a vertical
one) or on a grid you know has no face connections.
```

**`reciprocal`** inverts a variable, named bare or as `{var: name}`:

```yaml
dt_inv:
  reciprocal: {dt: {var: "dt"}}
```

**`lateral_divergence`** takes two flux sub-terms and forms their horizontal
divergence:

```yaml
volume_flux_divergence:
  lateral_divergence:
    Fx:
      product: {u_velocity: "UVELMASS", dyG: "dyG", dz: "drF"}
    Fy:
      product: {v_velocity: "VVELMASS", dxG: "dxG", dz: "drF"}
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
q = xbudget.BudgetQuery(ds, recipe)
q.var("heat_rhs_diffusion_lateral")   # the variable name
q.get_vars("heat_rhs")                # its operands
q.aggregate()                         # every top-level term, per budget
```

## Authoring in Python, exporting to YAML

Nothing says a recipe has to start life as a file. Build it as a dict, check it
against real data, then write it out:

```python
recipe = {
    "heat": {
        "lhs": {"sum": {"tendency": {"var": "heat_tendency"}}},
        "rhs": {
            "sum": {
                "advection": {"var": "advective_flux_convergence"},
                "surface_forcing": {
                    "product": {"flux": "surface_heat_flux", "area": "cell_area"},
                },
            },
        },
    }
}

xbudget.save_yaml(recipe, "my_model.yaml")
```

which writes:

```yaml
heat:
  lhs:
    sum:
      tendency:
        var: heat_tendency
  rhs:
    sum:
      advection:
        var: advective_flux_convergence
      surface_forcing:
        product:
          flux: surface_heat_flux
          area: cell_area
```

and reads back identically:

```python
>>> xbudget.load_yaml("my_model.yaml") == recipe
True
```

`save_yaml` validates before it writes, so a malformed recipe raises
`BudgetParseError` while you are authoring it rather than silently landing on
disk. Key order is preserved rather than alphabetized, because order is
meaningful: operands come back from `get_vars` in the order you wrote them.

```{note}
The round-trip is clean because `collect_budgets` never modifies your recipe —
it only adds derived variables to the dataset. So you can save a recipe at any
point and get back exactly what you wrote.
```

## When something is wrong

**Structural errors raise.** `parse_budgets` (and therefore `save_yaml` and
`collect_budgets`) raises `BudgetParseError` naming the offending path:

```python
>>> xbudget.parse_budgets({"heat": {"rhs": {"sum": "not a dict"}}})
BudgetParseError: 'sum' at heat/rhs must be a dict, got str.
```

**Missing diagnostics skip — but never silently forget.** If your dataset does
not have a variable a term references, that term is skipped and everything else
still builds. This is deliberate: one recipe can serve datasets with different
diagnostics available. The danger is a budget that looks complete but isn't — a
term quietly dropped, a residual you blame on numerics. xbudget is built so that
never happens without a trace.

Take the [Quickstart](quickstart.md) budget, but run it against a dataset that is
missing `surface_heat_flux`:

```python
>>> import warnings
>>> with warnings.catch_warnings(record=True) as w:
...     warnings.simplefilter("always")
...     xbudget.collect_budgets(ds, recipe)   # ds has no surface_heat_flux
>>> print(w[-1].message)
xbudget: missing diagnostic(s) ['surface_heat_flux']; 1 budget term(s) are now
incomplete: ['heat_rhs']. The affected variables carry
`xbudget_incomplete`/`xbudget_missing` attrs; query them with
BudgetQuery.missing(). Mark expected-absent terms `optional` in the recipe to
silence this, or use on_missing='raise' to fail instead.
```

One summary at the end of the run — not one warning per operand — naming both the
absent diagnostic and the *consequence*: which budget term is now incomplete.

**Incompleteness is a durable, queryable property, not just a warning.** Every
variable built from fewer inputs than its recipe describes is stamped, and the
stamp survives being written to disk and reopened:

```python
>>> ds["heat_rhs"].attrs
{'provenance': ['heat_rhs_advection', 'heat_rhs_diffusion'],
 'xbudget_path': ['heat', 'rhs'], 'xbudget_op': 'sum',
 'xbudget_incomplete': 1, 'xbudget_missing': ['surface_forcing']}
```

`provenance` says what went *in*; `xbudget_missing` says what was supposed to and
didn't. The query layer reads both back — including on a reopened dataset:

```python
>>> q = xbudget.BudgetQuery(ds, recipe)
>>> q.is_complete("heat_rhs")
False
>>> q.missing()
{('heat', 'rhs'): ['surface_forcing'],
 ('heat', 'rhs', 'surface_forcing'): ['surface_heat_flux']}
>>> q.incomplete_terms()
['heat_rhs']
>>> q.get_vars("heat_rhs")
{'var': 'heat_rhs', 'sum': ['heat_rhs_advection', 'heat_rhs_diffusion'],
 'missing': ['surface_forcing']}
```

Note that `aggregate()` shows only what actually materialized, so a dropped term
just isn't there — that is exactly the trap `missing()` exists to close. Reach
for `missing()` / `is_complete()` whenever a budget doesn't close.

**A missing product factor drops the term — it is not multiplied by zero.** A
`sum` keeps its surviving operands (and is flagged incomplete); a `product` needs
every factor, so if one is absent the whole term is *not built*. `q.var()` returns
`None` for it, rather than a variable full of zeros that would read as a real,
null contribution to whatever consumes it. (Before v0.7.0 the engine fabricated
that zero; see the CHANGELOG.)

**Choosing how loud to be: `on_missing`.** The default is `"warn"` (the summary
above). Two other policies:

```python
xbudget.collect_budgets(ds, recipe, on_missing="raise")   # fail on any required gap
xbudget.collect_budgets(ds, recipe, on_missing="ignore")  # skip silently
```

`"raise"` turns a recipe/dataset mismatch into a `MissingDiagnosticError` (its
`.missing` attribute lists every gap at once) — use it when you need a guarantee
that the recipe matched the data. `"ignore"` is silent but **still stamps the
`xbudget_incomplete` attributes**, so the record survives even when the warning
doesn't.

**Declaring an expected absence: `optional`.** Some diagnostics are legitimately
absent on some datasets, and you don't want a warning every run. Mark the term
`optional` instead of deleting it — the intent stays documented in the recipe,
and the term is dropped with no warning, no `on_missing="raise"` error, and no
`incomplete` flag on its parent:

```yaml
rhs:
  sum:
    advection: {var: "advective_flux_convergence"}
    eddy_bolus:            # only some model configurations output this
      optional: true
      product:
        flux: "bolus_flux"
        area: "cell_area"
```

`optional` covers the whole subtree beneath it. It is the honest alternative to
deleting a term to quiet a warning — deleting erases the fact that the term was
ever expected; `optional` keeps it.

**Stray keys warn.** A key that is neither `var` nor an operation is ignored with
a warning, usually meaning an enclosing `product:` was left out:

```yaml
# wrong: sign/density sit directly on the term and are ignored
flux_convergence:
  var: null
  sign: -1.0
  density: 1029.0

# right
flux_convergence:
  var: null
  product:
    var: null
    sign: -1.0
    density: 1029.0
```

A missing `product:` wrapper like this silently changes what the term computes —
its factors are dropped rather than multiplied in — and the only outward sign is
that warning. They are worth reading rather than filtering.

## Contributing a recipe

If you write a recipe for a model xbudget does not ship yet, please
[open an issue or a pull request](https://github.com/hdrake/xbudget/issues) —
the existing ones in `xbudget/recipes/` are good references, and
`MOM6.yaml` is the most complete.
