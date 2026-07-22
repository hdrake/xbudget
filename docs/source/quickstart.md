# Quickstart

This page closes a budget end to end in about thirty lines, with data you can
make up on the spot — no model output to download, and no `xgcm.Grid` needed.

The idea is simple. A budget says *this tendency* equals *the sum of these
processes*. Your model writes out the pieces; xbudget assembles them, gives each
one a predictable name, and lets you check that the two sides agree.

## Some data

Pretend this came out of an ocean model. Heat changes because of advection,
diffusion, and a surface flux — and because we are making the numbers up, we can
make the budget balance exactly:

```python
import numpy as np
import xarray as xr
import xbudget

rng = np.random.default_rng(0)
dims, coords = ("x", "y"), {"x": [0, 1, 2], "y": [0, 1, 2]}

advection = rng.random((3, 3))
diffusion = rng.random((3, 3))
surface_flux = rng.random((3, 3))       # W m-2
cell_area = np.full((3, 3), 1.0e6)      # m2

ds = xr.Dataset(
    {
        # the tendency the model reports, in W
        "heat_tendency": (dims, advection + diffusion + surface_flux * cell_area),
        "advective_flux_convergence": (dims, advection),
        "diffusive_flux_convergence": (dims, diffusion),
        "surface_heat_flux": (dims, surface_flux),
        "cell_area": (dims, cell_area),
    },
    coords=coords,
)
```

Note the surface term: the model reports a flux **per unit area**, so it has to
be multiplied by the cell area before it can join a budget written in watts.
That kind of bookkeeping is most of what xbudget does for you.

## The recipe

You describe the budget once, as a nested dict — the *recipe*. Each budget has a
`lhs` and an `rhs`; each side is built from operations (`sum`, `product`, and a
few others). You write `var` only to name a diagnostic; anything without one is
something xbudget derives for you:

```python
recipe = {
    "heat": {
        "lhs": {"sum": {"tendency": {"var": "heat_tendency"}}},
        "rhs": {
            "sum": {
                "advection": {"var": "advective_flux_convergence"},
                "diffusion": {"var": "diffusive_flux_convergence"},
                "surface_forcing": {
                    "product": {
                        "flux": "surface_heat_flux",
                        "area": "cell_area",
                    },
                },
            },
        },
    }
}
```

`{"var": "advective_flux_convergence"}` points straight at a diagnostic.
`surface_forcing` is the interesting one: it is a `product` of a diagnostic and a
grid metric, which xbudget multiplies out for you.

## Collect, then query

```python
xbudget.collect_budgets(ds, recipe)
q = xbudget.BudgetQuery(ds, recipe)
```

`collect_budgets` computes every term and writes it into `ds`. `BudgetQuery` is
how you find them afterwards:

```python
>>> q.aggregate()
{'heat': {'lhs': {'tendency': 'heat_lhs_tendency'},
          'rhs': {'advection': 'heat_rhs_advection',
                  'diffusion': 'heat_rhs_diffusion',
                  'surface_forcing': 'heat_rhs_surface_forcing'}}}
```

Each term is named after its path through the recipe, joined with underscores —
`heat` / `rhs` / `advection` becomes `heat_rhs_advection`. Nothing to memorize,
and nothing to parse back out.

## Does it close?

This is the question xbudget exists to answer:

```python
>>> residual = ds[q.var("heat_lhs")] - ds[q.var("heat_rhs")]
>>> float(abs(residual).max())
0.0
```

Zero, because we built the data to balance. With real model output you would
expect something small but non-zero, and a residual that is *not* small usually
means a term is missing from your recipe or is on the wrong side.

## Where did a term come from?

Every variable xbudget derives carries its own provenance:

```python
>>> ds[q.var("heat_rhs_surface_forcing")].attrs
{'provenance': ['surface_heat_flux', 'cell_area'],
 'xbudget_path': ['heat', 'rhs', 'surface_forcing'],
 'xbudget_op': 'product'}
```

Or ask the query directly, which also works before you go digging in the
dataset:

```python
>>> q.get_vars("heat_rhs")
{'var': 'heat_rhs', 'sum': ['heat_rhs_advection', 'heat_rhs_diffusion', 'heat_rhs_surface_forcing']}

>>> q.get_vars("heat_rhs_surface_forcing")
{'var': 'heat_rhs_surface_forcing', 'product': ['surface_heat_flux', 'cell_area']}
```

## Keeping the recipe

A recipe is just a dict, so you can save it and reuse it:

```python
xbudget.save_yaml(recipe, "my_model.yaml")
recipe = xbudget.load_yaml("my_model.yaml")
```

`save_yaml` validates before writing, so a malformed recipe fails while you are
writing it rather than the next time someone loads it.

## Your own model

Real recipes are bigger than this one — a full ocean heat budget can have
dozens of terms — but they are the same shape, and you only write one once.
xbudget ships several:

```python
recipe = xbudget.load_preset_budget("MOM6")
xbudget.collect_budgets(grid, recipe)
```

Currently shipped: `MOM6`, `MOM6_3Donly`, `MOM6_drift`, `MOM6_surface`, and
`ECCOV4r4_native`. If your model is not here, writing a recipe for it is a
one-time job — see [Writing a recipe](recipes.md) for the full schema,
the available operations, and how a recipe becomes variables.

Two things this page skipped, both of which real budgets need:

- **Grids.** Terms built by differencing a staggered flux (or by a lateral
  divergence) need an `xgcm.Grid` instead of a plain `Dataset`, so xbudget knows
  the discretization. Pass the grid to `collect_budgets` in exactly the same way.
- **Scale.** The examples below run against real model output, with dask.

For a full worked budget, see the
[MOM6 example](examples/MOM6_budget_examples_mass_heat_salt.ipynb).
