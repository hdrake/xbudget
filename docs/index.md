# xbudget

**Easy handling of budgets diagnosed from General Circulation Models with xarray.**

**xbudget** is a python package that facilitates the handling of budget
diagnostics for finite-volume General Circulation Models (currently only
supporting structured grids, i.e. grids that can be constructed as an
[`xgcm.Grid`](https://xgcm.readthedocs.io/en/stable/) object).

xbudget expects budgets which have a Left-Hand Side (LHS) equal to a Right-Hand
Side (RHS), typically in one of the following two forms:

$$
\frac{\partial}{\partial t} \int \rho \lambda \, dV =
\int \left[ - \nabla \cdot \left( \mathbf{F}_{\lambda} + \rho \lambda \mathbf{u} \right) \right] \, dV
$$

where $\lambda$ is the property density (or tracer concentration), $\mathbf{u}$
is the flow velocity, and $\mathbf{F}_{\lambda}$ is the sum of all non-advective
fluxes of $\lambda$.

xbudget ingests the budget diagnostics as an `xarray.Dataset` — or as an
`xgcm.Grid` when a budget uses staggered-grid operations (finite differences or
lateral flux divergences) — and uses structured metadata, in the form of a
nested dictionary (or `.yaml` file), to close such budgets. While this may seem
trivial for use cases in which there is a single flux to keep track of, total
non-advective fluxes in general circulation models can be composed of dozens of
contributing processes. Since budget diagnostics are often not output as
volume-integrated tendencies, xbudget allows for terms to be derived as sums,
products, differences, reciprocals, or lateral flux divergences (or some
combination of these), including on face-connected grids such as the ECCO LLC90
tiles. For example, ocean heat tendency due to air-sea heat fluxes might be
derived from the difference between vertical heat fluxes across depth interfaces,
summed over longwave, shortwave, sensible, and latent components of the flux, and
multiplied by the ocean cell area.

While drafting a `.yaml` file from scratch for a new model can be daunting, it
only needs to be done once — then closing budgets is a breeze!

New here? The [Quickstart](quickstart.md) closes a budget in about thirty lines,
with made-up data and no downloads. [Writing a recipe](recipes.md) is the
reference for writing a `.yaml` file for your own model.

```{toctree}
:maxdepth: 2
:caption: Contents

installation
quickstart
recipes
examples/handling_missing_diagnostics
examples/MOM6_budget_examples_mass_heat_salt
examples/eccov4r4_budget_examples_mass_heat_salt
examples/eccov4r4_heat_budget_decomposition
contributing
api
```
