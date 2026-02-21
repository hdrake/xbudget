
xbudget: easy handling of budgets diagnosed from General Circulation Models with xarray
========================================================================================

*xbudget* is a python package that facilitates the handling of budget diagnostics for finite-volume General Circulation Models (currently only supporting structured grids).

xbudget expects budgets which have a Left-Hand Side (LHS) equal to a Right-Hand Side (RHS), typically in one of the following two forms:

.. math::
   \frac{\partial}{\partial t} \int \rho \lambda \, dV = \int \left[ - \nabla \cdot \left( \mathbf{F}_{\lambda} + \rho \lambda \mathbf{u} \right) \right] \, dV 

where :math:`\lambda` is the property density (or tracer concentration), :math:`\mathbf{u}` is the flow velocity, and :math:`\mathbf{F}_{\lambda}` is the sum of all non-advective fluxes of :math:`\lambda`.

xbudget ingests an `xgcm.Grid` object containing the budget diagnostics and uses structured metadata, in the form of a nested dictionary (or `.yaml` file), to close such budgets. While this may seem trivial for use cases in which there is a single flux to keep track of, total non-advective fluxes in general circulation models can be composed of dozens of contributing processes. Since budget diagnostics are often not output as volume-integrated tendencies, xbudget allows for terms to be derived as sums, products, or differences (or some combination of these). For example, ocean heat tendency due to air-sea heat fluxes might be derived from the difference between vertical heat fluxes across depth interfaces, summed over longwave, shortwave, sensible, and latent components of the flux, and multiplied by the ocean cell area.

While drafting a `.yaml` file from scratch for a new model can be daunting, it only needs to be done once -- then closing budgets is a breeze!

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   examples/MOM6_budget_examples_mass_heat_salt
