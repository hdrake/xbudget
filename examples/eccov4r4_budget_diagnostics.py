import numpy as np
import xarray as xr

def eccov4r4_geothermal_heat_flux_tendency(ds):
    """Compute the geothermal heat flux tendency for ECCOv4r4.

    This maps geothermal heat input at the ocean floor onto the model water
    column and expresses it as a volume-weighted potential temperature
    tendency.

    Args:
        ds: Dataset containing ECCOv4r4 heat-flux diagnostics and grid metrics.

    Returns:
        xr.DataArray: Volume-weighted geothermal heat flux tendency with
        units of degree_C m3 s-1.
    """
    geothermal_flux = ds["geothermalFlux"].copy(deep=True)

    cell_volume = ds["drF"] * ds["hFacC"] * ds["rA"]
    rho0 = 1029.0  # Seawater density (kg/m^3)
    c_p = 3994.0  # Heat capacity (J/kg/K)

    cell_mask = xr.where(ds["hFacC"] > 0, 1.0, 0.0)  # 1 at open C-grid cells
    shifted_cell_mask = cell_mask.shift(k=-1).fillna(0.0)  # Shift one level upward; the bottom layer becomes 0.

    # Expand the bottom geothermal flux into a 3D field aligned with the open-ocean mask.
    # The mask difference is 1 only at the deepest open cell in each column.
    geothermal_flux_3d = geothermal_flux * (cell_mask - shifted_cell_mask)
    geothermal_flux_3d = geothermal_flux_3d.transpose("k", "tile", "j", "i")

    # Convert geothermal heat flux from W/m^2 into a temperature tendency.
    geothermal_forcing = (geothermal_flux_3d / (rho0 * c_p)) / (ds["hFacC"] * ds["drF"])

    # Multiply by cell volume to express the result as a volume-integrated tendency.
    geothermal_heat_flux_tendency = geothermal_forcing * cell_volume
    geothermal_heat_flux_tendency.attrs = {
        "standard_name": "geothermal_heat_flux_convergence",
        "long_name": "Geothermal heat flux convergence",
        "units": "degree_C m3 s-1",
    }

    return geothermal_heat_flux_tendency.fillna(0.0)


def eccov4r4_penetrative_heat_flux_tendency(ds):
    """Compute the penetrative shortwave heat flux tendency for ECCOv4r4.

    This distributes incoming shortwave radiation through the water column
    using the prescribed double-exponential attenuation profile and
    expresses the resulting radiative heating as a volume-weighted
    potential temperature tendency.

    Args:
        ds: Dataset containing ECCOv4r4 shortwave heat-flux diagnostics and grid metrics.

    Returns:
        xr.DataArray: Volume-weighted penetrative heat flux tendency with
        units of degree_C m3 s-1.
    """
    cell_volume = ((ds["drF"] * ds["hFacC"]) * ds["rA"]).fillna(0.0)
    rho0 = 1029.0
    c_p = 3994.0

    shortwave_fraction = 0.62
    zeta1 = 0.6
    zeta2 = 20.0
    z_cutoff = -200

    cell_center_depth = ds["Z"].compute()
    cell_interface_depth = ds["Zp1"].compute()

    interface_depth = np.concatenate([cell_interface_depth.values[:-1], [np.nan]])

    decay = lambda z: shortwave_fraction * np.exp(z / zeta1) + (1.0 - shortwave_fraction) * np.exp(z / zeta2)

    # Evaluate the attenuation profile at the top and bottom interfaces of each cell.
    upper_decay = xr.DataArray(decay(interface_depth[:-1]), coords=[cell_center_depth.k], dims=["k"])
    lower_decay = xr.DataArray(decay(interface_depth[1:]), coords=[cell_center_depth.k], dims=["k"])

    cutoff_index = np.where(cell_center_depth < z_cutoff)[0][0]
    upper_decay.values[cutoff_index:] = 0
    lower_decay.values[cutoff_index - 1 :] = 0

    cell_mask = xr.where(ds["hFacC"] > 0, 1.0, 0.0)
    lower_cell_mask = xr.where(cell_mask.shift(k=-1) == 1.0, 1.0, 0.0)

    # Interior convergence is the vertical divergence of the attenuated shortwave flux.
    interior_shortwave_convergence = ((upper_decay * cell_mask - lower_decay * lower_cell_mask) * ds["oceQsw"])

    # The surface layer is handled separately so the concatenated result stays on the k grid.
    surface_shortwave_convergence = (((upper_decay[0] - lower_decay[0]) * ds["oceQsw"]) * cell_mask.isel(k=0))
    surface_shortwave_convergence = surface_shortwave_convergence.expand_dims(k=[ds["k"].isel(k=0).item()])

    shortwave_convergence = xr.concat([surface_shortwave_convergence, interior_shortwave_convergence.isel(k=slice(1, None))], dim="k").fillna(0.0)

    # Convert the radiative convergence into a volume-integrated temperature tendency.
    shortwave_heat_flux_tendency = (shortwave_convergence * cell_volume / (rho0 * c_p)) / (ds["hFacC"] * ds["drF"])

    shortwave_heat_flux_tendency.attrs = {
        "long_name": "volume-weighted sea water potential temperature tendency due to shortwave heating",
        "units": "degree_C m3 s-1",
    }

    return shortwave_heat_flux_tendency


def eccov4r4_nonpenetrative_heat_flux_tendency(ds):
    """Compute the non-penetrative surface heat flux tendency for ECCOv4r4.

    This isolates the component of the air-sea heat flux that is absorbed
    within the surface model layer, excluding penetrative shortwave
    radiation, and expresses it as a volume-weighted potential temperature
    tendency.

    Args:
        ds: Dataset containing ECCOv4r4 surface heat-flux diagnostics and grid metrics.

    Returns:
        xr.DataArray: Volume-weighted non-penetrative heat flux tendency with
        units of degree_C m3 s-1.
    """
    cell_volume = ((ds["drF"] * ds["hFacC"]) * ds["rA"]).fillna(0.0)

    rho0 = 1029.0
    c_p = 3994.0

    # Mask to the surface layer before applying the non-penetrative heat flux.

    cell_mask = xr.where(ds["hFacC"] > 0, 1.0, 0.0)
    surface_cell_mask = cell_mask * xr.where(ds["k"] == 0, 1.0, np.nan)

    nonshortwave_surface_forcing = ((ds["TFLUX"] - ds["oceQsw"]) * surface_cell_mask).fillna(0.0)  # Broadcast the surface forcing onto the 3D grid.

    # Remove the penetrative shortwave component before converting to a temperature tendency.
    nonshortwave_heat_flux = (nonshortwave_surface_forcing / (rho0 * c_p)) / (ds["hFacC"] * ds["drF"])

    # Multiply by cell volume to match the volume-integrated tendency convention used here.
    nonpenetrative_heat_flux_tendency = (nonshortwave_heat_flux * cell_volume).fillna(0.0)
    nonpenetrative_heat_flux_tendency.attrs = {
        "long_name": "volume-weighted sea water potential temperature tendency due to non-penetrative surface heat fluxes",
        "units": "degree_C m3 s-1",
    }

    return nonpenetrative_heat_flux_tendency
