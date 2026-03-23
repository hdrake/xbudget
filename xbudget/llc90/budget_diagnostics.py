import numpy as np
import xarray as xr


def calc_llc90_geothermal_heatflux_tendency(ds):
    geoflx_llc = ds["geothermalFlux"].copy(deep=True)

    volcello = ds["drF"] * ds["hFacC"] * ds["rA"]
    rho0 = 1029.0  # Seawater density (kg/m^3)
    c_p = 3994.0  # Heat capacity (J/kg/K)

    hFacC = xr.where(ds["hFacC"] > 0, 1.0, 0.0)  # 1 at the "open" C grid cells
    hFacC_shifted = hFacC.shift(k=-1).fillna(0.0)  # shifts everything one layer up but leaves the bottom with NaNs

    # Create 3d field of geothermal heat flux
    geoflx3d = geoflx_llc * (hFacC - hFacC_shifted)
    GEOFLX = geoflx3d.transpose("k", "tile", "j", "i")

    # Add geothermal heat flux to forcing field and convert from W/m^2 to degC/s
    G_geothermal_forcing = ((GEOFLX) / (rho0 * c_p)) / (ds["hFacC"] * ds["drF"])

    geothermal_heatflux_tendency = G_geothermal_forcing * volcello
    geothermal_heatflux_tendency.attrs = {
        "standard_name": "geothermal_heat_flux_convergence",
        "long_name": "Geothermal heat flux convergence",
        "units": "degree_C m3 s-1",
    }

    return geothermal_heatflux_tendency.fillna(0.0)


def llc90_penetrative_heat_flux_tendency(ds):
    volcello = ((ds["drF"] * ds["hFacC"]) * ds["rA"]).fillna(0.0)
    rho0 = 1029.0
    c_p = 3994.0

    R = 0.62
    zeta1 = 0.6
    zeta2 = 20.0
    z_cutoff = -200

    Z = ds["Z"].compute()
    Zp1 = ds["Zp1"].compute()

    RF = np.concatenate([Zp1.values[:-1], [np.nan]])

    decay = lambda z: R * np.exp(z / zeta1) + (1.0 - R) * np.exp(z / zeta2)

    q1 = xr.DataArray(decay(RF[:-1]), coords=[Z.k], dims=["k"])
    q2 = xr.DataArray(decay(RF[1:]), coords=[Z.k], dims=["k"])

    zCut = np.where(Z < z_cutoff)[0][0]
    q1.values[zCut:] = 0
    q2.values[zCut - 1 :] = 0

    mskC = xr.where(ds["hFacC"] > 0, 1.0, 0.0)
    mskC_down = xr.where(mskC.shift(k=-1) == 1.0, 1.0, 0.0)

    interior = ((q1 * mskC - q2 * mskC_down) * ds["oceQsw"])

    surface = (((q1[0] - q2[0]) * ds["oceQsw"]) * mskC.isel(k=0))
    surface = surface.expand_dims(k=[ds["k"].isel(k=0).item()])

    shortwave_convergence = xr.concat([surface, interior.isel(k=slice(1, None))], dim="k").fillna(0.0)

    G_shortwave_convergence = (shortwave_convergence * volcello / (rho0 * c_p)) / (ds["hFacC"] * ds["drF"])

    G_shortwave_convergence.attrs = {
        "long_name": "volume-weighted sea water potential temperature tendency due to shortwave heating",
        "units": "degree_C m3 s-1",
    }

    return G_shortwave_convergence


def llc90_nonpenetrative_heat_flux_tendency(ds):
    volcello = ((ds["drF"] * ds["hFacC"]) * ds["rA"]).fillna(0.0)

    rho0 = 1029.0
    c_p = 3994.0

    # shortwave_heatflux_convergence = llc90_shortwave_heat_tendency(ds)

    mskC = xr.where(ds["hFacC"] > 0, 1.0, 0.0)
    surface_mskC = mskC * xr.where(ds["k"] == 0, 1.0, np.nan)

    surf_forcing_no_shortwave = ((ds["TFLUX"] - ds["oceQsw"]) * surface_mskC).fillna(0.0)  # cast to 3D

    nonshortwave_surface_heat_flux = (surf_forcing_no_shortwave / (rho0 * c_p)) / (ds["hFacC"] * ds["drF"])

    G_nonshortwave_surface_flux = (nonshortwave_surface_heat_flux * volcello).fillna(0.0)
    G_nonshortwave_surface_flux.attrs = {
        "long_name": "volume-weighted sea water potential temperature tendency due to non-penetrative surface heat fluxes",
        "units": "degree_C m3 s-1",
    }

    return G_nonshortwave_surface_flux