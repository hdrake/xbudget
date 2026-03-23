import numpy as np
import xarray as xr

def calc_llc90_geothermal_heatflux_tendency(ds):
    def download_LLC90_geothermal_fluxes(ds): 
        import os
        import tempfile
        import requests
        import ecco_v4_py as ecco

        url = "https://github.com/ECCO-GROUP/ECCO-v4-Python-Tutorial/raw/master/misc/geothermalFlux.bin"

        with tempfile.TemporaryDirectory() as td:
            fp = os.path.join(td, "geothermalFlux.bin")

            with requests.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(fp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)

            geoflx = ecco.read_llc_to_tiles(td, "geothermalFlux.bin")

            # Convert numpy array to an xarray DataArray with matching dimensions as the monthly mean fields
            geoflx_llc = xr.DataArray(geoflx,coords={'tile': ds.tile.values,
                                                    'j': ds.j.values,
                                                    'i': ds.i.values},dims=['tile','j','i'])
            geoflx_llc.attrs = {'standard_name': 'geothermalFlux','long_name': '2D Geothermal heat flux','units': 'W/m^2'}

        ds["geothermalFlux"] = geoflx_llc #TODO: set as a coord

    if "geothermalFlux" not in ds: 
        print("geothermalFlux not found, retrieving from GitHub...")
        download_LLC90_geothermal_fluxes(ds)

    geoflx_llc = ds["geothermalFlux"].copy(deep=True)

    volcello = ds["drF"] * ds["hFacC"] * ds["rA"]
    rho0 = 1029.0 # Seawater density (kg/m^3)
    c_p = 3994.0 # Heat capacity (J/kg/K)
    
    hFacC = xr.where(ds["hFacC"] > 0, 1.0, 0.0) #1 at the "open" C grid cells 
    hFacC_shifted = hFacC.shift(k=-1).fillna(0.0) #shifts everything one layer up but leaves the bottom with NaNs
        
    # Create 3d field of geothermal heat flux
    geoflx3d = geoflx_llc * (hFacC - hFacC_shifted)
    GEOFLX = geoflx3d.transpose('k','tile','j','i')
    
    # Add geothermal heat flux to forcing field and convert from W/m^2 to degC/s
    G_geothermal_forcing = ((GEOFLX)/(rho0*c_p))/(ds["hFacC"]*ds["drF"])

    geothermal_heatflux_tendency = G_geothermal_forcing * volcello
    geothermal_heatflux_tendency.attrs = {'standard_name': 'geothermal_heat_flux_convergence',
                                        'long_name': 'Geothermal heat flux convergence',
                                        'units': 'degree_C m3 s-1'}
        
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
    q2.values[zCut - 1:] = 0

    mskC = xr.where(ds["hFacC"] > 0, 1.0, 0.0)
    mskC_down = xr.where(mskC.shift(k=-1) == 1.0, 1.0, 0.0)

    interior = ((q1 * mskC - q2 * mskC_down) * ds["oceQsw"])

    surface = (((q1[0] - q2[0]) * ds["oceQsw"]) * mskC.isel(k=0))
    surface = surface.expand_dims(k=[ds["k"].isel(k=0).item()])

    shortwave_convergence = xr.concat([surface, interior.isel(k=slice(1, None))], dim="k").fillna(0.0)

    G_shortwave_convergence = (shortwave_convergence * volcello  / (rho0 * c_p)) / (ds["hFacC"] * ds["drF"])

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

    surf_forcing_no_shortwave = ((ds["TFLUX"] - ds["oceQsw"]) * surface_mskC).fillna(0.0) #cast to 3D
    
    nonshortwave_surface_heat_flux = (surf_forcing_no_shortwave / (rho0 * c_p)) / (ds["hFacC"] * ds["drF"])

    G_nonshortwave_surface_flux = (nonshortwave_surface_heat_flux * volcello).fillna(0.0)
    G_nonshortwave_surface_flux.attrs = {
        "long_name": "volume-weighted sea water potential temperature tendency due to non-penetrative surface heat fluxes",
        "units": "degree_C m3 s-1",
    }

    return G_nonshortwave_surface_flux


def diff_2d_flux_llc90(grid, Fx, Fy):
    """Computes 2D flux divergences on an LLC90 grid.

    Why this exists:
      `xgcm.Grid.diff_2d_vector` (and related vector operators) rely on face-connection metadata. 
      Ideally, `xgcm.Grid.diff_2d_vector` could be used to calculate 
      divergences/convergences on LLC-type grids, but at the moment this is not available. 
      For now, we can use this function which is dask-native, and topology-correct for LLC90 grids.

    Args:
      grid: xgcm.Grid instance. Used to infer axis coordinate names and access center
        coordinates via `grid._ds`.
      Fx: X-face flux `xarray.DataArray` (e.g., `UVELMASS * dyG * drF`). Must be staggered
        on exactly one non-center X dimension (e.g., `i_g`) and include the LLC face
        dimension inferred from `grid._face_connections`.
      Fy: Y-face flux `xarray.DataArray` (e.g., `VVELMASS * dxG * drF`). Must be staggered
        on exactly one non-center Y dimension (e.g., `j_g`) and include the LLC face
        dimension inferred from `grid._face_connections`.

    Returns:
      Dict with keys {"X","Y"}:
        - out["X"] is the X-direction divergence contribution on C-points.
        - out["Y"] is the Y-direction divergence contribution on C-points.

      Combine as:
        div = out["X"] + out["Y"]
        conv = -(out["X"] + out["Y"])

    Raises:
      ValueError: If the LLC face dimension cannot be inferred uniquely from
        `grid._face_connections`, or if Fx/Fy staggering is ambiguous.
    """
    keys = list(getattr(grid, "_face_connections", {}).keys())
    if len(keys) != 1:
        raise ValueError("Could not infer a unique LLC face dimension from grid._face_connections.")
    face_dim = keys[0]

    ds = grid._ds

    def _center_dim(axis):
        return grid.axes[axis].coords["center"]

    def _staggered_dim(axis, da):
        candidates = [
            c for pos, c in grid.axes[axis].coords.items()
            if pos != "center" and c in da.dims
        ]
        if len(candidates) != 1:
            raise ValueError("Flux difference inconsistent with finite volume discretization.")
        return candidates[0]

    Xc, Yc = _center_dim("X"), _center_dim("Y")
    Xs = _staggered_dim("X", Fx)
    Ys = _staggered_dim("Y", Fy)

    xs_new = int(Fx[Xs].isel({Xs: -1}).values) + 1
    ys_new = int(Fy[Ys].isel({Ys: -1}).values) + 1

    faces = Fx[face_dim].values

    def x_neighbors(face):
        """Returns the +X edge slice for Fx on `face`, stitched from the LLC90 neighbor."""
        f = int(face)

        if 0 <= f <= 2:
            g = Fx.sel({face_dim: f + 3}).isel({Xs: 0})
        elif 3 <= f <= 5:
            g = (
                Fy.sel({face_dim: 12 - f})
                .isel({Ys: 0, Xc: slice(None, None, -1)})
                .rename({Xc: Yc})
                .assign_coords({Yc: ds[Yc]})
            )
        elif f == 6:
            g = Fx.sel({face_dim: 7}).isel({Xs: 0})
        elif 7 <= f <= 8:
            g = Fx.sel({face_dim: f + 1}).isel({Xs: 0})
        elif 10 <= f <= 11:
            g = Fx.sel({face_dim: f + 1}).isel({Xs: 0})
        else:
            g = xr.full_like(Fx.sel({face_dim: f}).isel({Xs: 0}), np.nan)

        return g.expand_dims({face_dim: [f], Xs: [xs_new]})

    def y_neighbors(face):
        """Returns the +Y edge slice for Fy on `face`, stitched from the LLC90 neighbor."""
        f = int(face)

        if 0 <= f <= 1:
            g = Fy.sel({face_dim: f + 1}).isel({Ys: 0})
        elif f == 2:
            g = (
                Fx.sel({face_dim: 6})
                .isel({Xs: 0, Yc: slice(None, None, -1)})
                .rename({Yc: Xc})
                .assign_coords({Xc: ds[Xc]})
            )
        elif 3 <= f <= 5:
            g = Fy.sel({face_dim: f + 1}).isel({Ys: 0})
        elif f == 6:
            g = (
                Fx.sel({face_dim: 10})
                .isel({Xs: 0, Yc: slice(None, None, -1)})
                .rename({Yc: Xc})
                .assign_coords({Xc: ds[Xc]})
            )
        elif 7 <= f <= 9:
            g = Fy.sel({face_dim: f + 3}).isel({Ys: 0})
        elif 10 <= f <= 12:
            g = (
                Fx.sel({face_dim: 12 - f})
                .isel({Xs: 0, Yc: slice(None, None, -1)})
                .rename({Yc: Xc})
                .assign_coords({Xc: ds[Xc]})
            )
        else:
            g = xr.full_like(Fy.sel({face_dim: f}).isel({Ys: 0}), np.nan)

        return g.expand_dims({face_dim: [f], Ys: [ys_new]})

    gx = xr.concat(
        [x_neighbors(f) for f in faces],
        dim=face_dim,
        coords="minimal",
        compat="override",
        join="override",
    )
    gy = xr.concat(
        [y_neighbors(f) for f in faces],
        dim=face_dim,
        coords="minimal",
        compat="override",
        join="override",
    )

    Fx_p = xr.concat([Fx, gx], dim=Xs, coords="minimal", compat="override", join="override").chunk({Xs: -1})
    Fy_p = xr.concat([Fy, gy], dim=Ys, coords="minimal", compat="override", join="override").chunk({Ys: -1})

    dFx = Fx_p.diff(Xs).rename({Xs: Xc}).assign_coords({Xc: ds[Xc]})
    dFy = Fy_p.diff(Ys).rename({Ys: Yc}).assign_coords({Yc: ds[Yc]})

    return {"X": dFx, "Y": dFy}