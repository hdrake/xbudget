import numpy as np
import xarray as xr
import xgcm

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
