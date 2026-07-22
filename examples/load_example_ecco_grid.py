"""Load the ECCO V4r4 (LLC90) example budget dataset used by the ECCO notebooks.

The data live on Zenodo (DOI `10.5281/zenodo.21479854`) as one netCDF per ECCO
PO.DAAC collection — monthly-mean 3-D volume/heat/salt fluxes, monthly surface
heat and freshwater fluxes, GM (bolus) velocities, month-boundary snapshots of
temperature/salinity/sea-surface-height, the grid geometry, and the geothermal
heat-flux input — covering the full year **2010** on the native 13-tile LLC90
grid. Together they carry every diagnostic the `ECCOV4r4_native` recipe needs to
close the mass, heat, and salt budgets.

This module downloads those collections (caching them under `../data`), assembles
them into the single budget-ready dataset the recipe expects, and builds the
face-connected `xgcm.Grid`.
"""
import os
import shutil
import urllib.request

import xarray as xr
import xgcm

# Zenodo record for the ECCO V4r4 (LLC90) example dataset.
# Concept DOI 10.5281/zenodo.21051424; this version 10.5281/zenodo.21479854.
ZENODO_RECORD = "21479854"
_BASE_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files"

GRID_FILE = "GRID_GEOMETRY_ECCO_V4r4_native_llc0090.nc"
GEOTHERMAL_FILE = "GEOTHERMAL_FLUX_ECCO_V4r4_native_llc0090.nc"

# Monthly-mean collections (one file each), all on the 12 months of 2010. The
# comment on each names the recipe diagnostics it supplies.
MONTHLY_FILES = [
    "OCEAN_TEMPERATURE_SALINITY_mon_mean_2010_ECCO_V4r4_native_llc0090.nc",       # THETA, SALT (tracer lambdas)
    "OCEAN_3D_VOLUME_FLUX_mon_mean_2010_ECCO_V4r4_native_llc0090.nc",             # UVELMASS, VVELMASS, WVELMASS
    "OCEAN_3D_TEMPERATURE_FLUX_mon_mean_2010_ECCO_V4r4_native_llc0090.nc",        # ADV{x,y,r}_TH, DF{xE,yE,rE,rI}_TH
    "OCEAN_3D_SALINITY_FLUX_mon_mean_2010_ECCO_V4r4_native_llc0090.nc",           # ADV{x,y,r}_SLT, DF*_SLT, oceSPtnd
    "OCEAN_AND_ICE_SURFACE_HEAT_FLUX_mon_mean_2010_ECCO_V4r4_native_llc0090.nc",  # TFLUX, oceQsw
    "OCEAN_AND_ICE_SURFACE_FW_FLUX_mon_mean_2010_ECCO_V4r4_native_llc0090.nc",    # SFLUX, oceFWflx
    "OCEAN_BOLUS_VELOCITY_mon_mean_2010_ECCO_V4r4_native_llc0090.nc",             # UVELSTAR, VVELSTAR, WVELSTAR
]

# Month-boundary snapshot collections (13 instants, 2010-01-01 … 2011-01-01).
# The engine differences these across the T axis to form each month's Eulerian
# tendency, so their `time` axis becomes the outer T coordinate `time_bounds`
# and each field is suffixed `_bounds` (THETA -> THETA_bounds, …).
SNAPSHOT_FILES = {
    "OCEAN_TEMPERATURE_SALINITY_snap_2010_ECCO_V4r4_native_llc0090.nc": ["THETA", "SALT"],
    "SEA_SURFACE_HEIGHT_snap_2010_ECCO_V4r4_native_llc0090.nc": ["ETAN"],
}


def download_ECCOV4r4_example_data(file_name, data_dir="../data"):
    """Download one collection file from Zenodo into `data_dir` (cached)."""
    destination_path = os.path.join(data_dir, file_name)
    if not os.path.exists(destination_path):
        os.makedirs(data_dir, exist_ok=True)
        print(f"File '{file_name}' being downloaded to {destination_path}.")
        with urllib.request.urlopen(f"{_BASE_URL}/{file_name}") as response, \
                open(destination_path, "wb") as out_file:
            shutil.copyfileobj(response, out_file)
        print(f"File '{file_name}' has completed download to {destination_path}.")
    else:
        print(f"File '{file_name}' already exists at {destination_path}. Skipping download.")
    return destination_path


def _load_snapshot_bounds(data_dir):
    """Snapshot fields -> `<VAR>_bounds` on the outer T coordinate `time_bounds`."""
    bounds = []
    for file_name, varnames in SNAPSHOT_FILES.items():
        ds = xr.open_dataset(download_ECCOV4r4_example_data(file_name, data_dir))
        for v in varnames:
            bounds.append(
                ds[v].rename({"time": "time_bounds"}).rename(f"{v}_bounds")
            )
    return xr.merge(bounds)


def load_ECCOV4r4_budget_diagnostics(data_dir="../data"):
    """Assemble the full ECCO V4r4 (2010, LLC90) budget dataset and build its grid.

    Downloads (once) and merges the grid geometry, the seven monthly-mean
    collections, the two month-boundary snapshot collections, and the geothermal
    input into a single dataset carrying every `ECCOV4r4_native` diagnostic, then
    returns the face-connected `xgcm.Grid`.
    """
    grid_ds = xr.open_dataset(download_ECCOV4r4_example_data(GRID_FILE, data_dir))
    monthly = xr.merge(
        [
            xr.open_dataset(
                download_ECCOV4r4_example_data(f, data_dir), chunks={}
            )
            for f in MONTHLY_FILES
        ],
        compat="override",
    )
    bounds = _load_snapshot_bounds(data_dir)
    geothermal = xr.open_dataset(download_ECCOV4r4_example_data(GEOTHERMAL_FILE, data_dir))

    ds = xr.merge([grid_ds, monthly, bounds, geothermal], compat="override").fillna(0.0)
    return construct_grid(ds)


def construct_grid(ds):
    # define the connectivity between faces
    face_connections = {'tile':
                        {0: {'X':  ((12, 'Y', False), (3, 'X', False)),
                             'Y':  (None,             (1, 'Y', False))},
                         1: {'X':  ((11, 'Y', False), (4, 'X', False)),
                             'Y':  ((0, 'Y', False),  (2, 'Y', False))},
                         2: {'X':  ((10, 'Y', False), (5, 'X', False)),
                             'Y':  ((1, 'Y', False),  (6, 'X', False))},
                         3: {'X':  ((0, 'X', False),  (9, 'Y', False)),
                             'Y':  (None,             (4, 'Y', False))},
                         4: {'X':  ((1, 'X', False),  (8, 'Y', False)),
                             'Y':  ((3, 'Y', False),  (5, 'Y', False))},
                         5: {'X':  ((2, 'X', False),  (7, 'Y', False)),
                             'Y':  ((4, 'Y', False),  (6, 'Y', False))},
                         6: {'X':  ((2, 'Y', False),  (7, 'X', False)),
                             'Y':  ((5, 'Y', False),  (10, 'X', False))},
                         7: {'X':  ((6, 'X', False),  (8, 'X', False)),
                             'Y':  ((5, 'X', False),  (10, 'Y', False))},
                         8: {'X':  ((7, 'X', False),  (9, 'X', False)),
                             'Y':  ((4, 'X', False),  (11, 'Y', False))},
                         9: {'X':  ((8, 'X', False),  None),
                             'Y':  ((3, 'X', False),  (12, 'Y', False))},
                         10: {'X': ((6, 'Y', False),  (11, 'X', False)),
                              'Y': ((7, 'Y', False),  (2, 'X', False))},
                         11: {'X': ((10, 'X', False), (12, 'X', False)),
                              'Y': ((8, 'Y', False),  (1, 'X', False))},
                         12: {'X': ((11, 'X', False), None),
                              'Y': ((9, 'Y', False),  (0, 'X', False))}}}

    coords = {
        "X": {"center": "i", "left": "i_g"},
        "Y": {"center": "j", "left": "j_g"},
        "T": {"center": "time", "outer": "time_bounds"},
        "Z": {"center": "k", "left": "k_l"},
    }

    metrics = {
        ("X",): ["dxG"],          # distances between two X faces
        ("Y",): ["dyG"],          # distances between two Y faces
        ("Z",): ["drF"],  # 1D Z distances between cell_boundaries
        ("X", "Y"): ["rA", "rAw", "rAs"],            # horizontal areas (cell center, west-face, south-face)
    }

    # xgcm >= 0.10 replaced `boundary` with `padding` and removed `periodic`;
    # the previous `periodic=False` is equivalent to padding="fill". X and Y
    # only pad at the outer edges of the LLC tiling, since `face_connections`
    # supplies the halo everywhere the 13 tiles meet.
    padding = {"X": "fill", "Y": "fill", "Z": "fill", "T": "fill"}
    # Set for every axis, not just Z: xgcm's default fill_value is changing from
    # 0.0 to nan, and these budgets rely on padded edges contributing zero.
    fill_value = 0.0

    grid = xgcm.Grid(
        ds,
        coords=coords,
        metrics=metrics,
        padding=padding,
        fill_value=fill_value,
        face_connections=face_connections,
        autoparse_metadata=False,
    )
    return grid
