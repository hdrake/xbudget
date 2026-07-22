"""End-to-end characterization ("golden") test for the MOM6 preset.

Pins the exact set of variables that ``collect_budgets`` materializes from the
example MOM6 grid, together with each variable's shape, provenance, and a
numerical fingerprint. This guards the absolute output of the engine.

The example dataset (~600 MB) is downloaded on demand from Zenodo and is not
tracked in git, so this test is skipped when the file is absent (e.g. in CI that
does not fetch it). The small reference fingerprint lives next to this file in
``characterization_MOM6.json`` and *is* tracked.

To regenerate the reference after an intended change::

    XBUDGET_REGEN_CHARN=1 python -m pytest \
        xbudget/tests/test_characterization.py -s
"""
import os
import json
import warnings

import numpy as np
import xarray as xr
import xgcm
import pytest

import xbudget

DATA_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "data",
        "MOM6_global_example_diagnostics_zlevels_v0_0_6.nc",
    )
)
REFERENCE_PATH = os.path.join(
    os.path.dirname(__file__), "characterization_MOM6.json"
)

# Tolerance for the numerical fingerprints. The recipe is deterministic, so the
# only expected drift is last-bit floating point from library version changes.
RTOL = 1e-9


def _build_example_grid():
    """Reconstruct the example xgcm.Grid from the local data file.

    Mirrors ``examples/load_example_model_grid.py`` but reads the already
    downloaded file directly (the example loader uses a relative ``../data``
    path that only resolves from the examples directory).
    """
    ds = xr.open_dataset(DATA_PATH).fillna(0.0)
    if "z_l" not in ds.dims:
        ds = ds.expand_dims(["z_l"]).assign_coords(
            {
                "z_l": xr.DataArray([3000], dims=("z_l",)),
                "z_i": xr.DataArray([0, 6000], dims=("z_i",)),
            }
        )
    # Chunk so the difference/rechunking code path is exercised, matching the
    # example loader.
    ds = ds.chunk({"xh": 100, "yh": 100, "xq": 100, "yq": 100, "time": 1})
    coords = {
        "X": {"center": "xh", "outer": "xq"},
        "Y": {"center": "yh", "outer": "yq"},
    }
    return xgcm.Grid(
        ds,
        coords=coords,
        metrics={("X", "Y"): "areacello"},
        padding={"X": "periodic", "Y": "extend"},
        autoparse_metadata=False,
    )


def _normalize_provenance(prov):
    """Make provenance JSON-comparable (numpy arrays/scalars -> lists/floats)."""
    if isinstance(prov, np.ndarray):
        prov = prov.tolist()
    if isinstance(prov, (list, tuple)):
        return [_normalize_provenance(p) for p in prov]
    if isinstance(prov, np.generic):
        return prov.item()
    return prov


def _fingerprint(ds, created):
    """Build a small, comparable fingerprint for each created variable."""
    fp = {}
    for name in created:
        da = ds[name]
        values = np.asarray(da.values, dtype="float64")
        fp[name] = {
            "dims": list(da.dims),
            "shape": list(da.shape),
            "nansum": float(np.nansum(values)),
            "nanmin": float(np.nanmin(values)) if values.size else 0.0,
            "nanmax": float(np.nanmax(values)) if values.size else 0.0,
            "provenance": _normalize_provenance(da.attrs.get("provenance")),
        }
    return fp


def _collect_example_budgets():
    """Run the MOM6 preset over the example grid; return (ds, created_names)."""
    grid = _build_example_grid()
    ds = grid._ds
    before = set(ds.data_vars)
    budget = xbudget.load_preset_budget("MOM6")
    with warnings.catch_warnings():
        # Missing-diagnostic warnings are expected for the example subset.
        warnings.simplefilter("ignore")
        xbudget.collect_budgets(grid, budget)
    created = sorted(set(ds.data_vars) - before)
    return ds, created


@pytest.mark.skipif(
    not os.path.exists(DATA_PATH),
    reason="example MOM6 dataset not present (download from Zenodo to run)",
)
def test_mom6_characterization_matches_reference():
    ds, created = _collect_example_budgets()
    fingerprint = _fingerprint(ds, created)

    if os.environ.get("XBUDGET_REGEN_CHARN"):
        with open(REFERENCE_PATH, "w") as f:
            json.dump(fingerprint, f, indent=2, sort_keys=True)
        pytest.skip(f"regenerated reference at {REFERENCE_PATH}")

    with open(REFERENCE_PATH) as f:
        reference = json.load(f)

    missing = set(reference) - set(fingerprint)
    extra = set(fingerprint) - set(reference)
    assert not missing, f"variables no longer produced: {sorted(missing)}"
    assert not extra, f"unexpected new variables: {sorted(extra)}"

    for name, ref in reference.items():
        got = fingerprint[name]
        assert got["dims"] == ref["dims"], f"{name}: dims changed"
        assert got["shape"] == ref["shape"], f"{name}: shape changed"
        assert got["provenance"] == ref["provenance"], (
            f"{name}: provenance changed"
        )
        for stat in ("nansum", "nanmin", "nanmax"):
            assert np.isclose(got[stat], ref[stat], rtol=RTOL, atol=0.0), (
                f"{name}: {stat} changed {ref[stat]} -> {got[stat]}"
            )
