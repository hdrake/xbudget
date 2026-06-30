"""Prove the typed evaluator is numerically equivalent to the legacy engine.

The new engine (parse -> evaluate) emits one variable per operation with the
simplified path-based names, collapsing the legacy engine's duplicate "copy"
variables. These tests run both engines on the same data and assert that every
new variable matches its legacy counterpart (mapped via the alias map) to
floating-point tolerance.

Two fixtures are used:
- a tiny synthetic grid that always runs (covers sum/product/difference,
  multi-operation terms, and missing-variable handling);
- the full example MOM6 grid, skipped when the (~600 MB) file is absent.
"""
import copy
import os
import warnings

import numpy as np
import xarray as xr
import xgcm
import pytest

import xbudget
from xbudget.parse import parse_budgets
from xbudget.evaluate import evaluate_budgets

OPS = {"sum", "product", "difference", "reciprocal", "lateral_divergence"}
RTOL = 1e-9


def _data(name):
    return os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data", name)
    )


DATA_PATH = _data("MOM6_global_example_diagnostics_zlevels_v0_0_6.nc")
ECCO_DATA_PATH = _data("ECCO_budget_terms.nc")


def _run_legacy(build_grid, preset):
    """Drive the legacy dict-walking engine (budget_fill_dict) directly.

    ``collect_budgets`` now uses the typed engine, so the equivalence oracle
    must call the legacy ``budget_fill_dict`` itself (the same loop the old
    ``collect_budgets`` performed).
    """
    grid = build_grid()
    ds = grid._ds
    before = set(ds.data_vars)
    preset = copy.deepcopy(preset)  # legacy engine mutates the dict
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for eq, sides in preset.items():
            for side in ("lhs", "rhs"):
                if side in sides:
                    xbudget.budget_fill_dict(grid, sides[side], f"{eq}_{side}")
    return ds, set(ds.data_vars) - before


def _run_new(build_grid, preset):
    grid = build_grid()
    ds = grid._ds
    before = set(ds.data_vars)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        alias_map, records = evaluate_budgets(grid, parse_budgets(preset))
    return ds, set(ds.data_vars) - before, alias_map, records


def _assert_equivalent(build_grid, preset):
    legacy_ds, legacy_new = _run_legacy(build_grid, preset)
    new_ds, new_new, alias_map, records = _run_new(build_grid, preset)

    # Every legacy variable (copies + operator-suffixed actuals) is reachable
    # from the alias map.
    unmapped = legacy_new - set(alias_map)
    assert not unmapped, f"legacy names with no new equivalent: {sorted(unmapped)}"

    # Every new variable matches its legacy counterpart numerically.
    assert set(records), "new engine produced no variables"
    for new_name, rec in records.items():
        legacy_actual = rec["legacy_actual"]
        assert legacy_actual in legacy_ds, (
            f"{new_name}: legacy actual {legacy_actual} missing"
        )
        a = np.asarray(new_ds[new_name].values, dtype="float64")
        b = np.asarray(legacy_ds[legacy_actual].values, dtype="float64")
        assert a.shape == b.shape, f"{new_name}: shape {a.shape} != {b.shape}"
        assert np.allclose(
            np.nan_to_num(a), np.nan_to_num(b), rtol=RTOL, atol=0.0
        ), f"{new_name}: values differ from legacy {legacy_actual}"

    # The new engine emits exactly one variable per legacy "producer": each
    # operator-suffixed actual, plus each leaf var (a bare legacy name with no
    # operator-suffixed sibling). The redundant legacy "copies" are dropped.
    legacy_actuals = {v for v in legacy_new if v.rsplit("_", 1)[-1] in OPS}
    bare_legacy = legacy_new - legacy_actuals
    legacy_leaves = {
        v for v in bare_legacy if not any(f"{v}_{op}" in legacy_new for op in OPS)
    }
    legacy_producers = legacy_actuals | legacy_leaves
    record_targets = {rec["legacy_actual"] for rec in records.values()}
    assert record_targets == legacy_producers, (
        f"new->legacy targets {sorted(record_targets)} != "
        f"legacy producers {sorted(legacy_producers)}"
    )
    # No two new variables collide on a name or a legacy target (injective).
    assert len(record_targets) == len(records)
    return records, alias_map


def _build_synthetic_grid():
    """A small chunked grid exercising sum, product, and difference."""
    rng = np.random.RandomState(0)
    ds = xr.Dataset(
        {
            "flux": xr.DataArray(rng.rand(5, 3), dims=("x_g", "y_c")),
            "diag_a": xr.DataArray(rng.rand(4, 3), dims=("x_c", "y_c")),
            "diag_b": xr.DataArray(rng.rand(4, 3), dims=("x_c", "y_c")),
            "area": xr.DataArray(rng.rand(4, 3) + 1.0, dims=("x_c", "y_c")),
        },
        coords={
            "x_g": np.arange(5),
            "x_c": np.arange(4) + 0.5,
            "y_c": np.arange(3),
        },
    ).chunk({"x_c": 2, "x_g": 2, "y_c": 3})
    return xgcm.Grid(
        ds,
        coords={"X": {"center": "x_c", "left": "x_g"}},
        periodic=False,
        autoparse_metadata=False,
    )


SYNTHETIC_PRESET = {
    "tracer": {
        "rhs": {
            "var": None,
            "sum": {
                "var": None,
                "diffusion": {
                    "var": None,
                    "product": {"var": None, "sign": -1.0, "d": "diag_a", "area": "area"},
                },
                # multi-operation term: a bulk product AND a finer sum
                "boundary": {
                    "var": None,
                    "product": {"var": None, "d": "diag_b", "area": "area"},
                    "sum": {
                        "var": None,
                        "convergence": {
                            "var": None,
                            "difference": {"var": None, "transport": "flux"},
                        },
                    },
                },
                # leaf term: var references an existing diagnostic directly
                "direct": {"var": "diag_a"},
                # references a diagnostic absent from the dataset
                "missing": {"var": None, "product": {"var": None, "d": "not_present"}},
            },
        }
    }
}


def test_equivalent_on_synthetic_grid():
    records, alias_map = _assert_equivalent(_build_synthetic_grid, SYNTHETIC_PRESET)
    # The simplified names drop operator infixes and the redundant copies.
    assert "tracer_rhs" in records
    assert "tracer_rhs_diffusion" in records
    assert "tracer_rhs_boundary" in records  # primary (product)
    assert "tracer_rhs_boundary_sum" in records  # second decomposition
    assert "tracer_rhs_boundary_convergence" in records  # difference leaf
    # leaf term: var references an existing diagnostic directly
    assert "tracer_rhs_direct" in records
    assert records["tracer_rhs_direct"]["op"] == "var"


def _build_mom6_grid():
    ds = xr.open_dataset(DATA_PATH).fillna(0.0)
    if "z_l" not in ds.dims:
        ds = ds.expand_dims(["z_l"]).assign_coords(
            {
                "z_l": xr.DataArray([3000], dims=("z_l",)),
                "z_i": xr.DataArray([0, 6000], dims=("z_i",)),
            }
        )
    ds = ds.chunk({"xh": 100, "yh": 100, "xq": 100, "yq": 100, "time": 1})
    return xgcm.Grid(
        ds,
        coords={"X": {"center": "xh", "outer": "xq"}, "Y": {"center": "yh", "outer": "yq"}},
        metrics={("X", "Y"): "areacello"},
        boundary={"X": "periodic", "Y": "extend"},
        autoparse_metadata=False,
    )


@pytest.mark.skipif(
    not os.path.exists(DATA_PATH),
    reason="example MOM6 dataset not present (download from Zenodo to run)",
)
def test_equivalent_on_mom6_example():
    records, alias_map = _assert_equivalent(
        _build_mom6_grid, xbudget.load_preset_budget("MOM6")
    )
    # 108 legacy variables collapse to 57 operation-named variables.
    assert len(records) == 57
    assert len(alias_map) == 108


# ECCOv4r4 LLC90 face connectivity (13 tiles), mirroring
# examples/load_example_ecco_grid.py.
_ECCO_FACE_CONNECTIONS = {
    "tile": {
        0: {"X": ((12, "Y", False), (3, "X", False)), "Y": (None, (1, "Y", False))},
        1: {"X": ((11, "Y", False), (4, "X", False)), "Y": ((0, "Y", False), (2, "Y", False))},
        2: {"X": ((10, "Y", False), (5, "X", False)), "Y": ((1, "Y", False), (6, "X", False))},
        3: {"X": ((0, "X", False), (9, "Y", False)), "Y": (None, (4, "Y", False))},
        4: {"X": ((1, "X", False), (8, "Y", False)), "Y": ((3, "Y", False), (5, "Y", False))},
        5: {"X": ((2, "X", False), (7, "Y", False)), "Y": ((4, "Y", False), (6, "Y", False))},
        6: {"X": ((2, "Y", False), (7, "X", False)), "Y": ((5, "Y", False), (10, "X", False))},
        7: {"X": ((6, "X", False), (8, "X", False)), "Y": ((5, "X", False), (10, "Y", False))},
        8: {"X": ((7, "X", False), (9, "X", False)), "Y": ((4, "X", False), (11, "Y", False))},
        9: {"X": ((8, "X", False), None), "Y": ((3, "X", False), (12, "Y", False))},
        10: {"X": ((6, "Y", False), (11, "X", False)), "Y": ((7, "Y", False), (2, "X", False))},
        11: {"X": ((10, "X", False), (12, "X", False)), "Y": ((8, "Y", False), (1, "X", False))},
        12: {"X": ((11, "X", False), None), "Y": ((9, "Y", False), (0, "X", False))},
    }
}


def _build_ecco_grid():
    ds = xr.open_dataset(ECCO_DATA_PATH).fillna(0.0)
    return xgcm.Grid(
        ds,
        coords={
            "X": {"center": "i", "left": "i_g"},
            "Y": {"center": "j", "left": "j_g"},
            "T": {"center": "time", "outer": "time_bounds"},
            "Z": {"center": "k", "left": "k_l"},
        },
        metrics={
            ("X",): ["dxG"],
            ("Y",): ["dyG"],
            ("Z",): ["drF"],
            ("X", "Y"): ["rA", "rAw", "rAs"],
        },
        boundary={"X": None, "Y": None, "Z": "fill", "T": None},
        periodic=False,
        fill_value={"Z": 0.0},
        face_connections=_ECCO_FACE_CONNECTIONS,
        autoparse_metadata=False,
    )


@pytest.mark.skipif(
    not os.path.exists(ECCO_DATA_PATH),
    reason="example ECCO LLC90 dataset not present (download from Zenodo to run)",
)
def test_equivalent_on_ecco_native_example():
    """The typed engine reproduces the legacy engine on the LLC90 ECCO budget.

    Exercises reciprocal, difference-of-sub-term, and the native-xgcm
    lateral_divergence on a face-connected (13-tile) grid. Slow (full 1.6 GB
    grid); skipped unless the data file is present.
    """
    records, alias_map = _assert_equivalent(
        _build_ecco_grid, xbudget.load_preset_budget("ECCOV4r4_native")
    )
    assert len(records) == 75
    assert len(alias_map) == 140
    # the (previously dropped) lateral eddy-bolus convergence is materialized
    assert "mass_rhs_advection_lateral_bolus_mass_flux_convergence" in records
