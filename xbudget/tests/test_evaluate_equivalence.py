"""Prove the typed evaluator is numerically equivalent to the legacy engine.

The new engine (parse -> evaluate) emits one variable per operation with the
simplified path-based names, collapsing the legacy engine's duplicate "copy"
variables. These tests run both engines on the same data and assert that every
new variable matches its legacy counterpart (mapped via the alias map) to
floating-point tolerance.

Three grids are used:
- a tiny synthetic grid that always runs (covers sum/product/difference,
  multi-operation terms, and missing-variable handling), from ``conftest.py``;
- the full example MOM6 grid, skipped when the (~600 MB) file is absent;
- the ECCO LLC90 grid, skipped when the (~1.6 GB) file is absent — the only
  coverage of ``reciprocal``, difference-of-a-sub-term, and face-connected
  ``lateral_divergence``.
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


def test_equivalent_on_synthetic_grid(synthetic_grid_builder, synthetic_preset):
    records, alias_map = _assert_equivalent(synthetic_grid_builder, synthetic_preset)
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
        padding={"X": "periodic", "Y": "extend"},
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
        padding={"X": "fill", "Y": "fill", "Z": "fill", "T": "fill"},
        fill_value=0.0,  # every axis: xgcm's default is changing 0.0 -> nan
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


def _assert_aggregate_equivalent(build_grid, preset, decompose):
    """v1 BudgetQuery.aggregate names the same terms, holding the same data.

    The names differ by design; the arrays behind them must not. This is the
    query-layer analogue of _assert_equivalent, on a real convention.
    """
    legacy_preset = copy.deepcopy(preset)
    legacy_grid = build_grid()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xbudget.collect_budgets(legacy_grid, legacy_preset, name_scheme="legacy")
        legacy_agg = xbudget.aggregate(legacy_preset, decompose=decompose)

    new_grid = build_grid()
    new_preset = copy.deepcopy(preset)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xbudget.collect_budgets(new_grid, new_preset)
    new_agg = xbudget.BudgetQuery(new_grid, new_preset).aggregate(decompose=decompose)

    assert set(legacy_agg) == set(new_agg)
    compared = 0
    for budget in new_agg:
        for side in ("lhs", "rhs"):
            if side not in new_agg[budget]:
                continue
            legacy_side, new_side = legacy_agg[budget][side], new_agg[budget][side]
            assert set(legacy_side) == set(new_side), (
                f"{budget}/{side}: labels {sorted(new_side)} != "
                f"legacy {sorted(legacy_side)}"
            )
            for label, new_name in new_side.items():
                a = np.asarray(new_grid._ds[new_name].values, dtype="float64")
                b = np.asarray(
                    legacy_grid._ds[legacy_side[label]].values, dtype="float64"
                )
                assert np.allclose(
                    np.nan_to_num(a), np.nan_to_num(b), rtol=RTOL, atol=0.0
                ), f"{budget}/{side}/{label} differs from legacy"
                compared += 1
    assert compared, "no terms compared"
    return new_agg


@pytest.mark.skipif(
    not os.path.exists(DATA_PATH),
    reason="example MOM6 dataset not present (download from Zenodo to run)",
)
def test_aggregate_matches_legacy_on_mom6_example():
    """Using the decompose list the MOM6 example notebook actually passes."""
    agg = _assert_aggregate_equivalent(
        _build_mom6_grid,
        xbudget.load_preset_budget("MOM6"),
        decompose=["surface_exchange_flux", "nonadvective", "diffusion"],
    )
    # decompose flattens a term into its parts, keyed parent_child
    assert "surface_exchange_flux_snow" in agg["mass"]["rhs"]
    assert "surface_exchange_flux" not in agg["mass"]["rhs"]


@pytest.mark.skipif(
    not os.path.exists(ECCO_DATA_PATH),
    reason="example ECCO LLC90 dataset not present (download from Zenodo to run)",
)
def test_aggregate_matches_legacy_on_ecco_native_example():
    """Using the decompose list the ECCO decomposition notebook actually passes."""
    _assert_aggregate_equivalent(
        _build_ecco_grid,
        xbudget.load_preset_budget("ECCOV4r4_native"),
        decompose=["advection", "diffusion", "surface_exchange_flux"],
    )
