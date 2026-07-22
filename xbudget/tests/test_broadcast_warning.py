"""Regression tests for issue #11: summing a 2D surface flux with a 3D flux
convergence silently broadcasts the surface flux over every vertical level.

The engine now emits a ``UserWarning`` when a ``sum`` mixes operands of
differing dimensionality, so the wrong finite-volume broadcast is at least
loud rather than silent. A well-formed budget whose summands all share the same
dimensions must stay quiet.
"""
import copy
import warnings

import numpy as np
import pytest
import xarray as xr
import xgcm

import xbudget
from xbudget.collect import _warn_if_summands_broadcast

from conftest import SYNTHETIC_PRESET


def _mixed_rank_grid():
    """A grid with a 3D flux convergence and a 2D surface flux on cell centers."""
    ds = xr.Dataset(
        {
            "conv3d": xr.DataArray(np.ones((2, 4, 3)), dims=("z", "x_c", "y_c")),
            "surf2d": xr.DataArray(np.ones((4, 3)) * 10.0, dims=("x_c", "y_c")),
        },
        coords={
            "z": np.arange(2),
            "x_c": np.arange(4) + 0.5,
            "y_c": np.arange(3),
        },
    ).chunk()
    return xgcm.Grid(
        ds,
        coords={"X": {"center": "x_c"}},
        padding="fill",
        autoparse_metadata=False,
    )


MIXED_RANK_PRESET = {
    "heat": {
        "rhs": {
            "var": None,
            "sum": {
                "var": None,
                "convergence": {"var": "conv3d"},
                "surface": {"var": "surf2d"},
            },
        }
    }
}


# -- the helper in isolation ------------------------------------------------


def test_helper_warns_on_mismatched_dims():
    a = xr.DataArray(np.ones((2, 3, 4)), dims=("z", "y", "x"))
    b = xr.DataArray(np.ones((3, 4)), dims=("y", "x"))
    with pytest.warns(UserWarning, match="mismatched dimensions"):
        _warn_if_summands_broadcast([a, b], "some_term")


def test_helper_quiet_on_matching_dims_and_scalars():
    a = xr.DataArray(np.ones((3, 4)), dims=("y", "x"))
    b = xr.DataArray(np.ones((3, 4)), dims=("y", "x"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning here fails the test
        _warn_if_summands_broadcast([a, b, -1.0, 1035.0], "some_term")
        _warn_if_summands_broadcast([a], "lone_term")


# -- end to end -------------------------------------------------------------


def test_mixed_rank_sum_warns():
    grid = _mixed_rank_grid()
    with pytest.warns(UserWarning, match="mismatched dimensions"):
        xbudget.collect_budgets(grid, copy.deepcopy(MIXED_RANK_PRESET))


def test_wellformed_sum_is_quiet(synthetic_grid):
    """The all-2D synthetic preset must not trip the broadcast warning."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        xbudget.collect_budgets(synthetic_grid, copy.deepcopy(SYNTHETIC_PRESET))
    broadcast = [w for w in record if "mismatched dimensions" in str(w.message)]
    assert broadcast == []
