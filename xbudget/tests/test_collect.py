"""Tests for the public ``collect_budgets`` entry point (typed engine)."""
import copy
import warnings

import numpy as np
import pytest
import xarray as xr
import xgcm

from xbudget.collect import collect_budgets


class TestCollectBudgets:
    """Test the collect_budgets function"""

    def test_collect_budgets_basic(self):
        """Test basic budget collection"""
        ds = xr.Dataset({
            "forcing_diag": xr.DataArray(np.random.rand(3, 3), dims=("x", "y")),
        }, coords={"x": [0, 1, 2], "y": [0, 1, 2]})

        recipe = {
            "heat": {
                "rhs": {
                    "sum": {
                        "forcing": {"var": "forcing_diag"},
                    },
                }
            }
        }

        collect_budgets(ds, recipe)
        # One variable per node, operator infixes dropped.
        assert "heat_rhs_forcing" in ds
        assert "heat_rhs" in ds
        # The redundant operator-suffixed names are not produced.
        assert "heat_rhs_sum" not in ds
        assert "heat_rhs_sum_forcing" not in ds

    def test_collect_budgets_does_not_mutate_recipe(self):
        """collect_budgets must not mutate the input recipe dict."""
        ds = xr.Dataset({
            "forcing_diag": xr.DataArray(np.random.rand(3, 3), dims=("x", "y")),
        }, coords={"x": [0, 1, 2], "y": [0, 1, 2]})
        recipe = {
            "heat": {"rhs": {"sum": {"forcing": {"var": "forcing_diag"}}}}
        }
        original = copy.deepcopy(recipe)
        collect_budgets(ds, recipe)
        assert recipe == original

    def test_collect_budgets_with_lhs_rhs(self):
        """Test budget collection with both lhs and rhs"""
        ds = xr.Dataset({
            "tendency_diag": xr.DataArray(np.random.rand(3, 3), dims=("x", "y")),
            "forcing_diag":  xr.DataArray(np.random.rand(3, 3), dims=("x", "y")),
        }, coords={"x": [0, 1, 2], "y": [0, 1, 2]})

        recipe = {
            "heat": {
                "lhs": {
                    "sum": {
                        "tendency": {"var": "tendency_diag"},
                    },
                },
                "rhs": {
                    "sum": {
                        "advection": {"var": "forcing_diag"},
                    },
                }
            }
        }

        collect_budgets(ds, recipe)
        assert "heat_lhs" in ds
        assert "heat_rhs" in ds

    def test_difference_without_grid_raises_value_error(self):
        """A `difference` op without an xgcm.Grid must raise a clear ValueError."""
        ds = xr.Dataset(
            {"flux": xr.DataArray(np.random.rand(5), dims=("x_g",))},
            coords={"x_g": np.arange(5)},
        )
        recipe = {"heat": {"rhs": {"difference": {"flux": "flux"}}}}

        with pytest.raises(ValueError, match="xgcm.Grid"):
            collect_budgets(ds, recipe)


# Two tiles joined along X: a miniature of the ECCO LLC face connections, which
# is what `lateral_divergence` exists for. Small enough to run in CI, where the
# real LLC90 dataset is unavailable.
FACE_CONNECTIONS = {"tile": {
    0: {"X": (None, (1, "X", False))},
    1: {"X": ((0, "X", False), None)},
}}


def _divergence_grid(chunks=None):
    """A small face-connected grid, enough to run a ``lateral_divergence``."""
    rng = np.random.RandomState(0)
    nt, nx, ny = 2, 4, 4
    ds = xr.Dataset(
        {
            "u": xr.DataArray(rng.rand(nt, nx, ny), dims=("tile", "x_g", "y_c")),
            "v": xr.DataArray(rng.rand(nt, nx, ny), dims=("tile", "x_c", "y_g")),
        },
        coords={
            "tile": np.arange(nt),
            "x_g": np.arange(nx), "x_c": np.arange(nx) + 0.5,
            "y_g": np.arange(ny), "y_c": np.arange(ny) + 0.5,
        },
    )
    if chunks is not None:
        ds = ds.chunk(chunks)
    return xgcm.Grid(
        ds,
        coords={
            "X": {"center": "x_c", "left": "x_g"},
            "Y": {"center": "y_c", "left": "y_g"},
        },
        face_connections=FACE_CONNECTIONS,
        padding="fill",
        autoparse_metadata=False,
    )


DIVERGENCE_RECIPE = {
    "mass": {"rhs": {"lateral_divergence": {
        "Fx": {"product": {"flux": "u"}},
        "Fy": {"product": {"flux": "v"}},
    }}}
}


CHUNKS = {"x_c": 2, "x_g": 2, "y_c": 2, "y_g": 2}


class TestLateralDivergenceRechunking:
    """``lateral_divergence`` gets the same rechunking treatment ``difference``
    does: a single chunk along the differenced dimensions while xgcm works, the
    caller's chunking restored afterwards."""

    @staticmethod
    def _run(chunks, allow_rechunk=True):
        grid = _divergence_grid(chunks)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            collect_budgets(grid, DIVERGENCE_RECIPE, allow_rechunk=allow_rechunk)
        return grid._ds["mass_rhs"]

    def test_rechunking_does_not_change_the_result(self):
        chunked = self._run(CHUNKS)
        unchunked = self._run(None)
        np.testing.assert_allclose(np.asarray(chunked), np.asarray(unchunked))

    def test_callers_chunking_is_restored(self):
        out = self._run(CHUNKS)
        # Not left as the single chunk the divergence was computed under.
        assert dict(zip(out.dims, out.chunks))["x_c"] == (2, 2)
        assert dict(zip(out.dims, out.chunks))["y_c"] == (2, 2)
