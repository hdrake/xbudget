"""Tests for the public ``collect_budgets`` entry point (typed engine)."""
import copy

import numpy as np
import pytest
import xarray as xr

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
