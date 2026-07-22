import pytest
import numpy as np
import xarray as xr
import copy
import xgcm
import dask.array as da
from xbudget.collect import (
    aggregate,
    disaggregate,
    deep_search,
    _deep_search,
    collect_budgets,
    budget_fill_dict,
    get_vars,
    _get_vars,
    flatten,
    flatten_lol,
)


# These four classes test the *legacy* recipe-reading helpers on purpose,
# so their deprecation warning is expected here rather than informative.
# TestLegacyHelperDeprecation below is what pins the warning itself.
@pytest.mark.filterwarnings("ignore::FutureWarning")
class TestDisaggregate:
    """Test the disaggregate function"""

    def test_disaggregate_basic(self):
        """Test basic disaggregation without decompose"""
        b = {
            "sum": {
                "advection": {"var": "advective_tendency"},
                "var": "heat_rhs_sum",
            },
            "var": "heat_rhs",
        }
        result = disaggregate(b)
        assert result == {"advection": "advective_tendency"}

    def test_disaggregate_with_decompose(self):
        """Test disaggregation with decompose parameter"""
        b = {
            "sum": {
                "advection": {
                    "var": "advective_tendency",
                    "sum": {
                        "horizontal": {"var": "advective_tendency_h"},
                        "vertical": {"var": "advective_tendency_v"},
                        "var": "heat_rhs_sum_advection_sum",
                    },
                },
                "var": "heat_rhs_sum",
            },
            "var": "heat_rhs",
        }
        result = disaggregate(b, decompose="advection")
        assert result == {
            "advection": {
                "horizontal": "advective_tendency_h",
                "vertical": "advective_tendency_v",
            }
        }

    def test_disaggregate_no_sum(self):
        """Test disaggregation when no sum key exists"""
        b = {"var": "some_variable"}
        result = disaggregate(b)
        assert result == {"var": "some_variable"}

    def test_disaggregate_with_none_values(self):
        """Test disaggregation ignores None values"""
        b = {
            "sum": {
                "advection": {"var": "advective_tendency"},
                "diffusion": None,
                "var": "heat_rhs_sum",
            },
            "var": "heat_rhs",
        }
        result = disaggregate(b)
        assert "diffusion" not in result


@pytest.mark.filterwarnings("ignore::FutureWarning")
class TestDeepSearch:
    """Test the deep_search and _deep_search functions"""

    def test_deep_search_simple_dict(self):
        """Test deep_search with simple nested dictionary"""
        b = {"advection": "advective_tendency"}
        result = deep_search(b)
        assert result == {"advection": "advective_tendency"}

    def test_deep_search_nested_dict(self):
        """Test deep_search with deeply nested dictionary"""
        b = {
            "advection": {
                "horizontal": "advective_tendency_h",
                "vertical": "advective_tendency_v",
            }
        }
        result = deep_search(b)
        assert result == {
            "advection_horizontal": "advective_tendency_h",
            "advection_vertical": "advective_tendency_v",
        }

    def test_deep_search_string_input(self):
        """Test deep_search with string input"""
        result = deep_search("variable_name")
        assert result == None

    def test_deep_search_complex_nesting(self):
        """Test deep_search with complex nested structure"""
        b = {
            "heat": {
                "rhs": {
                    "advection": {
                        "horizontal": "adv_h",
                        "vertical": "adv_v",
                    }
                }
            }
        }
        result = deep_search(b)
        assert result == {
            "heat_rhs_advection_horizontal": "adv_h",
            "heat_rhs_advection_vertical": "adv_v",
        }


@pytest.mark.filterwarnings("ignore::FutureWarning")
class TestAggregate:
    """Test the aggregate function"""

    def test_aggregate_basic(self):
        """Test basic aggregation"""
        recipe = {
            "heat": {
                "rhs": {
                    "sum": {
                        "advection": {"var": "advective_tendency"},
                        "var": "heat_rhs_sum",
                    },
                    "var": "heat_rhs",
                }
            }
        }
        result = aggregate(recipe)
        assert result["heat"]["rhs"] == {"advection": "advective_tendency"}

    def test_aggregate_with_decompose(self):
        """Test aggregation with decompose parameter"""
        recipe = {
            "heat": {
                "rhs": {
                    "sum": {
                        "advection": {
                            "var": "advective_tendency",
                            "sum": {
                                "horizontal": {"var": "advective_tendency_h"},
                                "vertical": {"var": "advective_tendency_v"},
                                "var": "heat_rhs_sum_advection_sum",
                            },
                        },
                        "var": "heat_rhs_sum",
                    },
                    "var": "heat_rhs",
                }
            }
        }
        result = aggregate(recipe, decompose="advection")
        assert "advection_horizontal" in result["heat"]["rhs"]
        assert "advection_vertical" in result["heat"]["rhs"]

    def test_aggregate_doesnt_modify_original(self):
        """Test that aggregate doesn't modify the original dictionary"""
        recipe = {
            "heat": {
                "rhs": {
                    "sum": {
                        "advection": {"var": "advective_tendency"},
                        "var": "heat_rhs_sum",
                    },
                    "var": "heat_rhs",
                }
            }
        }
        original_copy = copy.deepcopy(recipe)
        aggregate(recipe)
        assert recipe == original_copy

    def test_aggregate_with_both_lhs_rhs(self):
        """Test aggregation with both lhs and rhs"""
        recipe = {
            "heat": {
                "lhs": {
                    "sum": {
                        "tendency": {"var": "tendency_var"},
                        "var": "heat_lhs_sum",
                    },
                    "var": "heat_lhs",
                },
                "rhs": {
                    "sum": {
                        "advection": {"var": "advective_tendency"},
                        "var": "heat_rhs_sum",
                    },
                    "var": "heat_rhs",
                },
            }
        }
        result = aggregate(recipe)
        assert "tendency" in result["heat"]["lhs"]
        assert "advection" in result["heat"]["rhs"]


@pytest.mark.filterwarnings("ignore::FutureWarning")
class TestGetVars:
    """Test the get_vars and _get_vars functions"""

    def test_get_vars_simple(self):
        """Test get_vars with simple variable"""
        recipe = {
            "heat": {
                "rhs": {
                    "sum": {
                        "advection": {"var": "advective_tendency"},
                        "var": "heat_rhs_sum",
                    },
                    "var": "heat_rhs",
                }
            }
        }
        result = get_vars(recipe, "advective_tendency")
        assert result["var"] == "advective_tendency"

    def test_get_vars_list_input(self):
        """Test get_vars with list of terms"""
        recipe = {
            "heat": {
                "rhs": {
                    "sum": {
                        "advection": {"var": "advective_tendency"},
                        "diffusion": {"var": "diffusive_tendency"},
                        "var": "heat_rhs_sum",
                    },
                    "var": "heat_rhs",
                }
            }
        }
        result = get_vars(recipe, ["advective_tendency", "diffusive_tendency"])
        assert isinstance(result, list)
        assert len(result) == 2


class TestFlatten:
    """Test the flatten and flatten_lol functions"""

    def test_flatten_simple_list(self):
        """Test flatten with simple list"""
        result = list(flatten([1, 2, 3]))
        assert result == [1, 2, 3]

    def test_flatten_nested_lists(self):
        """Test flatten with nested lists"""
        result = list(flatten([1, [2, 3], 4]))
        assert result == [1, 2, 3, 4]

    def test_flatten_deeply_nested(self):
        """Test flatten with deeply nested lists"""
        result = list(flatten([1, [2, [3, 4]], 5]))
        assert result == [1, 2, 3, 4, 5]

    def test_flatten_lol_simple(self):
        """Test flatten_lol with simple list of lists"""
        result = flatten_lol([[1, 2], [3, 4]])
        assert result == [1, 2, 3, 4]

    def test_flatten_lol_mixed(self):
        """Test flatten_lol with mixed nesting"""
        result = flatten_lol([[1, [2, 3]], [4, 5]])
        assert result == [1, 2, 3, 4, 5]

    def test_flatten_lol_empty(self):
        """Test flatten_lol with empty list"""
        result = flatten_lol([])
        assert result == []


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
                        "var": None,
                    },
                    "var": None,
                }
            }
        }
        
        collect_budgets(ds, recipe)
        # New scheme: one variable per node, operator infixes dropped.
        assert "heat_rhs_forcing" in ds
        assert "heat_rhs" in ds
        # The redundant operator-suffixed names are gone by default.
        assert "heat_rhs_sum" not in ds
        assert "heat_rhs_sum_forcing" not in ds

    def test_collect_budgets_legacy_name_scheme(self):
        """name_scheme='legacy' reproduces the historical names and fills the dict."""
        ds = xr.Dataset({
            "forcing_diag": xr.DataArray(np.random.rand(3, 3), dims=("x", "y")),
        }, coords={"x": [0, 1, 2], "y": [0, 1, 2]})
        recipe = {
            "heat": {"rhs": {"sum": {"forcing": {"var": "forcing_diag"}, "var": None}, "var": None}}
        }
        with pytest.warns(FutureWarning, match="name_scheme"):
            collect_budgets(ds, recipe, name_scheme="legacy")
        # Historical variable names are produced...
        assert "heat_rhs_sum_forcing" in ds
        assert "heat_rhs_sum" in ds
        assert "heat_rhs" in ds
        # ...the simplified names are not (legacy mode is faithful to the old engine)...
        assert "heat_rhs_forcing" not in ds
        # ...and the recipe dict is filled in place (get_vars/aggregate rely on this).
        assert recipe["heat"]["rhs"]["var"] == "heat_rhs"

    def test_collect_budgets_does_not_mutate_recipe(self):
        """collect_budgets must not mutate the input recipe dict."""
        import copy as _copy
        ds = xr.Dataset({
            "forcing_diag": xr.DataArray(np.random.rand(3, 3), dims=("x", "y")),
        }, coords={"x": [0, 1, 2], "y": [0, 1, 2]})
        recipe = {
            "heat": {"rhs": {"sum": {"forcing": {"var": "forcing_diag"}, "var": None}, "var": None}}
        }
        original = _copy.deepcopy(recipe)
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
                        "var": None,
                    },
                    "var": None,
                },
                "rhs": {
                    "sum": {
                        "advection": {"var": "forcing_diag"},
                        "var": None,
                    },
                    "var": None,
                }
            }
        }
        
        collect_budgets(ds, recipe)
        assert "heat_lhs" in ds
        assert "heat_rhs" in ds


class TestBudgetFillDict:
    """Test the budget_fill_dict function"""

    def test_budget_fill_dict_sum_operation(self):
        """Test budget_fill_dict with sum operation"""
        ds = xr.Dataset({
            "advection": xr.DataArray(np.ones((3, 3)), dims=("x", "y")),
            "diffusion": xr.DataArray(np.ones((3, 3)), dims=("x", "y")),
        }, coords={"x": [0, 1, 2], "y": [0, 1, 2]})
        
        recipe = {
            "var": None,
            "sum": {
                "advection": {"var": "advection"},
                "diffusion": {"var": "diffusion"},
                "var": None,
            }
        }
        
        result = budget_fill_dict(ds, recipe, "heat_rhs")
        assert result is not None
        assert "heat_rhs_sum" in ds

    def test_budget_fill_dict_product_operation(self):
        """Test budget_fill_dict with product operation"""
        ds = xr.Dataset({
            "coeff": xr.DataArray(2.0 * np.ones((3, 3)), dims=("x", "y")),
            "var": xr.DataArray(3.0 * np.ones((3, 3)), dims=("x", "y")),
        }, coords={"x": [0, 1, 2], "y": [0, 1, 2]})
        
        recipe = {
            "var": None,
            "product": {
                "coeff": {"var": "coeff"},
                "var_part": {"var": "var"},
                "var": None,
            }
        }
        
        result = budget_fill_dict(ds, recipe, "heat_rhs")
        assert result is not None
        assert "heat_rhs_product" in ds
        # Check that product is correct (2.0 * 3.0 = 6.0)
        assert np.allclose(ds["heat_rhs_product"].values, 6.0)

    def test_budget_fill_dict_missing_variable_warning(self):
        """Test that missing variables generate warnings"""
        ds = xr.Dataset({
            "advection": xr.DataArray(np.ones((3, 3)), dims=("x", "y")),
        }, coords={"x": [0, 1, 2], "y": [0, 1, 2]})
        
        recipe = {
            "var": None,
            "sum": {
                "advection": {"var": "advection"},
                "missing_var": {"var": "missing_var"},
                "var": None,
            }
        }
        
        with pytest.warns(UserWarning):
            budget_fill_dict(ds, recipe, "heat_rhs")

    def test_budget_fill_dict_numeric_values(self):
        """Test budget_fill_dict with numeric values"""
        ds = xr.Dataset({
            "var": xr.DataArray(np.ones((3, 3)), dims=("x", "y")),
        }, coords={"x": [0, 1, 2], "y": [0, 1, 2]})
        
        recipe = {
            "var": None,
            "product": {
                "factor": 2.0,
                "var_part": {"var": "var"},
                "var": None,
            }
        }
        
        result = budget_fill_dict(ds, recipe, "heat_rhs")
        assert result is not None
        assert np.allclose(ds["heat_rhs_product"].values, 2.0)

    def test_budget_fill_dict_allow_rechunk(self):
        """Test the allow_rechunk option for the difference operation."""
        # Create a dataset with non-uniform chunks on the staggered grid,
        # which would cause issues for xgcm.grid.diff
        flux_data = da.from_array(np.random.rand(5, 3), chunks=((2, 2, 1), 3))
        ds_chunked = xr.Dataset(
            {
                "var": xr.DataArray(
                    flux_data,
                    dims=("x_g", "y_c"),
                )
            },
            coords={
                "x_g": np.arange(5),
                "x_c": np.arange(4) + 0.5,
                "y_c": np.arange(3),
            },
        )

        grid_params = {
            "coords": {"X": {"center": "x_c", "left": "x_g"}},
            "padding": "fill",
            "autoparse_metadata": False,
        }

        recipe = {
            "var": None,
            "difference": {"var_diff": "var", "var": None},
        }

        # 1. Test that allow_rechunk=False raises an error when passing a chunked 
        # dataset through budget_fill_dict
        with pytest.raises(ValueError):
            grid_fail = xgcm.Grid(ds_chunked.copy(deep=True), **grid_params)
            budget_fill_dict(
                grid_fail,
                copy.deepcopy(recipe),
                "tendency_rhs",
                allow_rechunk=False,
            )

        # 2. Test that shows allow_rechunk=True works
        grid_success = xgcm.Grid(ds_chunked.copy(deep=True), **grid_params)
        budget_fill_dict(
            grid_success,
            copy.deepcopy(recipe),
            "tendency_rhs",
            allow_rechunk=True,
        )
        tendency_rechunked = grid_success._ds["tendency_rhs_difference"]

        # 3. Compare with a correct result from an unchunked array
        grid_unchunked = xgcm.Grid(ds_chunked.chunk(-1), **grid_params)
        budget_fill_dict(
            grid_unchunked,
            copy.deepcopy(recipe),
            "tendency_rhs",
            allow_rechunk=False,
        )
        tendency_correct = grid_unchunked._ds["tendency_rhs_difference"]

        # The numerical results should be identical
        xr.testing.assert_allclose(tendency_rechunked, tendency_correct)

    def test_difference_without_grid_raises_value_error(self):
        """A `difference` op without an xgcm.Grid must raise a clear ValueError.

        Regression test: previously the grid guard was misplaced, so passing a
        plain Dataset reached an undefined ``staggered_axes`` and raised an
        opaque NameError instead.
        """
        ds = xr.Dataset(
            {"flux": xr.DataArray(np.random.rand(5), dims=("x_g",))},
            coords={"x_g": np.arange(5)},
        )
        recipe = {"var": None, "difference": {"flux": "flux", "var": None}}

        with pytest.raises(ValueError, match="xgcm.Grid"):
            budget_fill_dict(ds, recipe, "tendency_rhs")

    def test_difference_not_first_term_does_not_raise(self):
        """A `difference` term that is not evaluated first must not raise.

        Regression test: the `else: raise(...must be xgcm.Grid...)` was attached
        to `if var_pref is None` rather than to a grid check, so any difference
        reached after another operation in the same node spuriously errored even
        when a valid grid was supplied.
        """
        ds = xr.Dataset(
            {"flux": xr.DataArray(np.random.rand(5, 3), dims=("x_g", "y_c"))},
            coords={
                "x_g": np.arange(5),
                "x_c": np.arange(4) + 0.5,
                "y_c": np.arange(3),
            },
        )
        grid = xgcm.Grid(
            ds,
            coords={"X": {"center": "x_c", "left": "x_g"}},
            padding="fill",
            autoparse_metadata=False,
        )
        # A node with a `product` (evaluated first, sets the running variable)
        # followed by a `difference` (previously tripped the misplaced raise).
        recipe = {
            "var": None,
            "product": {"var": None, "scale": -1.0, "a": "flux"},
            "difference": {"var": None, "d": "flux"},
        }

        budget_fill_dict(grid, recipe, "tendency_rhs")

        assert "tendency_rhs_product" in grid._ds
        assert "tendency_rhs_difference" in grid._ds


class TestLegacyHelperDeprecation:
    """The recipe-reading query helpers warn, exactly once, and still work.

    "Exactly once" is the point: `disaggregate` recurses and `aggregate` calls
    it, so a warning in the recursive body would fire once per node and train
    users to filter it.
    """

    BUDGET = {
        "heat": {
            "rhs": {
                "sum": {
                    "advection": {"var": "advective_tendency"},
                    "var": "heat_rhs_sum",
                },
                "var": "heat_rhs",
            }
        }
    }

    def test_aggregate_warns_once(self):
        with pytest.warns(FutureWarning, match="BudgetQuery") as record:
            result = aggregate(copy.deepcopy(self.BUDGET))
        assert len(record) == 1
        assert result["heat"]["rhs"] == {"advection": "advective_tendency"}

    def test_aggregate_with_decompose_warns_once(self):
        """The recursive path must not multiply the warning."""
        budget = copy.deepcopy(self.BUDGET)
        budget["heat"]["rhs"]["sum"]["advection"]["sum"] = {
            "horizontal": {"var": "adv_h"},
            "vertical": {"var": "adv_v"},
            "var": "heat_rhs_sum_advection_sum",
        }
        with pytest.warns(FutureWarning) as record:
            aggregate(budget, decompose="advection")
        assert len(record) == 1

    def test_disaggregate_warns_once(self):
        with pytest.warns(FutureWarning, match="BudgetQuery") as record:
            disaggregate(copy.deepcopy(self.BUDGET)["heat"]["rhs"])
        assert len(record) == 1

    def test_deep_search_warns_once(self):
        with pytest.warns(FutureWarning, match="BudgetQuery") as record:
            deep_search({"advection": "advective_tendency"})
        assert len(record) == 1

    def test_get_vars_warns_once(self):
        with pytest.warns(FutureWarning, match="BudgetQuery") as record:
            get_vars(copy.deepcopy(self.BUDGET), "heat_rhs")
        assert len(record) == 1


class TestUnfilledRecipeIsLoud:
    """Querying a recipe the legacy engine never filled must not return {}.

    This is the shape of the downstream breakage in xwmt/xwmb: they call
    collect_budgets (now v1 by default, which does not touch the recipe) and
    then the legacy aggregate, which silently found nothing.
    """

    RECIPE = {
        "heat": {
            "rhs": {
                "var": None,
                "sum": {
                    "var": None,
                    "forcing": {
                        "var": None,
                        "product": {"var": None, "flux": "f", "area": "a"},
                    },
                },
            }
        }
    }

    def _ds(self):
        return xr.Dataset(
            {"f": (("x",), np.ones(3)), "a": (("x",), np.full(3, 2.0))},
            coords={"x": [0, 1, 2]},
        )

    @pytest.mark.filterwarnings("ignore::FutureWarning")
    def test_aggregate_after_v1_run_raises(self):
        """The xwmt pattern: collect with v1, then call the legacy aggregate."""
        d = copy.deepcopy(self.RECIPE)
        collect_budgets(self._ds(), d)  # v1 default: recipe untouched
        with pytest.raises(ValueError, match="BudgetQuery"):
            aggregate(d)

    @pytest.mark.filterwarnings("ignore::FutureWarning")
    def test_aggregate_on_uncollected_recipe_raises(self):
        with pytest.raises(ValueError, match="nothing to report"):
            aggregate(copy.deepcopy(self.RECIPE))

    @pytest.mark.filterwarnings("ignore::FutureWarning")
    def test_aggregate_after_legacy_run_still_works(self):
        d = copy.deepcopy(self.RECIPE)
        collect_budgets(self._ds(), d, name_scheme="legacy")
        assert aggregate(d)["heat"]["rhs"] == {"forcing": "heat_rhs_sum_forcing"}

    @pytest.mark.filterwarnings("ignore::FutureWarning")
    def test_get_vars_miss_on_unfilled_recipe_explains(self):
        with pytest.raises(ValueError, match="BudgetQuery"):
            get_vars(copy.deepcopy(self.RECIPE), "heat_rhs_sum_forcing")

    @pytest.mark.filterwarnings("ignore::FutureWarning")
    def test_recipe_with_no_derived_terms_is_not_unfilled(self):
        """A skeleton recipe (all placeholders, no operations) is not an error.

        MOM6_drift is shaped like this: there is nothing to fill, so there is
        nothing to complain about.
        """
        skeleton = {"heat": {"rhs": {"sum": {"advection": {"var": None}}}}}
        assert aggregate(skeleton)["heat"]["rhs"] == {}
