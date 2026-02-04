import pytest
import numpy as np
import xarray as xr
import copy
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


class TestAggregate:
    """Test the aggregate function"""

    def test_aggregate_basic(self):
        """Test basic aggregation"""
        xbudget_dict = {
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
        result = aggregate(xbudget_dict)
        assert result["heat"]["rhs"] == {"advection": "advective_tendency"}

    def test_aggregate_with_decompose(self):
        """Test aggregation with decompose parameter"""
        xbudget_dict = {
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
        result = aggregate(xbudget_dict, decompose="advection")
        assert "advection_horizontal" in result["heat"]["rhs"]
        assert "advection_vertical" in result["heat"]["rhs"]

    def test_aggregate_doesnt_modify_original(self):
        """Test that aggregate doesn't modify the original dictionary"""
        xbudget_dict = {
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
        original_copy = copy.deepcopy(xbudget_dict)
        aggregate(xbudget_dict)
        assert xbudget_dict == original_copy

    def test_aggregate_with_both_lhs_rhs(self):
        """Test aggregation with both lhs and rhs"""
        xbudget_dict = {
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
        result = aggregate(xbudget_dict)
        assert "tendency" in result["heat"]["lhs"]
        assert "advection" in result["heat"]["rhs"]


class TestGetVars:
    """Test the get_vars and _get_vars functions"""

    def test_get_vars_simple(self):
        """Test get_vars with simple variable"""
        xbudget_dict = {
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
        result = get_vars(xbudget_dict, "advective_tendency")
        assert result["var"] == "advective_tendency"

    def test_get_vars_list_input(self):
        """Test get_vars with list of terms"""
        xbudget_dict = {
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
        result = get_vars(xbudget_dict, ["advective_tendency", "diffusive_tendency"])
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
        
        xbudget_dict = {
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
        
        collect_budgets(ds, xbudget_dict)
        assert "heat_rhs_sum_forcing" in ds
        assert "heat_rhs_sum" in ds
        assert "heat_rhs" in ds
        

    def test_collect_budgets_with_lhs_rhs(self):
        """Test budget collection with both lhs and rhs"""
        ds = xr.Dataset({
            "tendency_diag": xr.DataArray(np.random.rand(3, 3), dims=("x", "y")),
            "forcing_diag":  xr.DataArray(np.random.rand(3, 3), dims=("x", "y")),
        }, coords={"x": [0, 1, 2], "y": [0, 1, 2]})
        
        xbudget_dict = {
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
        
        collect_budgets(ds, xbudget_dict)
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
        
        xbudget_dict = {
            "var": None,
            "sum": {
                "advection": {"var": "advection"},
                "diffusion": {"var": "diffusion"},
                "var": None,
            }
        }
        
        result = budget_fill_dict(ds, xbudget_dict, "heat_rhs")
        assert result is not None
        assert "heat_rhs_sum" in ds

    def test_budget_fill_dict_product_operation(self):
        """Test budget_fill_dict with product operation"""
        ds = xr.Dataset({
            "coeff": xr.DataArray(2.0 * np.ones((3, 3)), dims=("x", "y")),
            "var": xr.DataArray(3.0 * np.ones((3, 3)), dims=("x", "y")),
        }, coords={"x": [0, 1, 2], "y": [0, 1, 2]})
        
        xbudget_dict = {
            "var": None,
            "product": {
                "coeff": {"var": "coeff"},
                "var_part": {"var": "var"},
                "var": None,
            }
        }
        
        result = budget_fill_dict(ds, xbudget_dict, "heat_rhs")
        assert result is not None
        assert "heat_rhs_product" in ds
        # Check that product is correct (2.0 * 3.0 = 6.0)
        assert np.allclose(ds["heat_rhs_product"].values, 6.0)

    def test_budget_fill_dict_missing_variable_warning(self):
        """Test that missing variables generate warnings"""
        ds = xr.Dataset({
            "advection": xr.DataArray(np.ones((3, 3)), dims=("x", "y")),
        }, coords={"x": [0, 1, 2], "y": [0, 1, 2]})
        
        xbudget_dict = {
            "var": None,
            "sum": {
                "advection": {"var": "advection"},
                "missing_var": {"var": "missing_var"},
                "var": None,
            }
        }
        
        with pytest.warns(UserWarning):
            budget_fill_dict(ds, xbudget_dict, "heat_rhs")

    def test_budget_fill_dict_numeric_values(self):
        """Test budget_fill_dict with numeric values"""
        ds = xr.Dataset({
            "var": xr.DataArray(np.ones((3, 3)), dims=("x", "y")),
        }, coords={"x": [0, 1, 2], "y": [0, 1, 2]})
        
        xbudget_dict = {
            "var": None,
            "product": {
                "factor": 2.0,
                "var_part": {"var": "var"},
                "var": None,
            }
        }
        
        result = budget_fill_dict(ds, xbudget_dict, "heat_rhs")
        assert result is not None
        assert np.allclose(ds["heat_rhs_product"].values, 2.0)