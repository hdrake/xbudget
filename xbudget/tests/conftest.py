"""Shared synthetic fixtures.

The example MOM6/ECCO datasets are large Zenodo downloads that CI does not have,
so every test gated on them skips there. These small in-memory grids are what
actually exercises the engine and the query layer in CI, so they are built to
cover the same shapes the real recipes use: sums, products, constants, a
difference across a staggered axis, a multi-operation term, a leaf that names a
diagnostic directly, and a term whose diagnostic is absent.
"""
import numpy as np
import pytest
import xarray as xr
import xgcm


def build_synthetic_grid():
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
        padding="fill",
        autoparse_metadata=False,
    )


SYNTHETIC_PRESET = {
    "tracer": {
        "lambda": "tracer_concentration",  # budget metadata, not a side
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


SYNTHETIC_PRESET_SKIPS = {
    "tracer": {
        "rhs": {
            "var": None,
            "sum": {
                "var": None,
                # A term whose *structurally* primary operation (the product,
                # listed first) is skipped at run time because its diagnostic is
                # absent, leaving the sum to claim the bare name. Which operation
                # owns which name is therefore a runtime fact, not a property of
                # the recipe: predicting names from the recipe alone would
                # give this term's sum the name "tracer_rhs_renamed_sum", but the
                # evaluator emits "tracer_rhs_renamed".
                "renamed": {
                    "var": None,
                    "product": {"var": None, "d": "not_present"},
                    "sum": {
                        "var": None,
                        "a": {
                            "var": None,
                            "product": {"var": None, "d": "diag_a"},
                        },
                    },
                },
            },
        }
    }
}


@pytest.fixture
def synthetic_grid():
    return build_synthetic_grid()


@pytest.fixture
def synthetic_grid_builder():
    """The factory itself, for tests needing two independent identical grids."""
    return build_synthetic_grid


@pytest.fixture
def synthetic_preset():
    """A fresh copy for each test."""
    import copy

    return copy.deepcopy(SYNTHETIC_PRESET)


@pytest.fixture
def synthetic_preset_skips():
    import copy

    return copy.deepcopy(SYNTHETIC_PRESET_SKIPS)
