"""Tests for the recipe display layer (``xbudget.display``).

All CI-safe: they use the synthetic preset/grid from ``conftest`` (the real
MOM6/ECCO recipes only get an offline smoke test via ``load_preset_budget``, no
dataset), so they exercise the renderer everywhere the engine tests run.
"""
import warnings

import pytest

import xbudget
from xbudget.display import (
    render_budgets_html,
    render_budgets_text,
    show_recipe,
)
from xbudget.parse import parse_budgets

from conftest import SYNTHETIC_PRESET


@pytest.fixture
def preset():
    import copy

    return copy.deepcopy(SYNTHETIC_PRESET)


class TestShowRecipe:
    def test_html_is_a_collapsible_tree(self, preset):
        view = show_recipe(preset)
        h = view._repr_html_()
        # Native <details>/<summary> collapse, scoped style, no external deps.
        assert "<details" in h and "<summary" in h
        assert 'class="xbdg-wrap"' in h and "<style>" in h
        assert "<script" not in h and "http://" not in h and "https://" not in h

    def test_html_shows_terms_operators_diagnostics_constants(self, preset):
        h = show_recipe(preset)._repr_html_()
        assert "diffusion" in h and "convergence" in h  # term names
        assert "Σ" in h and "×" in h and "Δ" in h  # operator badges
        assert "diag_a" in h  # a raw diagnostic
        assert "-1." in h  # a scalar constant (the sign)

    def test_text_repr_is_an_ascii_tree(self, preset):
        t = repr(show_recipe(preset))
        assert "├─" in t or "└─" in t
        assert "tracer" in t and "diffusion" in t
        # Offline (no dataset): no resolved names, nothing marked unavailable.
        assert "->" not in t and "unavailable" not in t

    def test_scope_to_one_budget(self, preset):
        h = show_recipe(preset, "tracer")._repr_html_()
        assert "tracer" in h

    def test_unknown_budget_raises(self, preset):
        with pytest.raises(KeyError, match="No budget 'nope'"):
            show_recipe(preset, "nope")

    def test_html_escapes_content(self):
        # A diagnostic name with HTML-significant characters must be escaped,
        # not injected raw into the markup.
        recipe = {"b": {"rhs": {"var": None, "sum": {"x": {"var": "a<b&c"}}}}}
        h = show_recipe(recipe)._repr_html_()
        assert "a<b&c" not in h
        assert "a&lt;b&amp;c" in h


class TestUnitsAnnotation:
    """Units in square brackets after each term's name (the display's answer to
    "what does this term come out in?", available before any run)."""

    @staticmethod
    def _with_units(grid):
        grid._ds["diag_a"].attrs["units"] = "kg m-2 s-1"
        grid._ds["diag_b"].attrs["units"] = "kg m-2 s-1"
        grid._ds["area"].attrs["units"] = "m2"
        grid._ds["flux"].attrs["units"] = "kg s-1"
        return grid

    def test_declared_constants_annotate_without_a_dataset(self):
        recipe = {"b": {"rhs": {"product": {
            "rho": {"value": 1035.0, "units": "kg m-3"}, "d": "diag_a",
        }}}}
        t = repr(show_recipe(recipe))
        assert "rho: 1035. [kg m-3]" in t

    def test_bare_numbers_are_not_annotated(self, preset):
        # `sign: -1.` is dimensionless by default, not by declaration; a `[1]`
        # on every sign would be noise.
        t = repr(show_recipe(preset))
        assert "sign: -1." in t and "sign: -1. [" not in t

    def test_terms_annotate_from_the_dataset(self, preset, synthetic_grid):
        grid = self._with_units(synthetic_grid)
        t = repr(show_recipe(preset, "tracer", grid))
        # Composed upwards: (kg m-2 s-1) * m2 -> kg s-1, on the term itself...
        assert "diffusion [×] [kg.s-1]" in t
        # ...and each leaf shows the units it contributed.
        assert "area: area [m2]" in t

    def test_without_data_terms_stay_unannotated(self, preset):
        t = repr(show_recipe(preset, "tracer"))
        assert "diffusion [×]" in t and "diffusion [×] [" not in t

    def test_query_repr_prefers_the_stamped_units(self, preset, synthetic_grid):
        grid = self._with_units(synthetic_grid)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xbudget.collect_budgets(grid, preset)
        q = xbudget.BudgetQuery(grid, preset)
        h = q._repr_html_()
        stamped = grid._ds["tracer_rhs_diffusion"].attrs["units"]
        assert f'<span class="xbdg-units">[{stamped}]</span>' in h

    def test_reciprocal_operand_is_annotated_in_both_renderers(self):
        # The HTML tree is the one users actually see in a notebook; it was the
        # one leaf the units annotation missed.
        import numpy as np
        import xarray as xr

        ds = xr.Dataset({
            "dt": xr.DataArray(np.ones(3), dims=("t",), attrs={"units": "s"}),
        })
        view = show_recipe({"b": {"rhs": {"reciprocal": {"dt": {"var": "dt"}}}}},
                           None, ds)
        assert "of: dt [s]" in repr(view)
        html = view._repr_html_()
        assert '>dt</span><span class="xbdg-units">[s]</span>' in html

    def test_a_gap_does_not_blank_the_units_above_it(self, synthetic_grid):
        # One absent diagnostic used to leave every term above it unannotated --
        # including the budget roots, which are the ones worth reading. The
        # evaluator builds the root from the operands that remain, so the display
        # should say what it will come out in.
        grid = self._with_units(synthetic_grid)
        recipe = {"tracer": {"units": "kg s-1", "rhs": {"sum": {
            "diffusion": {"product": {"d": "diag_a", "area": "area"}},
            "absent": {"product": {"d": "not_present", "area": "area"}},
        }}}}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xbudget.collect_budgets(grid, recipe)
        stamped = grid._ds["tracer_rhs"].attrs["units"]
        root_line = repr(show_recipe(recipe, "tracer", grid)).splitlines()[2]
        assert f"[{stamped}]" in root_line
        # The term that will never be built is still, correctly, unannotated:
        # its line ends at the operator badge, with no units after it.
        absent_line = next(
            l for l in repr(show_recipe(recipe, "tracer", grid)).splitlines()
            if "absent" in l
        )
        assert absent_line.endswith("absent [×]")

    def test_query_repr_annotates_before_collecting(self, preset, synthetic_grid):
        # Nothing has been materialized, so there is no stamped attribute to
        # read; the display falls back to inferring from the recipe + dataset.
        grid = self._with_units(synthetic_grid)
        h = xbudget.BudgetQuery(grid, preset)._repr_html_()
        assert 'class="xbdg-units"' in h


class TestQueryRepr:
    def test_repr_html_annotates_and_greys_missing(self, preset, synthetic_grid):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xbudget.collect_budgets(synthetic_grid, preset)
            q = xbudget.BudgetQuery(synthetic_grid, preset)
            h = q._repr_html_()
        assert 'class="xbdg-wrap"' in h
        # A materialized term is annotated with its resolved variable name...
        assert "tracer_rhs_diffusion" in h
        # ...and the term whose diagnostic is absent is greyed / flagged.
        assert "xbdg-missing" in h and "unavailable" in h

    def test_repr_html_planned_when_no_dataset(self, preset):
        q = xbudget.BudgetQuery(None, preset)
        h = q._repr_html_()
        assert "planned" in h
        # Planned names are shown; nothing is greyed because nothing is checked.
        assert "tracer_rhs_diffusion" in h
        # Nothing greyed/flagged: the "unavailable" tag is only emitted when a
        # term resolves to None, which never happens with no dataset to check.
        assert "unavailable" not in h
        assert 'xbdg-term xbdg-missing"' not in h

    def test_text_repr_unchanged(self, preset):
        q = xbudget.BudgetQuery(None, preset)
        assert repr(q) == "<BudgetQuery: tracer (planned)>"


class TestShippedRecipesRender:
    """Every shipped recipe renders offline without raising (structure only)."""

    @pytest.mark.parametrize(
        "name",
        ["MOM6", "MOM6_3Donly", "MOM6_drift", "MOM6_surface", "ECCOV4r4_native"],
    )
    def test_offline_render(self, name):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            recipe = xbudget.load_preset_budget(name)
            view = show_recipe(recipe)
            html = view._repr_html_()
            text = repr(view)
        assert "<details" in html and len(text.splitlines()) > 1


class TestTolerance:
    def test_renders_malformed_recipe_the_parser_tolerates(self):
        # A term missing its enclosing `product:` — the parser warns and skips
        # the stray scalar key; the renderer must not choke on the result.
        recipe = {
            "b": {"rhs": {"var": None, "sum": {"t": {"var": None, "sign": -1.0}}}}
        }
        with pytest.warns(UserWarning):
            budgets = parse_budgets(recipe)
        # No warning expected from rendering itself.
        assert "<details" in render_budgets_html(budgets)
        assert "b" in render_budgets_text(budgets)
