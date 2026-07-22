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
