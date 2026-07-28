"""Human-friendly renderings of an xbudget recipe.

A recipe is a deeply nested provenance tree. Printed as raw JSON it is almost
unreadable; what a user actually wants is the *shape* — which terms sum into
which, what operation builds each, and (after a run) what each term is called in
the dataset. This module renders the **typed** tree (:mod:`xbudget.nodes`, via
:func:`xbudget.parse.parse_budgets`) rather than the raw dict, so it shows real
semantics (operator badges, scalars vs. diagnostics) and inherits the parser's
tolerance of the malformed/placeholder nodes real recipes carry.

Two renderings, one walker:

- ``_repr_html_`` — a collapsible tree of nested ``<details>``/``<summary>``
  elements (native HTML5, no JavaScript), styled after xarray's ``Dataset``
  repr so it feels at home in the same notebooks. Click an arrow to expand a
  level.
- ``__repr__`` — an indented ``├─``/``└─`` ASCII tree, for terminals and any
  context that does not render HTML.

Both annotate each node with its units in square brackets — ``density: 1035.
[kg m-3]``, ``lateral [×] [W]`` — inferred through the tree by
:func:`xbudget.units.infer_units`, so a recipe shows the same unit arithmetic
``assert_unit_consistency`` checks, before anything is evaluated.

Entry points:

- :func:`show_recipe` — inspect a recipe offline (no dataset). Returns a small
  displayable object; ``show_recipe(recipe, "heat", grid)`` scopes to one budget
  and reads its diagnostics' units from ``grid``.
- :meth:`xbudget.query.BudgetQuery._repr_html_` — the same tree, but annotated
  with each term's resolved variable name and with unmaterialized terms greyed
  out, so displaying a query in a notebook shows structure *and* run state.
"""
import html

from .nodes import (
    Constant,
    Difference,
    LateralDivergence,
    Product,
    Reciprocal,
    Sum,
    Term,
    VarRef,
)
from .parse import parse_budgets
from .units import format_units, infer_units

__all__ = ["show_recipe"]

# Operator glyphs shown as a badge on each term's summary line.
_OP_SYMBOLS = {
    "sum": "Σ",              # Σ
    "product": "×",         # ×
    "difference": "Δ",      # Δ
    "reciprocal": "1÷x",    # 1÷x
    "lateral_divergence": "∇·",  # ∇·
}

# Scoped, self-contained styling. Backgrounds/borders are neutral
# semi-transparent greys so the tree reads on both light and dark notebook
# themes without needing to detect which; a prefers-color-scheme override only
# fine-tunes the accent text colors. Emitted with every repr (like xarray's);
# duplicate identical <style> blocks across cells are harmless.
_CSS = """<style>
.xbdg-wrap {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial,
    sans-serif;
  font-size: 0.82rem;
  line-height: 1.6;
  color: inherit;
  --xbdg-diag: #0969da;
  --xbdg-const: #1a7f37;
  --xbdg-var: #8250df;
  --xbdg-muted: #808080;
  --xbdg-line: rgba(128, 128, 128, 0.35);
  --xbdg-chip: rgba(128, 128, 128, 0.16);
}
@media (prefers-color-scheme: dark) {
  .xbdg-wrap {
    --xbdg-diag: #4493f8;
    --xbdg-const: #3fb950;
    --xbdg-var: #bc8cff;
  }
}
.xbdg-wrap details { margin: 0; }
.xbdg-wrap summary {
  cursor: pointer;
  list-style-position: outside;
  padding: 1px 0;
  outline: none;
}
.xbdg-wrap summary:hover { background: var(--xbdg-chip); border-radius: 3px; }
.xbdg-wrap .xbdg-body {
  margin-left: 0.55em;
  padding-left: 0.7em;
  border-left: 1px solid var(--xbdg-line);
}
.xbdg-wrap .xbdg-header {
  font-weight: 600;
  margin-bottom: 0.3em;
  opacity: 0.85;
}
.xbdg-wrap .xbdg-name { font-weight: 600; }
.xbdg-wrap .xbdg-op {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.9em;
  background: var(--xbdg-chip);
  border-radius: 3px;
  padding: 0 0.4em;
  margin-left: 0.4em;
}
.xbdg-wrap .xbdg-meta {
  font-size: 0.9em;
  color: var(--xbdg-muted);
  margin-left: 0.5em;
}
.xbdg-wrap .xbdg-var {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  color: var(--xbdg-var);
  margin-left: 0.5em;
}
.xbdg-wrap .xbdg-tag {
  color: var(--xbdg-muted);
  font-style: italic;
  margin-left: 0.5em;
}
.xbdg-wrap .xbdg-units {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.9em;
  color: var(--xbdg-muted);
  margin-left: 0.4em;
}
.xbdg-wrap .xbdg-leaf { padding: 1px 0; }
.xbdg-wrap .xbdg-label { color: var(--xbdg-muted); }
.xbdg-wrap .xbdg-diag {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  color: var(--xbdg-diag);
}
.xbdg-wrap .xbdg-const {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  color: var(--xbdg-const);
}
.xbdg-wrap .xbdg-missing { opacity: 0.45; }
</style>"""


def _esc(text):
    return html.escape(str(text))


def _fmt_const(value):
    """Format a scalar factor/addend the way recipes write it (``1035.``, ``-1.``)."""
    if value == int(value):
        return f"{int(value)}."
    return f"{value:g}"


def _badge(kind):
    sym = _OP_SYMBOLS.get(kind, kind)
    return f'<span class="xbdg-op" title="{_esc(kind)}">{_esc(sym)}</span>'


# ---------------------------------------------------------------------------
# Units annotation
# ---------------------------------------------------------------------------


def _diagnostic_units_lookup(data):
    """A ``name -> units string`` reader over a grid/dataset (or nothing)."""
    ds = getattr(data, "_ds", data)
    if ds is None:
        return lambda _name: None

    def lookup(name):
        return ds[name].attrs.get("units") if name in ds else None

    return lookup


def units_annotator(data=None):
    """Build the ``node -> units string`` callback the renderers annotate with.

    Infers each node's units from the recipe tree (:func:`xbudget.units.infer_units`),
    reading raw diagnostics' units from ``data`` when one is given. Without
    ``data`` only declared constants carry units, so most terms stay unannotated.
    """
    lookup = _diagnostic_units_lookup(data)
    return lambda node: format_units(infer_units(node, lookup))


def _units_text(node, units):
    """The units to display for one node, or ``None`` to show none.

    A bare number (``sign: -1.``) is dimensionless by default rather than by
    declaration, so annotating it is noise; a constant that *declares* units is
    worth showing.
    """
    if units is None:
        return None
    if isinstance(node, Constant) and node.units == "1":
        return None
    return units(node)


def _units_html(node, units):
    text = _units_text(node, units)
    return "" if text is None else f'<span class="xbdg-units">[{_esc(text)}]</span>'


def _units_suffix(node, units):
    text = _units_text(node, units)
    return "" if text is None else f" [{text}]"


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


def _leaf_html(label, value, value_cls, units_html=""):
    return (
        f'<div class="xbdg-leaf">'
        f'<span class="xbdg-label">{_esc(label)}:</span> '
        f'<span class="{value_cls}">{_esc(value)}</span>{units_html}</div>'
    )


def _op_contents_html(op, resolve, units):
    """The operands of one operation, as HTML rows/sub-trees."""
    if isinstance(op, (Sum, Product)):
        parts = []
        for name, operand in op.terms:
            if isinstance(operand, Constant):
                parts.append(_leaf_html(
                    name, _fmt_const(operand.value), "xbdg-const",
                    _units_html(operand, units),
                ))
            elif isinstance(operand, VarRef):
                parts.append(_leaf_html(
                    name, operand.name, "xbdg-diag", _units_html(operand, units)
                ))
            else:  # Term
                parts.append(_term_html(operand, resolve, units))
        return "".join(parts)
    if isinstance(op, Difference):
        if isinstance(op.operand, VarRef):
            return _leaf_html(
                "of", op.operand.name, "xbdg-diag",
                _units_html(op.operand, units),
            )
        return _term_html(op.operand, resolve, units)
    if isinstance(op, Reciprocal):
        return _leaf_html("of", op.source, "xbdg-diag")
    if isinstance(op, LateralDivergence):
        return _term_html(op.fx, resolve, units) + _term_html(op.fy, resolve, units)
    return ""


def _annotation_html(term, resolve):
    """The var-name / unavailable annotation shown after a term's badges."""
    if resolve is None:
        return ""
    var = resolve(term)
    if var is None:
        return '<span class="xbdg-tag">unavailable</span>'
    return f'<span class="xbdg-var">{_esc(var)}</span>'


def _term_html(term, resolve, units):
    missing_cls = ""
    if resolve is not None and resolve(term) is None:
        missing_cls = " xbdg-missing"
    unit_html = _units_html(term, units)

    # A leaf term (no operations) is either a direct reference to a raw
    # diagnostic (explicit_var) or an empty placeholder.
    if not term.operations:
        value = term.explicit_var if isinstance(term.explicit_var, str) else "—"
        cls = "xbdg-diag" if isinstance(term.explicit_var, str) else "xbdg-label"
        return (
            f'<div class="xbdg-leaf{missing_cls}">'
            f'<span class="xbdg-name">{_esc(term.name)}</span> '
            f'<span class="{cls}">{_esc(value)}</span>{unit_html}</div>'
        )

    annot = _annotation_html(term, resolve)
    if len(term.operations) == 1:
        op = term.operations[0]
        summary = (
            f'<span class="xbdg-name">{_esc(term.name)}</span>'
            f"{_badge(op.kind)}{unit_html}{annot}"
        )
        body = _op_contents_html(op, resolve, units)
    else:
        # Several operations on one term (e.g. a bulk product AND an equivalent
        # finer sum): give each its own collapsible group so their operands do
        # not run together.
        badges = "".join(_badge(op.kind) for op in term.operations)
        summary = (
            f'<span class="xbdg-name">{_esc(term.name)}</span>'
            f"{badges}{unit_html}{annot}"
        )
        groups = []
        for op in term.operations:
            groups.append(
                f'<details><summary>{_badge(op.kind)}'
                f'<span class="xbdg-meta">{_esc(op.kind)}</span></summary>'
                f'<div class="xbdg-body">{_op_contents_html(op, resolve, units)}</div>'
                f"</details>"
            )
        body = "".join(groups)

    return (
        f'<details class="xbdg-term{missing_cls}"><summary>{summary}</summary>'
        f'<div class="xbdg-body">{body}</div></details>'
    )


def _budget_html(budget, resolve, units):
    meta = "".join(
        f'<span class="xbdg-meta">{_esc(k)}={_esc(v)}</span>'
        for k, v in budget.metadata.items()
    )
    sides = "".join(
        _term_html(term, resolve, units) for term in budget.sides.values()
    )
    return (
        f'<details class="xbdg-budget" open><summary>'
        f'<span class="xbdg-name">{_esc(budget.name)}</span>{meta}</summary>'
        f'<div class="xbdg-body">{sides}</div></details>'
    )


def render_budgets_html(budgets, resolve=None, units=None, header=None):
    """Render ``{name: Budget}`` as a collapsible HTML tree.

    ``resolve`` is an optional ``Term -> variable name or None`` callback
    (:meth:`xbudget.query.BudgetQuery._resolve_var`); when given, each term is
    annotated with its dataset variable name and unmaterialized terms are greyed
    out. ``units`` is an optional ``node -> units string or None`` callback (see
    :func:`units_annotator`); when given, each node is annotated with its units
    in square brackets. ``header`` is an optional title line.
    """
    head = f'<div class="xbdg-header">{_esc(header)}</div>' if header else ""
    body = "".join(_budget_html(b, resolve, units) for b in budgets.values())
    return f'<div class="xbdg-wrap">{_CSS}{head}{body}</div>'


# ---------------------------------------------------------------------------
# Text rendering (fallback for non-HTML contexts)
# ---------------------------------------------------------------------------


def _text_lines(label, node, resolve, units, prefix, is_last):
    """Yield the ASCII-tree lines for one node under ``label``."""
    connector = "└─ " if is_last else "├─ "
    child_prefix = prefix + ("   " if is_last else "│  ")
    unit_suffix = _units_suffix(node, units)

    if isinstance(node, Constant):
        yield f"{prefix}{connector}{label}: {_fmt_const(node.value)}{unit_suffix}"
        return
    if isinstance(node, VarRef):
        yield f"{prefix}{connector}{label}: {node.name}{unit_suffix}"
        return

    # A Term.
    term = node
    if not term.operations:
        value = term.explicit_var if isinstance(term.explicit_var, str) else "—"
        yield f"{prefix}{connector}{label}: {value}{unit_suffix}"
        return

    suffix = ""
    if resolve is not None:
        var = resolve(term)
        suffix = f"  -> {var}" if var is not None else "  (unavailable)"
    ops = " ".join(_OP_SYMBOLS.get(op.kind, op.kind) for op in term.operations)
    yield f"{prefix}{connector}{label} [{ops}]{unit_suffix}{suffix}"

    children = []  # (child_label, child_node)
    for op in term.operations:
        if isinstance(op, (Sum, Product)):
            children.extend(op.terms)
        elif isinstance(op, Difference):
            operand = op.operand
            children.append(("of", operand))
        elif isinstance(op, Reciprocal):
            children.append(("of", VarRef(op.source)))
        elif isinstance(op, LateralDivergence):
            children.append((op.fx.name, op.fx))
            children.append((op.fy.name, op.fy))
    for i, (child_label, child_node) in enumerate(children):
        yield from _text_lines(
            child_label, child_node, resolve, units, child_prefix,
            i == len(children) - 1,
        )


def render_budgets_text(budgets, resolve=None, units=None, header=None):
    """Render ``{name: Budget}`` as an indented ASCII tree.

    ``resolve`` and ``units`` are the same optional callbacks
    :func:`render_budgets_html` takes; a node's units are shown in square
    brackets after its name.
    """
    lines = []
    if header:
        lines.append(header)
    names = list(budgets)
    for i, name in enumerate(names):
        budget = budgets[name]
        meta = budget.metadata
        meta_str = (
            "  " + ", ".join(f"{k}={v}" for k, v in meta.items()) if meta else ""
        )
        lines.append(f"{name}{meta_str}")
        sides = list(budget.sides.items())
        for j, (_side, term) in enumerate(sides):
            lines.extend(
                _text_lines(term.name, term, resolve, units, "", j == len(sides) - 1)
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public offline entry point
# ---------------------------------------------------------------------------


class _RecipeView:
    """A displayable view of a recipe, returned by :func:`show_recipe`.

    Renders as a collapsible HTML tree in a notebook (``_repr_html_``) and as an
    ASCII tree in a terminal (``__repr__``).
    """

    def __init__(self, budgets, units=None):
        self._budgets = budgets
        self._units = units

    def _header(self):
        return "xbudget recipe · " + ", ".join(self._budgets)

    def _repr_html_(self):
        return render_budgets_html(
            self._budgets, units=self._units, header=self._header()
        )

    def __repr__(self):
        return render_budgets_text(
            self._budgets, units=self._units, header=self._header()
        )


def show_recipe(recipe, budget=None, data=None):
    """Display a recipe as a collapsible tree, annotated with each term's units.

    Parameters
    ----------
    recipe : dict
        A recipe in xbudget format (e.g. from
        :func:`~xbudget.presets.load_preset_budget`).
    budget : str, optional
        Show only this budget (``"mass"``, ``"heat"``, ``"salt"``). Default:
        all budgets in the recipe.
    data : xgcm.Grid, xr.Dataset, or None
        Where to read raw diagnostics' ``units`` attributes from. Units are
        inferred through the tree from its leaves, so passing the data you intend
        to collect against annotates the whole recipe — reading the tree top-down
        then shows *where* a unit stops being inferable. Without it only declared
        constants carry units, and most terms show none.

    Returns
    -------
    _RecipeView
        An object that renders as a collapsible HTML tree in a Jupyter/VSCode
        notebook and as an indented ASCII tree when ``repr()``-ed or printed.

    Examples
    --------
    >>> import xbudget
    >>> recipe = xbudget.load_preset_budget("MOM6")
    >>> xbudget.show_recipe(recipe, "heat", grid)   # doctest: +SKIP
    """
    budgets = parse_budgets(recipe)
    if budget is not None:
        if budget not in budgets:
            raise KeyError(
                f"No budget {budget!r} in recipe. Available: {list(budgets)}."
            )
        budgets = {budget: budgets[budget]}
    return _RecipeView(budgets, units=units_annotator(data))
