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

Entry points:

- :func:`show_recipe` — inspect a recipe offline (no dataset). Returns a small
  displayable object; ``show_recipe(recipe, "heat")`` scopes to one budget.
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
# HTML rendering
# ---------------------------------------------------------------------------


def _leaf_html(label, value, value_cls):
    return (
        f'<div class="xbdg-leaf">'
        f'<span class="xbdg-label">{_esc(label)}:</span> '
        f'<span class="{value_cls}">{_esc(value)}</span></div>'
    )


def _op_contents_html(op, resolve):
    """The operands of one operation, as HTML rows/sub-trees."""
    if isinstance(op, (Sum, Product)):
        parts = []
        for name, operand in op.terms:
            if isinstance(operand, Constant):
                parts.append(_leaf_html(name, _fmt_const(operand.value), "xbdg-const"))
            elif isinstance(operand, VarRef):
                parts.append(_leaf_html(name, operand.name, "xbdg-diag"))
            else:  # Term
                parts.append(_term_html(operand, resolve))
        return "".join(parts)
    if isinstance(op, Difference):
        if isinstance(op.operand, VarRef):
            return _leaf_html("of", op.operand.name, "xbdg-diag")
        return _term_html(op.operand, resolve)
    if isinstance(op, Reciprocal):
        return _leaf_html("of", op.source, "xbdg-diag")
    if isinstance(op, LateralDivergence):
        return _term_html(op.fx, resolve) + _term_html(op.fy, resolve)
    return ""


def _annotation_html(term, resolve):
    """The var-name / unavailable annotation shown after a term's badges."""
    if resolve is None:
        return ""
    var = resolve(term)
    if var is None:
        return '<span class="xbdg-tag">unavailable</span>'
    return f'<span class="xbdg-var">{_esc(var)}</span>'


def _term_html(term, resolve):
    missing_cls = ""
    if resolve is not None and resolve(term) is None:
        missing_cls = " xbdg-missing"

    # A leaf term (no operations) is either a direct reference to a raw
    # diagnostic (explicit_var) or an empty placeholder.
    if not term.operations:
        value = term.explicit_var if isinstance(term.explicit_var, str) else "—"
        cls = "xbdg-diag" if isinstance(term.explicit_var, str) else "xbdg-label"
        return (
            f'<div class="xbdg-leaf{missing_cls}">'
            f'<span class="xbdg-name">{_esc(term.name)}</span> '
            f'<span class="{cls}">{_esc(value)}</span></div>'
        )

    annot = _annotation_html(term, resolve)
    if len(term.operations) == 1:
        op = term.operations[0]
        summary = (
            f'<span class="xbdg-name">{_esc(term.name)}</span>'
            f"{_badge(op.kind)}{annot}"
        )
        body = _op_contents_html(op, resolve)
    else:
        # Several operations on one term (e.g. a bulk product AND an equivalent
        # finer sum): give each its own collapsible group so their operands do
        # not run together.
        badges = "".join(_badge(op.kind) for op in term.operations)
        summary = f'<span class="xbdg-name">{_esc(term.name)}</span>{badges}{annot}'
        groups = []
        for op in term.operations:
            groups.append(
                f'<details><summary>{_badge(op.kind)}'
                f'<span class="xbdg-meta">{_esc(op.kind)}</span></summary>'
                f'<div class="xbdg-body">{_op_contents_html(op, resolve)}</div>'
                f"</details>"
            )
        body = "".join(groups)

    return (
        f'<details class="xbdg-term{missing_cls}"><summary>{summary}</summary>'
        f'<div class="xbdg-body">{body}</div></details>'
    )


def _budget_html(budget, resolve):
    meta = "".join(
        f'<span class="xbdg-meta">{_esc(k)}={_esc(v)}</span>'
        for k, v in budget.metadata.items()
    )
    sides = "".join(_term_html(term, resolve) for term in budget.sides.values())
    return (
        f'<details class="xbdg-budget" open><summary>'
        f'<span class="xbdg-name">{_esc(budget.name)}</span>{meta}</summary>'
        f'<div class="xbdg-body">{sides}</div></details>'
    )


def render_budgets_html(budgets, resolve=None, header=None):
    """Render ``{name: Budget}`` as a collapsible HTML tree.

    ``resolve`` is an optional ``Term -> variable name or None`` callback
    (:meth:`xbudget.query.BudgetQuery._resolve_var`); when given, each term is
    annotated with its dataset variable name and unmaterialized terms are greyed
    out. ``header`` is an optional title line.
    """
    head = f'<div class="xbdg-header">{_esc(header)}</div>' if header else ""
    body = "".join(_budget_html(b, resolve) for b in budgets.values())
    return f'<div class="xbdg-wrap">{_CSS}{head}{body}</div>'


# ---------------------------------------------------------------------------
# Text rendering (fallback for non-HTML contexts)
# ---------------------------------------------------------------------------


def _text_lines(label, node, resolve, prefix, is_last):
    """Yield the ASCII-tree lines for one node under ``label``."""
    connector = "└─ " if is_last else "├─ "
    child_prefix = prefix + ("   " if is_last else "│  ")

    if isinstance(node, Constant):
        yield f"{prefix}{connector}{label}: {_fmt_const(node.value)}"
        return
    if isinstance(node, VarRef):
        yield f"{prefix}{connector}{label}: {node.name}"
        return

    # A Term.
    term = node
    if not term.operations:
        value = term.explicit_var if isinstance(term.explicit_var, str) else "—"
        yield f"{prefix}{connector}{label}: {value}"
        return

    suffix = ""
    if resolve is not None:
        var = resolve(term)
        suffix = f"  -> {var}" if var is not None else "  (unavailable)"
    ops = " ".join(_OP_SYMBOLS.get(op.kind, op.kind) for op in term.operations)
    yield f"{prefix}{connector}{label} [{ops}]{suffix}"

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
            child_label, child_node, resolve, child_prefix, i == len(children) - 1
        )


def render_budgets_text(budgets, resolve=None, header=None):
    """Render ``{name: Budget}`` as an indented ASCII tree."""
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
                _text_lines(term.name, term, resolve, "", j == len(sides) - 1)
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

    def __init__(self, budgets):
        self._budgets = budgets

    def _header(self):
        return "xbudget recipe · " + ", ".join(self._budgets)

    def _repr_html_(self):
        return render_budgets_html(self._budgets, header=self._header())

    def __repr__(self):
        return render_budgets_text(self._budgets, header=self._header())


def show_recipe(recipe, budget=None):
    """Display a recipe as a collapsible tree.

    Parameters
    ----------
    recipe : dict
        A recipe in xbudget format (e.g. from
        :func:`~xbudget.presets.load_preset_budget`).
    budget : str, optional
        Show only this budget (``"mass"``, ``"heat"``, ``"salt"``). Default:
        all budgets in the recipe.

    Returns
    -------
    _RecipeView
        An object that renders as a collapsible HTML tree in a Jupyter/VSCode
        notebook and as an indented ASCII tree when ``repr()``-ed or printed.

    Examples
    --------
    >>> import xbudget
    >>> recipe = xbudget.load_preset_budget("MOM6")
    >>> xbudget.show_recipe(recipe, "heat")   # doctest: +SKIP
    """
    budgets = parse_budgets(recipe)
    if budget is not None:
        if budget not in budgets:
            raise KeyError(
                f"No budget {budget!r} in recipe. Available: {list(budgets)}."
            )
        budgets = {budget: budgets[budget]}
    return _RecipeView(budgets)
