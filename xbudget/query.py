"""Query the variables an xbudget run materialized.

The evaluator (:mod:`xbudget.evaluate`) writes one variable per operation into
the dataset and names it by its term path. This module answers the questions a
user actually has afterwards: *what is this term called?*, *what went into it?*,
and *what are the top-level terms of this budget?* — without them needing to
know the naming rule or parse flat variable names.

It is the v1 replacement for the ``get_vars``/``aggregate`` helpers in
:mod:`xbudget.collect`, which only work after a (deprecated)
``name_scheme="legacy"`` run because they read ``var`` fields that the legacy
engine fills into the recipe by mutating it.

A :class:`BudgetQuery` is built from the same two things the run used — the data
and the recipe — so it also works on a dataset reopened from disk long after the
run, and (via ``explicit_var``, see :meth:`BudgetQuery._resolve_var`) on a recipe
a legacy run already filled in.
"""
import difflib

import xgcm

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
from .parse import parse_budgets, _resolve_recipe

__all__ = ["BudgetQuery"]


def _name(path):
    """Output variable name for a term path (mirrors ``evaluate._new_name``)."""
    return "_".join(path)


class BudgetQuery:
    """Look up the variables an xbudget run produced.

    Parameters
    ----------
    data : xgcm.Grid, xr.Dataset, or None
        The object passed to :func:`~xbudget.collect.collect_budgets`. Used to
        tell which terms were actually materialized: a term whose diagnostics
        were missing from the dataset is skipped by the evaluator (with a
        warning) and is reported here as unavailable rather than as a name that
        does not exist.

        ``None`` inspects a recipe offline, with no dataset to check
        against. Results are then *planned* names — what the recipe would
        emit if every diagnostic were present — which may differ from a real
        run (see :meth:`_resolve_var`).
    recipe : dict
        The recipe used for the run (e.g. from
        :func:`~xbudget.presets.load_preset_budget`).
    xbudget_dict : dict, optional
        Deprecated alias for ``recipe``; removed in xbudget v1.0.

    Examples
    --------
    >>> xbudget.collect_budgets(grid, recipe)
    >>> q = xbudget.BudgetQuery(grid, recipe)
    >>> q.var("heat_lhs_advection")
    'heat_lhs_advection'
    >>> grid._ds[q.var("heat_lhs_advection")]
    <xarray.DataArray 'heat_lhs_advection' ...>
    >>> q.aggregate()["heat"]["rhs"]
    {'advection': 'heat_rhs_advection', 'diffusion': 'heat_rhs_diffusion', ...}

    See also
    --------
    xbudget.parse_budgets, xbudget.evaluate_budgets
    """

    def __init__(self, data, recipe=None, *, xbudget_dict=None):
        recipe = _resolve_recipe(recipe, xbudget_dict, "BudgetQuery")
        if isinstance(data, xgcm.grid.Grid):
            self._ds = data._ds
        else:
            self._ds = data  # an xr.Dataset, or None
        self.budgets = parse_budgets(recipe)

        # Address indices, all built by walking the tree. Names are matched
        # exactly and never split on "_": real term names contain underscores
        # (e.g. "Eulerian_tendency"), so splitting would mis-parse them.
        self._by_name = {}       # v1 primary variable name -> Term
        self._by_op_name = {}    # v1 operation-suffixed name -> (Term, Operation)
        self._by_path = {}       # path tuple -> Term
        self._diagnostics = {}   # raw diagnostic name -> [term paths using it]
        self._legacy = {}        # legacy variable name -> v1 name
        self._duplicates = {}    # v1 name -> [colliding paths]

        for budget in self.budgets.values():
            for side, term in budget.sides.items():
                self._index_term(term, f"{budget.name}_{side}")

    # -- index construction -------------------------------------------------

    def _index_term(self, term, legacy_namepath):
        """Register a term and its descendants in the address indices."""
        base = _name(term.path)
        if base in self._by_name and self._by_name[base] is not term:
            # "_".join(path) is not injective: ("a", "b_c") and ("a", "b", "c")
            # both give "a_b_c". Record it rather than silently overwrite.
            self._duplicates.setdefault(base, [self._by_name[base].path])
            self._duplicates[base].append(term.path)
        self._by_name.setdefault(base, term)
        self._by_path[tuple(term.path)] = term
        self._legacy.setdefault(legacy_namepath, base)

        if isinstance(term.explicit_var, str):
            self._diagnostics.setdefault(term.explicit_var, []).append(
                tuple(term.path)
            )

        for op in term.operations:
            self._legacy.setdefault(f"{legacy_namepath}_{op.kind}", base)
            # A term with several operations emits one variable each; only the
            # primary gets the bare name, so the others are addressable only by
            # their suffixed name (e.g. "heat_rhs_boundary_sum").
            self._by_op_name.setdefault(f"{base}_{op.kind}", (term, op))
            self._index_op(op, term, legacy_namepath)

    def _index_op(self, op, term, legacy_namepath):
        """Register an operation's operands."""
        if isinstance(op, (Sum, Product)):
            for name, operand in op.terms:
                if isinstance(operand, Term):
                    self._index_term(
                        operand, f"{legacy_namepath}_{op.kind}_{name}"
                    )
                elif isinstance(operand, VarRef):
                    self._diagnostics.setdefault(operand.name, []).append(
                        tuple(term.path)
                    )
        elif isinstance(op, Difference):
            if isinstance(op.operand, VarRef):
                self._diagnostics.setdefault(op.operand.name, []).append(
                    tuple(term.path)
                )
            else:
                self._index_term(
                    op.operand,
                    f"{legacy_namepath}_difference_{op.operand.name}",
                )
        elif isinstance(op, Reciprocal):
            self._diagnostics.setdefault(op.source, []).append(tuple(term.path))
        elif isinstance(op, LateralDivergence):
            self._index_term(op.fx, f"{legacy_namepath}_Fx")
            self._index_term(op.fy, f"{legacy_namepath}_Fy")

    # -- resolution ---------------------------------------------------------

    def _in_ds(self, name):
        return self._ds is None or name in self._ds

    def _resolve_var(self, term):
        """The dataset variable holding this term's primary value, or None.

        Mirrors the naming in ``evaluate._Evaluator._eval_term``, which depends
        on facts only known at run time — if a term's primary operation was
        skipped because a diagnostic was missing, a sibling operation claims the
        bare name instead. So resolution checks the dataset rather than
        predicting from the recipe alone:

        1. ``"_".join(term.path)`` — the v1 primary name (the normal case);
        2. ``term.explicit_var`` — a leaf's raw diagnostic. This is also what
           makes a *legacy-filled* recipe resolve correctly: a legacy run writes
           each node's legacy variable name into its ``var`` field, so rung 1
           misses (v1 names are not in a legacy run's dataset) and this rung
           finds the legacy name that is;
        3. ``f"{base}_{op.kind}"`` — defensive. The current evaluator always
           gives the bare name to *something* when it emits anything at all, so
           rungs 1-2 are exhaustive today; this keeps resolution total if that
           ever changes, rather than silently reporting a live variable as
           missing.

        Returns ``None`` when nothing materialized, so callers can drop the term
        rather than hand back a name that would ``KeyError`` on lookup.
        """
        base = _name(term.path)
        if self._in_ds(base):
            return base
        if isinstance(term.explicit_var, str) and self._in_ds(term.explicit_var):
            return term.explicit_var
        for op in term.operations:
            suffixed = f"{base}_{op.kind}"
            if self._in_ds(suffixed):
                return suffixed
        return None

    def _lookup(self, address):
        """Resolve one address to a Term, or raise KeyError with a hint."""
        if isinstance(address, tuple):
            if address in self._by_path:
                return self._by_path[address]
            raise KeyError(
                f"No term at path {address!r}. Use BudgetQuery.terms() to list "
                f"the available paths."
            )
        if not isinstance(address, str):
            raise TypeError(
                f"Address must be a variable name (str), a path (tuple), or a "
                f"list of either; got {type(address).__name__}."
            )
        if address in self._by_name:
            return self._by_name[address]
        return None  # not a term; caller decides (diagnostic / error)

    def _unknown(self, address):
        """Build the KeyError for an address that matched nothing."""
        if address in self._legacy:
            return KeyError(
                f"{address!r} is a legacy (operator-infixed) variable name. The "
                f"v1 equivalent is {self._legacy[address]!r}. Legacy names are "
                f"only produced by collect_budgets(..., name_scheme='legacy'), "
                f"which is deprecated."
            )
        candidates = (
            list(self._by_name) + list(self._by_op_name) + list(self._diagnostics)
        )
        close = difflib.get_close_matches(address, candidates, n=3)
        hint = f" Did you mean: {', '.join(repr(c) for c in close)}?" if close else ""
        return KeyError(f"Unknown term or diagnostic {address!r}.{hint}")

    # -- public API ---------------------------------------------------------

    @property
    def alias_map(self):
        """Legacy variable name -> v1 variable name, for programmatic migration."""
        return dict(self._legacy)

    @property
    def duplicate_names(self):
        """v1 names produced by more than one term path (should be empty).

        ``"_".join(path)`` is not injective, so two distinct terms *can* collide
        on one output name — in which case the second silently overwrites the
        first in the dataset. Non-empty means the recipe needs a rename.
        """
        return {k: list(v) for k, v in self._duplicates.items()}

    def metadata(self, budget=None):
        """Return a budget's metadata (the non-``lhs``/``rhs`` keys of a recipe).

        Budget metadata declares the quantities the engine does not itself build
        but that describe the budget's state — e.g. the mass budget's layer
        ``thickness`` and its coordinate ``lambda``. Exposing it here gives
        downstream code (``xwmt``/``xwmb``) one engine-independent contract for
        those names instead of reaching into raw ``recipe`` keys.

        Parameters
        ----------
        budget : str or None
            A budget name (e.g. ``"mass"``, ``"heat"``). If ``None`` (default),
            return ``{budget_name: metadata_dict}`` for every budget.

        Returns
        -------
        dict
            The metadata dict for ``budget`` (e.g.
            ``{"lambda": "density", "thickness": "thkcello"}``), or a dict of
            them keyed by budget name. Shallow copies, so mutating the result
            never rewrites the parsed recipe (metadata values are scalars).
        """
        if budget is None:
            return {name: dict(b.metadata) for name, b in self.budgets.items()}
        if budget not in self.budgets:
            raise KeyError(
                f"No budget {budget!r}. Available budgets: {list(self.budgets)}."
            )
        return dict(self.budgets[budget].metadata)

    def thickness(self, budget="mass"):
        """The layer-thickness variable a budget declares, or ``None``.

        The mass budget's ``thickness`` (e.g. ``"thkcello"``) is its prognostic
        state variable — layer thickness in metres — and the core input
        ``xwmt.WaterMass`` needs to build its vertical (mass) metrics. Reading it
        through this accessor keeps that dependency explicit and independent of
        the recipe's dict layout or the naming scheme used to collect it.

        Returns ``None`` if the budget declares no ``thickness``. Raises
        ``KeyError`` if the recipe has no such budget at all — including the
        defaulted ``"mass"``, so a recipe without a mass budget is reported
        rather than silently indistinguishable from one that omits the key.
        """
        return self.metadata(budget).get("thickness")

    def lambda_var(self, budget):
        """The ``lambda`` coordinate variable a budget declares, or ``None``.

        (Named ``lambda_var`` because ``lambda`` is a Python keyword; the
        recipe key itself is ``lambda``.) Raises ``KeyError`` for an unknown
        budget, as :meth:`metadata` does.
        """
        return self.metadata(budget).get("lambda")

    def surface_lambda(self, budget):
        """The ``surface_lambda`` variable a budget declares, or ``None``.

        Raises ``KeyError`` for an unknown budget, as :meth:`metadata` does.
        """
        return self.metadata(budget).get("surface_lambda")

    def terms(self):
        """Map every term path to its variable name (``None`` if not materialized)."""
        return {path: self._resolve_var(term) for path, term in self._by_path.items()}

    def var(self, address):
        """The dataset variable name for a term, or ``None`` if not materialized.

        Parameters
        ----------
        address : str or tuple
            A v1 variable name (``"heat_lhs_advection"``) or a term path
            (``("heat", "lhs", "advection")``).

        Raises
        ------
        KeyError
            If the address matches no term. Legacy operator-infixed names get an
            error naming their v1 equivalent.
        """
        term = self._lookup(address)
        if term is not None:
            return self._resolve_var(term)
        if isinstance(address, str):
            if address in self._by_op_name:
                # An operation-suffixed name is its own answer, when emitted.
                return address if self._in_ds(address) else None
            if address in self._diagnostics:
                return address if self._in_ds(address) else None
        raise self._unknown(address)

    def get_vars(self, address):
        """Describe a term (or raw diagnostic): its variable and its operands.

        Parameters
        ----------
        address : str, tuple, or list
            A v1 variable name, a term path, or a raw diagnostic name. A list
            (or array) requests a batch and returns a list of results.

        Returns
        -------
        dict
            ``{"var": name}`` plus, for each of the term's operations, a
            ``"sum"``/``"product"``/... key listing its operands (sub-term
            variable names, raw diagnostic names, and constants, in recipe
            order). For a raw diagnostic: ``{"var": name, "referenced_by":
            [paths]}``.
        """
        # A tuple is one address (a path); any other sequence is a batch.
        if not isinstance(address, (str, tuple)) and hasattr(address, "__iter__"):
            return [self.get_vars(a) for a in address]

        term = self._lookup(address)
        if term is None and isinstance(address, str):
            if address in self._by_op_name:
                # One specific operation of a multi-operation term.
                op_term, op = self._by_op_name[address]
                out = {"var": address if self._in_ds(address) else None}
                operands = self._operands(op)
                if operands is not None:
                    out[op.kind] = operands
                return out
            if address in self._diagnostics:
                return {
                    "var": address,
                    "referenced_by": list(self._diagnostics[address]),
                }
        if term is None:
            raise self._unknown(address)

        out = {"var": self._resolve_var(term)}
        for op in term.operations:
            operands = self._operands(op)
            if operands is not None:
                out[op.kind] = operands
        return out

    def _operands(self, op):
        """List an operation's inputs as names/constants, in recipe order."""
        if isinstance(op, (Sum, Product)):
            out = []
            for _label, operand in op.terms:
                if isinstance(operand, Term):
                    resolved = self._resolve_var(operand)
                    if resolved is not None:
                        out.append(resolved)
                elif isinstance(operand, Constant):
                    out.append(operand.value)
                elif isinstance(operand, VarRef):
                    if self._in_ds(operand.name):
                        out.append(operand.name)
                    elif op.kind == "product":
                        # Mirror the evaluator, which multiplies in 0.0 for a
                        # missing factor rather than dropping it. Omitting it
                        # here would claim the variable equals the surviving
                        # factors when it is in fact identically zero.
                        out.append(0.0)
            return out
        if isinstance(op, Difference):
            if isinstance(op.operand, VarRef):
                return [op.operand.name]
            resolved = self._resolve_var(op.operand)
            return [resolved] if resolved is not None else []
        if isinstance(op, Reciprocal):
            return [op.source]
        if isinstance(op, LateralDivergence):
            fluxes = [self._resolve_var(op.fx), self._resolve_var(op.fy)]
            return [f for f in fluxes if f is not None]
        return None

    def aggregate(self, decompose=()):
        """Collapse each budget to its top-level terms and their variable names.

        Parameters
        ----------
        decompose : str or iterable of str, optional
            Term names to expand into *their* summed parts instead of reporting
            them whole. Expanded keys are joined to their parent's with an
            underscore (``"advection"`` -> ``"advection_lateral"``), applied
            recursively. Matching is exact.

        Returns
        -------
        dict
            ``{budget: {**metadata, side: {label: variable_name}}}``. Terms that
            were not materialized are omitted.

        Examples
        --------
        >>> q.aggregate()["heat"]["rhs"]
        {'advection': 'heat_rhs_advection', 'diffusion': 'heat_rhs_diffusion'}
        >>> q.aggregate(decompose=["diffusion"])["heat"]["rhs"]
        {'advection': 'heat_rhs_advection',
         'diffusion_lateral': 'heat_rhs_diffusion_lateral',
         'diffusion_vertical': 'heat_rhs_diffusion_vertical'}
        """
        if isinstance(decompose, str):
            decompose = {decompose}
        else:
            decompose = set(decompose)

        out = {}
        for name, budget in self.budgets.items():
            body = dict(budget.metadata)
            for side, term in budget.sides.items():
                body[side] = self._aggregate_term(term, decompose)
            out[name] = body
        return out

    @staticmethod
    def _sum_op(term):
        """The term's first ``Sum`` operation, or None."""
        return next((op for op in term.operations if isinstance(op, Sum)), None)

    def _aggregate_term(self, term, decompose):
        """Flatten a term to ``{label: variable_name}`` over its first sum."""
        sum_op = self._sum_op(term)
        if sum_op is None:
            # No summed decomposition to report: the term is its own answer.
            resolved = self._resolve_var(term)
            return {term.name: resolved} if resolved is not None else {}

        out = {}
        for label, operand in sum_op.terms:
            if isinstance(operand, Constant):
                continue
            if isinstance(operand, VarRef):
                if self._in_ds(operand.name):
                    out[label] = operand.name
                continue
            # Expand only if the operand has a sum of its own. Ask the tree
            # rather than inferring it from the shape of the recursive result:
            # a term with no sum returns {its_name: var}, which is
            # indistinguishable from a real one-child sum whose child happens to
            # share the parent's name.
            if label in decompose and self._sum_op(operand) is not None:
                for child, var in self._aggregate_term(operand, decompose).items():
                    out[f"{label}_{child}"] = var
                continue
            resolved = self._resolve_var(operand)
            if resolved is not None:
                out[label] = resolved
        return out

    # -- conveniences -------------------------------------------------------

    def __contains__(self, address):
        """True if ``address`` names something this recipe defines.

        Mirrors what :meth:`var` accepts: a term (by v1 name or path), an
        operation-suffixed name, or a raw diagnostic. ``_lookup`` *returns*
        ``None`` for an unrecognized string rather than raising -- only a bad
        path or type raises -- so testing for an exception alone reported every
        string as present.
        """
        try:
            if self._lookup(address) is not None:
                return True
        except (KeyError, TypeError):
            return False
        return address in self._by_op_name or address in self._diagnostics

    def __getitem__(self, address):
        return self.var(address)

    def _summary(self):
        n_terms = len(self._by_path)
        n_live = sum(1 for v in self.terms().values() if v is not None)
        budgets = ", ".join(self.budgets)
        scope = "planned" if self._ds is None else f"{n_live}/{n_terms} materialized"
        return f"BudgetQuery: {budgets} ({scope})"

    def __repr__(self):
        return f"<{self._summary()}>"

    def _repr_html_(self):
        """Collapsible tree, annotated with each term's resolved variable name.

        Terms whose diagnostics were missing from the dataset resolve to
        ``None`` (see :meth:`_resolve_var`) and are greyed out, so the display
        shows both the recipe's structure and what this run actually
        materialized.
        """
        from .display import render_budgets_html

        return render_budgets_html(
            self.budgets, resolve=self._resolve_var, header=self._summary()
        )
