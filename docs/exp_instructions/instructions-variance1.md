2026-08-04

# variance1: Is between-mouse variance in the REM diffusion constant larger than within-mouse variance?

## Question

Each mouse contributes several REM sessions, each with one diffusion constant *D*. Is the spread in
*D* driven more by **which animal** it came from, or by **which session** of that animal?

## Motivation

*D* is being treated as a property of the head-direction circuit. If it were, animals should differ
systematically and sessions within an animal should agree. If instead sessions within an animal
scatter as widely as animals do, then *D* is a property of the recording, not the circuit — and
every downstream claim about it is weaker than it looks.

Two obstacles rule out a naive variance comparison: sessions from one mouse are **not independent**,
and mice have **unequal session counts** (1–8 after gating). A random-intercept mixed model handles
both.

A confound sits on top of this. *D* inflates when the ring is undersampled, and **cell yield is
confounded with mouse identity** — Mouse20 is thin in every session. Ungated, that artefact loads
onto the between-mouse component and masquerades as biology. Hence the cell-count gate, and hence
experiment 3, which asks whether anything survives once cell count is matched.

## Experiments

- **variance1_exp1** — τ², σ² and the ICC at three cell-count gates (≥15, ≥20, ungated).
- **variance1_exp2** — the same decomposition at each fit window, to test whether the ICC is an
  artefact of which window's *D* is used.
- **variance1_exp3** — restrict to a cell-count-matched band of sessions and ask whether the
  between-mouse difference survives.

## Methods

Run everything with:

```bash
uv run hd-exp run variance1                          # defaults below
uv run hd-exp run variance1 --estimator D            # the other co-headline estimate
uv run hd-exp run variance1 --headline-gate 20
```

Code: `src/experiments/variance1.py`. Primitives it reuses: `src/variance.py` (the model),
`src/analysis/io.py` (loading and `log_D`), `src/analysis/stats.py` (Kruskal-Wallis, correlation).

**Input.** `outputs/results/diffusion_Mouse<m>.parquet`, produced by `uv run hd-diffusion
--scope all`. One row per (session, cell set).

**Model.** Random-intercept linear mixed model on log *D*, fitted by REML:

```
log(D) ~ 1 + (1 | mouse)
```

τ² (the random-intercept variance) is between-mouse; σ² (the residual) is between-session within
mouse; ICC = τ²/(τ²+σ²).

Three modelling choices, all load-bearing:

1. ***D* is log-transformed.** It is positive, spans ~0.3–7, and has mean-dependent spread. On the
   log scale the components read as fractional variability and normality is defensible.
2. **Sessions are gated on ADn cell count**, because the undersampling artefact is confounded with
   mouse identity (above).
3. **Intervals come from a mouse-level parametric bootstrap**, not Wald. Variance components are
   bounded at zero with skewed sampling distributions, and ~6 mice carry very little information
   about τ²; Wald intervals here would be actively misleading. A one-way ANOVA ICC is computed
   alongside as an independent cross-check on the decomposition itself.

**exp3 — the matched subset.** Restrict to sessions with 14–28 ADn cells (`--matched-band`), which
leaves 5 mice with per-mouse medians in a narrow range, then refit. Cell count is used because it is
the one quality variable that is unambiguously **exogenous**: it is fixed by the implant and cannot
be caused by the dynamics. Ring scatter and refit spread are *not* exogenous — a fast-drifting bump
is harder to track, so those measures are partly a **consequence** of *D*. Regressing them out
controls for a mediator and destroys real signal, which is why matching is used instead.

Reported alongside: Kruskal-Wallis across mice within the band (does *D* differ at all, without the
variance-partitioning ambition of an ICC), and the within-band correlation between cell count and
*D* (confirming the confound is actually removed).

**Defaults.** `cell_set=ADn`, `estimator=D_bout_aware`, `gates=(15, 20, 0)`, `headline_gate=15`,
`windows_ms=(200, 300, 400, 500)`, `matched_band=(14, 28)`, `n_bootstrap=2000`, `seed=0`.
