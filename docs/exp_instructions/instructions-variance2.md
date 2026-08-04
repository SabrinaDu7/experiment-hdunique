2026-08-04

# variance2: How does variance in D partition across bouts, sessions and mice?

## Question

A session is not one continuous stretch of REM but a handful of episodes minutes apart. Does the
variance in *D* live between mice, between sessions of a mouse, or between **bouts of a session**?

## Motivation

`variance1` models one *D* per session, which silently assumes a session's *D* is a well-defined
quantity. If bouts within a session disagree, that assumption fails: the session-level *D* is an
average over a distribution, and what the two-level model calls "within-mouse variance" is partly
bout-level noise rather than session-to-session biology.

A second question follows immediately. If bouts do differ, is that **biology or decode quality**? A
bout whose ring is poorly sampled would show an inflated *D* for reasons having nothing to do with
the circuit.

## Experiments

- **variance2_exp1** — three-level nested decomposition of log *D*: bout within session within mouse.
- **variance2_exp2** — how much of the bout-level variance is explained by decode quality, and what
  happens to the decomposition when it is regressed out.

## Methods

```bash
uv run hd-exp collect variance2      # builds the per-bout table (minutes)
uv run hd-exp run     variance2
```

Code: `src/experiments/variance2.py`; primitives from `src/analysis/stats.py` (`nested_variance`,
`demean_within`, `residualise`, `correlation`) and `src/analysis/io.py`.

**Input.** `outputs/results/bouts_Mouse<m>_ADn.parquet` — one row per REM bout, with its diffusion
constant and the sleep-architecture context around it.

**exp1 — the decomposition.** Searle's unbalanced nested ANOVA on log *D*. Method of moments rather
than REML because `statsmodels`' crossed variance-component optimiser does not converge on this
design, whereas this estimator is exact and needs no optimiser.

Note the resulting ICC answers a *different* question from `variance1`'s: it is "for a randomly
chosen **bout**", not "for a randomly chosen **session**". A session's *D* averages over its bouts
and is therefore less noisy, so the bout-level ICC is necessarily lower. Neither supersedes the
other.

**exp2 — decode quality.** Three per-bout quality measures, all **independent of the angular
dynamics**: ring radial coefficient of variation, fraction of points falling well inside the ring
(both computed from the cached embedding), and the spread of *D* across ring refits.

`nugget` is deliberately **excluded**. It is the intercept of a fit to the same diffusion curve
whose slope is *D*, so regressing *D* on it is partly tautological. It is reported in the results
for comparison only.

⚠️ The correction in this experiment is **over-aggressive by construction** and its output must be
read with that in mind: a fast-drifting bump is harder to track, so ring scatter and refit spread
are partly *consequences* of *D*. Regressing them out controls for a mediator. `variance1_exp3`
gives the honest control by matching on cell count, which is exogenous.

**Defaults.** `cell_set=ADn`, `d_column=D_200`, `off_ring_frac=0.6`,
`quality_columns=(ring_cv, ring_inward, D_std)`.
