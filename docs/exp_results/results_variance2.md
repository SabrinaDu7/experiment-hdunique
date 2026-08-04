2026-08-04

<!-- GENERATED from results_variance2.in — do not edit; edit the .in and re-run `hd-exp render`. -->

# variance2 — results

> **Question.** A session is not one continuous stretch of REM but a handful of episodes minutes
> apart. Does the variance in *D* live between mice, between sessions of a mouse, or between
> **bouts of a session**?

Methods: [`instructions-variance2.md`](../exp_instructions/instructions-variance2.md).

**537 bouts across 32 sessions and 6 mice.**

---

## variance2_exp1 — three-level decomposition

| component | variance | share |
|---|---|---|
| between-mouse | 0.325 | **34.3 %** |
| between-session, within mouse | 0.289 | **30.6 %** |
| **between-bout, within session** | 0.332 | **35.1 %** |

ICC(mouse) = 0.343; ICC(mouse + session) = 0.649.

Within-session spread: median SD of log *D* = 0.482; median max/min *D* across a
session's bouts = **6.4×**, worst 42×.

### Interpretation

**Bout-level variance is the largest single component, and it was previously invisible** — the
two-level model folded it into the session residual.

The consequence is that a session's *D* is not a session property. It is an average over a
distribution with a several-fold internal range, so any statement about "between-session variance"
is partly a statement about how many bouts a session happened to contain.

Read the ICC carefully: 0.343 answers *"for a randomly chosen **bout**"*, whereas
`variance1`'s ICC answers *"for a randomly chosen **session**"*. A session averages over its bouts
and is less noisy, so the bout-level figure is necessarily lower. They are not in conflict.

## variance2_exp2 — is that bout-level variance biology or decode quality?

Correlations with log *D*, computed **within session** so that between-session differences cannot
carry them:

| measure | rho_within_session | p |
|---|---|---|
| ring_cv | 0.347 | 0.000 |
| ring_inward | 0.474 | 0.000 |
| D_std | 0.403 | 0.000 |

Together these explain **33.0 %** of within-session variance.

Decomposition after regressing them out:

| component | share (raw) | share (quality-corrected) |
|---|---|---|
| between-mouse | 34.3 % | **2.2 %** |
| between-session | 30.6 % | 18.6 % |
| between-bout | 35.1 % | 79.1 % |

ICC(mouse) falls from 0.343 to **0.022**.

Per-mouse quality:

| mouse | cells | ring_cv | off_ring | D |
|---|---|---|---|---|
| 12 | 40.000 | 0.288 | 0.063 | 0.782 |
| 17 | 25.000 | 0.359 | 0.120 | 2.089 |
| 20 | 9.000 | 0.444 | 0.087 | 3.462 |
| 24 | 16.000 | 0.483 | 0.209 | 2.438 |
| 25 | 18.000 | 0.335 | 0.100 | 0.646 |
| 28 | 23.000 | 0.263 | 0.059 | 0.453 |

Correlation between cell count and ring scatter across bouts: ρ = -0.487.

### Interpretation

A third of bout-level variance is decode quality, established with measures **independent of the
angular dynamics** — so this is not the tautology that regressing *D* on its own `nugget` would be.

⚠️ **But the apparent collapse of between-mouse variance to 2.2 % must not be
read as "mice do not differ".** Quality tracks cell count, which is fixed by the implant, so the
correction removes real between-mouse variance along with the artefact. Worse, ring scatter and
refit spread are partly *consequences* of fast drift — a fast-moving bump is harder to track — so
regressing them out controls for a mediator.

The honest control is to match on cell count, which is exogenous. That is `variance1_exp3`, and
there the mouse effect **survives**.

---

## Answer to the question

**Variance is spread roughly evenly across all three levels**, with the bout level slightly the
largest — and it had not been measured before. The practical consequence is that a session's *D* is
a summary of a distribution rather than a fixed quantity, so the two-level model's "within-mouse"
component conflates session biology with bout-level noise.

About a third of the bout-level variance is decode quality. The remainder is either real
physiological variation or quality that these three measures do not capture.

## How to reproduce

```bash
uv run hd-exp collect variance2
uv run hd-exp run     variance2
uv run hd-exp check   variance2
```

## Next steps

- The two thirds of bout-level variance not explained by quality is unattributed. A quality measure
  independent of both cell count and the dynamics would settle it.
- Bout-level ICC with proper intervals, by parametric bootstrap over the nested design.

## Provenance

Generated 2026-08-04T21:29:21+00:00 from commit `f0e6970`.

| config | value |
|---|---|
| `cell_set` | `ADn` |
| `d_column` | `D_200` |
| `off_ring_frac` | `0.6` |
| `quality_columns` | `['ring_cv', 'ring_inward', 'D_std']` |

| input | value |
|---|---|
| `input_bouts` | `537` |
