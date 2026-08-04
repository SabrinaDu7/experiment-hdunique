2026-08-04

<!-- GENERATED from results_variance1.in — do not edit; edit the .in and re-run `hd-exp render`. -->

# variance1 — results

> **Question.** Each mouse contributes several REM sessions, each with one diffusion constant *D*.
> Is the spread in *D* driven more by **which animal** it came from, or by **which session** of that
> animal?

Methods and rationale: [`instructions-variance1.md`](../exp_instructions/instructions-variance1.md).

---

## variance1_exp1 — ICC by cell-count gate

| gate | sessions | mice | tau2 | sigma2 | ICC | ICC_lo | ICC_hi | ANOVA_ICC |
|---|---|---|---|---|---|---|---|---|
| >=15 ADn | 23 | 6 | 0.313 | 0.224 | 0.582 | 0.001 | 0.855 | 0.574 |
| >=20 ADn | 17 | 4 | 0.211 | 0.241 | 0.467 | 0.000 | 0.822 | 0.499 |
| ungated | 32 | 6 | 0.322 | 0.327 | 0.496 | 0.000 | 0.770 | 0.486 |

At the headline gate (≥15 ADn cells): **23 sessions across 6 mice**,
τ² = 0.313, σ² = 0.224, **ICC = 0.582**, 95 % CI
[0.001, 0.855]. The one-way ANOVA cross-check gives 0.574.

Per-mouse shrunken estimates:

| mouse | n_sessions | blup | cond_sd | mean_log_D | raw_mean_log_D |
|---|---|---|---|---|---|
| 28 | 3 | -0.535 | 0.246 | -0.375 | -0.503 |
| 25 | 4 | -0.480 | 0.218 | -0.320 | -0.406 |
| 12 | 5 | -0.332 | 0.198 | -0.171 | -0.219 |
| 24 | 2 | 0.408 | 0.287 | 0.568 | 0.715 |
| 17 | 8 | 0.427 | 0.160 | 0.587 | 0.625 |
| 20 | 1 | 0.512 | 0.362 | 0.672 | 1.040 |

### Interpretation

**The ICC point estimate is not the finding; its interval is.** The ICC sits around 0.6, which taken
alone would say mouse identity explains most of the variance. But the confidence interval reaches
essentially to zero, so the data are also consistent with mouse identity explaining *none* of it.

The ANOVA cross-check agreeing to within ~0.01 matters here: it says the decomposition itself is
sound, so the width is sampling uncertainty rather than a broken estimator. With ~6 mice, τ² is
estimated from 5 degrees of freedom no matter how many sessions feed it — **more sessions cannot
fix this; only more mice can.**

Note also that the lower bound sitting at zero is partly a small-group boundary effect: bootstrap
replicates pile up on τ² = 0. That is not evidence *for* τ² = 0, so the null should not be
over-read in either direction.

## variance1_exp2 — does the fit window change the answer?

| window_ms | sessions | mice | tau2 | sigma2 | ICC | ICC_lo | ICC_hi | ANOVA_ICC |
|---|---|---|---|---|---|---|---|---|
| 200 | 23 | 6 | 0.313 | 0.224 | 0.582 | 0.001 | 0.855 | 0.574 |
| 300 | 23 | 6 | 0.273 | 0.188 | 0.592 | 0.002 | 0.858 | 0.582 |
| 400 | 23 | 6 | 0.237 | 0.162 | 0.594 | 0.002 | 0.859 | 0.585 |
| 500 | 23 | 6 | 0.208 | 0.141 | 0.595 | 0.002 | 0.860 | 0.585 |

ICC ranges from 0.582 to 0.595 across the four windows.

### Interpretation

Both variance components shrink as the window widens — *D* saturates, so its spread compresses —
but their **ratio is flat**. The decomposition is therefore not an artefact of which window's *D*
is decomposed. The window choice rescales the absolute variances and leaves the question of
between- versus within-mouse untouched.

## variance1_exp3 — does the mouse effect survive matching on cell count?

Restricted to sessions with **14-28 ADn cells**: 21 sessions,
5 mice.

| mouse | sessions | median_cells | median_D |
|---|---|---|---|
| 25 | 5 | 18.00 | 0.66 |
| 28 | 4 | 21.50 | 0.91 |
| 17 | 8 | 25.00 | 1.58 |
| 24 | 3 | 16.00 | 2.06 |
| 20 | 1 | 16.00 | 2.83 |

- **ICC = 0.647**, 95 % CI [0.004, 0.889] (ANOVA 0.635)
- **Kruskal-Wallis across mice: p = 0.0079**
- Spread in median *D* across mice: **4.3×**
- Correlation between cell count and *D* within the band: ρ = -0.115

### Interpretation

**The between-mouse difference survives matching, and strengthens.** Within a narrow cell-count
band the mice still differ several-fold in median *D*, the difference is significant by
Kruskal-Wallis, and the ICC is if anything higher than on the full data — while the correlation
between cell count and *D* has collapsed, confirming the confound really was removed.

This is the experiment that rescues the mouse effect from a pessimistic reading. Correcting *D* for
continuous decode-quality measures (ring scatter, refit spread) appears to abolish the between-mouse
variance entirely — but those measures are partly **downstream** of *D*, because a fast-drifting
bump is harder to track. Matching on cell count, which is fixed by the implant and cannot be caused
by the dynamics, is the honest control, and it leaves the effect standing.

---

## Answer to the question

**Two different questions were being asked at once, and they have different answers.**

*Do mice differ?* — **Yes, with support.** At matched cell count the difference is several-fold and
significant (p = 0.0079), and it is not explained by the most obvious recording
confound.

*Is between-mouse variance larger than within-mouse variance?* — **Not answerable here.** The ICC
lands around 0.582 but its interval spans almost the whole range at every gate and every
fit window. That is a power limit set by having ~6 mice, and no gate, estimator or additional
session repairs it.

The defensible summary is therefore: *mouse identity matters, but this dataset cannot say how much
of the variance it accounts for.*

## How to reproduce

```bash
uv run hd-diffusion --scope all      # only if outputs/results/diffusion_Mouse*.parquet is absent
uv run hd-exp run variance1
```

Variants:

```bash
uv run hd-exp run variance1 --estimator D            # the other co-headline estimate of D
uv run hd-exp run variance1 --headline-gate 20
uv run hd-exp run variance1 --matched-band 16 26
uv run hd-exp check variance1                        # re-run and diff against committed values
```

## Next steps

- **More mice is the only fix** for the ICC's interval. Failing that, reframe at the bout level
  where n is in the hundreds — see `variance2`.
- The matched-band result rests on 5 mice, one of which contributes a single session. Worth
  repeating with a minimum session count per mouse.
- Decode quality is confounded with the implant. A quality measure genuinely independent of both
  cell count and the dynamics would let this be tested directly rather than by matching.

## Provenance

Generated 2026-08-04T16:32:42+00:00 from commit `4e7dc54`.

| config | value |
|---|---|
| `cell_set` | `ADn` |
| `estimator` | `D_bout_aware` |
| `gates` | `[15, 20, 0]` |
| `headline_gate` | `15` |
| `matched_band` | `[14, 28]` |
| `mice` | `[12, 17, 20, 24, 25, 28]` |
| `n_bootstrap` | `2000` |
| `seed` | `0` |
| `windows_ms` | `[200, 300, 400, 500]` |

| input | value |
|---|---|
| `estimator` | `D_bout_aware` |
| `input_sessions` | `32` |
