# Between-mouse vs between-session variance of REM diffusion — results

**Date:** 2026-08-02
**Produced by:** `uv run hd-variance`, `uv run hd-variance-by-window`
**Reproduce:** [`REPRODUCING.md`](./REPRODUCING.md) §2
**Input:** the ADn rows of `diffusion_Mouse<m>.parquet` ([`rem-diffusion.md`](./rem-diffusion.md))

Run on **both co-headline estimates** of *D* (`--estimator D` and `--estimator D_bout_aware`). The
conclusion is the same either way; tables below give both.

Each mouse contributes several sessions, each with one diffusion constant *D*. How much of the
spread in *D* is **between mice**, and how much is **between sessions within a mouse**?

---

## The headline is a null result ⚠️

**With ~6 mice, between-mouse and within-mouse variability are not separable.** The ICC lands at
0.47–0.59 depending on the gate, but its confidence interval spans essentially the whole range and
**includes zero at every gate**. The data are consistent with mouse identity explaining none of the
variance in *D*.

The reportable statement is *"these components are not separable at this sample size"* — **not**
*"mice differ from each other"*, and equally **not** *"mice do not differ"*. This is a power limit,
not a fixable analysis choice: no gate, estimator or extra session repairs it.

---

## Model

A random-intercept linear mixed model on log *D*, fitted by REML:

```
log(D) ~ 1 + (1 | mouse)
```

| Component | Meaning |
|---|---|
| **τ²** (random-intercept variance) | between-mouse variance |
| **σ²** (residual variance) | between-session, within-mouse variance |
| **ICC** = τ²/(τ²+σ²) | fraction of variability attributable to mouse identity |

Three modelling choices, all load-bearing:

1. **Sessions are gated on ADn cell count** (default ≥ 15). *D* inflates when the ring is
   undersampled, and cell yield is **confounded with mouse identity** — Mouse20 recorded 5/6/9/10/16
   ADn cells across *every* session — so ungated the artefact loads onto τ² and masquerades as a
   between-mouse difference. (But see the caveat below: cell count is a noisy proxy.)
2. ***D* is log-transformed.** It is positive, spans 0.34–7.3, and has mean-dependent spread. On the
   log scale the components read as *fractional* variability and normality is defensible; on the raw
   scale the few high-*D* sessions would dominate both components.
3. **CIs come from a mouse-level parametric bootstrap, not Wald.** Variance components are bounded
   at zero with skewed sampling distributions, and ~6 mice carry very little information about τ².
   Wald intervals here would be actively misleading.

**σ² is an upper bound on within-mouse biology.** It absorbs everything the mouse intercept does
not, which includes *D*-estimation noise as well as real session-to-session variation. It is also a
single pooled parameter under an assumed-common variance, not an average of per-mouse variances —
the per-mouse variances are in fact wildly heterogeneous, so σ² describes no individual mouse well.

---

## Results (ADn, `n_bootstrap=2000`, seed 0)

**Estimator `D`** (pairs pooled across the whole concatenated trace):

| Gate | sessions | mice | τ² | σ² | **ICC** | ICC 95% CI | ANOVA ICC |
|---|---|---|---|---|---|---|---|
| **≥15 ADn** (default) | 23 | 6 | 0.290 | 0.204 | **0.587** | **[0.003, 0.857]** | 0.579 |
| ≥20 ADn | 17 | 4 | 0.193 | 0.216 | **0.471** | **[0.000, 0.825]** | 0.505 |
| ungated (sensitivity) | 32 | 6 | 0.302 | 0.305 | **0.498** | **[0.000, 0.771]** | 0.488 |

**Estimator `D_bout_aware`** (cross-bout pairs excluded):

| Gate | sessions | mice | τ² | σ² | **ICC** | ICC 95% CI | ANOVA ICC |
|---|---|---|---|---|---|---|---|
| **≥15 ADn** (default) | 23 | 6 | 0.313 | 0.224 | **0.582** | **[0.001, 0.855]** | 0.574 |
| ≥20 ADn | 17 | 4 | 0.211 | 0.241 | **0.467** | **[0.000, 0.822]** | 0.499 |
| ungated (sensitivity) | 32 | 6 | 0.322 | 0.327 | **0.496** | **[0.000, 0.770]** | 0.486 |

**The ICC is insensitive to which estimate is decomposed** — 0.587 vs 0.582, 0.471 vs 0.467,
0.498 vs 0.496. Both components rise by ~8 % under `D_bout_aware` (the cross-bout bias is not
perfectly uniform, so removing it slightly widens the spread of log *D*), but their **ratio is
unchanged**. The cross-bout issue in *D* therefore does not touch the variance conclusion at all.

**The ANOVA cross-check agrees to within 0.04 at every gate**, so the decomposition itself is sound
— the uncertainty is sampling, not method.

Figure: `variance_by_mouse_ADn_min15_200ms_D.png` (and `..._D_bout_aware.png`). Panel A shows every session as a dot with the mouse
mean as a bar (spread of the bars is between-mouse; spread within a column is within-mouse; open
marks flag < 20 cells). Panel B shows each mouse's shrunken estimate ± 1.96 conditional SD —
intervals straddling the grand-mean line is the visual form of "the ICC CI includes zero". *No
violin or box plot: with 1–8 sessions per mouse a KDE would invent a confident-looking density from
one or two points.*

### Sessions per mouse under each gate

| Mouse | ≥15 ADn | ≥20 ADn | D range (≥15 gate) |
|---|---|---|---|
| Mouse12 | 5 | 5 | 0.447 – 1.279 |
| Mouse17 | 8 | 7 | 1.195 – 3.458 |
| Mouse20 | 1 | 0 (dropped) | 2.843 |
| Mouse24 | 2 | 0 (dropped) | 2.047 – 2.078 |
| Mouse25 | 4 | 2 | 0.555 – 1.011 |
| Mouse28 | 3 | 3 | 0.344 – 1.220 |
| **total** | **23 / 6 mice** | **17 / 4 mice** | |

### Per-mouse shrunken estimates (≥15 gate)

| Mouse | n | BLUP | cond. SD | mean log D | raw mean log D |
|---|---|---|---|---|---|
| Mouse28 | 3 | −0.504 | 0.235 | −0.308 | −0.426 |
| Mouse25 | 4 | −0.464 | 0.208 | −0.268 | −0.349 |
| Mouse12 | 5 | −0.334 | 0.189 | −0.138 | −0.185 |
| Mouse24 | 2 | +0.391 | 0.275 | +0.586 | +0.724 |
| Mouse17 | 8 | +0.412 | 0.153 | +0.608 | +0.644 |
| Mouse20 | 1 | +0.498 | 0.346 | +0.694 | +1.045 |

Note how hard Mouse20 (n = 1) is shrunk: raw mean +1.045 → +0.694, with the widest conditional SD.

---

## The ICC does not depend on the fit window

The same decomposition on the wider origin-forced windows (`D_300`…`D_500` — the **saturation
diagnostics** from [`rem-diffusion.md`](./rem-diffusion.md), *not* alternative estimates of *D*):

| Window | τ² | σ² | **ICC** | ICC 95% CI | ANOVA ICC |
|---|---|---|---|---|---|
| **200 ms** (headline) | 0.290 | 0.204 | **0.587** | [0.003, 0.857] | 0.579 |
| 300 ms | 0.252 | 0.171 | 0.595 | [0.002, 0.860] | 0.587 |
| 400 ms | 0.218 | 0.148 | 0.595 | [0.002, 0.860] | 0.587 |
| 500 ms | 0.190 | 0.130 | 0.594 | [0.002, 0.859] | 0.585 |

Same sweep on `D_bout_aware` — equally flat, at the same level:

| Window | τ² | σ² | **ICC** | ICC 95% CI | ANOVA ICC |
|---|---|---|---|---|---|
| **200 ms** | 0.313 | 0.224 | **0.582** | [0.001, 0.855] | 0.574 |
| 300 ms | 0.273 | 0.188 | 0.592 | [0.002, 0.858] | 0.582 |
| 400 ms | 0.237 | 0.162 | 0.594 | [0.002, 0.859] | 0.585 |
| 500 ms | 0.208 | 0.141 | 0.595 | [0.002, 0.860] | 0.585 |

Both components shrink together as the window widens — *D* saturates, so its spread compresses — but
their **ratio is flat**. The variance conclusion is robust to the window choice; the window only
rescales the absolute variances. (`variance_by_window_ADn_min15_D.csv`, `variance_by_window_ADn_min15_D_bout_aware.csv`.)

---

## Observations

- **Do not over-read the null in the other direction either.** τ²'s lower bound sits at zero partly
  because bootstrap replicates pile up on the τ² = 0 boundary — a known small-group pathology, not
  evidence *for* τ² = 0. Panel A does show visible separation (Mouse17/20/24 high, Mouse25/28 low);
  the model simply cannot certify it exceeds chance from six draws. **"We can't tell" is the honest
  read.**

- **More sessions will not tighten the CI — only more mice would.** τ² is estimated from 6 groups
  (5 d.f.) however many sessions feed it. Adding sessions constrains σ² but cannot isolate τ² any
  better, so the ICC's uncertainty barely moves. The point estimate is also unstable to the gate
  (0.587 vs 0.471), because changing the gate changes the *D* distribution, not just the sample size.

- **⚠️ At the ≥15 gate a meaningful share of τ² rests on mice with 1–2 observations.**
  Mouse20-130517 has exactly 16 ADn cells and survives the gate; it is that mouse's **only** session,
  so it contributes nothing to σ² while acting as a high-leverage point on τ². Mouse24 contributes
  two near-identical sessions. The ≥20 gate is the cleaner design — it removes exactly those two
  thin-column mice, and τ² duly falls 0.290 → 0.193 — but it **costs the paper session**
  (Mouse25-140130, 17 ADn, is itself dropped) and leaves only 4 mice. At ≥20, τ² ≈ σ² (0.193 vs
  0.216): between- and within-mouse variability are indistinguishable.

- **⚠️ The cell-count gate is the weakest link, and a better gate exists.**
  [`rem-diffusion.md`](./rem-diffusion.md) shows cell count is only a *noisy* proxy for ring quality:
  the ≥15 gate discards **Mouse20-130520** (10 cells, D = 0.93, nugget −0.005) and
  **Mouse25-140206** (14 cells, D = 0.91, nugget −0.009), both clean by every other diagnostic,
  while keeping **Mouse24-131217/8** (16 cells, D ≈ 2.0, nugget ≈ +0.05). **Gating on `nugget` would
  be better justified.** That is not done here because a nugget gate **selects on the outcome** —
  the nugget is the intercept of the same fit whose slope is *D* — so it would truncate the *D*
  distribution and produce components that are not comparable to an exogenous gate's. Resolving this
  properly needs a quality measure independent of the diffusion curve.

- **τ² + σ² exceeds var(log D)** at every gate — by 16.0 % at ≥15 (0.494 vs 0.426), 10.8 % ungated
  (0.607 vs 0.548) and 6.2 % at ≥20 (0.409 vs 0.385). Expected, not a bug: REML's precision-weighted grand mean means the components do not decompose the raw marginal
  variance additively under an unbalanced design. It is an order-of-magnitude sanity check only; the
  ANOVA ICC is the real validation.

- Bootstrap `ConvergenceWarning`s are normal and suppressed — replicates landing on the τ² = 0
  boundary are exactly what produces the CI's lower edge near zero. All 2000/2000 replicates
  converged at every gate reported here, and the primary fits converge cleanly.

---

## Differences from the predecessor repo's numbers

| Gate | source | here | why |
|---|---|---|---|
| ≥15 ADn | ICC 0.592 (τ² 0.280, σ² 0.193) | ICC 0.587 (τ² 0.290, σ² 0.204) | the origin-forced estimator shifts D by ~1 % |
| ≥20 ADn | ICC 0.478 | ICC 0.471 | same |
| **ungated** | **ICC 0.627, 30 sessions** | **ICC 0.498, 32 sessions** | **the source figure is stale** |

The source's ungated fit predates two sessions that a later NWB re-download added —
**Mouse20-130520** and **Mouse24-131216** — and was never re-run, while the source's own diffusion
table includes both. 32 is the correct count. Both fall below the ≥15 gate, so the headline is
untouched; only the sensitivity fit moves.

It moves in an interpretable direction: Mouse20-130520's D = 0.93 is far out of line with Mouse20's
other sessions (2.8–7.3), so including it inflates within-mouse variance (σ² 0.237 → 0.305) and
deflates between-mouse variance (τ² 0.399 → 0.302). **This strengthens the original argument** that
the ungated ICC was inflated by an undersampling artefact confounded with mouse identity — with the
complete session set it falls from 0.627 to 0.498, much closer to the gated values.

## Open

- Whether to require a minimum **session count per mouse** (e.g. ≥ 3) instead of, or alongside, the
  cell gate. That would drop Mouse20 and Mouse24 for the right reason — they cannot inform
  within-mouse variance — while keeping Mouse25-140130 at the ≥15 cell gate.
