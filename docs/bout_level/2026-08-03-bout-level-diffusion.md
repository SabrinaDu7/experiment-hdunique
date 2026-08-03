# REM diffusion per bout: bouts within sessions within mice

**Date:** 2026-08-03
**Branch:** `sdu/bout-level-diffusion`
**Code:** `src/hdunique/bouts.py`, `src/hdunique/cli/bout_diffusion.py` (`hd-bouts`)
**Reproduce:** [`REPRODUCING-bouts.md`](./REPRODUCING-bouts.md)
**Outputs:** `bouts_Mouse<m>_ADn.parquet` — one row per REM bout

The published pipeline gives **one *D* per session**, pooled over every REM bout it contains. A
session is not a continuous stretch of REM: it is a handful of episodes minutes apart, and nothing
guarantees they share a drift rate. This computes *D* per bout, so the hidden level becomes visible.

**537 bouts across 32 sessions and 6 mice (ADn).**

---

## Headline

| Finding | |
|---|---|
| **Bout-level variance is the largest of the three components** | 35.1 % of all variance in log *D*, and it was previously invisible — folded into the session-level residual |
| **Within-session spread is large** | median session has a **6.4× range** in *D* across its bouts; the worst has 42× |
| **The bout-context effect is a between-session confound** | raw β = −0.499 (p = 7×10⁻⁶) → **−0.053 (p = 0.5)** once session is controlled |
| **The 10 s bout-merge rule is a no-op on this dataset** | smallest REM-to-REM gap is **11 s**, median **597 s** |

---

## 1. Preprocessing: the merge rule is inert here

The plan was to merge REM bouts separated by less than 10 s, on the reasoning that a brief
misscored intrusion should not split one episode into two.

**Measured across all 643 REM-to-REM gaps in the dataset: none is ≤ 10 s.** The minimum is 11 s,
the median 597 s (about 10 minutes), and only 2 gaps (0.3 %) fall below 20 s. DANDI's scoring
evidently enforces a minimum epoch duration, so brief intrusions are already absorbed.

Consequences:

- The merge is **exactly** a no-op at the default, so the existing decode cache stays valid and no
  refit was needed — the whole dataset was analysable immediately rather than three sessions.
- `merge_close_bouts` is implemented anyway and `--merge-gap-s` is exposed, because the rule is
  sound and other scorings may need it. Raising it above 11 s **changes the REM epochs and
  therefore invalidates the cache**; `hd-bouts` refuses rather than silently mismatching.

## 2. Bout context: entry gives no contrast, exit does

| | count |
|---|---|
| REM bouts preceded by Non-REM | 675 / 682 |
| REM bouts preceded by Awake | **7** |
| REM bouts followed by Awake | 590 |
| REM bouts followed by Non-REM | 92 |

REM is essentially always entered from Non-REM, so "before wake" is not a testable contrast in this
dataset. The exit is: a bout either ends in an awakening or returns to Non-REM.

*(Counts here are over all REM epochs in the state tables; the analysis below uses the 537 bouts
that survive the `n_samples` cap and have a finite D.)*

## 3. Three-level variance decomposition

Nested random-effects model on log *D*, bout within session within mouse, variance components by
Searle's unbalanced nested ANOVA (statsmodels' crossed-VC optimiser did not converge on this
design; the ANOVA estimator is exact and needs no optimiser).

| Component | variance | share |
|---|---|---|
| between-mouse | 0.325 | **34.3 %** |
| between-session, within mouse | 0.289 | **30.6 %** |
| **between-bout, within session** | **0.332** | **35.1 %** |

**ICC(mouse) = 0.343**, ICC(mouse + session) = 0.649.

**Read the ICC carefully — it answers a different question from the two-level one.** The published
ICC of 0.587 is *"for a randomly chosen **session**, what fraction of the variance in its D is
attributable to mouse identity?"*. The 0.343 here is *"for a randomly chosen **bout**"*. A session's
*D* averages over its bouts, so it is a less noisy quantity; the bout-level ICC is necessarily lower.
Neither supersedes the other. What is new is the **decomposition of the remainder**: what the
two-level model called within-mouse variance is roughly half session-level and half bout-level.

**This is the finding that matters for the original hypothesis.** The two-level model implicitly
treated a session's *D* as a well-defined per-session quantity. It is not — it is an average over a
distribution with a 6.4× median range. Any statement about "between-session variance" is partly a
statement about how many bouts each session happened to contain and how they were weighted.

### Within-session spread

- median within-session SD of log *D*: **0.48**
- median within-session max/min *D*: **6.4×** (range 1.6× to 42×)

Concretely, Mouse25-140130 — a paper-target session whose pooled *D* is 0.959 — contains bouts
ranging from **0.068 to 2.034**.

**This confirms the mixture prediction** made when interpreting the long-timescale results
([long_D doc](../long_D/2026-08-03-long-timescale-diffusion.md) §"What's actually left"): if *D* is
heterogeneous across bouts, ⟨cos Δα⟩ decays slower than a single exponential and the measured *D*
falls with the fit window, drifting from the mean of the bout *D*s toward the slowest. The predicted
within-session spread was "roughly 0.3 to >1"; the observed median range is 6.4×. So the
sub-diffusion documented at 1–5 s has a concrete mechanism, and it is heterogeneity rather than
confinement — at least in part.

## 4. Context effects: one apparent, none surviving

**Exit state.** Bouts that return to Non-REM look much faster than bouts that end in an awakening —
until session identity is accounted for:

| Model | β (to-wake) | p |
|---|---|---|
| Raw, ignoring session | **−0.499** | 7 × 10⁻⁶ |
| Session as random effect | −0.053 | 0.50 |
| Session fixed effects | −0.032 | 0.68 |

The raw effect is a **between-session confound**: Non-REM-exit bouts are concentrated in
high-*D* sessions (mean log *D* +0.53 versus −0.09 elsewhere), and only 18 of 32 sessions contain
both exit types at all. Within a session, exit context does not predict *D*.

This is worth dwelling on, because it is exactly the failure mode that motivated the analysis: a
plausible biological story with a highly significant p-value that is entirely an artefact of
pooling across sessions.

**Position in the session.** No effect: spearman −0.005 (p = 0.91) raw, +0.030 (p = 0.49)
session-demeaned. *(An apparent increase across the night in the first three sessions inspected did
not survive the full dataset — noted here because it was the initial impression.)*

**Bout duration.** No effect: spearman +0.055 (p = 0.20) raw, −0.056 (p = 0.19) demeaned. This also
retires a specific artefact worry from the long-timescale work — that long bouts dominate long lags,
so a duration–*D* correlation could manufacture the falling *D*(τ). There is no such correlation.

## 4b. Is bout-level variance biology or decode quality? ⚠️

Three per-bout quality measures were tested. `nugget` is **mechanically linked** to *D* (both come
from the same diffusion curve), so it is reported but cannot settle anything. The other two are
independent of the angular dynamics: **ring radial scatter** and **fraction of off-ring points**
are computed from the cached embedding, and **refit spread** (`D_std`) from the disagreement between
ring fits.

| measure | independent of D? | ρ with log D, within session |
|---|---|---|
| `nugget` | **no** — same curve | +0.459 |
| ring radial scatter | yes | +0.347 |
| fraction of off-ring points | yes | **+0.474** |
| refit spread (`D_std`) | yes | +0.403 |

The independent off-ring measure is *as* predictive as the tautological one, so this is a real
effect. Together the independent measures explain **33 % of within-session variance in log D**.

**Correcting for them changes the decomposition drastically:**

| | between-mouse | between-session | between-bout | ICC(mouse) |
|---|---|---|---|---|
| raw log *D* | 0.325 (34.3 %) | 0.289 (30.6 %) | 0.332 (35.1 %) | 0.343 |
| **quality-corrected** | **0.007 (2.2 %)** | 0.060 (18.6 %) | 0.255 (79.1 %) | **0.022** |

Between-mouse variance essentially vanishes. **Almost all of the apparent between-mouse difference
in *D* is explained by how well each animal's ring was sampled.**

**But the correction is not clean, and must not be read as "mice do not differ".** Decode quality is
itself a property of the animal's implant:

| mouse | median cells | ring scatter | off-ring | refit spread | median *D* |
|---|---|---|---|---|---|
| Mouse12 | 40 | 0.288 | 0.063 | 0.076 | 0.78 |
| Mouse28 | 23 | 0.263 | 0.059 | 0.008 | 0.45 |
| Mouse25 | 18 | 0.335 | 0.100 | 0.016 | 0.65 |
| Mouse24 | 16 | 0.483 | 0.209 | 0.648 | 2.44 |
| Mouse17 | 25 | 0.359 | 0.120 | 0.635 | 2.09 |
| Mouse20 | 9 | 0.444 | 0.087 | 0.746 | 3.46 |

ρ(cell count, ring scatter) = −0.487 and ρ(cell count, *D*) = −0.437 across bouts. Cell count is
fixed by the implant, so regressing quality out removes real between-mouse variance along with the
artefact. The defensible conclusion is **not** that mouse identity is irrelevant, but that in this
dataset **between-mouse differences in *D* are not separable from between-mouse differences in
recording quality**. That is a sharper version of the same "cannot resolve" verdict the two-level
model reached — now with a named mechanism rather than just wide error bars.

## 4c. Exit state, the unconfounded tests

**Paired within-session** (13 sessions with ≥3 bouts of each type): median within-session difference
in log *D* (Non-REM-exit minus wake-exit) = **+0.121**, higher in 9 of 13 sessions,
**Wilcoxon p = 0.45**. Consistent with the session-fixed-effects result; no effect.

**Three sessions inspected directly** (`bout_exit_strip.png`), chosen to span the quality range:

| Session | cells | → Awake | → Non-REM | ratio | MWU p |
|---|---|---|---|---|---|
| Mouse12-120808 | 39 | n=10, median 1.26 | n=4, median 1.31 | 1.04× | 0.84 |
| Mouse17-130129 | 25 | n=12, median 1.40 | n=7, median 1.65 | 1.18× | 0.77 |
| Mouse20-130515 | 6 | n=10, median 4.88 | n=8, median 4.88 | 1.00× | 0.83 |

The two distributions sit on top of each other in all three. What the panels *do* show is the thing
that actually matters: the **within-group** spread dwarfs any between-group difference, and the
three sessions occupy completely different *D* ranges (1.3, 1.5, 4.9) that track cell count
(39, 25, 6) rather than anything about bout context.

## 4d. Cell-count-matched subset: the between-mouse effect survives ⚠️ (revises §4b)

§4b showed that regressing out decode quality collapses between-mouse variance to ~2 %. That
correction is **over-aggressive**, and the matched-subset test shows why.

Cell count is the one quality variable that is unambiguously **exogenous** — fixed by the implant,
and impossible for the dynamics to cause. Ring scatter and refit spread are not: a genuinely
fast-drifting bump is *harder to track*, so it produces more off-ring excursions and less stable
fits. Regressing those out therefore controls for a partial **consequence** of *D*, not just a cause.

Restricting to sessions with **14–28 ADn cells** — 21 sessions, 5 mice, per-mouse medians 16–25 —
removes the cell-count confound directly:

| Mouse | sessions | median cells | median *D* | median nugget |
|---|---|---|---|---|
| Mouse25 | 5 | 18 | **0.66** | −0.01 |
| Mouse28 | 4 | 21.5 | **0.91** | −0.00 |
| Mouse17 | 8 | 25 | 1.58 | 0.03 |
| Mouse24 | 3 | 16 | 2.06 | 0.06 |
| Mouse20 | 1 | 16 | **2.83** | 0.08 |

| | value |
|---|---|
| ICC | **0.647**, 95 % CI [0.004, 0.889] (ANOVA cross-check 0.635) |
| Kruskal–Wallis, *D* differs across mice | **p = 0.0079** |
| ρ(cell count, *D*) *within* the band | **−0.115** (confound removed) |

**The between-mouse difference survives, and strengthens** — a 4.3× spread in median *D* at matched
cell count, significant by Kruskal–Wallis. The ICC is if anything higher than on the full data
(0.647 vs 0.587), while its CI stays wide because 5 mice is still 5 mice.

**Revised conclusion.** Mice do appear to differ, and it is not simply cell count. What remains
unresolvable is the *variance ratio*: the ICC point estimate moves around and its interval spans
almost the whole range. So "is there a mouse effect?" has support (p = 0.008); "is between-mouse
variance larger than within-mouse?" does not, and will not at n = 5–6.

## 4e. The long-lag sub-diffusion is NOT a mixing artefact

The decisive test proposed in the long-timescale doc, now run: compute the anomalous exponent α over
1–5 s **within single bouts** (no mixing possible) and compare with the pooled value.

| | median α (1–5 s) | IQR |
|---|---|---|
| pooled across bouts | 0.54 | 0.38–0.75 |
| **within single bouts** | **0.50** | 0.33–0.76 |

Difference −0.004, **Wilcoxon p = 0.98**, higher in 15 of 32 sessions.

**The sub-diffusion is fully present inside a single REM bout.** Bout-to-bout heterogeneity is real
(§3, 6.4× median range) but it does *not* cause the falling *D*(τ) — that prediction, made when
interpreting the long-timescale results, is **wrong** and is corrected here.

So the remaining explanation is genuine confinement: the decoded angle is not freely diffusing on
the ring, something restores it. Consistent with that, occupancy is **not uniform** — median
circular resultant of the decoded-angle distribution is 0.205, with 17 of 32 sessions above 0.2, so
some directions are visited far more than others.

**Whether those preferred directions are real or a decode artefact is open.** Occupancy uniformity
correlates with cell count (ρ = +0.328), i.e. lumpiness is worse with fewer cells, which is what an
unevenly sampled spline ring would produce: nearest-point decoding preferentially assigns points to
densely-fit arcs, manufacturing attraction. That is the leading alternative to a biological
restoring force and is not yet excluded.

## 4f. Synthetic ring control: how much of the sub-diffusion is instrument? (all 32 sessions)

`scripts/synthetic_ring_control.py`. A walk that is **free by construction** is pushed through each
session's own fitted ring and its own *D* calculation. α = 1 means "free walk"; anything less is
manufactured somewhere. Three quantities are separated:

| column | what it isolates |
|---|---|
| `alpha_truth` | the **estimator** alone — α of the synthetic angles before any decoding |
| `alpha_clean` | + the ring **parameterisation** (points placed exactly on the curve) |
| `alpha_noisy` | + **projection of off-ring noise** (empirical residuals resampled) |
| `alpha_real − alpha_noisy` | the remainder: **dynamics** |

### Results

| | median α |
|---|---|
| free walk, before decoding (`truth`) | **1.06** |
| free walk, points exactly on ring (`clean`) | 1.06 |
| free walk + realistic off-ring noise (`noisy`) | **0.88** |
| **real data** | **0.54** |

**How the raw gap (1.00 → 0.54) splits, by median:**

| source | contribution |
|---|---|
| estimator blindness | **−0.055** (none; the estimator slightly *overshoots* on synthetic data) |
| decoder / off-ring noise | **+0.154** |
| **dynamics (unexplained)** | **+0.286** |

**The dynamics are the largest single component, not the instrument.** The estimator's blindness —
which looked decisive on a four-session preview — bites only **2 of 32 sessions**
(Mouse20-130514 and -130515, 5 and 6 cells, the fastest and thinnest in the dataset), where
`alpha_truth` falls to 0.27 and 0.33. Everywhere else the estimator is honest.

### The deficit is robust

`deficit = alpha_real − alpha_noisy`, i.e. how far real data sits below a free walk that went
through the *same* decoder at the *same* speed with the *same* bout structure:

- median **−0.286**, IQR −0.377 to −0.191
- negative in **29 of 32** sessions, Wilcoxon **p = 2.6 × 10⁻⁸**
- **independent of cell count** (ρ = +0.197, p = 0.28) and of speed (ρ = −0.218, p = 0.23)

Restricting to the **15 sessions where the machinery is verified honest** (`truth` > 0.9 *and*
`noisy` > 0.9, so neither estimator nor decoder is doing damage): deficit median **−0.249**,
negative in **15 of 15**, **p = 6.1 × 10⁻⁵**.

**Conclusion: the decoded REM angle is genuinely sub-diffusive.** It is not the ring
parameterisation (`clean` ≈ `truth` everywhere), not the estimator (except in 2 thin sessions), and
not fully explained by projection noise. Something restores the bump toward preferred directions.

### Caveat that bounds this

Off-ring residuals are resampled **i.i.d.**, destroying their temporal correlation. Real off-ring
excursions are likely persistent, which would make the decoder *more* damaging than this control
allows — so `alpha_noisy` is optimistic and the −0.25 deficit is an **upper bound**. Resampling
residuals in contiguous blocks is the fix, and should be done before the effect size is quoted.

The control also starts at the **embedding**, not at spikes, so it does not test rate estimation or
Isomap. A full end-to-end synthetic (spikes from tuning curves) is the stronger version.

## 5. What this does and does not settle

**Settles:**
- Session-level *D* is an average over a broad within-session distribution, not a session property.
- Bout-level variance is a third of the total and was previously mis-attributed.
- The sub-diffusion at long lags has a mechanism consistent with rate heterogeneity.
- Bout context (exit state, timing, duration) does **not** explain within-session spread — tested
  raw, session-demeaned, session-FE, and paired within-session.
- A third of bout-level variance is decode quality, and *between-mouse* variance is almost entirely
  attributable to it (§4b) — though quality is confounded with the implant, so this bounds rather
  than refutes a mouse effect.

**Does not settle:**
- **What the *remaining* bout-to-bout variation is.** Decode quality accounts for 33 % of it (§4b);
  the other two thirds could be real physiological variation or quality these three proxies miss.
- **Whether between-mouse variance exceeds within-mouse variance.** Still 6 mice. The bout level
  adds hundreds of observations but no degrees of freedom at the mouse level, so the CI on the
  mouse component is no narrower than before. This analysis sharpens *what σ² means*; it does not
  add power to the original question.
- **Whether the confinement is biological.** §4e rules out mixing, leaving a genuine restoring
  tendency — but non-uniform ring occupancy could equally come from uneven spline sampling, and
  that correlates with cell count. Distinguishing them needs a synthetic control: decode a *known*
  free random walk through the same fitted ring and see whether α < 1 appears anyway.

## 6. Open next steps

1. ~~Within-bout α~~ — done (§4e): sub-diffusion is real, not mixing.
2. ~~Is bout *D* decode quality?~~ — done (§4b, §4d): partly, but the naive correction over-corrects.
3. **Synthetic ring control.** Push a known free random walk through each session's fitted ring and
   measure α. If α < 1 emerges from a genuinely free walk, the confinement in §4e is a decode
   artefact. This is now the single highest-value test.
4. **Bout-level ICC with proper CIs**, by parametric bootstrap over the nested design.
