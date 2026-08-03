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
- **Confinement versus heterogeneity.** Heterogeneity is now demonstrated, but the decisive test
  proposed earlier — within-bout α versus pooled α — has not been run.

## 6. Open next steps

1. **Within-bout α.** If per-bout α ≈ 1 while pooled α ≈ 0.53, the sub-diffusion is *entirely* a
   mixing artefact. Cheap, decisive.
2. **Is bout *D* explained by bout decode quality?** Regress per-bout log *D* on per-bout `nugget`.
   If it absorbs most of the bout-level variance, that 35 % is measurement, not biology.
3. **Bout-level ICC with proper CIs**, by parametric bootstrap over the nested design.
