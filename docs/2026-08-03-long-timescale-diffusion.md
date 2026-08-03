# REM diffusion at 200 ms, 500 ms and 5 s — methods, wrapping, and results

**Date:** 2026-08-03
**Branch:** `sdu/longer-D-timescale`
**Code:** `src/hdunique/timescale.py`, `src/hdunique/cli/timescale.py` (`hd-timescale`)
**Reproduce:** [`REPRODUCING-timescale.md`](./REPRODUCING-timescale.md)
**Outputs:** `timescale_Mouse<m>_ADn.parquet`, `Mouse<m>_timescale_msd_ADn.png`

The published diffusion constant is fitted over the first **200 ms**. This asks what happens when
the same quantity is measured over **500 ms** and **5 s** instead. All 32 ADn sessions across all
6 mice were run.

**Answer in one line:** the process is diffusive at 200–500 ms and clearly **sub-diffusive** by
1–5 s, so *D* is not a single number — it depends on the window, and by 5 s it has fallen to a
median of **0.33×** its 200 ms value.

Getting there required dealing with circular wrapping, which is not a detail at 5 s: it is the
difference between a measurement and an artefact.

---

## 1. The wrapping problem

The decoded angle lives on a circle. Every estimator here compares α(t+τ) with α(t), and the
difference is taken the short way round, into (−π, π]. That is harmless while the typical
displacement is far below π. It is not harmless at 5 s.

**A wrapped difference cannot exceed π, so ⟨Δα²⟩ is bounded above by π²/3 ≈ 3.29 rad²** — the
variance of a uniformly distributed angle. Any session whose true displacement approaches that
ceiling has its diffusion constant silently compressed toward zero, and no amount of fitting
recovers it.

This is not hypothetical. Measured wrapped ⟨Δα²⟩ at 5 s, as a fraction of the ceiling
(`wrapped_saturation`), runs from **0.32 to 0.81, median 0.51** across the 32 sessions. Half the
dataset is at least halfway into a hard ceiling by 5 s. For Mouse20-130514 the wrapped curve is
essentially flat from 1 s onward (2.40 → 2.45 → 2.69 at 1, 2, 5 s) — it has stopped measuring
anything.

**So a 5 s fit on wrapped differences measures the ceiling, not the dynamics.** It must be
circumvented.

## 2. Two circumventions, and why both are needed

### 2.1 `unwrapped` — lift the trajectory onto the real line

Accumulate the per-bin signed steps: α̃(t) = Σ signed_diff(α(k), α(k−1)). Displacements then grow
without bound and there is no ceiling. This is the natural first answer, and it is what "unwrapping"
usually means.

**It is exact only if no true per-bin step exceeds π.** Where a step does, its direction is guessed
— and because the result is a cumulative sum, the error is carried forward for the rest of the bout.
Decode jitter is thereby integrated into a spurious random walk that never returns.

Diagnostic: **`unwrap_risk`**, the fraction of per-bin steps with |step| > π/2. It ranges from
**0.001 to 0.128** across sessions, i.e. from negligible to one step in eight being ambiguous.

### 2.2 `circular` — never unwrap at all

For a wrapped Gaussian displacement of variance σ², the mean resultant length is exactly
⟨cos Δα⟩ = exp(−σ²/2). So

> **σ²(τ) = −2 · ln ⟨cos Δα(τ)⟩**

recovers the *unwrapped* variance from *wrapped* data, with no unwrapping decision anywhere. There
is no hard ceiling: as the angle decorrelates, ⟨cos Δα⟩ → 0 and the estimate → ∞.

Verified on a synthetic wrapped random walk with a known rate (D = 1.0, so ⟨Δα²⟩ = τ):

| τ (s) | truth | `wrapped` | `unwrapped` | `circular` |
|---|---|---|---|---|
| 0.1 | 0.100 | 0.100 | 0.100 | 0.100 |
| 1.0 | 0.996 | 0.991 | 0.996 | 0.998 |
| 2.0 | 1.997 | 1.842 | 1.997 | 2.008 |
| **5.0** | **5.046** | **2.942** ✗ | **5.046** ✓ | **4.933** ✓ |

`wrapped` under-reads by 42 % at 5 s; both circumventions track the truth.

### 2.3 The two fail in opposite directions, which is what makes them useful

`unwrapped` integrates jitter, so it is an **upper** bound. `circular` assumes Gaussianity, and
non-Gaussian jitter pushes it **down**, so it is closer to a lower bound. Their ratio
**`unwrapped_over_circular`** is therefore a direct measure of how much the long-lag estimate can be
trusted. It is 1.4 for the cleanest sessions and 9.1 for the worst.

## 3. Which circumvention wins — a falsification test, not a preference

The two can be adjudicated rather than merely compared, because they make incompatible predictions
about a *third* measurable quantity.

If the unwrapped MSD at 5 s is correct, then the surviving angular correlation must be
R = ⟨cos Δα⟩ ≈ exp(−MSD/2). R is measured directly. So:

| Session | `unwrap_risk` | R measured | R implied by `unwrapped` | ratio |
|---|---|---|---|---|
| Mouse28-140313 | 0.001 | 0.516 | 0.39 | **1.3** |
| Mouse25-140130 | 0.014 | 0.299 | 0.17 | **1.8** |
| Mouse17-130128 | 0.046 | 0.495 | 8.3 × 10⁻³ | **59** |
| Mouse20-130515 | 0.073 | 0.164 | 1.4 × 10⁻⁵ | **11 758** |
| Mouse20-130514 | 0.128 | 0.196 | 5.7 × 10⁻⁶ | **34 663** |

For the high-risk sessions the unwrapped MSD implies the angle has forgotten where it started
several times over, while the data say it is still ~50 % correlated. **`unwrapped` is refuted for
those sessions, not merely noisy.** Its apparent winding is misresolved steps accumulating.

Meanwhile `circular` remains sensitive everywhere: **R at 5 s spans 0.164–0.652**, never near the
~0.05 floor where −2·ln R would be noise-dominated.

> **`circular` is therefore the estimator of record at long lags**, with `unwrapped` retained as a
> cross-check that is valid only where `unwrap_risk` is small. At 200–500 ms, where nothing wraps,
> the plain `wrapped` estimator remains correct and is what the published `D` uses.

### 3.1 A caveat that cuts the right way

`circular` is exact for a *Gaussian* displacement. At 200 ms the decoded displacement is strongly
leptokurtic — **median kurtosis 16.5** against a Gaussian's 3.0 — a spike of near-zero steps plus
rare jumps. For such a law ⟨cos Δα⟩ is larger than a Gaussian of equal variance would give, so
`circular` **under-reads at short lags**: median ratio 0.75 against `wrapped` at 200 ms. On a
synthetic Gaussian the two agree to 1.000, confirming the machinery rather than the data is
innocent.

By 5 s the central limit theorem has done its work: **median kurtosis 2.99**, i.e. Gaussian to two
decimals. So the bias is largest at 200 ms and gone at 5 s — which **inflates** the measured
D(5 s)/D(200 ms) ratio. The fall reported below is therefore a **conservative bound**; the true
fall is steeper.

## 4. Method as run

Everything downstream of the ring fit is already cached (`outputs/results/cache/*.npz` holds each
run's per-refit decoded angles), so this analysis needed **no refitting** — it is a two-minute
recompute over all 32 sessions.

1. **Split** each session's concatenated decoded trace back into its REM bouts using the cached
   `bout_lengths`. Everything is **bout-aware**: a pair spanning the gap between two bouts is
   meaningless at any lag and catastrophic at 5 s. Bouts are long enough to support it — each session's
   median bout is 17–104 s, and only **2 of 537** bouts across the dataset are shorter than the 5 s lag.
2. **Compute** ⟨Δα²⟩(τ) at every lag from 100 ms to 5 s in 100 ms steps (50 lags), by all three
   estimators, per refit, then average over the 5 cached refits.
3. **Fit** D at each window (200 ms, 500 ms, 5 s) with the repo's existing origin-forced estimator
   `diffusion.window_slope`, unchanged.
4. **Diagnose** with `unwrap_risk`, `wrapped_saturation`, `unwrapped_over_circular`,
   `resultant_at_max_lag`, `kurtosis_200`, `kurtosis_max_lag`, and the anomalous exponent α.

**Cross-check:** recomputing the 200 ms and 500 ms fits from the new `wrapped` curve reproduces the
published `D_bout_aware` and `D_bout_aware_500` columns with **max |difference| = 0.0e+00** across
all 32 sessions. The new code path is the same estimator on a longer lag range.

### The estimator-robust measure

Comparing *D* across windows mixes a real effect with each estimator's bias. The **anomalous
exponent α**, the log-log slope of ⟨Δα²⟩ against τ, does not: ⟨Δα²⟩ ∝ τ^α with α = 1 for ordinary
diffusion, and any *multiplicative* estimator bias cancels out of a log-log slope entirely.

| estimator | α over 0.1–0.5 s | α over 1–5 s |
|---|---|---|
| `wrapped` | 0.87 | 0.41 |
| `circular` | **0.93** | **0.53** |

Both agree: **near-diffusive at short lag, strongly sub-diffusive by 1–5 s.** This is the finding
that does not depend on which circumvention you believe.

---

## 5. Results — all 32 ADn sessions, all 6 mice

*D* in rad²/s from the `circular` estimator. "published D" is the existing 200 ms `D_bout_aware`
(wrapped), for reference. "u/c agree" flags `unwrapped_over_circular` < 2, i.e. sessions where both
circumventions concur. ⭐ = paper-target session.

| Session | cells | published D | D 200 ms | D 500 ms | **D 5 s** | D(5s)/D(200ms) | α short → long | R(5 s) | u/c | u/c agree |
|---|---|---|---|---|---|---|---|---|---|---|
| Mouse12-120806 | 39 | 1.202 | 0.808 | 0.762 | **0.332** | 0.41 | 0.94 → 0.53 | 0.485 | 2.7 | no |
| Mouse12-120807 | 40 | 1.258 | 0.805 | 0.747 | **0.317** | 0.39 | 0.92 → 0.54 | 0.516 | 2.2 | no |
| Mouse12-120808 | 39 | 1.212 | 0.892 | 0.805 | **0.338** | 0.38 | 0.88 → 0.56 | 0.477 | 1.8 | yes |
| Mouse12-120809 | 49 | 0.434 | 0.344 | 0.383 | **0.281** | 0.82 | 1.13 → 0.75 | 0.524 | 1.6 | yes |
| Mouse12-120810 | 44 | 0.421 | 0.377 | 0.432 | **0.344** | 0.91 | 1.16 → 0.86 | 0.432 | 1.6 | yes |
| Mouse17-130125 | 3 | 3.333 | 2.377 | 2.118 | **0.352** | 0.15 | 0.94 → 0.12 | 0.524 | 9.1 | no |
| Mouse17-130128 | 19 | 3.436 | 2.603 | 2.090 | **0.397** | 0.15 | 0.78 → 0.14 | 0.495 | 6.6 | no |
| Mouse17-130129 | 25 | 1.524 | 1.085 | 0.938 | **0.357** | 0.33 | 0.85 → 0.54 | 0.473 | 3.0 | no |
| Mouse17-130130 | 26 | 1.161 | 0.844 | 0.796 | **0.381** | 0.45 | 0.96 → 0.64 | 0.432 | 2.2 | no |
| Mouse17-130131 | 22 | 2.500 | 1.888 | 1.762 | **0.721** | 0.38 | 0.95 → 0.57 | 0.228 | 2.8 | no |
| Mouse17-130201 | 26 | 3.257 | 2.338 | 2.063 | **0.599** | 0.26 | 0.88 → 0.35 | 0.308 | 4.9 | no |
| Mouse17-130202 | 25 | 1.627 | 1.266 | 1.141 | **0.311** | 0.25 | 0.92 → 0.34 | 0.544 | 4.9 | no |
| Mouse17-130203 | 28 | 1.499 | 1.157 | 1.062 | **0.376** | 0.33 | 0.93 → 0.45 | 0.466 | 3.2 | no |
| Mouse17-130204 | 23 | 1.234 | 0.901 | 0.784 | **0.235** | 0.26 | 0.87 → 0.41 | 0.620 | 3.3 | no |
| Mouse20-130514 | 5 | 7.321 | 5.881 | 4.930 | **0.876** | 0.15 | 0.85 → 0.16 | 0.196 | 7.3 | no |
| Mouse20-130515 | 6 | 4.763 | 3.897 | 3.587 | **0.903** | 0.23 | 0.97 → 0.33 | 0.164 | 6.0 | no |
| Mouse20-130516 | 9 | 3.421 | 2.546 | 2.186 | **0.587** | 0.23 | 0.87 → 0.32 | 0.318 | 4.9 | no |
| Mouse20-130517 | 16 | 2.828 | 2.042 | 1.746 | **0.499** | 0.24 | 0.86 → 0.40 | 0.357 | 4.3 | no |
| Mouse20-130520 | 10 | 0.912 | 0.764 | 0.699 | **0.230** | 0.30 | 0.95 → 0.43 | 0.618 | 2.9 | no |
| Mouse24-131216 | 14 | 2.849 | 2.145 | 1.802 | **0.571** | 0.27 | 0.82 → 0.42 | 0.331 | 2.0 | yes |
| Mouse24-131217 | 16 | 2.059 | 1.503 | 1.273 | **0.355** | 0.24 | 0.84 → 0.34 | 0.502 | 3.7 | no |
| Mouse24-131218 | 16 | 2.029 | 1.589 | 1.366 | **0.497** | 0.31 | 0.85 → 0.53 | 0.344 | 3.4 | no |
| Mouse25-140124 | 10 | 2.591 | 1.819 | 1.417 | **0.334** | 0.18 | 0.76 → 0.33 | 0.518 | 4.5 | no |
| Mouse25-140130 ⭐ | 17 | 0.959 | 0.733 | 0.785 | **0.516** | 0.70 | 1.10 → 0.77 | 0.299 | 1.4 | yes |
| Mouse25-140131 | 20 | 0.662 | 0.550 | 0.587 | **0.404** | 0.73 | 1.08 → 0.79 | 0.386 | 1.6 | yes |
| Mouse25-140204 | 22 | 0.512 | 0.438 | 0.486 | **0.351** | 0.80 | 1.15 → 0.81 | 0.437 | 1.8 | yes |
| Mouse25-140205 | 18 | 0.606 | 0.519 | 0.594 | **0.417** | 0.80 | 1.21 → 0.75 | 0.377 | 1.5 | yes |
| Mouse25-140206 | 14 | 0.878 | 0.660 | 0.689 | **0.352** | 0.53 | 1.10 → 0.62 | 0.456 | 2.4 | no |
| Mouse28-140311 | 14 | 1.291 | 0.901 | 0.937 | **0.437** | 0.48 | 1.07 → 0.62 | 0.371 | 2.6 | no |
| Mouse28-140313 ⭐ | 24 | 0.300 | 0.261 | 0.358 | **0.282** | 1.08 | 1.42 → 0.80 | 0.516 | 1.4 | yes |
| Mouse28-140317 | 20 | 1.195 | 0.658 | 0.527 | **0.196** | 0.30 | 0.76 → 0.65 | 0.652 | 1.6 | yes |
| Mouse28-140318 | 23 | 0.617 | 0.437 | 0.493 | **0.365** | 0.84 | 1.16 → 0.75 | 0.429 | 1.9 | yes |

Per-mouse MSD figures: `Mouse<m>_timescale_msd_ADn.png` — log-log, all three estimators, with the
π²/3 ceiling and the 200 ms rate extrapolated as a slope-1 guide.

### 5.1 200 ms → 500 ms: nothing happens

**D(500 ms)/D(200 ms) = 0.92 median, IQR 0.86–1.05.** Widening the window from 200 ms to 500 ms
does not materially change the answer, in any session. Anyone choosing between those two windows is
choosing between equivalent measurements.

This matches the existing `D_500` column in the main sweep, which was already known to be flat for
slow sessions, and extends the observation to every session.

### 5.2 200 ms → 5 s: a large, systematic fall

**D(5 s)/D(200 ms) = 0.33 median, IQR 0.25–0.58, range 0.15–1.08.**

**The fall is strongly predicted by how fast the session was to begin with**
(spearman(D 200 ms, ratio) = **−0.891**, p = 8 × 10⁻¹²):

| | n | median D(5 s)/D(200 ms) |
|---|---|---|
| slow sessions (D₂₀₀ < 1) | 17 | **0.53** |
| fast sessions (D₂₀₀ ≥ 2) | 8 | **0.23** |

Only one session (Mouse28-140313 ⭐, the slowest in the dataset) shows no fall at all
(ratio 1.08). Every other session falls.

### 5.3 The paper-target sessions

| Session | D 200 ms | D 500 ms | D 5 s | α short → long |
|---|---|---|---|---|
| **Mouse25-140130** ⭐ | 0.733 | 0.785 | **0.516** | 1.10 → 0.77 |
| **Mouse28-140313** ⭐ | 0.261 | 0.358 | **0.282** | 1.42 → 0.80 |

Both are among the cleanest in the dataset (u/c = 1.4, low `unwrap_risk`), and both are among the
*least* sub-diffusive — α ≈ 0.8 over 1–5 s where the fast sessions reach 0.12–0.35. Mouse25-140130
still sits well above the paper's 0.52 target at 200 ms under DANDI REM scoring; at 5 s it lands at
0.516, but that coincidence is not evidence of anything, since the paper's number is a 200 ms
measurement under different REM scoring.

---

## 6. Interpretation

**The angle is diffusive on the timescale the paper measures, and confined on the timescale of
seconds.** α ≈ 0.93 over 0.1–0.5 s is ordinary diffusion; α ≈ 0.53 over 1–5 s is not.

Two mechanisms are consistent with this, and the data here separate them only partly:

1. **The ring is bounded.** A walk on a circle cannot spread beyond the circle, so its MSD must
   saturate once it has explored an appreciable fraction of the ring. This is geometry, not biology,
   and it necessarily contributes. It predicts exactly the observed pattern — that faster sessions
   saturate sooner (they explore the ring sooner), which is the −0.891 correlation above.
2. **A genuine restoring tendency**, e.g. attractor dynamics pulling the decoded state back, would
   also produce α < 1, and would do so *before* the ring is fully explored.

**These are not distinguished here.** Mechanism 1 alone is sufficient to explain the whole effect,
so the honest reading is that this analysis establishes *that* D is window-dependent and by how
much, not *why*. Separating the two would need either a control on a synthetic bounded walk with a
matched D, or the confinement length estimated independently.

**The practical consequence stands regardless: "the REM diffusion constant" is not a single
number.** Quoting one requires quoting the window. At 200 ms and 500 ms the answer is stable; at
5 s it is a different, smaller quantity, and for fast sessions the 5 s number is closer to a
statement about the size of the ring than about the rate of drift.

---

## 7. Limitations

- **The 5 s estimate is only fully corroborated in 11 of 32 sessions** (those with u/c < 2). In the
  other 21 the two circumventions disagree by 2–9×, and the reported value is `circular`'s — which
  the R test supports over `unwrapped`, but which remains a single estimator without an independent
  check.
- **`circular` under-reads at short lags** by ~25 % because the displacement is leptokurtic
  (§3.1). Absolute *D* values in the table are therefore not comparable with the published
  `wrapped` 200 ms numbers; the *ratios across windows* and the *exponents* are the meaningful
  quantities. The published column is included in the table only for orientation.
- **No confidence intervals.** The epoch bootstrap in `diffusion.bootstrap_ci` is built for the
  200 ms window and would need a 5 s epoch — which most bouts can supply only a handful of, so the
  CI would be very wide. Not attempted.
- **ADn only**, matching the rest of the repo. The PoS and ADn+PoS caches exist and
  `hd-timescale --cell-set PoS` runs, but PoS is near-noise and the wrapping problem is far worse
  there.
- **Bout length caps the longest usable lag.** 5 s is comfortable (median bout 30–100 s), but the
  approach does not extend much further without losing most bouts.
- **The two mechanisms in §6 are not separated.**

## 8. What changed in the code

Nothing in the existing pipeline. `timescale.py` and `cli/timescale.py` are new; `config.py` gained
`TIMESCALE_WINDOWS_MS` and `TIMESCALE_MAX_LAG`; `plotting.py` gained `plot_timescale_curves`. The
existing estimator, sweep, cache and parquets are untouched, and all previously published artefacts
remain byte-identical.
