# REM diffusion constants — results

**Date:** 2026-08-02
**Produced by:** `uv run hd-diffusion --scope all` → `$OUTPUT_PATH/results/diffusion_Mouse<m>.parquet`
**Reproduce:** [`REPRODUCING.md`](./REPRODUCING.md) §1

*D* is the diffusion constant of the SPUD-decoded head-direction angle during REM sleep, in rad²/s:
the slope of ⟨Δα²⟩ against lag, fitted through the origin over the first 200 ms.

**Two co-headline estimates are reported.** `D` pools every pair in the concatenated trace, as the
source pipeline did. **`D_bout_aware`** excludes pairs straddling a REM-bout boundary — pairs
separated by minutes of unscored sleep, which are not measurements of diffusion at that lag. It is
**lower in every single session** (see below), and it is the quantity the confidence interval
belongs to. Where they disagree materially, `D_bout_aware` is the one to trust.

> ⚠️ **These values use DANDI's REM scoring, not the scoring behind the paper's numbers, so they
> are not directly comparable to the paper's reported *D*.** The two scorings disagree
> substantially on some sessions. See [`methods.md`](../methods.md) §5.

**Settings** (all of them the config defaults): 12-knot ring, best of 10 restarts by `fit_err`,
5 refits, `fit_frac=1.0`, dt = σ = 100 ms, Isomap → 3-D with 5 neighbours, first 15 000 samples,
seed 0. **65 runs across 39 sessions** — 32 ADn, 13 ADn+PoS, 20 PoS. Mouse32-140820 is excluded
(0 cells, 0 REM epochs).

---

## ADn — the analysis set (32 sessions)

⭐ marks a session for which the paper reports a value.

| Session | ADn | bouts | **D** ± std | **D bout-aware** [95% CI] | D: 200→300→400→500 ms | r²: 200→500 | D₅₀₀/D | nugget |
|---|---|---|---|---|---|---|---|---|
| Mouse12-120806 | 39 | 13 | **1.219** ± 0.178 | 1.202 [0.973, 1.319] | 1.22 → 1.18 → 1.14 → 1.10 | 0.999 → 0.997 | 0.90 | +0.017 |
| Mouse12-120807 | 40 | 18 | **1.279** ± 0.179 | 1.258 [1.017, 1.366] | 1.28 → 1.23 → 1.18 → 1.13 | 0.999 → 0.996 | 0.88 | +0.024 |
| Mouse12-120808 | 39 | 14 | **1.220** ± 0.062 | 1.212 [1.080, 1.379] | 1.22 → 1.16 → 1.11 → 1.06 | 0.998 → 0.995 | 0.87 | +0.031 |
| Mouse12-120809 | 49 | 25 | **0.467** ± 0.012 | 0.434 [0.357, 0.513] | 0.47 → 0.48 → 0.49 → 0.49 | 1.000 → 1.000 | 1.05 | -0.006 |
| Mouse12-120810 | 44 | 12 | **0.447** ± 0.006 | 0.421 [0.389, 0.531] | 0.45 → 0.47 → 0.49 → 0.50 | 0.999 → 0.998 | 1.13 | -0.011 |
| Mouse17-130125 | 3 | 28 | **3.354** ± 0.421 | 3.333 [2.943, 3.456] | 3.35 → 3.28 → 3.04 → 2.74 | 1.000 → 0.982 | 0.82 | +0.015 |
| Mouse17-130128 | 19 | 15 | **3.458** ± 0.650 | 3.436 [3.045, 3.573] | 3.46 → 3.13 → 2.84 → 2.56 | 0.998 → 0.978 | 0.74 | +0.137 |
| Mouse17-130129 | 25 | 19 | **1.546** ± 0.230 | 1.524 [1.388, 1.694] | 1.55 → 1.46 → 1.37 → 1.28 | 0.999 → 0.991 | 0.83 | +0.037 |
| Mouse17-130130 | 26 | 22 | **1.195** ± 0.433 | 1.161 [1.025, 1.346] | 1.19 → 1.17 → 1.13 → 1.08 | 0.999 → 0.997 | 0.91 | +0.008 |
| Mouse17-130131 | 22 | 22 | **2.546** ± 0.480 | 2.500 [2.148, 2.519] | 2.55 → 2.46 → 2.35 → 2.23 | 1.000 → 0.995 | 0.87 | +0.029 |
| Mouse17-130201 | 26 | 20 | **3.279** ± 0.578 | 3.257 [2.968, 3.545] | 3.28 → 3.07 → 2.86 → 2.66 | 0.999 → 0.989 | 0.81 | +0.090 |
| Mouse17-130202 | 25 | 24 | **1.666** ± 0.196 | 1.627 [1.422, 1.738] | 1.67 → 1.61 → 1.53 → 1.45 | 1.000 → 0.994 | 0.87 | +0.017 |
| Mouse17-130203 | 28 | 21 | **1.539** ± 0.226 | 1.499 [1.260, 1.640] | 1.54 → 1.50 → 1.44 → 1.37 | 1.000 → 0.996 | 0.89 | +0.013 |
| Mouse17-130204 | 23 | 22 | **1.261** ± 0.390 | 1.234 [0.968, 1.189] | 1.26 → 1.19 → 1.12 → 1.05 | 0.999 → 0.992 | 0.83 | +0.029 |
| Mouse20-130514 | 5 | 12 | **7.331** ± 0.876 | 7.321 [6.949, 7.821] | 7.33 → 6.57 → 5.81 → 5.12 | 0.998 → 0.966 | 0.70 | +0.306 |
| Mouse20-130515 | 6 | 23 | **4.781** ± 0.726 | 4.763 [4.393, 4.871] | 4.78 → 4.59 → 4.24 → 3.86 | 0.999 → 0.984 | 0.81 | +0.052 |
| Mouse20-130516 | 9 | 20 | **3.439** ± 0.700 | 3.421 [3.053, 3.529] | 3.44 → 3.22 → 2.96 → 2.71 | 0.999 → 0.984 | 0.79 | +0.088 |
| Mouse20-130517 | 16 | 18 | **2.843** ± 0.299 | 2.828 [2.645, 3.149] | 2.84 → 2.65 → 2.44 → 2.25 | 0.999 → 0.986 | 0.79 | +0.079 |
| Mouse20-130520 | 10 | 17 | **0.928** ± 0.052 | 0.912 [0.798, 1.005] | 0.93 → 0.93 → 0.89 → 0.85 | 0.999 → 0.996 | 0.91 | -0.005 |
| Mouse24-131216 | 14 | 21 | **2.870** ± 0.511 | 2.849 [2.899, 3.407] | 2.87 → 2.66 → 2.46 → 2.27 | 0.998 → 0.987 | 0.79 | +0.092 |
| Mouse24-131217 | 16 | 16 | **2.078** ± 0.610 | 2.059 [1.598, 1.964] | 2.08 → 1.95 → 1.81 → 1.67 | 0.999 → 0.987 | 0.80 | +0.057 |
| Mouse24-131218 | 16 | 14 | **2.047** ± 0.409 | 2.029 [1.753, 2.110] | 2.05 → 1.94 → 1.82 → 1.71 | 0.999 → 0.991 | 0.83 | +0.049 |
| Mouse25-140124 | 10 | 18 | **2.609** ± 0.282 | 2.591 [2.314, 2.859] | 2.61 → 2.37 → 2.13 → 1.92 | 0.998 → 0.976 | 0.73 | +0.101 |
| Mouse25-140130 ⭐ | 17 | 16 | **1.011** ± 0.003 | 0.959 [0.764, 1.011] | 1.01 → 1.04 → 1.05 → 1.03 | 0.999 → 1.000 | 1.02 | -0.016 |
| Mouse25-140131 | 20 | 22 | **0.701** ± 0.007 | 0.662 [0.521, 0.695] | 0.70 → 0.72 → 0.73 → 0.73 | 1.000 → 1.000 | 1.03 | -0.007 |
| Mouse25-140204 | 22 | 31 | **0.555** ± 0.002 | 0.512 [0.460, 0.609] | 0.56 → 0.58 → 0.60 → 0.60 | 0.998 → 0.999 | 1.09 | -0.015 |
| Mouse25-140205 | 18 | 10 | **0.628** ± 0.004 | 0.606 [0.509, 0.718] | 0.63 → 0.67 → 0.69 → 0.70 | 0.996 → 0.999 | 1.11 | -0.023 |
| Mouse25-140206 | 14 | 13 | **0.906** ± 0.024 | 0.878 [0.738, 1.051] | 0.91 → 0.92 → 0.91 → 0.88 | 0.999 → 0.999 | 0.98 | -0.009 |
| Mouse28-140311 | 14 | 9 | **1.315** ± 0.020 | 1.291 [1.093, 1.665] | 1.31 → 1.31 → 1.29 → 1.27 | 1.000 → 0.999 | 0.96 | +0.001 |
| Mouse28-140313 ⭐ | 24 | 10 | **0.344** ± 0.005 | 0.300 [0.216, 0.347] | 0.34 → 0.39 → 0.43 → 0.45 | 0.991 → 0.992 | 1.31 | -0.023 |
| Mouse28-140317 | 20 | 3 | **1.220** ± 0.821 | 1.195 [0.488, 1.548] | 1.22 → 1.12 → 1.03 → 0.94 | 0.996 → 0.987 | 0.77 | +0.044 |
| Mouse28-140318 | 23 | 18 | **0.663** ± 0.006 | 0.617 [0.412, 0.762] | 0.66 → 0.68 → 0.69 → 0.69 | 0.999 → 1.000 | 1.04 | -0.009 |

Per-mouse figures: `Mouse<m>_rem_diffusion_grid_ADn_200ms.png` (`uv run hd-diffusion-grid`).

---

## Reading the columns

- **`D` ± `D_std`** — the headline, and the spread across the 5 refits. **`D_std` is not a
  confidence interval.** It shrinks by construction as `n_restarts` rises (the block-minimum over
  restarts makes surviving rings converge), so read it as *residual ring instability at these
  settings*. There is no CI on D in this repo — see [the port doc](./2026-08-02-port-rem-diffusion-and-variance.md) §D5.
- **The 200→500 ms progression** is a **linearity probe, not four estimates of D.** All four windows
  refit the *same* diffusion curve over progressively more lags. Flat ⇒ genuinely diffusive;
  monotonically falling ⇒ the curve saturates inside the window (crossover τ_c ≈ π²/6D).
  **`D_500` is not "the better D"** — it is trustworthy only where it barely differs from `D`.
- **`nugget`** — the intercept of a *free*-intercept fit to the three measured lags: the
  decode-jitter floor that the origin-forced fit is obliged to absorb into its slope. **This is the
  best single quality flag.** Clean sessions sit at ≈0; inflated ones are clearly positive.
- **`r2`** at 200 ms spans only 0.991–1.000 across every session and **cannot report a bad fit** —
  two measured points against a one-parameter model. Use `r2_500` and `nugget`.
- **`D_bout_aware` [95% CI]** — the co-headline estimate, with pairs straddling REM-bout
  boundaries excluded, and **the paper's bootstrap**: 200 ms epochs resampled with replacement
  1000×, 2.5/97.5 percentiles. Note the epoch bootstrap is *inherently* bout-aware — an epoch
  cannot straddle a boundary — so the interval belongs to `D_bout_aware`, not to `D`. **This is a
  genuine confidence interval, unlike `D_std`.**
  - It covers sampling variability of one ring, **not** ring-to-ring variability. In the three
    sessions with the largest `D_std` (Mouse17-130204, Mouse24-131216/7) the CI excludes the
    multi-refit mean — read `D_std` and the CI together, they measure different things.
  - Widths are ~24 % of *D* at the median (10–89 %), wider than the paper's ±0.04 on 1.1. The
    squared angular changes have heavy tails: on a synthetic random walk with the same epoch count
    the interval is ~3× narrower, so the excess is real decode jitter, not a bug.

## Observations

**D inflates when the ring is undersampled, but cell count is a noisy proxy for it.** The trend is
real (Mouse17-130125: 3 cells → 3.35; Mouse20-130514: 5 cells → 7.33) but breaks down in the middle:
at **10 ADn cells** Mouse20-130520 gives **0.93** while Mouse25-140124 gives **2.61**; at **14
cells** Mouse25-140206 gives **0.91** but Mouse24-131216 gives **2.87**. Same cell count, ~3× the D.

**`nugget` separates them where cell count fails**, because it measures the actual mechanism:

| cells | session | D | nugget |
|---|---|---|---|
| 10 | Mouse20-130520 | **0.93** | **−0.005** |
| 10 | Mouse25-140124 | **2.61** | **+0.101** |
| 14 | Mouse25-140206 | 0.91 | −0.009 |
| 14 | Mouse24-131216 | 2.87 | +0.092 |

It is not perfect — Mouse17-130125 has D = 3.35 with a nugget of only +0.015 — so treat it as the
best single flag, not a law. **Consequence for gating:** a blunt "drop < 15 ADn cells" rule discards
Mouse20-130520 and Mouse25-140206, which look clean by every other diagnostic, while keeping
Mouse24-131217/8 (16 cells, D ≈ 2.0, nugget ≈ +0.05). **Gating on `nugget` is better justified than
gating on cell count**, and the variance decomposition's cell-count gate should be read with that in
mind.

**`D_std` corroborates.** Stable low-D sessions have near-zero spread (Mouse25-140130 ±0.003,
Mouse25-140204 ±0.002, Mouse28-140313 ±0.005) while inflated few-cell ones swing wildly
(Mouse20-130514 ±0.876, Mouse28-140317 ±0.821, Mouse17-130128 ±0.650). High D and high `D_std`
co-occur — both signalling an under-constrained ring.

**Saturation tracks D exactly as the circular geometry predicts** (τ_c ≈ π²/6D):

| D range | n | median D: 200→500 ms | median r²: 200→500 | D₅₀₀/D |
|---|---|---|---|---|
| D < 1 | 9 | 0.63 → 0.69 (**flat**) | 0.999 → 0.999 | **1.05** |
| 1 ≤ D < 2 | 11 | 1.26 → 1.10 | 0.999 → 0.996 | 0.88 |
| 2 ≤ D < 4 | 10 | 2.86 → 2.26 | 0.999 → 0.986 | 0.80 |
| D ≥ 4 | 2 | 6.06 → 4.49 | 0.999 → **0.975** | 0.75 |

Slow sessions are still linear at 500 ms (τ_c ≈ 3 s, far beyond the window); fast sessions have
saturated well before it. This is *why* the paper fits 200 ms — the shortest window that clears the
smoothing kernel while staying left of the knee.

**Mouse28-140313 ⭐ *rises* across windows** (0.34 → 0.45, ratio 1.31) — the opposite of saturation,
and the signature of the σ = 100 ms kernel suppressing the shortest lag. Its nugget is negative
(−0.023): the 100 ms point sits *below* the diffusive line. So for the slowest sessions D is biased
somewhat low.

**Mouse25-140130 ⭐ is flat to within 4 % across every window with r² ≈ 1.000 throughout.** Under
DANDI REM it sits at 1.011 — about **twice** the 0.52 the paper reports for this session. That gap
cannot be a fit-window artefact, and the predecessor repo established it is not explained by cell
set, cell count, or decode quality either. It remains unexplained.

## Why ADn only

The other cell sets are in the parquets (`--cell-areas PoS`, `--cell-areas ADn PoS`) purely to
source this decision:

- **PoS alone is near-noise.** D = 1.5–8.2 across the 20 PoS runs, with `D_std` up to ±1.37 and
  nuggets reaching +0.39. Postsubiculum alone does not form a usable ring.
- **Adding PoS to ADn barely moves D**, and where it does it mostly adds variance
  (Mouse24-131216: 2.87 → 3.61 with `D_std` ±1.30). The ring is carried by ADn.

## Cross-bout pairs bias D upward

The diffusion curve is computed on the concatenated decoded angle, so at lag *k* it includes
*k*·(*B*−1) pairs that straddle REM-bout boundaries — separated by minutes of unscored sleep, and
contributing ~π²-scale squared differences. `D_bout_aware` excludes them.

**The bias is one-directional: `D_bout_aware < D` in every one of the 32 ADn sessions.** Median
−1.7 %, range **−12.9 % to −0.1 %**, with **7** sessions beyond 5 %. It is worst for the *slow, clean*
sessions the analysis leans on — including **Mouse25-140130 ⭐ at −5.2 % (1.011 → 0.959)** and
**Mouse28-140313 ⭐ at −12.9 % (0.344 → 0.300)** — because a roughly fixed absolute contamination is
a larger fraction of a shallower curve.

No conclusion above changes — the ordering of sessions is preserved and Mouse25-140130 remains ~2×
the paper's 0.52 either way. Both estimates are reported side by side rather than one being
designated *the* estimate; the variance decomposition is run on both and is
[insensitive to the choice](./results-variance-decomposition.md).

Note that the paper's own bootstrap procedure is inherently bout-aware, which is an independent
argument that `D_bout_aware` is the quantity the paper's method actually targets.
