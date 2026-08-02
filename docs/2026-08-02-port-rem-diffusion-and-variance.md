# Porting the REM diffusion + variance pipeline into this repo

**Date:** 2026-08-02
**Source:** `SPUD_Analysis-of-manifold-structure-in-head-direction-data`, package
`reproduce_pynapple/`, inventoried in that repo's
`docs/2026-07-29-port-manifest-rem-diffusion-and-variance.md` (the port manifest).
**Result:** `src/hdunique/` + `src/spud/`, reproducing the sweep and the decomposition from DANDI
alone.

This document records **what was moved, why it was changed, what broke, and what is still open.**
It is the port record, not a results doc — results live in
[`rem-diffusion.md`](./rem-diffusion.md) and
[`variance-decomposition.md`](./variance-decomposition.md); instructions live in
[`REPRODUCING.md`](./REPRODUCING.md).

---

## 1. What was ported

The source package mixed two data paths (CRCNS files and DANDI/pynapple), three analyses (wake
decoding RMSE, REM diffusion, variance decomposition) and a set of untracked one-off scripts. Only
the DANDI REM diffusion sweep and the variance decomposition came across.

| Source | Here | Note |
|---|---|---|
| `shared_scripts/{angle_fns, dim_red_fns, fit_helper_fns, manifold_fit_and_decode_fns}.py` | `src/spud/` | Verbatim. Only edit: `import shared_scripts.X` → `import spud.X`. |
| `read_in_data/rate_functions.py` | `src/spud/kernel_rates.py` | Only the two functions used (`gaussian_wind_fn`, `get_kernel_sum`); the rest was CRCNS file parsing plus `sys.path` manipulation. |
| `env.py` | `src/hdunique/env.py` | Rewritten: named accessors, a stated variable contract, an actionable error when unset. |
| `reproduce_pynapple/config.py` | `src/hdunique/config.py` | Rewritten; see D3, D5. |
| `reproduce_pynapple/loader.py` | `src/hdunique/loader.py` | DANDI half only; see D1. |
| `reproduce_pynapple/rates.py` | `src/hdunique/rates.py` | Plus `bout_lengths`, new; see Q1. |
| `reproduce/run_spud_tests/manifold.py` | `src/hdunique/manifold.py` | Only the pieces the REM path uses. The parallel cross-validation harness was wake-only and is gone. |
| `reproduce_pynapple/main_diffusion.py` (449 lines) | `src/hdunique/diffusion.py` + `sweep.py` + `cli/diffusion.py` | Split: the estimator, the per-session/cache/parquet plumbing, and the CLI. |
| `reproduce_pynapple/variance.py`, `main_variance.py` | `src/hdunique/variance.py`, `cli/variance.py` | Near-verbatim; see D5. |
| `reproduce_pynapple/plotting.py` | `src/hdunique/plotting.py` | Diffusion + variance figures. The wake summary figure needed CRCNS head angles and is gone. |
| `reproduce_pynapple/main_diffusion_grid.py` | `src/hdunique/cli/diffusion_grid.py` | Now shares the estimator; see P2. |
| `throwaway/variance_by_window.py` (untracked) | `src/hdunique/cli/variance_by_window.py` | Promoted into the package and re-derived; see P3. |

**Not ported:** `main_rmse.py` and `main_plot.py` (wake decoding — needs CRCNS `.ang` and
`.states.Wake`), `compare.py` (a CRCNS-vs-DANDI differ, meaningless without CRCNS),
`nb_diffusion_by_mouse.py` (a notebook that re-implemented `pd.read_parquet`),
`throwaway/plot_diffusion_by_cellset.py` (see D2), and `shared_scripts/binned_spikes_class.py` +
`general_file_fns.py` (CRCNS-only, and the former carries a known off-by-one shank-index bug).

Line count for the pipeline itself fell from ~1 400 to ~1 050 while gaining a module boundary
between the estimator and its plumbing.

---

## 2. Decisions

These were put to the user with the port manifest's flags and answered explicitly.

### D1 — DANDI only; the CRCNS path is dropped

The source package could read REM epochs from either CRCNS `.states.REM` or the DANDI `states`
table, and those two scorings disagree substantially (on Mouse28-140313, 23 bouts / 1389 s versus
10 bouts / 803 s, which swings D by ~2.7×). The published sweep used DANDI. **Decision: keep DANDI,
delete the CRCNS half.**

Consequence, stated everywhere it matters: **absolute D here is not directly comparable to the
paper's reported values**, because the paper's REM scoring is not the one being used. The parquet
now carries a `rem_source` column so a row can never be silently misattributed — the manifest's
flag 7.1 was precisely that this provenance existed only in prose.

### D2 — ADn is the analysis cell set

Postsubiculum alone does not form a usable ring (D = 1.3–8.3 across sessions, with large
refit-to-refit spread), and the union mostly adds variance without moving D. **Decision: ADn is the
analysis set.** `cell_areas` remains a config knob so PoS and the union stay reachable, and the
shipped parquets retain those rows so the claim above is sourced rather than asserted. The source
repo's dedicated cell-set comparison plot was dropped: the parquet already carries the numbers, and
a figure supporting a decision already made is not worth a script.

This also retires the source docs' "all cells per session (ADn ∪ PoS)" variance table, which had no
code behind it (manifest flag 7.9) and whose lower ICC was traced to two near-noise PoS-only
sessions leaking into the within-mouse component.

### D3 — Published settings are the defaults

In the source, `DiffusionConfig` defaulted to `n_restarts=5, n_bootstrap=200` while the published
parquets recorded `n_restarts=10, n_bootstrap=0` — so running the sweep as documented did not
reproduce the documented tables (manifest flag 7.4). **Every default here is the published
setting**, and `uv run hd-diffusion --scope all` with no flags is the command of record.

### D4 — The 200 ms fit is genuinely forced through the origin

See P2 — this is both a decision and the one change that moves the numbers.

### D6 — Both diffusion estimates are co-headline

Rather than designate one estimator, the sweep reports **`D`** (pairs pooled across the concatenated
trace, as the source pipeline did) and **`D_bout_aware`** (cross-bout pairs excluded) side by side,
with the variance decomposition runnable on either via `--estimator`. See Q1, now resolved.

### D5 — The bootstrap is reimplemented to the paper's specification

The source implementation resampled per-bin squared angular changes at lags 1 and 2, on a single
refit's decoded angle — a different statistic wearing the paper's name — and it was switched off in
the published sweep anyway, leaving `D_boot_lo`/`D_boot_hi` NaN in every row. Rather than port a
mislabelled statistic, **the paper's procedure was written from scratch** (`diffusion.bootstrap_ci`):
resample whole 200 ms epochs with replacement, as many as the data holds, recompute *D* from each
resampling, take the 2.5/97.5 percentiles, 1000×.

Resampling whole epochs rather than individual pairs is the point — it preserves the within-epoch
correlation between lags, making it a bootstrap over the *data* rather than over the summary
statistic. Validated on a synthetic random walk with a known slope of 1.600: the estimator recovers
1.5951 and the interval [1.5727, 1.6284] covers the truth.

**An epoch cannot straddle a bout boundary**, so the interval this returns is a CI for
`D_bout_aware`, not for `D` — an independent argument that the bout-aware quantity is what the
paper's own method targets. Columns are named `D_bout_ci_lo` / `D_bout_ci_hi` accordingly.

`D_std` is still reported alongside, and the two measure different things: the CI covers sampling
variability of *one* ring; `D_std` covers ring-to-ring variability. In the three sessions with the
largest `D_std` the CI excludes the multi-refit mean, which is informative rather than contradictory.

---

## 3. Problems found and fixed

### P1 — `bout_lengths` did not exist, so bout structure was unrecoverable

The rate matrix concatenates REM bouts, and nothing downstream recorded where the seams were. Added
`rates.bout_lengths` and `sweep.truncate_bouts` (the `n_samples=15000` cap truncates the
concatenated trace, so the bout structure has to be truncated to match). Cached alongside the
decoded angles.

**This turned out to be a correctness check in its own right:** across all 64 imported cache
entries, the independently recomputed bout lengths summed to *exactly* the stored embedding length.
That is strong evidence that this repo's binning reproduces the source's, since the two were
computed by different code from different starting points.

### P2 — Two different "through the origin" fits, neither of which was through the origin

The source had `main_diffusion._origin_slope`, which prepended `(0,0)` as a **data point** to a
`np.polyfit` with a **free intercept**, and `main_diffusion_grid._fit_through_origin`, a genuinely
forced zero-intercept fit. The parquet used the first; the grid figures used the second; both were
described as "slope through the origin". So the number on the figure was not the number in the
table.

**Fixed:** one estimator, `diffusion.slope_through_origin`, genuinely forced (`D = Σxy/Σx²`), used
by the sweep, the grid and every figure. At zero lag the squared angular change is exactly zero, so
the intercept is not a free parameter — forcing it is the physically correct choice, and it is what
the user chose.

**This changes the published numbers.** For the 200 ms window the two estimators are, with
c₀ = ⟨Δα²⟩(100 ms) and c₁ = ⟨Δα²⟩(200 ms):

| | formula | uses |
|---|---|---|
| old (free intercept, origin as a point) | `D = 5·c₁` | **only the 200 ms point** |
| new (forced through origin) | `D = 2·c₀ + 4·c₁` | both measured points |

The old fit's algebra collapses to a single data point — the 100 ms measurement had no influence on
D at all. Measured effect over the 64 imported runs:

| | value |
|---|---|
| median change | **+1.05 %** |
| mean change | +0.89 % |
| range | −4.3 % … +15.9 % |
| max \|ΔD\| | 0.36 rad²/s |

The change is **systematic, not noise**: its sign tracks the sign of the `nugget` (the free-
intercept fit's intercept) in 59 of 64 runs — 18/18 for negative nuggets. That is exactly what the
algebra predicts, since a positive nugget means the 100 ms point sits above the diffusive line, and
only the new estimator sees that point. No conclusion in either results doc changes.

The `r2` columns also change definition: r² is now measured against the zero-intercept model
(SS_tot = Σy²), consistent with the model actually fitted. `nugget`, `D_freeint` and `r2_freeint`
are unaffected — they are a deliberately free-intercept diagnostic — and were verified identical to
the source's to 7e-16.

### P3 — The fit-window ICC table was not reproducible as captioned

The source doc's "ICC is invariant to the fit window" table was captioned as the ADn ≥15 gate over
23 sessions, but the only script that produced it (untracked, in `throwaway/`) gated on
`n_cells >= 10` and additionally reported rounds gated on `nugget ≤ t` — a gate that **selects on
the outcome**, since the nugget is the intercept of the same fit whose slope is D
(the script's own docstring reported spearman(nugget, log D) = 0.861).

**Fixed:** promoted to `cli/variance_by_window.py`, tracked, gating on the same ADn cell count as
the headline decomposition and nothing else. The nugget-gated rounds are not reproduced; a gate
that conditions on the outcome does not belong in a variance decomposition. The table is re-derived
in [`variance-decomposition.md`](./variance-decomposition.md).

### P4 — The cache trusted decoded angles it had not verified

`_cache_path` keyed on `(session_id, cell_set)`, and the stored signature covered only the
*embedding* parameters. The decoded angles depend on `n_restarts`, `n_refits`, `fit_frac`, `seed`
and the ring-fit parameters, none of which were checked — so re-running at a different `n_restarts`
silently reused another setting's decodes, and `--recompute-diffusion-only` would rebuild a parquet
from them without complaint.

**Fixed:** `sweep.cache_signature` covers every parameter the decode depends on. A mismatched entry
is recomputed, not trusted.

### P5 — Stale names and dead code

`main_rmse.py` documented itself as `m3_main`, `main_plot.py` as `plot_summary`, and `main_rmse.py`
contained a bare `1` statement — a stray no-op. All resolved by not porting those modules. The
source's `run_spud_tests` also read a module constant `_DALPHA` while its callers passed
`cfg.dalpha`; here `dalpha` comes from the config on every path.

---

## 4. Validation

Three independent checks, in increasing strength.

### 4.1 Bout structure agrees (all 64 imported runs)

As described in P1: independently recomputed bout lengths sum to exactly the stored embedding
length in every entry (64 in the bulk import, plus Mouse28-140313 ADn imported separately = 65).
Confirms binning, epoch reading and the `n_samples` truncation all match.

### 4.2 The free-intercept diagnostics are bit-comparable

`nugget` recomputed here matches the source's parquet to **7 × 10⁻¹⁶** across all 64 bulk-imported
runs. Since the
nugget is a least-squares fit to the diffusion curve, this confirms the diffusion curve itself —
and therefore the decoded angles and everything upstream — is being read identically.

### 4.3 A cold recompute from the NWBs reproduces the imported decode

The strongest check: `throwaway/verify_cache.py` re-runs a whole session from the NWB files —
rates, Isomap, all 50 ring fits, decode — and diffs against the imported cache. Two sessions were
run cold, one small and one large:

| Session | points | runtime | max \|Δembed\| | max \|Δdecoded\| | D cold | D imported |
|---|---|---|---|---|---|---|
| Mouse28-140317 ADn | 1 927 | 6 min | 1.9e−06 | 2.4e−07 | 1.220138 | 1.220138 |
| Mouse28-140313 ADn | 8 020 | 18 min | **0** | **0** | 0.3442509925 | 0.3442509925 |

Mouse28-140313 was recomputed through the full CLI, which rewrote the cache entry; diffing that
against a copy of the imported entry gives **max |Δ| = 0 on both the embedding and the decoded
angles** — bit-identical, with identical bout lengths and identical cache signatures. (The
1.9e−06 on the smaller session is float32 cache storage precision, since that comparison was made
against the file rather than between two files.)

**Conclusion: the imported cache is exactly what this repo's code produces from the raw data.** The
pipeline is deterministic at fixed seed, as intended — `spud.fit_manifold` seeds nothing of its own,
so `sweep.run_session` seeds the global NumPy RNG per refit.

### 4.4 Downstream results reproduce

The variance decomposition at the headline gate lands within 0.01 of the source repo's:

| | source | here | Δ |
|---|---|---|---|
| sessions / mice | 23 / 6 | 23 / 6 | — |
| τ² | 0.280 | 0.290 | +0.010 |
| σ² | 0.193 | 0.204 | +0.011 |
| ICC | 0.592 | 0.587 | −0.005 |
| ANOVA ICC | 0.584 | 0.579 | −0.005 |

The small shifts are the expected consequence of P2 (the origin-forced estimator), not drift.

### 4.5 One genuine discrepancy: the source's ungated sensitivity is stale ⚠️

The **ungated** sensitivity fit does *not* reproduce, and the reason is a real inconsistency in the
source repo rather than an error here:

| | source | here |
|---|---|---|
| sessions | 30 | **32** |
| τ² | 0.399 | 0.302 |
| σ² | 0.237 | 0.305 |
| ICC | **0.627** | **0.498** |

The source's variance doc (dated 2026-07-16) was computed before two sessions —
**Mouse20-130520** (10 ADn, D = 0.93) and **Mouse24-131216** (14 ADn, D = 2.87) — were added to the
dataset by a later NWB re-download. The source's *diffusion* doc lists both; its *variance* doc
predates them and was never re-run. 32 is the correct current count.

The gated headline is untouched, because both sessions fall below the ≥15 cell gate. Only the
ungated sensitivity moves, and it moves **a lot**, in an interpretable direction: Mouse20-130520's
D = 0.93 is wildly out of line with Mouse20's other sessions (2.8–7.3), so adding it inflates
within-mouse variance (σ² 0.237 → 0.305) and deflates between-mouse variance (τ² 0.399 → 0.302).

This *strengthens* the source's own argument. That doc argued the ungated ICC was inflated because
undersampling is confounded with mouse identity; with the complete session set the ungated ICC
falls from 0.627 to 0.498, much closer to the gated values, exactly as that argument predicts.

---

## 5. Open questions for the user

### Q1 — Cross-bout pairs inflate D ✅ RESOLVED: both estimates are co-headline

`spud.get_diffusion_curve` is applied to the **concatenated** decoded angle, so at lag *k* it forms
*k*·(*B*−1) pairs that straddle bout boundaries, where *B* is the bout count. Those pairs are
separated by minutes of unscored sleep, so their squared angular difference is ~π²-scale — orders
of magnitude above the ~0.1 rad² the curve is measuring.

The source pipeline did this, and the published numbers include it. I did **not** change the
headline, because excluding them is a methodological change and yours to make. Instead the sweep
now reports `D_bout_aware` alongside `D`: the identical estimator with boundary-crossing pairs
excluded.

**It is not negligible, and it is not random.** Across the 31 ADn sessions:

| | value |
|---|---|
| direction | **negative in every single session** — D is always biased *up* |
| median | −1.7 % |
| range | −7.8 % … −0.1 % |
| \|change\| > 5 % | 6 of 31 sessions |

The bias is largest for **low-D sessions**, as expected: a roughly fixed absolute contamination is a
larger fraction of a shallower curve. The worst affected are exactly the clean, slow sessions the
analysis leans on — including **Mouse25-140130, the paper-target session, at −5.2 % (1.011 →
0.959)**. In the ADn+PoS set Mouse28-140313 moves −20 % (0.266 → 0.212).

No conclusion in either results doc flips: Mouse25-140130 remains ~2× the paper's 0.52, the
low-cell sessions remain inflated, and the ICC is a ratio of variances of log D so a near-uniform
multiplicative shift largely cancels. But the diffusion constants are biased upward by a knowable,
one-directional amount, and the affected sessions are not a random subset.

**Resolved: both are reported as co-headline estimates** (D6), with `D_bout_aware` carrying the
confidence interval and flagged as the one to trust where they disagree materially. The variance
decomposition runs on either and **the ICC is insensitive to the choice** (0.587 vs 0.582 at the
≥15 gate), so nothing downstream hinges on it.

Corrected figures after the full 32-session table was complete: the range is **−12.9 % to −0.1 %**
with **7** sessions beyond 5 %, not the −7.8 % / 6 sessions first reported — that earlier figure was
computed while Mouse28-140313 (the worst affected, −12.9 %) was still being recomputed and was
missing from the table.

### Q2 — The paper's epoch bootstrap ✅ RESOLVED: implemented

See D5.

---

## 6. Provenance of the shipped cache

The decode cache **is** shipped (tracked in git, ~22 MB), so `--recompute-only` and the figure
commands work on a fresh clone without the multi-hour sweep.

It was **imported** from the predecessor repo (`throwaway/migrate_cache.py`) rather
than recomputed during the port, because the decode is the expensive step (~50 ring fits per
session, 8–15 minutes each) and the change this port makes to the estimator is strictly downstream
of it. The import is justified only by §4, which is why §4.3 recomputes cold from the raw data
rather than taking the import on trust.

The cache is a convenience, not a dependency: delete it and `hd-diffusion --scope all` rebuilds
everything from the NWB files.
