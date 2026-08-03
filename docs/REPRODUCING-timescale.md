# Reproducing the long-timescale diffusion results

Every number, table and figure in
[`2026-08-03-long-timescale-diffusion.md`](./2026-08-03-long-timescale-diffusion.md) is produced by
the commands below. Nothing there is quoted without a command here that regenerates it.

Setup (data, environment, `uv sync`) is identical to the main pipeline — follow
[`REPRODUCING.md`](./REPRODUCING.md) §0 first, then come back here.

> **⚠️ The `OUTPUT_PATH` trap.** An exported `OUTPUT_PATH` from another project wins over `.envrc`,
> because the code reads `os.environ` first. Every command below is shown with it set explicitly.
> Check with `echo $OUTPUT_PATH` if results land somewhere unexpected.

```bash
cd experiment-hdunique
export OUTPUT_PATH="$PWD/outputs"     # or rely on .envrc if nothing else is exported
```

---

## 1. The whole analysis, in one command

```bash
uv run hd-timescale
```

**Runtime: ~2 minutes for all 32 ADn sessions.** It reads the decode cache
(`$OUTPUT_PATH/results/cache/*.npz`) and never touches the NWB files, so there is no Isomap and no
ring fitting — the expensive work was already done and cached by `hd-diffusion`.

Writes, per mouse:
- `$OUTPUT_PATH/results/timescale_Mouse<m>_ADn.parquet`
- `$OUTPUT_PATH/results/Mouse<m>_timescale_msd_ADn.png`

and prints one line per session:

```
Mouse25-140130 [ADn]  200ms:0.96/0.73  500ms:1.02/0.79  5000ms:0.72/0.52
                      risk=0.014  sat=0.66  u/c=1.4  R=0.299  α=1.10→0.77
```

where each `Xms:a/b` is D from the `unwrapped`/`circular` estimators at that window.

**Prerequisite:** the decode cache must be present. It ships with the repo; if you deleted it, run
`uv run hd-diffusion --scope all` first (hours — see [`REPRODUCING.md`](./REPRODUCING.md) §1).

### Variants

```bash
uv run hd-timescale --no-make-plot            # parquets only
uv run hd-timescale --cell-set PoS            # the diagnostic cell sets
uv run hd-timescale --cell-set ADn+PoS
uv run hd-timescale --max-lag 100             # out to 10 s instead of 5 s
uv run hd-timescale --windows-ms 200 1000 3000
```

---

## 2. Parquet schema

One row per session. Columns beyond the identifying ones:

| Column | Meaning |
|---|---|
| `D_<window>_<method>` | D in rad²/s, origin-forced fit over that window, from that estimator. `window` ∈ {200, 500, 5000} ms, `method` ∈ {wrapped, unwrapped, circular} |
| `r2_<window>_<method>` | r² of that fit, against the zero-intercept model |
| `curve_<method>` | the full 50-point MSD curve, 100 ms … 5 s |
| `alpha_short_<method>` | log-log slope of the curve over 0.1–0.5 s (1 = ordinary diffusion) |
| `alpha_long_<method>` | log-log slope over 1–5 s |
| `unwrap_risk` | fraction of per-bin steps with \|step\| > π/2 — ambiguous unwrap directions |
| `wrapped_saturation` | wrapped ⟨Δα²⟩ at 5 s, as a fraction of the π²/3 ceiling |
| `unwrapped_over_circular` | ratio of the two circumventions at 5 s; the trust flag |
| `resultant_at_max_lag` | ⟨cos Δα⟩ at 5 s — surviving angular correlation |
| `decorrelation_s` | first lag where ⟨cos Δα⟩ < 1/e, or `inf` |
| `kurtosis_200`, `kurtosis_max_lag` | displacement kurtosis at 200 ms and 5 s (Gaussian = 3) |
| `pairs_at_max_lag` | within-bout pairs available at the longest lag |
| `wrapped_ceiling` | π²/3, stored so the parquet is self-describing |

---

## 3. Regenerating each table in the results doc

### §5 — the 32-session table

```bash
uv run python - <<'EOF'
import glob, pandas as pd
df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("outputs/results/timescale_Mouse*_ADn.parquet"))])
pub = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("outputs/results/diffusion_Mouse*.parquet"))])
pub = pub[pub.cell_set == "ADn"].set_index("session_id")
df["published_D"] = df.session_id.map(pub.D_bout_aware)
df["ratio_5s_200ms"] = df.D_5000_circular / df.D_200_circular
cols = ["session_id", "n_cells", "published_D", "D_200_circular", "D_500_circular",
        "D_5000_circular", "ratio_5s_200ms", "alpha_short_circular", "alpha_long_circular",
        "resultant_at_max_lag", "unwrapped_over_circular"]
print(df.sort_values(["mouse", "session"])[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
EOF
```

### §5.1–5.2 — the window ratios and the speed correlation

```bash
uv run python - <<'EOF'
import glob, pandas as pd
from scipy.stats import spearmanr
df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("outputs/results/timescale_Mouse*_ADn.parquet"))])
for w in (500, 5000):
    r = df[f"D_{w}_circular"] / df.D_200_circular
    print(f"D_{w}/D_200: median {r.median():.2f}  IQR {r.quantile(.25):.2f}-{r.quantile(.75):.2f}"
          f"  range {r.min():.2f}-{r.max():.2f}")
r5 = df.D_5000_circular / df.D_200_circular
print("slow (D200<1), n=%d: median %.2f" % ((df.D_200_circular < 1).sum(), r5[df.D_200_circular < 1].median()))
print("fast (D200>=2), n=%d: median %.2f" % ((df.D_200_circular >= 2).sum(), r5[df.D_200_circular >= 2].median()))
print("spearman(D_200, ratio) = %+.3f (p=%.1e)" % spearmanr(df.D_200_circular, r5))
EOF
```

### §4 — the anomalous exponents

```bash
uv run python - <<'EOF'
import glob, pandas as pd
df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("outputs/results/timescale_Mouse*_ADn.parquet"))])
for m in ("wrapped", "circular"):
    print(f"{m:9s} alpha 0.1-0.5s median {df[f'alpha_short_{m}'].median():.2f}"
          f"   alpha 1-5s median {df[f'alpha_long_{m}'].median():.2f}")
print("kurtosis: 200ms median %.1f   5s median %.2f" % (df.kurtosis_200.median(), df.kurtosis_max_lag.median()))
print("saturation: median %.2f range %.2f-%.2f" % (df.wrapped_saturation.median(), df.wrapped_saturation.min(), df.wrapped_saturation.max()))
print("unwrap_risk range %.3f-%.3f" % (df.unwrap_risk.min(), df.unwrap_risk.max()))
print("R at 5s range %.3f-%.3f" % (df.resultant_at_max_lag.min(), df.resultant_at_max_lag.max()))
print("u/c < 2 in %d/%d sessions" % ((df.unwrapped_over_circular < 2).sum(), len(df)))
EOF
```

---

## 4. Verifying the method itself

These are the checks the results doc rests on. Each is a few seconds.

### 4.1 Synthetic validation of the estimators (§2.2 table)

A wrapped random walk with a known rate. `wrapped` must under-read at 5 s; both circumventions must
recover the truth.

```bash
uv run python - <<'EOF'
import numpy as np
rng = np.random.default_rng(0); dt, D, N = 0.1, 1.0, 200_000
true = np.cumsum(rng.normal(0, np.sqrt(D * dt), N)); wrapped = true % (2 * np.pi)
sad = lambda a, b: (a - b + np.pi) % (2 * np.pi) - np.pi
uw = np.cumsum(np.concatenate([[0], sad(wrapped[1:], wrapped[:-1])]))
print(f"{'tau(s)':>6} {'truth':>10} {'wrapped':>10} {'unwrapped':>10} {'circular':>10}")
for L in (1, 2, 5, 10, 20, 50):
    d = sad(wrapped[L:], wrapped[:-L])
    print(f"{L*dt:6.1f} {np.mean((true[L:]-true[:-L])**2):10.3f} {np.mean(d**2):10.3f} "
          f"{np.mean((uw[L:]-uw[:-L])**2):10.3f} {-2*np.log(np.mean(np.cos(d))):10.3f}")
print(f"wrapped ceiling = pi^2/3 = {np.pi**2/3:.3f}")
EOF
```

Expected: at τ = 5 s, truth ≈ 5.05, wrapped ≈ 2.94 (saturated), unwrapped ≈ 5.05, circular ≈ 4.93.

### 4.2 The falsification test that picks `circular` (§3)

If the unwrapped MSD were right, ⟨cos Δα⟩ would have to equal exp(−MSD/2). Compare with the
measured value.

```bash
uv run python - <<'EOF'
import glob, numpy as np, pandas as pd
df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("outputs/results/timescale_Mouse*_ADn.parquet"))])
df["R_implied"] = np.exp(-df.D_5000_unwrapped * 5.0 / 2.0)
df["ratio"] = df.resultant_at_max_lag / df.R_implied
print(df.sort_values("unwrap_risk")[["session_id", "unwrap_risk", "resultant_at_max_lag", "R_implied", "ratio"]]
        .to_string(index=False, float_format=lambda v: f"{v:.4g}"))
EOF
```

Expected: ratio ≈ 1–4 for low-risk sessions; ~10⁴ for Mouse20-130514/130515, where `unwrapped` is
refuted.

### 4.3 The circular estimator is exact on a Gaussian, biased on this data (§3.1)

```bash
uv run python - <<'EOF'
import numpy as np
from scipy.stats import kurtosis
rng = np.random.default_rng(0)
for s in (0.2, 0.5, 1.0):
    g = rng.normal(0, s, 400_000); gw = (g + np.pi) % (2 * np.pi) - np.pi
    w, c = np.mean(gw**2), -2 * np.log(np.mean(np.cos(gw)))
    print(f"sigma={s}  kurtosis={kurtosis(g, fisher=False):.2f}  wrapped={w:.4f}  circular={c:.4f}  ratio={c/w:.3f}")
EOF
```

Expected: kurtosis 3.00 and ratio 1.000 in every row — the machinery is unbiased; the ~0.75 ratio
seen on real 200 ms data is the data's leptokurtosis, not the estimator.

### 4.4 The new code reproduces the published estimator where they overlap

```bash
uv run python - <<'EOF'
import glob, numpy as np, pandas as pd
from hdunique import diffusion as dif, timescale
from hdunique.sweep import iter_cache
pub = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("outputs/results/diffusion_Mouse*.parquet"))])
pub = pub[pub.cell_set == "ADn"].set_index("session_id")
lags = tuple(range(1, 51)); worst = 0.0
for e in iter_cache(cell_set="ADn"):
    curve = np.mean([timescale.msd_curve(angles=t, bout_lengths=e.bout_lengths, lags=lags, method="wrapped")
                     for t in e.decoded], axis=0)
    for w, col in ((200, "D_bout_aware"), (500, "D_bout_aware_500")):
        d, _ = dif.window_slope(curve=curve, dt=0.1, window_ms=w)
        worst = max(worst, abs(d - pub.loc[e.meta["session_id"], col]))
print(f"max |difference| vs published D_bout_aware columns: {worst:.3e}")
EOF
```

Expected: `0.000e+00`. The `wrapped` curve is the published estimator on a longer lag range.

### 4.5 Bouts are long enough for a 5 s lag

```bash
uv run python - <<'EOF'
import numpy as np
from hdunique.sweep import iter_cache
short = 0; total = 0
for e in iter_cache(cell_set="ADn"):
    bl = e.bout_lengths; total += len(bl); short += sum(1 for b in bl if b <= 50)
    print(f"{e.meta['session_id']:16s} bouts={len(bl):3d} min={min(bl):5d} median={int(np.median(bl)):5d} max={max(bl):5d}")
print(f"\nbouts shorter than the 5 s lag (50 bins): {short}/{total}")
EOF
```

Expected: `2/537` — 5 s is comfortably supported.

---

## 5. Determinism

`hd-timescale` performs no fitting and draws no random numbers: it is a deterministic function of
the cached decoded angles. Re-running it reproduces its parquets exactly. The only stochastic step
in the whole chain is the ring fit, which happened upstream in `hd-diffusion` and is seeded there
(see [`REPRODUCING.md`](./REPRODUCING.md) §5).
