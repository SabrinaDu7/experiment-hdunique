# Reproducing the per-bout diffusion results

Every number in [`2026-08-03-bout-level-diffusion.md`](./2026-08-03-bout-level-diffusion.md) comes
from the commands below. Setup is identical to the main pipeline — follow
[`../porting/REPRODUCING.md`](../porting/REPRODUCING.md) §0 first.

```bash
cd experiment-hdunique
export OUTPUT_PATH="$PWD/outputs"
```

## 1. Per-bout diffusion constants

```bash
uv run hd-bouts                                          # every cached ADn session
uv run hd-bouts --sessions 25-140130 28-140313 12-120806 # the three focus sessions
```

Writes `bouts_Mouse<m>_ADn.parquet`, one row per REM bout: `bout_index`, `start_s`, `duration_s`,
`prev_state`, `next_state`, `time_frac`, `n_bins`, `D_200`, `D_500`, `D_std`, `nugget`, plus
`truncated` / `short` flags.

Minutes, not hours: decoded angles come from the cache, and only the state epochs are read from the
NWB files.

> **The cache was built from unmerged REM epochs.** `--merge-gap-s` above ~11 s changes the epochs
> and invalidates it; `hd-bouts` raises rather than silently mismatching. To use a real merge rule,
> re-run `hd-diffusion` with the same rule first.

## 2. The merge rule is a no-op (§1)

```bash
uv run python - <<'EOF'
import numpy as np, pandas as pd
from hdunique import loader
gaps = []
for m, s in loader.list_sessions():
    try: d = loader.load_session(mouse=m, session=s)
    except Exception: continue
    if "states" not in d.keys(): continue
    rem = d["states"].as_dataframe().query("label == 'REM'").sort_values("start")
    if len(rem) < 2: continue
    gaps += list(rem.start.to_numpy()[1:] - rem.end.to_numpy()[:-1])
g = pd.Series(gaps)
print(f"n gaps {len(g)}, min {g.min():.0f}s, median {g.median():.0f}s")
for t in (10, 20, 30, 60):
    print(f"  <= {t:3d}s: {(g <= t).sum()}")
EOF
```

Expected: `n gaps 643, min 11s, median 597s`, and `<= 10s: 0`.

## 3. Bout context counts (§2)

```bash
uv run python - <<'EOF'
import pandas as pd
from hdunique import loader
rows = []
for m, s in loader.list_sessions():
    try: d = loader.load_session(mouse=m, session=s)
    except Exception: continue
    if "states" not in d.keys(): continue
    st = d["states"].as_dataframe().sort_values("start").reset_index(drop=True)
    for i, r in st.iterrows():
        if r.label != "REM": continue
        rows.append({"prev": st.label.iloc[i-1] if i else "start",
                     "next": st.label.iloc[i+1] if i < len(st)-1 else "end"})
b = pd.DataFrame(rows)
print(len(b), "REM bouts"); print("prev:", dict(b.prev.value_counts())); print("next:", dict(b.next.value_counts()))
EOF
```

Expected: 682 bouts; preceded by Non-REM 675 / Awake 7; followed by Awake 590 / Non-REM 92.

## 4. Three-level variance decomposition (§3)

Searle's unbalanced nested ANOVA. Used in preference to `statsmodels` crossed variance components,
which fails to converge on this design; the ANOVA estimator is exact and optimiser-free.

```bash
uv run python - <<'EOF'
import glob, numpy as np, pandas as pd
df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("outputs/results/bouts_Mouse*_ADn.parquet"))])
df = df[np.isfinite(df.D_200) & (df.D_200 > 0)].copy(); df["y"] = np.log(df.D_200)
N, A = len(df), df.mouse.nunique()
n_ij = df.groupby(["mouse", "session_id"]).size(); n_i = df.groupby("mouse").size(); J = len(n_ij)
gm = df.y.mean(); mi = df.groupby("mouse").y.mean(); mij = df.groupby(["mouse", "session_id"]).y.mean()
SSA = sum(n_i[i] * (mi[i] - gm) ** 2 for i in n_i.index)
SSB = sum(n_ij[k] * (mij[k] - mi[k[0]]) ** 2 for k in n_ij.index)
SSE = sum(((g.y - g.y.mean()) ** 2).sum() for _, g in df.groupby(["mouse", "session_id"]))
MSA, MSB, MSE = SSA / (A - 1), SSB / (J - A), SSE / (N - J)
s2 = sum(n_ij[k] ** 2 / n_i[k[0]] for k in n_ij.index)
k2 = (N - s2) / (J - A); k1 = (s2 - sum(n_ij ** 2) / N) / (A - 1); k0 = (N - sum(n_i ** 2) / N) / (A - 1)
se = MSE; sb = max((MSB - MSE) / k2, 0); sa = max((MSA - MSE - k1 * sb) / k0, 0); tot = sa + sb + se
print(f"{N} bouts / {J} sessions / {A} mice")
for name, v in (("between-mouse", sa), ("between-session", sb), ("between-bout", se)):
    print(f"  {name:18s} {v:.3f}  ({100*v/tot:4.1f}%)")
print(f"  ICC(mouse) {sa/tot:.3f}   ICC(mouse+session) {(sa+sb)/tot:.3f}")
print(f"  within-session SD of log D: median {df.groupby('session_id').y.std().median():.2f}")
print(f"  within-session max/min D:   median {df.groupby('session_id').D_200.agg(lambda x: x.max()/x.min()).median():.1f}x")
EOF
```

Expected: 537 / 32 / 6; between-mouse 0.325 (34.3 %), between-session 0.289 (30.6 %), between-bout
0.332 (35.1 %); ICC(mouse) 0.343; within-session SD 0.48; max/min 6.4×.

## 5. Context effects (§4)

```bash
uv run python - <<'EOF'
import glob, warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("outputs/results/bouts_Mouse*_ADn.parquet"))])
df = df[np.isfinite(df.D_200) & (df.D_200 > 0)].copy()
df["log_D"] = np.log(df.D_200); df["to_wake"] = (df.next_state == "Awake").astype(int)
for label, m in (("raw", smf.ols("log_D ~ to_wake", df).fit()),
                 ("session RE", smf.mixedlm("log_D ~ to_wake", df, groups=df.session_id).fit()),
                 ("session FE", smf.ols("log_D ~ to_wake + C(session_id)", df).fit())):
    print(f"  exit-state, {label:11s}: beta {m.params['to_wake']:+.3f}  p {m.pvalues['to_wake']:.2g}")
d = df.copy()
for c in ("log_D", "time_frac", "duration_s"):
    d[c + "_c"] = d[c] - d.groupby("session_id")[c].transform("mean")
print(f"  position raw {spearmanr(df.time_frac, df.log_D)[0]:+.3f} / demeaned {spearmanr(d.time_frac_c, d.log_D_c)[0]:+.3f}")
print(f"  duration raw {spearmanr(df.duration_s, df.log_D)[0]:+.3f} / demeaned {spearmanr(d.duration_s_c, d.log_D_c)[0]:+.3f}")
EOF
```

Expected: exit-state β −0.499 (p 7e-06) raw → −0.053 (p 0.5) with session RE → −0.032 (p 0.68) with
session FE. Position and duration correlations all |ρ| < 0.06, p > 0.19.

## 6. Determinism

`hd-bouts` does no fitting and draws no random numbers — it is a deterministic function of the
cached decoded angles plus the NWB state tables. Re-running reproduces its parquets exactly.
