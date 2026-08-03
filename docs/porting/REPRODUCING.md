# Reproducing every result in this repo

Every number and figure in [`rem-diffusion.md`](./results-rem-diffusion.md) and
[`variance-decomposition.md`](./results-variance-decomposition.md) is produced by the commands below. No
result is quoted anywhere in this repo without a command here that regenerates it.

---

## 0. Setup

### 0.1 Code

```bash
git clone <this repo> && cd experiment-hdunique
uv sync
```

### 0.2 Data — DANDI dandiset 000056

The only input is [DANDI 000056](https://dandiarchive.org/dandiset/000056) (Peyrache et al. 2015,
NWB conversion). No CRCNS files are needed.

```bash
uv run --with dandi dandi download DANDI:000056
```

This produces `000056/sub-Mouse12/…`, `000056/sub-Mouse17/…`, and so on — 40 NWB files.

> **⚠️ The NWB files were relabelled upstream on 2026-07-16**, adding `location` (ADn / PoS)
> annotations to many previously unlabelled sessions. Cell counts here assume the relabelled
> files. If a session reports 0 cells, you have an older download; re-download it.

### 0.3 Environment

```bash
cp .envrc.example .envrc
$EDITOR .envrc          # point DANDI_DATA_ROOT at the download; set OUTPUT_PATH
direnv allow            # or: source .envrc
```

| Variable | Meaning |
|---|---|
| `DANDI_DATA_ROOT` | The 000056 download directory (contains `sub-Mouse*/`). |
| `OUTPUT_PATH` | Where parquets, the decode cache and figures are written. |

> **⚠️ An exported `OUTPUT_PATH` from another project wins over `.envrc`.** The code reads
> `os.environ` first. If results land somewhere unexpected, check `echo $OUTPUT_PATH`.

### 0.4 Sanity check

```bash
uv run python -c "
from hdunique.env import dandi_root, results_dir
from hdunique import loader
print('data:   ', dandi_root(), dandi_root().exists())
print('results:', results_dir())
print('sessions found:', len(loader.list_sessions()))
"
```

Expect `sessions found: 40`.

---

## 1. The REM diffusion sweep

**Every default is the published setting**, so the sweep needs no flags:

```bash
uv run hd-diffusion --scope all
```

Writes `$OUTPUT_PATH/results/diffusion_Mouse{12,17,20,24,25,28}.parquet` and populates
`$OUTPUT_PATH/results/cache/`.

**Runtime: several hours.** The cost is the ring fits — `n_refits` (5) × `n_restarts` (10) = 50
spline fits per session, each a k-means init plus an iterative optimisation over up to 15 000
points. Budget roughly 8–15 minutes per session on one core.

Subsets, for a quicker check:

```bash
uv run hd-diffusion --scope session --mouse 28 --session 140313 --make-plot
uv run hd-diffusion --scope mouse --mouse 25
```

> These are only quick **if the shipped cache covers them** (it covers all 65 published runs). A
> session that is not cached costs the full 8–15 minutes.

The diagnostic cell sets (used only to justify the ADn-only choice, see
[`rem-diffusion.md`](./results-rem-diffusion.md) §"Why ADn only"):

```bash
uv run hd-diffusion --scope all --cell-areas PoS
uv run hd-diffusion --scope all --cell-areas ADn PoS
```

### Re-deriving metrics without refitting

The cache holds each run's embedding and per-refit decoded angles, so any change to the diffusion
metric, lag window or fit is a seconds-long recompute:

```bash
uv run hd-diffusion --recompute-only
```

The cache is keyed on a **signature covering every parameter the decode depends on** (rates,
embedding, ring fit, restarts, refits, fit fraction, seed, cell areas). Change any of them and the
entry is recomputed rather than silently reused.

---

## 2. Variance decomposition

Requires the parquets from step 1.

```bash
uv run hd-variance                                        # headline: >=15 ADn cells
uv run hd-variance --min-adn-cells 20                     # stricter gate
uv run hd-variance --min-adn-cells 0                      # ungated sensitivity
uv run hd-variance --estimator D_bout_aware               # the bout-aware estimate of D
```

Prints τ², σ², the ICC with bootstrap CIs and the ANOVA cross-check, and writes
`variance_by_mouse_ADn_min<N>_<window>ms_<estimator>.png`.

Runtime: ~40 s per gate (2000 parametric-bootstrap LMM refits each).

> `ConvergenceWarning`s during the bootstrap are expected and are suppressed. Replicates landing on
> the τ² = 0 boundary are exactly what produces the CI's lower edge near zero; the primary fits
> converge cleanly.

### Fit-window robustness

```bash
uv run hd-variance-by-window
uv run hd-variance-by-window --estimator D_bout_aware
```

Writes `variance_by_window_ADn_min15_<estimator>.csv` — one row per fit window
(200/300/400/500 ms).

---

## 3. Figures

```bash
uv run hd-diffusion-grid                    # per-mouse grid of diffusion curves, 200 ms window
uv run hd-diffusion-grid --window-ms 500
```

Reads the cache only; no NWB access, seconds to run.

---

## 4. End-to-end, from nothing

```bash
uv sync
uv run --with dandi dandi download DANDI:000056
cp .envrc.example .envrc && $EDITOR .envrc && source .envrc

uv run hd-diffusion --scope all          # hours
uv run hd-variance                       # minutes
uv run hd-variance-by-window             # minutes
uv run hd-diffusion-grid                 # seconds
```

---

## 5. Determinism

The sweep is deterministic at fixed `seed`. `spud.fit_manifold` seeds nothing of its own — its
k-means initialisation draws from the **global** NumPy RNG — so `sweep.run_session` seeds that
global RNG once per refit (`np.random.seed(cfg.seed + refit)`). Remove that and D swings between
identical runs.

Verified: re-running Mouse28-140313 cold from the NWB files reproduces its cached decoded angles
**bit-identically** (max |Δ| = 0), and Mouse28-140317 to float32 storage precision. See
[`2026-08-02-port-rem-diffusion-and-variance.md`](./2026-08-02-port-rem-diffusion-and-variance.md)
§"Validation".

## 6. Provenance of the shipped cache

The `outputs/results/cache/` entries are **tracked in git and ship with the repo** (~22 MB), so a
fresh clone can run `--recompute-only`, `hd-variance` and the figure commands immediately, without
first spending hours on the sweep.

They were imported from the predecessor repository, where the same pipeline and the same settings
produced them, rather than recomputed from scratch during the port
(`scripts/migrate_cache.py`). They were then **validated** by recomputing whole sessions cold from
the NWBs and diffing (`scripts/verify_cache.py`): Mouse28-140313 ADn reproduced
**bit-identically**. See the port doc's validation section.

If you would rather not trust the import, delete the cache and run step 1 from scratch; the
pipeline does not depend on it.
