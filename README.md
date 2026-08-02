# experiment-hdunique

**How fast does the head-direction system's internal compass drift during REM sleep, and how much
of that drift is a property of the animal rather than the recording session?**

This repo decodes head direction from anterodorsal thalamic (ADn) population activity during REM
sleep without using any measured head angle, measures the diffusion constant *D* of the decoded
angle, and decomposes the variability in *D* into between-mouse and within-mouse components.

The decoding method is SPUD (spline parameterization for unsupervised decoding) from
**Chaudhuri et al. (2019),** [*The intrinsic population dynamics of a canonical cognitive
circuit*](https://www.nature.com/articles/s41593-019-0460-x), Nature Neuroscience. The data is
[DANDI dandiset 000056](https://dandiarchive.org/dandiset/000056) (Peyrache et al. 2015), read via
[pynapple](https://pynapple.org/).

**Headline finding: the variance decomposition is a null result.** The ICC lands around 0.5, but
with only ~6 mice its confidence interval spans nearly the whole range and includes zero. The
honest statement is *"between-mouse and within-mouse variability are not separable at this sample
size"*, not *"mice differ from each other"*. Details in
[`docs/variance-decomposition.md`](./docs/variance-decomposition.md).

---

## Start here

| If you want to… | Read |
|---|---|
| Know what the paper specifies, and where we depart from it | [`docs/methods.md`](./docs/methods.md) |
| Run everything yourself | [`docs/REPRODUCING.md`](./docs/REPRODUCING.md) |
| See the diffusion constants per session | [`docs/rem-diffusion.md`](./docs/rem-diffusion.md) |
| See the between/within-mouse decomposition | [`docs/variance-decomposition.md`](./docs/variance-decomposition.md) |
| Understand how this code got here, and what was wrong before | [`docs/2026-08-02-port-rem-diffusion-and-variance.md`](./docs/2026-08-02-port-rem-diffusion-and-variance.md) |

## Quick start

```bash
uv sync
uv run --with dandi dandi download DANDI:000056     # the only input data
cp .envrc.example .envrc && $EDITOR .envrc          # point at the download; set OUTPUT_PATH
source .envrc                                        # or: direnv allow

uv run hd-diffusion --scope all                      # the sweep (hours)
uv run hd-variance                                   # the decomposition (minutes)
```

Full instructions, including quicker subsets and the diagnostic runs, are in
[`docs/REPRODUCING.md`](./docs/REPRODUCING.md).

## Repository structure

```
src/
  spud/          Vendored analysis code from Chaudhuri et al. (2019), kept as the authors wrote it
                 so it can be diffed against upstream. Exempt from this repo's style rules.
                 angle_fns, dim_red_fns, fit_helper_fns, manifold_fit_and_decode_fns, kernel_rates

  hdunique/      The pipeline. One flat module per stage, and nothing does I/O at import.
    env.py         Where data comes from and results go (the DANDI_DATA_ROOT / OUTPUT_PATH contract)
    config.py      DiffusionConfig, VarianceGateConfig / VarianceConfig and the shared constants.
                   Every default is the setting that produced the published results.
    loader.py      DANDI/pynapple: sessions, spikes, REM epochs, cell selection by brain area
    rates.py       Sum-of-Gaussians firing rates, computed per REM bout
    manifold.py    Isomap embedding, the 12-knot ring fit, and the per-refit decode
    diffusion.py   The diffusion curve, the origin-forced fit and its bootstrap — the estimator
    sweep.py       One session -> one result row; the CacheEntry record and the parquet tables
    variance.py    The random-intercept LMM, its bootstrap, and the ANOVA cross-check
    plotting.py    Every figure, plus the DiffusionPanel record both diffusion figures draw
    cli/           Entry points. Each is argument parsing and reporting only; the science is above.
      diffusion.py          -> hd-diffusion
      diffusion_grid.py     -> hd-diffusion-grid
      variance.py           -> hd-variance
      variance_by_window.py -> hd-variance-by-window

docs/            Methods, results, reproduction instructions, and the port record
scripts/         Provenance tools that are not part of the pipeline: migrate_cache.py (how the
                 shipped decode cache got here) and verify_cache.py (the cold recompute that
                 validates it). Tracked, because the port record cites them as evidence.
outputs/results/ Parquets and the decode cache (tracked), figures (not tracked)
```

`hdunique/` is ten modules, so it is deliberately **flat**: an `io/` or `analysis/` subpackage
would add import depth without removing a single decision about where code goes. `cli/` is the one
subpackage that earns its keep, because it is the only group with a shared contract (each module
exposes `run(cfg=...)` plus a `main()` console script).

The dependency order is strictly one-way: `cli/ → sweep, variance → diffusion, manifold, rates →
loader, config, env → spud`. Nothing in `hdunique/` is imported by `spud/`, and `scripts/` imports
`hdunique/` but nothing imports `scripts/`.

### Where the results live

| Artefact | Path |
|---|---|
| Per-session diffusion constants | `$OUTPUT_PATH/results/diffusion_Mouse<m>.parquet` |
| Decode cache (embedding + decoded angles) | `$OUTPUT_PATH/results/cache/*.npz` |
| Figures | `$OUTPUT_PATH/results/*.png` |

The parquets are the primary artefact. The cache exists so that any new diffusion metric is a
seconds-long recompute instead of hours of refitting.

## Scope

- **ADn cells only.** The ring is carried by ADn; postsubiculum alone does not form a usable ring
  and adding it to ADn mostly adds variance. PoS runs remain reachable as a diagnostic.
- **DANDI REM scoring.** The pipeline reads one public dataset end to end. This differs from the
  scoring used for the original paper, so absolute *D* values here are **not** directly comparable
  to the paper's reported numbers — see [`docs/methods.md`](./docs/methods.md) §5.
- **No wake decoding.** The wake RMSE decode-quality gate lived in the predecessor repo and needs
  CRCNS head-angle files; it is not ported.
