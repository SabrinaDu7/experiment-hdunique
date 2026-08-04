# experiment-hdunique

This is a repo aimed at testing the following hypothesis:
- What we see during REM sleep is intrinsic, it’s not plastic; it reflects internal dynamics that are UNIQUE to an animal.
- The only thing that can explain the REM sleep activity variance is animal identity. Nothing else.


To test this hypothesis, we need to answer the following questions:
1. Q1: Do population and cell-level metrics vary more between animals or within animals (ie within sessions)?
2. Q2: Do population and cell-level metrics vary more between animals when they are in REM versus in wake (to isolate REM from just animal activity in general). *[04-08-2026: Currently no experiments]*

To answer Q1, we are analyzing REM activity across metrics, animals, and datasets.
| Datasets | Animals | Metrics |
| -------- | -------- | -------- |
| Peyrache et al. 2015  | 7  | REM diffusion constant  |
| xx  | xx  | xx  |

## Metrics
- Population-level:
  - **REM Diffusion Constant**: How fast does the head-direction system's internal compass drift during REM sleep, and how much of that drift is a property of the animal rather than the recording session?
- Cell-level:

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
[`docs/variance-decomposition.md`](./docs/porting/results-variance-decomposition.md).

---

## Start here

| If you want to… | Read |
|---|---|
| Know what the paper specifies, and where we depart from it | [`docs/methods.md`](./docs/methods.md) |
| Run everything yourself | [`docs/REPRODUCING.md`](./docs/porting/REPRODUCING.md) |
| See the diffusion constants per session | [`docs/rem-diffusion.md`](./docs/porting/results-rem-diffusion.md) |
| See the between/within-mouse decomposition | [`docs/variance-decomposition.md`](./docs/porting/results-variance-decomposition.md) |
| See how *D* depends on the measurement window (200 ms vs 500 ms vs 5 s) | [`docs/2026-08-03-long-timescale-diffusion.md`](./docs/long_D/2026-08-03-long-timescale-diffusion.md) |
| Reproduce that timescale analysis | [`docs/REPRODUCING-timescale.md`](./docs/long_D/REPRODUCING-timescale.md) |
| Understand how this code got here, and what was wrong before | [`docs/2026-08-02-port-rem-diffusion-and-variance.md`](./docs/porting/2026-08-02-port-rem-diffusion-and-variance.md) |

## Quick start

```bash
uv sync
uv run --with dandi dandi download DANDI:000056     # the only input data
cp .envrc.example .envrc && $EDITOR .envrc          # point at the download; set OUTPUT_PATH
source .envrc                                        # or: direnv allow

uv run hd-diffusion --scope all                      # the sweep (hours)
uv run hd-variance                                   # the decomposition (minutes)
uv run hd-timescale                                  # D at 200 ms / 500 ms / 5 s (~2 min)
```

Full instructions, including quicker subsets and the diagnostic runs, are in
[`docs/REPRODUCING.md`](./docs/porting/REPRODUCING.md).

## Repository structure

**One scientific question is the unit of work.** Each has an id (`diffusion1`), an instructions doc
stating the question and its methods, a results doc generated from a template, and a thin experiment
module. `uv run hd-exp list` enumerates them.

```
docs/
  exp_instructions/instructions-<qid>.md    the question, its motivation, its methods
  exp_results/results_<qid>.in              hand-written template: prose + @TOKEN@
  exp_results/results_<qid>.md              GENERATED — never edit; edit the .in
  methods.md                                the paper's spec and our departures from it
  porting/ long_D/ bout_level/              historical narrative records
src/
  config.py env.py                          constants; where data and outputs live
  loader.py rates.py manifold.py            DANDI -> rates -> Isomap -> ring
  diffusion.py timescale.py bouts.py        the estimators
  variance.py sweep.py                      the mixed model; the cache and parquet plumbing
  analysis/    io, curves, stats, values, render   reusable primitives
  figures/     base, strips, curves               one panel grammar per figure type
  collect/     timescale, bouts                   sweeps that build per-session tables
  experiments/ <qid>.py + registry                one module per question
  cli/         exp.py (hd-exp), diffusion.py (hd-diffusion)
  spud/        vendored code from Chaudhuri et al. (2019), unchanged
scripts/       provenance tools: migrate_cache, verify_cache, synthetic_ring_control
tests/
outputs/
  cache/       decode cache (tracked, ~22 MB) — lets a fresh clone run every analysis
  results/     per-session tables and <qid>_values.json
  figures/     <qid>_<description>.png
```

Only two console scripts survive: **`hd-diffusion`**, the one expensive collector (NWB → cache), and
**`hd-exp`**, which runs everything else. Analyses that used to be separate CLIs are now experiments
sharing the primitives in `analysis/` and `figures/`.

### How a question runs

```bash
uv run hd-exp list                  # what exists
uv run hd-exp collect diffusion1    # only for questions needing new tables
uv run hd-exp run     diffusion1    # analyse + render
uv run hd-exp check   diffusion1    # recompute and diff against committed values
```

`analyse` writes named values plus provenance (resolved config, git commit, timestamp) to
`<qid>_values.json`; `render` substitutes them into the `.in`. **An unresolved token is a hard
error**, so a results file cannot render with a gap, and re-running an analysis cannot overwrite
prose.

> **Note on imports.** Modules are top-level names (`config`, `env`, `loader`), so an inherited
> `PYTHONPATH` from another checkout can shadow them. Nothing here needs `PYTHONPATH`; `env.py`
> raises with an explanation if it detects the shadow.

### Where the results live

| Artefact | Path |
|---|---|
| Per-session diffusion constants | `$OUTPUT_PATH/results/diffusion_Mouse<m>.parquet` |
| Decode cache (embedding + decoded angles) | `$OUTPUT_PATH/results/cache/*.npz` |
| D across measurement windows | `$OUTPUT_PATH/results/timescale_Mouse<m>_ADn.parquet` |
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
