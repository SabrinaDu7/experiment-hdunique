# experiment-hdunique

## Intro to this repo
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
  - **Angular speed**: How quickly the brain's estimate of heading is changing.
- Cell-level:
  - **Tuning to angular speed**: bin the animal's behavioral angular velocity and plot mean firing rate against it.
  - **Neuron intrinsic timescale**: Denoted τ and also called the spike-train autocorrelation decay constant. Measures how long a neuron's activity stays correlated with itself over time.

---

## Start here

Work in this repo is organised as **questions**. Each has an id (`diffusion1`, `variance1`, …) and
four assets sharing that id: an instructions doc, a results template, a generated results doc, and
an experiment module.

| If you want to… | Read |
|---|---|
| **Add or change an experiment** | [`CLAUDE.md`](./CLAUDE.md) — the workflow, where files go, the rules |
| See what questions exist | `uv run hd-exp list` |
| Read a question's methods | `docs/exp_instructions/instructions-<qid>.md` |
| Read a question's results | `docs/exp_results/results_<qid>.md` |
| The original diffusion-constant methods (the paper's spec) | [`docs/methods.md`](./docs/methods.md) |
| How this code got here, and what was wrong before | [`docs/porting/2026-08-02-port-rem-diffusion-and-variance.md`](./docs/porting/2026-08-02-port-rem-diffusion-and-variance.md) |

### Questions so far

| id | Question | Results |
|---|---|---|
| `diffusion1` | How does *D* depend on the measurement window? | [results](./docs/exp_results/results_diffusion1.md) |
| `diffusion2` | Is the long-lag sub-diffusion real, or made by the decoder/estimator? | [results](./docs/exp_results/results_diffusion2.md) |
| `variance1` | Is between-mouse variance in *D* larger than within-mouse? | [results](./docs/exp_results/results_variance1.md) |
| `variance2` | How does variance partition across bouts, sessions and mice? | [results](./docs/exp_results/results_variance2.md) |
| `bouts1` | Does REM bout context predict *D*? | [results](./docs/exp_results/results_bouts1.md) |

`docs/porting/`, `docs/long_D/` and `docs/bout_level/` are historical narrative records from before
this structure existed. They are kept for provenance; new work does not go there.

## Quick start

```bash
uv sync
uv run --with dandi dandi download DANDI:000056     # the only input data
cp .envrc.example .envrc && $EDITOR .envrc          # point at the download; set OUTPUT_PATH
source .envrc                                        # or: direnv allow

uv run hd-diffusion --scope all                      # NWB -> outputs/cache (hours; once)
uv run hd-exp run variance1                          # any question (seconds to minutes)
```

## Repository structure

```
docs/
  exp_instructions/instructions-<qid>.md    the question and its methods      (hand-written)
  exp_results/results_<qid>.in              prose + @TOKEN@                   (hand-written)
  exp_results/results_<qid>.md              the rendered document             (GENERATED)
  methods.md                                the paper's spec and our departures from it
  porting/ long_D/ bout_level/              historical records
src/
  config.py env.py                          constants; where data and outputs live
  loader.py rates.py manifold.py            DANDI -> rates -> Isomap -> ring
  diffusion.py timescale.py bouts.py        the estimators
  variance.py sweep.py                      the mixed model; cache and parquet plumbing
  analysis/   io, stats, curves, values, render     reusable primitives
  figures/    base, strips, curves                  shared figure grammars
  collect/    timescale, bouts                      sweeps building per-session tables
  experiments/<qid>.py + registry                   one module per question
  cli/        exp.py (hd-exp), diffusion.py (hd-diffusion)
  spud/       vendored from Chaudhuri et al. (2019), unchanged
scripts/      provenance tools: migrate_cache, verify_cache, synthetic_ring_control
tests/
outputs/
  cache/      decode cache (tracked, ~22 MB) — every question reads it, nothing recomputes it
  results/    per-session tables and <qid>_values.json
  figures/    <qid>_<expid>_<description>.png
```

Two console scripts only: **`hd-diffusion`** (the one expensive collector) and **`hd-exp`**
(everything else).

### How a result stays trustworthy

`analyse()` writes named values plus provenance — resolved config, git commit, timestamp — to
`<qid>_values.json`. `render` substitutes them into the `.in`. **A token with no value is a hard
error**, so a results file cannot render with a gap, and re-running an analysis cannot overwrite a
word of interpretation. `hd-exp check <qid>` recomputes and fails on any drift.

## Scope

- **ADn cells only.** The ring is carried by ADn; postsubiculum alone does not form a usable ring
  and adding it to ADn mostly adds variance. PoS runs remain reachable as a diagnostic.
- **DANDI REM scoring.** The pipeline reads one public dataset end to end. This differs from the
  scoring used for the original paper, so absolute *D* values here are **not** directly comparable
  to the paper's reported numbers — see [`docs/methods.md`](./docs/methods.md) §5.
- **No wake decoding.** The wake RMSE decode-quality gate lived in the predecessor repo and needs
  CRCNS head-angle files; it is not ported.
