2026-08-04

# bouts1: Does the sleep-architecture context a REM bout sits in predict its diffusion constant?

## Question

REM bouts differ in when they occur, how long they last, and what state follows them. Do any of
those predict how fast the decoded head direction drifts?

## Motivation

`variance2` shows bout-level variance is the largest single component. If bout context explains it,
then the session-level *D* is a weighted average whose value depends on that session's sleep
architecture — a recording fact, not a circuit fact. That would matter for every between-mouse
comparison, because sleep architecture is partly a property of the animal.

This is also a test of the analysis itself. A plausible biological story with a small p-value that
turns out to be a pooling artefact is exactly the failure mode worth guarding against.

## Experiments

- **bouts1_exp1** — exit state, position in session and bout duration, each tested raw and with
  session identity controlled.
- **bouts1_exp2** — the exit-state contrast within individual sessions, paired across sessions and
  shown for three named ones.

## Methods

```bash
uv run hd-exp collect bouts1
uv run hd-exp run     bouts1
```

Code: `src/experiments/bouts1.py`; primitives from `src/analysis/stats.py`
(`effect_within_and_between`, `demean_within`, `correlation`, `paired_difference`, `rank_sum`).

**Input.** The per-bout table, as for `variance2`.

**Context available.** REM is essentially always *entered* from Non-REM in this dataset, so entry
offers no contrast. The exit does: a bout either ends in an awakening or returns to Non-REM.

**exp1 — three estimates of the same effect.** `effect_within_and_between` fits the exit-state
effect ignoring session, with session as a random effect, and with session as fixed effects. A
predictor that looks strong raw and vanishes under the other two is a **between-session confound**,
not an effect of the predictor. Reporting all three makes that visible rather than leaving it to be
discovered later.

The continuous variables (position in session, bout duration) are tested both raw and
session-demeaned, for the same reason.

**exp2 — the paired test.** Only sessions with at least `min_per_arm` bouts of *each* exit type
carry within-session information; the contrast is taken within each such session and compared across
them by Wilcoxon signed-rank. Three sessions spanning the cell-count range are then shown
individually, because a per-session view shows the within-group spread that a difference-in-means
hides.

**Defaults.** `cell_set=ADn`, `d_column=D_200`, `exit_state=Awake`,
`context_columns=(time_frac, duration_s)`, `min_per_arm=3`.
