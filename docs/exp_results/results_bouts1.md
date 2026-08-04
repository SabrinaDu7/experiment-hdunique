2026-08-04

<!-- GENERATED from results_bouts1.in — do not edit; edit the .in and re-run `hd-exp render`. -->

# bouts1 — results

> **Question.** REM bouts differ in when they occur, how long they last, and what state follows
> them. Do any of those predict how fast the decoded head direction drifts?

Methods: [`instructions-bouts1.md`](../exp_instructions/instructions-bouts1.md).

**537 bouts.** Entered from Non-REM in 531 cases against
6 otherwise, so **entry offers no usable contrast**. Exit does:

| next_state | count |
|---|---|
| Awake | 456 |
| Non-REM | 81 |

---

## bouts1_exp1 — exit state, position and duration

The exit-state effect, estimated three ways:

| model | beta | p |
|---|---|---|
| raw | -0.499 | 0.000 |
| group_random | -0.053 | 0.498 |
| group_fixed | -0.032 | 0.677 |

Raw β = **-0.499** (p = 7.4e-06) collapses to **-0.032**
(p = 0.68) once session identity is held fixed.

Which mice supply the minority exit type at all — 4 of 6:

| mouse | count |
|---|---|
| 12 | 9 |
| 17 | 42 |
| 20 | 27 |
| 24 | 3 |

Continuous context:

| variable | rho_raw | p_raw | rho_demeaned | p_demeaned |
|---|---|---|---|---|
| time_frac | -0.005 | 0.914 | 0.030 | 0.489 |
| duration_s | 0.055 | 0.203 | -0.056 | 0.193 |

### Interpretation

**The apparent exit-state effect is a between-session confound.** It is large and highly significant
raw, and gone once session is controlled. The reason is visible in the composition: the minority
exit type is supplied by only some mice, concentrated in the fast ones, so "compare the two exit
groups" is in practice "compare those mice to the rest" — the between-mouse difference wearing a
context costume.

This is worth dwelling on because it is exactly the failure mode the experiment was built to catch:
a plausible biological story with p ≈ 10⁻⁶ that survives no control at all.

Position in session and bout duration show nothing either way (all |ρ| ≤ 0.055).
The duration null also retires a specific artefact worry raised elsewhere — that long bouts
dominating long lags could manufacture the falling *D*(τ) seen in `diffusion1`. No such correlation
exists.

## bouts1_exp2 — the within-session contrast

Only **13 of 32 sessions** contain enough bouts of both exit
types to carry within-session information. Across those:

- median within-session difference in log *D*: **0.121**
- higher in 9 of 13 sessions
- **Wilcoxon signed-rank p = 0.45**

Three sessions spanning the cell-count range:

| session | cells | n_Awake | median_Awake | n_other | median_other | ratio | p |
|---|---|---|---|---|---|---|---|
| Mouse12-120808 | 39 | 10 | 1.26 | 4 | 1.31 | 1.04 | 0.84 |
| Mouse17-130129 | 25 | 12 | 1.40 | 7 | 1.65 | 1.18 | 0.77 |
| Mouse20-130515 | 6 | 10 | 4.88 | 8 | 4.88 | 1.00 | 0.83 |

### Interpretation

The paired test agrees with the fixed-effects model: no effect. In the three named sessions the two
distributions sit on top of each other, and the largest ratio between arms is only
0.16 in log units.

What the per-session view *does* show is more informative than the null: the **within-group spread
dwarfs any between-group difference**, and the three sessions occupy quite different *D* ranges that
track cell count rather than anything about sleep architecture.

---

## Answer to the question

**No.** Neither exit state, position in the session, nor bout duration predicts a bout's diffusion
constant once session identity is accounted for. The one effect that looked real is a pooling
artefact.

So the large bout-level variance found in `variance2` is **not** explained by the sleep-architecture
context of the bout. Whatever drives it is either within-session physiology on a shorter timescale
or decode quality varying bout to bout.

## How to reproduce

```bash
uv run hd-exp collect bouts1
uv run hd-exp run     bouts1
uv run hd-exp check   bouts1
```

Variants:

```bash
uv run hd-exp run bouts1 --min-per-arm 5
uv run hd-exp run bouts1 --plot-sessions 25-140130 28-140313
```

## Next steps

- Within-session physiological covariates on a shorter timescale (theta power, microarousals) are
  the obvious next candidates, none of which is in the current per-bout table.
- The minority exit arm is supplied by a subset of mice. A design with balanced exit types per
  mouse would test context cleanly rather than by statistical control.

## Provenance

Generated 2026-08-04T21:29:23+00:00 from commit `f0e6970`.

| config | value |
|---|---|
| `cell_set` | `ADn` |
| `context_columns` | `['time_frac', 'duration_s']` |
| `d_column` | `D_200` |
| `exit_state` | `Awake` |
| `min_per_arm` | `3` |
| `plot_sessions` | `['12-120808', '17-130129', '20-130515']` |

| input | value |
|---|---|
| `input_bouts` | `537` |
