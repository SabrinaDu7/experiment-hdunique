2026-08-05

# angular1: How fast does the head-direction signal turn, in wake and in REM, across animals?

## Question

What is the mean angular velocity of the head-direction representation in each animal and session —
during wake, where the head is tracked, and during REM, where it is not?

## Motivation

Everything this repo has measured so far is a *diffusion* constant: how the decoded angle spreads
over a lag. That conflates two things — how fast the bump moves, and how little it is restrained.
Angular velocity is the first of those on its own, and it is the quantity a between-animal
comparison most obviously ought to control for: an animal whose bump simply moves faster will show a
larger *D* without any difference in the dynamics we care about.

REM is the interesting case and the hard one. There is no head to track, so the velocity has to come
from the spikes alone. Peyrache et al. (2015) solved this, and this question implements their
solution.

## Experiments

- **angular1_exp1** — mean angular speed per session in wake and in REM, and how it varies across
  animals.
- **angular1_exp2** — the validation. In wake the answer is independently known from the tracking
  LEDs, so the estimator can be checked against it.
- **angular1_exp3** — one decoder per wake bout, to separate a broken decoder from a broken
  estimator. This is the experiment that localises exp2's failure.

## Methods

The estimator is from **Peyrache, Lacroix, Petersen & Buzsáki (2015)**, [*Internally organized
mechanisms of the head direction sense*](https://www.nature.com/articles/nn.3968), Nature
Neuroscience, "Estimation of angular velocity". The data is
[DANDI dandiset 000056](https://dandiarchive.org/dandiset/000056), the same recordings.

```bash
uv run hd-exp collect angular1     # ~4 min per session, ~2 h for the full set
uv run hd-exp run     angular1
```

Code: `src/experiments/angular1.py`; the estimator is `src/metrics/angular_speed.py`; the measured
ground truth is `src/decode/head_direction.py`; the synthetic verification is
`tests/test_angular_speed.py`.

### The idea

Two head-direction cells with different preferred directions fire in sequence when the head sweeps
past both. *How long* the second lags the first depends on *how fast* the head turned. So the
temporal cross-correlogram of a cell pair carries the angular velocity, even with no tracking.

Made precise: let ρ(θ) be the correlation between the two cells' **tuning curves** offset by an
angle θ. If the head turns at speed *v*, a time lag τ corresponds to an angular offset *v*·τ, so the
temporal cross-correlation at lag τ should equal ρ(*v*·τ). The head does not turn at one fixed
speed, so this is averaged over a distribution of speeds:

> C(τ) ≈ ⟨ρ(*v*·τ)⟩ᵥ

with *v* drawn from a χ²(3) prior — a one-parameter family whose only free parameter is the mean
speed. Fitting that one parameter to the observed correlogram is the estimate.

### Per session

1. **Tuning curves** for every ADn cell, from wake, where the head angle is measured from the LEDs
   (60 bins, occupancy-normalised). Cells peaking below 1 Hz are dropped as untuned.
   The curves come from wake and are then applied unchanged to REM — that is the point: a cell's
   preferred direction is a property of the cell, not of the brain state.
2. **ρ(θ)** for each pair, by rotating one curve against the other. Pairs never reaching |ρ| = 0.3
   are dropped: their correlogram carries no angular information and the fit would be unconstrained.
3. **Observed correlogram** per pair, as a Pearson correlation between the two binned rate traces
   (20 ms bins) at each lag out to ±1 s, computed by FFT. Bouts are binned separately and
   concatenated so no bin straddles a gap. Rates are high-passed at 5 s first — see below.
4. **Fit one speed across all pairs at once** by matching the predicted correlograms' *shape*, over
   a coarse geometric sweep from 0.05 to 30 rad/s followed by a bounded refinement. A session needs
   3 usable pairs to report an estimate.
5. The **per-pair median**, which was the first implementation, is computed alongside and reported
   for comparison. It is given the joint fit's lag window, so the two differ only in whether the
   pairs are pooled.

### Five implementation choices that are not cosmetic

Each was forced by an observed failure — the first three by the synthetic test, the last two by real
data. They are recorded because none is obvious from the paper's description and all five change the
answer.

**Shape, not amplitude.** Binned spike counts are Poisson, and that noise attenuates a measured
correlation towards zero however fast the head moved. A cost function comparing raw amplitudes
therefore prefers an implausibly high speed, whose prediction is flat and small, purely because the
data are noisy — on synthetic data with a true speed of 3 rad/s it returned 59. The fit allows the
prediction a free amplitude and offset (`shape_cost`, solved in closed form), so only the shape of
the fall-off is compared and the fit stays one-dimensional.

**The lag window has to suit the speed.** The window must be long enough for the head to sweep an
appreciable angle and short enough not to wrap several times. Below about 1 rad of total sweep the
correlogram is too flat to separate speeds and the estimate runs high — at 0.25 rad, a true 0.5
rad/s came back as 1.87. Above about 6 rad it wraps and does the same. Since the right window
depends on the answer, it is set iteratively: fit at a 1 s window, then re-fit with the window that
first estimate implies (`TARGET_SWEEP_RAD` = 2.5 rad). The full correlogram is computed once and
each round slices it, so iterating is free.

**Rotation direction.** ρ(θ) must peak at the angle by which the two tuning curves are genuinely
offset. An off-by-a-sign rotation puts the peak at the mirror angle, which is silently wrong rather
than obviously broken — a real bug here, caught only by a test asserting the peak location.

**Pairs are fitted jointly, not separately and averaged.** All the pairs in a session watched the
same head turn, so there is one speed to find. Fitting each alone and taking the median discards
that, and on real data it fails: single-pair estimates scatter by several times their own median and
fail *one-sided* — a pair with no usable signal runs high rather than scattering symmetrically — so
the median inherits the tail instead of averaging it away. Fitting jointly removes the failure mode
rather than averaging over it, because a wrong speed must pay a cost in every pair at once. Verified
on synthetic data with as many uninformative pairs mixed in as real ones.

**Slow rate fluctuation is removed first, and lags are capped at 1 s.** Firing rates co-vary on
timescales of seconds for reasons unrelated to turning — brain state, arousal, drift. That produces a
broad correlogram component the model cannot distinguish from a very slow sweep, and it fits it by
driving the speed towards zero: uncorrected, sessions returned 0.05 rad/s, the minimum of the search
grid, against a measured 0.63. Rates are high-passed at 5 s and lags capped at head-turn timescales.
This is a genuine tension with the previous point — a slow sweep wants a *wide* window to reach the
1–6 rad the fit needs, and that is exactly where the confound lives.

## angular1_exp3 — one decoder per wake bout

exp2 establishes that the estimate does not track measured head speed. It cannot say *why*, because
two different things could be wrong: the population decode, or the cell-pair estimator built on top
of it. exp3 separates them.

### Why the bout is the unit

DANDI's `states` table scores a whole day-long recording, so `"Awake"` is not one behavioural
session. Mouse28-140313 has 67 "Awake" intervals totalling 17045 s: several multi-thousand-second
bouts separated by sleep, plus a tail of brief arousals. The rest of this repo pools them, embeds the
concatenation and fits one ring to it.

That is measurably wrong. Held-out circular RMSE of the decoded angle against measured head
direction, same code, three different definitions of wake:

| session | all DANDI "Awake" | longest contiguous bout | CRCNS `.states.Wake` | published |
|---|---|---|---|---|
| Mouse28-140313 | 0.968 | 0.396 | 0.387 | 0.358 |
| Mouse12-120808 | 1.449 | 1.500 | 0.698 | 0.361 |
| Mouse25-140130 | 1.480 | 0.426 | 0.424 | 0.405 |

Chance is π/√3 = 1.81 rad, so the pooled numbers are close to no information at all, while a single
contiguous bout reproduces the published decode. **Pooling heterogeneous epochs, not the decoder,
was the fault.** (The predecessor repo never hit this because it read wake from the CRCNS
`.states.Wake` file, which happens to contain the single behavioural interval; it had no bout
selection logic at all.)

No heuristic is used to guess which bout is "the real one". Every wake bout of at least 300 s gets
its own decoder and its own quality number, and the quality number decides what is usable.

### Per bout

1. **Select** contiguous wake bouts of at least `min_bout_s` (default 300 s = 3000 rate bins).
   Bouts closer than `merge_gap_s` are merged first; the default 0 keeps DANDI's scoring as given.
2. **Embed and fit that bout alone** — Isomap, then the 12-knot ring, exactly as elsewhere.
3. **Split internally**: fit on `train_frac` of the bout's points, decode the held-out remainder,
   register to the measured angle by the offset-and-flip grid search, and take the circular RMSE.
   Repeated over `n_splits` splits. A bout passes if its mean RMSE is below `rmse_threshold`
   (0.5 rad, the published bar).
4. **Decode the whole bout** from the first split's ring, and measure three angular speeds on it:

   | quantity | source | what it tests |
   |---|---|---|
   | measured | tracking LEDs | ground truth |
   | decoded | angular speed of this bout's decoded ring angle | the decoder |
   | correlogram | the Peyrache cell-pair estimator on the same bout | that estimator |

**The logic.** All three are measured on the same bout, so they are directly comparable. If decoded
tracks measured but correlogram does not, the decode is sound and the cell-pair estimator is the
broken part. If neither tracks measured on bouts whose RMSE says the decode is good, the fault is
further upstream.

Angular speed needs no registration — it is invariant to the arbitrary shift and flip the ring
parameterisation carries — which is exactly what makes the same measurement possible in REM, where
there is no measured angle to register against. Registration is used only for the RMSE and for the
per-bin error drawn in the figure.

Code: `src/decode/bout_decode.py` (per-bout decode), `loader.contiguous_bouts` / `loader.longest_bout`
(bout selection), `figures.panels.bout_speed_traces` (the figure).

```bash
uv run hd-exp collect angular1 --stages bouts --sessions 25-140130 28-140313   # ~2.5 h
uv run hd-exp run angular1
```

Traces are cached to `outputs/cache/angular1_traces_<session>.npz` so the figure can be redrawn
without repeating the decode.

## Is the ground truth itself right?

Every claim above grades an estimate against the measured head direction, so that measurement had to
be checked before any of it means anything. Four checks, run on Mouse25-140130 and Mouse28-140313.

**1. Round trip.** Differentiate the measured angle to a *signed* angular velocity, integrate it
back, compare. Max error 8×10⁻¹³ rad. This cannot validate the magnitude — the reconstruction is
exact by construction — but it does confirm the wrapping, the sign convention and the time base.
`angular_velocity` / `integrate_velocity`, pinned by a test.

**2. An independent implementation of the same signal.** The CRCNS `.ang` files hold Neuroscope's own
head direction for these sessions, from the same LEDs, computed by different code and sample-aligned
with the DANDI series. Angular speed on **identical samples**:

| session | path ours | path CRCNS | ratio | net@1s ours | net@1s CRCNS | ratio |
|---|---|---|---|---|---|---|
| Mouse25-140130 | 1.176 | 1.057 | 1.11 | 0.464 | 0.454 | **1.02** |
| Mouse28-140313 | 1.006 | 0.899 | 1.12 | 0.407 | 0.395 | **1.03** |
| Mouse12-120808 | 4.318 | 1.317 | 3.28 | 0.815 | 0.551 | 1.48 |
| Mouse17-130130 | 2.249 | 0.978 | 2.30 | 0.556 | 0.430 | 1.29 |

Verified to 2–3% for Mouse25 and Mouse28. **Not** for Mouse12 and Mouse17, where the LED-derived
angle is markedly noisier than `.ang` — a caveat on any measured speed for those animals.

**3. The LED-to-head-axis convention differs per animal.** Comparing our angle to `.ang` across the
19 sessions with CRCNS data gives a constant rotation, tightly consistent within each animal:
Mouse25 ≈ 0, Mouse12 and Mouse17 ≈ +π/2, Mouse28 ≈ ±π (circular correlation 0.84–0.997). A constant
rotation cancels in differencing, in tuning-curve correlations ρ(θ), and in the decode RMSE, which
fits an offset and flip anyway — so nothing computed here is affected. It would matter the moment an
absolute preferred direction is reported.

**4. Path-length speed is not a well-defined quantity; net speed is.** Decimating the same wake bout:

| effective rate | 39 Hz | 19.5 Hz | 9.8 Hz | 4.9 Hz | 2.4 Hz |
|---|---|---|---|---|---|
| path speed, no smoothing | 1.675 | 1.474 | 1.174 | 0.917 | 0.688 |
| **net speed at τ = 1 s** | **0.467** | **0.473** | **0.474** | **0.474** | **0.419** |

Path length falls 2.4× purely with sampling rate, because finer sampling accumulates more tracking
jitter: it measures the instrument as much as the head, and two signals at different effective
bandwidths cannot be compared with it. Net speed at a fixed τ is invariant, so **net speed is the
ground truth used for grading**, with path length reported alongside. Pinned by a test.

Script: `scripts/verify_measured_speed.py` (checks 1–4 on real sessions);
`tests/test_head_direction.py` (the properties, on synthetic traces).

## Verification

The estimator makes a strong claim — a number for a quantity nobody measured — so it is checked
three ways.

**1. Synthetic recovery (`tests/test_angular_speed.py`).** A head-direction trace is generated at a
*chosen* mean speed, von Mises cells are given Poisson spikes from it, and the full estimator is run
on those spikes. It must return the speed it was given, within a factor of two, at true speeds of
0.5, 1, 3 and 6 rad/s; and the four estimates must come back in the right order. This is the check
that the implementation measures what it claims, and it is what caught all three bugs above.

**2. Measured ground truth in wake (angular1_exp2).** During wake the head angle is measured from
the LEDs, so the answer is independently known. The estimate is compared against it per session:
correlation across sessions, and the ratio of estimate to truth.

The comparison is made against the **net** angular speed, not the path length, and the distinction
decides whether the estimator looks calibrated or looks broken by a factor of two:

- *path-length* speed is mean |dθ/dt| — a head that turns left then right has moved twice;
- *net* speed is the mean displacement between the two ends of a window of length τ, divided by τ —
  those two turns cancel.

The estimator reads net displacement, because a lag τ enters its model as a single angular offset
*v*·τ. Path length is an upper bound on the net speed and the gap widens with τ. Both are reported;
the net speed at τ = 1 s is the like-for-like comparison. Reporting only the path-length ratio would
attribute a genuine difference of definition to a fault in the estimator.

**3. REM against wake.** Not ground truth, but a sanity constraint: whatever the absolute
calibration, REM and wake are estimated by identical machinery on the same cells, so their ratio is
interpretable even if the scale is not.

## Outcome

**The cell-pair estimator fails; the decode does not.** The estimator recovers a known speed from
synthetic spikes, but on these recordings its session-to-session variation does not track the
measured head speed at all, so the exp1 numbers should not feed the between-animal comparison this
question was written for.

exp3 localises that. On wake bouts whose own decode is verified against measured head direction,
angular speed read straight off the decoded ring angle **is** the measured speed. So the quantity is
recoverable from these recordings — just not by cell-pair correlograms, and only where a decode
carries a quality number.

Without `angular1_exp2` the speeds would have looked entirely reasonable and gone into the
comparison; without `angular1_exp3` the failure would have been blamed on the decode.

## Known limitations

- **A χ²(3) prior is assumed, not fitted.** The paper's choice. `SPEED_MODELS` also offers χ²(1),
  exponential and Rayleigh; `--speed-model` switches between them, and their spread is the honest
  measure of how much this assumption matters.
- **Tuning curves are estimated in wake and assumed to hold in REM.** If preferred directions drift
  between states, the REM estimate is biased by an amount this design cannot see.
- **A session's estimate is one number for the whole state.** Speed certainly varies within a
  session — wake in particular mixes long immobile stretches with brief fast turns — and nothing
  here resolves that. This is a leading suspect for the validation failure.
- **Removing slow rate covariation also removes slow real sweeps.** The high-pass cannot tell the
  two apart, so genuinely slow turning is attenuated along with the confound.
