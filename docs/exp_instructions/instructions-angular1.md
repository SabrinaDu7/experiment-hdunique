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

**The validation fails.** The estimator recovers a known speed from synthetic spikes, but on these
recordings its session-to-session variation does not track the measured head speed at all. The
numbers in [`results_angular1.md`](../exp_results/results_angular1.md) should not be used for the
between-animal comparison this question was written for; that document reports the failure, the
diagnostics that localise it, and what is most likely wrong.

The value delivered here is therefore the verification apparatus, not the metric: without
`angular1_exp2` the speeds would have looked entirely reasonable and gone into the comparison.

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
