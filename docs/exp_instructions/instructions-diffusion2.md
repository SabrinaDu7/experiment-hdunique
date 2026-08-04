2026-08-04

# diffusion2: Is the long-lag sub-diffusion real, or manufactured by the decoder and the estimator?

## Question

`diffusion1` finds the decoded angle is sub-diffusive at long lags — it covers less ground than a
free random walk would. Is that the head-direction system, or is it our instrument?

## Motivation

Three different things produce "*D* falls with the window", and only one is about the brain:

1. **The brain** — the bump really is restrained.
2. **The decoder** — noisy points projected onto a ring can be pulled toward densely-sampled arcs,
   faking attraction that is not in the data.
3. **The estimator** — once the angle has fully scrambled, −2·ln⟨cos Δα⟩ cannot distinguish "fast"
   from "very fast" and reports slowing where it has simply gone blind.

Every number in `diffusion1` mixes all three. Nothing in that analysis can separate them, because
there is no case where the true answer is known.

## Experiments

- **diffusion2_exp1** — push a walk that is **free by construction** through each session's own
  fitted ring and its own *D* calculation, and see what comes back.

## Methods

```bash
uv run hd-exp collect diffusion2     # ~30 s per session, ~20 min for the full set
uv run hd-exp run     diffusion2
```

Code: `src/experiments/diffusion2.py`; the control itself is `scripts/synthetic_ring_control.py`.

**The logic is calibration.** α = 1 means "free walk". If the machinery returns 1 on data known to
be free, it is honest and a real value below 1 is the brain. If it returns less than 1 on a free
walk, it is lying and the real-data result means nothing.

Per session:

1. **Refit the ring** on the cached Isomap embedding (single fit, seed 0 — verified to reproduce
   the cached decode to a circular correlation of 0.999).
2. **Measure the decoder's real working noise**: the 3-D offset of every real embedded point from
   its nearest point on the curve.
3. **Simulate a free wrapped random walk** per bout at the session's own measured rate, so speed
   and bout structure match the real data. ⟨Δα²⟩ = D·τ exactly, no confinement anywhere.
4. **Place it on the ring** and decode it back through the *same* ring, twice: `clean` (points
   exactly on the curve, isolating the parameterisation) and `noisy` (plus resampled residuals,
   adding projection noise).
5. **Measure α** identically to the real analysis.

Differencing the nested outputs attributes the shortfall to each stage: `1 − alpha_truth` is the
estimator, `alpha_truth − alpha_noisy` the decoder, and `alpha_real − alpha_noisy` — the
**deficit** — is the only part that could be dynamics.

⚠️ **The deficit is an upper bound.** Residuals are resampled i.i.d., which destroys their temporal
correlation. Real off-ring excursions are likely persistent, which would make the decoder *more*
damaging than this control allows, so `alpha_noisy` is optimistic. Resampling residuals in
contiguous blocks is the fix and has not been done.

⚠️ The control starts at the **embedding**, not at spikes, so it does not test rate estimation or
Isomap. A full end-to-end synthetic would simulate spikes from tuning curves.

**Defaults.** `cell_set=ADn`, all cached sessions, `honest_alpha=0.9`, `seed=0`.
