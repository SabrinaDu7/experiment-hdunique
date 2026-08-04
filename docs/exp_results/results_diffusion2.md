2026-08-04

<!-- GENERATED from results_diffusion2.in — do not edit; edit the .in and re-run `hd-exp render`. -->

# diffusion2 — results

> **Question.** `diffusion1` finds the decoded angle is sub-diffusive at long lags. Is that the
> head-direction system, or is it our instrument?

Methods: [`instructions-diffusion2.md`](../exp_instructions/instructions-diffusion2.md).

**32 sessions.** α = 1 means "free walk"; anything less was manufactured somewhere.

---

## diffusion2_exp1 — a known-free walk through the same machinery

| stage | median α |
|---|---|
| free walk, before decoding | **1.06** |
| free walk, points exactly on the ring | 1.06 |
| free walk + realistic off-ring noise | **0.88** |
| **real data** | **0.54** |

How the shortfall from a free walk divides:

| source | contribution |
|---|---|
| estimator blindness | -0.055 |
| decoder / off-ring noise | +0.154 |
| **dynamics (unexplained)** | **+0.286** |

The estimator fails in only **2 of 32** sessions
(Mouse20-130514, Mouse20-130515):

| session_id | n_cells | alpha_truth | alpha_noisy | alpha_real |
|---|---|---|---|---|
| Mouse20-130514 | 5 | 0.27 | 0.05 | 0.19 |
| Mouse20-130515 | 6 | 0.33 | 0.30 | 0.34 |

**Deficit** — real data against a free walk through the *same* decoder, at the *same* speed, with
the *same* bout structure:

- median **-0.286**, IQR -0.377 to -0.191
- negative in **29 of 32** sessions, p = 2.6e-08
- independent of cell count (ρ = 0.197, p = 0.28) and of speed
  (ρ = -0.218, p = 0.23)

Restricted to the **15 sessions where the machinery is verified honest** on both counts:
deficit **-0.249**, negative in 15 of 15,
p = 6.1e-05.

| session_id | n_cells | alpha_real | alpha_truth | alpha_noisy | deficit |
|---|---|---|---|---|---|
| Mouse24-131216 | 14 | 0.406 | 1.525 | 1.399 | -0.994 |
| Mouse17-130125 | 3 | 0.108 | 1.197 | 1.056 | -0.948 |
| Mouse25-140124 | 10 | 0.348 | 1.060 | 0.893 | -0.546 |
| Mouse17-130204 | 23 | 0.400 | 1.093 | 0.854 | -0.454 |
| Mouse12-120808 | 39 | 0.552 | 1.050 | 1.002 | -0.449 |
| Mouse25-140206 | 14 | 0.618 | 1.108 | 1.058 | -0.440 |
| Mouse20-130517 | 16 | 0.399 | 1.072 | 0.810 | -0.411 |
| Mouse17-130203 | 28 | 0.441 | 1.051 | 0.843 | -0.402 |
| Mouse28-140311 | 14 | 0.623 | 1.096 | 0.992 | -0.369 |
| Mouse12-120807 | 40 | 0.559 | 1.042 | 0.911 | -0.352 |
| Mouse20-130520 | 10 | 0.447 | 1.003 | 0.795 | -0.348 |
| Mouse20-130516 | 9 | 0.302 | 1.360 | 0.621 | -0.319 |
| Mouse24-131218 | 16 | 0.546 | 1.207 | 0.858 | -0.311 |
| Mouse17-130202 | 25 | 0.375 | 1.010 | 0.678 | -0.303 |
| Mouse28-140317 | 20 | 0.820 | 1.172 | 1.115 | -0.295 |
| Mouse17-130201 | 26 | 0.341 | 1.063 | 0.634 | -0.293 |
| Mouse17-130129 | 25 | 0.538 | 1.019 | 0.817 | -0.279 |
| Mouse17-130128 | 19 | 0.141 | 1.002 | 0.399 | -0.258 |
| Mouse25-140205 | 18 | 0.749 | 1.065 | 0.999 | -0.249 |
| Mouse28-140313 | 24 | 0.796 | 1.062 | 1.034 | -0.238 |
| Mouse25-140204 | 22 | 0.814 | 1.081 | 1.049 | -0.235 |
| Mouse25-140130 | 17 | 0.774 | 1.028 | 1.002 | -0.228 |
| Mouse17-130130 | 26 | 0.623 | 0.960 | 0.845 | -0.222 |
| Mouse12-120810 | 44 | 0.861 | 1.097 | 1.056 | -0.195 |
| Mouse28-140318 | 23 | 0.745 | 1.098 | 0.925 | -0.180 |
| Mouse12-120809 | 49 | 0.750 | 1.045 | 0.911 | -0.161 |
| Mouse25-140131 | 20 | 0.788 | 1.010 | 0.947 | -0.159 |
| Mouse24-131217 | 16 | 0.378 | 0.953 | 0.525 | -0.147 |
| Mouse12-120806 | 39 | 0.561 | 1.002 | 0.612 | -0.052 |
| Mouse20-130515 | 6 | 0.336 | 0.332 | 0.303 | 0.032 |
| Mouse20-130514 | 5 | 0.189 | 0.271 | 0.046 | 0.143 |
| Mouse17-130131 | 22 | 0.501 | 0.970 | 0.339 | 0.161 |

### Interpretation

**The ring parameterisation is innocent.** `clean` ≈ `truth` everywhere: placing points exactly on
the fitted curve and decoding them recovers the walk. Uneven arc-length sampling — the most obvious
way a spline ring could fake attraction — is not happening.

**The estimator is honest almost everywhere.** It fails only in the two thinnest, fastest sessions,
where the angle has fully decorrelated by 5 s and no estimator could recover a rate. Elsewhere a
free walk comes back as free.

**Projection noise matters, but is not the whole story**, and the residual deficit is
**independent of both cell count and speed** — so it is not a recording-quality artefact wearing a
different hat.

**A real effect survives all three.** The deficit is negative in almost every session and in every
one of the 15 sessions where the machinery is verified clean.

---

## Answer to the question

**Mostly the brain, but the raw number badly overstated it.**

The decoded angle is genuinely sub-diffusive: something restores the bump toward preferred
directions during REM, when there is no vestibular input to anchor it to. But the honest effect size
is the deficit against a matched free walk — about -0.249 — not the raw α ≈
0.54, which conflates dynamics with a decoder that costs +0.154 on its own.

⚠️ **Treat -0.249 as an upper bound.** Off-ring residuals are resampled i.i.d.,
which destroys their temporal correlation and so flatters the decoder. Real off-ring excursions
likely come in runs, which would make the decoder more damaging and the true deficit smaller.

## How to reproduce

```bash
uv run hd-exp collect diffusion2     # ~20 min
uv run hd-exp run     diffusion2
uv run hd-exp check   diffusion2
```

## Next steps

- **Block-resample the residuals** before quoting the effect size. Small change, and it is the one
  caveat standing between this and a firm number.
- A full end-to-end synthetic, simulating spikes from tuning curves rather than starting at the
  embedding, would additionally test rate estimation and Isomap.

## Provenance

Generated 2026-08-04T16:26:35+00:00 from commit `a19075c`.

| config | value |
|---|---|
| `cell_set` | `ADn` |
| `honest_alpha` | `0.9` |
| `seed` | `0` |
| `sessions` | `[]` |

| input | value |
|---|---|
| `input_sessions` | `32` |
