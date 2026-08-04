2026-08-04

# diffusion1: How does the REM diffusion constant depend on the measurement window?

## Question

*D* is defined as the slope of ⟨Δα²⟩ against lag, fitted over the first 200 ms. Does the answer
change if the same quantity is measured over 500 ms, or 5 s?

## Motivation

A diffusion *constant* should be constant: for a plain random walk the same number comes back
whatever window it is fitted over. If it does not, the process is not a plain random walk, and
"the REM diffusion constant" is not a single number — quoting one requires quoting the window.

Answering this needs circular wrapping dealt with first. Angular differences fold into (−π, π], so
⟨Δα²⟩ has a hard ceiling at **π²/3 ≈ 3.29 rad²**. That is irrelevant at 200 ms and dominant at 5 s.

## Experiments

- **diffusion1_exp1** — *D* refitted at every window from 200 ms to 5 s, per session, plus the
  dimensionless shape of the curve.
- **diffusion1_exp2** — the same comparison split by cell set, to see whether the pattern is a
  property of the recording or of the analysis.

## Methods

The decoding method is SPUD (spline parameterization for unsupervised decoding) from
**Chaudhuri et al. (2019),** [*The intrinsic population dynamics of a canonical cognitive
circuit*](https://www.nature.com/articles/s41593-019-0460-x), Nature Neuroscience. The data is
[DANDI dandiset 000056](https://dandiarchive.org/dandiset/000056) (Peyrache et al. 2015).

```bash
uv run hd-exp collect diffusion1     # long-lag curves for all three cell sets (~5 min)
uv run hd-exp run     diffusion1
uv run hd-exp run     diffusion1 --estimator wrapped --cell-set PoS
```

Code: `src/experiments/diffusion1.py`; curves from `src/timescale.py`, fitting from
`src/diffusion.py` (`window_slope`), figure from `src/figures/strips.py`.

**Three estimators of the same ⟨Δα²⟩**, computed side by side because they fail differently:

| method | definition | fails when |
|---|---|---|
| `wrapped` | ⟨signed_diff(α(t+τ), α(t))²⟩ | displacement approaches π — hard ceiling at π²/3 |
| `unwrapped` | cumulative-sum the per-bin steps, then difference | any single step exceeds π; the error then integrates forever |
| `circular` | −2·ln⟨cos Δα⟩ | ⟨cos Δα⟩ approaches 0 |

`circular` is exact for a wrapped Gaussian (⟨cos Δα⟩ = exp(−σ²/2)) and needs no unwrapping
decision, so it is the default at long lags. `wrapped` remains correct at 200–500 ms and is what
the published *D* uses.

**Everything is bout-aware.** A pair spanning the gap between two REM bouts is meaningless at any
lag and catastrophic at 5 s.

**The anomalous exponent α** — the log-log slope of ⟨Δα²⟩ against τ — is reported alongside *D*
because it is **dimensionless**, so any multiplicative bias in an estimator cancels out of it. α = 1
is ordinary diffusion. It is the comparison that does not depend on which estimator you believe.

⚠️ `circular` assumes a Gaussian displacement. At 200 ms the decoded displacement is strongly
leptokurtic, so `circular` under-reads there and by 5 s does not. That inflates the measured
long/short ratio, making the reported fall a **conservative bound**.

**Defaults.** `cell_set=ADn`, `windows_ms=(200, 500, 1000, 2000, 3000, 4000, 5000)`,
`reference_ms=200`, `estimator=circular`, `cell_sets=(ADn, ADn+PoS, PoS)`.
