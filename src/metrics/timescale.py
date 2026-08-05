"""Long-lag behaviour of the decoded angle, where circular wrapping stops being ignorable.

The mean-squared-displacement machinery itself lives in `metrics.diffusion`; this module adds the
diagnostics that only matter once lags are long enough for wrapping to bite.

The 200 ms estimator in `diffusion.py` measures ⟨Δα²⟩ from *wrapped* angular differences, which
fold into (−π, π]. That is harmless while the typical displacement is far below π. It stops being
harmless well before 5 s: a wrapped difference cannot exceed π, so ⟨Δα²⟩ is bounded above by
**π²/3 ≈ 3.29 rad²** — the variance of a uniform angle. Any session whose true displacement
approaches that ceiling has its diffusion constant silently compressed toward zero.

Three estimators of the same underlying ⟨Δα²⟩(τ) are therefore computed side by side:

| method | definition | fails when |
|---|---|---|
| `wrapped` | ⟨signed_diff(α(t+τ), α(t))²⟩ | displacement approaches π (hard ceiling at π²/3) |
| `unwrapped` | cumulative-sum the per-bin signed steps, then take plain differences | a single per-bin step exceeds π, whose sign is then guessed wrong and integrated forever |
| `circular` | −2·ln⟨cos Δα⟩ | ⟨cos Δα⟩ approaches 0 (no hard ceiling, but the log blows up) |

`circular` is exact for a wrapped Gaussian random walk: if the unwrapped displacement is
N(0, σ²) then ⟨cos Δα⟩ = exp(−σ²/2), so −2·ln⟨cos Δα⟩ recovers σ² whatever the wrapping does.

**The two circumventions fail in opposite directions, which is what makes them useful together.**
`unwrapped` integrates decode jitter into a spurious random walk, so it is an *upper* bound;
`circular` assumes the displacement is Gaussian, and non-diffusive jitter pushes it *down*. Where
they agree the long-lag estimate is trustworthy; where they diverge the session cannot support one,
and `unwrapped_over_circular` quantifies that directly.

All curves here are **bout-aware** — a pair spanning the gap between two REM bouts is meaningless at
any lag, and catastrophic at 5 s.
"""

import numpy as np
from beartype import beartype
from jaxtyping import Float, jaxtyped

import spud.angle_fns as af
from metrics.diffusion import (
    METHODS,
    UNWRAP_RISK_THRESHOLD,
    WRAPPED_CEILING,
    msd_curve,
    split_bouts,
    unwrap_bout,
)


@jaxtyped(typechecker=beartype)
def unwrap_risk(*, angles: Float[np.ndarray, " time"], bout_lengths: list[int]) -> float:
    """Fraction of per-bin steps whose unwrap direction is ambiguous (|step| > π/2).

    This is the trustworthiness flag for the `unwrapped` estimator: every such step is a coin flip
    that the cumulative sum then carries forward for the rest of the bout. Clean sessions sit near
    0.001; a session above ~0.05 should not be read at long lags.
    """
    steps = np.concatenate(
        [af.shifted_angular_diffs(b, 1) for b in split_bouts(angles=angles, bout_lengths=bout_lengths) if len(b) > 1]
    )
    return float((np.abs(steps) > UNWRAP_RISK_THRESHOLD).mean())


@jaxtyped(typechecker=beartype)
def resultant_length(
    *, angles: Float[np.ndarray, " time"], bout_lengths: list[int], lag: int
) -> float:
    """Mean resultant length ⟨cos Δα⟩ at one lag — how much memory of the angle survives.

    This is what sets the ceiling on *any* estimator, not just the wrapped one. Once the
    displacement distribution is uniform on the circle, ⟨cos Δα⟩ = 0 and the data no longer contain
    the information needed to recover a rate: fast and very fast look identical. The `circular`
    estimator is −2·ln of this, so it loses sensitivity exactly as this approaches 0 and any small
    positive bias then dominates. Below ~0.05 the long-lag estimate should be read as a lower bound.
    """
    bouts = [b for b in split_bouts(angles=angles, bout_lengths=bout_lengths) if len(b) > lag]
    if not bouts:
        return float("nan")
    pooled = np.concatenate([af.shifted_angular_diffs(b, lag) for b in bouts])
    return float(np.mean(np.cos(pooled)))


@jaxtyped(typechecker=beartype)
def decorrelation_lag(
    *, angles: Float[np.ndarray, " time"], bout_lengths: list[int], lags: tuple[int, ...]
) -> int:
    """First lag (in bins) at which ⟨cos Δα⟩ falls below 1/e, or 0 if it never does.

    Beyond this lag the angle has essentially forgotten where it started, so no estimator can
    recover a diffusion rate from it — the honest answer there is "faster than this window can
    resolve", not a number.
    """
    for lag in lags:
        if resultant_length(angles=angles, bout_lengths=bout_lengths, lag=lag) < np.exp(-1.0):
            return lag
    return 0


@jaxtyped(typechecker=beartype)
def anomalous_exponent(
    *, curve: Float[np.ndarray, " lag"], lags_s: Float[np.ndarray, " lag"], lo_s: float, hi_s: float
) -> float:
    """Log-log slope of MSD against lag over [lo_s, hi_s]: the anomalous-diffusion exponent α.

    ⟨Δα²⟩ ∝ τ^α, so α = 1 is ordinary diffusion, α < 1 sub-diffusive, α > 1 super-diffusive. This is
    the estimator-robust way to ask whether the process is still diffusive at a given timescale:
    unlike D it is dimensionless, so a multiplicative bias in the estimator cancels out of it
    entirely. Returns NaN if the window holds fewer than two usable points.
    """
    curve = np.asarray(curve, dtype=float)
    usable = (lags_s >= lo_s) & (lags_s <= hi_s) & np.isfinite(curve) & (curve > 0)
    if usable.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(lags_s[usable]), np.log(curve[usable]), 1)[0])


@jaxtyped(typechecker=beartype)
def displacement_kurtosis(
    *, angles: Float[np.ndarray, " time"], bout_lengths: list[int], lag: int
) -> float:
    """Kurtosis of the wrapped displacement at one lag (Gaussian = 3).

    The `circular` estimator is exact only for a Gaussian displacement. At short lags the decoded
    angle is strongly leptokurtic — a spike of near-zero steps plus rare jumps — which makes
    ⟨cos Δα⟩ larger than a Gaussian of the same variance would give, so `circular` under-reads.
    By several seconds the central limit theorem has done its work and the bias goes away. Reporting
    this makes the direction of that bias checkable rather than assumed.
    """
    bouts = [b for b in split_bouts(angles=angles, bout_lengths=bout_lengths) if len(b) > lag]
    if not bouts:
        return float("nan")
    pooled = np.concatenate([af.shifted_angular_diffs(b, lag) for b in bouts])
    centred = pooled - pooled.mean()
    variance = float(np.mean(centred**2))
    return float(np.mean(centred**4) / variance**2) if variance > 0 else float("nan")


@jaxtyped(typechecker=beartype)
def pairs_per_lag(*, bout_lengths: list[int], lags: tuple[int, ...]) -> list[int]:
    """Number of within-bout pairs available at each lag — how much data each point rests on.

    Long lags are supported by far fewer pairs than short ones (a 5 s lag needs a bout longer than
    5 s), so this is what stops a thinly-supported long-lag point being read as if it were as solid
    as the 200 ms one.
    """
    return [int(sum(max(0, length - lag) for length in bout_lengths)) for lag in lags]


#: Re-exported so callers can keep importing the MSD machinery from here; the implementation is in
#: `metrics.diffusion`, which is the single definition of a diffusion curve in this repo.
__all__ = [
    "METHODS", "UNWRAP_RISK_THRESHOLD", "WRAPPED_CEILING", "anomalous_exponent", "decorrelation_lag",
    "displacement_kurtosis", "msd_curve", "pairs_per_lag", "resultant_length", "split_bouts",
    "unwrap_bout", "unwrap_risk",
]
