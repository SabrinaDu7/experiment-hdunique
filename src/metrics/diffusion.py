"""Diffusion curve and diffusion constant of a decoded angle trajectory.

The paper: *"The diffusion curve at time shift tau is D(tau) = <(alpha(t+tau) - alpha(t))^2>_t,
where the average is taken over all pairs of time points separated by tau. To compute diffusion
constants, we fitted a straight line to the first 200 ms of the squared change in decoded angle
against time."*

D is reported as the **slope** of that line (not slope/2), matching the paper's wording. Note this
is twice the diffusion coefficient of the physics convention.

The line is forced through the origin: at zero lag the squared change is exactly zero, so the
intercept is not a free parameter. (The source repo instead added (0,0) as a data point to a
free-intercept fit while calling it "through the origin" — see
docs/porting/2026-08-02-port-rem-diffusion-and-variance.md, problem P2.)
"""

import numpy as np
from beartype import beartype
from jaxtyping import Float, jaxtyped

import spud.angle_fns as af

#: Upper bound on wrapped ⟨Δα²⟩: the variance of a uniformly distributed angle, π²/3.
WRAPPED_CEILING: float = float(np.pi**2 / 3.0)

#: Per-bin steps above this magnitude have an ambiguous unwrap direction. π/2 is deliberately
#: conservative: the sign is only truly a coin flip at π, but a step of π/2 already means the
#: decoded angle moved a quarter of the ring in 100 ms.
UNWRAP_RISK_THRESHOLD: float = float(np.pi / 2)

#: Estimator names, in the order they are reported.
METHODS: tuple[str, ...] = ("wrapped", "unwrapped", "circular")


@jaxtyped(typechecker=beartype)
def split_bouts(
    *, angles: Float[np.ndarray, " time"], bout_lengths: list[int]
) -> list[Float[np.ndarray, " _"]]:
    """Split a concatenated decoded trace back into its REM bouts."""
    return list(np.split(angles, np.cumsum(bout_lengths)[:-1]))


@jaxtyped(typechecker=beartype)
def unwrap_bout(*, bout: Float[np.ndarray, " time"]) -> Float[np.ndarray, " time"]:
    """Lift one bout's wrapped angles onto the real line by accumulating signed per-bin steps.

    Each step is taken as the shortest way round, so the result is exact as long as the true
    per-bin motion never exceeds π. Unwrapping is done per bout and never across a bout boundary,
    where the true displacement is unknowable.
    """
    if len(bout) < 2:
        return np.zeros(len(bout))
    steps = af.shifted_angular_diffs(bout, 1)
    return np.cumsum(np.concatenate([[0.0], steps]))


@jaxtyped(typechecker=beartype)
def msd_curve(
    *,
    angles: Float[np.ndarray, " time"],
    bout_lengths: list[int],
    lags: tuple[int, ...],
    method: str,
) -> Float[np.ndarray, " lag"]:
    """Mean squared angular displacement at each lag (in bins), bout-aware, in rad².

    `method` selects one of the three estimators documented at the top of this module. A lag longer
    than a bout simply contributes no pairs from that bout; a lag longer than *every* bout yields
    NaN at that lag rather than a silently empty mean.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; expected one of {METHODS}.")

    bouts = split_bouts(angles=angles, bout_lengths=bout_lengths)
    if method == "unwrapped":
        bouts = [unwrap_bout(bout=b) for b in bouts]

    out = []
    for lag in lags:
        usable = [b for b in bouts if len(b) > lag]
        if not usable:
            out.append(float("nan"))
            continue
        if method == "unwrapped":
            pooled = np.concatenate([b[lag:] - b[:-lag] for b in usable])
            out.append(float(np.mean(pooled**2)))
        else:
            pooled = np.concatenate([af.shifted_angular_diffs(b, lag) for b in usable])
            if method == "wrapped":
                out.append(float(np.mean(pooled**2)))
            else:  # circular
                resultant = float(np.mean(np.cos(pooled)))
                out.append(float(-2.0 * np.log(resultant)) if resultant > 0 else float("nan"))
    return np.array(out)


@jaxtyped(typechecker=beartype)
def diffusion_curve(
    *,
    angles: Float[np.ndarray, " time"],
    lags: tuple[int, ...],
    bout_lengths: list[int] | None = None,
) -> Float[np.ndarray, " lag"]:
    """Mean squared angular change <(alpha(t+tau) - alpha(t))^2> at each lag, in rad^2.

    A thin alias for `msd_curve(method="wrapped")`, kept because "diffusion curve" is the name the
    paper uses. `bout_lengths=None` means pool across the whole concatenated trace — expressed as a
    single bout spanning everything, which is exactly what pooling is.
    """
    return msd_curve(
        angles=angles,
        bout_lengths=bout_lengths if bout_lengths is not None else [len(angles)],
        lags=lags,
        method="wrapped",
    )


@jaxtyped(typechecker=beartype)
def slope_through_origin(
    *, lags_s: Float[np.ndarray, " lag"], values: Float[np.ndarray, " lag"]
) -> tuple[float, float]:
    """Least-squares line forced through (0, 0): D = sum(xy)/sum(x^2). Returns (D, r2).

    r2 is measured against the zero-intercept model, so the total sum of squares is sum(y^2)
    (variation about zero, not about the mean) — consistent with the fitted model having no
    intercept.
    """
    x, y = np.asarray(lags_s, dtype=float), np.asarray(values, dtype=float)
    d = float(np.sum(x * y) / np.sum(x * x))
    ss_tot = float(np.sum(y**2))
    r2 = 1.0 - float(np.sum((y - d * x) ** 2)) / ss_tot if ss_tot > 0 else float("nan")
    return d, r2


@jaxtyped(typechecker=beartype)
def window_lags(*, dt: float, window_ms: int) -> Float[np.ndarray, " lag"]:
    """Lag axis (s) the origin-forced fit spans: every multiple of `dt` up to `window_ms`.

    Defined once so the fit, the bootstrap and the figures all agree on how many measured lags a
    named window covers — the source repo recomputed this rounding at each call site.
    """
    n = round(window_ms / 1000.0 / dt)
    return np.arange(1, n + 1, dtype=float) * dt


@jaxtyped(typechecker=beartype)
def window_slope(
    *, curve: Float[np.ndarray, " lag"], dt: float, window_ms: int
) -> tuple[float, float]:
    """D and r2 from the origin-forced fit over the first `window_ms` of the diffusion curve.

    Every window reads the *same* curve, so the 200 -> 500 ms progression is a linearity probe:
    flat means genuinely diffusive, monotonically falling means the curve saturates inside the
    window (crossover tau_c ~ pi^2 / 6D). It is not a set of alternative estimates of D.
    """
    lags_s = window_lags(dt=dt, window_ms=window_ms)
    return slope_through_origin(lags_s=lags_s, values=curve[: len(lags_s)])


@jaxtyped(typechecker=beartype)
def free_intercept_fit(
    *, curve: Float[np.ndarray, " lag"], dt: float, n_lags: int = 3
) -> tuple[float, float, float]:
    """Free-intercept OLS over the first `n_lags` *measured* lags. Returns (slope, nugget, r2).

    This is a diagnostic, not the estimator. Unlike the origin-forced fit it has a real residual
    degree of freedom, so it can detect curvature; and its intercept — the `nugget` — estimates the
    decode-jitter floor that the origin-forced fit is obliged to absorb into its slope. Clean
    sessions sit near zero; an inflated D shows up as a clearly positive nugget.
    """
    x = np.arange(1, n_lags + 1, dtype=float) * dt
    y = np.asarray(curve[:n_lags], dtype=float)
    design = np.vstack([x, np.ones_like(x)]).T
    (slope, intercept), *_ = np.linalg.lstsq(design, y, rcond=None)
    yhat = slope * x + intercept
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum((y - yhat) ** 2)) / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), r2


@jaxtyped(typechecker=beartype)
def epoch_starts(*, bout_lengths: list[int], epoch_bins: int) -> Float[np.ndarray, " epoch"]:
    """Start indices of non-overlapping `epoch_bins`-long epochs tiling each bout.

    Epochs never straddle a bout boundary, so every within-epoch pair is a genuine measurement at
    that lag. A bout shorter than one epoch contributes nothing.
    """
    starts: list[int] = []
    offset = 0
    for length in bout_lengths:
        n_epochs = length // epoch_bins
        starts.extend(offset + i * epoch_bins for i in range(n_epochs))
        offset += length
    return np.array(starts, dtype=float)


@jaxtyped(typechecker=beartype)
def bootstrap_ci(
    *,
    angles: Float[np.ndarray, " time"],
    bout_lengths: list[int],
    dt: float,
    window_ms: int,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    """The paper's bootstrap: resample `window_ms` epochs with replacement, as many as the data
    holds, recompute D from each resampling, and return the 2.5 / 97.5 percentiles.

    An epoch spans the whole fit window, so it carries one pair at every lag the fit uses (three
    100 ms bins for the 200 ms window: two lag-1 pairs and one lag-2 pair). Resampling whole epochs
    rather than individual pairs preserves the within-epoch correlation between lags, which is what
    makes this a bootstrap over the *data* rather than over the summary statistic.

    Returns (nan, nan) if `n_boot <= 0` or the session has too few complete epochs.
    """
    lags_s = window_lags(dt=dt, window_ms=window_ms)
    n_lags = len(lags_s)
    starts = epoch_starts(bout_lengths=bout_lengths, epoch_bins=n_lags + 1).astype(int)
    if n_boot <= 0 or len(starts) < 2:
        return float("nan"), float("nan")

    # Squared angular change at each lag, for every epoch: (n_epochs, n_lags).
    per_epoch = np.empty((len(starts), n_lags))
    for lag in range(1, n_lags + 1):
        diffs = af.signed_angular_diff(angles[starts + lag], angles[starts])
        per_epoch[:, lag - 1] = diffs**2

    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(starts), len(starts))
        estimates[b], _ = slope_through_origin(lags_s=lags_s, values=per_epoch[idx].mean(axis=0))
    return float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def window_column(*, stat: str, window_ms: int, headline_ms: int) -> str:
    """Parquet column name for a per-window statistic: the headline window is unsuffixed ("D"),
    wider windows are suffixed ("D_500"). Defined once so writer and readers agree."""
    return stat if window_ms == headline_ms else f"{stat}_{window_ms}"
