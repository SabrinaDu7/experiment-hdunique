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
docs/2026-08-02-port-rem-diffusion-and-variance.md, problem P2.)
"""

import numpy as np
from beartype import beartype
from jaxtyping import Float, jaxtyped

import spud.angle_fns as af


@jaxtyped(typechecker=beartype)
def diffusion_curve(
    *,
    angles: Float[np.ndarray, " time"],
    lags: tuple[int, ...],
    bout_lengths: list[int] | None = None,
) -> Float[np.ndarray, " lag"]:
    """Mean squared angular change <(alpha(t+tau) - alpha(t))^2> at each lag, in rad^2.

    `bout_lengths` splits `angles` into the bouts it was concatenated from, so that no pair
    straddles a bout boundary. Pass None to pool across the whole concatenated trace, which is what
    the source pipeline did and what the published numbers use; the boundary-crossing pairs are
    separated by minutes of unscored sleep and inflate the curve slightly (see
    docs/2026-08-02-port-rem-diffusion-and-variance.md, open question Q1).
    """
    if bout_lengths is None:
        return af.get_diffusion_curve(angles, np.asarray(lags))

    segments = np.split(angles, np.cumsum(bout_lengths)[:-1])
    out = []
    for lag in lags:
        diffs = [af.shifted_angular_diffs(s, lag) for s in segments]
        pooled = np.concatenate([d for d in diffs if len(d) > 0])
        out.append(float(np.mean(pooled**2)))
    return np.array(out)


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
def window_slope(
    *, curve: Float[np.ndarray, " lag"], dt: float, window_ms: int
) -> tuple[float, float]:
    """D and r2 from the origin-forced fit over the first `window_ms` of the diffusion curve.

    Every window reads the *same* curve, so the 200 -> 500 ms progression is a linearity probe:
    flat means genuinely diffusive, monotonically falling means the curve saturates inside the
    window (crossover tau_c ~ pi^2 / 6D). It is not a set of alternative estimates of D.
    """
    n = round(window_ms / 1000.0 / dt)
    lags_s = np.arange(1, n + 1, dtype=float) * dt
    return slope_through_origin(lags_s=lags_s, values=curve[:n])


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


def window_column(*, stat: str, window_ms: int, headline_ms: int) -> str:
    """Parquet column name for a per-window statistic: the headline window is unsuffixed ("D"),
    wider windows are suffixed ("D_500"). Defined once so writer and readers agree."""
    return stat if window_ms == headline_ms else f"{stat}_{window_ms}"
