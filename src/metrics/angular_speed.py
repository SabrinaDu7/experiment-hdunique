"""Angular speed of the head-direction signal, estimated from spikes alone.

The method is Peyrache et al. (2015), *Internally organized mechanisms of the head direction sense*
(Nat. Neurosci. 18:569), "Estimation of angular velocity". Its value is that it needs **no measured
head angle**, so it works during sleep, where the animal is not tracked and the head is not moving
but the internal compass still is.

The argument, in one line: under independent rate coding, the temporal cross-correlation between two
HD cells at lag τ is approximately the angular correlation between their tuning curves evaluated at
the angle the head swept in that time. So

    C(τ)  ≈  ⟨ ρ(v·τ) ⟩_v

where ρ(θ) is the correlation between the two tuning curves at angular offset θ, and the average is
over the distribution of angular speeds v. Fitting the mean of that distribution to the observed
C(τ) recovers the mean angular speed.

Following the paper, the speed distribution is taken to be **χ² with 3 degrees of freedom**, scaled
to the mean being fitted — a skewed, strictly positive family. The paper reports the result is not
sensitive to that choice among skewed positive distributions, and `SPEED_MODELS` here makes the
alternatives testable rather than assumed.

Two properties make this checkable rather than merely plausible, and both are exercised in
`experiments/angular1.py`:

- during **wake** the measured head angle exists, so the estimate can be compared against the
  angular speed actually swept;
- on **synthetic** spikes generated from a known velocity trace, the estimator must return the
  velocity that generated them.
"""

import dataclasses

import numpy as np
from beartype import beartype
from jaxtyping import Float, jaxtyped
from scipy import optimize, signal, stats

#: Distribution families for the angular-speed prior, all strictly positive and skewed. The paper
#: uses chi2 with 3 d.o.f.; the others exist so "the result is insensitive to this" can be tested.
SPEED_MODELS: tuple[str, ...] = ("chi2_3", "chi2_1", "exponential", "rayleigh")


@jaxtyped(typechecker=beartype)
def tuning_curves(
    *,
    spike_times: dict[int, Float[np.ndarray, " _"]],
    angles: Float[np.ndarray, " time"],
    times: Float[np.ndarray, " time"],
    n_bins: int = 60,
    min_occupancy_s: float = 0.5,
) -> tuple[Float[np.ndarray, "unit bin"], Float[np.ndarray, " bin"]]:
    """Firing rate of each unit as a function of head direction, in Hz.

    Bins with less occupancy than `min_occupancy_s` become NaN rather than a rate divided by almost
    nothing — an unvisited direction should read as unknown, not as a spike.
    """
    edges = np.linspace(0.0, 2.0 * np.pi, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    dt = float(np.median(np.diff(times)))
    occupancy = np.histogram(angles, bins=edges)[0] * dt

    curves = np.full((len(spike_times), n_bins), np.nan)
    for row, unit in enumerate(sorted(spike_times)):
        spikes = spike_times[unit]
        inside = spikes[(spikes >= times[0]) & (spikes <= times[-1])]
        if not len(inside):
            continue
        at = np.interp(inside, times, np.unwrap(angles)) % (2.0 * np.pi)
        counts = np.histogram(at, bins=edges)[0]
        usable = occupancy >= min_occupancy_s
        curves[row, usable] = counts[usable] / occupancy[usable]
    return curves, centres


@jaxtyped(typechecker=beartype)
def angular_correlation(
    *, curve_a: Float[np.ndarray, " bin"], curve_b: Float[np.ndarray, " bin"]
) -> Float[np.ndarray, " bin"]:
    """Correlation between two tuning curves as a function of angular offset θ, ρ(θ).

    ρ(θ) is the Pearson correlation between curve A and curve B rotated by θ, evaluated at every bin
    offset. It peaks at the pair's preferred-direction difference, and its width is set by how broad
    the tuning is — the two features that let a *temporal* correlation be read as an angle swept.
    """
    a, b = np.asarray(curve_a, float), np.asarray(curve_b, float)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3:
        return np.full(len(a), np.nan)
    a = np.where(valid, a, np.nanmean(a[valid]))
    b = np.where(valid, b, np.nanmean(b[valid]))
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a**2).sum() * (b**2).sum())
    if denom == 0:
        return np.full(len(a), np.nan)
    # roll by -shift so index `shift` means "B evaluated at phi + shift", i.e. B rotated forward.
    # Rolling the other way returns the mirror image, which silently fits the wrong direction of
    # travel and biases every speed estimate.
    return np.array([float((a * np.roll(b, -shift)).sum() / denom) for shift in range(len(a))])


@jaxtyped(typechecker=beartype)
def highpass_rate(
    *, rate: Float[np.ndarray, " time"], dt: float, cutoff_s: float
) -> Float[np.ndarray, " time"]:
    """Remove variation slower than `cutoff_s` by subtracting a running mean.

    Firing rates share slow fluctuations that have nothing to do with head turning — brain state,
    arousal, drift in the recording. Those produce a broad component in the cross-correlogram that
    persists over many seconds, and the model has no way to tell it from a very slow sweep: it fits
    the broad component by driving the speed towards zero. On real sessions this pinned the estimate
    at the bottom of the search grid regardless of how fast the animal was actually turning.

    Head turns live on a sub-second timescale, so removing everything slower costs no signal and
    removes the confound.
    """
    width = max(1, int(np.rint(cutoff_s / dt)))
    if width >= len(rate):
        return rate - rate.mean()
    kernel = np.ones(width) / width
    return rate - np.convolve(rate, kernel, mode="same")


@jaxtyped(typechecker=beartype)
def rate_cross_correlation(
    *,
    rate_a: Float[np.ndarray, " time"],
    rate_b: Float[np.ndarray, " time"],
    max_lag_bins: int,
) -> Float[np.ndarray, " lag"]:
    """Pearson correlation between two binned rate traces at each lag from -max to +max.

    A correlation rather than a spike count, because that is what the model predicts: rho(theta) is
    a correlation, so the observed quantity has to be one too for the comparison to mean anything.

    Computed by FFT. A direct loop over lags is O(N x lags), and with half a million rate bins and a
    couple of hundred lags that is minutes per cell pair - enough to make the sweep impractical.
    """
    a = np.asarray(rate_a, float) - np.mean(rate_a)
    b = np.asarray(rate_b, float) - np.mean(rate_b)
    norm = np.sqrt((a**2).sum() * (b**2).sum())
    if norm == 0 or max_lag_bins >= len(a):
        return np.full(2 * max_lag_bins + 1, np.nan)

    full = signal.correlate(a, b, mode="full", method="fft")
    centre = len(a) - 1
    return full[centre - max_lag_bins : centre + max_lag_bins + 1] / norm


def speed_quadrature(
    *, mean_speed: float, model: str = "chi2_3", n_points: int = 128
) -> tuple[np.ndarray, np.ndarray]:
    """Quadrature nodes and weights for the angular-speed distribution with the given mean.

    The distribution shape is fixed by `model` and only its scale is fitted, which is what makes the
    fit one-dimensional. Nodes are placed at equal probability so the average is a plain mean over
    nodes — no weighting error to get wrong.
    """
    families = {
        "chi2_3": stats.chi2(df=3), "chi2_1": stats.chi2(df=1),
        "exponential": stats.expon(), "rayleigh": stats.rayleigh(),
    }
    if model not in families:
        raise ValueError(f"Unknown speed model {model!r}; expected one of {SPEED_MODELS}.")
    dist = families[model]
    probs = (np.arange(n_points) + 0.5) / n_points
    nodes = dist.ppf(probs)
    return nodes / dist.mean() * mean_speed, np.full(n_points, 1.0 / n_points)


@jaxtyped(typechecker=beartype)
def predicted_correlation(
    *,
    rho: Float[np.ndarray, " bin"],
    lags_s: Float[np.ndarray, " lag"],
    mean_speed: float,
    model: str = "chi2_3",
    n_points: int = 128,
) -> Float[np.ndarray, " lag"]:
    """The cross-correlation this pair should show if the head swept at `mean_speed` on average.

    For each speed in the quadrature, a lag τ corresponds to an angular offset v·τ; ρ is looked up
    there (periodically, since it lives on the circle) and averaged over speeds. Signed lags carry
    the direction of travel, which is why the asymmetry of ρ survives the averaging.
    """
    nodes, weights = speed_quadrature(mean_speed=mean_speed, model=model, n_points=n_points)
    n_bins = len(rho)
    offsets = np.outer(nodes, lags_s)  # (speed, lag) angular offset in radians
    index = np.rint(offsets / (2.0 * np.pi) * n_bins).astype(int) % n_bins
    return (rho[index] * weights[:, None]).sum(axis=0)


@dataclasses.dataclass(frozen=True)
class SpeedFit:
    """Result of fitting one pair's cross-correlogram."""

    mean_speed: float
    cost: float
    n_lags: int


@jaxtyped(typechecker=beartype)
def shape_cost(
    *,
    predicted: Float[np.ndarray, " lag"],
    observed: Float[np.ndarray, " lag"],
) -> float:
    """Residual after allowing the prediction any amplitude and offset — a comparison of *shape*.

    The model predicts how correlation falls away with lag, not how large it is. Its absolute size
    is set by firing statistics: binned spike counts are Poisson, and that noise attenuates a
    measured correlation towards zero regardless of how fast the head moved. Comparing raw
    amplitudes therefore rewards implausibly high speeds, whose prediction is flat and small, purely
    because the data are noisy — a failure the synthetic test catches immediately.

    Scaling and offset are solved in closed form, so the fit stays one-dimensional in the speed.
    """
    pred = np.asarray(predicted, float)
    obs = np.asarray(observed, float)
    if np.ptp(pred) < 1e-12:
        # A flat prediction can only match a flat observation; give it the variance it fails to
        # explain rather than a free pass through the intercept.
        return float(np.sum((obs - obs.mean()) ** 2))
    design = np.vstack([pred, np.ones_like(pred)]).T
    coeffs, *_ = np.linalg.lstsq(design, obs, rcond=None)
    return float(np.sum((design @ coeffs - obs) ** 2))


@jaxtyped(typechecker=beartype)
def fit_mean_speed(
    *,
    rho: Float[np.ndarray, " bin"],
    observed: Float[np.ndarray, " lag"],
    lags_s: Float[np.ndarray, " lag"],
    model: str = "chi2_3",
    coarse_range: tuple[float, float] = (0.05, 30.0),
    coarse_points: int = 80,
) -> SpeedFit:
    """Mean angular speed whose predicted cross-correlogram best matches the observed one.

    A coarse sweep over a broad speed range first, then a bounded refinement seeded from its
    minimum, as the paper describes. The coarse pass is not an optimisation detail: the cost surface
    has local minima wherever a wrong speed happens to align rho's peak with the observed one, and a
    derivative-based search from an arbitrary start finds them.
    """
    finite = np.isfinite(observed) & np.isfinite(lags_s)
    if finite.sum() < 5 or not np.isfinite(rho).all():
        return SpeedFit(mean_speed=float("nan"), cost=float("nan"), n_lags=int(finite.sum()))

    obs, lags = observed[finite], lags_s[finite]

    def cost(speed: float) -> float:
        if speed <= 0:
            return np.inf
        pred = predicted_correlation(rho=rho, lags_s=lags, mean_speed=float(speed), model=model)
        return shape_cost(predicted=pred, observed=obs)

    grid = np.geomspace(*coarse_range, coarse_points)
    costs = np.array([cost(v) for v in grid])
    best_index = int(np.argmin(costs))
    seed = float(grid[best_index])

    lo = float(grid[max(best_index - 1, 0)])
    hi = float(grid[min(best_index + 1, len(grid) - 1)])
    result = optimize.minimize_scalar(cost, bounds=(lo, hi), method="bounded")
    best = float(result.x) if result.success and result.fun <= costs[best_index] else seed
    return SpeedFit(mean_speed=best, cost=cost(best), n_lags=int(finite.sum()))


#: Angle the head should sweep across the full lag window for the fit to be well constrained.
#: Below about 1 rad the correlogram is too flat to distinguish speeds and the estimate runs high;
#: above about 5 rad it wraps several times and does the same. Measured on synthetic data whose
#: true speed is known — see `tests/test_angular_speed.py`.
TARGET_SWEEP_RAD: float = 2.5
USABLE_SWEEP_RAD: tuple[float, float] = (1.0, 6.0)


@jaxtyped(typechecker=beartype)
def fit_with_adaptive_window(
    *,
    rho: Float[np.ndarray, " bin"],
    rate_a: Float[np.ndarray, " time"],
    rate_b: Float[np.ndarray, " time"],
    dt: float,
    initial_lag_s: float = 1.0,
    max_lag_s: float = 10.0,
    rounds: int = 2,
    model: str = "chi2_3",
) -> SpeedFit:
    """Fit the mean speed, choosing the lag window to suit the speed being fitted.

    The lag window is not a free stylistic choice: it must be long enough for the head to sweep an
    appreciable angle and short enough not to wrap repeatedly. On synthetic data with a known speed,
    a window covering under ~1 rad of sweep overestimates badly, and one covering more than ~6 rad
    does too. Since the right window depends on the answer, it is set iteratively - fit once at a
    default window, then re-fit with the window the first estimate implies.

    The full correlogram is computed once and each round slices it, so iterating costs nothing
    beyond the fitting itself.
    """
    widest = int(np.rint(max_lag_s / dt))
    if widest < 3 or widest >= len(rate_a) // 2:
        return SpeedFit(mean_speed=float("nan"), cost=float("nan"), n_lags=0)
    full = rate_cross_correlation(rate_a=rate_a, rate_b=rate_b, max_lag_bins=widest)
    if not np.isfinite(full).any():
        return SpeedFit(mean_speed=float("nan"), cost=float("nan"), n_lags=0)

    lag_s = initial_lag_s
    fit = SpeedFit(mean_speed=float("nan"), cost=float("nan"), n_lags=0)
    for _ in range(max(1, rounds)):
        half = int(np.rint(lag_s / dt))
        if half < 3 or half > widest:
            break
        window = slice(widest - half, widest + half + 1)
        lags_s = np.arange(-half, half + 1, dtype=float) * dt
        fit = fit_mean_speed(rho=rho, observed=full[window], lags_s=lags_s, model=model)
        if not np.isfinite(fit.mean_speed) or fit.mean_speed <= 0:
            break
        lag_s = float(np.clip(TARGET_SWEEP_RAD / fit.mean_speed, 0.1, max_lag_s))
    return fit


@jaxtyped(typechecker=beartype)
def fit_population_speed(
    *,
    rhos: Float[np.ndarray, "pair bin"],
    correlograms: Float[np.ndarray, "pair lag"],
    dt: float,
    initial_lag_s: float = 1.0,
    max_lag_s: float = 10.0,
    rounds: int = 2,
    model: str = "chi2_3",
    coarse_range: tuple[float, float] = (0.05, 30.0),
    coarse_points: int = 80,
) -> SpeedFit:
    """One mean speed explaining *every* pair's correlogram at once.

    All the pairs in a session watched the same head turn, so there is one speed to find, not one per
    pair. Fitting each pair alone and taking a median throws that away, and on real data it fails:
    single-pair fits scatter by several times their own median, the failures are one-sided (a pair
    with no usable signal runs high rather than scattering symmetrically), and the median inherits
    that tail instead of averaging it out.

    Fitting jointly removes the failure mode rather than averaging over it. A speed that is wrong for
    the session must pay a cost in every pair simultaneously, so a handful of uninformative pairs can
    no longer set the answer; they contribute a nearly flat cost curve and are outvoted.

    Each pair's residual is normalised by its own variance before summing, so a pair with a strong
    correlation does not dominate merely by being large — the pairs vote, weighted by how well the
    shared speed explains them rather than by their amplitude.

    `correlograms` holds each pair's *full* correlation out to `max_lag_s`, centred on zero lag, as
    `rate_cross_correlation` returns it. The lag window is narrowed iteratively exactly as in
    `fit_with_adaptive_window`, by slicing.
    """
    widest = (correlograms.shape[1] - 1) // 2
    if len(rhos) == 0 or widest < 3:
        return SpeedFit(mean_speed=float("nan"), cost=float("nan"), n_lags=0)

    usable = np.isfinite(correlograms).all(axis=1) & np.isfinite(rhos).all(axis=1)
    if not usable.any():
        return SpeedFit(mean_speed=float("nan"), cost=float("nan"), n_lags=0)
    rhos, correlograms = rhos[usable], correlograms[usable]

    lag_s = initial_lag_s
    fit = SpeedFit(mean_speed=float("nan"), cost=float("nan"), n_lags=0)
    for _ in range(max(1, rounds)):
        half = int(np.rint(lag_s / dt))
        if half < 3 or half > widest:
            break
        observed = correlograms[:, widest - half : widest + half + 1]
        lags_s = np.arange(-half, half + 1, dtype=float) * dt
        # Normalising by each pair's own variance makes the summands comparable; a pair with no
        # variance to explain is dropped rather than contributing a division by zero.
        scale = observed.var(axis=1)
        keep = scale > 0
        if not keep.any():
            break

        def cost(speed: float, *, obs=observed[keep], sc=scale[keep], rs=rhos[keep], lg=lags_s):
            if speed <= 0:
                return np.inf
            total = 0.0
            for rho, row, var in zip(rs, obs, sc, strict=True):
                pred = predicted_correlation(
                    rho=rho, lags_s=lg, mean_speed=float(speed), model=model
                )
                total += shape_cost(predicted=pred, observed=row) / (var * len(lg))
            return total

        grid = np.geomspace(*coarse_range, coarse_points)
        costs = np.array([cost(v) for v in grid])
        best_index = int(np.argmin(costs))
        lo = float(grid[max(best_index - 1, 0)])
        hi = float(grid[min(best_index + 1, len(grid) - 1)])
        refined = optimize.minimize_scalar(cost, bounds=(lo, hi), method="bounded")
        best = (
            float(refined.x)
            if refined.success and refined.fun <= costs[best_index]
            else float(grid[best_index])
        )
        fit = SpeedFit(mean_speed=best, cost=float(cost(best)), n_lags=int(keep.sum()))
        if not np.isfinite(best) or best <= 0:
            break
        lag_s = float(np.clip(TARGET_SWEEP_RAD / best, 0.1, max_lag_s))
    return fit
