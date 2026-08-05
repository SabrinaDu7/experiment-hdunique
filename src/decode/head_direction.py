"""Head direction and angular speed, measured from the tracking LEDs.

The decoded angle used elsewhere in this repo is *unsupervised* — inferred from population activity
with no reference to where the animal was actually looking. This module provides the measured
quantity instead: the head direction implied by the two tracking LEDs, and the angular speed derived
from it.

That measured signal only exists during wake, when the animal is being tracked. Its value here is
as **ground truth**: an estimator that claims to recover angular speed from spikes alone can be
checked against it during wake, and only then trusted during REM where no such check is possible.
"""

import numpy as np
import pynapple as nap
from beartype import beartype
from jaxtyping import Float, jaxtyped

#: LED pairs to try, in order. The DANDI conversion exposes them under two naming schemes.
LED_KEYS: tuple[tuple[str, str], ...] = (
    ("RedLED", "BlueLED"),
    ("SubjectPosition/RedLED", "SubjectPosition/BlueLED"),
)


def head_direction(*, data: nap.NWBFile) -> nap.Tsd:
    """Head direction in [0, 2π), from the vector joining the two tracking LEDs.

    Samples where either LED is missing are dropped rather than interpolated: a fabricated position
    would produce a fabricated angular velocity, which is the very quantity being measured.
    """
    keys = set(data.keys())
    for red_key, blue_key in LED_KEYS:
        if red_key in keys and blue_key in keys:
            red, blue = data[red_key], data[blue_key]
            break
    else:
        raise KeyError(f"No LED pair in this NWB; looked for {LED_KEYS}.")

    delta = np.asarray(red.values, dtype=float) - np.asarray(blue.values, dtype=float)
    angle = np.arctan2(delta[:, 1], delta[:, 0]) % (2.0 * np.pi)
    good = np.isfinite(angle) & np.isfinite(delta).all(axis=1)
    return nap.Tsd(t=np.asarray(red.index, dtype=float)[good], d=angle[good])


@jaxtyped(typechecker=beartype)
def angular_speed(
    *, angles: Float[np.ndarray, " time"], times: Float[np.ndarray, " time"], smooth_s: float = 0.0
) -> Float[np.ndarray, " _"]:
    """Unsigned angular speed in rad/s from a head-direction trace.

    Differences are taken the short way round, so a wrap from 2π to 0 reads as a small step rather
    than a full turn. `smooth_s` box-filters the angle before differencing; tracking jitter otherwise
    inflates the speed, and at 39 Hz that inflation is not small.
    """
    if smooth_s > 0:
        dt = float(np.median(np.diff(times)))
        width = max(1, int(np.rint(smooth_s / dt)))
        kernel = np.ones(width) / width
        unwrapped = np.unwrap(angles)
        angles = np.convolve(unwrapped, kernel, mode="same") % (2.0 * np.pi)

    delta = (np.diff(angles) + np.pi) % (2.0 * np.pi) - np.pi
    return np.abs(delta / np.diff(times))


@jaxtyped(typechecker=beartype)
def angular_velocity(
    *, angles: Float[np.ndarray, " time"], times: Float[np.ndarray, " time"], smooth_s: float = 0.0
) -> Float[np.ndarray, " _"]:
    """Signed angular velocity in rad/s — positive anticlockwise.

    `angular_speed` is the absolute value of this. The signed form exists so the derivative can be
    integrated back to the angle it came from, which is the only way to check that the wrapping,
    the sign convention and the time base are all consistent (see `integrate_velocity`).
    """
    if smooth_s > 0:
        dt = float(np.median(np.diff(times)))
        width = max(1, int(np.rint(smooth_s / dt)))
        kernel = np.ones(width) / width
        angles = np.convolve(np.unwrap(angles), kernel, mode="same") % (2.0 * np.pi)
    delta = (np.diff(angles) + np.pi) % (2.0 * np.pi) - np.pi
    return delta / np.diff(times)


@jaxtyped(typechecker=beartype)
def integrate_velocity(
    *,
    velocity: Float[np.ndarray, " step"],
    times: Float[np.ndarray, " time"],
    initial_angle: float,
) -> Float[np.ndarray, " time"]:
    """Rebuild the angle by accumulating `velocity` from `initial_angle`.

    The inverse of `angular_velocity`. Round-tripping through the pair cannot validate the magnitude
    of the velocity against anything external — the reconstruction is exact by construction — but it
    does test everything around it: that differences are taken the short way round, that the sign
    convention is consistent, and that each step is divided by the interval it actually spans rather
    than by an assumed constant.
    """
    return (initial_angle + np.concatenate([[0.0], np.cumsum(velocity * np.diff(times))])) % (
        2.0 * np.pi
    )


@jaxtyped(typechecker=beartype)
def net_speed(
    *, angles: Float[np.ndarray, " time"], times: Float[np.ndarray, " time"], tau_s: float
) -> float:
    """Mean *net* angle swept per second, measured over windows of length `tau_s`.

    Distinct from `angular_speed`, and the distinction matters. `angular_speed` is path length per
    second: it counts a head that turns left then right as having moved twice. This counts only the
    displacement between the two ends of a window, so those two turns cancel.

    The cross-correlogram estimator reads net displacement — a lag τ enters its model as a single
    angular offset v·τ — so this, not the path length, is the quantity it should be checked against.
    Path length is an upper bound on it and the gap grows with τ.
    """
    step = int(np.rint(tau_s / float(np.median(np.diff(times)))))
    if step < 1 or step >= len(angles):
        return float("nan")
    # Only pairs genuinely tau apart: a pair spanning a gap between epochs is not a real excursion.
    spanned = times[step:] - times[:-step]
    usable = np.abs(spanned - tau_s) < 0.25 * tau_s
    if usable.sum() < 10:
        return float("nan")
    delta = (angles[step:] - angles[:-step] + np.pi) % (2.0 * np.pi) - np.pi
    return float(np.mean(np.abs(delta[usable])) / tau_s)


def measured_speed_in(
    *,
    data: nap.NWBFile,
    epochs: nap.IntervalSet,
    smooth_s: float = 0.1,
    net_taus_s: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0),
) -> dict[str, float]:
    """Measured angular speed inside the given epochs, in rad/s, both ways.

    Returns the path-length speed (`mean`, `median`) and the net displacement speed at each lag in
    `net_taus_s` (`net_0.25` and so on). Both are reported because they answer different questions
    and the spike-based estimator is only comparable to the second.

    Returns NaN when the epochs contain no tracking — which is the normal case for sleep, and the
    reason the spike-based estimator exists at all.
    """
    hd = head_direction(data=data).restrict(epochs)
    times = np.asarray(hd.index, dtype=float)
    if len(times) < 3:
        out = {"mean": float("nan"), "median": float("nan"), "n_samples": 0}
        return out | {f"net_{tau:g}": float("nan") for tau in net_taus_s}

    angles = np.asarray(hd.values, dtype=float)
    speed = angular_speed(angles=angles, times=times, smooth_s=smooth_s)
    # Drop samples spanning a gap between epochs, where the "step" is not a real movement.
    gaps = np.diff(times)
    usable = speed[gaps < 5.0 * np.median(gaps)]
    out = {
        "mean": float(np.mean(usable)),
        "median": float(np.median(usable)),
        "n_samples": len(usable),
    }
    return out | {
        f"net_{tau:g}": net_speed(angles=angles, times=times, tau_s=tau) for tau in net_taus_s
    }
