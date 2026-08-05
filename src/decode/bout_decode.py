"""Fit one decoder per bout, rather than one decoder per state.

The rest of the pipeline pools every epoch of a state, embeds the concatenation and fits a single
ring to it. On wake that is measurably wrong: decoding all of DANDI's "Awake" intervals together
gives a held-out circular RMSE of 0.97-1.48 rad against measured head direction, where chance is
pi/sqrt(3) = 1.81 — while the same code on a single contiguous wake bout gives 0.40-0.43, matching
the published 0.36-0.41. Pooling heterogeneous epochs, not the decoder, was the fault.

So the unit here is the bout: each is embedded and fitted on its own, and split internally into
train and test so its decode has a quality number attached that rests on held-out points.

Angular speed needs no alignment — it is invariant to the arbitrary shift and flip the ring
parameterisation carries — which is what makes the same measurement possible in REM, where there is
no measured angle to align to.
"""

import dataclasses
import itertools

import numpy as np
import pynapple as nap
from beartype import beartype
from jaxtyping import Float, jaxtyped

import spud.manifold_fit_and_decode_fns as mff
from core.config import DiffusionConfig
from decode import head_direction, manifold, rates


@dataclasses.dataclass(frozen=True)
class BoutDecode:
    """One bout's decode, its measured counterpart, and how well the two agree."""

    #: Ring angle at every bin of the bout, in [0, 2pi). Carries an arbitrary shift and flip.
    decoded: np.ndarray
    #: Measured head direction binned onto the same grid; NaN where the bout has no tracking.
    measured: np.ndarray
    #: Bin centres, in seconds.
    times: np.ndarray
    #: Held-out circular RMSE against the measured angle, mean and s.d. over splits.
    rmse: float
    rmse_sd: float
    n_bins: int
    duration_s: float


@jaxtyped(typechecker=beartype)
def binned_measured_angle(
    *, hd: nap.Tsd, start: float, end: float, dt: float
) -> Float[np.ndarray, " time"]:
    """Circular mean of the measured head direction in each rate bin of one bout.

    Bins with no tracking samples return NaN rather than an interpolated value: a fabricated angle
    would be scored as if it were ground truth.
    """
    edges = np.arange(float(start), float(end), dt)
    restricted = hd.restrict(nap.IntervalSet(start=[start], end=[end]))
    times = np.asarray(restricted.index, dtype=float)
    angles = np.asarray(restricted.values, dtype=float)

    idx = np.searchsorted(times, edges)
    out = np.full(len(edges) - 1, np.nan)
    for i, (lo, hi) in enumerate(itertools.pairwise(idx)):
        if hi > lo:
            segment = angles[lo:hi]
            out[i] = np.arctan2(np.sin(segment).mean(), np.cos(segment).mean()) % (2.0 * np.pi)
    return out


def decode_bout(
    *,
    units: nap.TsGroup,
    start: float,
    end: float,
    cfg: DiffusionConfig,
    measured: Float[np.ndarray, " time"] | None = None,
    train_frac: float = 0.8,
    n_splits: int = 5,
    seed: int = 0,
) -> BoutDecode | None:
    """Embed one bout, fit its own ring, and decode every bin of it.

    `n_splits` independent train/test splits give the held-out RMSE; the returned trace comes from
    the first split's fit, so the decode being measured and the decode being scored are the same
    object. Returns None when the bout is too short to embed.

    Pass `measured=None` for a state with no tracking: the trace still decodes and the RMSE is NaN.
    """
    epoch = nap.IntervalSet(start=[start], end=[end])
    rate_matrix = rates.rate_matrix(units=units, epochs=epoch, dt=cfg.dt, sigma=cfg.sigma)
    n = min(len(rate_matrix), cfg.n_samples)
    if n < 500:
        return None
    rate_matrix = rate_matrix[:n]
    times = np.arange(float(start), float(end), cfg.dt)[: n + 1][:-1] + cfg.dt / 2.0

    if measured is None:
        measured_binned = np.full(n, np.nan)
    else:
        measured_binned = np.asarray(measured, dtype=float)[:n]

    embed = manifold.isomap_embed(rates=rate_matrix, cfg=cfg)
    rng = np.random.default_rng(seed)

    trace, scores = None, []
    for split in range(max(1, n_splits)):
        # spud's k-means init draws from the global NumPy RNG, so seeding it makes each fit
        # reproducible; the split itself uses its own generator.
        np.random.seed(seed + split)
        train = rng.choice(len(embed), int(train_frac * len(embed)), replace=False)
        test = np.setdiff1d(np.arange(len(embed)), train)
        fit = manifold.fit_best_ring(points=embed[train], cfg=cfg)

        if np.isfinite(measured_binned[test]).sum() >= 100:
            _, mse = mff.decode_from_passed_fit(
                embed[test], fit["tt"][:-1], fit["curve"][:-1], measured_binned[test]
            )
            scores.append(np.sqrt(mse))
        if trace is None:
            decoded, _ = mff.decode_from_passed_fit(
                embed, fit["tt"][:-1], fit["curve"][:-1], np.zeros(len(embed))
            )
            trace = np.asarray(decoded, dtype=float)

    return BoutDecode(
        decoded=trace,
        measured=measured_binned,
        times=times[:n],
        rmse=float(np.mean(scores)) if scores else float("nan"),
        rmse_sd=float(np.std(scores)) if scores else float("nan"),
        n_bins=n,
        duration_s=float(end) - float(start),
    )


def speed_summary(*, angles: Float[np.ndarray, " time"], times: Float[np.ndarray, " time"],
                  net_tau_s: float, smooth_s: float) -> dict[str, float]:
    """Path-length and net angular speed of one trace, the two measures used throughout.

    NaN gaps are dropped before differencing rather than interpolated, so an untracked stretch
    contributes nothing instead of contributing a fabricated step.
    """
    good = np.isfinite(angles)
    if good.sum() < 50:
        return {"path": float("nan"), "net": float("nan")}
    a, t = angles[good], times[good]
    speed = head_direction.angular_speed(angles=a, times=t, smooth_s=smooth_s)
    gaps = np.diff(t)
    usable = speed[gaps < 5.0 * np.median(gaps)]
    return {
        "path": float(np.mean(usable)) if len(usable) else float("nan"),
        "net": head_direction.net_speed(angles=a, times=t, tau_s=net_tau_s),
    }
