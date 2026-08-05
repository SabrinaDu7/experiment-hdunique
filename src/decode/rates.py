"""Firing rates, exactly as the paper specifies: a Gaussian kernel of s.d. 100 ms summed at each
exact spike time, evaluated at 100 ms bin centres, computed **per bout from that bout's spikes
only** and concatenated afterwards.

This is the original pipeline's `get_kernel_sum` formulation, not a histogram-then-filter
approximation; the two differ slightly at bin discretisation and bout edges.
"""

import numpy as np
import pynapple as nap
from beartype import beartype
from jaxtyping import Float, jaxtyped

from spud.kernel_rates import gaussian_wind_fn, get_kernel_sum


@jaxtyped(typechecker=beartype)
def _bout_rates(
    *,
    spike_times: dict[int, Float[np.ndarray, " _"]],
    unit_ids: list[int],
    start: float,
    end: float,
    dt: float,
    sigma: float,
) -> Float[np.ndarray, "time units"]:
    """Sum-of-Gaussians rate per unit at the bin centres of one bout, using in-bout spikes only."""
    def wind_fn(mu: float, x: Float[np.ndarray, " _"]) -> Float[np.ndarray, " _"]:
        return gaussian_wind_fn(mu, sigma, x)

    bin_edges = np.arange(start, end, dt)
    bin_centres = bin_edges[:-1] + dt / 2.0

    columns = []
    for uid in unit_ids:
        spk = spike_times[uid]
        in_bout = spk[(spk >= start) & (spk < end)]
        columns.append(get_kernel_sum(list(in_bout), bin_centres, wind_fn))
    return np.array(columns).T


@jaxtyped(typechecker=beartype)
def rate_matrix(
    *, units: nap.TsGroup, epochs: nap.IntervalSet, dt: float, sigma: float
) -> Float[np.ndarray, "time units"]:
    """Rate matrix (time x cells) over the given epochs, bouts concatenated in temporal order.

    Column order is ascending unit id. Note that concatenation means the returned array is not
    contiguous in time across bout boundaries — anything computing time-lagged quantities from it
    must respect the bout structure (see `diffusion.diffusion_curve`).
    """
    unit_ids = sorted(int(i) for i in units.index)
    spike_times = {uid: np.asarray(units[uid].t, dtype=float) for uid in unit_ids}
    blocks = [
        _bout_rates(
            spike_times=spike_times,
            unit_ids=unit_ids,
            start=float(start),
            end=float(end),
            dt=dt,
            sigma=sigma,
        )
        for start, end in zip(np.asarray(epochs.start), np.asarray(epochs.end))
    ]
    return np.vstack(blocks)


@jaxtyped(typechecker=beartype)
def bout_lengths(*, epochs: nap.IntervalSet, dt: float) -> list[int]:
    """Number of rate bins each bout contributes, in the same order `rate_matrix` concatenates
    them. Needed to split the concatenated decoded angle back into per-bout segments."""
    return [
        len(np.arange(float(start), float(end), dt)) - 1
        for start, end in zip(np.asarray(epochs.start), np.asarray(epochs.end))
    ]
