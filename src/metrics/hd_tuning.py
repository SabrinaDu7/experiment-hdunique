"""How strongly, and how reliably, does a cell encode head direction?

The decode assumes the ADn population carries a head-direction ring. On the two sessions examined so
far that assumption holds in some wake bouts and fails in others — among bouts where the animal
samples the whole circle, the median tuning strength of its ADn cells predicts that bout's decode
error at rho = -0.93. So tuning is not a descriptive statistic here; it is the precondition the
decode rests on, and it varies within a session.

Three things are measured per cell, and they are not the same thing:

- **strength** — how sharply the firing rate depends on direction *now* (mean vector length).
- **reliability** — whether the same tuning is there in both halves of the data, which strength alone
  cannot tell you: a cell that fires in one burst while the animal happened to face one way looks
  sharply tuned and is not.
- **significance** — whether the strength beats what the same spikes produce with their relationship
  to head direction destroyed, which controls for firing rate and for uneven occupancy.
"""

import dataclasses

import numpy as np
from beartype import beartype
from jaxtyping import Float, jaxtyped

from metrics.angular_speed import tuning_curves


@dataclasses.dataclass(frozen=True)
class CellTuning:
    """One cell's head-direction tuning within one stretch of data."""

    mvl: float
    preferred: float
    peak_hz: float
    mean_hz: float
    #: Correlation between tuning curves built from the two halves of the data.
    reliability: float
    #: Fraction of shuffles whose mean vector length reaches the observed one.
    shuffle_p: float


@jaxtyped(typechecker=beartype)
def mean_vector(*, curve: Float[np.ndarray, " bin"], centres: Float[np.ndarray, " bin"]) -> complex:
    """Occupancy-normalised mean resultant vector of one tuning curve.

    Its length is the standard measure of head-direction tuning strength (0 = flat, 1 = all firing
    at one direction) and its angle is the preferred direction. Unvisited bins are dropped rather
    than treated as zero firing, which would fake sharp tuning for a cell that simply was not
    observed facing that way.
    """
    usable = np.isfinite(curve)
    if usable.sum() < 3 or np.nansum(curve[usable]) <= 0:
        return complex(np.nan)
    weights = curve[usable] / np.sum(curve[usable])
    return complex(np.sum(weights * np.exp(1j * centres[usable])))


@jaxtyped(typechecker=beartype)
def _curve_for(
    *,
    spikes: Float[np.ndarray, " _"],
    angles: Float[np.ndarray, " time"],
    times: Float[np.ndarray, " time"],
    n_bins: int,
    min_occupancy_s: float,
) -> Float[np.ndarray, " bin"]:
    """One cell's tuning curve, via the shared implementation."""
    curves, _ = tuning_curves(
        spike_times={0: spikes}, angles=angles, times=times,
        n_bins=n_bins, min_occupancy_s=min_occupancy_s,
    )
    return curves[0]


@dataclasses.dataclass(frozen=True)
class _Gridded:
    """Spikes and head direction on the same fixed grid, so a shuffle is a roll of an array.

    The shuffle null needs a tuning curve per surrogate, and computing each by interpolating the
    angle at every shifted spike time costs an O(n log n) search over the whole trace — 200 of those
    per cell put the sweep in the hours. Counting spikes onto the head-direction sample grid once
    turns each surrogate into `np.roll` plus a `bincount`, which is a hundred times faster and is
    the same null: a circular shift of the spike train against the head-direction trace.
    """

    bin_index: np.ndarray
    counts: np.ndarray
    occupancy: np.ndarray
    dt: float


def _grid(
    *, spikes: np.ndarray, angles: np.ndarray, times: np.ndarray, n_bins: int
) -> _Gridded:
    """Put one cell's spikes and the head direction on a common grid."""
    dt = float(np.median(np.diff(times)))
    bin_index = np.minimum((angles / (2.0 * np.pi) * n_bins).astype(int), n_bins - 1)
    edges = np.concatenate([times - dt / 2.0, [times[-1] + dt / 2.0]])
    counts = np.histogram(spikes, bins=edges)[0].astype(float)
    occupancy = np.bincount(bin_index, minlength=n_bins) * dt
    return _Gridded(bin_index=bin_index, counts=counts, occupancy=occupancy, dt=dt)


def _gridded_curve(*, grid: _Gridded, counts: np.ndarray, min_occupancy_s: float) -> np.ndarray:
    """Tuning curve from already-gridded counts."""
    total = np.bincount(grid.bin_index, weights=counts, minlength=len(grid.occupancy))
    usable = grid.occupancy >= min_occupancy_s
    curve = np.full(len(grid.occupancy), np.nan)
    curve[usable] = total[usable] / grid.occupancy[usable]
    return curve


def cell_tuning(
    *,
    spikes: Float[np.ndarray, " _"],
    angles: Float[np.ndarray, " time"],
    times: Float[np.ndarray, " time"],
    n_bins: int = 60,
    min_occupancy_s: float = 0.5,
    n_shuffles: int = 200,
    seed: int = 0,
) -> CellTuning:
    """Strength, preferred direction, split-half reliability and a shuffle p for one cell.

    The null circularly shifts the spike train against the head-direction trace by a random offset.
    That preserves the cell's firing rate and its burst structure and the animal's occupancy, and
    destroys only the alignment between them — so a significant result cannot be produced by a cell
    simply firing a lot, or by the animal spending most of its time facing one way.
    """
    curve = _curve_for(
        spikes=spikes, angles=angles, times=times,
        n_bins=n_bins, min_occupancy_s=min_occupancy_s,
    )
    centres = np.linspace(0.0, 2.0 * np.pi, n_bins, endpoint=False) + np.pi / n_bins
    vector = mean_vector(curve=curve, centres=centres)
    if not np.isfinite(vector):
        return CellTuning(*(float("nan"),) * 5, shuffle_p=float("nan"))

    half = len(times) // 2
    halves = []
    for lo, hi in ((0, half), (half, len(times))):
        halves.append(
            _curve_for(
                spikes=spikes, angles=angles[lo:hi], times=times[lo:hi],
                n_bins=n_bins, min_occupancy_s=min_occupancy_s / 2.0,
            )
        )
    shared = np.isfinite(halves[0]) & np.isfinite(halves[1])
    reliability = (
        float(np.corrcoef(halves[0][shared], halves[1][shared])[0, 1])
        if shared.sum() >= 5 and np.ptp(halves[0][shared]) > 0 and np.ptp(halves[1][shared]) > 0
        else float("nan")
    )

    rng = np.random.default_rng(seed)
    observed = abs(vector)

    # The null is computed on the grid, and so is the statistic it is compared against — otherwise
    # the observed value and its null would come from two slightly different estimators and the p
    # would carry that difference rather than the effect.
    grid = _grid(spikes=spikes, angles=angles, times=times, n_bins=n_bins)
    gridded_observed = abs(mean_vector(
        curve=_gridded_curve(grid=grid, counts=grid.counts, min_occupancy_s=min_occupancy_s),
        centres=centres,
    ))
    beaten = 0
    if np.isfinite(gridded_observed):
        shifts = rng.integers(len(grid.counts) // 20, len(grid.counts) - len(grid.counts) // 20,
                              size=max(1, n_shuffles))
        for shift in shifts:
            null = mean_vector(
                curve=_gridded_curve(
                    grid=grid, counts=np.roll(grid.counts, int(shift)),
                    min_occupancy_s=min_occupancy_s,
                ),
                centres=centres,
            )
            if np.isfinite(null) and abs(null) >= gridded_observed:
                beaten += 1
    else:
        beaten = max(1, n_shuffles)  # no usable gridded statistic: report no significance

    duration = float(times[-1] - times[0]) or float("nan")
    return CellTuning(
        mvl=observed,
        preferred=float(np.angle(vector) % (2.0 * np.pi)),
        peak_hz=float(np.nanmax(curve)),
        mean_hz=float(len(spikes[(spikes >= times[0]) & (spikes <= times[-1])]) / duration),
        reliability=reliability,
        shuffle_p=(beaten + 1) / (max(1, n_shuffles) + 1),
    )


def preferred_direction_consistency(*, preferred: Float[np.ndarray, " bout"]) -> float:
    """How closely a cell's preferred direction agrees across the bouts it was measured in.

    1 means identical every time, 0 means unrelated. A cell can clear the strength and significance
    bars in every bout and still point somewhere different in each — that cell is not carrying a
    stable direction and should not be trusted to carry one into sleep, where nothing can check it.
    """
    usable = preferred[np.isfinite(preferred)]
    if len(usable) < 2:
        return float("nan")
    return float(np.abs(np.mean(np.exp(1j * usable))))
