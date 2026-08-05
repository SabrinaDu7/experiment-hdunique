"""Does the tuning measurement measure tuning, and does the fast null match the slow one?"""

import numpy as np
import pytest

from metrics.hd_tuning import (
    _grid,
    _gridded_curve,
    cell_tuning,
    mean_vector,
    preferred_direction_consistency,
)

DT = 1.0 / 39.0


def _session(*, kappa: float, peak_hz: float, duration: float = 900.0, seed: int = 0):
    """A head-direction trace covering the circle, and Poisson spikes from a von Mises cell."""
    rng = np.random.default_rng(seed)
    times = np.arange(0.0, duration, DT)
    angles = np.cumsum(rng.normal(0, 0.15, len(times))) % (2.0 * np.pi)
    rate = peak_hz * np.exp(kappa * (np.cos(angles - 1.0) - 1.0)) if kappa > 0 else \
        np.full(len(times), peak_hz)
    counts = rng.poisson(rate * DT)
    spikes = np.repeat(times, counts) + rng.uniform(0, DT, counts.sum())
    return times, angles, np.sort(spikes)


def test_a_tuned_cell_scores_high_and_an_untuned_one_does_not() -> None:
    """The basic requirement, and the one every threshold in tuning1 rests on."""
    times, angles, spikes = _session(kappa=4.0, peak_hz=30.0)
    tuned = cell_tuning(spikes=spikes, angles=angles, times=times, n_shuffles=100)

    times, angles, spikes = _session(kappa=0.0, peak_hz=30.0, seed=1)
    flat = cell_tuning(spikes=spikes, angles=angles, times=times, n_shuffles=100)

    assert tuned.mvl > 4 * flat.mvl, f"tuned {tuned.mvl:.3f} vs untuned {flat.mvl:.3f}"
    assert tuned.shuffle_p <= 0.05 < flat.shuffle_p


def test_preferred_direction_is_recovered() -> None:
    """A preferred direction that is not where the cell actually fires makes every downstream
    consistency measure meaningless."""
    times, angles, spikes = _session(kappa=4.0, peak_hz=30.0)
    got = cell_tuning(spikes=spikes, angles=angles, times=times, n_shuffles=20).preferred
    assert np.isclose((got - 1.0 + np.pi) % (2 * np.pi) - np.pi, 0.0, atol=0.3)


def test_an_untuned_cell_is_not_rescued_by_a_high_firing_rate() -> None:
    """The shuffle null exists for this: significance must not be buyable with spike count."""
    times, angles, spikes = _session(kappa=0.0, peak_hz=100.0, seed=2)
    assert cell_tuning(spikes=spikes, angles=angles, times=times, n_shuffles=100).shuffle_p > 0.05


def test_gridded_curve_matches_the_interpolated_one() -> None:
    """The fast null bins spikes onto the head-direction grid instead of interpolating the angle at
    each spike. That is a hundred times faster, and only legitimate if it gives the same curve."""
    from metrics.angular_speed import tuning_curves

    times, angles, spikes = _session(kappa=4.0, peak_hz=30.0)
    grid = _grid(spikes=spikes, angles=angles, times=times, n_bins=60)
    fast = _gridded_curve(grid=grid, counts=grid.counts, min_occupancy_s=0.5)
    slow = tuning_curves(spike_times={0: spikes}, angles=angles, times=times, n_bins=60)[0][0]

    shared = np.isfinite(fast) & np.isfinite(slow)
    assert shared.sum() > 40
    assert np.corrcoef(fast[shared], slow[shared])[0, 1] > 0.99
    centres = np.linspace(0, 2 * np.pi, 60, endpoint=False) + np.pi / 60
    assert np.isclose(
        abs(mean_vector(curve=fast, centres=centres)),
        abs(mean_vector(curve=slow, centres=centres)), rtol=0.1
    )


@pytest.mark.parametrize(
    ("angles", "expected"),
    [(np.array([1.0, 1.05, 0.95]), 1.0), (np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2]), 0.0)],
)
def test_preferred_consistency_spans_agreement_and_disagreement(angles, expected) -> None:
    """A cell can be strongly tuned in every bout and point somewhere different in each; that cell
    must not pass as reliable."""
    assert np.isclose(preferred_direction_consistency(preferred=angles), expected, atol=0.05)
