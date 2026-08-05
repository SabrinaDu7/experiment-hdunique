"""Can the estimator recover an angular speed it was given?

Everything downstream rests on the claim that a temporal cross-correlation between two HD cells
encodes the angle the head swept. The only way to know the implementation of that claim works is to
generate spikes from a head-direction trace whose speed we chose, and check the number comes back.
"""

import numpy as np
import pytest

from metrics.angular_speed import (
    angular_correlation,
    fit_population_speed,
    fit_with_adaptive_window,
    predicted_correlation,
    rate_cross_correlation,
    speed_quadrature,
    tuning_curves,
)

DT = 0.02  # s, rate-bin width
DURATION = 1200.0  # s of synthetic recording
MAX_LAG_S = 10.0  # widest correlogram lag, as in `experiments.angular1`


def _von_mises(centres: np.ndarray, preferred: float, kappa: float, peak: float) -> np.ndarray:
    """A tuning curve with the shape HD cells actually have."""
    return peak * np.exp(kappa * (np.cos(centres - preferred) - 1.0))


def _simulate(*, mean_speed: float, seed: int = 0, n_cells: int = 8):
    """A head-direction trace at a known mean angular speed, and Poisson spikes tuned to it."""
    rng = np.random.default_rng(seed)
    n = int(DURATION / DT)
    times = np.arange(n) * DT

    # Speed drawn from the same family the estimator assumes, direction flipping slowly so the
    # trace explores the whole circle rather than winding one way forever.
    speed = rng.chisquare(3, size=n) / 3.0 * mean_speed
    direction = np.sign(np.sin(2 * np.pi * times / 30.0))
    angles = np.cumsum(speed * direction * DT) % (2.0 * np.pi)

    preferred = np.linspace(0, 2 * np.pi, n_cells, endpoint=False)
    spikes = {}
    for i, pref in enumerate(preferred):
        rate = _von_mises(angles, pref, kappa=4.0, peak=30.0)
        counts = rng.poisson(rate * DT)
        idx = np.repeat(np.arange(n), counts)
        spikes[i] = times[idx] + rng.uniform(0, DT, len(idx))
    return times, angles, spikes


def _estimate(*, mean_speed: float, seed: int = 0) -> float:
    """Run the full estimator on synthetic data and return the median pair estimate."""
    times, angles, spikes = _simulate(mean_speed=mean_speed, seed=seed)
    curves, _ = tuning_curves(spike_times=spikes, angles=angles, times=times, n_bins=60)

    edges = np.arange(times[0], times[-1] + DT, DT)
    rates = np.array([np.histogram(spikes[u], bins=edges)[0].astype(float) for u in sorted(spikes)])

    estimates = []
    for a in range(len(curves)):
        for b in range(a + 1, len(curves)):
            rho = angular_correlation(curve_a=curves[a], curve_b=curves[b])
            fit = fit_with_adaptive_window(rho=rho, rate_a=rates[a], rate_b=rates[b], dt=DT)
            if np.isfinite(fit.mean_speed):
                estimates.append(fit.mean_speed)
    return float(np.median(estimates))


@pytest.mark.parametrize("true_speed", [0.5, 1.0, 3.0, 6.0])
def test_recovers_the_speed_it_was_given(true_speed: float) -> None:
    """The headline check: spikes made at a known speed must estimate back to it."""
    got = _estimate(mean_speed=true_speed)
    assert 0.5 * true_speed < got < 2.0 * true_speed, (
        f"true {true_speed:.2f} rad/s, estimated {got:.2f} — outside a factor of two, so the "
        "estimator is not measuring what it claims"
    )


def _joint_estimate(*, mean_speed: float, seed: int = 0, n_cells: int = 8, uninformative: int = 0):
    """The joint estimator on the same synthetic data, optionally with junk pairs mixed in."""
    times, angles, spikes = _simulate(mean_speed=mean_speed, seed=seed, n_cells=n_cells)
    curves, _ = tuning_curves(spike_times=spikes, angles=angles, times=times, n_bins=60)
    edges = np.arange(times[0], times[-1] + DT, DT)
    rates = np.array([np.histogram(spikes[u], bins=edges)[0].astype(float) for u in sorted(spikes)])

    rng = np.random.default_rng(seed + 99)
    # Out to the same widest lag the pipeline uses; a narrower one cannot hold the window a slow
    # sweep needs, and the fit would fail for a reason that is the test's fault, not the code's.
    widest = int(np.rint(MAX_LAG_S / DT))
    rhos, cgrams = [], []
    for a in range(len(curves)):
        for b in range(a + 1, len(curves)):
            rhos.append(angular_correlation(curve_a=curves[a], curve_b=curves[b]))
            cgrams.append(
                rate_cross_correlation(rate_a=rates[a], rate_b=rates[b], max_lag_bins=widest)
            )
    # Pairs carrying no angular information: a real tuning curve shape against pure noise.
    for _ in range(uninformative):
        rhos.append(rhos[0].copy())
        cgrams.append(rng.normal(0.0, 1e-3, size=2 * widest + 1))

    return fit_population_speed(
        rhos=np.array(rhos), correlograms=np.array(cgrams), dt=DT
    ).mean_speed


@pytest.mark.parametrize("true_speed", [0.5, 1.0, 3.0])
def test_joint_fit_recovers_the_speed_it_was_given(true_speed: float) -> None:
    """Pooling the pairs must not cost accuracy on data where the per-pair fit already works."""
    got = _joint_estimate(mean_speed=true_speed)
    assert 0.5 * true_speed < got < 2.0 * true_speed, f"true {true_speed:.2f}, joint fit {got:.2f}"


def test_joint_fit_survives_pairs_that_carry_no_signal() -> None:
    """The reason the joint fit exists.

    On real data the per-pair estimates scatter by several times their own median and fail one-sided,
    so a median over pairs inherits the tail. Mixing uninformative pairs into synthetic data
    reproduces that situation: a speed wrong for the session pays a cost in every real pair at once,
    so the junk pairs should be outvoted rather than averaged in.
    """
    true_speed = 1.0
    clean = _joint_estimate(mean_speed=true_speed)
    polluted = _joint_estimate(mean_speed=true_speed, uninformative=28)  # as many as the real pairs
    assert 0.5 * true_speed < polluted < 2.0 * true_speed, (
        f"junk pairs moved the joint fit from {clean:.2f} to {polluted:.2f} (true {true_speed:.2f})"
    )


def test_estimates_are_ordered_across_the_whole_range() -> None:
    """Ordering must survive even where the absolute scale is imperfect.

    Weaker than the point estimate but harder to satisfy by accident, and it is the property every
    between-animal comparison actually relies on.
    """
    got = [_estimate(mean_speed=v) for v in (0.5, 1.0, 3.0, 6.0)]
    assert got == sorted(got), f"estimates not monotonic in the true speed: {got}"


def test_speed_quadrature_has_the_requested_mean() -> None:
    """If the prior's mean were not the fitted parameter, the fit would target the wrong thing."""
    for model in ("chi2_3", "chi2_1", "exponential", "rayleigh"):
        nodes, weights = speed_quadrature(mean_speed=2.5, model=model, n_points=2048)
        assert np.isclose((nodes * weights).sum(), 2.5, rtol=0.02), model


def test_angular_correlation_peaks_at_the_offset_between_two_curves() -> None:
    """rho(theta) must peak where the two tuning curves are actually offset, or the mapping from
    time lag to swept angle is meaningless."""
    centres = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    a = _von_mises(centres, 0.0, 4.0, 30.0)
    b = _von_mises(centres, np.pi / 2, 4.0, 30.0)
    rho = angular_correlation(curve_a=a, curve_b=b)
    peak_angle = centres[int(np.argmax(rho))]
    assert np.isclose(peak_angle, np.pi / 2, atol=2 * np.pi / 60 * 2)


def test_prediction_is_flat_when_the_head_never_moves() -> None:
    """At zero speed every lag maps to zero angular offset, so the predicted correlation cannot
    vary with lag. A model that still bends is reading lag as something other than swept angle."""
    centres = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    rho = angular_correlation(
        curve_a=_von_mises(centres, 0.0, 4.0, 30.0), curve_b=_von_mises(centres, 0.3, 4.0, 30.0)
    )
    pred = predicted_correlation(
        rho=rho, lags_s=np.linspace(-0.5, 0.5, 41), mean_speed=1e-6
    )
    assert np.ptp(pred) < 1e-6
