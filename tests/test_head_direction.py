"""Does the measured ground truth measure what `angular1`'s validation claims it does?

`angular1_exp2` grades the spike-based estimator against the head angle measured from the tracking
LEDs, and the grade depends entirely on *which* measured speed it is compared to. Path length and
net displacement differ by a factor that grows with the window, so if these two are not what they
claim to be, the validation reports a fault in the estimator that is really a difference of
definition.
"""

import numpy as np
import pytest

from decode.head_direction import (
    angular_speed,
    angular_velocity,
    integrate_velocity,
    net_speed,
)

DT = 1.0 / 39.0  # s, the tracking sample interval in this dataset


def _trace(*, duration: float = 600.0, speed: float = 2.0, flip_period: float = 4.0, seed: int = 0):
    """A head-direction trace that reverses direction periodically, as a real head does."""
    rng = np.random.default_rng(seed)
    times = np.arange(0.0, duration, DT)
    direction = np.sign(np.sin(2 * np.pi * times / flip_period))
    step = rng.chisquare(3, size=len(times)) / 3.0 * speed * direction * DT
    return times, np.cumsum(step) % (2.0 * np.pi)


def test_constant_turn_gives_back_its_own_speed_both_ways() -> None:
    """With no reversals there is nothing to cancel, so net and path length must agree."""
    times = np.arange(0.0, 600.0, DT)
    angles = (1.5 * times) % (2.0 * np.pi)
    assert np.isclose(np.mean(angular_speed(angles=angles, times=times)), 1.5, rtol=0.01)
    assert np.isclose(net_speed(angles=angles, times=times, tau_s=0.25), 1.5, rtol=0.05)


def test_net_speed_is_below_path_length_when_the_head_reverses() -> None:
    """The premise of the validation: reversals cancel in the net measure and accumulate in path
    length, so the two are not interchangeable ground truths."""
    times, angles = _trace()
    path = float(np.mean(angular_speed(angles=angles, times=times)))
    net = net_speed(angles=angles, times=times, tau_s=1.0)
    assert net < path, f"net {net:.2f} should be below path length {path:.2f} for a reversing head"


@pytest.mark.parametrize("flip_period", [2.0, 4.0, 8.0])
def test_net_speed_falls_as_the_window_lengthens(flip_period: float) -> None:
    """More reversals fit inside a longer window, so more of the motion cancels. If net speed did
    not fall with tau it would not be measuring displacement, and the tau reported alongside the
    validation ratio would be meaningless."""
    times, angles = _trace(flip_period=flip_period)
    speeds = [net_speed(angles=angles, times=times, tau_s=tau) for tau in (0.25, 0.5, 1.0, 2.0)]
    assert speeds == sorted(speeds, reverse=True), f"net speed not decreasing in tau: {speeds}"


def test_wrapping_is_not_read_as_a_full_turn() -> None:
    """A crossing of 2pi is a small step, not a large one. Both measures difference the short way
    round; if either did not, every wrap would add a spurious 2pi."""
    times = np.arange(0.0, 10.0, DT)
    angles = (0.1 * times + 6.2) % (2.0 * np.pi)  # crosses 2pi early and keeps going
    assert np.max(angular_speed(angles=angles, times=times)) < 1.0
    assert np.isclose(net_speed(angles=angles, times=times, tau_s=0.5), 0.1, rtol=0.05)


def test_velocity_integrates_back_to_the_angle_it_came_from() -> None:
    """The round trip must close to machine precision.

    This cannot validate the *magnitude* of the velocity against anything external — the
    reconstruction is exact by construction — but it does test everything around it: differencing
    the short way round, a consistent sign convention, and dividing each step by the interval it
    actually spans. Any of those wrong and the angle does not come back.
    """
    times, angles = _trace(duration=300.0)
    velocity = angular_velocity(angles=angles, times=times)
    rebuilt = integrate_velocity(velocity=velocity, times=times, initial_angle=float(angles[0]))
    error = np.abs((rebuilt - angles + np.pi) % (2.0 * np.pi) - np.pi)
    assert error.max() < 1e-9, f"round trip does not close: max error {error.max():.2e} rad"


def test_signed_velocity_is_the_signed_form_of_the_speed() -> None:
    """The two must not drift apart; `angular_speed` is documented as the absolute value."""
    times, angles = _trace(duration=300.0)
    assert np.allclose(
        np.abs(angular_velocity(angles=angles, times=times)),
        angular_speed(angles=angles, times=times),
    )


@pytest.mark.parametrize("stride", [1, 2, 4, 8])
def test_net_speed_barely_moves_with_the_sampling_rate(stride: int) -> None:
    """Net speed at a fixed tau must be a property of the head, not of the sampling rate.

    Path length is not: on real wake tracking it falls by a factor of ~2.4 between 39 Hz and 2.4 Hz,
    because finer sampling accumulates more tracking jitter. That makes path length unusable as a
    ground truth to grade an estimator against, and net speed usable. This test pins the property
    the choice rests on.
    """
    times, angles = _trace(duration=900.0, flip_period=4.0)
    reference = net_speed(angles=angles, times=times, tau_s=1.0)
    decimated = net_speed(angles=angles[::stride], times=times[::stride], tau_s=1.0)
    assert np.isclose(decimated, reference, rtol=0.1), (
        f"net speed moved from {reference:.3f} to {decimated:.3f} on {stride}x decimation"
    )


def test_failed_led_detections_are_dropped_not_used_as_positions() -> None:
    """`-1` marks a failed detection, and it is finite — so an isfinite check keeps it.

    Treating (-1, -1) as a position the animal was at is not a small error: failed detections run to
    20.5% of samples in Mouse12-120808, and keeping them inflated net angular speed by 1.48x against
    an independent computation of the same signal. Dropping them brings the agreement to 1.000.
    """
    n = 1000
    times = np.arange(n) * DT
    angle = np.linspace(0, 4 * np.pi, n)
    red = np.column_stack([np.cos(angle), np.sin(angle)]) * 10.0 + 50.0
    blue = np.full_like(red, 50.0)
    bad = np.zeros(n, dtype=bool)
    bad[100:150] = True
    red[bad] = -1.0

    class _Series:
        def __init__(self, values):
            self.values = values
            self.index = times

    class _File(dict):
        pass

    data = _File({"RedLED": _Series(red), "BlueLED": _Series(blue)})
    from decode.head_direction import head_direction

    hd = head_direction(data=data)
    assert len(hd) == n - bad.sum(), "failed detections were not dropped"
    assert not np.any(np.isclose(np.asarray(hd.index), times[bad][:, None]).any(axis=0))
