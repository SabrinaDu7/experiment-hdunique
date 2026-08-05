"""Does the measured ground truth measure what `angular1`'s validation claims it does?

`angular1_exp2` grades the spike-based estimator against the head angle measured from the tracking
LEDs, and the grade depends entirely on *which* measured speed it is compared to. Path length and
net displacement differ by a factor that grows with the window, so if these two are not what they
claim to be, the validation reports a fault in the estimator that is really a difference of
definition.
"""

import numpy as np
import pytest

from decode.head_direction import angular_speed, net_speed

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
