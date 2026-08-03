"""Per-bout diffusion constants, and the sleep-architecture context each bout sits in.

The published pipeline gives one *D* per session, pooled over every REM bout it contains. That
hides a level: a session is a handful of REM episodes minutes apart, and nothing guarantees they
share a drift rate. If they do not, the session-level *D* is a weighted average over a distribution,
and the variance decomposition's within-mouse component is partly bout-level spread rather than
session-to-session biology.

This module computes *D* for each bout separately and labels each bout with the state that preceded
and followed it, so that the three-level structure (bout within session within mouse) can be fitted
and so that context effects — notably whether a bout ended in an awakening — can be tested.

Two facts about this dataset shape what is worth asking (both measured, see
docs/bout_level/2026-08-03-bout-level-diffusion.md):

- **Merging REM bouts separated by a short gap is a no-op here.** The smallest REM-to-REM gap in
  the whole dataset is 11 s and the median is 597 s, so no threshold at or below 10 s merges
  anything. `merge_close_bouts` is provided anyway, because the rule is a reasonable one and the
  data could change; it is simply inert at the default.
- **REM is essentially always entered from Non-REM** (675 of 682 bouts), so the informative contrast
  is the *exit*: 590 bouts end in an awakening, 92 return to Non-REM.
"""

import numpy as np
import pandas as pd
import pynapple as nap
from beartype import beartype
from jaxtyping import Float, jaxtyped

from hdunique import diffusion as dif

#: Label used when a bout is the first or last epoch in the recording.
BOUNDARY_STATE: str = "none"


def merge_close_bouts(*, epochs: nap.IntervalSet, max_gap_s: float) -> nap.IntervalSet:
    """Merge consecutive REM epochs separated by no more than `max_gap_s`.

    A brief non-REM intrusion inside what is really one REM episode would otherwise split it into
    two bouts, shortening the usable lag range and inflating the apparent bout count. Merging spans
    the gap, so the intervening (differently scored) time is absorbed into the bout — which is only
    the right call if the gap really is mis-scored REM rather than a genuine arousal. Keep
    `max_gap_s` short for that reason.
    """
    start, end = np.asarray(epochs.start, dtype=float), np.asarray(epochs.end, dtype=float)
    if len(start) == 0:
        return epochs
    starts, ends = [start[0]], [end[0]]
    for s, e in zip(start[1:], end[1:], strict=True):
        if s - ends[-1] <= max_gap_s:
            ends[-1] = e
        else:
            starts.append(s)
            ends.append(e)
    return nap.IntervalSet(start=np.array(starts), end=np.array(ends))


def bout_context(*, data: nap.NWBFile, epochs: nap.IntervalSet) -> pd.DataFrame:
    """Label each REM bout with the states around it and where it sits in the recording.

    Returns one row per bout: start, duration, the state immediately before and after, and the
    bout's index within the session. The DANDI state epochs tile the recording contiguously, so
    "the state before" is simply the preceding epoch's label.
    """
    states = data["states"].as_dataframe().sort_values("start").reset_index(drop=True)
    labels = states["label"].to_numpy()

    rows = []
    for index, (start, end) in enumerate(
        zip(np.asarray(epochs.start), np.asarray(epochs.end), strict=True)
    ):
        before = states.index[states["end"] <= start + 1e-6]
        after = states.index[states["start"] >= end - 1e-6]
        rows.append(
            {
                "bout_index": index,
                "start_s": float(start),
                "end_s": float(end),
                "duration_s": float(end - start),
                "prev_state": labels[before[-1]] if len(before) else BOUNDARY_STATE,
                "next_state": labels[after[0]] if len(after) else BOUNDARY_STATE,
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame):
        # Where the bout sits in the night, as a fraction of the recording, so sessions of
        # different lengths are comparable.
        span = float(states["end"].max() - states["start"].min())
        frame["time_frac"] = (frame["start_s"] - float(states["start"].min())) / span
    return frame


@jaxtyped(typechecker=beartype)
def bout_diffusion(
    *,
    angles: Float[np.ndarray, " time"],
    dt: float,
    windows_ms: tuple[int, ...],
    lags: tuple[int, ...],
) -> dict[str, float]:
    """Diffusion constants for a single bout, using the published wrapped estimator.

    Wrapping is not a concern at these windows (see the long-timescale doc), so this is exactly the
    headline estimator applied to one bout instead of to the pooled session. Returns NaN for any
    window the bout is too short to support, rather than a number resting on no pairs.
    """
    out: dict[str, float] = {}
    usable = tuple(lag for lag in lags if len(angles) > lag)
    if not usable:
        return {f"D_{w}": float("nan") for w in windows_ms} | {"nugget": float("nan")}

    curve = np.array(
        [float(np.mean(dif.af.shifted_angular_diffs(angles, lag) ** 2)) for lag in usable]
    )
    for window_ms in windows_ms:
        n_needed = round(window_ms / 1000.0 / dt)
        if len(curve) < n_needed:
            out[f"D_{window_ms}"] = float("nan")
            continue
        out[f"D_{window_ms}"], _ = dif.window_slope(curve=curve, dt=dt, window_ms=window_ms)
    out["nugget"] = (
        dif.free_intercept_fit(curve=curve, dt=dt)[1] if len(curve) >= 3 else float("nan")
    )
    return out
