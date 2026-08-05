"""DANDI 000056 loading via pynapple: sessions, spikes, REM epochs, cell selection.

Everything here reads the NWB files only. The source repo's CRCNS path (`.ang` / `.states.*` files,
`spike_counts`, shank-index matching) is deliberately not ported — see
docs/porting/2026-08-02-port-rem-diffusion-and-variance.md, decision D1.
"""

import re
from pathlib import Path

import numpy as np
import pynapple as nap

from core.env import dandi_root

_NWB_RE = re.compile(r"sub-Mouse(\d+)_ses-Mouse\d+-(\d+)_")


def nwb_path(*, mouse: int, session: int) -> Path:
    """Path to one session's NWB file under the DANDI root."""
    return (
        dandi_root()
        / f"sub-Mouse{mouse}"
        / f"sub-Mouse{mouse}_ses-Mouse{mouse}-{session}_behavior+ecephys.nwb"
    )


def list_sessions(*, mouse: int | None = None) -> list[tuple[int, int]]:
    """Enumerate (mouse, session) pairs from the NWB filenames under the DANDI root, sorted.
    Pass `mouse` to restrict to one animal."""
    out: list[tuple[int, int]] = []
    for nwb in dandi_root().rglob("*.nwb"):
        m = _NWB_RE.search(nwb.name)
        if m and (mouse is None or int(m.group(1)) == mouse):
            out.append((int(m.group(1)), int(m.group(2))))
    return sorted(set(out))


def load_session(*, mouse: int, session: int) -> nap.NWBFile:
    """Open one session's NWB file as a pynapple NWBFile."""
    return nap.load_file(str(nwb_path(mouse=mouse, session=session)))


def get_units(*, data: nap.NWBFile) -> nap.TsGroup:
    """Spike trains for every sorted unit in the session."""
    return data["units"]


def load_state_epochs(*, data: nap.NWBFile, state: str = "REM") -> nap.IntervalSet:
    """Epochs of one sleep/wake state from the NWB `states` interval set (DANDI's own scoring).

    Returns an empty IntervalSet if the session has no `states` table or no epochs of that label.
    """
    # pynapple's NWBFile exposes its contents through .keys(), not __contains__.
    if "states" not in data.keys():  # noqa: SIM118
        return nap.IntervalSet(start=[], end=[])
    df = data["states"].as_dataframe()
    rows = df[df["label"] == state].sort_values("start")
    return nap.IntervalSet(start=rows["start"].to_numpy(), end=rows["end"].to_numpy())


def contiguous_bouts(
    *, epochs: nap.IntervalSet, merge_gap_s: float = 0.0, min_duration_s: float = 0.0
) -> nap.IntervalSet:
    """Merge epochs separated by no more than `merge_gap_s`, then drop those shorter than
    `min_duration_s`.

    DANDI's `states` table labels a whole day-long recording, so "Awake" is not one behavioural
    session: it is several multi-thousand-second bouts separated by sleep, plus a tail of brief
    arousals. Treating that union as a single stretch pools genuinely different behaviour — measured
    head speed differs about two-fold between a session's longest wake bout and the rest of its wake.

    Anything estimating a per-state quantity should choose its bouts explicitly rather than inherit
    that union by default.
    """
    start, end = np.asarray(epochs.start, dtype=float), np.asarray(epochs.end, dtype=float)
    if len(start) == 0:
        return nap.IntervalSet(start=[], end=[])

    order = np.argsort(start)
    start, end = start[order], end[order]
    merged_start, merged_end = [start[0]], [end[0]]
    for s, e in zip(start[1:], end[1:], strict=True):
        if s - merged_end[-1] <= merge_gap_s:
            merged_end[-1] = max(merged_end[-1], e)
        else:
            merged_start.append(s)
            merged_end.append(e)

    keep = np.array(merged_end) - np.array(merged_start) >= min_duration_s
    return nap.IntervalSet(start=np.array(merged_start)[keep], end=np.array(merged_end)[keep])


def longest_bout(*, epochs: nap.IntervalSet) -> nap.IntervalSet:
    """The single longest epoch, as a one-interval IntervalSet; empty in, empty out."""
    start, end = np.asarray(epochs.start, dtype=float), np.asarray(epochs.end, dtype=float)
    if len(start) == 0:
        return nap.IntervalSet(start=[], end=[])
    i = int(np.argmax(end - start))
    return nap.IntervalSet(start=[start[i]], end=[end[i]])


def select_units_by_area(
    *, units: nap.TsGroup, areas: tuple[str, ...]
) -> tuple[nap.TsGroup, list[int]]:
    """Select units whose DANDI per-unit `location` is one of `areas`.

    Returns (selected TsGroup, selected unit ids ascending). Empty if the session carries no
    `location` metadata — many DANDI sessions were unlabelled before the 2026-07-16 relabelling.
    """
    if "location" not in units.metadata.columns:
        return units[[]], []
    loc = units.get_info("location")
    ids = sorted(int(i) for i in loc[loc.isin(areas)].index)
    return units[ids], ids


def count_units_by_area(*, units: nap.TsGroup, unit_ids: list[int]) -> dict[str, int]:
    """Per-area breakdown of the selected units, keyed by `location` value."""
    if not unit_ids or "location" not in units.metadata.columns:
        return {}
    sel = units.get_info("location").loc[unit_ids]
    return {str(area): int((sel == area).sum()) for area in sorted(set(sel))}
