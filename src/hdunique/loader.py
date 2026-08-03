"""DANDI 000056 loading via pynapple: sessions, spikes, REM epochs, cell selection.

Everything here reads the NWB files only. The source repo's CRCNS path (`.ang` / `.states.*` files,
`spike_counts`, shank-index matching) is deliberately not ported — see
docs/porting/2026-08-02-port-rem-diffusion-and-variance.md, decision D1.
"""

import re
from pathlib import Path

import pynapple as nap

from hdunique.env import dandi_root

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
