"""Entry point for per-bout REM diffusion constants.

    Called by an experiment's collect(); not a console script.
    uv run hd-bouts --sessions 25-140130 28-140313 12-120806
    uv run hd-bouts --merge-gap-s 20                    # merge REM bouts closer than 20 s

Writes one parquet per mouse, one row per REM bout, with the bout's diffusion constant and the
sleep-architecture context it sits in.

The decoded angles come from the cache, so this is seconds per session — but note the cache was
built from *unmerged* REM epochs. Raising `--merge-gap-s` above the smallest real gap changes the
epochs and therefore invalidates the cache; the command refuses rather than quietly mismatching.
"""

import dataclasses

import numpy as np
import pandas as pd

from core.config import DIFFUSION_LAGS, HEADLINE_WINDOW_MS
from core.env import results_dir
from decode import loader
from decode.sweep import iter_cache
from figures import panels
from metrics import bouts

#: Lags used for the per-bout curve: the same 100..500 ms the headline estimator fits over.
BOUT_LAGS: tuple[int, ...] = DIFFUSION_LAGS


@dataclasses.dataclass(frozen=True)
class BoutConfig:
    """Which cached runs to break into bouts, and how bouts are defined."""

    cell_set: str = "ADn"
    #: Restrict to these `<mouse>-<session>` pairs; empty means every cached session.
    sessions: tuple[str, ...] = ()
    #: Merge REM epochs separated by no more than this. The smallest real gap in this dataset is
    #: 11 s, so the default is deliberately inert — see `bouts.merge_close_bouts`.
    merge_gap_s: float = 10.0
    dt: float = 0.1
    windows_ms: tuple[int, ...] = (200, 500)
    #: Bouts shorter than this contribute too few pairs to fit; they are kept but flagged.
    min_duration_s: float = 10.0
    #: Sessions to draw an exit-state strip figure for, as `<mouse>-<session>`. Empty draws none.
    plot_sessions: tuple[str, ...] = ()


def session_rows(*, cfg: BoutConfig, entry: object) -> list[dict[str, object]]:
    """One row per REM bout of a single cached session.

    The cache stores bout lengths in the same order as the session's REM epochs, truncated by the
    `n_samples` cap, so the two are aligned by position and the context frame is trimmed to match.
    """
    meta = entry.meta  # type: ignore[attr-defined]
    mouse, session = int(meta["mouse"]), int(meta["session"])
    data = loader.load_session(mouse=mouse, session=session)
    epochs = loader.load_state_epochs(data=data, state="REM")
    merged = bouts.merge_close_bouts(epochs=epochs, max_gap_s=cfg.merge_gap_s)
    if len(merged) != len(epochs):
        raise RuntimeError(
            f"{meta['session_id']}: --merge-gap-s {cfg.merge_gap_s} merges "
            f"{len(epochs)} REM epochs into {len(merged)}, which changes the rates and invalidates "
            "the decode cache. Re-run `hd-diffusion` with the same merge rule first."
        )

    context = bouts.bout_context(data=data, epochs=merged)
    lengths = entry.bout_lengths  # type: ignore[attr-defined]
    context = context.iloc[: len(lengths)].reset_index(drop=True)

    rows: list[dict[str, object]] = []
    offset = 0
    for i, n_bins in enumerate(lengths):
        # Average the per-bout D over the cached refits, matching how session-level D is formed.
        per_refit = [
            bouts.bout_diffusion(
                angles=np.asarray(trace[offset : offset + n_bins], dtype=float),
                dt=cfg.dt,
                windows_ms=cfg.windows_ms,
                lags=BOUT_LAGS,
            )
            for trace in entry.decoded  # type: ignore[attr-defined]
        ]
        offset += n_bins
        stats = {
            key: float(np.mean([r[key] for r in per_refit]))
            for key in per_refit[0]
        }
        stats_std = float(np.std([r[f"D_{HEADLINE_WINDOW_MS}"] for r in per_refit]))
        rows.append(
            {
                "mouse": mouse,
                "session": session,
                "session_id": meta["session_id"],
                "cell_set": meta["cell_set"],
                "n_cells": meta["n_cells"],
                **context.iloc[i].to_dict(),
                "n_bins": int(n_bins),
                # The cap can truncate the final bout, so record whether this row is a full bout.
                "truncated": bool(n_bins < round(context.iloc[i]["duration_s"] / cfg.dt) - 1),
                "short": bool(context.iloc[i]["duration_s"] < cfg.min_duration_s),
                **stats,
                "D_std": stats_std,
            }
        )
    return rows


def run(*, cfg: BoutConfig) -> None:
    """Break every requested session into bouts and write one parquet per mouse."""
    wanted = {tuple(int(p) for p in s.split("-")) for s in cfg.sessions}
    by_mouse: dict[int, list[dict[str, object]]] = {}
    for entry in iter_cache(cell_set=cfg.cell_set):
        key = (int(entry.meta["mouse"]), int(entry.meta["session"]))
        if wanted and key not in wanted:
            continue
        rows = session_rows(cfg=cfg, entry=entry)
        by_mouse.setdefault(key[0], []).extend(rows)
        d_col = f"D_{HEADLINE_WINDOW_MS}"
        finite = [r[d_col] for r in rows if np.isfinite(r[d_col])]
        print(
            f"  {entry.meta['session_id']}: {len(rows)} bouts, "
            f"D range {min(finite):.2f}-{max(finite):.2f}, median {np.median(finite):.2f}"
        )

    if not by_mouse:
        print("No matching cached sessions.")
        return

    if cfg.plot_sessions:
        every = pd.DataFrame([r for rows in by_mouse.values() for r in rows])
        wanted_ids = [f"Mouse{m}-{s}" for m, s in (tuple(int(p) for p in x.split("-"))
                                                   for x in cfg.plot_sessions)]
        frames = {sid: every[every.session_id == sid] for sid in wanted_ids}
        frames = {k: v for k, v in frames.items() if len(v)}
        if frames:
            results_dir().mkdir(parents=True, exist_ok=True)
            panels.plot_bout_exit_strip(
                frames=frames, save_path=results_dir() / "bout_exit_strip.png"
            )

    results_dir().mkdir(parents=True, exist_ok=True)
    for mouse, rows in sorted(by_mouse.items()):
        frame = pd.DataFrame(rows).sort_values(["session", "bout_index"]).reset_index(drop=True)
        path = results_dir() / f"bouts_Mouse{mouse}_{cfg.cell_set}.parquet"
        frame.to_parquet(path, index=False)
        print(f"  -> {len(frame)} bouts in {path.name}")

