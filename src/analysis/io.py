"""Reading the pipeline's artefacts, in one place.

Before this module the repo had four parquet writers (only one of which merged on key), three
copies of the `"<mouse>-<session>"` spec parser, and two unrelated ways to glob per-mouse parquets.
Analyses also reached for `outputs/results/...` as a relative path, which quietly ignored
`$OUTPUT_PATH`. Everything that touches a result file now goes through here.
"""

from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from core.env import results_dir
from decode.sweep import CacheEntry, iter_cache

#: Filename stems of the per-mouse tables the pipeline writes.
DIFFUSION_STEM = "diffusion"
TIMESCALE_STEM = "timescale"
BOUTS_STEM = "bouts"


def parse_session_spec(spec: str) -> tuple[int, int]:
    """Parse a `"<mouse>-<session>"` spec, e.g. `"28-140313"` -> `(28, 140313)`."""
    mouse, _, session = spec.partition("-")
    if not session:
        raise ValueError(f"Bad session spec {spec!r}; expected '<mouse>-<session>' e.g. '28-140313'")
    return int(mouse), int(session)


def select_sessions(
    *, cell_set: str = "ADn", sessions: Iterable[str] = ()
) -> Iterator[CacheEntry]:
    """Cached runs of one cell set, optionally restricted to given `"<mouse>-<session>"` specs.

    `iter_cache` can only filter by cell set, so three call sites had each written their own
    session filter; this is that filter, once.
    """
    wanted = {parse_session_spec(s) for s in sessions}
    for entry in iter_cache(cell_set=cell_set):
        key = (int(entry.meta["mouse"]), int(entry.meta["session"]))
        if not wanted or key in wanted:
            yield entry


def load_tables(*, stem: str, cell_set: str | None = None) -> pd.DataFrame:
    """Concatenate the per-mouse tables of one kind into a single frame.

    `stem` is one of the module constants. `cell_set` is required for the tables whose filename
    carries it (timescale, bouts) and ignored for those that do not (diffusion, which stores the
    cell set as a column instead).
    """
    pattern = (
        f"{stem}_Mouse*.parquet" if cell_set is None else f"{stem}_Mouse*_{cell_set}.parquet"
    )
    paths = sorted(results_dir().glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No {stem} tables matching {pattern} in {results_dir()}. "
            "Run the collection step for this question first."
        )
    frame = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    sort_cols = [c for c in ("mouse", "session", "bout_index") if c in frame.columns]
    return frame.sort_values(sort_cols).reset_index(drop=True)


def load_diffusion(*, cell_set: str = "ADn") -> pd.DataFrame:
    """The per-session diffusion table, restricted to one cell set."""
    frame = load_tables(stem=DIFFUSION_STEM)
    return frame[frame["cell_set"] == cell_set].reset_index(drop=True)


def load_timescale(*, cell_set: str = "ADn") -> pd.DataFrame:
    """The per-session long-lag table for one cell set."""
    return load_tables(stem=TIMESCALE_STEM, cell_set=cell_set)


def load_bouts(*, cell_set: str = "ADn") -> pd.DataFrame:
    """The per-bout table for one cell set."""
    return load_tables(stem=BOUTS_STEM, cell_set=cell_set)


def with_log_d(*, frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Add `log_D` from `column`, dropping rows where D is missing or non-positive.

    Four different expressions for "drop unusable D" had accumulated across the codebase and its
    docs; this is the one definition. Rows are dropped rather than carried as NaN so that every
    downstream count is the count actually modelled.
    """
    out = frame[np.isfinite(frame[column]) & (frame[column] > 0)].copy()
    out["log_D"] = np.log(out[column])
    return out.reset_index(drop=True)


def load_shards(*, name: str) -> pd.DataFrame | None:
    """Concatenate `<name>.parquet` and any `<name>_shard*.parquet` written by parallel workers.

    A long sweep is split across processes by session, each writing its own shard, because the work
    is embarrassingly parallel and running it on one core wastes most of the machine. Readers should
    not have to know whether a table arrived in one piece or eight.
    """
    paths = sorted(results_dir().glob(f"{name}.parquet")) + sorted(
        results_dir().glob(f"{name}_shard*.parquet")
    )
    if not paths:
        return None
    frame = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    keys = [c for c in ("mouse", "session", "bout_index") if c in frame.columns]
    if keys:
        frame = frame.drop_duplicates(subset=keys, keep="last").sort_values(keys)
    return frame.reset_index(drop=True)


def shard_name(*, name: str, shard: int, n_shards: int) -> str:
    """Table name for one worker's slice; unsuffixed when the sweep is not split."""
    return name if n_shards <= 1 else f"{name}_shard{shard}"


def save_table(*, frame: pd.DataFrame, name: str) -> Path:
    """Write a result table to `outputs/results/<name>.parquet` and return its path."""
    results_dir().mkdir(parents=True, exist_ok=True)
    path = results_dir() / f"{name}.parquet"
    frame.to_parquet(path, index=False)
    return path
