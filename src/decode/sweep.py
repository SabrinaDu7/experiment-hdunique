"""The REM diffusion sweep: one session -> one result row, plus the cache and parquet plumbing.

Pipeline per session (all of it the original M3 method, sourced from DANDI):
  ADn spikes -> sum-of-Gaussians rates over the REM bouts -> sqrt -> Isomap 3-D
  -> 12-knot ring, best of `n_restarts` by fit_err -> decode every REM point
  -> diffusion curve at lags 100..500 ms -> D = origin-forced slope over the first 200 ms.

Repeated `n_refits` times to give D mean +/- std.
"""

import dataclasses
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
from beartype import beartype
from jaxtyping import Float, jaxtyped

from core.config import (
    DIFFUSION_LAGS,
    FIT_WINDOWS_MS,
    HEADLINE_WINDOW_MS,
    PAPER_REM_TARGETS,
    DiffusionConfig,
)
from core.env import cache_dir, results_dir
from decode import loader, manifold, rates
from metrics import diffusion as dif

#: Parquet rows are replaced on this key; other cell sets / fit fractions are preserved.
MERGE_KEY: list[str] = ["session_id", "cell_set", "fit_frac"]


# --- cache -------------------------------------------------------------------------------------
@jaxtyped(typechecker=beartype)
@dataclasses.dataclass(frozen=True)
class CacheEntry:
    """One cached run: everything downstream of the expensive fit, and the metadata describing it.

    Holding it as a record rather than a loose dict means readers of the cache (the sweep, the
    parquet rebuild, the grid figures) all see the same fields with the same types.
    """

    meta: dict[str, object]
    embed: Float[np.ndarray, "time dim"]
    decoded: Float[np.ndarray, "refit time"]
    bout_lengths: list[int]


def cache_path(*, session_id: str, cell_set: str) -> Path:
    """Cache file for one (session, cell set) run."""
    return cache_dir() / f"{session_id}__{cell_set.replace('+', '-')}.npz"


def cache_signature(*, cfg: DiffusionConfig) -> dict[str, object]:
    """Every parameter the cached embedding *and* decoded angles depend on.

    The source repo guarded only the embedding parameters, so re-running at a different
    `n_restarts` silently reused decodes from another setting. Guarding both means a mismatched
    cache is recomputed instead of trusted — see docs, problem P5.
    """
    return {
        "n_samples": cfg.n_samples,
        "n_neighbors": cfg.n_neighbors,
        "fit_dim": cfg.fit_dim,
        "dt": cfg.dt,
        "sigma": cfg.sigma,
        "n_knots": cfg.n_knots,
        "knot_order": cfg.knot_order,
        "penalty": cfg.penalty,
        "dalpha": cfg.dalpha,
        "n_restarts": cfg.n_restarts,
        "n_refits": cfg.n_refits,
        "fit_frac": cfg.fit_frac,
        "seed": cfg.seed,
        "rem_label": cfg.rem_label,
        "cell_areas": list(cfg.cell_areas),
    }


@jaxtyped(typechecker=beartype)
def save_cache(
    *,
    cfg: DiffusionConfig,
    embed: Float[np.ndarray, "time dim"],
    decoded: Float[np.ndarray, "refit time"],
    bout_lengths: list[int],
    meta: dict[str, object],
) -> None:
    """Persist the embedding, the per-refit decoded angles, the bout structure and the metadata."""
    cache_dir().mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path(session_id=cfg.session_id, cell_set=cfg.cell_set),
        embed=embed.astype(np.float32),
        decoded=decoded.astype(np.float32),
        bout_lengths=np.array(bout_lengths, dtype=np.int64),
        meta=np.array(json.dumps(meta)),
        signature=np.array(json.dumps(cache_signature(cfg=cfg))),
    )


def read_cache_file(*, path: Path) -> CacheEntry:
    """Read one cache file, widening the float32 storage back to float64. No signature check."""
    z = np.load(path, allow_pickle=False)
    return CacheEntry(
        meta=json.loads(str(z["meta"])),
        embed=z["embed"].astype(float),
        decoded=z["decoded"].astype(float),
        bout_lengths=[int(v) for v in z["bout_lengths"]],
    )


def load_cache(*, cfg: DiffusionConfig) -> CacheEntry | None:
    """Read a cache entry if it exists and its signature matches the current config, else None."""
    path = cache_path(session_id=cfg.session_id, cell_set=cfg.cell_set)
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=False)
    if json.loads(str(z["signature"])) != cache_signature(cfg=cfg):
        return None
    return read_cache_file(path=path)


def iter_cache(*, cell_set: str | None = None) -> Iterator[CacheEntry]:
    """Every cache entry on disk, in session order, optionally restricted to one cell set.

    Used by the commands that work purely from cached decodes (the parquet rebuild and the grid
    figures), which is what makes them seconds rather than hours.
    """
    entries = [read_cache_file(path=path) for path in sorted(cache_dir().glob("*.npz"))]
    for entry in sorted(entries, key=lambda e: (int(e.meta["mouse"]), int(e.meta["session"]))):
        if cell_set is None or entry.meta["cell_set"] == cell_set:
            yield entry


# --- metrics -----------------------------------------------------------------------------------
@jaxtyped(typechecker=beartype)
def row_from_decodes(
    *,
    cfg: DiffusionConfig,
    meta: dict[str, object],
    decoded: Float[np.ndarray, "refit time"],
    bout_lengths: list[int],
) -> dict[str, object]:
    """Aggregate per-refit decoded angles into one result row.

    Two co-headline estimates, both origin-forced slopes averaged over refits: `D` pools every pair
    in the concatenated trace, and `D_bout_aware` excludes pairs straddling a REM-bout boundary.
    `nugget` (free-intercept) is a quality diagnostic, and `D_bout_ci_*` is the paper's epoch bootstrap.
    """
    per_window: dict[int, dict[str, list[float]]] = {
        ms: {"D": [], "r2": []} for ms in FIT_WINDOWS_MS
    }
    curves = []
    for trace in decoded:
        curve = dif.diffusion_curve(angles=trace, lags=DIFFUSION_LAGS)
        curves.append(curve)
        for ms in FIT_WINDOWS_MS:
            d, r2 = dif.window_slope(curve=curve, dt=cfg.dt, window_ms=ms)
            per_window[ms]["D"].append(d)
            per_window[ms]["r2"].append(r2)

    mean_curve = np.mean(curves, axis=0)
    d_free, nugget, r2_free = dif.free_intercept_fit(curve=mean_curve, dt=cfg.dt)

    # Co-headline: the same estimator with boundary-crossing pairs excluded. A pair spanning the
    # gap between two REM bouts is not a measurement of diffusion at that lag, so this is the
    # cleaner quantity; `D` is retained because it is what the source pipeline's tables report.
    bout_curves = [
        dif.diffusion_curve(angles=t, lags=DIFFUSION_LAGS, bout_lengths=bout_lengths)
        for t in decoded
    ]
    mean_bout_curve = np.mean(bout_curves, axis=0)

    # The bootstrap resamples data, so it needs one decoded trace rather than the mean curve; use
    # the refit whose D is the median, so the CI describes a representative ring, not an extreme.
    # Note the paper's epoch bootstrap is *inherently* bout-aware -- an epoch cannot straddle a
    # bout boundary -- so the interval it returns is a CI for `D_bout_aware`, not for `D`.
    median_trace = np.asarray(
        decoded[int(np.argsort(per_window[HEADLINE_WINDOW_MS]["D"])[len(decoded) // 2])],
        dtype=float,
    )
    ci_lo, ci_hi = dif.bootstrap_ci(
        angles=median_trace,
        bout_lengths=bout_lengths,
        dt=cfg.dt,
        window_ms=HEADLINE_WINDOW_MS,
        n_boot=cfg.n_bootstrap,
        seed=cfg.seed,
    )

    row: dict[str, object] = {
        **meta,
        "curve_rad2": [float(v) for v in mean_curve],
        "curve_bout_aware_rad2": [float(v) for v in mean_bout_curve],
        "D_freeint": d_free,
        "nugget": nugget,
        "r2_freeint": r2_free,
        "D_bout_ci_lo": ci_lo,
        "D_bout_ci_hi": ci_hi,
        "n_bootstrap": cfg.n_bootstrap,
    }
    for ms in FIT_WINDOWS_MS:
        d_bout, r2_bout = dif.window_slope(curve=mean_bout_curve, dt=cfg.dt, window_ms=ms)
        row[dif.window_column(stat="D_bout_aware", window_ms=ms, headline_ms=HEADLINE_WINDOW_MS)] = (
            d_bout
        )
        row[
            dif.window_column(stat="r2_bout_aware", window_ms=ms, headline_ms=HEADLINE_WINDOW_MS)
        ] = r2_bout
        ds, r2s = np.array(per_window[ms]["D"]), np.array(per_window[ms]["r2"])
        row[dif.window_column(stat="D", window_ms=ms, headline_ms=HEADLINE_WINDOW_MS)] = float(
            ds.mean()
        )
        row[dif.window_column(stat="D_std", window_ms=ms, headline_ms=HEADLINE_WINDOW_MS)] = float(
            ds.std()
        )
        row[dif.window_column(stat="r2", window_ms=ms, headline_ms=HEADLINE_WINDOW_MS)] = float(
            r2s.mean()
        )
    return row


# --- one session -------------------------------------------------------------------------------
def run_session(*, cfg: DiffusionConfig) -> dict[str, object] | None:
    """Compute the REM diffusion row for one session, or None if it has no usable data.

    A session is skipped when it has no cells of the requested set, no REM epochs, or when a
    multi-area set collapses to a single area (which would duplicate that area's own run).
    """
    cached = load_cache(cfg=cfg) if cfg.use_cache else None
    if cached is not None:
        return row_from_decodes(
            cfg=cfg,
            meta=cached.meta,
            decoded=cached.decoded,
            bout_lengths=cached.bout_lengths,
        )

    data = loader.load_session(mouse=cfg.mouse, session=cfg.session)
    units = loader.get_units(data=data)
    units_sel, unit_ids = loader.select_units_by_area(units=units, areas=cfg.cell_areas)
    rem = loader.load_state_epochs(data=data, state=cfg.rem_label)

    if not unit_ids or len(rem) == 0:
        print(f"  skip {cfg.session_id}: {len(unit_ids)} {cfg.cell_set} cells, {len(rem)} REM bouts")
        return None

    by_area = loader.count_units_by_area(units=units, unit_ids=unit_ids)
    if len(cfg.cell_areas) > 1 and any(by_area.get(a, 0) == 0 for a in cfg.cell_areas):
        print(f"  skip {cfg.session_id} [{cfg.cell_set}]: only one area present, duplicates its run")
        return None

    rate_mat = rates.rate_matrix(units=units_sel, epochs=rem, dt=cfg.dt, sigma=cfg.sigma)
    lengths = rates.bout_lengths(epochs=rem, dt=cfg.dt)

    # The n_samples cap truncates the concatenated trace, so truncate the bout structure to match.
    rate_mat = rate_mat[: cfg.n_samples]
    lengths = truncate_bouts(lengths=lengths, n_kept=len(rate_mat))

    embed = manifold.isomap_embed(rates=rate_mat, cfg=cfg)
    decoded = manifold.decode_refits(embed=embed, cfg=cfg)

    meta: dict[str, object] = {
        "mouse": cfg.mouse,
        "session": cfg.session,
        "session_id": cfg.session_id,
        "cell_set": cfg.cell_set,
        "n_cells": len(unit_ids),
        "n_adn": by_area.get("ADn", 0),
        "n_pos": by_area.get("PoS", 0),
        "n_samples": len(embed),
        "n_rem_bouts": len(rem),
        "fit_frac": cfg.fit_frac,
        "n_refits": cfg.n_refits,
        "n_restarts": cfg.n_restarts,
        "rem_source": "dandi_states",
        "paper_target": PAPER_REM_TARGETS.get(cfg.session_id),
    }
    if cfg.use_cache:
        save_cache(cfg=cfg, embed=embed, decoded=decoded, bout_lengths=lengths, meta=meta)

    return row_from_decodes(cfg=cfg, meta=meta, decoded=decoded, bout_lengths=lengths)


def truncate_bouts(*, lengths: list[int], n_kept: int) -> list[int]:
    """Trim the per-bout bin counts to the first `n_kept` bins of the concatenated trace."""
    out: list[int] = []
    remaining = n_kept
    for length in lengths:
        if remaining <= 0:
            break
        out.append(min(length, remaining))
        remaining -= out[-1]
    return out


# --- parquet -----------------------------------------------------------------------------------
def parquet_path(*, mouse: int) -> Path:
    """Result table for one mouse."""
    return results_dir() / f"diffusion_Mouse{mouse}.parquet"


def save_parquet(*, mouse: int, rows: list[dict[str, object]]) -> None:
    """Merge rows into the mouse's parquet on (session_id, cell_set, fit_frac)."""
    results_dir().mkdir(parents=True, exist_ok=True)
    path = parquet_path(mouse=mouse)
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_parquet(path)
        if set(MERGE_KEY).issubset(old.columns):
            keys = set(map(tuple, new[MERGE_KEY].to_numpy()))
            old = old[~old[MERGE_KEY].apply(tuple, axis=1).isin(keys)]
        new = pd.concat([old, new], ignore_index=True)
    new = new.sort_values(["fit_frac", "cell_set", "session"]).reset_index(drop=True)
    new.to_parquet(path, index=False)
    print(f"  -> {len(rows)} row(s) written; {len(new)} total in {path.name}")


def load_all_mice(*, mice: tuple[int, ...]) -> pd.DataFrame:
    """Concatenate every mouse's diffusion parquet into one long table."""
    frames = []
    for mouse in mice:
        path = parquet_path(mouse=mouse)
        if not path.exists():
            raise FileNotFoundError(f"No parquet for Mouse{mouse} at {path}. Run the sweep first.")
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True).sort_values(["mouse", "session"]).reset_index(
        drop=True
    )
