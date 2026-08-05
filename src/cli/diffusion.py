"""Entry point for the REM diffusion sweep.

    uv run hd-diffusion --scope all                              # the published sweep, no flags
    uv run hd-diffusion --scope session --mouse 28 --session 140313 --make-plot
    uv run hd-diffusion --scope mouse --mouse 25
    uv run hd-diffusion --recompute-only                         # rebuild parquets from cache

Every default is the setting that produced the published tables, so `--scope all` alone is the
command of record. See docs/porting/REPRODUCING.md.
"""

import dataclasses
import time

import numpy as np
import tyro

from core.config import DIFFUSION_LAGS, HEADLINE_WINDOW_MS, DiffusionConfig
from core.env import cache_dir, results_dir
from decode import loader, sweep
from figures import panels
from metrics import diffusion as dif


def _print_row(*, row: dict[str, object], elapsed: float) -> None:
    """One line per session: D, the diagnostics, and the cells / bouts behind them."""
    target = row.get("paper_target")
    print(
        f"  {row['session_id']} [{row['cell_set']}]: "
        f"D={row['D']:.3f}±{row['D_std']:.3f}  "
        f"D_bout={row['D_bout_aware']:.3f} "
        f"[{row['D_bout_ci_lo']:.3f}, {row['D_bout_ci_hi']:.3f}]  nugget={row['nugget']:+.3f}  "
        f"({row['n_cells']} cells, {row['n_rem_bouts']} bouts) [{elapsed:.0f}s]"
        + (f"  paper≈{target}" if target else "")
    )


def _plot_session(*, cfg: DiffusionConfig, row: dict[str, object]) -> None:
    """Save the single-session diffusion-curve figure."""
    panels.plot_session_diffusion(
        panel=panels.DiffusionPanel(
            title=f"{row['session_id']}  ({row['n_cells']} {row['cell_set']} cells)",
            lags_s=np.array(DIFFUSION_LAGS, dtype=float) * cfg.dt,
            curve=np.array(row["curve_rad2"], dtype=float),
            d=float(row["D"]),
            r2=float(row["r2"]),
            n_cells=int(row["n_cells"]),
        ),
        n_fit=len(dif.window_lags(dt=cfg.dt, window_ms=HEADLINE_WINDOW_MS)),
        window_ms=HEADLINE_WINDOW_MS,
        save_path=results_dir() / f"{cfg.session_id}_{cfg.cell_set}_rem_diffusion.png",
    )


def run_recompute_only(*, cfg: DiffusionConfig) -> None:
    """Rebuild the parquets straight from cached decoded angles — no NWB, Isomap or ring fitting."""
    by_mouse: dict[int, list[dict[str, object]]] = {}
    for entry in sweep.iter_cache():
        row = sweep.row_from_decodes(
            cfg=cfg, meta=entry.meta, decoded=entry.decoded, bout_lengths=entry.bout_lengths
        )
        by_mouse.setdefault(int(entry.meta["mouse"]), []).append(row)
        _print_row(row=row, elapsed=0.0)
    if not by_mouse:
        print(f"No cache in {cache_dir()}; run a full sweep first.")
        return
    for mouse, rows in by_mouse.items():
        sweep.save_parquet(mouse=mouse, rows=rows)


def run(*, cfg: DiffusionConfig) -> None:
    """Run the sweep at the configured scope and write one parquet per mouse."""
    if cfg.recompute_only:
        run_recompute_only(cfg=cfg)
        return

    if cfg.scope == "session":
        sessions = [(cfg.mouse, cfg.session)]
    elif cfg.scope == "mouse":
        sessions = loader.list_sessions(mouse=cfg.mouse)
    else:
        sessions = loader.list_sessions()

    by_mouse: dict[int, list[int]] = {}
    for mouse, session in sessions:
        by_mouse.setdefault(mouse, []).append(session)

    for mouse, mouse_sessions in by_mouse.items():
        rows: list[dict[str, object]] = []
        for session in mouse_sessions:
            print(f"[Mouse{mouse}-{session}]")
            t0 = time.time()
            session_cfg = dataclasses.replace(cfg, mouse=mouse, session=session)
            row = sweep.run_session(cfg=session_cfg)
            if row is None:
                continue
            _print_row(row=row, elapsed=time.time() - t0)
            rows.append(row)
            if cfg.make_plot:
                _plot_session(cfg=session_cfg, row=row)
        if rows:
            sweep.save_parquet(mouse=mouse, rows=rows)
        else:
            print(f"  Mouse{mouse}: no usable sessions, no parquet written")


def main() -> None:
    """Console-script entry point for `hd-diffusion`."""
    run(cfg=tyro.cli(DiffusionConfig))


if __name__ == "__main__":
    main()
