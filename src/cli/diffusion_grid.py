"""Per-mouse grid of REM diffusion curves, one panel per session, built from the decode cache.

    uv run hd-diffusion-grid                     # every mouse, 200 ms window
    uv run hd-diffusion-grid --window-ms 500

No NWB, Isomap or ring fitting: the cached decoded angles are enough.
"""

import dataclasses

import numpy as np
import tyro

import diffusion as dif
import sweep
from config import DIFFUSION_LAGS, HEADLINE_WINDOW_MS
from env import cache_dir, results_dir
from plotting import DiffusionPanel, plot_mouse_diffusion_grid


@dataclasses.dataclass(frozen=True)
class GridConfig:
    """Which cached runs to draw, and over what fit window."""

    window_ms: int = HEADLINE_WINDOW_MS
    #: Cell set to draw. ADn is the analysis set; the ring is carried by ADn.
    cell_set: str = "ADn"
    dt: float = 0.1


def _panels_by_mouse(*, cfg: GridConfig) -> dict[int, list[DiffusionPanel]]:
    """Read every matching cache entry and build one panel per session, in session order."""
    out: dict[int, list[DiffusionPanel]] = {}
    for entry in sweep.iter_cache(cell_set=cfg.cell_set):
        curve = np.mean(
            [dif.diffusion_curve(angles=t, lags=DIFFUSION_LAGS) for t in entry.decoded], axis=0
        )
        d, r2 = dif.window_slope(curve=curve, dt=cfg.dt, window_ms=cfg.window_ms)
        out.setdefault(int(entry.meta["mouse"]), []).append(
            DiffusionPanel(
                title=f"{entry.meta['session_id']}  ({entry.meta['n_cells']} cells)",
                lags_s=np.array(DIFFUSION_LAGS, dtype=float) * cfg.dt,
                curve=curve,
                d=d,
                r2=r2,
                n_cells=int(entry.meta["n_cells"]),
            )
        )
    return out


def run(*, cfg: GridConfig) -> None:
    """Write one grid figure per mouse."""
    panels_by_mouse = _panels_by_mouse(cfg=cfg)
    if not panels_by_mouse:
        print(f"No cached runs for cell_set={cfg.cell_set} in {cache_dir()}.")
        return
    n_fit = len(dif.window_lags(dt=cfg.dt, window_ms=cfg.window_ms))
    results_dir().mkdir(parents=True, exist_ok=True)
    for mouse, panels in sorted(panels_by_mouse.items()):
        plot_mouse_diffusion_grid(
            mouse=mouse,
            panels=panels,
            n_fit=n_fit,
            window_ms=cfg.window_ms,
            cell_set=cfg.cell_set,
            save_path=results_dir()
            / f"Mouse{mouse}_rem_diffusion_grid_{cfg.cell_set}_{cfg.window_ms}ms.png",
        )


def main() -> None:
    """Console-script entry point for `hd-diffusion-grid`."""
    run(cfg=tyro.cli(GridConfig))


if __name__ == "__main__":
    main()
