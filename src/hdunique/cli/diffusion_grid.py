"""Per-mouse grid of REM diffusion curves, one panel per session, built from the decode cache.

    uv run hd-diffusion-grid                     # every mouse, 200 ms window
    uv run hd-diffusion-grid --window-ms 500

No NWB, Isomap or ring fitting: the cached decoded angles are enough.
"""

import dataclasses
import json

import numpy as np
import tyro

from hdunique import diffusion as dif
from hdunique.config import DIFFUSION_LAGS, HEADLINE_WINDOW_MS
from hdunique.env import cache_dir, results_dir
from hdunique.plotting import plot_mouse_diffusion_grid


@dataclasses.dataclass(frozen=True)
class GridConfig:
    """Which cached runs to draw, and over what fit window."""

    window_ms: int = HEADLINE_WINDOW_MS
    #: Cell set to draw. ADn is the analysis set; the ring is carried by ADn.
    cell_set: str = "ADn"
    dt: float = 0.1


def _panels_by_mouse(*, cfg: GridConfig) -> dict[int, list[dict[str, object]]]:
    """Read every matching cache entry and build one panel spec per session."""
    out: dict[int, list[dict[str, object]]] = {}
    for path in sorted(cache_dir().glob("*.npz")):
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        if meta["cell_set"] != cfg.cell_set:
            continue
        curves = [
            dif.diffusion_curve(angles=t, lags=DIFFUSION_LAGS)
            for t in z["decoded"].astype(float)
        ]
        curve = np.mean(curves, axis=0)
        d, r2 = dif.window_slope(curve=curve, dt=cfg.dt, window_ms=cfg.window_ms)
        out.setdefault(int(meta["mouse"]), []).append(
            {
                "lags_s": np.array(DIFFUSION_LAGS, dtype=float) * cfg.dt,
                "curve": curve,
                "D": d,
                "r2": r2,
                "n_cells": int(meta["n_cells"]),
                "session": int(meta["session"]),
                "title": f"{meta['session_id']}  ({meta['n_cells']} cells)",
            }
        )
    return out


def run(*, cfg: GridConfig) -> None:
    """Write one grid figure per mouse."""
    panels_by_mouse = _panels_by_mouse(cfg=cfg)
    if not panels_by_mouse:
        print(f"No cached runs for cell_set={cfg.cell_set} in {cache_dir()}.")
        return
    n_fit = round(cfg.window_ms / 1000.0 / cfg.dt)
    results_dir().mkdir(parents=True, exist_ok=True)
    for mouse, panels in sorted(panels_by_mouse.items()):
        plot_mouse_diffusion_grid(
            mouse=mouse,
            panels=sorted(panels, key=lambda p: p["session"]),
            n_fit=n_fit,
            window_ms=cfg.window_ms,
            cell_set=cfg.cell_set,
            save_path=results_dir()
            / f"Mouse{mouse}_rem_diffusion_grid_{cfg.cell_set}_{cfg.window_ms}ms.png",
        )


def main() -> None:
    run(cfg=tyro.cli(GridConfig))


if __name__ == "__main__":
    main()
