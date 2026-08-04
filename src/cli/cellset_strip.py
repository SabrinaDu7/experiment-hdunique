"""Entry point for the per-cell-set REM diffusion strip plots, one figure per measurement window.

    uv run hd-cellset-strip                                  # 1, 2, 3, 4, 5 s
    uv run hd-cellset-strip --windows-ms 200 500
    uv run hd-cellset-strip --estimator wrapped

Reads the timescale parquets, which store each session's full 50-point MSD curve, so *D* at any
window up to 5 s is refitted from what is already on disk — no cache access and no recompute of the
curves. Run `hd-timescale` for each cell set first.

The `circular` estimator is the default because these windows are long enough for wrapping to
matter; see docs/long_D/2026-08-03-long-timescale-diffusion.md §3.
"""

import dataclasses
import glob

import numpy as np
import pandas as pd
import tyro

import diffusion as dif
from env import results_dir
from plotting import WELL_SAMPLED_CELLS, plot_cellset_strip

#: Cell sets to draw, in panel order. ADn is the analysis set; the other two are diagnostics.
CELL_SETS: tuple[str, ...] = ("ADn", "ADn+PoS", "PoS")


@dataclasses.dataclass(frozen=True)
class CellSetStripConfig:
    """Which windows and estimator to draw, and how to flag undersampled sessions."""

    #: Measurement windows, in ms. Each gets its own figure, all on a shared D axis.
    windows_ms: tuple[int, ...] = (1000, 2000, 3000, 4000, 5000)
    #: Which of the three wrapping-aware estimators to fit.
    estimator: str = "circular"
    cell_sets: tuple[str, ...] = CELL_SETS
    dt: float = 0.1
    #: Below this cell count a session is drawn with an open marker.
    well_sampled: int = WELL_SAMPLED_CELLS


def load_curves(*, cell_set: str) -> pd.DataFrame:
    """All sessions of one cell set, from the timescale parquets. Empty frame if none exist."""
    paths = sorted(glob.glob(str(results_dir() / f"timescale_Mouse*_{cell_set}.parquet")))
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def fit_window(
    *, frame: pd.DataFrame, window_ms: int, estimator: str, dt: float
) -> pd.DataFrame:
    """Refit D at one window from each session's stored MSD curve, and add `log_D`.

    Refitting from the stored curve rather than re-reading the cache keeps this command a
    seconds-long operation on the parquets alone, and guarantees it uses exactly the curve the
    timescale results were computed from.
    """
    out = frame.copy()
    out["D"] = [
        dif.window_slope(
            curve=np.asarray(curve, dtype=float), dt=dt, window_ms=window_ms
        )[0]
        for curve in out[f"curve_{estimator}"]
    ]
    out["log_D"] = np.log(out["D"].where(out["D"] > 0))
    return out


def drop_unmeasurable(*, frame: pd.DataFrame, cell_set: str, window_ms: int) -> pd.DataFrame:
    """Drop sessions whose D is undefined at this window, naming them rather than hiding them.

    The `circular` estimator is −2·ln⟨cos Δα⟩, so it is undefined once the mean resultant reaches
    zero and goes negative — the angle is then fully decorrelated and the data no longer contain a
    rate to recover. That is a real limit of the measurement, not a defect to paper over, so the
    session is named on the way out.
    """
    ok = np.isfinite(frame["log_D"])
    for row in frame[~ok].itertuples():
        print(
            f"    drop {row.session_id} [{cell_set}] at {window_ms / 1000:g} s: "
            f"D undefined ({row.n_cells} cells, R at 5 s = {row.resultant_at_max_lag:.3f} "
            "— fully decorrelated)"
        )
    return frame[ok].reset_index(drop=True)


def run(*, cfg: CellSetStripConfig) -> None:
    """Write one strip figure per window, on a D axis shared across every window and cell set."""
    raw = {cs: load_curves(cell_set=cs) for cs in cfg.cell_sets}
    missing = [cs for cs, f in raw.items() if f.empty]
    if missing:
        print(f"No timescale parquets for {missing}; run `hd-timescale --cell-set <set>` first.")
    raw = {cs: f for cs, f in raw.items() if not f.empty}
    if not raw:
        return

    fitted = {
        window_ms: {
            cs: drop_unmeasurable(
                frame=fit_window(
                    frame=f, window_ms=window_ms, estimator=cfg.estimator, dt=cfg.dt
                ),
                cell_set=cs,
                window_ms=window_ms,
            )
            for cs, f in raw.items()
        }
        for window_ms in cfg.windows_ms
    }

    # One D range for every figure, so the five windows can be read against each other directly.
    all_log_d = np.concatenate(
        [f["log_D"].to_numpy() for per_set in fitted.values() for f in per_set.values()]
    )
    ylim = (float(all_log_d.min()) - 0.25, float(all_log_d.max()) + 0.25)

    results_dir().mkdir(parents=True, exist_ok=True)
    for window_ms, per_set in fitted.items():
        plot_cellset_strip(
            frames=per_set,
            window_ms=window_ms,
            estimator=cfg.estimator,
            well_sampled=cfg.well_sampled,
            ylim=ylim,
            save_path=results_dir()
            / f"diffusion_by_cellset_{window_ms}ms_{cfg.estimator}.png",
        )
        summary = "   ".join(
            f"{cs}: median {f['D'].median():.2f}" for cs, f in per_set.items()
        )
        print(f"    {window_ms / 1000:g} s -> {summary}")


def main() -> None:
    """Console-script entry point for `hd-cellset-strip`."""
    run(cfg=tyro.cli(CellSetStripConfig))


if __name__ == "__main__":
    main()
