"""How does the REM diffusion constant depend on the measurement window?"""

import dataclasses

import numpy as np
import pandas as pd

from analysis import io, stats
from analysis.values import Values
from diffusion import window_slope
from env import figures_dir
from figures.strips import plot_grouped_strip

QUESTION_ID = "diffusion1"
EXPERIMENTS = (
    "diffusion1_exp1",  # D at each window, per session
    "diffusion1_exp2",  # the same, split by cell set
)


@dataclasses.dataclass(frozen=True)
class Config:
    """Which windows, which estimator, and which cell sets to compare."""

    #: Analysis cell set for exp1.
    cell_set: str = "ADn"
    #: Windows the diffusion constant is refitted over, in ms.
    windows_ms: tuple[int, ...] = (200, 500, 1000, 2000, 3000, 4000, 5000)
    #: The window treated as the reference the others are compared against.
    reference_ms: int = 200
    #: Estimator. `circular` (-2 ln <cos>) is the one that survives wrapping at long lags;
    #: `wrapped` is correct at short lags and has a hard ceiling of pi^2/3 beyond them.
    estimator: str = "circular"
    #: Cell sets compared in exp2.
    cell_sets: tuple[str, ...] = ("ADn", "ADn+PoS", "PoS")
    #: Windows shown side by side in the exp1 figure, on the analysis cell set alone.
    figure_windows_ms: tuple[int, ...] = (200, 1000, 5000)
    dt: float = 0.1
    #: Below this cell count a session is drawn with an open marker.
    well_sampled: int = 20


def collect(*, cfg: Config) -> None:
    """Build the long-lag curves for every cell set (delegates to the timescale sweep)."""
    from collect.timescale import TimescaleConfig
    from collect.timescale import run as run_timescale

    for cell_set in cfg.cell_sets:
        run_timescale(cfg=TimescaleConfig(cell_set=cell_set, make_plot=False))


def _fit_windows(*, frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Refit D at each window from each session's stored MSD curve.

    Refitting from the stored curve rather than the cache keeps this a seconds-long operation and
    guarantees every window reads the *same* curve — which is what makes the progression across
    windows a property of the curve's shape rather than of the fitting.
    """
    out = frame.copy()
    for window_ms in cfg.windows_ms:
        out[f"D_{window_ms}"] = [
            window_slope(curve=np.asarray(c, dtype=float), dt=cfg.dt, window_ms=window_ms)[0]
            for c in out[f"curve_{cfg.estimator}"]
        ]
    return out


def analyse(*, cfg: Config, values: Values) -> None:
    """Report D at every window, its dependence on session speed, and the cell-set comparison."""
    frame = _fit_windows(frame=io.load_timescale(cell_set=cfg.cell_set), cfg=cfg)
    values.note("input_sessions", len(frame))
    ref = f"D_{cfg.reference_ms}"

    # --- exp1: how D moves with the window ---
    per_window = pd.DataFrame(
        {
            "window_ms": cfg.windows_ms,
            "median_D": [frame[f"D_{w}"].median() for w in cfg.windows_ms],
            "ratio_to_reference": [
                (frame[f"D_{w}"] / frame[ref]).median() for w in cfg.windows_ms
            ],
        }
    )
    values.table("WINDOW_TABLE", per_window, floatfmt=".3f")
    values.scalar("REFERENCE_MS", cfg.reference_ms, fmt="d")
    values.scalar("N_SESSIONS", len(frame), fmt="d")

    longest = max(cfg.windows_ms)
    ratio = frame[f"D_{longest}"] / frame[ref]
    values.scalar("LONGEST_MS", longest, fmt="d")
    values.scalar("LONGEST_RATIO_MEDIAN", float(ratio.median()), fmt=".2f")
    values.scalar("LONGEST_RATIO_LO", float(ratio.quantile(0.25)), fmt=".2f")
    values.scalar("LONGEST_RATIO_HI", float(ratio.quantile(0.75)), fmt=".2f")
    short = frame[f"D_{500}"] / frame[ref] if 500 in cfg.windows_ms else ratio
    values.scalar("SHORT_RATIO_MEDIAN", float(short.median()), fmt=".2f")

    # Faster sessions should fall further if the cause is the angle exploring the whole ring.
    speed = stats.correlation(x=frame[ref], y=ratio)
    values.scalar("SPEED_RATIO_RHO", speed["rho"])
    values.scalar("SPEED_RATIO_P", speed["p"], fmt=".1e")

    # The dimensionless shape of the curve, immune to a multiplicative estimator bias.
    values.scalar("ALPHA_SHORT", float(frame[f"alpha_short_{cfg.estimator}"].median()), fmt=".2f")
    values.scalar("ALPHA_LONG", float(frame[f"alpha_long_{cfg.estimator}"].median()), fmt=".2f")
    values.scalar("ALPHA_SHORT_WRAPPED", float(frame["alpha_short_wrapped"].median()), fmt=".2f")
    values.scalar("ALPHA_LONG_WRAPPED", float(frame["alpha_long_wrapped"].median()), fmt=".2f")

    # Wrapping: how far into its own ceiling the naive estimator has run by the longest lag.
    values.scalar("SATURATION_MEDIAN", float(frame["wrapped_saturation"].median()), fmt=".2f")
    values.scalar("SATURATION_MAX", float(frame["wrapped_saturation"].max()), fmt=".2f")
    values.scalar("RESULTANT_MIN", float(frame["resultant_at_max_lag"].min()), fmt=".3f")
    values.scalar("RESULTANT_MAX", float(frame["resultant_at_max_lag"].max()), fmt=".3f")

    per_session = frame[["session_id", "n_cells", ref, f"D_{longest}"]].copy()
    per_session["ratio"] = ratio
    values.table("PER_SESSION", per_session.sort_values("ratio"), floatfmt=".3f")

    # Every session's D at each window, side by side on one axis. The panels share a D scale, so
    # the compression from short to long windows is the thing the figure shows.
    windows = {
        f"{w / 1000:g} s" if w >= 1000 else f"{w} ms": io.with_log_d(frame=frame, column=f"D_{w}")
        for w in cfg.figure_windows_ms
    }
    path = figures_dir() / f"{QUESTION_ID}_exp1_windows_{cfg.cell_set}.png"
    plot_grouped_strip(
        panels=windows,
        group="mouse",
        title=f"REM diffusion by measurement window ({cfg.cell_set} only, "
              f"{cfg.estimator} estimator)",
        well_sampled=cfg.well_sampled,
        label_prefix="Mouse",
        save_path=path,
    )
    values.figure(
        "FIG_WINDOWS", path,
        caption=f"D at each measurement window, {cfg.cell_set} cells only",
    )

    # --- exp2: the same windows, split by cell set ---
    frames = {}
    for cell_set in cfg.cell_sets:
        try:
            frames[cell_set] = _fit_windows(frame=io.load_timescale(cell_set=cell_set), cfg=cfg)
        except FileNotFoundError:
            continue
    rows = []
    for cell_set, sub in frames.items():
        row = {"cell_set": cell_set, "sessions": len(sub)}
        row |= {f"D_{w}": float(sub[f"D_{w}"].median()) for w in cfg.windows_ms}
        rows.append(row)
    values.table("CELLSET_TABLE", pd.DataFrame(rows), floatfmt=".2f")

    if {"ADn", "PoS"} <= frames.keys():
        for window_ms in (cfg.reference_ms, longest):
            gap = frames["PoS"][f"D_{window_ms}"].median() / frames["ADn"][f"D_{window_ms}"].median()
            values.scalar(f"POS_OVER_ADN_{window_ms}", float(gap), fmt=".1f")

    spreads = {
        cs: float(np.log(sub[f"D_{w}"]).quantile(0.75) - np.log(sub[f"D_{w}"]).quantile(0.25))
        for cs, sub in frames.items()
        for w in (cfg.reference_ms,)
    }
    values.scalar("ADN_IQR_SHORT", spreads.get("ADn", float("nan")), fmt=".2f")
    adn = frames["ADn"]
    values.scalar(
        "ADN_IQR_LONG",
        float(np.log(adn[f"D_{longest}"]).quantile(0.75) - np.log(adn[f"D_{longest}"]).quantile(0.25)),
        fmt=".2f",
    )

