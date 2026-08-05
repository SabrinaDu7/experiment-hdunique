"""Entry point for the long-timescale diffusion comparison: D at 200 ms, 500 ms and 5 s.

    Called by an experiment's collect(); not a console script.
    uv run hd-timescale --cell-set PoS
    uv run hd-timescale --no-make-plot

Reads the decode cache only — no NWB loading, no Isomap, no ring fitting — so the whole sweep is a
couple of minutes. Writes one parquet per mouse plus a per-mouse figure.

The point of the exercise is that the 200 ms diffusion constant and the 5 s one need not agree, and
that comparing them is only meaningful once circular wrapping is dealt with. See
docs/long_D/2026-08-03-long-timescale-diffusion.md.
"""

import dataclasses

import numpy as np
import pandas as pd

from core.config import TIMESCALE_MAX_LAG, TIMESCALE_WINDOWS_MS
from core.env import results_dir
from decode.sweep import iter_cache
from figures import panels
from metrics import diffusion as dif
from metrics import timescale


@dataclasses.dataclass(frozen=True)
class TimescaleConfig:
    """Which cached runs to analyse, and over what lags."""

    #: Cell set to read from the cache. ADn is the analysis set.
    cell_set: str = "ADn"
    #: Decode bin width; must match the cached runs.
    dt: float = 0.1
    #: Longest lag, in bins.
    max_lag: int = TIMESCALE_MAX_LAG
    #: Fit windows, in ms.
    windows_ms: tuple[int, ...] = TIMESCALE_WINDOWS_MS
    #: Lag ranges (s) the short- and long-timescale anomalous exponents are fitted over.
    alpha_short_s: tuple[float, float] = (0.1, 0.5)
    alpha_long_s: tuple[float, float] = (1.0, 5.0)
    make_plot: bool = True


def timescale_row(
    *, cfg: TimescaleConfig, meta: dict[str, object], decoded: np.ndarray, bout_lengths: list[int]
) -> dict[str, object]:
    """One session's row: each estimator's MSD curve, and D fitted at each window from each.

    Curves are averaged over the cached refits before fitting, matching how the 200 ms sweep
    aggregates, so the only thing that differs from the published `D_bout_aware` is the lag range
    and the wrapping treatment.
    """
    lags = tuple(range(1, cfg.max_lag + 1))
    lags_s = np.arange(1, cfg.max_lag + 1, dtype=float) * cfg.dt
    row: dict[str, object] = {
        **{k: meta[k] for k in ("mouse", "session", "session_id", "cell_set", "n_cells", "n_adn")},
        "n_rem_bouts": len(bout_lengths),
        "n_samples": int(sum(bout_lengths)),
        "unwrap_risk": timescale.unwrap_risk(angles=decoded[0], bout_lengths=bout_lengths),
        "wrapped_ceiling": timescale.WRAPPED_CEILING,
        "kurtosis_200": timescale.displacement_kurtosis(
            angles=decoded[0], bout_lengths=bout_lengths, lag=2
        ),
        "kurtosis_max_lag": timescale.displacement_kurtosis(
            angles=decoded[0], bout_lengths=bout_lengths, lag=cfg.max_lag
        ),
    }

    curves: dict[str, np.ndarray] = {}
    for method in timescale.METHODS:
        per_refit = [
            timescale.msd_curve(
                angles=trace, bout_lengths=bout_lengths, lags=lags, method=method
            )
            for trace in decoded
        ]
        curve = np.mean(per_refit, axis=0)
        curves[method] = curve
        row[f"curve_{method}"] = [float(v) for v in curve]
        # Dimensionless shape of the curve, so a multiplicative estimator bias cancels.
        row[f"alpha_short_{method}"] = timescale.anomalous_exponent(
            curve=curve, lags_s=lags_s, lo_s=cfg.alpha_short_s[0], hi_s=cfg.alpha_short_s[1]
        )
        row[f"alpha_long_{method}"] = timescale.anomalous_exponent(
            curve=curve, lags_s=lags_s, lo_s=cfg.alpha_long_s[0], hi_s=cfg.alpha_long_s[1]
        )
        for window_ms in cfg.windows_ms:
            d, r2 = dif.window_slope(curve=curve, dt=cfg.dt, window_ms=window_ms)
            row[f"D_{window_ms}_{method}"] = d
            row[f"r2_{window_ms}_{method}"] = r2

    # How far the wrapped curve has run into its own ceiling by the longest lag, and how far the
    # two circumventions have parted company. Together these say whether 5 s is readable at all.
    row["wrapped_saturation"] = float(curves["wrapped"][-1] / timescale.WRAPPED_CEILING)
    row["unwrapped_over_circular"] = float(curves["unwrapped"][-1] / curves["circular"][-1])
    row["pairs_at_max_lag"] = timescale.pairs_per_lag(
        bout_lengths=bout_lengths, lags=(cfg.max_lag,)
    )[0]

    # Sensitivity floor shared by every estimator: how much angular memory survives to the
    # longest lag, and when it ran out. Averaged over refits like the curves themselves.
    row["resultant_at_max_lag"] = float(
        np.mean([
            timescale.resultant_length(angles=t, bout_lengths=bout_lengths, lag=cfg.max_lag)
            for t in decoded
        ])
    )
    decorr = int(np.median([
        timescale.decorrelation_lag(angles=t, bout_lengths=bout_lengths, lags=lags)
        for t in decoded
    ]))
    row["decorrelation_s"] = decorr * cfg.dt if decorr else float("inf")
    return row


def print_row(*, row: dict[str, object], windows_ms: tuple[int, ...]) -> None:
    """One line per session: D at each window under each circumvention, plus the two flags."""
    fields = "  ".join(
        f"{w}ms:{row[f'D_{w}_unwrapped']:.2f}/{row[f'D_{w}_circular']:.2f}" for w in windows_ms
    )
    print(
        f"  {row['session_id']} [{row['cell_set']}]  {fields}   "
        f"risk={row['unwrap_risk']:.3f}  sat={row['wrapped_saturation']:.2f}  "
        f"u/c={row['unwrapped_over_circular']:.1f}  R={row['resultant_at_max_lag']:.3f}  "
        f"α={row['alpha_short_circular']:.2f}→{row['alpha_long_circular']:.2f}"
    )


def run(*, cfg: TimescaleConfig) -> None:
    """Analyse every cached run of the requested cell set and write one parquet per mouse."""
    by_mouse: dict[int, list[dict[str, object]]] = {}
    for entry in iter_cache(cell_set=cfg.cell_set):
        row = timescale_row(
            cfg=cfg, meta=entry.meta, decoded=entry.decoded, bout_lengths=entry.bout_lengths
        )
        print_row(row=row, windows_ms=cfg.windows_ms)
        by_mouse.setdefault(int(entry.meta["mouse"]), []).append(row)

    if not by_mouse:
        print(f"No cached runs for cell_set={cfg.cell_set}.")
        return

    results_dir().mkdir(parents=True, exist_ok=True)
    for mouse, rows in sorted(by_mouse.items()):
        frame = pd.DataFrame(rows).sort_values("session").reset_index(drop=True)
        path = results_dir() / f"timescale_Mouse{mouse}_{cfg.cell_set}.parquet"
        frame.to_parquet(path, index=False)
        print(f"  -> {len(frame)} row(s) in {path.name}")
        if cfg.make_plot:
            panels.plot_timescale_curves(
                mouse=mouse,
                rows=rows,
                dt=cfg.dt,
                max_lag=cfg.max_lag,
                cell_set=cfg.cell_set,
                save_path=results_dir()
                / f"Mouse{mouse}_timescale_msd_{cfg.cell_set}.png",
            )

