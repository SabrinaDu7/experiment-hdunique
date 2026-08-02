"""Variance decomposition repeated across fit windows, as a robustness check.

    uv run hd-variance-by-window

Fits log(D) ~ 1 + (1|mouse) once per origin-forced fit window (200/300/400/500 ms) at the standard
ADn gate, and writes one CSV row per window.

The wider windows are a **saturation diagnostic, not alternative estimates of D** (see
docs/rem-diffusion.md). The question this answers is narrow: does the *ratio* of the variance
components depend on which window D is read from? An ICC that moved across windows would mean the
saturation is distributed unevenly across mice; a flat ICC means the decomposition is not an
artefact of the window choice.
"""

import dataclasses

import pandas as pd
import tyro

from hdunique import variance
from hdunique.config import DANDI_MICE, FIT_WINDOWS_MS
from hdunique.env import results_dir
from hdunique.sweep import load_all_mice


@dataclasses.dataclass(frozen=True)
class WindowSweepConfig:
    """Gate and bootstrap settings, matched to the headline decomposition."""

    min_adn_cells: int = 15
    cell_set: str = "ADn"
    mice: tuple[int, ...] = DANDI_MICE
    n_bootstrap: int = 2000
    seed: int = 0


def summarise(*, df: pd.DataFrame, window_ms: int, cfg: "WindowSweepConfig") -> dict[str, object]:
    """Fit the LMM for one window and return a single summary row."""
    comp = variance.variance_components(result=variance.fit_lmm(df=df))
    boot = variance.bootstrap_components(df=df, n_boot=cfg.n_bootstrap, seed=cfg.seed)
    lo, hi = variance.component_cis(boot=boot)["icc"]
    return {
        "window_ms": window_ms,
        "n_sessions": len(df),
        "n_mice": df["mouse"].nunique(),
        "tau2": comp["tau2_between_mouse"],
        "sigma2": comp["sigma2_within_mouse"],
        "ICC": comp["icc"],
        "ICC_lo": lo,
        "ICC_hi": hi,
        "ANOVA_ICC": variance.anova_icc(df=df),
    }


def run(*, cfg: WindowSweepConfig) -> None:
    """Fit every window at the configured gate and write the CSV."""
    import warnings

    warnings.filterwarnings("ignore")
    raw = load_all_mice(mice=cfg.mice)

    rows = []
    for window_ms in FIT_WINDOWS_MS:
        gated = variance.gate_sessions(
            df=raw, cell_set=cfg.cell_set, min_cells=cfg.min_adn_cells, window_ms=window_ms
        )
        rows.append(summarise(df=gated, window_ms=window_ms, cfg=cfg))
        print(f"  done {window_ms} ms")

    out = pd.DataFrame(rows)
    with pd.option_context("display.float_format", lambda v: f"{v:.3f}", "display.width", 200):
        print(
            f"\n=== LMM variance decomposition by fit window "
            f"({cfg.cell_set}, >={cfg.min_adn_cells} ADn, n_bootstrap={cfg.n_bootstrap}) ==="
        )
        print(out.to_string(index=False))

    path = results_dir() / f"variance_by_window_{cfg.cell_set}_min{cfg.min_adn_cells}.csv"
    out.to_csv(path, index=False)
    print(f"\nsaved {path}")


def main() -> None:
    run(cfg=tyro.cli(WindowSweepConfig))


if __name__ == "__main__":
    main()
