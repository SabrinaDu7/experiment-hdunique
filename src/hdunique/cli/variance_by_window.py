"""Variance decomposition repeated across fit windows, as a robustness check.

    uv run hd-variance-by-window

Fits log(D) ~ 1 + (1|mouse) once per origin-forced fit window (200/300/400/500 ms) at the standard
ADn gate, and writes one CSV row per window.

The wider windows are a **saturation diagnostic, not alternative estimates of D** (see
docs/porting/results-rem-diffusion.md). The question this answers is narrow: does the *ratio* of the variance
components depend on which window D is read from? An ICC that moved across windows would mean the
saturation is distributed unevenly across mice; a flat ICC means the decomposition is not an
artefact of the window choice.
"""

import warnings

import pandas as pd
import tyro

from hdunique import variance
from hdunique.config import FIT_WINDOWS_MS, VarianceGateConfig
from hdunique.env import results_dir
from hdunique.sweep import load_all_mice


def summarise(*, df: pd.DataFrame, window_ms: int, cfg: VarianceGateConfig) -> dict[str, object]:
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


def run(*, cfg: VarianceGateConfig) -> None:
    """Fit every window at the configured gate and write the CSV."""
    warnings.filterwarnings("ignore")  # boundary tau^2=0 bootstrap replicates warn by design
    raw = load_all_mice(mice=cfg.mice)

    rows = []
    for window_ms in FIT_WINDOWS_MS:
        gated = variance.gate_sessions(
            df=raw, cell_set=cfg.cell_set, min_cells=cfg.min_adn_cells, window_ms=window_ms,
            estimator=cfg.estimator,
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

    path = (
        results_dir()
        / f"variance_by_window_{cfg.cell_set}_min{cfg.min_adn_cells}_{cfg.estimator}.csv"
    )
    out.to_csv(path, index=False)
    print(f"\nsaved {path}")


def main() -> None:
    """Console-script entry point for `hd-variance-by-window`."""
    run(cfg=tyro.cli(VarianceGateConfig))


if __name__ == "__main__":
    main()
