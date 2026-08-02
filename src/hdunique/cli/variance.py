"""Entry point for the between-mouse vs between-session variance decomposition.

    uv run hd-variance                                    # default gate (>=15 ADn cells)
    uv run hd-variance --min-adn-cells 20 --no-sensitivity-ungated
    uv run hd-variance --window-ms 500

Reports tau^2 / sigma^2 / ICC with parametric-bootstrap CIs and a one-way ANOVA cross-check. With
`--sensitivity-ungated` the ungated fit is printed alongside, so the gate's effect on the
decomposition is visible rather than hidden. See docs/variance-decomposition.md.
"""

import warnings

import numpy as np
import pandas as pd
import tyro

from hdunique import variance
from hdunique.config import VarianceConfig
from hdunique.env import results_dir
from hdunique.plotting import plot_variance_summary
from hdunique.sweep import load_all_mice


def print_gated_table(*, df: pd.DataFrame, cfg: VarianceConfig) -> None:
    """Print how many sessions per mouse survived the gate, and their D range."""
    print(
        f"=== gated sessions: cell_set={cfg.cell_set}, n_adn >= {cfg.min_adn_cells}, "
        f"window={cfg.window_ms} ms, estimator={cfg.estimator} ===\n"
        f"  {len(df)} session(s) across {df['mouse'].nunique()} mice"
    )
    d_col = variance.d_column(window_ms=cfg.window_ms, estimator=cfg.estimator)
    counts = df.groupby("mouse").agg(
        n_sessions=(d_col, "size"), D_min=(d_col, "min"), D_max=(d_col, "max")
    )
    with pd.option_context("display.float_format", lambda v: f"{v:.3f}"):
        print(counts.to_string(), "\n")


def print_components(*, df: pd.DataFrame, cfg: VarianceConfig, label: str) -> None:
    """Fit the LMM and print tau^2 / sigma^2 / ICC with bootstrap CIs and the ANOVA cross-check."""
    comp = variance.variance_components(result=variance.fit_lmm(df=df))
    boot = variance.bootstrap_components(df=df, n_boot=cfg.n_bootstrap, seed=cfg.seed)
    cis = variance.component_cis(boot=boot)

    print(f"=== {label}: log(D) ~ 1 + (1 | mouse), REML ===")
    print(
        f"  {len(df)} sessions, {df['mouse'].nunique()} mice, "
        f"{len(boot)}/{cfg.n_bootstrap} bootstrap fits converged"
    )
    for key, name in [
        ("tau2_between_mouse", "between-mouse  tau^2"),
        ("sigma2_within_mouse", "within-mouse sigma^2"),
        ("icc", "ICC = tau^2/(tau^2+sigma^2)"),
    ]:
        lo, hi = cis[key]
        print(f"  {name:<28} {comp[key]:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"  {'ANOVA ICC (cross-check)':<28} {variance.anova_icc(df=df):.3f}")
    # Order-of-magnitude sanity check only: REML's precision-weighted grand mean means the
    # components do not decompose the raw marginal variance additively under an unbalanced design.
    print(
        f"  {'var(log_D), unpartitioned':<28} {np.var(df['log_D'], ddof=1):.3f}"
        f"   (vs tau^2+sigma^2 = {comp['total']:.3f})\n"
    )


def run(*, cfg: VarianceConfig) -> None:
    """Load the parquets, gate, fit, report and (optionally) plot."""
    warnings.filterwarnings("ignore")  # boundary tau^2=0 bootstrap replicates warn by design
    raw = load_all_mice(mice=cfg.mice)
    gated = variance.gate_sessions(
        df=raw, cell_set=cfg.cell_set, min_cells=cfg.min_adn_cells, window_ms=cfg.window_ms,
        estimator=cfg.estimator,
    )
    print_gated_table(df=gated, cfg=cfg)
    print_components(df=gated, cfg=cfg, label=f"GATED (n_adn >= {cfg.min_adn_cells})")

    if cfg.sensitivity_ungated:
        ungated = variance.gate_sessions(
            df=raw, cell_set=cfg.cell_set, min_cells=0, window_ms=cfg.window_ms,
            estimator=cfg.estimator,
        )
        print_components(df=ungated, cfg=cfg, label="SENSITIVITY: ungated (all sessions)")

    if cfg.make_plot:
        effects = variance.mouse_effects(result=variance.fit_lmm(df=gated), df=gated)
        with pd.option_context("display.float_format", lambda v: f"{v:.3f}"):
            print("=== per-mouse shrunken effects ===")
            print(effects.to_string(index=False), "\n")
        plot_variance_summary(
            df=gated,
            effects=effects,
            cfg=cfg,
            save_path=results_dir()
            / f"variance_by_mouse_{cfg.cell_set}_min{cfg.min_adn_cells}"
            f"_{cfg.window_ms}ms_{cfg.estimator}.png",
        )


def main() -> None:
    """Console-script entry point for `hd-variance`."""
    run(cfg=tyro.cli(VarianceConfig))


if __name__ == "__main__":
    main()
