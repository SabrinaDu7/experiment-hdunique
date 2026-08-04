"""Is between-mouse variance in the REM diffusion constant larger than within-mouse variance?"""

import dataclasses

import pandas as pd

from analysis import io, stats
from analysis.values import Values
from config import DANDI_MICE, FIT_WINDOWS_MS
from env import figures_dir
from figures.strips import plot_grouped_strip
from variance import (
    anova_icc,
    bootstrap_components,
    component_cis,
    d_column,
    fit_lmm,
    gate_sessions,
    mouse_effects,
)

QUESTION_ID = "variance1"
EXPERIMENTS = (
    "variance1_exp1",  # ICC at each cell-count gate
    "variance1_exp2",  # is the ICC an artefact of the fit window?
    "variance1_exp3",  # does the mouse effect survive matching on cell count?
)


@dataclasses.dataclass(frozen=True)
class Config:
    """Which sessions enter the decomposition, and how its intervals are drawn."""

    #: Cell set to decompose. The ring is carried by ADn; PoS alone is near-noise.
    cell_set: str = "ADn"
    #: Which co-headline estimate of D to use.
    estimator: str = "D_bout_aware"
    #: Cell-count gates to report, in order. 0 means ungated.
    gates: tuple[int, ...] = (15, 20, 0)
    #: The gate whose numbers are the headline.
    headline_gate: int = 15
    #: Fit windows for the robustness check (exp2).
    windows_ms: tuple[int, ...] = FIT_WINDOWS_MS
    #: Cell-count band for the matched-subset test (exp3), inclusive.
    matched_band: tuple[int, int] = (14, 28)
    mice: tuple[int, ...] = DANDI_MICE
    #: Parametric-bootstrap replicates. Variance components are bounded at zero with skewed
    #: sampling distributions, so Wald intervals would mislead.
    n_bootstrap: int = 2000
    seed: int = 0


def _decompose(*, frame: pd.DataFrame, cfg: Config) -> dict[str, object]:
    """Fit the two-level model on one gated frame and summarise it with bootstrap intervals."""
    from variance import variance_components

    components = variance_components(result=fit_lmm(df=frame))
    boot = bootstrap_components(df=frame, n_boot=cfg.n_bootstrap, seed=cfg.seed)
    cis = component_cis(boot=boot)
    lo, hi = cis["icc"]
    return {
        "sessions": len(frame),
        "mice": int(frame["mouse"].nunique()),
        "tau2": components["tau2_between_mouse"],
        "sigma2": components["sigma2_within_mouse"],
        "ICC": components["icc"],
        "ICC_lo": lo,
        "ICC_hi": hi,
        "ANOVA_ICC": anova_icc(df=frame),
    }


def _exp1_gates(*, cfg: Config, raw: pd.DataFrame, values: Values) -> pd.DataFrame:
    """ICC at each cell-count gate, headline first."""
    rows = []
    for gate in cfg.gates:
        gated = gate_sessions(
            df=raw, cell_set=cfg.cell_set, min_cells=gate,
            window_ms=200, estimator=cfg.estimator,
        )
        rows.append({"gate": f">={gate} ADn" if gate else "ungated", **_decompose(frame=gated, cfg=cfg)})
        if gate == cfg.headline_gate:
            for key in ("tau2", "sigma2", "ICC", "ICC_lo", "ICC_hi", "ANOVA_ICC"):
                values.scalar(f"HEADLINE_{key.upper()}", rows[-1][key])
            values.scalar("HEADLINE_SESSIONS", rows[-1]["sessions"], fmt="d")
            values.scalar("HEADLINE_MICE", rows[-1]["mice"], fmt="d")
            effects = mouse_effects(result=fit_lmm(df=gated), df=gated)
            values.table("PER_MOUSE_EFFECTS", effects, floatfmt=".3f")
            path = figures_dir() / f"{QUESTION_ID}_exp1_by_mouse_min{gate}.png"
            plot_grouped_strip(
                panels={f"≥{gate} ADn cells": gated}, group="mouse",
                title=f"REM diffusion by mouse ({cfg.cell_set}, gated at ≥{gate} cells)",
                label_prefix="Mouse", save_path=path,
            )
            values.figure("FIG_BY_MOUSE", path,
                          caption=f"every gated session, grouped by mouse (≥{gate} cells)")
    frame = pd.DataFrame(rows)
    values.table("GATE_TABLE", frame, floatfmt=".3f")
    return frame


def _exp2_windows(*, cfg: Config, raw: pd.DataFrame, values: Values) -> None:
    """Does the ICC depend on which fit window's D is decomposed?"""
    rows = []
    for window_ms in cfg.windows_ms:
        gated = gate_sessions(
            df=raw, cell_set=cfg.cell_set, min_cells=cfg.headline_gate,
            window_ms=window_ms, estimator=cfg.estimator,
        )
        rows.append({"window_ms": window_ms, **_decompose(frame=gated, cfg=cfg)})
    frame = pd.DataFrame(rows)
    values.table("WINDOW_TABLE", frame, floatfmt=".3f")
    values.scalar("WINDOW_ICC_MIN", frame["ICC"].min())
    values.scalar("WINDOW_ICC_MAX", frame["ICC"].max())


def _exp3_matched(*, cfg: Config, raw: pd.DataFrame, values: Values) -> None:
    """Does the mouse effect survive matching on cell count?

    Cell count is the one quality variable that is unambiguously exogenous — fixed by the implant,
    and impossible for the dynamics to cause. Ring scatter and refit spread are not, so regressing
    them out would control for a partial consequence of D rather than a cause of it.
    """
    lo, hi = cfg.matched_band
    band = raw[(raw["n_adn"] >= lo) & (raw["n_adn"] <= hi)]
    band = io.with_log_d(frame=band, column=d_column(window_ms=200, estimator=cfg.estimator))

    summary = _decompose(frame=band, cfg=cfg)
    for key in ("tau2", "sigma2", "ICC", "ICC_lo", "ICC_hi", "ANOVA_ICC"):
        values.scalar(f"MATCHED_{key.upper()}", summary[key])
    values.scalar("MATCHED_SESSIONS", summary["sessions"], fmt="d")
    values.scalar("MATCHED_MICE", summary["mice"], fmt="d")
    values.scalar("MATCHED_BAND", f"{lo}-{hi}")

    d_col = d_column(window_ms=200, estimator=cfg.estimator)
    kruskal = stats.compare_groups(values=band[d_col], groups=band["mouse"])
    values.scalar("MATCHED_KRUSKAL_P", kruskal["p"], fmt=".4f")
    corr = stats.correlation(x=band["n_adn"], y=band[d_col])
    values.scalar("MATCHED_CELLS_D_RHO", corr["rho"])

    per_mouse = (
        band.groupby("mouse")
        .agg(sessions=(d_col, "size"), median_cells=("n_adn", "median"), median_D=(d_col, "median"))
        .reset_index()
        .sort_values("median_D")
    )
    values.table("MATCHED_PER_MOUSE", per_mouse, floatfmt=".2f")
    path = figures_dir() / f"{QUESTION_ID}_exp3_matched_band.png"
    plot_grouped_strip(
        panels={f"{lo}–{hi} ADn cells": band}, group="mouse",
        title=f"REM diffusion by mouse, cell count matched to {lo}–{hi} ({cfg.cell_set})",
        label_prefix="Mouse", save_path=path,
    )
    values.figure("FIG_MATCHED", path,
                  caption=f"the same comparison restricted to {lo}–{hi} ADn cells")
    values.scalar("MATCHED_D_SPREAD", per_mouse["median_D"].max() / per_mouse["median_D"].min(),
                  fmt=".1f")


def analyse(*, cfg: Config, values: Values) -> None:
    """Run all three experiments and record every number the results template needs."""
    raw = io.load_diffusion(cell_set=cfg.cell_set)
    values.note("input_sessions", len(raw))
    values.note("estimator", cfg.estimator)

    _exp1_gates(cfg=cfg, raw=raw, values=values)
    _exp2_windows(cfg=cfg, raw=raw, values=values)
    _exp3_matched(cfg=cfg, raw=raw, values=values)
