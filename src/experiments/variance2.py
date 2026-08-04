"""How does variance in D partition across bouts, sessions and mice?"""

import dataclasses

import numpy as np
import pandas as pd

from analysis import io, stats
from analysis.values import Values
from env import figures_dir
from figures.strips import plot_grouped_strip
from sweep import iter_cache

QUESTION_ID = "variance2"
EXPERIMENTS = (
    "variance2_exp1",  # three-level nested decomposition
    "variance2_exp2",  # how much of the bout level is decode quality?
)


@dataclasses.dataclass(frozen=True)
class Config:
    """Which bouts enter the decomposition, and which quality measures are regressed out."""

    cell_set: str = "ADn"
    #: Per-bout diffusion constant to decompose.
    d_column: str = "D_200"
    #: Fraction of a bout's embedded points counted as "off ring" below this share of the
    #: session's median radius.
    off_ring_frac: float = 0.6
    #: Quality measures for exp2. All are independent of the angular dynamics; `nugget` is
    #: deliberately excluded because it comes from the same curve as D.
    quality_columns: tuple[str, ...] = ("ring_cv", "ring_inward", "D_std")


def collect(*, cfg: Config) -> None:
    """Build the per-bout table this question needs (delegates to the bout sweep)."""
    from collect.bouts import BoutConfig
    from collect.bouts import run as run_bouts

    run_bouts(cfg=BoutConfig(cell_set=cfg.cell_set))


def ring_quality(*, cell_set: str, off_ring_frac: float) -> pd.DataFrame:
    """Per-bout decode quality, measured from the embedding rather than the decoded angle.

    Two measures: how tightly the bout's points hug the ring (radial coefficient of variation), and
    what fraction fall well inside it. Both describe how well the manifold was sampled during that
    bout and are independent of how fast the angle moved — unlike `nugget`, which is derived from
    the same diffusion curve as D and so cannot be used to explain it.
    """
    rows = []
    for entry in iter_cache(cell_set=cell_set):
        embed = np.asarray(entry.embed, dtype=float)
        radius = np.linalg.norm(embed - embed.mean(axis=0), axis=1)
        threshold = off_ring_frac * float(np.median(radius))
        offset = 0
        for index, length in enumerate(entry.bout_lengths):
            block = radius[offset : offset + length]
            offset += length
            rows.append(
                {
                    "session_id": entry.meta["session_id"],
                    "bout_index": index,
                    "ring_cv": float(block.std() / block.mean()),
                    "ring_inward": float(np.mean(block < threshold)),
                }
            )
    return pd.DataFrame(rows)


def _record(*, values: Values, prefix: str, parts: dict[str, float]) -> None:
    """Record one nested decomposition under a common prefix."""
    for key in ("var_outer", "var_inner", "var_resid"):
        values.scalar(f"{prefix}_{key.upper()}", parts[key])
        values.scalar(f"{prefix}_{key.upper()}_PCT", 100 * parts[f"frac_{key.split('_')[1]}"],
                      fmt=".1f")
    values.scalar(f"{prefix}_ICC_MOUSE", parts["icc_outer"])
    values.scalar(f"{prefix}_ICC_MOUSE_SESSION", parts["icc_outer_inner"])


def analyse(*, cfg: Config, values: Values) -> None:
    """Decompose bout-level D across three levels, then ask how much is decode quality."""
    bouts = io.with_log_d(frame=io.load_bouts(cell_set=cfg.cell_set), column=cfg.d_column)
    values.note("input_bouts", len(bouts))

    # --- exp1: the three-level split ---
    raw = stats.nested_variance(frame=bouts, outcome="log_D", outer="mouse", inner="session_id")
    _record(values=values, prefix="RAW", parts=raw)
    for key, name in (("n", "BOUTS"), ("n_inner", "SESSIONS"), ("n_outer", "MICE")):
        values.scalar(f"N_{name}", raw[key], fmt="d")

    within = bouts.groupby("session_id")["log_D"].std()
    spread = bouts.groupby("session_id")[cfg.d_column].agg(lambda x: x.max() / x.min())
    values.scalar("WITHIN_SESSION_SD", float(within.median()))
    values.scalar("WITHIN_SESSION_SPREAD", float(spread.median()), fmt=".1f")
    values.scalar("WITHIN_SESSION_SPREAD_MAX", float(spread.max()), fmt=".0f")

    path = figures_dir() / f"{QUESTION_ID}_exp1_bouts_by_mouse.png"
    plot_grouped_strip(
        panels={"every REM bout": bouts}, group="mouse",
        title=f"Per-bout REM diffusion ({cfg.cell_set}) — spread within a column is "
              "within-mouse, and includes the bout level",
        label_prefix="Mouse", statistic="median", save_path=path,
    )
    values.figure("FIG_BOUTS_BY_MOUSE", path, caption="every bout, grouped by mouse")

    # --- exp2: how much of the bout level is decode quality? ---
    quality = ring_quality(cell_set=cfg.cell_set, off_ring_frac=cfg.off_ring_frac)
    merged = bouts.merge(quality, on=["session_id", "bout_index"], how="left")
    centred = stats.demean_within(
        frame=merged, columns=["log_D", *cfg.quality_columns], group="session_id"
    )

    rows = []
    for column in cfg.quality_columns:
        corr = stats.correlation(x=centred[f"{column}_c"], y=centred["log_D_c"])
        rows.append({"measure": column, "rho_within_session": corr["rho"], "p": corr["p"]})
    values.table("QUALITY_CORRELATIONS", pd.DataFrame(rows), floatfmt=".3f")

    explained = stats.residualise(
        frame=centred, outcome="log_D_c", predictors=[f"{c}_c" for c in cfg.quality_columns]
    )
    values.scalar(
        "QUALITY_VARIANCE_EXPLAINED",
        100 * (1 - float(explained.var()) / float(centred["log_D_c"].var())), fmt=".1f",
    )

    merged["resid"] = stats.residualise(
        frame=merged, outcome="log_D", predictors=list(cfg.quality_columns)
    )
    corrected = stats.nested_variance(
        frame=merged, outcome="resid", outer="mouse", inner="session_id"
    )
    _record(values=values, prefix="CORRECTED", parts=corrected)

    per_mouse = (
        merged.groupby("mouse")
        .agg(cells=("n_cells", "median"), ring_cv=("ring_cv", "median"),
             off_ring=("ring_inward", "median"), D=(cfg.d_column, "median"))
        .reset_index()
    )
    values.table("QUALITY_BY_MOUSE", per_mouse, floatfmt=".3f")
    values.scalar(
        "CELLS_QUALITY_RHO",
        stats.correlation(x=merged["n_cells"], y=merged["ring_cv"])["rho"],
    )
