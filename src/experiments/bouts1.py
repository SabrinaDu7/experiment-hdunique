"""Does the sleep-architecture context a REM bout sits in predict its diffusion constant?"""

import dataclasses

import numpy as np
import pandas as pd

from analysis import io, stats
from analysis.values import Values
from env import figures_dir
from figures.strips import plot_grouped_strip

QUESTION_ID = "bouts1"
EXPERIMENTS = (
    "bouts1_exp1",  # exit state, position in session, bout duration
    "bouts1_exp2",  # the exit-state contrast within individual sessions
)


@dataclasses.dataclass(frozen=True)
class Config:
    """Which bouts, which context variables, and which sessions to inspect individually."""

    cell_set: str = "ADn"
    d_column: str = "D_200"
    #: The state a bout exits to that defines the contrast; everything else is the other arm.
    exit_state: str = "Awake"
    #: Continuous context variables tested against D.
    context_columns: tuple[str, ...] = ("time_frac", "duration_s")
    #: Minimum bouts of each exit type for a session to carry within-session information.
    min_per_arm: int = 3
    #: Sessions drawn individually in the figure.
    plot_sessions: tuple[str, ...] = ("12-120808", "17-130129", "20-130515")


def collect(*, cfg: Config) -> None:
    """Build the per-bout table this question needs (delegates to the bout sweep)."""
    from collect.bouts import BoutConfig
    from collect.bouts import run as run_bouts

    run_bouts(cfg=BoutConfig(cell_set=cfg.cell_set))


def analyse(*, cfg: Config, values: Values) -> None:
    """Test every context variable both raw and with session identity controlled."""
    bouts = io.with_log_d(frame=io.load_bouts(cell_set=cfg.cell_set), column=cfg.d_column)
    bouts["to_exit"] = (bouts["next_state"] == cfg.exit_state).astype(int)
    values.note("input_bouts", len(bouts))

    # --- composition: which contrasts the data can actually support ---
    counts = bouts["next_state"].value_counts()
    values.table("EXIT_COUNTS", counts.reset_index(), floatfmt=".0f")
    values.scalar("N_BOUTS", len(bouts), fmt="d")
    entered = bouts["prev_state"].value_counts()
    values.scalar("ENTERED_FROM_TOP", str(entered.index[0]))
    values.scalar("ENTERED_FROM_TOP_N", int(entered.iloc[0]), fmt="d")
    values.scalar("ENTERED_FROM_OTHER_N", int(entered.iloc[1:].sum()), fmt="d")

    # --- exp1: the exit-state effect, three ways ---
    models = stats.effect_within_and_between(
        frame=bouts, outcome="log_D", predictor="to_exit", group="session_id"
    )
    values.table("EXIT_MODELS", models, floatfmt=".3f")
    values.scalar("EXIT_BETA_RAW", float(models.loc[models.model == "raw", "beta"].iloc[0]))
    values.scalar("EXIT_P_RAW", float(models.loc[models.model == "raw", "p"].iloc[0]), fmt=".2g")
    values.scalar(
        "EXIT_BETA_FIXED", float(models.loc[models.model == "group_fixed", "beta"].iloc[0])
    )
    values.scalar(
        "EXIT_P_FIXED", float(models.loc[models.model == "group_fixed", "p"].iloc[0]), fmt=".2f"
    )

    # which mice supply the minority arm at all -- the confound made concrete
    minority = bouts[bouts["to_exit"] == 0]
    values.table(
        "MINORITY_BY_MOUSE",
        minority["mouse"].value_counts().sort_index().reset_index(),
        floatfmt=".0f",
    )
    values.scalar("MINORITY_MICE", int(minority["mouse"].nunique()), fmt="d")
    values.scalar("TOTAL_MICE", int(bouts["mouse"].nunique()), fmt="d")

    # --- exp1 continued: continuous context, raw and session-demeaned ---
    centred = stats.demean_within(
        frame=bouts, columns=["log_D", *cfg.context_columns], group="session_id"
    )
    rows = []
    for column in cfg.context_columns:
        raw = stats.correlation(x=bouts[column], y=bouts["log_D"])
        dem = stats.correlation(x=centred[f"{column}_c"], y=centred["log_D_c"])
        rows.append(
            {"variable": column, "rho_raw": raw["rho"], "p_raw": raw["p"],
             "rho_demeaned": dem["rho"], "p_demeaned": dem["p"]}
        )
    values.table("CONTEXT_CORRELATIONS", pd.DataFrame(rows), floatfmt=".3f")
    values.scalar("CONTEXT_MAX_ABS_RHO", float(max(abs(r["rho_raw"]) for r in rows)))

    # --- exp2: the paired within-session contrast ---
    per_session = bouts.pivot_table(
        index="session_id", columns="to_exit", values=cfg.d_column, aggfunc="size"
    ).fillna(0)
    eligible = per_session[
        (per_session.get(0, 0) >= cfg.min_per_arm) & (per_session.get(1, 0) >= cfg.min_per_arm)
    ].index
    values.scalar("PAIRED_SESSIONS", len(eligible), fmt="d")
    values.scalar("PAIRED_SESSIONS_TOTAL", int(bouts["session_id"].nunique()), fmt="d")

    paired = (
        bouts[bouts["session_id"].isin(eligible)]
        .groupby(["session_id", "to_exit"])["log_D"].median().unstack()
    )
    result = stats.paired_difference(a=paired[0], b=paired[1])
    values.scalar("PAIRED_MEDIAN_DIFF", result["median_diff"])
    values.scalar("PAIRED_N_HIGHER", result["n_positive"], fmt="d")
    values.scalar("PAIRED_P", result["p"], fmt=".2f")

    # --- exp2: the named sessions, side by side ---
    rows = []
    for spec in cfg.plot_sessions:
        mouse, session = io.parse_session_spec(spec)
        sub = bouts[(bouts["mouse"] == mouse) & (bouts["session"] == session)]
        arms = {k: sub[sub["to_exit"] == v][cfg.d_column] for k, v in (("exit", 1), ("other", 0))}
        rows.append(
            {
                "session": f"Mouse{mouse}-{session}",
                "cells": int(sub["n_cells"].iloc[0]),
                f"n_{cfg.exit_state}": len(arms["exit"]),
                f"median_{cfg.exit_state}": float(arms["exit"].median()),
                "n_other": len(arms["other"]),
                "median_other": float(arms["other"].median()),
                "ratio": float(arms["other"].median() / arms["exit"].median()),
                "p": stats.rank_sum(a=arms["exit"], b=arms["other"])["p"],
            }
        )
    values.table("NAMED_SESSIONS", pd.DataFrame(rows), floatfmt=".2f")

    named = bouts[bouts["session_id"].isin(r["session"] for r in rows)].copy()
    named["exit"] = np.where(named["to_exit"] == 1, cfg.exit_state, "other")
    path = figures_dir() / f"{QUESTION_ID}_exp2_exit_state.png"
    plot_grouped_strip(
        panels={sid: named[named.session_id == sid] for sid in (r["session"] for r in rows)},
        group="exit",
        title=f"Per-bout REM diffusion split by the state each bout exits to ({cfg.cell_set})",
        statistic="median", count_col=None, save_path=path,
    )
    values.figure("FIG_EXIT_STATE", path,
                  caption="three sessions, bouts split by exit state")
    values.scalar("NAMED_MAX_RATIO", float(max(abs(np.log(r["ratio"])) for r in rows)), fmt=".2f")
