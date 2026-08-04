"""Is the long-lag sub-diffusion real, or manufactured by the decoder and the estimator?"""

import dataclasses

import pandas as pd

from analysis import io, stats
from analysis.values import Values
from env import results_dir

QUESTION_ID = "diffusion2"
EXPERIMENTS = ("diffusion2_exp1",)  # synthetic free walk pushed through each session's own ring

#: Where the control's per-session table lives.
CONTROL_TABLE = "synthetic_ring_control"


@dataclasses.dataclass(frozen=True)
class Config:
    """Which sessions to control, and what counts as a trustworthy machine."""

    cell_set: str = "ADn"
    #: Restrict to these `<mouse>-<session>` specs; empty means every cached session.
    sessions: tuple[str, ...] = ()
    #: A session's machinery is "verified honest" when a free walk survives both the estimator and
    #: the decoder above this alpha. Only those sessions can carry a clean claim about dynamics.
    honest_alpha: float = 0.9
    seed: int = 0


def collect(*, cfg: Config) -> None:
    """Push a known-free walk through every session's refitted ring (~30 s per session)."""
    import sys

    sys.path.insert(0, str(results_dir().parent.parent / "scripts"))
    from synthetic_ring_control import ControlConfig
    from synthetic_ring_control import run as run_control

    run_control(cfg=ControlConfig(sessions=cfg.sessions, cell_set=cfg.cell_set, seed=cfg.seed))


def analyse(*, cfg: Config, values: Values) -> None:
    """Split the observed sub-diffusion into estimator, decoder and dynamics.

    The three synthetic columns are nested: `alpha_truth` is the free walk measured before any
    decoding, `alpha_clean` adds the ring's parameterisation, `alpha_noisy` adds realistic off-ring
    scatter. Differencing them attributes the shortfall from a free walk to each stage in turn, and
    whatever remains against the real data is the only part that could be dynamics.
    """
    path = results_dir() / f"{CONTROL_TABLE}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No control table at {path}. Run `hd-exp collect diffusion2` first (~20 min)."
        )
    frame = pd.read_csv(path)
    values.note("input_sessions", len(frame))

    frame["deficit"] = frame["alpha_real"] - frame["alpha_noisy"]
    frame["estimator_loss"] = 1.0 - frame["alpha_truth"]
    frame["decoder_loss"] = frame["alpha_truth"] - frame["alpha_noisy"]

    values.scalar("N_SESSIONS", len(frame), fmt="d")
    for column, name in (
        ("alpha_truth", "TRUTH"), ("alpha_clean", "CLEAN"),
        ("alpha_noisy", "NOISY"), ("alpha_real", "REAL"),
    ):
        values.scalar(f"ALPHA_{name}", float(frame[column].median()), fmt=".2f")

    # How the observed shortfall from a free walk divides between the three stages.
    values.scalar("LOSS_ESTIMATOR", float(frame["estimator_loss"].median()), fmt="+.3f")
    values.scalar("LOSS_DECODER", float(frame["decoder_loss"].median()), fmt="+.3f")
    values.scalar("LOSS_DYNAMICS", float(-frame["deficit"].median()), fmt="+.3f")

    # The estimator only fails where the angle has fully decorrelated; name those sessions.
    blind = frame[frame["alpha_truth"] < cfg.honest_alpha]
    values.scalar("N_ESTIMATOR_BLIND", len(blind), fmt="d")
    values.scalar("ESTIMATOR_BLIND_SESSIONS", ", ".join(blind["session_id"]) or "none")
    values.table(
        "BLIND_TABLE",
        blind[["session_id", "n_cells", "alpha_truth", "alpha_noisy", "alpha_real"]],
        floatfmt=".2f",
    )

    values.scalar("DEFICIT_MEDIAN", float(frame["deficit"].median()), fmt="+.3f")
    values.scalar("DEFICIT_LO", float(frame["deficit"].quantile(0.25)), fmt="+.3f")
    values.scalar("DEFICIT_HI", float(frame["deficit"].quantile(0.75)), fmt="+.3f")
    values.scalar("DEFICIT_N_NEGATIVE", int((frame["deficit"] < 0).sum()), fmt="d")

    sign = stats.paired_difference(a=frame["alpha_real"], b=frame["alpha_noisy"])
    values.scalar("DEFICIT_P", sign["p"], fmt=".1e")

    # A quality artefact would track cell count or speed. Neither does.
    for column, name in (("n_cells", "CELLS"), ("d_target", "SPEED")):
        corr = stats.correlation(x=frame[column], y=frame["deficit"])
        values.scalar(f"DEFICIT_{name}_RHO", corr["rho"])
        values.scalar(f"DEFICIT_{name}_P", corr["p"], fmt=".2f")

    # Restricted to sessions where neither the estimator nor the decoder is doing damage.
    honest = frame[
        (frame["alpha_truth"] > cfg.honest_alpha) & (frame["alpha_noisy"] > cfg.honest_alpha)
    ]
    values.scalar("N_HONEST", len(honest), fmt="d")
    values.scalar("HONEST_DEFICIT_MEDIAN", float(honest["deficit"].median()), fmt="+.3f")
    values.scalar("HONEST_N_NEGATIVE", int((honest["deficit"] < 0).sum()), fmt="d")
    values.scalar(
        "HONEST_P",
        stats.paired_difference(a=honest["alpha_real"], b=honest["alpha_noisy"])["p"], fmt=".1e",
    )

    shown = frame.sort_values("deficit")[
        ["session_id", "n_cells", "alpha_real", "alpha_truth", "alpha_noisy", "deficit"]
    ]
    values.table("PER_SESSION", shown, floatfmt=".3f")

    io.save_table(frame=frame, name=f"{QUESTION_ID}_control")
