"""How fast does the head-direction signal turn, in wake and in REM, across animals and sessions?"""

import dataclasses

import numpy as np
import pandas as pd

from analysis import io, stats
from analysis.values import Values
from core.config import DANDI_MICE
from core.env import results_dir
from decode import head_direction, loader
from metrics import angular_speed as angspeed

QUESTION_ID = "angular1"
EXPERIMENTS = (
    "angular1_exp1",  # mean angular speed per session, wake and REM
    "angular1_exp2",  # validation: the wake estimate against the measured head angle
)


@dataclasses.dataclass(frozen=True)
class Config:
    """Which sessions and cells, and the estimator's parameters."""

    cell_areas: tuple[str, ...] = ("ADn",)
    mice: tuple[int, ...] = DANDI_MICE
    #: Restrict to these `<mouse>-<session>` specs; empty means every session with usable data.
    sessions: tuple[str, ...] = ()
    #: Rate-bin width for the cross-correlograms, in seconds.
    dt: float = 0.02
    #: Head-direction bins for the tuning curves.
    n_hd_bins: int = 60
    #: A cell needs this peak rate in its tuning curve to be treated as head-direction tuned.
    min_peak_rate_hz: float = 1.0
    #: A pair needs its tuning curves to reach at least this |rho| somewhere, or the correlogram
    #: carries no angular information and the fit is unconstrained.
    min_rho: float = 0.3
    #: Minimum pairs for a session-level estimate to be reported.
    min_pairs: int = 3
    #: Widest lag the correlograms are computed to, in seconds. The fit narrows from here. Capped
    #: at head-turn timescales: beyond a second or so the correlogram carries slow shared rate
    #: drift, which the model cannot distinguish from a very slow sweep.
    max_lag_s: float = 1.0
    #: Rate variation slower than this is removed before correlating; 0 disables. Same confound.
    detrend_s: float = 5.0
    #: Angular-speed prior. See `metrics.angular_speed.SPEED_MODELS`.
    speed_model: str = "chi2_3"
    #: Smoothing applied to the measured head angle before differencing, in seconds.
    measured_smooth_s: float = 0.1
    #: States to estimate in. Wake is the one with a ground truth to check against.
    states: tuple[str, ...] = ("Awake", "REM")
    #: Lags at which the measured *net* angular speed is recorded, for the validation in exp2.
    net_taus_s: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
    #: The lag the validation compares at. The estimator reads net displacement over roughly this
    #: much of a window, so it is the like-for-like ground truth.
    validation_tau_s: float = 1.0


def _binned_rates(
    *, spike_times: dict[int, np.ndarray], epochs, dt: float
) -> tuple[np.ndarray, int]:
    """Spike counts per unit in fixed bins, concatenated across the epochs' bouts.

    Bouts are binned separately and then concatenated, so no bin straddles a gap. The
    cross-correlogram still pairs across a seam, which costs a handful of bins out of many
    thousands; the alternative of dropping them entirely would complicate the lag bookkeeping for
    no measurable gain.
    """
    blocks, unit_ids = [], sorted(spike_times)
    for start, end in zip(np.asarray(epochs.start), np.asarray(epochs.end), strict=True):
        edges = np.arange(float(start), float(end), dt)
        if len(edges) < 3:
            continue
        blocks.append(
            np.array([np.histogram(spike_times[u], bins=edges)[0] for u in unit_ids], dtype=float)
        )
    if not blocks:
        return np.zeros((len(unit_ids), 0)), 0
    return np.hstack(blocks), len(blocks)


def _session_estimate(*, cfg: Config, mouse: int, session: int) -> dict[str, object] | None:
    """Estimate mean angular speed in each state for one session, plus the wake ground truth."""
    data = loader.load_session(mouse=mouse, session=session)
    units = loader.get_units(data=data)
    selected, unit_ids = loader.select_units_by_area(units=units, areas=cfg.cell_areas)
    if len(unit_ids) < 2:
        return None

    wake = loader.load_state_epochs(data=data, state="Awake")
    if len(wake) == 0:
        return None

    # Tuning curves come from wake, where the head angle is measured. They are then applied to REM,
    # which is the whole point: the preferred directions are a property of the cells, not the state.
    hd = head_direction.head_direction(data=data).restrict(wake)
    hd_times = np.asarray(hd.index, dtype=float)
    if len(hd_times) < 100:
        return None
    spike_times = {u: np.asarray(selected[u].t, dtype=float) for u in unit_ids}
    curves, _ = angspeed.tuning_curves(
        spike_times=spike_times,
        angles=np.asarray(hd.values, dtype=float),
        times=hd_times,
        n_bins=cfg.n_hd_bins,
    )

    tuned = [
        i for i in range(len(curves))
        if np.isfinite(curves[i]).sum() > cfg.n_hd_bins // 2
        and np.nanmax(curves[i]) >= cfg.min_peak_rate_hz
    ]
    if len(tuned) < 2:
        return None

    row: dict[str, object] = {
        "mouse": mouse, "session": session, "session_id": f"Mouse{mouse}-{session}",
        "n_cells": len(unit_ids), "n_tuned": len(tuned),
    }

    for state in cfg.states:
        epochs = loader.load_state_epochs(data=data, state=state)
        prefix = state.lower()
        if len(epochs) == 0:
            row[f"{prefix}_speed"] = float("nan")
            row[f"{prefix}_pairs"] = 0
            continue
        rates, n_bouts = _binned_rates(spike_times=spike_times, epochs=epochs, dt=cfg.dt)
        widest = int(np.rint(cfg.max_lag_s / cfg.dt))
        if rates.shape[1] <= 2 * widest:
            row[f"{prefix}_speed"] = float("nan")
            row[f"{prefix}_pairs"] = 0
            continue

        if cfg.detrend_s > 0:
            rates = np.array([
                angspeed.highpass_rate(rate=r, dt=cfg.dt, cutoff_s=cfg.detrend_s) for r in rates
            ])

        # Every pair's correlogram, computed once at the widest lag; both estimators slice it.
        rhos, correlograms = [], []
        for a_i, a in enumerate(tuned):
            for b in tuned[a_i + 1 :]:
                rho = angspeed.angular_correlation(curve_a=curves[a], curve_b=curves[b])
                if not np.isfinite(rho).all() or np.nanmax(np.abs(rho)) < cfg.min_rho:
                    continue
                rhos.append(rho)
                correlograms.append(
                    angspeed.rate_cross_correlation(
                        rate_a=rates[a], rate_b=rates[b], max_lag_bins=widest
                    )
                )
        row[f"{prefix}_pairs"] = len(rhos)
        row[f"{prefix}_bouts"] = n_bouts
        if len(rhos) < cfg.min_pairs:
            row[f"{prefix}_speed"] = float("nan")
            row[f"{prefix}_speed_pairwise"] = float("nan")
            row[f"{prefix}_speed_iqr"] = float("nan")
            continue

        rhos, correlograms = np.array(rhos), np.array(correlograms)
        joint = angspeed.fit_population_speed(
            rhos=rhos, correlograms=correlograms, dt=cfg.dt,
            max_lag_s=cfg.max_lag_s, model=cfg.speed_model,
        )
        row[f"{prefix}_speed"] = joint.mean_speed

        # The per-pair estimator, kept for comparison. It is given the joint fit's window rather
        # than iterating its own, so the two differ only in whether the pairs are pooled.
        if np.isfinite(joint.mean_speed) and joint.mean_speed > 0:
            half = int(np.rint(np.clip(
                angspeed.TARGET_SWEEP_RAD / joint.mean_speed, 0.1, cfg.max_lag_s
            ) / cfg.dt))
            half = int(np.clip(half, 3, widest))
            lags_s = np.arange(-half, half + 1, dtype=float) * cfg.dt
            window = slice(widest - half, widest + half + 1)
            pairwise = [
                angspeed.fit_mean_speed(
                    rho=rho, observed=row_c[window], lags_s=lags_s, model=cfg.speed_model
                ).mean_speed
                for rho, row_c in zip(rhos, correlograms, strict=True)
            ]
            pairwise = np.array([v for v in pairwise if np.isfinite(v)])
        else:
            pairwise = np.array([])
        row[f"{prefix}_speed_pairwise"] = (
            float(np.median(pairwise)) if len(pairwise) >= cfg.min_pairs else float("nan")
        )
        row[f"{prefix}_speed_iqr"] = (
            float(np.subtract(*np.percentile(pairwise, [75, 25])))
            if len(pairwise) >= cfg.min_pairs
            else float("nan")
        )

    measured = head_direction.measured_speed_in(
        data=data, epochs=wake, smooth_s=cfg.measured_smooth_s, net_taus_s=cfg.net_taus_s
    )
    row["measured_wake_speed"] = measured["mean"]
    row["measured_wake_median"] = measured["median"]
    for tau in cfg.net_taus_s:
        row[f"measured_wake_net_{tau:g}"] = measured[f"net_{tau:g}"]
    row["measured_wake_net"] = measured[f"net_{cfg.validation_tau_s:g}"]
    return row


def collect(*, cfg: Config) -> None:
    """Estimate angular speed for every session and write the table this question analyses."""
    wanted = {io.parse_session_spec(s) for s in cfg.sessions}
    rows = []
    for mouse, session in loader.list_sessions():
        if mouse not in cfg.mice or (wanted and (mouse, session) not in wanted):
            continue
        try:
            row = _session_estimate(cfg=cfg, mouse=mouse, session=session)
        except Exception as exc:  # noqa: BLE001 - one bad session must not stop the sweep
            print(f"  skip Mouse{mouse}-{session}: {exc}")
            continue
        if row is None:
            continue
        rows.append(row)
        print(
            f"  {row['session_id']}: wake {row.get('awake_speed', float('nan')):.2f} "
            f"(net {row['measured_wake_net']:.2f}, path {row['measured_wake_speed']:.2f})  "
            f"REM {row.get('rem_speed', float('nan')):.2f} rad/s  [{row['n_tuned']} tuned cells]"
        )
    if not rows:
        print("No usable sessions.")
        return
    frame = pd.DataFrame(rows)
    path = io.save_table(frame=frame, name=f"{QUESTION_ID}_speeds")
    print(f"  -> {len(frame)} sessions in {path.name}")


def analyse(*, cfg: Config, values: Values) -> None:
    """Report angular speed by state and animal, and check the wake estimate against ground truth."""
    frame = pd.read_parquet(results_dir() / f"{QUESTION_ID}_speeds.parquet")
    values.note("input_sessions", len(frame))
    values.scalar("N_SESSIONS", len(frame), fmt="d")
    values.scalar("N_MICE", int(frame["mouse"].nunique()), fmt="d")

    # --- exp1: speed by state ---
    usable = frame[np.isfinite(frame["awake_speed"]) & np.isfinite(frame["rem_speed"])]
    values.scalar("N_BOTH_STATES", len(usable), fmt="d")
    for state in ("awake", "rem"):
        column = frame[f"{state}_speed"].dropna()
        values.scalar(f"{state.upper()}_MEDIAN", float(column.median()), fmt=".2f")
        values.scalar(f"{state.upper()}_LO", float(column.quantile(0.25)), fmt=".2f")
        values.scalar(f"{state.upper()}_HI", float(column.quantile(0.75)), fmt=".2f")

    ratio = usable["rem_speed"] / usable["awake_speed"]
    values.scalar("REM_OVER_WAKE", float(ratio.median()), fmt=".2f")
    paired = stats.paired_difference(a=usable["rem_speed"], b=usable["awake_speed"])
    values.scalar("PAIRED_P", paired["p"], fmt=".2g")
    values.scalar("PAIRED_N_REM_HIGHER", paired["n_positive"], fmt="d")

    per_mouse = (
        frame.groupby("mouse")
        .agg(sessions=("session", "size"), wake=("awake_speed", "median"),
             rem=("rem_speed", "median"), tuned=("n_tuned", "median"))
        .reset_index()
    )
    values.table("PER_MOUSE", per_mouse, floatfmt=".2f")
    values.table(
        "PER_SESSION",
        frame[
            [
                "session_id", "n_tuned", "awake_speed", "awake_speed_pairwise", "rem_speed",
                "measured_wake_net", "measured_wake_speed",
            ]
        ]
        .sort_values("session_id"),
        floatfmt=".2f",
    )

    # --- exp2: does the wake estimate match the measured angular speed? ---
    # The like-for-like comparison is against the *net* displacement rate, which is what the
    # correlogram model reads. The path-length speed is reported alongside as the upper bound.
    check = frame[np.isfinite(frame["awake_speed"]) & np.isfinite(frame["measured_wake_net"])]
    values.scalar("VALIDATION_N", len(check), fmt="d")
    values.scalar("VALIDATION_TAU", cfg.validation_tau_s, fmt=".2f")
    for label, column in (("NET", "measured_wake_net"), ("PATH", "measured_wake_speed")):
        corr = stats.correlation(x=check[column], y=check["awake_speed"])
        values.scalar(f"VALIDATION_{label}_RHO", corr["rho"])
        values.scalar(f"VALIDATION_{label}_P", corr["p"], fmt=".2g")
        scale = check["awake_speed"] / check[column]
        values.scalar(f"VALIDATION_{label}_RATIO", float(scale.median()), fmt=".2f")
        values.scalar(f"VALIDATION_{label}_RATIO_LO", float(scale.quantile(0.25)), fmt=".2f")
        values.scalar(f"VALIDATION_{label}_RATIO_HI", float(scale.quantile(0.75)), fmt=".2f")
        values.scalar(
            f"VALIDATION_{label}_WITHIN_2X", int(((scale > 0.5) & (scale < 2.0)).sum()), fmt="d"
        )

    # The measured net speed must itself fall with tau if net and path length differ as claimed.
    net_by_tau = pd.DataFrame(
        {
            "tau_s": list(cfg.net_taus_s),
            "measured_net_speed": [
                float(frame[f"measured_wake_net_{tau:g}"].median()) for tau in cfg.net_taus_s
            ],
        }
    )
    values.table("NET_BY_TAU", net_by_tau, floatfmt=".2f")

    # How the two quantities' spreads compare. A ratio well above 1 means the estimator's
    # session-to-session variation is its own noise, not variation it has detected.
    for label, column in (("EST", "awake_speed"), ("TRUTH", "measured_wake_net")):
        values.scalar(f"SPREAD_{label}_LO", float(check[column].min()), fmt=".2f")
        values.scalar(f"SPREAD_{label}_HI", float(check[column].max()), fmt=".2f")
        values.scalar(f"SPREAD_{label}_FOLD", float(check[column].max() / check[column].min()),
                      fmt=".0f")

    # Pooling the pairs was meant to fix the per-pair scatter; whether it did is a result.
    pairwise = frame[np.isfinite(frame["awake_speed_pairwise"])
                     & np.isfinite(frame["measured_wake_net"])]
    corr = stats.correlation(x=pairwise["measured_wake_net"], y=pairwise["awake_speed_pairwise"])
    values.scalar("PAIRWISE_RHO", corr["rho"])
    values.scalar("PAIRWISE_P", corr["p"], fmt=".2g")
    values.scalar(
        "PAIRWISE_SCATTER",
        float((pairwise["awake_speed_iqr"] / pairwise["awake_speed_pairwise"]).median()),
        fmt=".1f",
    )
    agreement = stats.correlation(x=pairwise["awake_speed"], y=pairwise["awake_speed_pairwise"])
    values.scalar("ESTIMATOR_AGREEMENT_RHO", agreement["rho"])

    # If the estimate tracked head speed, it should not track cell count instead.
    for label, column in (("CELLS", "n_tuned"), ("PAIRS", "awake_pairs")):
        corr = stats.correlation(x=check[column], y=check["awake_speed"])
        values.scalar(f"CONFOUND_{label}_RHO", corr["rho"])
        values.scalar(f"CONFOUND_{label}_P", corr["p"], fmt=".2g")

    # Does restricting to the best-sampled sessions rescue it? If the correlation does not improve
    # with sampling, the failure is not a power problem.
    gates = pd.DataFrame(
        [
            {
                "min_tuned_cells": gate,
                "sessions": int((check["n_tuned"] >= gate).sum()),
                "rho": stats.correlation(
                    x=check.loc[check["n_tuned"] >= gate, "measured_wake_net"],
                    y=check.loc[check["n_tuned"] >= gate, "awake_speed"],
                )["rho"],
            }
            for gate in (0, 10, 15, 20)
            if (check["n_tuned"] >= gate).sum() >= 5
        ]
    )
    values.table("VALIDATION_BY_GATE", gates, floatfmt=".3f")
