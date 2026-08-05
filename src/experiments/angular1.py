"""How fast does the head-direction signal turn, in wake and in REM, across animals and sessions?"""

import dataclasses

import numpy as np
import pandas as pd
import pynapple as nap

from analysis import io, stats
from analysis.values import Values
from core.config import DANDI_MICE, DiffusionConfig
from core.env import cache_dir, results_dir
from decode import bout_decode, head_direction, loader
from figures import panels
from metrics import angular_speed as angspeed

QUESTION_ID = "angular1"
EXPERIMENTS = (
    "angular1_exp1",  # mean angular speed per session, wake and REM
    "angular1_exp2",  # validation: the wake estimate against the measured head angle
    "angular1_exp3",  # per-wake-bout decoders: is the decoded angular speed the measured one?
    "angular1_exp4",  # measured angular speed per wake bout, every animal and session
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

    # --- exp3: one decoder per wake bout ---
    #: Shortest wake bout given its own decoder. At dt=0.1 s this is 3000 rate bins, enough to
    #: embed and fit a ring; shorter bouts are dropped rather than fitted on too little.
    min_bout_s: float = 300.0
    #: Wake epochs closer together than this are merged first. 0 keeps DANDI's scoring as given.
    merge_gap_s: float = 0.0
    #: Train fraction and number of splits for each bout's held-out decode RMSE.
    train_frac: float = 0.8
    n_splits: int = 3
    #: Best-of-N ring per split, selected on the unsupervised spline cost. 5 is the value the
    #: published wake decode used; the sweep-wide default of 10 triples the cost for no measured
    #: gain here.
    n_restarts: int = 5
    #: A bout's decode must reach this RMSE against measured head direction to be called usable.
    #: 0.5 rad is the bar the published wake decode clears; chance is pi/sqrt(3) = 1.81.
    rmse_threshold: float = 0.5
    #: Box-smoothing applied to both speed traces before drawing them. Raw 10 Hz angular speed is
    #: dominated by bin-to-bin noise in measured and decoded alike.
    trace_smooth_s: float = 5.0
    seed: int = 0
    #: Which collection stages `collect` runs. "sessions" is the per-session sweep behind exp1/exp2
    #: (hours); "bouts" is the per-wake-bout decode behind exp3; "measured" is the tracking-only
    #: sweep behind exp4 (a minute, since it decodes nothing).
    stages: tuple[str, ...] = ("sessions", "bouts", "measured")
    #: exp4 reports every qualifying bout, with no cap - it costs nothing to include them all.
    measured_min_bout_s: float = 300.0


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


def _correlogram_speed(
    *, cfg: Config, spike_times: dict[int, np.ndarray], hd: nap.Tsd, start: float, end: float
) -> float:
    """The cell-pair estimator restricted to one bout, with that bout's own tuning curves."""
    epoch = nap.IntervalSet(start=[start], end=[end])
    restricted = hd.restrict(epoch)
    times = np.asarray(restricted.index, dtype=float)
    if len(times) < 500:
        return float("nan")
    curves, _ = angspeed.tuning_curves(
        spike_times=spike_times, angles=np.asarray(restricted.values, dtype=float),
        times=times, n_bins=cfg.n_hd_bins,
    )
    tuned = [
        i for i in range(len(curves))
        if np.isfinite(curves[i]).sum() > cfg.n_hd_bins // 2
        and np.nanmax(curves[i]) >= cfg.min_peak_rate_hz
    ]
    if len(tuned) < 2:
        return float("nan")

    rate_mat, _ = _binned_rates(spike_times=spike_times, epochs=epoch, dt=cfg.dt)
    widest = int(np.rint(cfg.max_lag_s / cfg.dt))
    if rate_mat.shape[1] <= 2 * widest:
        return float("nan")
    if cfg.detrend_s > 0:
        rate_mat = np.array([
            angspeed.highpass_rate(rate=r, dt=cfg.dt, cutoff_s=cfg.detrend_s) for r in rate_mat
        ])

    rhos, correlograms = [], []
    for a_i, a in enumerate(tuned):
        for b in tuned[a_i + 1 :]:
            rho = angspeed.angular_correlation(curve_a=curves[a], curve_b=curves[b])
            if not np.isfinite(rho).all() or np.nanmax(np.abs(rho)) < cfg.min_rho:
                continue
            rhos.append(rho)
            correlograms.append(
                angspeed.rate_cross_correlation(
                    rate_a=rate_mat[a], rate_b=rate_mat[b], max_lag_bins=widest
                )
            )
    if len(rhos) < cfg.min_pairs:
        return float("nan")
    return angspeed.fit_population_speed(
        rhos=np.array(rhos), correlograms=np.array(correlograms), dt=cfg.dt,
        max_lag_s=cfg.max_lag_s, model=cfg.speed_model,
    ).mean_speed


def _bout_rows(*, cfg: Config, mouse: int, session: int) -> tuple[list[dict], list[dict]]:
    """One row per long wake bout: its own decoder, and every angular speed measurable on it.

    Three quantities sit side by side for the same bout, which is what makes this diagnostic rather
    than merely descriptive:

    - **measured** — from the tracking LEDs. Ground truth.
    - **decoded** — the angular speed of this bout's own decoded ring angle. Tests the decoder.
    - **correlogram** — the Peyrache cell-pair estimator on the same bout. Tests that estimator.

    If decoded matches measured but correlogram does not, the fault is in the estimator and not in
    the decode. That is the question exp3 exists to settle. Returns (summary rows, trace rows).
    """
    decode_cfg = DiffusionConfig(
        mouse=mouse, session=session, cell_areas=cfg.cell_areas, n_restarts=cfg.n_restarts
    )
    data = loader.load_session(mouse=mouse, session=session)
    units = loader.get_units(data=data)
    selected, unit_ids = loader.select_units_by_area(units=units, areas=cfg.cell_areas)
    if len(unit_ids) < 3:
        return [], []

    wake = loader.load_state_epochs(data=data, state="Awake")
    bouts = loader.contiguous_bouts(
        epochs=wake, merge_gap_s=cfg.merge_gap_s, min_duration_s=cfg.min_bout_s
    )
    if len(bouts) == 0:
        return [], []

    hd = head_direction.head_direction(data=data)
    spike_times = {u: np.asarray(selected[u].t, dtype=float) for u in unit_ids}

    rows: list[dict] = []
    traces: list[dict] = []
    for index, (start, end) in enumerate(
        zip(np.asarray(bouts.start), np.asarray(bouts.end), strict=True)
    ):
        measured = bout_decode.binned_measured_angle(
            hd=hd, start=float(start), end=float(end), dt=decode_cfg.dt
        )
        result = bout_decode.decode_bout(
            units=selected, start=float(start), end=float(end), cfg=decode_cfg,
            measured=measured, train_frac=cfg.train_frac, n_splits=cfg.n_splits, seed=cfg.seed,
        )
        if result is None:
            continue

        kwargs = {"net_tau_s": cfg.validation_tau_s, "smooth_s": cfg.measured_smooth_s}
        true_speed = bout_decode.speed_summary(
            angles=result.measured, times=result.times, **kwargs
        )
        decoded_speed = bout_decode.speed_summary(
            angles=result.decoded, times=result.times, **kwargs
        )
        rows.append({
            "mouse": mouse, "session": session, "session_id": f"Mouse{mouse}-{session}",
            "bout_index": index, "bout_start": float(start), "duration_s": result.duration_s,
            "n_bins": result.n_bins, "n_cells": len(unit_ids),
            "rmse": result.rmse, "rmse_sd": result.rmse_sd,
            "usable": bool(result.rmse < cfg.rmse_threshold),
            "measured_path": true_speed["path"], "measured_net": true_speed["net"],
            "decoded_path": decoded_speed["path"], "decoded_net": decoded_speed["net"],
            "correlogram_speed": _correlogram_speed(
                cfg=cfg, spike_times=spike_times, hd=hd, start=float(start), end=float(end)
            ),
        })
        traces.append({
            "session_id": f"Mouse{mouse}-{session}", "bout_index": index,
            "times": result.times, "measured": result.measured, "decoded": result.decoded,
            "rmse": result.rmse, "usable": bool(result.rmse < cfg.rmse_threshold),
        })
    return rows, traces


def _traces_path(*, session_id: str):
    """Where one session's per-bout decoded and measured traces are cached."""
    return cache_dir() / f"{QUESTION_ID}_traces_{session_id}.npz"


def _save_traces(*, session_id: str, traces: list[dict]) -> None:
    """Cache the raw traces so the figure can be redrawn without re-running the decode.

    Each bout's decode costs minutes, so a figure that can only be changed by recomputing it is a
    figure that does not get changed.
    """
    cache_dir().mkdir(parents=True, exist_ok=True)
    flat = {}
    for entry in traces:
        key = entry["bout_index"]
        for field in ("times", "measured", "decoded"):
            flat[f"{key}_{field}"] = entry[field]
        flat[f"{key}_meta"] = np.array([entry["rmse"], float(entry["usable"])])
    np.savez_compressed(_traces_path(session_id=session_id), **flat)


def load_traces(*, session_id: str) -> list[dict]:
    """Read back what `_save_traces` wrote, in bout order."""
    path = _traces_path(session_id=session_id)
    if not path.exists():
        return []
    data = np.load(path)
    indices = sorted({int(k.split("_")[0]) for k in data.files})
    out = []
    for index in indices:
        rmse, usable = data[f"{index}_meta"]
        out.append({
            "bout_index": index, "times": data[f"{index}_times"],
            "measured": data[f"{index}_measured"], "decoded": data[f"{index}_decoded"],
            "rmse": float(rmse), "usable": bool(usable),
        })
    return out


def collect_bouts(*, cfg: Config) -> None:
    """Per-wake-bout decoders and speeds, for exp3. Also draws the trace figure per session."""
    wanted = {io.parse_session_spec(s) for s in cfg.sessions}
    rows: list[dict] = []
    for mouse, session in loader.list_sessions():
        if mouse not in cfg.mice or (wanted and (mouse, session) not in wanted):
            continue
        try:
            got, traces = _bout_rows(cfg=cfg, mouse=mouse, session=session)
        except Exception as exc:  # noqa: BLE001 - one bad session must not stop the sweep
            print(f"  skip Mouse{mouse}-{session}: {exc}")
            continue
        for row in got:
            print(
                f"  {row['session_id']} bout {row['bout_index']} "
                f"({row['duration_s']:.0f}s): RMSE {row['rmse']:.3f} "
                f"{'PASS' if row['usable'] else 'FAIL'}  "
                f"path measured {row['measured_path']:.2f} vs decoded {row['decoded_path']:.2f}  "
                f"correlogram {row['correlogram_speed']:.2f}",
                flush=True,
            )
        if traces:
            _save_traces(session_id=f"Mouse{mouse}-{session}", traces=traces)
        rows.extend(got)
    if not rows:
        print("No usable wake bouts.")
        return
    path = io.save_table(frame=pd.DataFrame(rows), name=f"{QUESTION_ID}_bouts")
    print(f"  -> {len(rows)} bouts in {path.name}")


def collect_measured(*, cfg: Config) -> None:
    """Measured angular speed of every wake bout in every session, for exp4.

    This needs only the tracking LEDs, so unlike the decode it covers the whole dataset in about a
    minute. It is the ground-truth answer to the question this whole topic asks — how fast does the
    head turn, per animal — against which every estimate is graded.
    """
    wanted = {io.parse_session_spec(s) for s in cfg.sessions}
    rows = []
    for mouse, session in loader.list_sessions():
        if mouse not in cfg.mice or (wanted and (mouse, session) not in wanted):
            continue
        try:
            data = loader.load_session(mouse=mouse, session=session)
            hd = head_direction.head_direction(data=data)
            bouts = loader.contiguous_bouts(
                epochs=loader.load_state_epochs(data=data, state="Awake"),
                merge_gap_s=cfg.merge_gap_s, min_duration_s=cfg.measured_min_bout_s,
            )
        except Exception as exc:  # noqa: BLE001 - one bad session must not stop the sweep
            print(f"  skip Mouse{mouse}-{session}: {exc}")
            continue

        for index, (start, end) in enumerate(
            zip(np.asarray(bouts.start), np.asarray(bouts.end), strict=True)
        ):
            inside = hd.restrict(nap.IntervalSet(start=[start], end=[end]))
            angles = np.asarray(inside.values, dtype=float)
            times = np.asarray(inside.index, dtype=float)
            if len(times) < 500:
                continue
            speeds = bout_decode.speed_summary(
                angles=angles, times=times,
                net_tau_s=cfg.validation_tau_s, smooth_s=cfg.measured_smooth_s,
            )
            rows.append({
                "mouse": mouse, "session": session, "session_id": f"Mouse{mouse}-{session}",
                "bout_index": index, "bout_start": float(start),
                "duration_s": float(end) - float(start), "n_samples": len(times),
                # Fraction of the bout the tracker actually saw. A bout mostly reconstructed from
                # a handful of samples is not a bout whose speed means much.
                "tracked_fraction": float(len(times) * np.median(np.diff(times))
                                          / (float(end) - float(start))),
                "measured_net": speeds["net"], "measured_path": speeds["path"],
            })
        if rows and rows[-1]["session_id"] == f"Mouse{mouse}-{session}":
            got = [r for r in rows if r["session_id"] == f"Mouse{mouse}-{session}"]
            print(f"  Mouse{mouse}-{session}: {len(got)} bouts, net "
                  f"{np.nanmin([r['measured_net'] for r in got]):.2f}-"
                  f"{np.nanmax([r['measured_net'] for r in got]):.2f} rad/s", flush=True)
    if not rows:
        print("No usable wake bouts.")
        return
    path = io.save_table(frame=pd.DataFrame(rows), name=f"{QUESTION_ID}_measured")
    print(f"  -> {len(rows)} bouts in {path.name}")


def collect(*, cfg: Config) -> None:
    """Run the requested collection stages. See `Config.stages`."""
    if "measured" in cfg.stages:
        collect_measured(cfg=cfg)
    if "bouts" in cfg.stages:
        collect_bouts(cfg=cfg)
    if "sessions" in cfg.stages:
        collect_sessions(cfg=cfg)


def collect_sessions(*, cfg: Config) -> None:
    """Estimate angular speed for every session and write the table exp1/exp2 analyse."""
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

    _exp3_bouts(cfg=cfg, values=values)
    _exp4_measured(cfg=cfg, values=values)


def _exp3_bouts(*, cfg: Config, values: Values) -> None:
    """Per-wake-bout decoders: does the decoded angular speed match the measured one?

    Each bout carries its own held-out decode RMSE, so for the first time a speed estimate can be
    conditioned on whether the decode it rests on is any good. That is what separates "the decoder
    is broken" from "the estimator built on top of it is broken".
    """
    path = results_dir() / f"{QUESTION_ID}_bouts.parquet"
    if not path.exists():
        values.scalar("BOUT_N", 0, fmt="d")
        values.scalar("BOUT_N_USABLE", 0, fmt="d")
        values.table("BOUT_TABLE", pd.DataFrame([{"note": "run: hd-exp collect angular1 --stages bouts"}]))
        for token in ("BOUT_RMSE_MEDIAN", "BOUT_DECODED_RATIO", "BOUT_CORRELOGRAM_RATIO",
                      "BOUT_RMSE_PASS", "BOUT_RMSE_FAIL", "BOUT_SPEED_ERR_PASS",
                      "BOUT_SPEED_ERR_FAIL", "BOUT_CORR_SPREAD", "BOUT_MEASURED_SPREAD"):
            values.scalar(token, float("nan"))
        values.scalar("BOUT_FIGURES", "(not generated)")
        return

    frame = pd.read_parquet(path)
    values.scalar("BOUT_N", len(frame), fmt="d")
    values.scalar("BOUT_N_USABLE", int(frame["usable"].sum()), fmt="d")
    values.scalar("BOUT_RMSE_MEDIAN", float(frame["rmse"].median()), fmt=".2f")

    passed, failed = frame[frame["usable"]], frame[~frame["usable"]]
    values.scalar("BOUT_RMSE_PASS", float(passed["rmse"].median()) if len(passed) else float("nan"))
    values.scalar("BOUT_RMSE_FAIL", float(failed["rmse"].median()) if len(failed) else float("nan"))

    # The headline comparison: on bouts whose decode is verified good, does decoded speed match
    # measured speed? And does the correlogram estimator, on those same bouts?
    #
    # Net speed, not path length. Path length is not a well-defined property of the head: on real
    # tracking it falls by a factor of 2.4 between 39 Hz and 2.4 Hz sampling, because finer sampling
    # accumulates more jitter. Net displacement at a fixed tau is invariant to that, so it is the
    # only one of the two that can grade an estimator. Both are carried in the table.
    for label, column in (("DECODED", "decoded_net"), ("CORRELOGRAM", "correlogram_speed")):
        ratio = frame[column] / frame["measured_net"]
        values.scalar(f"BOUT_{label}_RATIO",
                      float(ratio[frame["usable"]].median()) if len(passed) else float("nan"),
                      fmt=".2f")
    path_ratio = (frame["decoded_path"] / frame["measured_path"])[frame["usable"]]
    values.scalar("BOUT_DECODED_PATH_RATIO",
                  float(path_ratio.median()) if len(passed) else float("nan"), fmt=".2f")

    for label, subset in (("PASS", passed), ("FAIL", failed)):
        err = (subset["decoded_net"] - subset["measured_net"]).abs()
        values.scalar(f"BOUT_SPEED_ERR_{label}",
                      float(err.median()) if len(subset) else float("nan"), fmt=".2f")

    # A number that does not vary when the truth varies is not measuring the truth.
    for label, column in (("CORR", "correlogram_speed"), ("MEASURED", "measured_net")):
        col = frame[column].dropna()
        values.scalar(f"BOUT_{label}_SPREAD",
                      float(col.max() / col.min()) if len(col) and col.min() > 0 else float("nan"),
                      fmt=".1f")

    values.table(
        "BOUT_TABLE",
        frame[["session_id", "bout_index", "duration_s", "rmse", "usable",
               "measured_net", "decoded_net", "measured_path", "decoded_path",
               "correlogram_speed"]],
        floatfmt=".2f",
    )

    for session_id in frame["session_id"].unique():
        traces = load_traces(session_id=session_id)
        if not traces:
            continue
        figure = panels.bout_speed_traces(
            traces=traces, session_id=session_id, smooth_s=cfg.trace_smooth_s,
            name=f"{QUESTION_ID}_exp3_traces_{session_id}",
        )
        values.figure(f"FIG_BOUT_TRACE_{session_id.replace('-', '_')}", figure,
                      caption=f"{session_id}: measured and decoded angular speed, one decoder per "
                              "wake bout (green = decode passes the 0.5 rad bar)")


def _exp4_measured(*, cfg: Config, values: Values) -> None:
    """Measured angular speed per wake bout across every animal and session.

    The ground truth, standing on its own. It needs no decode and no estimator, so it is the one
    part of this question that is not contingent on anything working — and it is the yardstick every
    other number here is measured against.
    """
    path = results_dir() / f"{QUESTION_ID}_measured.parquet"
    tokens = ("MEASURED_N_BOUTS", "MEASURED_N_SESSIONS", "MEASURED_N_MICE", "MEASURED_MEDIAN",
              "MEASURED_LO", "MEASURED_HI", "MEASURED_WITHIN_SESSION_SPREAD",
              "MEASURED_BETWEEN_MOUSE_SPREAD", "MEASURED_KRUSKAL_P", "MEASURED_ICC",
              "MEASURED_VAR_MOUSE", "MEASURED_VAR_SESSION", "MEASURED_VAR_BOUT")
    if not path.exists():
        for token in tokens:
            values.scalar(token, float("nan"))
        values.table("MEASURED_PER_MOUSE", pd.DataFrame(
            [{"note": "run: hd-exp collect angular1 --stages measured"}]))
        return

    frame = pd.read_parquet(path)
    usable = frame[np.isfinite(frame["measured_net"])]
    values.scalar("MEASURED_N_BOUTS", len(usable), fmt="d")
    values.scalar("MEASURED_N_SESSIONS", int(usable["session_id"].nunique()), fmt="d")
    values.scalar("MEASURED_N_MICE", int(usable["mouse"].nunique()), fmt="d")
    values.scalar("MEASURED_MEDIAN", float(usable["measured_net"].median()), fmt=".2f")
    values.scalar("MEASURED_LO", float(usable["measured_net"].quantile(0.25)), fmt=".2f")
    values.scalar("MEASURED_HI", float(usable["measured_net"].quantile(0.75)), fmt=".2f")

    # Is the head speed a property of the animal, or does it vary just as much bout to bout?
    within = usable.groupby("session_id")["measured_net"].agg(lambda s: s.max() / s.min())
    values.scalar("MEASURED_WITHIN_SESSION_SPREAD", float(within.median()), fmt=".1f")
    per_mouse_median = usable.groupby("mouse")["measured_net"].median()
    values.scalar("MEASURED_BETWEEN_MOUSE_SPREAD",
                  float(per_mouse_median.max() / per_mouse_median.min()), fmt=".1f")
    kruskal = stats.compare_groups(values=usable["measured_net"], groups=usable["mouse"])
    values.scalar("MEASURED_KRUSKAL_P", kruskal["p"], fmt=".2g")

    # Three levels, not two: bouts sit inside sessions inside mice, and a two-level model that
    # treats bouts as independent draws from a mouse would attribute session structure to the
    # animal. Searle's nested ANOVA is exact here and needs no optimiser, unlike the LMM, which
    # does not converge on this design.
    components = stats.nested_variance(
        frame=usable.assign(log_speed=np.log(usable["measured_net"])),
        outcome="log_speed", outer="mouse", inner="session",
    )
    total = sum(max(0.0, components[k]) for k in ("var_outer", "var_inner", "var_resid"))
    for label, key in (("MOUSE", "var_outer"), ("SESSION", "var_inner"), ("BOUT", "var_resid")):
        values.scalar(f"MEASURED_VAR_{label}",
                      100.0 * max(0.0, components[key]) / total if total > 0 else float("nan"),
                      fmt=".1f")
    values.scalar("MEASURED_ICC",
                  max(0.0, components["var_outer"]) / total if total > 0 else float("nan"),
                  fmt=".3f")

    per_mouse = (
        usable.groupby("mouse")
        .agg(sessions=("session_id", "nunique"), bouts=("measured_net", "size"),
             median=("measured_net", "median"), lo=("measured_net", "min"),
             hi=("measured_net", "max"))
        .reset_index()
    )
    values.table("MEASURED_PER_MOUSE", per_mouse, floatfmt=".2f")

    decoded = None
    decode_path = results_dir() / f"{QUESTION_ID}_bouts.parquet"
    if decode_path.exists():
        decoded = pd.read_parquet(decode_path)
    figure = panels.bout_speed_by_animal(
        measured=usable, decoded=decoded, tau_s=cfg.validation_tau_s,
        name=f"{QUESTION_ID}_exp4_measured_speed_by_animal",
    )
    values.figure("FIG_MEASURED_BY_ANIMAL", figure,
                  caption="Measured net angular speed of every wake bout, by animal and session "
                          "(open circles: decoded speed where the decode is verified)")
