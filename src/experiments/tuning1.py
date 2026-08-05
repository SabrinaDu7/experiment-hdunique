"""Does head-direction tuning govern decode quality, and which ADn cells are reliably tuned?"""

import dataclasses

import numpy as np
import pandas as pd
import pynapple as nap

from analysis import io, stats
from analysis.values import Values
from core.config import DANDI_MICE, DiffusionConfig
from core.env import results_dir
from decode import bout_decode, head_direction, loader
from metrics import hd_tuning

QUESTION_ID = "tuning1"
EXPERIMENTS = (
    "tuning1_exp1",  # which ADn cells are reliably head-direction tuned during wake
    "tuning1_exp2",  # does a bout's tuning quality predict its decode error
    "tuning1_exp3",  # does decoding from the reliable cells alone beat the whole population
)


@dataclasses.dataclass(frozen=True)
class Config:
    """Which bouts and cells enter, and where the bar for "reliably tuned" sits."""

    cell_areas: tuple[str, ...] = ("ADn",)
    mice: tuple[int, ...] = DANDI_MICE
    #: Restrict to these `<mouse>-<session>` specs; empty means every session.
    sessions: tuple[str, ...] = ()

    # --- which wake bouts ---
    #: Shortest wake bout considered. Below this the tuning curve rests on too little occupancy.
    min_bout_s: float = 300.0
    merge_gap_s: float = 0.0
    #: Cap on bouts per session, longest first. The per-bout decode is minutes; without a cap the
    #: sweep is tens of hours and the extra bouts are the shortest and least informative.
    max_bouts_per_session: int = 4

    # --- tuning ---
    n_hd_bins: int = 60
    min_occupancy_s: float = 0.5
    n_shuffles: int = 200
    #: A cell counts as tuned in a bout when it clears all three bars. Strength alone is not enough:
    #: a cell that fires one burst while the animal faces one way scores high and is not tuned.
    min_mvl: float = 0.2
    max_shuffle_p: float = 0.01
    min_reliability: float = 0.3
    #: A cell is *reliably* tuned in a session when it is tuned in at least this fraction of the
    #: session's bouts and points the same way in them.
    min_bout_fraction: float = 0.5
    min_preferred_consistency: float = 0.8

    # --- decode ---
    train_frac: float = 0.8
    n_splits: int = 3
    n_restarts: int = 5
    rmse_threshold: float = 0.5
    #: Also decode from the reliably-tuned subset alone (exp3). Doubles the decode cost.
    decode_reliable_subset: bool = True
    #: A subset decode needs at least this many cells to be worth attempting.
    min_subset_cells: int = 5

    seed: int = 0
    #: Stages `collect` runs: "cells" is tuning only (minutes); "decode" adds the per-bout decode.
    stages: tuple[str, ...] = ("cells", "decode")


def _wake_bouts(*, cfg: Config, data: nap.NWBFile) -> nap.IntervalSet:
    """The session's longest qualifying wake bouts, in time order."""
    bouts = loader.contiguous_bouts(
        epochs=loader.load_state_epochs(data=data, state="Awake"),
        merge_gap_s=cfg.merge_gap_s,
        min_duration_s=cfg.min_bout_s,
    )
    start, end = np.asarray(bouts.start, float), np.asarray(bouts.end, float)
    if len(start) <= cfg.max_bouts_per_session:
        return bouts
    keep = np.sort(np.argsort(end - start)[::-1][: cfg.max_bouts_per_session])
    return nap.IntervalSet(start=start[keep], end=end[keep])


def _cell_rows(*, cfg: Config, mouse: int, session: int) -> list[dict]:
    """Per-cell, per-bout tuning for one session. No decoding, so this is cheap."""
    data = loader.load_session(mouse=mouse, session=session)
    selected, unit_ids = loader.select_units_by_area(units=data["units"], areas=cfg.cell_areas)
    if not unit_ids:
        return []
    bouts = _wake_bouts(cfg=cfg, data=data)
    if len(bouts) == 0:
        return []

    hd = head_direction.head_direction(data=data)
    rows = []
    for index, (start, end) in enumerate(
        zip(np.asarray(bouts.start), np.asarray(bouts.end), strict=True)
    ):
        window = nap.IntervalSet(start=[start], end=[end])
        inside = hd.restrict(window)
        angles = np.asarray(inside.values, dtype=float)
        times = np.asarray(inside.index, dtype=float)
        if len(times) < 500:
            continue
        for unit in unit_ids:
            spikes = np.asarray(selected[unit].t, dtype=float)
            tuning = hd_tuning.cell_tuning(
                spikes=spikes, angles=angles, times=times,
                n_bins=cfg.n_hd_bins, min_occupancy_s=cfg.min_occupancy_s,
                n_shuffles=cfg.n_shuffles, seed=cfg.seed + unit,
            )
            rows.append({
                "mouse": mouse, "session": session, "session_id": f"Mouse{mouse}-{session}",
                "unit": int(unit), "bout_index": index,
                "bout_start": float(start), "duration_s": float(end) - float(start),
                **dataclasses.asdict(tuning),
            })
    return rows


def _is_tuned(*, frame: pd.DataFrame, cfg: Config) -> pd.Series:
    """Per-row verdict: did this cell clear all three bars in this bout?"""
    return (
        (frame["mvl"] >= cfg.min_mvl)
        & (frame["shuffle_p"] <= cfg.max_shuffle_p)
        & (frame["reliability"] >= cfg.min_reliability)
    )


def reliable_cells(*, frame: pd.DataFrame, cfg: Config) -> dict[tuple[int, int], list[int]]:
    """The reliably-tuned unit ids per (mouse, session).

    Reliable means tuned in most of the session's wake bouts *and* pointing the same way in them.
    Both halves matter: consistency without strength admits a silent cell, and strength without
    consistency admits a cell whose preferred direction wanders, which is exactly the cell that
    cannot be trusted to carry a direction into sleep where nothing can check it.
    """
    frame = frame.assign(tuned=_is_tuned(frame=frame, cfg=cfg))
    out: dict[tuple[int, int], list[int]] = {}
    for (mouse, session), group in frame.groupby(["mouse", "session"]):
        n_bouts = group["bout_index"].nunique()
        chosen = []
        for unit, cell in group.groupby("unit"):
            fraction = cell["tuned"].sum() / max(1, n_bouts)
            consistency = hd_tuning.preferred_direction_consistency(
                preferred=cell.loc[cell["tuned"], "preferred"].to_numpy(dtype=float)
            )
            if fraction >= cfg.min_bout_fraction and (
                n_bouts < 2 or (np.isfinite(consistency)
                                and consistency >= cfg.min_preferred_consistency)
            ):
                chosen.append(int(unit))
        out[(int(mouse), int(session))] = sorted(chosen)
    return out


def _decode_rows(*, cfg: Config, mouse: int, session: int, subset: list[int]) -> list[dict]:
    """Per-bout decode from the whole ADn population and, optionally, from the reliable subset."""
    decode_cfg = DiffusionConfig(
        mouse=mouse, session=session, cell_areas=cfg.cell_areas, n_restarts=cfg.n_restarts
    )
    data = loader.load_session(mouse=mouse, session=session)
    selected, unit_ids = loader.select_units_by_area(units=data["units"], areas=cfg.cell_areas)
    if len(unit_ids) < 3:
        return []
    bouts = _wake_bouts(cfg=cfg, data=data)
    hd = head_direction.head_direction(data=data)

    populations = {"all": selected}
    if cfg.decode_reliable_subset and len(subset) >= cfg.min_subset_cells:
        populations["reliable"] = selected[subset]

    rows = []
    for index, (start, end) in enumerate(
        zip(np.asarray(bouts.start), np.asarray(bouts.end), strict=True)
    ):
        measured = bout_decode.binned_measured_angle(
            hd=hd, start=float(start), end=float(end), dt=decode_cfg.dt
        )
        row = {
            "mouse": mouse, "session": session, "session_id": f"Mouse{mouse}-{session}",
            "bout_index": index, "bout_start": float(start),
            "duration_s": float(end) - float(start),
            "n_cells": len(unit_ids), "n_reliable": len(subset),
        }
        for label, units in populations.items():
            result = bout_decode.decode_bout(
                units=units, start=float(start), end=float(end), cfg=decode_cfg,
                measured=measured, train_frac=cfg.train_frac, n_splits=cfg.n_splits,
                seed=cfg.seed,
            )
            row[f"rmse_{label}"] = result.rmse if result else float("nan")
        rows.append(row)
        print(
            f"  {row['session_id']} bout {index} ({row['duration_s']:.0f}s): "
            f"all {row.get('rmse_all', float('nan')):.3f}"
            + (f"  reliable ({len(subset)} cells) {row['rmse_reliable']:.3f}"
               if "rmse_reliable" in row else "  reliable n/a"),
            flush=True,
        )
    return rows


def collect(*, cfg: Config) -> None:
    """Tuning first, then the decode that depends on knowing which cells are reliable."""
    wanted = {io.parse_session_spec(s) for s in cfg.sessions}
    targets = [
        (mouse, session) for mouse, session in loader.list_sessions()
        if mouse in cfg.mice and (not wanted or (mouse, session) in wanted)
    ]

    if "cells" in cfg.stages:
        rows = []
        for mouse, session in targets:
            try:
                got = _cell_rows(cfg=cfg, mouse=mouse, session=session)
            except Exception as exc:  # noqa: BLE001 - one bad session must not stop the sweep
                print(f"  skip Mouse{mouse}-{session}: {exc}")
                continue
            if got:
                frame = pd.DataFrame(got)
                print(f"  Mouse{mouse}-{session}: {frame['unit'].nunique()} ADn cells x "
                      f"{frame['bout_index'].nunique()} bouts, "
                      f"median MVL {frame['mvl'].median():.3f}", flush=True)
            rows.extend(got)
        if rows:
            path = io.save_table(frame=pd.DataFrame(rows), name=f"{QUESTION_ID}_cells")
            print(f"  -> {len(rows)} cell-bout rows in {path.name}")

    if "decode" in cfg.stages:
        cells = pd.read_parquet(results_dir() / f"{QUESTION_ID}_cells.parquet")
        subsets = reliable_cells(frame=cells, cfg=cfg)
        rows = []
        for mouse, session in targets:
            try:
                rows.extend(_decode_rows(
                    cfg=cfg, mouse=mouse, session=session,
                    subset=subsets.get((mouse, session), []),
                ))
            except Exception as exc:  # noqa: BLE001
                print(f"  skip Mouse{mouse}-{session}: {exc}")
        if rows:
            path = io.save_table(frame=pd.DataFrame(rows), name=f"{QUESTION_ID}_decode")
            print(f"  -> {len(rows)} bouts in {path.name}")


def _bout_tuning(*, cells: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Population tuning summary per bout, the predictor exp2 tests."""
    tagged = cells.assign(tuned=_is_tuned(frame=cells, cfg=cfg))
    return (
        tagged.groupby(["mouse", "session", "session_id", "bout_index"])
        .agg(median_mvl=("mvl", "median"), median_reliability=("reliability", "median"),
             n_tuned=("tuned", "sum"), n_cells=("unit", "nunique"))
        .reset_index()
        .assign(tuned_fraction=lambda f: f["n_tuned"] / f["n_cells"])
    )


def analyse(*, cfg: Config, values: Values) -> None:
    """Report which cells are reliably tuned, and whether tuning predicts decode quality."""
    cells = pd.read_parquet(results_dir() / f"{QUESTION_ID}_cells.parquet")
    values.note("input_cell_bout_rows", len(cells))
    values.scalar("N_SESSIONS", int(cells["session_id"].nunique()), fmt="d")
    values.scalar("N_MICE", int(cells["mouse"].nunique()), fmt="d")
    values.scalar("N_CELLS", int(cells.groupby(["mouse", "session"])["unit"].nunique().sum()),
                  fmt="d")
    values.scalar("N_BOUTS", int(cells.groupby(["mouse", "session"])["bout_index"].nunique().sum()),
                  fmt="d")

    # --- exp1: which cells are reliably tuned ---
    tagged = cells.assign(tuned=_is_tuned(frame=cells, cfg=cfg))
    values.scalar("TUNED_ROW_FRACTION", float(tagged["tuned"].mean()), fmt=".2f")
    for name, column in (("MVL", "mvl"), ("RELIABILITY", "reliability")):
        values.scalar(f"{name}_MEDIAN", float(cells[column].median()), fmt=".2f")
        values.scalar(f"{name}_LO", float(cells[column].quantile(0.25)), fmt=".2f")
        values.scalar(f"{name}_HI", float(cells[column].quantile(0.75)), fmt=".2f")

    subsets = reliable_cells(frame=cells, cfg=cfg)
    per_session = []
    for (mouse, session), units in subsets.items():
        group = cells[(cells["mouse"] == mouse) & (cells["session"] == session)]
        per_session.append({
            "session_id": f"Mouse{mouse}-{session}",
            "bouts": int(group["bout_index"].nunique()),
            "adn_cells": int(group["unit"].nunique()),
            "reliable": len(units),
            "fraction": len(units) / max(1, group["unit"].nunique()),
        })
    session_frame = pd.DataFrame(per_session).sort_values("session_id")
    values.table("RELIABLE_PER_SESSION", session_frame, floatfmt=".2f")
    values.scalar("RELIABLE_FRACTION_MEDIAN", float(session_frame["fraction"].median()), fmt=".2f")
    values.scalar("RELIABLE_FRACTION_LO", float(session_frame["fraction"].min()), fmt=".2f")
    values.scalar("RELIABLE_FRACTION_HI", float(session_frame["fraction"].max()), fmt=".2f")

    per_mouse = (
        session_frame.assign(mouse=session_frame["session_id"].str.extract(r"Mouse(\d+)")[0])
        .groupby("mouse")
        .agg(sessions=("session_id", "size"), adn_cells=("adn_cells", "median"),
             reliable=("reliable", "median"), fraction=("fraction", "median"))
        .reset_index()
    )
    values.table("RELIABLE_PER_MOUSE", per_mouse, floatfmt=".2f")

    # Does a cell tuned in one bout stay tuned in the next? The premise of carrying cells into sleep.
    consistency = (
        tagged[tagged["tuned"]].groupby(["mouse", "session", "unit"])["preferred"]
        .apply(lambda p: hd_tuning.preferred_direction_consistency(
            preferred=p.to_numpy(dtype=float)))
        .dropna()
    )
    values.scalar("PREFERRED_CONSISTENCY_MEDIAN", float(consistency.median()), fmt=".2f")
    values.scalar("PREFERRED_CONSISTENCY_ABOVE_0_8",
                  float((consistency >= 0.8).mean()), fmt=".2f")

    _exp2_and_3(cfg=cfg, cells=cells, values=values)


def _exp2_and_3(*, cfg: Config, cells: pd.DataFrame, values: Values) -> None:
    """Tuning against decode error, and the reliable subset against the whole population."""
    path = results_dir() / f"{QUESTION_ID}_decode.parquet"
    tokens = ("DECODE_N", "TUNING_RMSE_RHO", "TUNING_RMSE_P", "TUNED_FRACTION_RHO",
              "TUNED_FRACTION_P", "CELLS_RMSE_RHO", "CELLS_RMSE_P", "DECODE_PASS_ALL",
              "SUBSET_N", "SUBSET_RMSE_MEDIAN", "ALL_RMSE_MEDIAN", "SUBSET_BETTER",
              "SUBSET_WILCOXON_P", "SUBSET_MEDIAN_CELLS")
    if not path.exists():
        for token in tokens:
            values.scalar(token, float("nan"))
        values.table("DECODE_TABLE", pd.DataFrame(
            [{"note": "run: hd-exp collect tuning1 --stages decode"}]))
        return

    decode = pd.read_parquet(path)
    bout = _bout_tuning(cells=cells, cfg=cfg)
    merged = decode.merge(bout, on=["mouse", "session", "session_id", "bout_index"], how="inner")
    values.scalar("DECODE_N", len(merged), fmt="d")
    values.scalar("DECODE_PASS_ALL", int((merged["rmse_all"] < cfg.rmse_threshold).sum()), fmt="d")

    for name, column in (("TUNING", "median_mvl"), ("TUNED_FRACTION", "tuned_fraction"),
                         ("CELLS", "n_cells")):
        usable = merged[np.isfinite(merged[column]) & np.isfinite(merged["rmse_all"])]
        corr = stats.correlation(x=usable[column], y=usable["rmse_all"])
        values.scalar(f"{name}_RMSE_RHO", corr["rho"])
        values.scalar(f"{name}_RMSE_P", corr["p"], fmt=".2g")

    values.table(
        "DECODE_TABLE",
        merged[["session_id", "bout_index", "duration_s", "n_cells", "n_reliable",
                "median_mvl", "tuned_fraction", "rmse_all"]
               + (["rmse_reliable"] if "rmse_reliable" in merged else [])]
        .sort_values(["session_id", "bout_index"]),
        floatfmt=".2f",
    )

    # --- exp3: reliable subset against the whole population ---
    if "rmse_reliable" not in merged:
        for token in ("SUBSET_N", "SUBSET_RMSE_MEDIAN", "ALL_RMSE_MEDIAN", "SUBSET_BETTER",
                      "SUBSET_WILCOXON_P", "SUBSET_MEDIAN_CELLS"):
            values.scalar(token, float("nan"))
        return
    paired = merged[np.isfinite(merged["rmse_reliable"]) & np.isfinite(merged["rmse_all"])]
    values.scalar("SUBSET_N", len(paired), fmt="d")
    values.scalar("SUBSET_MEDIAN_CELLS", float(paired["n_reliable"].median()), fmt=".0f")
    values.scalar("SUBSET_RMSE_MEDIAN", float(paired["rmse_reliable"].median()), fmt=".2f")
    values.scalar("ALL_RMSE_MEDIAN", float(paired["rmse_all"].median()), fmt=".2f")
    if len(paired) >= 5:
        test = stats.paired_difference(a=paired["rmse_all"], b=paired["rmse_reliable"])
        values.scalar("SUBSET_BETTER", test["n_positive"], fmt="d")
        values.scalar("SUBSET_WILCOXON_P", test["p"], fmt=".2g")
    else:
        values.scalar("SUBSET_BETTER", int((paired["rmse_reliable"] < paired["rmse_all"]).sum()),
                      fmt="d")
        values.scalar("SUBSET_WILCOXON_P", float("nan"))
