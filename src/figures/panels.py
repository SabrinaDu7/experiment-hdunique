"""Figures: per-session and per-mouse diffusion curves, and the variance-decomposition summary.

All fit lines here are origin-forced, matching the estimator in `diffusion.py`, so what is drawn
is what is tabulated.
"""

import dataclasses
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from beartype import beartype
from jaxtyping import Float, jaxtyped

from core.config import VarianceConfig

#: Raw-D labels placed on a log-D axis, so the axis reads in rad^2/s.
D_TICKS = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
#: At or above this cell count the ring is well constrained; below it D is closer to an upper bound.
WELL_SAMPLED_CELLS = 20


# --- diffusion curves ---------------------------------------------------------------------------
@jaxtyped(typechecker=beartype)
@dataclasses.dataclass(frozen=True)
class DiffusionPanel:
    """Everything one diffusion-curve panel draws: the measured curve, its fit and its labels.

    The single-session figure and the per-mouse grid draw identical panels, so both callers build
    this and neither owns the drawing.
    """

    title: str
    lags_s: Float[np.ndarray, " lag"]
    curve: Float[np.ndarray, " lag"]
    d: float
    r2: float
    n_cells: int


def _cell_gauge(*, ax: plt.Axes, n_cells: int, max_cells: int) -> None:
    """Small vertical gauge whose fill height is n_cells / the mouse's maximum, with the count
    printed below — a visual cue for whether a steep D coincides with an undersampled ring."""
    x0, width = 0.90, 0.035
    y0, height = 0.30, 0.55
    frac = (n_cells / max_cells) if max_cells > 0 else 0.0
    ax.add_patch(
        plt.Rectangle((x0, y0), width, height, transform=ax.transAxes, fill=False, ec="0.6",
                      lw=0.8, zorder=5)
    )
    ax.add_patch(
        plt.Rectangle((x0, y0), width, height * frac, transform=ax.transAxes, facecolor="C0",
                      ec="none", zorder=6)
    )
    ax.text(x0 + width / 2, y0 - 0.04, str(n_cells), transform=ax.transAxes, ha="center",
            va="top", fontsize=8, color="0.25")


def _diffusion_panel(
    *,
    ax: plt.Axes,
    panel: DiffusionPanel,
    n_fit: int,
    max_cells: int,
    fontsize: int = 8,
) -> None:
    """One diffusion-curve panel: the measured points, the origin-forced fit over the first `n_fit`
    lags (solid) extrapolated across the rest (dashed), D and r2 annotated, plus a cell gauge."""
    lags_s, d = panel.lags_s, panel.d
    x = np.concatenate([[0.0], lags_s])
    y = np.concatenate([[0.0], panel.curve])
    ax.scatter(x[1 + n_fit :], y[1 + n_fit :], color="0.6", s=28, zorder=3)
    ax.scatter(x[: 1 + n_fit], y[: 1 + n_fit], color="C0", s=28, zorder=4)
    ax.plot([0.0, lags_s[-1]], [0.0, d * lags_s[-1]], "C3--", lw=1.0)
    ax.plot([0.0, lags_s[n_fit - 1]], [0.0, d * lags_s[n_fit - 1]], "C3-", lw=2.5)
    ax.text(
        0.04, 0.96, f"D = {d:.3f} rad²/s\nr² = {panel.r2:.3f}", transform=ax.transAxes, va="top",
        fontsize=fontsize, bbox={"boxstyle": "round", "fc": "white", "ec": "0.7"},
    )
    _cell_gauge(ax=ax, n_cells=panel.n_cells, max_cells=max_cells)
    ax.set_title(panel.title, fontsize=9)
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)


def plot_session_diffusion(
    *, panel: DiffusionPanel, n_fit: int, window_ms: int, save_path: Path
) -> None:
    """Single-session diffusion-curve figure."""
    fig, ax = plt.subplots(figsize=(6.5, 5), constrained_layout=True)
    _diffusion_panel(ax=ax, panel=panel, n_fit=n_fit, max_cells=panel.n_cells, fontsize=10)
    ax.set_xlabel("lag τ (s)")
    ax.set_ylabel("⟨Δα²⟩ (rad²)")
    fig.suptitle(f"REM diffusion ({window_ms} ms fit window)", fontsize=12)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  saved {save_path}")


def plot_mouse_diffusion_grid(
    *, mouse: int, panels: list[DiffusionPanel], n_fit: int, window_ms: int, cell_set: str,
    save_path: Path,
) -> None:
    """One figure per mouse: a grid of diffusion-curve panels, one per session, on a shared
    y-limit so slopes are visually comparable panel to panel."""
    n = len(panels)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), constrained_layout=True, squeeze=False
    )
    flat = axes.ravel()
    ymax = max(float(np.max(p.curve)) for p in panels) * 1.05
    max_cells = max(p.n_cells for p in panels)
    for ax, panel in zip(flat, panels):
        _diffusion_panel(ax=ax, panel=panel, n_fit=n_fit, max_cells=max_cells)
        ax.set_ylim(0.0, ymax)
    for ax in flat[n:]:
        ax.axis("off")
    for ax in axes[-1, :]:
        ax.set_xlabel("lag τ (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("⟨Δα²⟩ (rad²)")
    fig.suptitle(
        f"Mouse{mouse}  REM diffusion  ({window_ms} ms fit window, {cell_set})", fontsize=13
    )
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  saved {save_path}")


# --- variance decomposition ----------------------------------------------------------------------
def _strip_panel(
    *, ax: plt.Axes, df: pd.DataFrame, order: list[int], colors: dict[int, str], seed: int,
    count_col: str = "n_cells", well_sampled: int = WELL_SAMPLED_CELLS,
) -> None:
    """Every session as a dot with the mouse mean as a bar.

    Two different cell-count thresholds are in play and the labelling must keep them apart: the
    *gate* (`VarianceConfig.min_adn_cells`, named in the figure title) decides which sessions appear
    at all, while `well_sampled` here only decides filled versus open markers among those that did.
    Open marks flag sessions whose ring may be undersampled, kept visible rather than silently gated
    away. Cell count is only a noisy proxy for ring quality (`nugget` separates clean from dirty
    better), so the mark is a flag to check, not a verdict."""
    rng = np.random.default_rng(seed)
    for i, mouse in enumerate(order):
        sub = df[df["mouse"] == mouse]
        if not len(sub):
            continue
        x = i + rng.uniform(-0.09, 0.09, len(sub))
        well = (sub[count_col] >= well_sampled).to_numpy()
        ax.scatter(x[well], sub["log_D"][well], s=42, color=colors[mouse], zorder=3)
        ax.scatter(x[~well], sub["log_D"][~well], s=42, facecolors="none",
                   edgecolors=colors[mouse], lw=1.4, zorder=3)
        ax.hlines(sub["log_D"].mean(), i - 0.28, i + 0.28, color=colors[mouse], lw=3, zorder=4)
    ax.axhline(df["log_D"].mean(), ls="--", color="0.55", lw=1.0, zorder=1)
    ax.set_xticks(range(len(order)), [f"Mouse{m}" for m in order], rotation=45, ha="right")
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylabel("D (rad²/s, log scale)")
    ax.set_title("A — every session (bar = mouse mean)", fontsize=10, loc="left")
    ax.legend(
        handles=[
            plt.Line2D([], [], ls="none", marker="o", color="0.35",
                       label=f"≥{well_sampled} cells (well sampled)"),
            plt.Line2D([], [], ls="none", marker="o", mfc="none", color="0.35",
                       label=f"<{well_sampled} cells (D may be inflated)"),
            plt.Line2D([], [], ls="--", color="0.55", label="grand mean"),
        ],
        frameon=False, fontsize=8, loc="upper left",
    )


def _caterpillar_panel(
    *, ax: plt.Axes, effects: pd.DataFrame, grand_mean: float, colors: dict[int, str]
) -> None:
    """Each mouse's shrunken estimate ±1.96 conditional SD. Mice with few sessions are pulled
    toward the grand mean and get visibly wider bars."""
    for i, row in enumerate(effects.itertuples()):
        lo = row.mean_log_D - 1.96 * row.cond_sd
        hi = row.mean_log_D + 1.96 * row.cond_sd
        ax.plot([lo, hi], [i, i], color=colors[row.mouse], lw=2, zorder=3, solid_capstyle="round")
        ax.scatter(row.mean_log_D, i, s=48, color=colors[row.mouse], zorder=4)
        ax.text(hi + 0.06, i, f"n={row.n_sessions}", va="center", fontsize=8, color="0.35")
    ax.axvline(grand_mean, ls="--", color="0.55", lw=1.0, zorder=1)
    ax.set_yticks(range(len(effects)), [f"Mouse{m}" for m in effects["mouse"]])
    ax.set_ylim(-0.6, len(effects) - 0.4)
    ax.set_xlabel("shrunken mean D (rad²/s, log scale)")
    ax.set_title("B — model estimate ±95% (bar width = uncertainty)", fontsize=10, loc="left")


def plot_variance_summary(
    *, df: pd.DataFrame, effects: pd.DataFrame, cfg: VarianceConfig, save_path: Path
) -> None:
    """Two-panel read of the decomposition. A = the raw sessions (spread of the bars is
    between-mouse, spread within a column is within-mouse); B = the model's shrunken per-mouse
    estimates with their uncertainty — intervals straddling the grand mean is the visual form of
    "the ICC CI includes zero". No violin or box plot: with 1-8 sessions per mouse a KDE would
    invent a confident-looking density from one or two points."""
    order = [int(m) for m in effects["mouse"]]
    colors = {m: f"C{i}" for i, m in enumerate(order)}
    grand_mean = float(df["log_D"].mean())

    fig, (ax_strip, ax_cat) = plt.subplots(
        1, 2, figsize=(5.0 + 0.9 * len(order), 4.8), constrained_layout=True
    )
    _strip_panel(ax=ax_strip, df=df, order=order, colors=colors, seed=cfg.seed)
    _caterpillar_panel(ax=ax_cat, effects=effects, grand_mean=grand_mean, colors=colors)

    lo, hi = df["log_D"].min() - 0.25, df["log_D"].max() + 0.25
    ticks = D_TICKS[(np.log(D_TICKS) > lo) & (np.log(D_TICKS) < hi)]
    ax_strip.set_ylim(lo, hi)
    ax_strip.set_yticks(np.log(ticks), [f"{t:g}" for t in ticks])
    ax_cat.set_xlim(lo, hi + 0.3)
    ax_cat.set_xticks(np.log(ticks), [f"{t:g}" for t in ticks])

    fig.suptitle(
        f"REM diffusion across mice and sessions "
        f"({cfg.cell_set}, gated at ≥{cfg.min_adn_cells} cells, {cfg.window_ms} ms window)",
        fontsize=12,
    )
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  saved {save_path}")


# --- long-timescale diffusion ---------------------------------------------------------------------
def _timescale_panel(
    *, ax: plt.Axes, row: dict[str, object], lags_s: np.ndarray, ceiling: float
) -> None:
    """One session's MSD curves under all three estimators, log-log, with the wrapped ceiling drawn.

    Log-log is the right frame here: pure diffusion is a straight line of slope 1, so sub-diffusive
    flattening and the wrapped ceiling are both immediately visible as departures from that slope.
    """
    styles = {"wrapped": ("C7", "o", "wrapped"),
              "unwrapped": ("C0", "s", "unwrapped"),
              "circular": ("C3", "^", "circular")}
    for method, (color, marker, label) in styles.items():
        curve = np.asarray(row[f"curve_{method}"], dtype=float)
        ax.plot(lags_s, curve, color=color, marker=marker, ms=2.5, lw=1.2, label=label)

    ax.axhline(ceiling, ls=":", color="0.4", lw=1.0)
    ax.text(lags_s[0], ceiling * 1.08, "π²/3", fontsize=7, color="0.4")

    # Slope-1 guide anchored on the 200 ms circular fit: what pure diffusion at the short-lag rate
    # would look like if it continued. Departure from it IS the result.
    d200 = float(row["D_200_circular"])
    ax.plot(lags_s, d200 * lags_s, ls="--", color="0.55", lw=1.0)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(
        f"{row['session_id']}  ({row['n_cells']} cells)\n"
        f"risk={row['unwrap_risk']:.3f}  u/c={row['unwrapped_over_circular']:.1f}",
        fontsize=8,
    )
    ax.tick_params(labelsize=7)


def plot_timescale_curves(
    *, mouse: int, rows: list[dict[str, object]], dt: float, max_lag: int, cell_set: str,
    save_path: Path,
) -> None:
    """One figure per mouse: the MSD curve of every session under all three estimators.

    The dashed grey line is the 200 ms rate extrapolated as if diffusion continued unchanged; the
    dotted line is the wrapped estimator's hard ceiling. A session that is genuinely diffusive out
    to 5 s tracks the dashed line; one that is not falls below it.
    """
    lags_s = np.arange(1, max_lag + 1, dtype=float) * dt
    n = len(rows)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.9 * ncols, 3.3 * nrows), constrained_layout=True, squeeze=False
    )
    flat = axes.ravel()
    for ax, row in zip(flat, sorted(rows, key=lambda r: int(r["session"])), strict=False):
        _timescale_panel(ax=ax, row=row, lags_s=lags_s, ceiling=float(row["wrapped_ceiling"]))
    for ax in flat[n:]:
        ax.axis("off")
    for ax in axes[-1, :]:
        ax.set_xlabel("lag τ (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("⟨Δα²⟩ (rad²)")
    flat[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle(
        f"Mouse{mouse}  REM diffusion across timescales ({cell_set}); "
        f"dashed = 200 ms rate extrapolated",
        fontsize=12,
    )
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  saved {save_path}")


def plot_cellset_strip(
    *, frames: dict[str, pd.DataFrame], window_ms: int, estimator: str, well_sampled: int,
    ylim: tuple[float, float], save_path: Path,
) -> None:
    """REM diffusion at one measurement window, one strip panel per cell set.

    Same visual grammar as `plot_variance_summary`'s panel A — one dot per session, x-jittered
    within its mouse, mouse mean as a bar, grand mean dashed, log-D axis with raw-D tick labels —
    repeated per cell set so the three can be read against each other. A mouse holds the same x slot
    and colour in every panel, and `ylim` is shared across panels *and* across the figures for the
    other windows, so the whole set is directly comparable.

    No gate is applied: every session with that cell set is drawn, with open marks flagging the
    ones below `well_sampled`. At these lags the low-cell sessions are exactly the ones whose
    estimate is least trustworthy, so hiding them would hide the caveat.
    """
    order = sorted({int(m) for f in frames.values() for m in f["mouse"]})
    colors = {m: f"C{i}" for i, m in enumerate(order)}
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(
        1, len(frames), figsize=(3.6 * len(frames) + 1.2, 4.6), constrained_layout=True,
        sharey=True, squeeze=False,
    )
    for ax, (cell_set, frame) in zip(axes[0], frames.items(), strict=True):
        for i, mouse in enumerate(order):
            sub = frame[frame["mouse"] == mouse]
            if not len(sub):
                continue
            x = i + rng.uniform(-0.09, 0.09, len(sub))
            well = (sub["n_cells"] >= well_sampled).to_numpy()
            ax.scatter(x[well], sub["log_D"][well], s=42, color=colors[mouse], zorder=3)
            ax.scatter(x[~well], sub["log_D"][~well], s=42, facecolors="none",
                       edgecolors=colors[mouse], lw=1.4, zorder=3)
            ax.hlines(sub["log_D"].mean(), i - 0.28, i + 0.28, color=colors[mouse], lw=3, zorder=4)
        if len(frame):
            ax.axhline(frame["log_D"].mean(), ls="--", color="0.55", lw=1.0, zorder=1)
        ax.set_xticks(range(len(order)), [f"Mouse{m}" for m in order], rotation=45, ha="right")
        ax.set_xlim(-0.6, len(order) - 0.4)
        ax.set_title(f"{cell_set}  (n = {len(frame)} sessions)", fontsize=10, loc="left")

    axes[0][0].set_ylabel("D (rad²/s, log scale)")
    axes[0][0].set_ylim(*ylim)
    ticks = D_TICKS[(np.log(D_TICKS) > ylim[0]) & (np.log(D_TICKS) < ylim[1])]
    axes[0][0].set_yticks(np.log(ticks), [f"{t:g}" for t in ticks])
    axes[0][0].legend(
        handles=[
            plt.Line2D([], [], ls="none", marker="o", color="0.35",
                       label=f"≥{well_sampled} cells (well sampled)"),
            plt.Line2D([], [], ls="none", marker="o", mfc="none", color="0.35",
                       label=f"<{well_sampled} cells (D may be inflated)"),
            plt.Line2D([], [], ls="--", color="0.55", label="grand mean (this panel)"),
        ],
        frameon=False, fontsize=8, loc="lower left",
    )
    fig.suptitle(
        f"REM diffusion by cell set — {window_ms / 1000:g} s measurement window "
        f"({estimator} estimator); bar = mouse mean",
        fontsize=12,
    )
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  saved {save_path}")


def plot_bout_exit_strip(
    *, frames: dict[str, pd.DataFrame], save_path: Path
) -> None:
    """Per-bout D split by what the bout exited to, one panel per session.

    Deliberately a strip of every bout rather than a summary: with 3-19 bouts per group a box or
    violin would invent a density. The bar is the group median. Because the comparison that matters
    is *within* a session, each panel is self-contained and the panels share a log-D axis only so
    that session-to-session scale differences stay visible.
    """
    fig, axes = plt.subplots(
        1, len(frames), figsize=(3.1 * len(frames) + 1.0, 4.4), constrained_layout=True,
        sharey=True, squeeze=False,
    )
    order = ["Awake", "Non-REM"]
    colors = {"Awake": "C1", "Non-REM": "C0"}
    rng = np.random.default_rng(0)

    for ax, (session_id, frame) in zip(axes[0], frames.items(), strict=True):
        for i, state in enumerate(order):
            sub = frame[frame["next_state"] == state]
            if not len(sub):
                continue
            x = i + rng.uniform(-0.12, 0.12, len(sub))
            ax.scatter(x, np.log(sub["D_200"]), s=44, color=colors[state], alpha=0.85, zorder=3)
            ax.hlines(np.log(sub["D_200"]).median(), i - 0.3, i + 0.3, color=colors[state],
                      lw=3, zorder=4)
        ax.set_xticks(range(len(order)), [f"exit →\n{s}" for s in order])
        ax.set_xlim(-0.6, len(order) - 0.4)
        n_cells = int(frame["n_cells"].iloc[0])
        counts = frame["next_state"].value_counts()
        ax.set_title(
            f"{session_id}  ({n_cells} cells)\n"
            f"n = {counts.get('Awake', 0)} vs {counts.get('Non-REM', 0)} bouts",
            fontsize=9,
        )

    axes[0][0].set_ylabel("D (rad²/s, log scale)")
    lo, hi = axes[0][0].get_ylim()
    ticks = D_TICKS[(np.log(D_TICKS) > lo) & (np.log(D_TICKS) < hi)]
    axes[0][0].set_yticks(np.log(ticks), [f"{t:g}" for t in ticks])
    fig.suptitle(
        "Per-bout REM diffusion, split by the state each bout exited to", fontsize=12
    )
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  saved {save_path}")
