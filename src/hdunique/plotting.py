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

from hdunique.config import VarianceConfig

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
    """Every session as a dot with the mouse mean as a bar. Open marks flag low-cell sessions,
    kept visible rather than silently gated away. Cell count is only a noisy proxy for ring quality
    (`nugget` separates clean from dirty better), so the mark is a flag to check, not a verdict."""
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
            plt.Line2D([], [], ls="none", marker="o", color="0.35", label=f"≥{well_sampled} cells"),
            plt.Line2D([], [], ls="none", marker="o", mfc="none", color="0.35",
                       label=f"<{well_sampled} cells"),
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
        f"({cfg.cell_set}, ≥{cfg.min_adn_cells} cells, {cfg.window_ms} ms window)",
        fontsize=12,
    )
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  saved {save_path}")
