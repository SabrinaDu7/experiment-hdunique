"""The one strip-plot grammar this project uses.

Three near-identical copies of this had accumulated — the variance summary's panel A, the cell-set
comparison, and the bout exit-state figure — each re-implementing the same jitter, the same group
mean bar, the same open/filled marker split and the same log-D axis. This is that figure, once.

Deliberately a strip of every point rather than a box or violin: with a handful of observations per
group, a KDE or quartile box invents a confident-looking density from almost nothing.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#: Raw-D labels placed on a log-D axis, so the axis reads in rad^2/s.
D_TICKS = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0])


def log_d_axis(*, ax: plt.Axes, lo: float, hi: float, axis: str = "y") -> None:
    """Label a log-D axis with raw-D ticks, so it reads in rad^2/s rather than log units."""
    ticks = D_TICKS[(np.log(D_TICKS) > lo) & (np.log(D_TICKS) < hi)]
    labels = [f"{t:g}" for t in ticks]
    if axis == "y":
        ax.set_ylim(lo, hi)
        ax.set_yticks(np.log(ticks), labels)
    else:
        ax.set_xlim(lo, hi)
        ax.set_xticks(np.log(ticks), labels)


def save_figure(*, fig: plt.Figure, path: Path) -> Path:
    """Write a figure and report where it went, creating the directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path.name}")
    return path


def strip_panel(
    *,
    ax: plt.Axes,
    frame: pd.DataFrame,
    group: str,
    order: list,
    colors: dict,
    value: str = "log_D",
    count_col: str | None = "n_cells",
    well_sampled: int = 20,
    seed: int = 0,
    statistic: str = "mean",
) -> None:
    """One panel: every observation as a jittered dot, each group's centre as a bar.

    `count_col` drives the filled/open split — open marks flag observations whose ring may be
    undersampled. They are drawn rather than dropped, because the point of the flag is to let a
    reader judge, and silently gating them away would hide the very confound they indicate. Pass
    None to draw every point filled.
    """
    rng = np.random.default_rng(seed)
    for i, key in enumerate(order):
        sub = frame[frame[group] == key]
        if not len(sub):
            continue
        x = i + rng.uniform(-0.11, 0.11, len(sub))
        if count_col is None or count_col not in sub:
            ax.scatter(x, sub[value], s=42, color=colors[key], zorder=3)
        else:
            well = (sub[count_col] >= well_sampled).to_numpy()
            ax.scatter(x[well], sub[value][well], s=42, color=colors[key], zorder=3)
            ax.scatter(x[~well], sub[value][~well], s=42, facecolors="none",
                       edgecolors=colors[key], lw=1.4, zorder=3)
        centre = sub[value].mean() if statistic == "mean" else sub[value].median()
        ax.hlines(centre, i - 0.28, i + 0.28, color=colors[key], lw=3, zorder=4)

    if len(frame):
        ax.axhline(frame[value].mean(), ls="--", color="0.55", lw=1.0, zorder=1)
    ax.set_xticks(range(len(order)), [str(k) for k in order], rotation=45, ha="right")
    ax.set_xlim(-0.6, len(order) - 0.4)


def plot_grouped_strip(
    *,
    panels: dict[str, pd.DataFrame],
    group: str,
    title: str,
    save_path: Path,
    value: str = "log_D",
    count_col: str | None = "n_cells",
    well_sampled: int = 20,
    statistic: str = "mean",
) -> Path:
    """One strip panel per entry in `panels`, sharing a single log-D axis.

    The shared axis is the point: panels are only worth putting side by side if the reader can
    compare heights across them.
    """
    order = sorted({k for f in panels.values() for k in f[group]})
    colors = {k: f"C{i}" for i, k in enumerate(order)}
    everything = pd.concat(panels.values()) if panels else pd.DataFrame({value: []})
    lo = float(everything[value].min()) - 0.25
    hi = float(everything[value].max()) + 0.25

    fig, axes = plt.subplots(
        1, len(panels), figsize=(3.5 * len(panels) + 1.2, 4.6),
        constrained_layout=True, sharey=True, squeeze=False,
    )
    for ax, (name, frame) in zip(axes[0], panels.items(), strict=True):
        strip_panel(ax=ax, frame=frame, group=group, order=order, colors=colors, value=value,
                    count_col=count_col, well_sampled=well_sampled, statistic=statistic)
        ax.set_title(f"{name}  (n = {len(frame)})", fontsize=10, loc="left")

    axes[0][0].set_ylabel("D (rad²/s, log scale)")
    log_d_axis(ax=axes[0][0], lo=lo, hi=hi)
    if count_col is not None:
        axes[0][0].legend(
            handles=[
                plt.Line2D([], [], ls="none", marker="o", color="0.35",
                           label=f"≥{well_sampled} cells"),
                plt.Line2D([], [], ls="none", marker="o", mfc="none", color="0.35",
                           label=f"<{well_sampled} cells"),
                plt.Line2D([], [], ls="--", color="0.55", label="grand mean"),
            ],
            frameon=False, fontsize=8, loc="lower left",
        )
    fig.suptitle(title, fontsize=12)
    return save_figure(fig=fig, path=save_path)
