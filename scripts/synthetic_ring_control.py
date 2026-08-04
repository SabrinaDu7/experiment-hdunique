"""Does the decoder itself manufacture sub-diffusion?

The long-lag analysis found the decoded REM angle is sub-diffusive (anomalous exponent α ≈ 0.5 over
1–5 s) *within single bouts*, which rules out bout-to-bout mixing and leaves genuine confinement —
something restoring the bump toward preferred directions. Decoded-angle occupancy is indeed
non-uniform, but that lumpiness correlates with cell count, which is exactly what an unevenly
sampled spline ring would produce: nearest-point decoding preferentially assigns points to
densely-fit arcs, manufacturing attraction where there is none.

This script settles it by pushing a walk we *know* is free through each session's own fitted ring:

1. Refit the ring on the session's cached Isomap embedding (single fit, seed 0 — verified to
   reproduce the cached decode to a circular correlation of 0.999).
2. Measure the real off-ring residuals: the offset of each real embedded point from its nearest
   point on the curve. This is the decoder's actual working noise.
3. Simulate a **free** wrapped random walk in ring-angle coordinates, per bout, at the session's own
   measured D — so speed and bout structure match the real data.
4. Place those synthetic angles onto the fitted curve, and add residual noise resampled from (2).
5. Decode the synthetic points through the *same* ring and measure α over 1–5 s.

A faithful decoder returns α = 1, because the walk was free by construction. Any shortfall is
manufactured by the ring geometry plus projection noise.

Two controls are run per session: `clean` (points exactly on the curve, no noise) isolates the
parameterisation, and `noisy` (empirical residuals added) isolates geometry + projection together.

    uv run python scripts/synthetic_ring_control.py --sessions 28-140313 25-140130 17-130128
"""

import dataclasses

import numpy as np
import pandas as pd
import tyro
from beartype import beartype
from jaxtyping import Float, jaxtyped

import spud.manifold_fit_and_decode_fns as mff
import timescale
from config import DiffusionConfig
from env import results_dir
from manifold import fit_best_ring
from sweep import iter_cache

#: Lags and fit range for the anomalous exponent, matching the long-timescale analysis.
LAGS: tuple[int, ...] = tuple(range(1, 51))
LAGS_S = np.arange(1, 51, dtype=float) * 0.1
ALPHA_RANGE: tuple[float, float] = (1.0, 5.0)


@dataclasses.dataclass(frozen=True)
class ControlConfig:
    """Which sessions to run the control on."""

    #: `<mouse>-<session>` pairs; empty runs every cached ADn session (~30 s each).
    sessions: tuple[str, ...] = ()
    cell_set: str = "ADn"
    dt: float = 0.1
    seed: int = 0


@jaxtyped(typechecker=beartype)
def nearest_on_curve(
    *, points: Float[np.ndarray, "n dim"], curve: Float[np.ndarray, "k dim"]
) -> tuple[Float[np.ndarray, " n"], Float[np.ndarray, "n dim"]]:
    """For each point, the index of the closest curve vertex and the offset to it.

    Brute force over the 200-odd curve vertices, which is what the decoder effectively does too, so
    the residuals measured here are the residuals the decoder actually sees.
    """
    d2 = ((points[:, None, :] - curve[None, :, :]) ** 2).sum(-1)
    idx = np.argmin(d2, axis=1)
    return idx.astype(float), points - curve[idx]


@jaxtyped(typechecker=beartype)
def curve_at_angle(
    *, angles: Float[np.ndarray, " n"], curve: Float[np.ndarray, "k dim"]
) -> Float[np.ndarray, "n dim"]:
    """Place angles in [0, 2π) onto the fitted curve by periodic linear interpolation."""
    k = len(curve)
    pos = (angles % (2.0 * np.pi)) / (2.0 * np.pi) * k
    lo = np.floor(pos).astype(int) % k
    hi = (lo + 1) % k
    frac = (pos - np.floor(pos))[:, None]
    return curve[lo] * (1.0 - frac) + curve[hi] * frac


@jaxtyped(typechecker=beartype)
def free_walk(
    *, bout_lengths: list[int], d_target: float, dt: float, rng: np.random.Generator
) -> Float[np.ndarray, " time"]:
    """A free wrapped random walk per bout, concatenated, with ⟨Δα²⟩ = d_target·τ by construction."""
    step = np.sqrt(max(d_target, 1e-9) * dt)
    return np.concatenate(
        [np.cumsum(rng.normal(0.0, step, n)) % (2.0 * np.pi) for n in bout_lengths]
    )


def run_session(*, cfg: ControlConfig, entry: object) -> dict[str, object]:
    """Refit the ring, run both synthetic controls through it, and report α."""
    meta = entry.meta  # type: ignore[attr-defined]
    embed = np.asarray(entry.embed, dtype=float)  # type: ignore[attr-defined]
    bout_lengths = entry.bout_lengths  # type: ignore[attr-defined]
    real = np.asarray(entry.decoded[0], dtype=float)  # type: ignore[attr-defined]

    np.random.seed(cfg.seed)
    fit = fit_best_ring(points=embed, cfg=DiffusionConfig(n_restarts=1))
    curve = np.asarray(fit["curve"])[:-1]
    tt = fit["tt"][:-1]

    _, residuals = nearest_on_curve(points=embed, curve=curve)

    def alpha_of(angles: np.ndarray) -> float:
        c = timescale.msd_curve(
            angles=angles, bout_lengths=bout_lengths, lags=LAGS, method="circular"
        )
        return timescale.anomalous_exponent(
            curve=c, lags_s=LAGS_S, lo_s=ALPHA_RANGE[0], hi_s=ALPHA_RANGE[1]
        )

    # Speed the synthetic walk at the session's own measured short-lag rate.
    real_curve = timescale.msd_curve(
        angles=real, bout_lengths=bout_lengths, lags=LAGS, method="circular"
    )
    d_target = float(real_curve[1] / 0.2)

    rng = np.random.default_rng(cfg.seed)
    truth = free_walk(bout_lengths=bout_lengths, d_target=d_target, dt=cfg.dt, rng=rng)
    on_ring = curve_at_angle(angles=truth, curve=curve)

    out: dict[str, object] = {
        "session_id": meta["session_id"],
        "n_cells": meta["n_cells"],
        "d_target": d_target,
        "alpha_real": alpha_of(real),
        "alpha_truth": alpha_of(truth),
        "residual_median": float(np.median(np.linalg.norm(residuals, axis=1))),
    }
    for label, points in (
        ("clean", on_ring),
        ("noisy", on_ring + residuals[rng.integers(0, len(residuals), len(on_ring))]),
    ):
        decoded, _ = mff.decode_from_passed_fit(points, tt, curve, np.zeros(len(points)))
        out[f"alpha_{label}"] = alpha_of(np.asarray(decoded, dtype=float))
    return out


def run(*, cfg: ControlConfig) -> None:
    """Run the control on the requested sessions and write a summary table."""
    wanted = {tuple(int(p) for p in s.split("-")) for s in cfg.sessions}
    rows = []
    for entry in iter_cache(cell_set=cfg.cell_set):
        if wanted and (int(entry.meta["mouse"]), int(entry.meta["session"])) not in wanted:
            continue
        row = run_session(cfg=cfg, entry=entry)
        rows.append(row)
        print(
            f"  {row['session_id']:16s} ({row['n_cells']:2d} cells)  "
            f"α real {row['alpha_real']:.2f} | truth {row['alpha_truth']:.2f} "
            f"| decoded-clean {row['alpha_clean']:.2f} | decoded-noisy {row['alpha_noisy']:.2f}"
        )
    if not rows:
        print("No matching cached sessions.")
        return

    frame = pd.DataFrame(rows)
    results_dir().mkdir(parents=True, exist_ok=True)
    path = results_dir() / "synthetic_ring_control.csv"
    frame.to_csv(path, index=False)
    print(f"\n  medians: real {frame.alpha_real.median():.2f}  truth {frame.alpha_truth.median():.2f}"
          f"  clean {frame.alpha_clean.median():.2f}  noisy {frame.alpha_noisy.median():.2f}")
    print(f"  saved {path}")


def main() -> None:
    """Entry point."""
    run(cfg=tyro.cli(ControlConfig))


if __name__ == "__main__":
    main()
