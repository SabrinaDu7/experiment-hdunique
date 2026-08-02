"""Isomap embedding and the 12-knot ring fit / decode, wrapping the vendored `spud` code."""

import numpy as np
from beartype import beartype
from jaxtyping import Float, jaxtyped

import spud.dim_red_fns as drf
import spud.manifold_fit_and_decode_fns as mff
from hdunique.config import DiffusionConfig


@jaxtyped(typechecker=beartype)
def isomap_embed(
    *, rates: Float[np.ndarray, "time units"], cfg: DiffusionConfig
) -> Float[np.ndarray, "time dim"]:
    """Variance-stabilise (square root) and embed the rate matrix with Isomap. Deterministic."""
    return drf.run_dim_red(
        rates,
        params={"n_neighbors": cfg.n_neighbors, "target_dim": cfg.fit_dim},
        method="iso",
    )


def _fit_params(*, cfg: DiffusionConfig) -> dict[str, object]:
    """Spline-fit parameters in the shape `spud.fit_manifold` expects."""
    return {
        "dalpha": cfg.dalpha,
        "knot_order": cfg.knot_order,
        "penalty_type": cfg.penalty,
        "nKnots": cfg.n_knots,
    }


@jaxtyped(typechecker=beartype)
def fit_best_ring(
    *, points: Float[np.ndarray, "time dim"], cfg: DiffusionConfig
) -> dict[str, object]:
    """Fit the ring `cfg.n_restarts` times from different k-means inits; keep the lowest `fit_err`.

    `fit_err` is the unsupervised spline cost the fit minimises — sum of point-to-curve distances
    times curve length — so selecting on it needs no ground-truth angle and applies to REM exactly
    as it does to wake.
    """
    fits = (mff.fit_manifold(points, _fit_params(cfg=cfg)) for _ in range(max(1, cfg.n_restarts)))
    return min(fits, key=lambda f: f["fit_err"])


@jaxtyped(typechecker=beartype)
def decode_ring_angle(
    *, embed: Float[np.ndarray, "time dim"], cfg: DiffusionConfig, rng: np.random.Generator
) -> Float[np.ndarray, " time"]:
    """Fit one ring on `cfg.fit_frac` of the points, then decode **all** of them to a ring angle.

    Nothing is held out: REM has no measured head direction, so there is nothing to score against
    and no reason to shrink the fit set. The reference trace passed to the decoder is zeros — it
    only fixes a global shift and flip, and D is invariant to both.
    """
    if cfg.fit_frac >= 1.0:
        points = embed
    else:
        n_fit = round(cfg.fit_frac * len(embed))
        points = embed[rng.choice(len(embed), n_fit, replace=False)]

    fit = fit_best_ring(points=points, cfg=cfg)
    reference = np.zeros(len(embed))
    decoded, _ = mff.decode_from_passed_fit(embed, fit["tt"][:-1], fit["curve"][:-1], reference)
    return np.asarray(decoded, dtype=float)


@jaxtyped(typechecker=beartype)
def decode_refits(
    *, embed: Float[np.ndarray, "time dim"], cfg: DiffusionConfig
) -> Float[np.ndarray, "refit time"]:
    """Decode the embedding `cfg.n_refits` times from independent ring fits, giving D mean +/- std.

    `spud.fit_manifold` seeds nothing of its own — its k-means init draws from the global NumPy RNG
    — so seeding that RNG once per refit is what makes the sweep reproducible.
    """
    traces = []
    for refit in range(cfg.n_refits):
        np.random.seed(cfg.seed + refit)
        traces.append(
            decode_ring_angle(embed=embed, cfg=cfg, rng=np.random.default_rng(cfg.seed + refit))
        )
    return np.array(traces)
