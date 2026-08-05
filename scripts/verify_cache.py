"""Validate the migrated cache by recomputing whole sessions from the NWBs.

`migrate_cache.py` imports decoded REM angles that were produced by the predecessor repo. Those
angles are only usable if this repo's code, run cold from the NWB files, reproduces them. This
script does that check: for each named session it forces a full recompute (embedding, ring fits,
decode) and diffs the result against what is in the cache.

The decode is deterministic given the seed — `fit_manifold` draws its k-means init from the global
NumPy RNG, which the sweep seeds per refit — so agreement should be essentially exact. A large
divergence means the port changed the pipeline, not just the reporting.

    uv run python scripts/verify_cache.py --sessions 28-140313 25-140130
"""

import dataclasses

import numpy as np
import tyro

from core.config import DIFFUSION_LAGS, HEADLINE_WINDOW_MS, DiffusionConfig
from decode import loader, manifold, rates, sweep
from metrics import diffusion as dif


@dataclasses.dataclass(frozen=True)
class VerifyConfig:
    """Sessions to verify, as `<mouse>-<session>` strings."""

    sessions: tuple[str, ...] = ("28-140313",)
    cell_areas: tuple[str, ...] = ("ADn",)


def recompute(*, cfg: DiffusionConfig) -> tuple[np.ndarray, np.ndarray]:
    """Run the full pipeline cold for one session. Returns (embedding, decoded angles).

    Deliberately mirrors `sweep.run_session` without its cache, so what is compared is the
    pipeline's own output rather than anything the cache handed back.
    """
    data = loader.load_session(mouse=cfg.mouse, session=cfg.session)
    units = loader.get_units(data=data)
    units_sel, _ = loader.select_units_by_area(units=units, areas=cfg.cell_areas)
    rem = loader.load_state_epochs(data=data, state=cfg.rem_label)

    rate_mat = rates.rate_matrix(units=units_sel, epochs=rem, dt=cfg.dt, sigma=cfg.sigma)
    embed = manifold.isomap_embed(rates=rate_mat[: cfg.n_samples], cfg=cfg)
    return embed, manifold.decode_refits(embed=embed, cfg=cfg)


def verify_one(*, mouse: int, session: int, cell_areas: tuple[str, ...]) -> None:
    """Recompute one session and report the divergence from its cached entry."""
    cfg = DiffusionConfig(mouse=mouse, session=session, cell_areas=cell_areas)
    cached = sweep.load_cache(cfg=cfg)
    if cached is None:
        print(f"{cfg.session_id} [{cfg.cell_set}]: no matching cache entry")
        return

    embed, decoded = recompute(cfg=cfg)
    cached_embed, cached_decoded = cached.embed, cached.decoded

    print(f"\n=== {cfg.session_id} [{cfg.cell_set}] ===")
    print(f"  shapes: embed {embed.shape} vs {cached_embed.shape}; "
          f"decoded {decoded.shape} vs {cached_decoded.shape}")
    if embed.shape != cached_embed.shape or decoded.shape != cached_decoded.shape:
        print("  SHAPE MISMATCH — the pipeline changed upstream of the decode")
        return

    # The cache stores float32, so ~1e-7 relative differences are storage precision, not drift.
    print(f"  max |Δembed|   = {np.max(np.abs(embed - cached_embed)):.3e}")
    print(f"  max |Δdecoded| = {np.max(np.abs(decoded - cached_decoded)):.3e}")

    for label, traces in [("recomputed", decoded), ("cached", cached_decoded)]:
        curve = np.mean(
            [dif.diffusion_curve(angles=t, lags=DIFFUSION_LAGS) for t in traces], axis=0
        )
        d, r2 = dif.window_slope(curve=curve, dt=cfg.dt, window_ms=HEADLINE_WINDOW_MS)
        print(f"  {label:<11} D = {d:.6f}  r² = {r2:.6f}")


def run(*, cfg: VerifyConfig) -> None:
    """Verify every requested session."""
    for spec in cfg.sessions:
        mouse, session = spec.split("-")
        verify_one(mouse=int(mouse), session=int(session), cell_areas=cfg.cell_areas)


def main() -> None:
    """Entry point: `uv run python scripts/verify_cache.py`."""
    run(cfg=tyro.cli(VerifyConfig))


if __name__ == "__main__":
    main()
