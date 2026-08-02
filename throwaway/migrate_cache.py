"""One-off: migrate the predecessor repo's decode cache into this repo's cache format.

The source repo (SPUD_Analysis-of-manifold-structure-in-head-direction-data,
outputs/results/pynapple_m3/cache/) stored, per (session, cell set), the Isomap embedding and the
per-refit decoded REM angles produced by exactly the settings that are now this repo's defaults
(n_restarts=10, n_refits=5, fit_frac=1.0, seed=0, 12 knots, dt=sigma=0.1, Isomap 3-D nn=5).

Those decoded angles are the expensive part (~50 ring fits per session). The change this port makes
— forcing the 200 ms fit through the origin — is downstream of the decode, so the cached angles are
still valid and D can simply be recomputed from them.

This script rewrites each entry into the new schema:
  dec -> decoded, embed_sig -> signature (with the fit parameters added), + bout_lengths
`bout_lengths` is not in the old cache and is recomputed from the NWB REM epochs (cheap: epochs
only, no spikes).

Migrated entries are NOT trusted blindly: `verify_cache.py` re-runs whole sessions from the NWBs
and checks the result matches. Run that before believing anything here.

    uv run python throwaway/migrate_cache.py --source-dir <old repo>/outputs/results/pynapple_m3/cache
"""

import dataclasses
import json
from pathlib import Path

import numpy as np
import tyro

from hdunique import loader, rates, sweep
from hdunique.config import DiffusionConfig
from hdunique.env import cache_dir


@dataclasses.dataclass(frozen=True)
class MigrateConfig:
    """Where the old cache lives, and whether to actually write."""

    source_dir: Path
    dry_run: bool = False


def _cfg_for(*, meta: dict[str, object]) -> DiffusionConfig:
    """Rebuild the config that produced a cached entry, from its stored metadata."""
    return DiffusionConfig(
        mouse=int(meta["mouse"]),
        session=int(meta["session"]),
        cell_areas=tuple(str(meta["cell_set"]).split("+")),
        n_restarts=int(meta["n_restarts"]),
        n_refits=int(meta["n_iter_isomap_fit"]),
        fit_frac=float(meta["fit_frac"]),
    )


def migrate_one(*, path: Path, dry_run: bool) -> str:
    """Migrate a single old cache file. Returns a one-line status."""
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    cfg = _cfg_for(meta=meta)

    # The old defaults must match this repo's, or the cached decodes do not describe our pipeline.
    defaults = DiffusionConfig()
    mismatched = {
        name: (getattr(cfg, name), getattr(defaults, name))
        for name in ("n_restarts", "n_refits", "fit_frac")
        if getattr(cfg, name) != getattr(defaults, name)
    }
    if mismatched:
        return f"SKIP {path.name}: settings differ from defaults {mismatched}"

    data = loader.load_session(mouse=cfg.mouse, session=cfg.session)
    rem = loader.load_state_epochs(data=data, state=cfg.rem_label)
    lengths = rates.bout_lengths(epochs=rem, dt=cfg.dt)

    embed = z["embed"].astype(float)
    lengths = sweep.truncate_bouts(lengths=lengths, n_kept=len(embed))
    if sum(lengths) != len(embed):
        return f"SKIP {path.name}: bout lengths sum to {sum(lengths)}, embedding has {len(embed)}"

    # Rename the old metadata key so the new schema is self-consistent.
    new_meta = {k: v for k, v in meta.items() if k != "n_iter_isomap_fit"}
    new_meta["n_refits"] = int(meta["n_iter_isomap_fit"])
    new_meta["rem_source"] = "dandi_states"
    new_meta.pop("paper_target", None)
    from hdunique.config import PAPER_REM_TARGETS

    new_meta["paper_target"] = PAPER_REM_TARGETS.get(str(meta["session_id"]))

    if not dry_run:
        sweep.save_cache(
            cfg=cfg,
            embed=embed,
            decoded=z["dec"].astype(float),
            bout_lengths=lengths,
            meta=new_meta,
        )
    return f"ok   {path.name}: {len(lengths)} bouts, {len(embed)} pts"


def run(*, cfg: MigrateConfig) -> None:
    """Migrate every entry in the source cache directory."""
    files = sorted(cfg.source_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files in {cfg.source_dir}")
    cache_dir().mkdir(parents=True, exist_ok=True)
    print(f"migrating {len(files)} entries -> {cache_dir()}")
    for path in files:
        print(" ", migrate_one(path=path, dry_run=cfg.dry_run))


def main() -> None:
    run(cfg=tyro.cli(MigrateConfig))


if __name__ == "__main__":
    main()
