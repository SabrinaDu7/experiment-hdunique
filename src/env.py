"""Environment contract: where the data lives and where results go."""

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

#: Environment variables this repo reads. See `.envrc.example`.
REQUIRED_VARS: tuple[str, ...] = ("DANDI_DATA_ROOT", "OUTPUT_PATH")


def get_env_var(*, name: str) -> Path:
    """Read a path-valued environment variable, falling back to parsing `.envrc` at the repo root."""
    if name not in os.environ:
        envrc = _REPO_ROOT / ".envrc"
        if envrc.exists():
            for line in envrc.read_text().splitlines():
                if line.startswith(f"export {name}="):
                    os.environ[name] = line.split("=", 1)[1].strip().strip('"')
                    break
    if name not in os.environ:
        raise KeyError(
            f"{name} is not set. Copy .envrc.example to .envrc and edit it, "
            f"or export {name} directly. Required: {', '.join(REQUIRED_VARS)}."
        )
    return Path(os.environ[name]).expanduser()


def dandi_root() -> Path:
    """Root of the local DANDI 000056 download (contains `sub-Mouse*/`)."""
    return get_env_var(name="DANDI_DATA_ROOT")


def results_dir() -> Path:
    """Directory holding parquets, the decode cache and figures."""
    return get_env_var(name="OUTPUT_PATH") / "results"


def cache_dir() -> Path:
    """Directory holding the per-run Isomap embedding + decoded-angle cache."""
    return results_dir() / "cache"
