"""Collecting the numbers an experiment produces, for substitution into its results template.

An analysis never writes prose. It writes **values** — named scalars, tables and figure paths —
into `outputs/results/<question_id>_values.json`. The results `.md` is then rendered from a
hand-written `.in` template that references those names as `@TOKEN@`.

That split is the point: re-running an analysis refreshes every number and **cannot touch a word of
interpretation**, while a number that no longer has a value becomes a hard error at render time
rather than a stale figure silently surviving in the document.

Alongside the values, every file records the provenance needed to answer "which settings produced
this?" — the resolved config, the git commit, the timestamp. That question came up repeatedly while
this analysis was being built, and it is answered here structurally rather than by memory.
"""

import dataclasses
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from env import results_dir


def _git_commit() -> str:
    """Short hash of HEAD, or "unknown" outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, cwd=Path(__file__).parent,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _markdown_table(*, frame: pd.DataFrame, floatfmt: str) -> str:
    """Render a frame as a GitHub-flavoured markdown table."""
    def cell(v: object) -> str:
        return format(v, floatfmt) if isinstance(v, float) else str(v)

    header = list(frame.columns)
    rows = [[cell(v) for v in row] for row in frame.itertuples(index=False)]
    lines = ["| " + " | ".join(str(h) for h in header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


@dataclasses.dataclass
class Values:
    """Named results for one question, plus the provenance describing how they were made.

    Tokens are recorded as already-formatted strings, so the formatting decision lives next to the
    computation that knows what precision is meaningful — not in the template, and not in whatever
    the renderer happens to default to.
    """

    question_id: str
    config: dict[str, Any]
    tokens: dict[str, str] = dataclasses.field(default_factory=dict)
    notes: dict[str, Any] = dataclasses.field(default_factory=dict)

    def _set(self, name: str, rendered: str) -> None:
        if name in self.tokens:
            raise KeyError(f"{self.question_id}: token {name!r} set twice")
        self.tokens[name] = rendered

    def scalar(self, name: str, value: float | str, *, fmt: str = ".3f") -> None:
        """Record a single number (or short string) under `name`."""
        self._set(name, format(value, fmt) if isinstance(value, float) else str(value))

    def table(self, name: str, frame: pd.DataFrame, *, floatfmt: str = ".3f") -> None:
        """Record a dataframe under `name`, rendered as a markdown table."""
        self._set(name, _markdown_table(frame=frame, floatfmt=floatfmt))

    def figure(self, name: str, path: Path, *, caption: str = "") -> None:
        """Record a figure under `name`, rendered as a markdown image link.

        The path is stored relative to the repository root so the link resolves both on disk and on
        a git host, whatever `$OUTPUT_PATH` happens to be on the machine that produced it.
        """
        repo_root = Path(__file__).resolve().parent.parent.parent
        try:
            rel = path.resolve().relative_to(repo_root)
        except ValueError:
            rel = path
        self._set(name, f"![{caption or name}](../../{rel})")

    def note(self, name: str, value: Any) -> None:
        """Record a value for provenance only — not substitutable into the template."""
        self.notes[name] = value

    def to_dict(self) -> dict[str, Any]:
        """The full record, values plus provenance."""
        return {
            "question_id": self.question_id,
            "generated": datetime.now(UTC).isoformat(timespec="seconds"),
            "git_commit": _git_commit(),
            "config": self.config,
            "notes": self.notes,
            "tokens": self.tokens,
        }

    def path(self) -> Path:
        """Where this question's values live."""
        return results_dir() / f"{self.question_id}_values.json"

    def save(self) -> Path:
        """Write the values file and return its path."""
        results_dir().mkdir(parents=True, exist_ok=True)
        target = self.path()
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        return target


def load_values(*, question_id: str) -> dict[str, Any]:
    """Read a previously saved values file."""
    path = results_dir() / f"{question_id}_values.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No values for {question_id!r} at {path}. Run `hd-exp analyse {question_id}` first."
        )
    return json.loads(path.read_text())
