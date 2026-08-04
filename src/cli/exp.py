"""`hd-exp` — run a question end to end.

    uv run hd-exp list
    uv run hd-exp run diffusion1                      # analyse + render
    uv run hd-exp run diffusion1 --cell-set PoS       # every config field is a flag
    uv run hd-exp collect  bouts1                     # only for questions that need new data
    uv run hd-exp analyse  variance1
    uv run hd-exp render   variance1
    uv run hd-exp check    variance1                  # re-run and diff against committed values

Each question owns its own `Config` dataclass, so `tyro` exposes exactly that question's
parameters — no shared flag surface that half the questions ignore.

`check` is what makes correctness mechanical rather than a judgement call: it recomputes and
compares against the committed `values.json`, exiting non-zero on any drift.
"""

import dataclasses
import sys
from pathlib import Path
from typing import Any

import tyro

import experiments
from analysis.render import render
from analysis.values import Values, load_values

#: Repository root, so docs are found regardless of the working directory.
DOCS_ROOT = Path(__file__).resolve().parent.parent.parent / "docs"


def _config_for(*, question_id: str, argv: list[str]) -> Any:
    """Parse the question's own Config from the remaining command-line arguments."""
    module = experiments.load(question_id)
    return tyro.cli(module.Config, args=argv)


def cmd_list() -> int:
    """Print every registered question, its experiments, and whether it has a collect step."""
    for question_id in experiments.QUESTION_IDS:
        module = experiments.load(question_id)
        collect = "collect+analyse" if hasattr(module, "collect") else "analyse only"
        doc = (module.__doc__ or "").strip().splitlines()
        print(f"  {question_id:12s} [{collect:15s}] {doc[0] if doc else ''}")
        for name in getattr(module, "EXPERIMENTS", ()):
            print(f"      {name}")
    return 0


def cmd_collect(*, question_id: str, argv: list[str]) -> int:
    """Run a question's data-collection step, if it declares one."""
    module = experiments.load(question_id)
    if not hasattr(module, "collect"):
        print(f"{question_id} has no collect step; it reads the shared decode cache. "
              f"Run `hd-exp analyse {question_id}`.")
        return 0
    module.collect(cfg=_config_for(question_id=question_id, argv=argv))
    return 0


def cmd_analyse(*, question_id: str, argv: list[str]) -> int:
    """Run a question's analysis and write its values file."""
    module = experiments.load(question_id)
    cfg = _config_for(question_id=question_id, argv=argv)
    values = Values(question_id=question_id, config=_as_dict(cfg))
    module.analyse(cfg=cfg, values=values)
    path = values.save()
    print(f"  {len(values.tokens)} value(s) -> {path.name}")
    return 0


def cmd_render(*, question_id: str) -> int:
    """Render a question's results document from its template and values."""
    out = render(
        question_id=question_id, values=load_values(question_id=question_id), docs_root=DOCS_ROOT
    )
    print(f"  rendered {out.relative_to(DOCS_ROOT.parent)}")
    return 0


def cmd_check(*, question_id: str, argv: list[str]) -> int:
    """Recompute and compare against the committed values file; non-zero exit on drift."""
    before = load_values(question_id=question_id)["tokens"]
    module = experiments.load(question_id)
    cfg = _config_for(question_id=question_id, argv=argv)
    values = Values(question_id=question_id, config=_as_dict(cfg))
    module.analyse(cfg=cfg, values=values)
    after = values.tokens

    drift = sorted(
        k for k in set(before) | set(after) if before.get(k) != after.get(k)
    )
    if drift:
        print(f"  DRIFT in {len(drift)} value(s) for {question_id}:")
        for key in drift:
            print(f"    {key}: committed {before.get(key, '<absent>')!r} "
                  f"-> recomputed {after.get(key, '<absent>')!r}")
        return 1
    print(f"  {question_id}: all {len(after)} values reproduce exactly")
    return 0


def _as_dict(cfg: object) -> dict[str, Any]:
    """A config as a JSON-safe dict, for the provenance record."""
    return {
        k: list(v) if isinstance(v, tuple) else v
        for k, v in dataclasses.asdict(cfg).items()  # type: ignore[call-overload]
    }


def main() -> None:
    """Dispatch on the sub-command, leaving the rest of argv for the question's own config."""
    argv = sys.argv[1:]
    if not argv or argv[0] in {"-h", "--help", "list"}:
        if not argv or argv[0] == "list":
            raise SystemExit(cmd_list())
        print(__doc__)
        raise SystemExit(0)

    command, rest = argv[0], argv[1:]
    if command not in {"collect", "analyse", "render", "run", "check"}:
        print(f"Unknown command {command!r}. Use: list, collect, analyse, render, run, check.")
        raise SystemExit(2)
    if not rest:
        print(f"`hd-exp {command}` needs a question id. Try `hd-exp list`.")
        raise SystemExit(2)

    question_id, flags = rest[0], rest[1:]
    if command == "render":
        raise SystemExit(cmd_render(question_id=question_id))
    if command == "collect":
        raise SystemExit(cmd_collect(question_id=question_id, argv=flags))
    if command == "analyse":
        raise SystemExit(cmd_analyse(question_id=question_id, argv=flags))
    if command == "check":
        raise SystemExit(cmd_check(question_id=question_id, argv=flags))

    code = cmd_analyse(question_id=question_id, argv=flags)
    raise SystemExit(code or cmd_render(question_id=question_id))


if __name__ == "__main__":
    main()
