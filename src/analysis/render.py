"""Rendering a results document from its template and the values an analysis produced.

`docs/exp_results/results_<qid>.in` is hand-written: every section, every sentence of
interpretation, with `@TOKEN@` wherever a number belongs. `outputs/results/<qid>_values.json` holds
those numbers. This module substitutes one into the other to produce `results_<qid>.md`.

Two rules make the output trustworthy:

- **An unresolved token is an error.** If the template asks for `@D_MEDIAN@` and the analysis did
  not produce it, rendering fails loudly. A results file therefore cannot quietly contain a gap, and
  cannot quietly keep a number whose analysis has been removed.
- **The `.md` is never edited by hand.** It carries a banner saying so. All prose belongs in the
  `.in`, which no script ever writes.
"""

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TOKEN = re.compile(r"@([A-Z0-9_]+)@")

#: Written at the top of every generated file, under the date.
BANNER = "<!-- GENERATED from {stem}.in — do not edit; edit the .in and re-run `hd-exp render`. -->"


def instructions_path(*, question_id: str, docs_root: Path) -> Path:
    """Where a question's instructions live."""
    return docs_root / "exp_instructions" / f"instructions-{question_id}.md"


def template_path(*, question_id: str, docs_root: Path) -> Path:
    """Where a question's results template lives."""
    return docs_root / "exp_results" / f"results_{question_id}.in"


def output_path(*, question_id: str, docs_root: Path) -> Path:
    """Where a question's rendered results land."""
    return docs_root / "exp_results" / f"results_{question_id}.md"


def provenance_block(*, values: dict[str, Any]) -> str:
    """A short, uniform footer naming exactly what produced these numbers.

    Rendered into every results document because "which settings gave this number?" is the question
    that is hardest to answer after the fact and cheapest to answer at write time.
    """
    config = values.get("config", {})
    rows = "\n".join(f"| `{k}` | `{v}` |" for k, v in sorted(config.items()))
    notes = values.get("notes", {})
    note_rows = "\n".join(f"| `{k}` | `{v}` |" for k, v in sorted(notes.items()))
    return (
        "## Provenance\n\n"
        f"Generated {values.get('generated', '?')} from commit `{values.get('git_commit', '?')}`.\n\n"
        "| config | value |\n|---|---|\n" + rows + "\n"
        + (f"\n| input | value |\n|---|---|\n{note_rows}\n" if note_rows else "")
    )


def render(*, question_id: str, values: dict[str, Any], docs_root: Path) -> Path:
    """Substitute `values` into the question's template and write the `.md`.

    Raises if the template is missing, or if it references a token the analysis did not produce.
    """
    template = template_path(question_id=question_id, docs_root=docs_root)
    if not template.exists():
        raise FileNotFoundError(
            f"No template for {question_id!r} at {template}. "
            "Write the .in first — it holds the prose; the analysis only supplies numbers."
        )

    tokens = dict(values.get("tokens", {}))
    body = template.read_text()

    # `@FIGURES@` is a convenience: every figure this analysis produced, in the order it made them,
    # for templates that do not care where each one lands. Placing `@FIG_*@` individually still
    # works and gives control over position.
    figure_tokens = [k for k in tokens if k.startswith("FIG_")]
    if "@FIGURES@" in body:
        tokens["FIGURES"] = "\n\n".join(tokens[k] for k in figure_tokens)

    missing = sorted({m.group(1) for m in TOKEN.finditer(body)} - set(tokens))
    if missing:
        raise KeyError(
            f"{question_id}: template references {len(missing)} token(s) the analysis did not "
            f"produce: {', '.join(missing)}. Either the analysis changed or the template is ahead "
            "of it; a results file must never render with a gap."
        )

    referenced = {m.group(1) for m in TOKEN.finditer(body)}

    # A figure that was generated but never referenced would sit on disk unmentioned, which is the
    # same failure as a stale number: the document would not describe what the analysis produced.
    orphan_figures = sorted(set(figure_tokens) - referenced) if "FIGURES" not in tokens else []
    if orphan_figures:
        raise KeyError(
            f"{question_id}: the analysis produced {len(orphan_figures)} figure(s) the template "
            f"never references: {', '.join(orphan_figures)}. Add the token(s), or use @FIGURES@ "
            "to include every figure automatically."
        )

    rendered = TOKEN.sub(lambda m: tokens[m.group(1)], body)
    # `@FIGURES@` consumes every FIG_ token, so those are used even though they are not named.
    consumed = set(figure_tokens) if "FIGURES" in tokens else set()
    unused = sorted(set(tokens) - referenced - consumed - {"FIGURES"})

    date = datetime.now(UTC).strftime("%Y-%m-%d")
    head = f"{date}\n\n{BANNER.format(stem=f'results_{question_id}')}\n\n"
    out = output_path(question_id=question_id, docs_root=docs_root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(head + rendered.rstrip() + "\n\n" + provenance_block(values=values))

    if unused:
        print(f"  note: {len(unused)} value(s) computed but not used by the template: "
              f"{', '.join(unused)}")
    return out
