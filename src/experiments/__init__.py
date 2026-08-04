"""Registry of scientific questions.

Each question is one module exposing `QUESTION_ID`, `EXPERIMENTS`, a `Config` dataclass and an
`analyse(*, cfg, values)` function; optionally also `collect(*, cfg)` when it needs data beyond the
shared decode cache. `hd-exp` discovers everything from here, so adding a question is adding a
module and one line below.
"""

import importlib
from types import ModuleType

#: Question ids, in the order `hd-exp list` shows them.
QUESTION_IDS: tuple[str, ...] = (
    "diffusion1",
    "diffusion2",
    "variance1",
    "variance2",
    "bouts1",
)

#: Questions whose module is not written yet. Listed so `hd-exp list` reports them as planned
#: rather than crashing, and so the registry stays the single record of what exists.
PLANNED: frozenset[str] = frozenset()


def load(question_id: str) -> ModuleType:
    """Import a question's module, with an actionable error for an unknown id."""
    if question_id not in QUESTION_IDS:
        raise KeyError(
            f"Unknown question {question_id!r}. Known: {', '.join(QUESTION_IDS)}. "
            "Add new ones to experiments/QUESTION_IDS."
        )
    if question_id in PLANNED:
        raise NotImplementedError(
            f"{question_id} is registered but not implemented yet. "
            "See docs/exp_instructions/ for the questions that are."
        )
    return importlib.import_module(f"experiments.{question_id}")
