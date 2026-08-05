# On inconsistencies
- General:This document is going to underly our data curation. It needs to be 100% correct and up-to-date. If you find a disrepancy as you work, fix it unless it's due to a major methodological choice, in which case report it to the use immediately and together you will decide on how to resolve it and update the methods.
- External inconsistencies: If while you search for things outside of this repo, you find contradicting information to what is INSIDE the repo, flag it to the user and together we will update things accordingly.
- Internal inconsistencies: If two or more sources inside this repo contradict each other, check which one is most recently changed (perhaps something is stale) and perhaps easy to fix. If two methodologies lead to the generation of internal inconsistencies, then you need to revisit the methodologies with the user and resolve the internal inconsistencies.

# Checkability
- To ensure correctness, everything should be easily sourced. Results, findings, etc., should all point to the data and code that was used to generate them. We should be able to independently reproduce the results.

# Coding style
- Everything is typed.
- Return type is always stated explicitly (use `None` too).
- Use `jaxtyped` (with `beartype`) for any tensor-typed arguments or returns.

# Skills
- This repo uses pynapple package a lot. There is a Claude skill for that. Leverage it.

# Experiment workflow

**One question is the unit of work.** Every question has an id — `<topic><n>`, e.g. `diffusion1`,
`variance2` — and that id names all four of its assets. Never add a loose analysis script.

## Where things go

| Path | What lives there | Who writes it |
|---|---|---|
| `docs/exp_instructions/instructions-<qid>.md` | the question, its motivation, its experiments, its methods with code pointers | **you, agent-assisted** |
| `docs/exp_results/results_<qid>.in` | the results **template**: all prose and interpretation, with `@TOKEN@` where numbers go | **you, by hand** |
| `docs/exp_results/results_<qid>.md` | the rendered document | **generated — never edit** |
| `docs/fyi.md` | quirks in the data or tooling that would change how you write *unrelated* code | **you, as you find them** |
| `src/experiments/<qid>.py` | `Config`, optional `collect()`, `analyse()` | you |
| `src/core/` | configuration surface and path contract | rarely |
| `src/decode/` | raw recordings -> decoded angle, and the cache | rarely |
| `src/metrics/` | **the measured quantities — a new metric goes here** | you |
| `src/analysis/` | reusable primitives: `io`, `stats`, `values`, `render` | you, when logic would otherwise repeat |
| `src/figures/` | figure grammars, shared across questions | you |
| `src/collect/` | sweeps that build per-session tables from the decode cache | you |
| `outputs/cache/` | the decode cache — expensive, tracked, shared by every question | `hd-diffusion` |
| `outputs/results/` | per-session tables and `<qid>_values.json` | `analyse()` |
| `outputs/figures/<qid>_<expid>_<desc>.png` | figures | `analyse()` |

Every problem gets written up in the results document of the question that found it. A problem also
goes in `docs/fyi.md` when knowing it would change how someone writes code for a *different*
question — a data sentinel, a naming convention, a measure that is not what it looks like. Keep the
entry short and link the results document for the full story.

`docs/porting/`, `docs/long_D/` and `docs/bout_level/` are **historical narrative records** from
before this structure. Do not add to them.

## Adding a new question

1. **Pick an id** and add it to `QUESTION_IDS` in `src/experiments/__init__.py`.
2. **Write `docs/exp_instructions/instructions-<qid>.md`** first — question, motivation,
   one line per `<qid>_exp<n>`, then methods with explicit code pointers. Writing the methods
   before the code is what keeps the two honest.
3. **Write `src/experiments/<qid>.py`**:
   ```python
   QUESTION_ID = "<qid>"
   EXPERIMENTS = ("<qid>_exp1",)

   @dataclasses.dataclass(frozen=True)
   class Config:
       """Every field becomes a documented --flag via tyro."""
       cell_set: str = "ADn"

   def collect(*, cfg: Config) -> None: ...   # optional: only if new data is needed
   def analyse(*, cfg: Config, values: Values) -> None: ...
   ```
   Keep it **under ~120 lines**. If it grows past that, the logic belongs in `src/analysis/`.
   **Reuse aggressively** — most analyses differ only by config, so check `analysis/` and
   `figures/` before writing anything new.
4. **Emit values, never prose**: `values.scalar(...)`, `values.table(...)`, `values.figure(...)`.
5. **Write `docs/exp_results/results_<qid>.in`** — prose and interpretation, `@TOKEN@` for numbers.
6. **Run it**: `uv run hd-exp run <qid>`.

## Rules the tooling enforces

- **A template token with no value is a hard error.** A results file can never render with a gap.
- **A figure that is generated but never referenced is a hard error.** Use `@FIGURES@` to include
  every figure automatically, or place `@FIG_*@` individually.
- **`hd-exp check <qid>`** recomputes and diffs against the committed `values.json`, exiting
  non-zero on drift. Run it before committing a result.
- Every `values.json` records the resolved config, git commit and timestamp, and those appear as a
  provenance block in the `.md`. Never state a number without it.

## Figures

Be sparing. A figure earns its place by showing something a table cannot — spread, shape, overlap.
One per experiment at most, and usually fewer.

## Commands

```bash
uv run hd-exp list                 # every question and its experiments
uv run hd-exp collect <qid>        # only for questions that declare collect()
uv run hd-exp run     <qid>        # analyse + render
uv run hd-exp check   <qid>        # recompute and diff against committed values
uv run hd-diffusion --scope all    # the one expensive collector: NWB -> outputs/cache (hours)

cargo build --release --manifest-path rust/hd-exp-list/Cargo.toml   # optional: fast `list`
```

`hd-exp list` is implemented in Rust (`rust/hd-exp-list/`), because listing questions otherwise
had to import every question module — ~2.6 s of sklearn/scipy/pandas/pynapple — to read three
attributes off them. The binary reads `QUESTION_IDS`, `PLANNED`, each docstring, each
`EXPERIMENTS` and each `collect` out of the source text instead, in ~2 ms. `src/experiments/`
remains the single source of truth; the binary keeps no copy of the registry.

If the binary is not built, `hd-exp list` falls back to the original by-import implementation
(`cli.exp._list_python`) and behaves identically, just slowly. `tests/test_exp_list.py` pins the
two to byte-identical output, so a question module written in a form the parser cannot read fails
the suite rather than silently vanishing from the listing.

The other sub-commands genuinely need the scientific stack, so they still pay for it — but their
imports now live inside the functions that use them, not at module scope.

Modules are top-level names (`config`, `env`, `loader`), so an inherited `PYTHONPATH` from another
checkout can shadow them; `env.py` raises with an explanation if it detects that.
