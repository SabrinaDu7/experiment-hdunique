"""The render step is what stands between an analysis and a published number, so its failure
modes matter more than its happy path: a missing value must stop the build, not leave a gap."""

import json
from pathlib import Path

import pandas as pd
import pytest

from analysis.render import render
from analysis.values import Values


def _values(**tokens: str) -> dict:
    return {"question_id": "q1", "generated": "2026-08-04T00:00:00+00:00",
            "git_commit": "abc1234", "config": {"cell_set": "ADn"}, "notes": {}, "tokens": tokens}


def _docs(tmp_path: Path, template: str) -> Path:
    (tmp_path / "exp_results").mkdir(parents=True)
    (tmp_path / "exp_results" / "results_q1.in").write_text(template)
    return tmp_path


def test_substitutes_tokens_and_dates_the_file(tmp_path: Path) -> None:
    docs = _docs(tmp_path, "# Q1\n\nMedian was @D_MEDIAN@ rad^2/s.\n")
    out = render(question_id="q1", values=_values(D_MEDIAN="0.90"), docs_root=docs)
    text = out.read_text()
    assert text.splitlines()[0].count("-") == 2, "first line must be the generation date"
    assert "Median was 0.90 rad^2/s." in text
    assert "@D_MEDIAN@" not in text


def test_missing_token_is_an_error(tmp_path: Path) -> None:
    docs = _docs(tmp_path, "Median @D_MEDIAN@, spread @D_IQR@.\n")
    with pytest.raises(KeyError, match="D_IQR"):
        render(question_id="q1", values=_values(D_MEDIAN="0.90"), docs_root=docs)


def test_missing_template_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "exp_results").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="results_q1.in"):
        render(question_id="q1", values=_values(), docs_root=tmp_path)


def test_provenance_records_the_config(tmp_path: Path) -> None:
    docs = _docs(tmp_path, "# Q1\n")
    text = render(question_id="q1", values=_values(), docs_root=docs).read_text()
    assert "`cell_set`" in text and "`ADn`" in text
    assert "abc1234" in text


def test_generated_banner_warns_against_editing(tmp_path: Path) -> None:
    docs = _docs(tmp_path, "# Q1\n")
    assert "do not edit" in render(question_id="q1", values=_values(), docs_root=docs).read_text()


def test_values_rejects_duplicate_token() -> None:
    v = Values(question_id="q1", config={})
    v.scalar("A", 1.0)
    with pytest.raises(KeyError, match="twice"):
        v.scalar("A", 2.0)


def test_values_table_renders_markdown() -> None:
    v = Values(question_id="q1", config={})
    v.table("T", pd.DataFrame({"mouse": [12, 17], "D": [0.5, 1.25]}), floatfmt=".2f")
    assert "| mouse | D |" in v.tokens["T"]
    assert "| 12 | 0.50 |" in v.tokens["T"]


def test_values_roundtrip_is_json_safe() -> None:
    v = Values(question_id="q1", config={"windows": [200, 500]})
    v.scalar("A", 1.0)
    assert json.loads(json.dumps(v.to_dict()))["tokens"]["A"] == "1.000"


def test_figure_names_the_file_it_points_at(tmp_path: Path) -> None:
    """A results document must say which file a figure is, not just embed it."""
    v = Values(question_id="q1", config={})
    v.figure("FIG_WINDOWS", tmp_path / "q1_exp1_windows.png", caption="windows")
    rendered = v.tokens["FIG_WINDOWS"]
    assert "q1_exp1_windows.png" in rendered, "the filename must appear in the document"
    assert "![windows](" in rendered
    assert v.notes["figures"]["FIG_WINDOWS"].endswith("q1_exp1_windows.png")


def test_figure_token_is_forced_into_the_prefix_render_looks_for(tmp_path: Path) -> None:
    """A figure recorded under a name without the `FIG_` prefix must still be found.

    `@FIGURES@` collects tokens by that prefix, and the unreferenced-figure error keys off it too.
    A figure named anything else is invisible to both: it renders nothing and raises nothing, which
    is exactly how two real figures went missing from a results document.
    """
    v = Values(question_id="q1", config={})
    v.figure("BOUT_TRACE", tmp_path / "q1_exp3_traces.png", caption="traces")
    assert "FIG_BOUT_TRACE" in v.tokens
    assert [k for k in v.tokens if k.startswith("FIG_")] == ["FIG_BOUT_TRACE"]


def test_figures_token_includes_every_figure(tmp_path: Path) -> None:
    """@FIGURES@ saves the template from naming each figure when placement does not matter."""
    docs = _docs(tmp_path, "# Q1\n\n@FIGURES@\n")
    vals = _values(FIG_A="![a](../../a.png)", FIG_B="![b](../../b.png)")
    text = render(question_id="q1", values=vals, docs_root=docs).read_text()
    assert "![a](../../a.png)" in text and "![b](../../b.png)" in text


def test_unreferenced_figure_is_an_error(tmp_path: Path) -> None:
    """A generated figure the document never mentions is as wrong as a stale number."""
    docs = _docs(tmp_path, "# Q1\n\nno figures here\n")
    with pytest.raises(KeyError, match="FIG_A"):
        render(question_id="q1", values=_values(FIG_A="![a](../../a.png)"), docs_root=docs)


def test_shards_merge_back_into_one_table(tmp_path: Path, monkeypatch) -> None:
    """A sweep split across processes must read back as if it had run in one.

    Each worker writes its own table; the analysis is written against a single frame and should not
    know the difference. Overlapping keys resolve to the last write, so re-running one shard fixes
    that shard without duplicating its rows.
    """
    import pandas as pd

    from analysis import io

    monkeypatch.setenv("OUTPUT_PATH", str(tmp_path))
    (tmp_path / "results").mkdir(parents=True, exist_ok=True)

    io.save_table(frame=pd.DataFrame({"mouse": [12], "session": [1], "v": [1.0]}),
                  name=io.shard_name(name="t", shard=0, n_shards=3))
    io.save_table(frame=pd.DataFrame({"mouse": [17], "session": [2], "v": [2.0]}),
                  name=io.shard_name(name="t", shard=1, n_shards=3))

    merged = io.load_shards(name="t")
    assert list(merged["mouse"]) == [12, 17]
    assert io.load_shards(name="absent") is None

    # A shard re-run must replace its rows, not append them.
    io.save_table(frame=pd.DataFrame({"mouse": [17], "session": [2], "v": [99.0]}),
                  name=io.shard_name(name="t", shard=1, n_shards=3))
    again = io.load_shards(name="t")
    assert len(again) == 2 and float(again.loc[again["mouse"] == 17, "v"].iloc[0]) == 99.0
