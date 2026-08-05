"""The Rust `list` must agree with the Python one it replaced.

`_list_python` imports each question module and reads its real attributes; the Rust binary parses
the same facts out of the source text. That is a duplication of logic, so it is pinned here: if a
question module ever uses a form the parser does not understand, this fails rather than quietly
dropping a line from `hd-exp list`.
"""

import io
import subprocess
from contextlib import redirect_stdout

import pytest

from cli.exp import LIST_BIN, REPO_ROOT, _list_python


def test_rust_list_matches_python_list() -> None:
    """Both implementations produce byte-identical output."""
    if not LIST_BIN.is_file():
        pytest.skip(f"{LIST_BIN.relative_to(REPO_ROOT)} not built; run cargo build --release")

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        assert _list_python() == 0
    expected = buffer.getvalue()

    result = subprocess.run(
        [str(LIST_BIN), str(REPO_ROOT)], capture_output=True, text=True, check=True
    )
    assert result.stdout == expected


def test_rust_list_reads_the_registry() -> None:
    """Every registered id appears, so the binary is not carrying a stale copy of QUESTION_IDS."""
    if not LIST_BIN.is_file():
        pytest.skip(f"{LIST_BIN.relative_to(REPO_ROOT)} not built; run cargo build --release")

    import experiments

    result = subprocess.run(
        [str(LIST_BIN), str(REPO_ROOT)], capture_output=True, text=True, check=True
    )
    for question_id in experiments.QUESTION_IDS:
        assert question_id in result.stdout
