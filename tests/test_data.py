import json

import pytest

from inkling_qlora.data import read_records


def test_read_records(tmp_path) -> None:
    p = tmp_path / "d.jsonl"
    row = {"prompt": [{"role": "user", "content": "q"}], "completion": [{"role": "assistant", "content": "a"}]}
    p.write_text(json.dumps(row) + "\n\n" + json.dumps(row) + "\n", encoding="utf-8")
    assert read_records(p) == [row, row]


def test_read_records_rejects_plain_strings(tmp_path) -> None:
    p = tmp_path / "d.jsonl"
    p.write_text(json.dumps({"prompt": "q", "completion": "a"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="prompt"):
        read_records(p)
