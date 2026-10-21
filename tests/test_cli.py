import json
import subprocess
import sys
from pathlib import Path

import pytest

from rubric_scorer.cli import load_criteria_set, main

ROOT = Path(__file__).resolve().parents[1]
CRITERIA = str(ROOT / "examples" / "criteria.json")
RESPONSES = str(ROOT / "examples" / "responses.json")


def test_cli_score_json_matches_documented_examples(capsys):
    assert main(["score", "--criteria", CRITERIA, "--responses", RESPONSES, "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [r["normalized_score"] for r in rows] == [100, 56, 0]
    assert rows[-1]["hard_failed"]


def test_cli_human_readable_output(capsys):
    assert main(["score", "--criteria", CRITERIA, "--responses", RESPONSES]) == 0
    out = capsys.readouterr().out
    assert "84% / 16% (OK)" in out
    assert "56.0/100" in out
    assert "HARD FAIL" in out


def test_cli_validate_success(capsys):
    assert main(["validate-criteria", "--criteria", CRITERIA]) == 0
    assert capsys.readouterr().out.strip().endswith("OK")


def test_ratio_failure_uses_distinct_exit_code(tmp_path, capsys):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"name": "trace-only", "criteria": [
        {"id": "a", "description": "A", "weight": 5, "category": "trace"}
    ]}), encoding="utf-8")
    assert main(["validate-criteria", "--criteria", str(path)]) == 1
    assert "outside" in capsys.readouterr().out


@pytest.mark.parametrize("data", [[], {}, {"name": "empty", "criteria": []},
                                    {"name": "wrong-type", "criteria": "bad"}])
def test_invalid_criteria_shapes_rejected(tmp_path, data):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        load_criteria_set(path)


def test_malformed_json_has_clear_error_and_no_traceback(tmp_path, capsys):
    path = tmp_path / "broken.json"
    path.write_text("{broken", encoding="utf-8")
    assert main(["validate-criteria", "--criteria", str(path)]) == 2
    err = capsys.readouterr().err
    assert "Error:" in err
    assert "Traceback" not in err


def test_missing_file_returns_error(tmp_path, capsys):
    assert main(["validate-criteria", "--criteria", str(tmp_path / "missing.json")]) == 2
    assert "Error:" in capsys.readouterr().err


def test_malformed_annotation_is_reported_without_traceback(tmp_path, capsys):
    path = tmp_path / "responses.json"
    path.write_text(json.dumps([{"response_id": "r", "results": {},
                                 "hallucinations": [{"category": "bad"}]}]), encoding="utf-8")
    assert main(["score", "--criteria", CRITERIA, "--responses", str(path)]) == 2
    assert "Error:" in capsys.readouterr().err


def test_module_entrypoint_end_to_end():
    process = subprocess.run(
        [sys.executable, "-m", "rubric_scorer.cli", "score", "--criteria", CRITERIA,
         "--responses", RESPONSES, "--json"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)[0]["normalized_score"] == 100
