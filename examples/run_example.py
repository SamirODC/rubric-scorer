"""
Minimal library usage example (no CLI).

Run with:  python examples/run_example.py
"""

import json
from pathlib import Path

from rubric_scorer.cli import load_criteria_set
from rubric_scorer.scorer import RubricScorer

HERE = Path(__file__).parent


def main() -> None:
    criteria_set = load_criteria_set(str(HERE / "criteria.json"))
    responses = json.loads((HERE / "responses.json").read_text())

    scorer = RubricScorer(criteria_set)
    results = scorer.score_batch(responses)

    for r in results:
        label = "HARD FAIL" if r.hard_failed else f"{r.normalized_score:.1f}/100"
        print(f"{r.response_id:24s} -> {label}")
        if r.failed_criteria:
            print(f"   failed criteria: {r.failed_criteria}")
        if r.hallucination_summary and r.hallucination_summary["total"]:
            print(f"   hallucinations: {r.hallucination_summary}")


if __name__ == "__main__":
    main()
