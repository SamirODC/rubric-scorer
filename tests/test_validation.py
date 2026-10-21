import math

import pytest

from rubric_scorer import CriteriaSet, Criterion, RubricScorer
from rubric_scorer.hallucination import (
    HallucinationAnnotation,
    HallucinationCategory,
    Severity,
    summarize_hallucinations,
)


def rubric():
    return CriteriaSet("validation").add(Criterion("correct", "Correct answer", 5))


@pytest.mark.parametrize("weight", [math.nan, math.inf, -math.inf])
def test_non_finite_weights_rejected(weight):
    with pytest.raises(ValueError, match="finite"):
        Criterion("bad", "Invalid weight", weight)


@pytest.mark.parametrize("weight", [True, "5", None])
def test_non_numeric_weights_rejected(weight):
    with pytest.raises(TypeError, match="weight"):
        Criterion("bad", "Invalid weight", weight)


def test_duplicate_ids_rejected_on_add_and_direct_construction():
    cs = rubric()
    with pytest.raises(ValueError, match="Duplicate"):
        cs.add(Criterion("correct", "Duplicate", 2))
    with pytest.raises(ValueError, match="unique"):
        RubricScorer(CriteriaSet("duplicates", [cs.criteria[0], cs.criteria[0]]))


@pytest.mark.parametrize("criteria", [[], [Criterion("penalty", "Penalty", -5)]])
def test_undefined_normalization_rejected(criteria):
    with pytest.raises(ValueError):
        RubricScorer(CriteriaSet("invalid", criteria))


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None, []])
def test_non_boolean_judgments_rejected(value):
    with pytest.raises(TypeError, match="booleans"):
        RubricScorer(rubric()).score("r", {"correct": value})


def test_unknown_judgment_rejected():
    with pytest.raises(ValueError, match="Unknown criterion"):
        RubricScorer(rubric()).score("r", {"typo": True})


def test_negative_weight_is_a_penalty_when_true_and_score_clips_at_zero():
    cs = rubric().add(Criterion("violation", "Violation present", -5))
    result = RubricScorer(cs).score("r", {"correct": True, "violation": True})
    assert result.raw_score == 0
    assert result.normalized_score == 0
    assert not result.hard_failed
    result = RubricScorer(cs).score("r", {"correct": False, "violation": True})
    assert result.raw_score == -5
    assert result.normalized_score == 0


def test_missing_noncritical_judgment_is_documented_without_automatic_hard_fail():
    result = RubricScorer(rubric()).score("r", {})
    assert result.failed_criteria == ["correct"]
    assert result.normalized_score == 0
    assert not result.hard_failed
    assert "Missing evaluation" in result.notes[0]


def test_scorer_snapshots_rubric_to_prevent_later_mutation():
    cs = rubric()
    scorer = RubricScorer(cs)
    cs.add(Criterion("extra", "Added later", 1))
    assert scorer.score("r", {"correct": True}).normalized_score == 100


def test_batch_converts_annotations_and_is_json_serializable():
    import json

    rows = [{"response_id": "r", "results": {"correct": True},
             "hallucinations": [{"category": "external_facts", "severity": "critical",
                                 "span": "invented claim", "explanation": "Unsupported"}]}]
    result = RubricScorer(rubric()).score_batch(rows)[0]
    assert result.normalized_score == 0
    assert result.hard_failed
    assert json.loads(json.dumps(result.to_dict()))["hallucination_summary"]["total"] == 1


def test_minor_annotation_does_not_automatically_zero_score():
    label = HallucinationAnnotation("grounding", "minor", "claim", "Weak evidence")
    result = RubricScorer(rubric()).score("r", {"correct": True}, [label])
    assert result.normalized_score == 100
    assert result.hallucination_summary["by_category"]["grounding"] == 1
    assert not result.hallucination_summary["has_critical"]


@pytest.mark.parametrize("category,severity", [("unknown", "minor"), ("grounding", "bad")])
def test_unknown_annotation_labels_rejected(category, severity):
    with pytest.raises(ValueError):
        HallucinationAnnotation(category, severity, "text", "explanation")


@pytest.mark.parametrize("index", [-1, True, 1.5])
def test_invalid_annotation_step_rejected(index):
    with pytest.raises(ValueError, match="step_index"):
        HallucinationAnnotation("grounding", "minor", "claim", "explanation", index)


def test_summary_counts_all_categories_and_severities():
    labels = [
        HallucinationAnnotation(HallucinationCategory.GROUNDING, Severity.MAJOR, "a", "b"),
        HallucinationAnnotation(HallucinationCategory.GROUNDING, Severity.CRITICAL, "c", "d"),
    ]
    summary = summarize_hallucinations(labels)
    assert summary["total"] == 2
    assert summary["by_category"]["grounding"] == 2
    assert summary["by_category"]["exploration"] == 0
    assert summary["by_severity"]["critical"] == 1
    assert summary["has_critical"]


@pytest.mark.parametrize("row", [None, {}, {"response_id": "r"}])
def test_invalid_batch_rows_rejected(row):
    with pytest.raises((TypeError, ValueError)):
        RubricScorer(rubric()).score_batch([row])
