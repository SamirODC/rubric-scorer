import pytest

from rubric_scorer.criteria import Criterion, CriteriaSet, CriterionCategory
from rubric_scorer.hallucination import (
    HallucinationAnnotation,
    HallucinationCategory,
    Severity,
)
from rubric_scorer.scorer import RubricScorer


def make_basic_set() -> CriteriaSet:
    cs = CriteriaSet(name="test-rubric")
    cs.add(Criterion("answers_question", "Answers the question", 4,
                      CriterionCategory.OUTCOME))
    cs.add(Criterion("no_fabrication", "No fabricated facts", 5,
                      CriterionCategory.OUTCOME, hard_fail=True))
    cs.add(Criterion("clear_reasoning", "Reasoning steps are clear", 2,
                      CriterionCategory.TRACE))
    return cs


class TestCriterion:
    def test_weight_out_of_range_raises(self):
        with pytest.raises(ValueError):
            Criterion("bad", "desc", 10)

    def test_valid_weight_ok(self):
        c = Criterion("ok", "desc", -5)
        assert c.weight == -5


class TestCriteriaSet:
    def test_outcome_trace_ratio(self):
        cs = make_basic_set()
        outcome_share, trace_share = cs.outcome_trace_ratio()
        assert outcome_share == pytest.approx(9 / 11)
        assert trace_share == pytest.approx(2 / 11)

    def test_validate_ratio_within_tolerance(self):
        cs = make_basic_set()
        # 9/11 ~= 0.818, within 0.15 of 0.8
        assert cs.validate_ratio(target_outcome_share=0.8, tolerance=0.15)

    def test_validate_ratio_out_of_tolerance(self):
        cs = make_basic_set()
        assert not cs.validate_ratio(target_outcome_share=0.5, tolerance=0.05)

    def test_hard_fail_criteria(self):
        cs = make_basic_set()
        ids = [c.id for c in cs.hard_fail_criteria()]
        assert ids == ["no_fabrication"]

    def test_max_min_possible(self):
        cs = make_basic_set()
        assert cs.max_possible_score() == 11
        assert cs.min_possible_score() == 0


class TestRubricScorer:
    def test_full_pass_scores_100(self):
        cs = make_basic_set()
        scorer = RubricScorer(cs)
        result = scorer.score(
            "r1",
            {"answers_question": True, "no_fabrication": True,
             "clear_reasoning": True},
        )
        assert result.normalized_score == 100.0
        assert not result.hard_failed
        assert result.failed_criteria == []

    def test_partial_pass_scores_between(self):
        cs = make_basic_set()
        scorer = RubricScorer(cs)
        result = scorer.score(
            "r2",
            {"answers_question": True, "no_fabrication": True,
             "clear_reasoning": False},
        )
        assert 0 < result.normalized_score < 100
        assert result.failed_criteria == ["clear_reasoning"]

    def test_hard_fail_forces_zero(self):
        cs = make_basic_set()
        scorer = RubricScorer(cs)
        result = scorer.score(
            "r3",
            {"answers_question": True, "no_fabrication": False,
             "clear_reasoning": True},
        )
        assert result.hard_failed
        assert result.normalized_score == 0.0

    def test_missing_criterion_treated_as_failed(self):
        cs = make_basic_set()
        scorer = RubricScorer(cs)
        result = scorer.score("r4", {"answers_question": True})
        assert "no_fabrication" in result.failed_criteria
        assert result.hard_failed
        assert any("Missing evaluation" in n for n in result.notes)

    def test_critical_hallucination_forces_zero(self):
        cs = make_basic_set()
        scorer = RubricScorer(cs)
        hallu = HallucinationAnnotation(
            category=HallucinationCategory.EXTERNAL_FACTS,
            severity=Severity.CRITICAL,
            span="fabricated stat",
            explanation="Made up a number.",
        )
        result = scorer.score(
            "r5",
            {"answers_question": True, "no_fabrication": True,
             "clear_reasoning": True},
            hallucinations=[hallu],
        )
        assert result.hard_failed
        assert result.normalized_score == 0.0
        assert result.hallucination_summary["has_critical"]

    def test_score_batch(self):
        cs = make_basic_set()
        scorer = RubricScorer(cs)
        responses = [
            {"response_id": "a",
             "results": {"answers_question": True, "no_fabrication": True,
                         "clear_reasoning": True}},
            {"response_id": "b",
             "results": {"answers_question": False, "no_fabrication": True,
                         "clear_reasoning": True}},
        ]
        results = scorer.score_batch(responses)
        assert len(results) == 2
        assert results[0].normalized_score == 100.0
        assert results[1].normalized_score < 100.0
