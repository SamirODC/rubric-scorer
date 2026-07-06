"""
rubric-scorer
=============

A small, opinionated library for grading AI/LLM outputs with structured,
outcome-oriented rubrics instead of ad-hoc thumbs up/down judgments.

Built from real-world RLHF and hallucination-annotation workflows: every
criterion is atomic, binary, and weighted (-5 to +5), and any criterion
can be flagged as a "hard fail" that overrides the numeric total.
"""

from .criteria import Criterion, CriteriaSet, CriterionCategory
from .scorer import RubricScorer, ScoreResult
from .hallucination import HallucinationCategory, HallucinationAnnotation

__all__ = [
    "Criterion",
    "CriteriaSet",
    "CriterionCategory",
    "RubricScorer",
    "ScoreResult",
    "HallucinationCategory",
    "HallucinationAnnotation",
]

__version__ = "0.1.0"
