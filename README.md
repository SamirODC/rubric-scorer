# rubric-scorer

Rubric-based scoring for AI/LLM outputs — built from real RLHF and hallucination-annotation work, not a toy example.

Most "LLM eval" scripts boil down to a single float from a judge model. That's fast, but it hides *why* a response is good or bad, and it can't be audited by a second annotator. `rubric-scorer` implements the pattern used in production annotation pipelines instead: a rubric is a set of small, atomic, weighted criteria, each one independently verifiable, with explicit hard-fail conditions for anything that should zero out a response no matter what else it got right.

## Why this exists

Two years of RLHF annotation, hallucination labeling, and rubric design across several eval platforms surfaced the same recurring failure: rubrics that reward *plausibility* instead of *correctness*. A confident, well-structured, fabricated answer routinely outscores an honest "I'm not sure" — unless the rubric is explicitly built to catch that.

This library encodes three rules that fix most of that:

1. **Atomic criteria.** Each criterion tests exactly one thing. No compound criteria like "accurate and well-formatted" — split it into two.
2. **Outcome over trace, ~80/20.** Rubrics should mostly judge whether the *final answer* is right, not whether the reasoning steps looked tidy. `CriteriaSet.validate_ratio()` checks this automatically so a rubric doesn't drift into rewarding "showed good work" over "got the right answer."
3. **Hard fails override the total.** Some failures (fabricated citations, unsafe recommendations, critical hallucinations) shouldn't just cost points — they should zero the response. `RubricScorer` enforces that regardless of how well everything else scored.

## Install

```bash
git clone https://github.com/SamirODC/rubric-scorer.git
cd rubric-scorer
pip install -e ".[dev]"
```

## Quick example

```python
from rubric_scorer import Criterion, CriteriaSet, CriterionCategory, RubricScorer

rubric = CriteriaSet(name="clinical-qa-v1")
rubric.add(Criterion("answers_question", "Directly answers what was asked", 4))
rubric.add(Criterion(
    "no_fabricated_citation", "Doesn't invent a source that doesn't exist",
    weight=5, category=CriterionCategory.OUTCOME, hard_fail=True,
))
rubric.add(Criterion(
    "logical_step_order", "Reasoning steps are coherent", weight=2,
    category=CriterionCategory.TRACE,
))

scorer = RubricScorer(rubric)
result = scorer.score(
    response_id="resp_001",
    results={
        "answers_question": True,
        "no_fabricated_citation": False,   # triggers hard fail
        "logical_step_order": True,
    },
)

print(result.normalized_score)  # 0.0 — hard fail overrides everything else
print(result.failed_criteria)   # ['no_fabricated_citation']
```

## CLI

```bash
rubric-score score --criteria examples/criteria.json --responses examples/responses.json
rubric-score validate-criteria --criteria examples/criteria.json
```

Example output:

```
Rubric: clinical-qa-hallucination-rubric-v1  (7 criteria)
Outcome/Trace weight split: 84% / 16% (OK)
------------------------------------------------------------
resp_clean_answer    100.0/100
resp_shaky_reasoning 56.0/100
  failed: correct_dosage_or_units, acknowledges_uncertainty, logical_step_order, cites_evidence_used
resp_fabricated_citation HARD FAIL
  failed: no_fabricated_citation
  note: Hard-fail criterion triggered — normalized score forced to 0 regardless of other criteria.
  note: Critical hallucination detected — normalized score forced to 0.
```

## Hallucination annotation

`rubric_scorer.hallucination` implements a 4-category taxonomy for annotating multi-step / agentic outputs, matching the categories used in production agentic-task labeling:

| Category | What it catches |
|---|---|
| `exploration` | Agent claims to have checked/searched something it never actually looked at |
| `grounding` | Conclusion doesn't follow from the evidence actually gathered |
| `external_facts` | A flat-out wrong fact about the world (wrong dosage, wrong date, wrong API) |
| `incorrect_action` | Agent claims to have taken an action it didn't actually take |

Each `HallucinationAnnotation` has a severity (`minor` / `major` / `critical`); any `critical` annotation automatically forces a hard fail in `RubricScorer`, the same way a fabricated-citation criterion would.

## Project structure

```
rubric_scorer/
  criteria.py       # Criterion / CriteriaSet — the rubric definition
  scorer.py         # RubricScorer / ScoreResult — the scoring engine
  hallucination.py  # 4-category hallucination taxonomy + annotations
  cli.py            # `rubric-score` command-line tool
examples/
  criteria.json     # A worked clinical-QA rubric (84/16 outcome/trace split)
  responses.json    # Three responses: clean pass, partial fail, hard fail
  run_example.py    # Library usage without the CLI
tests/
  test_scorer.py    # Unit tests for criteria, ratio validation, hard fails
```

## Run tests

```bash
pytest
```

## Background

Built by [Samir Delgadillo](mailto:delgadillo.samir.omar@gmail.com) — Medical Doctor and AI evaluation specialist with 2+ years of hands-on RLHF annotation, hallucination labeling, and rubric design across Alignerr, Outlier, Telus, RWS, and AfterQuery. This project is a distilled, reusable version of the rubric methodology used day-to-day in that work.

## License

MIT — see [LICENSE](LICENSE).
