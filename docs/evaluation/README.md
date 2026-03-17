# Evaluation Framework Artifacts

This directory contains the v0.1/v0.1.1 trial-labeling framework for failure-analysis evaluation.

## Files

- `task-success-definition.md`: layered success criteria and gating rules.
- `root-cause-taxonomy.md`: closed-set taxonomy and scoring semantics.
- `evidence-rubric.md`: evidence grading and pass constraints.
- `evaluation-enums.md`: centralized enum source of truth.
- `failure-eval-record-template.json`: storage-ready record schema example.

## Intended use

1. Freeze case metadata (`case_risk_level`, `case_local_only`, optional `risk_rationale`) before labeling.
2. Label root cause, evidence quality, actionability, task execution state, and E2E validation using referenced docs.
3. Persist records with the JSON template shape for downstream analytics.
