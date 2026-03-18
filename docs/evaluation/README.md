# Evaluation Framework Artifacts

This directory contains the v0.1/v0.1.1 trial-labeling framework for failure-analysis evaluation.
It also includes a compact Lite Eval path for quick skill smoke tests.

## Files

- `lite-eval.md`: compact quick-test rubric (5 required fields).
- `lite-eval-record-template.json`: minimal storage-ready template for Lite Eval.

## Intended use

### Full framework (default)

1. Freeze case metadata (`case_risk_level`, `case_local_only`, optional `risk_rationale`) before labeling.
2. Label root cause, evidence quality, actionability, task execution state, and E2E validation using referenced docs.
3. Persist records with the JSON template shape for downstream analytics.

### Lite Eval (quick smoke test)

1. Score only 5 required fields from `lite-eval.md`.
2. Apply the hard gate for `overall_skill_pass`.
3. Persist with `lite-eval-record-template.json`.
