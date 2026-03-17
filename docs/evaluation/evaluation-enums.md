# Evaluation Enums (v0.1.1)

This file is the single source of truth for enum values used by evaluation docs and record templates.

## case metadata

- `case_risk_level`: `normal | high`

## skill layer

- `actionability`: `not_actionable | partially_actionable | actionable`
- `actionability_blocker_code`:
  - `missing_steps`
  - `unsafe_operation`
  - `ambiguous_target`
  - `requires_unavailable_permission`
  - `not_verifiable`
  - `other`

## evidence

- `log_evidence_quality`: `E0 | E1 | E2 | E3`
- `doc_evidence_quality`: `E0 | E1 | E2 | E3`
- `mapping_quality`: `E0 | E1 | E2 | E3`
- `negative_tags` (allowed values):
  - `pseudo_evidence_chain`
  - `conflicting_evidence_ignored`
  - `citation_mismatch`

## task layer

- `task_outcome`:
  - `success`
  - `failed_due_to_skill_output`
  - `blocked_by_external_dependency`
  - `not_executed_yet`

## e2e layer

- `result_validation_method`:
  - `unit_test`
  - `integration_test`
  - `numerical_tolerance_check`
  - `baseline_output_match`
  - `business_e2e_script`
  - `human_spot_check`

- `result_validation_artifact.artifact_type`:
  - `command`
  - `log_path`
  - `report_path`
  - `screenshot`
  - `human_note`
