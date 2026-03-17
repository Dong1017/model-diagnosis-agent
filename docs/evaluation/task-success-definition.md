# Task Success Definition (v0.1)

## 1. Evaluation layers

Use three independent success layers for each case.

- **Skill success (diagnosis layer)**: evaluates analysis quality only.
- **Task success (execution layer)**: evaluates whether recommended actions were actually executed and verified.
- **End-to-end success (business layer)**: evaluates whether business result is restored and verified correct.

Do not replace these three with a single pass rate.

## 2. Case metadata required before scoring

- `case_risk_level`: `normal | high` (drives high-risk evidence requirements).
- `case_local_only`: boolean (declares whether log-only justification is allowed).

Both fields must be assigned in case metadata before scoring starts.
- `risk_rationale` is recommended to justify why `case_risk_level` is set (especially when `high`).

Enum values are centrally maintained in [evaluation-enums.md](evaluation-enums.md).

## 3. Skill success rule (explicit)

`skill_success=true` only if all are true:

1. `root_cause_pass=true`
2. `evidence_pass=true`
3. `actionability=actionable` (**hard gate**)

### v0.1 default threshold

- `root_cause_pass=true` means:
  - primary cause hits **L2+**, OR
  - acceptable alternative cause hits **L2+** (`topk_hit_l2_or_above=true`)
- L1-only hit is insufficient for `skill_success=true`.
- For high-risk cases, require L3 hit (`primary_hit_l3=true`).

### Required skill-layer fields

- `root_cause_pass`: boolean (derived from structured root-cause scoring)
- `evidence_pass`: boolean (derived from evidence rubric)
- `actionability`: `not_actionable | partially_actionable | actionable`
- `actionability_blocker_code`: enum:
  - `missing_steps`
  - `unsafe_operation`
  - `ambiguous_target`
  - `requires_unavailable_permission`
  - `not_verifiable`
  - `other`
- `actionability_blocker_detail`: text (required when actionable != actionable)

## 4. Task success states

Task layer must separate skill-caused failures from external blockers and from no execution.

### Required execution-state fields

- `repair_action_executed`: boolean
- `executed_action_id`: string (required when `repair_action_executed=true`)
- `repair_verification_passed`: `true | false | null` (`null` required when `repair_action_executed=false`)

Consistency rule:

- If `repair_action_executed=false`, then `executed_action_id=null` and `repair_verification_passed=null`.

### Task outcome enum

- `success`
- `failed_due_to_skill_output`
- `blocked_by_external_dependency`
- `not_executed_yet`

### External dependency examples

- Missing permission/access
- Unavailable compute resource
- Driver/CANN/CUDA upgrade constraints
- Network or dependency mirror unavailable

## 5. End-to-end success definition

`e2e_success=true` requires both:

1. Runtime issue resolved (`error_resolved=true`)
2. Result correctness validated (`result_validation_pass=true`)

### Required traceability fields

- `result_validation_method`: enum, one of
  - `unit_test`
  - `integration_test`
  - `numerical_tolerance_check`
  - `baseline_output_match`
  - `business_e2e_script`
  - `human_spot_check`
- `result_validation_artifact` object:
  - `artifact_type`: `command | log_path | report_path | screenshot | human_note`
  - `artifact_value`: string
- `result_validation_pass`: boolean

## 6. Multi-solution handling

To avoid "shotgun" suggestions, record:

- `num_distinct_solutions_proposed`: integer
- `max_counted_alternatives`: integer (v0.1 default: `2`)
- `primary_solution_correct`: boolean
- `fallback_solution_correct`: `true | false | null` (`null` when no fallback is proposed)
- `has_viable_path`: boolean (true if any proposed path is executable and validated)

Distinct rule:

- If two suggestions share the same root cause and same repair action with only parameter tuning differences, count as one.
- Count as distinct only when repair path or dependency assumptions differ.

## 7. Conversation metrics (renamed)

- `turns_to_close`: total turns until closure criteria met
- `model_info_requests`: number of model-initiated information requests
- `premature_conclusion`: boolean

`premature_conclusion=true` when required key inputs are missing and the model outputs any of:

- explicit primary root-cause judgment,
- explicit repair action recommendation,
- explicit exclusion of other major branches.

It is also true when the model requested key info but gave high-confidence conclusion before info was provided.

Closure must be judged by acceptance criteria, not only "user said solved".

## 8. Failure attribution fill rule

`failure_analysis.failure_primary_tag` is nullable.

- Required when `skill_success=false` OR `task_outcome=failed_due_to_skill_output`.
- Must be null for fully successful cases (`skill_success=true` and `task_outcome=success`).
- Must be null when `task_outcome=blocked_by_external_dependency` (external blocker is not a skill failure).
- Primary tag should identify the earliest dominant failure source in the chain, not the final symptom.
