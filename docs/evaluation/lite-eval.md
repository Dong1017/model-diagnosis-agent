# Lite Eval (Quick Skill Test)

This compact rubric is for fast smoke testing of diagnosis skills.
It intentionally trades detail for speed and consistency.

## Fields (required)

1. `diagnosis_correct`: `yes | partial | no`
2. `evidence_ok`: `yes | no`
3. `actionable`: `yes | no`
4. `execution_result`: `success | failed | blocked | not_run`
5. `overall_skill_pass`: `yes | no`

Optional:

- `notes`: short free text (1-2 lines).

## Scoring Rules

### 1) diagnosis_correct

- `yes`: predicted primary root cause matches gold at L2+.
- `partial`: only L1 match, or acceptable alternative at L2+.
- `no`: otherwise.

### 2) evidence_ok

- `yes`: at least one concrete case-log clue supports the conclusion, and no obvious contradiction is ignored.
- `no`: missing concrete evidence or contradiction ignored.

### 3) actionable

- `yes`: suggested repair steps are executable and can be verified.
- `no`: steps are missing/unsafe/ambiguous/not verifiable.

### 4) execution_result

- `success`: action executed and verification passed.
- `failed`: action executed but verification failed.
- `blocked`: action could not be executed due to external dependency.
- `not_run`: action not executed yet.

### 5) overall_skill_pass (hard gate)

Set `overall_skill_pass=yes` only if all are true:

- `diagnosis_correct` is `yes` or `partial`
- `evidence_ok=yes`
- `actionable=yes`

Else set `overall_skill_pass=no`.

## Suggested Use

Use Lite Eval for:

- early experiments
- CI smoke checks
- rapid regression spot checks

Use the full framework when you need deeper attribution, evidence granularity, or business E2E validation.
