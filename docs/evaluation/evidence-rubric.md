# Evidence Rubric (v0.1)

## 1. Evidence levels

- `E0` No valid evidence.
- `E1` Relevant evidence mentioned, but cannot support conclusion.
- `E2` Partially supporting evidence; supports intermediate judgment only.
- `E3` Sufficient evidence with explicit mapping: evidence -> rule -> conclusion.

## 2. Evaluate three channels independently

- `log_evidence_quality`: `E0|E1|E2|E3`
- `doc_evidence_quality`: `E0|E1|E2|E3`
- `mapping_quality`: `E0|E1|E2|E3`

`mapping_quality` checks whether the model explicitly connects case logs with external documentation/rules.

## 3. Negative evidence-chain tags

- `pseudo_evidence_chain`: conclusion-first citation padding
- `conflicting_evidence_ignored`: contradictory evidence exists but was ignored
- `citation_mismatch`: cited source does not support claimed statement

## 4. Evidence pass rule (explicit)

`evidence_pass=true` only if all are true:

- `mapping_quality >= E2`
- `conflicting_evidence_ignored` not present in `negative_tags`

Additional constraints:

- For high-risk conclusions, `mapping_quality` must be `E3`.
- If conclusion depends on external rule/doc, `doc_evidence_quality=E0` => `evidence_pass=false`.
- If conclusion is fully case-local (log-only) by case design, `doc_evidence_quality=E0` is allowed only when `case_local_only=true`.
- If `pseudo_evidence_chain` is present in high-risk conclusions, `evidence_pass=false`.


## 5. Case metadata ownership

- `case_risk_level` must be set in case metadata before labeling (`normal | high`).
- High-risk rules in this rubric apply only when `case_risk_level=high`.
- `case_local_only` must be pre-declared in case metadata; reviewers must not decide it ad hoc during scoring.


Enum values referenced in this rubric are centrally maintained in [evaluation-enums.md](evaluation-enums.md).
