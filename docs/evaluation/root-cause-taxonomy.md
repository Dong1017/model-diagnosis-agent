# Root Cause Taxonomy (v0.1, closed set)

## 1. Scope note

This v0.1 taxonomy is defined for MindSpore failure-analysis workflow.
`framework_mindspore` and `backend_runtime` are intentional names in this scope.

`backend_runtime` groups CANN/ACLNN/HCCL/NCCL in L1 for stable early labeling;
finer separation (e.g., CANN vs distributed backend) is captured at L2/L3 in v0.1.

## 2. L1 closed categories

`l1_category` must be one of:

1. `platform_environment`
2. `scripts_application_logic`
3. `framework_mindspore`
4. `backend_runtime`
5. `dependency_third_party`
6. `data_input`
7. `configuration`

Notes:

- `platform_environment` includes OS/device/driver/runtime env setup issues.
- `configuration` is **not** a sub-type of scripts; it is standalone (context/env/flags/system config).
- `dependency_third_party` is separated from platform to isolate ecosystem/package problems.
- `data_input` is in scope for failure diagnosis and must be tracked.

## 3. L2 mechanism examples (non-exhaustive)

- `version_incompatibility`
- `missing_env_var_or_path`
- `operator_not_supported`
- `invalid_shape_or_dtype`
- `invalid_axis_or_index`
- `distributed_comm_timeout`
- `resource_exhaustion`
- `permission_or_access_denied`
- `api_contract_change`

## 4. L3 trigger pattern

`l3_trigger` should capture concrete mechanism (error code/pattern/condition), e.g.:

- `EZ1001_duplicate_dims_in_axis`
- `507010_device_heartbeat_lost`
- `gen_ops_yaml_missing_py_method`

## 5. Scoring rule

- L1 hit only: low partial credit
- L1+L2 hit: major credit
- L1+L2+L3 hit: full credit

Additional scoring fields:

- `predicted_primary_cause`
- `predicted_alternative_causes` (Top-k list)
- `max_alternative_causes` (v0.1 default: `2`; entries over limit do not count for `topk_hit`)
- `primary_hit`: boolean
- `topk_hit`: boolean
- `topk_hit_l2_or_above`: boolean
- `primary_hit_l3`: boolean

If primary misses but alternative hits, count partial credit via `topk_hit`.

`acceptable alternative` means the candidate belongs to that case's pre-defined gold acceptable root-cause set; alternatives outside this set must not be counted for `topk_hit_l2_or_above`.


## 6. Acceptable set minimal structure

`acceptable_root_cause_set` must be a list of structured objects, each using the same shape as root-cause prediction:

- `l1_category` (required)
- `l2_mechanism` (required)
- `l3_trigger` (optional, nullable)

Free-text alternatives are not allowed.
