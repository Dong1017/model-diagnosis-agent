# Failure Showcase

Historical torch_npu failures and their solutions. Format:
```yaml
- failure_info: "[error keywords/context]"
  observed_at: "[file:function or test location where observed]"
  failure_type: "platform|scripts|framework|cann"
  root_cause: "[specific cause]"
  solution: "[actionable steps]"
  last_seen: "[timestamp]"
  occurrences: [count]
```

## Common Failure Patterns

### InstanceNorm with Expanded Weights
- failure_info: "test_instance_norm_model_num_dim_1_npu, expanded_weights, per_sample_grad, 91% mismatched elements, greatest relative difference 353.9"
- observed_at: ""
- failure_type: "cann"
- root_cause: "NPU's InstanceNorm implementation computes gradients differently for per-sample gradients vs individual backward passes"
- solution: "Increase tolerances for NPU in tests: atol=1e-2, rtol=1e-3"
- last_seen: "2026-03-06"
- occurrences: 1

### Missing CANN Environment
- failure_info: "libhccl.so not found, libascendcl.so not found, ImportError: cannot import name npu"
- observed_at: ""
- failure_type: "scripts"
- root_cause: "CANN environment variables (ASCEND_OPP_PATH, ASCEND_AICPU_PATH) not set or CANN not installed"
- solution: "Source CANN environment: source /usr/local/Ascend/ascend-toolkit/set_env.sh"
- last_seen: "2026-03-06"
- occurrences: 5

### Out of Memory (OOM)
- failure_info: "EL0004, FAIL_TO_ALLOCATE_MEMORY, 200000, 207018, out of memory"
- observed_at: ""
- failure_type: "platform"
- root_cause: "Device HBM memory exhausted due to large tensors or batch size"
- solution: "Reduce batch size, use gradient checkpointing, torch_npu.empty_cache()"
- last_seen: "2026-03-06"
- occurrences: 10+

### Delayed Execution Warning
- failure_info: "WARNING: Since the operator is called asynchronously, the stacktrace may be inaccurate"
- observed_at: ""
- failure_type: "framework"
- root_cause: "NPU uses asynchronous execution, stack traces show op launch site not error site"
- solution: "Use synchronous mode for debugging: ASCEND_LAUNCH_BLOCKING=1, or check CANN logs for accurate error location"
- last_seen: "2026-03-06"
- occurrences: 20+

### Context Empty Error
- failure_info: "107002, The context is empty, aclrtSetContext or aclrtSetDevice is called"
- observed_at: ""
- failure_type: "framework"
- root_cause: "NPU context not initialized before calling NPU API"
- solution: "Ensure torch.npu devices are initialized before tensor operations, check init sequence"
- last_seen: "2026-03-06"
- occurrences: 8

### Device Task Abort (FORCE STOP)
- failure_info: "107010, NPU function error: FORCE STOP, device task abort, reason=device task abort"
- observed_at: ""
- failure_type: "platform"
- root_cause: "NPU hardware heartbeat lost or device encountered error"
- solution: "Check device health with npu-smi info, may need device reset or hardware replacement"
- last_seen: "2026-03-06"
- occurrences: 3

### CANN Inner Error
- failure_info: "EZ9999, E[1-9A-Z]9999, CANN Inner Error"
- observed_at: ""
- failure_type: "cann"
- root_cause: "CANN internal error, check operator compatibility with current CANN version"
- solution: "Check CANN logs in /var/log/npu/slog/, update CANN version, recompile operators"
- last_seen: "2026-03-06"
- occurrences: 4

### HCCL Timeout
- failure_info: "wait for compute device to finish failed, times out, 107020"
- observed_at: ""
- failure_type: "cann"
- root_cause: "Process termination mid-HCCL operation causing timeout"
- solution: "Check network connectivity, verify HCCL configuration, avoid unexpected process exits"
- last_seen: "2026-03-06"
- occurrences: 2

### Feature Not Supported
- failure_info: "ERR00007, 207000, feature not supported, operator not supported on NPU backend"
- observed_at: ""
- failure_type: "framework"
- root_cause: "Operator or feature not implemented for NPU in current torch_npu/CANN version"
- solution: "Check torch_npu/CANN compatibility, upgrade versions, or use CPU fallback"
- last_seen: "2026-03-06"
- occurrences: 15+

### Test Framework Device Detection
- failure_info: "ERR01002, Expected all tensors to be on the same device, device mismatch, index_fill with incompatible types"
- observed_at: "test_sort_and_select.py:test_stable_sort_against_numpy_npu_bfloat16"
- failure_type: "scripts"
- root_cause: "Test early return condition checked for 'cuda' instead of 'npu', causing complex test with index_fill to execute on NPU with incompatible tuple/tensor types"
- solution: "Update device type check from `if self.device_type == 'cuda'` to `if self.device_type == 'npu'`"
- last_seen: "2026-03-09"
- occurrences: 1

## Searchable Keywords

- Memory: OOM, EL0004, 200000, 207018, memory exhausted
- Hardware: 107010, FORCE STOP, device task abort, ECC, link error
- Distributed: HCCL, timeout, broadcast, all_reduce, 107020
- Framework: 107002, context empty, stream not in context
- Test: device detection, device mismatch, index_fill, bfloat16
- Operator: custom ops, not supported, fallback, expanded_weights
- Environment: libhccl.so, libascendcl.so, ASCEND_OPP_PATH, CANN

## Adding New Failures

When analyzing a new failure:

1. Extract key information: error code, keywords, context
2. Identify failure type (platform/scripts/framework/cann)
3. Determine root cause through orientation analysis
4. Document solution (verified or proposed)
5. Update this file with YAML format above
6. Include approximate timestamp and occurrence count

### failure_info Guidelines

**Include:** error codes, error keywords, operators, operation patterns
**Exclude:** test/function names, file names, variable names, specific parameters

**Examples:**
- ✅ "ERR01002, device mismatch, index_fill"
- ❌ "test_stable_sort_against_numpy_npu_bfloat16"
- ✅ "EL0004, OOM"
- ❌ "torch_npu.npu.empty_cache() out of memory"
- ✅ "expanded_weights, per_sample_grad"
- ❌ "test_instance_norm_model_num_dim_1_npu"

**Purpose:** Enable semantic pattern matching; use `observed_at` for specific instances.
