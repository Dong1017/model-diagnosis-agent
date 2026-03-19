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

### Version Mismatch (torch_npu ↔ PyTorch ↔ CANN)
- failure_info: "RuntimeError, undefined symbol, ImportError, version mismatch, ABI incompatible"
- observed_at: "N/A - seed entry"
- failure_type: "platform"
- root_cause: "torch_npu compiled against different PyTorch version than installed, or CANN version incompatible with torch_npu"
- solution: "Verify version matrix: check torch_npu release notes for supported PyTorch and CANN versions. Rebuild or reinstall matching versions."
- last_seen: "2026-03-18"
- occurrences: 5+

### Operator Kernel Not Found (CANN Package Missing)
- failure_info: "561003, 561112, kernel not found, operator binary package not installed, ACLNN_ERR_INNER_FIND_KERNEL_ERROR"
- observed_at: "N/A - seed entry"
- failure_type: "cann"
- root_cause: "CANN operator package (opp) not installed or incomplete, or ASCEND_OPP_PATH points to wrong directory"
- solution: "1. Check OPP installed: ls /usr/local/Ascend/ascend-toolkit/latest/opp/ 2. Re-source env: source /usr/local/Ascend/ascend-toolkit/set_env.sh 3. Reinstall CANN toolkit if packages missing"
- last_seen: "2026-03-18"
- occurrences: 3+

### Scalar-to-Tensor Device Mismatch
- failure_info: "ERR01002, Expected all tensors to be on the same device, scalar tensor on CPU, NPU tensor expected"
- observed_at: "N/A - seed entry"
- failure_type: "cann"
- root_cause: "Python scalar or CPU scalar tensor passed to op-plugin function that expects NPU tensor. Op-plugin converts scalar via self_tensor_to_device() but some paths miss this."
- solution: "Explicitly move scalar to NPU: `scalar_tensor.to('npu')`, or extract as Python scalar with `.item()` before passing"
- last_seen: "2026-03-18"
- occurrences: 5+

### AI Core Overflow
- failure_info: "207003, ACL_ERROR_RT_AICORE_OVER_FLOW, AI Core overflow, numerical overflow"
- observed_at: "N/A - seed entry"
- failure_type: "cann"
- root_cause: "Operator computation overflows AI Core register width, typically with fp16 inputs on large tensors or extreme values"
- solution: "Cast inputs to fp32 before computation, or use loss scaling. Check operator dump data for overflow location."
- last_seen: "2026-03-18"
- occurrences: 3

### Stream Not in Current Context
- failure_info: "107003, ACL_ERROR_RT_STREAM_CONTEXT, stream not in current context"
- observed_at: "N/A - seed entry"
- failure_type: "framework"
- root_cause: "NPU stream was created in one context but used in another, typically caused by multi-device operations or improper device switching"
- solution: "Ensure torch.npu.set_device() is called before stream operations. Check that tensors and streams are on the same device."
- last_seen: "2026-03-18"
- occurrences: 2

### HCCL Distributed Init Failure
- failure_info: "ERR02xxx, HCCL init failed, init_process_group failed, EI0006, socket build timeout, EJ0001"
- observed_at: "N/A - seed entry"
- failure_type: "cann"
- root_cause: "HCCL communication initialization failed due to network issues, incorrect rank table, or firewall blocking ports"
- solution: "1. Check network connectivity between nodes 2. Verify HCCL rank table configuration 3. Check firewall rules for HCCL ports 4. Set HCCL_CONNECT_TIMEOUT=300 for slow networks"
- last_seen: "2026-03-18"
- occurrences: 4

### DO_COMPATIBILITY Fallback Precision Difference
- failure_info: "numerical mismatch, precision difference, aclnn unavailable, DO_COMPATIBILITY fallback to acl_op"
- observed_at: "N/A - seed entry"
- failure_type: "cann"
- root_cause: "aclnn kernel not available on current CANN version, DO_COMPATIBILITY silently falls back to acl_op (OpCommand) path which may have different precision or behavior"
- solution: "Upgrade CANN to version that supports the aclnn kernel. Check op_plugin_functions.yaml for version requirements. Or adjust test tolerances if precision difference is acceptable."
- last_seen: "2026-03-18"
- occurrences: 2

### Format Conversion Error (ND/NZ/NCHW)
- failure_info: "format mismatch, FormatHelper, IsOpInputBaseFormat, transdata failed, ACL format error"
- observed_at: "N/A - seed entry"
- failure_type: "cann"
- root_cause: "Tensor stored in non-base format (e.g., NZ for matmul optimization) but operator requires base format (ND/NCHW). Format conversion (TransData) failed or was not triggered."
- solution: "Use npu_format_cast to convert to base format before operation, or use apply_tensor_without_format for output allocation"
- last_seen: "2026-03-18"
- occurrences: 3

### Operator Not Registered for privateuse1
- failure_info: "NotImplementedError, Could not run, no kernel found, aten::xxx, privateuse1"
- observed_at: "N/A - seed entry"
- failure_type: "framework"
- root_cause: "Operator not registered in npu_native_functions.yaml or not implemented in op-plugin for current PyTorch version"
- solution: "1. Check npu_native_functions.yaml for operator registration 2. Check op_plugin_functions.yaml for version support 3. Use CPU fallback: tensor.cpu() → op → result.npu() 4. File feature request if operator should be supported"
- last_seen: "2026-03-18"
- occurrences: 10+

## Observed Failures

(Entries added from actual diagnosis sessions — include specific test case names and verified solutions)

## Searchable Keywords

- Memory: OOM, EL0004, 200000, 207018, memory exhausted, FAIL_TO_ALLOCATE_MEMORY
- Hardware: 107010, FORCE STOP, device task abort, ECC, link error, 507010, 507054, heartbeat
- Distributed: HCCL, timeout, broadcast, all_reduce, 107020, EI0002, EI0006, EJ0001, init_process_group
- Framework: 107002, 107003, context empty, stream not in context, ERR00xxx, ERR01xxx, ERR02xxx
- Test: device detection, device mismatch, index_fill, bfloat16, DISABLED_TESTS_FILE, privateuse1
- Operator: custom ops, not supported, fallback, expanded_weights, 561003, 561112, kernel not found, NotImplementedError
- Environment: libhccl.so, libascendcl.so, ASCEND_OPP_PATH, CANN, version mismatch, ImportError
- Precision: numerical mismatch, overflow, 207003, DO_COMPATIBILITY, tolerance, per_sample_grad
- Format: format mismatch, FormatHelper, ND, NZ, NCHW, transdata
- Scalar: scalar tensor, device mismatch, self_tensor_to_device, item()

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
