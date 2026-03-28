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

### AI Core Overflow
- failure_info: "207003, ACL_ERROR_RT_AICORE_OVER_FLOW, overflow, fp16, mixed precision, loss becomes NaN"
- observed_at: ""
- failure_type: "cann"
- root_cause: "AI Core numeric overflow, usually triggered by fp16 or mixed-precision computation on large values or unstable gradients"
- solution: "Stabilize the overflowing path first: cast sensitive ops to fp32, tune or reduce loss scaling, inspect gradients for inf/nan, then re-check any downstream timeout or communication failures"
- last_seen: "2026-03-19"
- occurrences: 2

### Version Mismatch (torch_npu -> PyTorch -> CANN)
- failure_info: "symbol not found, import fails after PyTorch upgrade, ABI mismatch, torch_npu version mismatch, CANN compatibility"
- observed_at: ""
- failure_type: "platform"
- root_cause: "Installed torch_npu build is not compatible with the current PyTorch and/or CANN version after upgrade"
- solution: "Reinstall or rebuild torch_npu for the exact PyTorch version in use, then verify the matching CANN compatibility matrix before retrying"
- last_seen: "2026-03-19"
- occurrences: 3

### Stream Not in Current Context
- failure_info: "stream not in current context, current stream, stream mismatch, aclrtSynchronizeStream, aclrtSetCurrentContext"
- observed_at: ""
- failure_type: "framework"
- root_cause: "A stream created or recorded under one NPU context is later used from a different current context or device, so runtime synchronization happens against the wrong context"
- solution: "Ensure the correct device/context is set before stream use, avoid cross-context stream reuse, and trace stream creation plus synchronization points in torch_npu or test code"
- last_seen: "2026-03-19"
- occurrences: 2

### Test Framework Device Detection
- failure_info: "ERR01002, Expected all tensors to be on the same device, device mismatch, index_fill with incompatible types"
- observed_at: "test_sort_and_select.py:test_stable_sort_against_numpy_npu_bfloat16"
- failure_type: "scripts"
- root_cause: "Test early return condition checked for 'cuda' instead of 'npu', causing complex test with index_fill to execute on NPU with incompatible tuple/tensor types"
- solution: "Update device type check from `if self.device_type == 'cuda'` to `if self.device_type == 'npu'`"
- last_seen: "2026-03-09"
- occurrences: 1

### Per-Sample Gradient Threshold Issue
- failure_info: "per_sample_grad, batch_size threshold, aclnnIm2col, gradient accuracy, expanded_weights,(Conv1d|GroupNorm|LayerNorm)"
- observed_at: "call_for_per_sample_grads with batch_size>=32"
- failure_type: "cann"
- root_cause: "PyTorch's per-sample gradient computation (torch.nn.utils.call_for_per_sample_grads) uses THRESHOLD=32 in conv_utils.py to switch between group-based algorithm and unfold-based algorithm. When batch_size >= 32, unfold path triggers CANN's aclnnIm2col optimization which changes floating-point operation order compared to group-based algorithm, causing numerical differences between per-sample gradients and cumulative individual backward passes"
- solution: "Workaround: use batch_size < 32 for per-sample gradient tests on NPU, or increase tolerance (atol=1e-2, rtol=1e-3). Long-term: address CANN aclnnIm2col numerical precision in optimization path, or use group-based algorithm unconditionally for per-sample gradients"
- last_seen: "2026-03-11"
- occurrences: 1

### Test Framework CUDA Dependency
- failure_info: "TypeError: '>=' not supported between 'NoneType' and 'tuple', SM53OrLater, torch.cuda.get_device_capability"
- observed_at: "test_linalg.py:5731 (class TestLinalg test definition)"
- failure_type: "scripts"
- root_cause: "Test imports torch.testing._internal.common_cuda which contains CUDA-specific lazy evaluation (SM53OrLater = LazyVal(lambda: torch.cuda.is_available() and torch.cuda.get_device_capability() >= (5, 3))) that fails when running on NPU because get_device_capability() returns None"
- solution: "Skip common_cuda import for NPU tests, or add None check: torch.cuda.is_available() and torch.cuda.get_device_capability() is not None and ..."
- last_seen: "2026-03-12"
- occurrences: 1

### _convert_weight_to_int4pack Not Implemented for NPU
- failure_info: "Could not run 'aten::_convert_weight_to_int4pack' with arguments from 'CPU' backend, PrivateUse1 dispatcher not registered, int4 quantization, weight packing"
- observed_at: "test_linalg.py test__int4_mm, test_mps.py test__int4_mm"
- failure_type: "framework|cann"
- root_cause: "Two issues: (1) Framework: torch_npu doesn't register aten::_convert_weight_to_int4pack dispatcher entry, only provides torch_npu.npu_convert_weight_to_int4pack in separate namespace. (2) CANN: No ACLNN kernel for weight packing equivalent to CUDA's matrix_to_m16n8k16_Bint4_layout; packing is done on CPU via npu_convert_weight_to_int4pack. Also: Input format differs (CUDA: uint8[N,K] with pre-packed int4; NPU: int32[K,N] unpacked), value range differs (CUDA: unsigned [0,15]; NPU: signed [-8,7])"
- solution: "Short-term: Modify test to use NPU-specific flow with torch_npu.npu_convert_weight_to_int4pack and signed int4 quantization. Long-term: Add NPU dispatcher entry in native_functions.yaml and implement _convert_weight_to_int4pack_npu with format conversion"
- last_seen: "2026-03-12"
- occurrences: 1

### AMP Custom Fwd Deprecation Warning Not Implemented on NPU
- failure_info: "IndexError, list index out of range, custom_fwd deprecation warning, torch.cuda.amp.custom_fwd is deprecated"
- observed_at: "test_cuda.py:test_autocast_custom_deprecated_warning"
- failure_type: "scripts"
- root_cause: "Test checks for CUDA-specific deprecation warning from torch.cuda.amp.custom_fwd. When adapted to NPU via adapt_testcases_to_npu.py (which converts torch.cuda.amp → torch.npu.amp), the test uses torch_npu.npu.amp.custom_fwd which does NOT emit a deprecation warning, causing the warning list to be empty and w[0] to raise IndexError"
- solution: "Add test_cuda.py::TestCudaAutocast::test_autocast_custom_deprecated_warning to disabled tests list (.pytorch-disabled-tests.json) as it tests CUDA-specific behavior not applicable to NPU's amp implementation. NPU's custom_fwd in torch_npu/npu/amp/autocast_mode.py doesn't have deprecation mechanism for backward compatibility reasons"
- last_seen: "2026-03-12"
- occurrences: 1

### CUDA Low-Level Test Framework APIs Missing on NPU
- failure_info: "AttributeError, module 'torch._C' has no attribute '(_cuda_setStream|_cuda_setDevice|_cuda_sleep|_cuda_attach_out_of_memory_observer)', CudaNonDefaultStream, device_sleep, OOM observer"
- observed_at: "test_cuda.py::TestCudaMallocAsync::test_notifies_oom, test_cuda.py::TestCuda (various tests)"
- failure_type: "scripts|framework"
- root_cause: "PyTorch test framework uses torch._C._cuda_* low-level APIs (_cuda_setStream, _cuda_setDevice, _cuda_sleep, _cuda_attach_out_of_memory_observer, etc.) which don't exist on NPU. These are CUDA C++ extension APIs. NPU has equivalents (_npu_setStream, _npu_setDevice, _npu_attach_out_of_memory_observer) but _cuda_sleep has no CANN equivalent (no device-side sleep kernel like CUDA's cudaSleep). Tests using CudaNonDefaultStream, device_sleep() or OOM observer attach trigger these missing APIs."
- solution: "For script code: Change torch._C._cuda_attach_out_of_memory_observer(cb) to torch_npu._C._npu_attach_out_of_memory_observer(cb). For the case that the call chains of torch_npu and cuda are compatible and kernel is ready, the torch_npu framework may need revise. Otherwise, ask CANN team for kernel implementation help."
- note: "CANN lacks device-side sleep kernel equivalent to CUDA's cudaSleep(). This is a feature gap, not a CANN error/bug. The no-op is sufficient since _sleep is only used for RPC/profiling timing delays."
- last_seen: "2026-03-16"
- occurrences: 2

### functionalize_rng_ops CUDARngStateHelper Missing NPU Support
- failure_info: "IndexError, tuple index out of range, functionalize_rng_ops, CUDARngStateHelper, default_generators"
- observed_at: "test_functionalization_of_rng_ops.py::NegativeTest::test_on_cpu"
- failure_type: "framework"
- root_cause: "PyTorch's functorch/aot_autograd uses CUDARngStateHelper.get_torch_state_as_tuple() for RNG functionalization, which calls torch.cuda._get_rng_state_offset() that accesses torch.cuda.default_generators[idx]. On NPU (which uses CPU-only PyTorch build), the default_generators tuple is empty, causing IndexError."
- solution: "Added patch_cuda_rng_state_helper() to torch_npu/utils/_inductor.py that overwrites CUDARngStateHelper.get_torch_state_as_tuple() to use torch_npu.npu.initial_seed() and torch_npu.npu._get_rng_state_offset() instead of CUDA equivalents when NPU is available. Both torch._prims_common.CUDARngStateHelper and torch._prims.rng_prims.CUDARngStateHelper instances are patched."
- last_seen: "2026-03-17"
- occurrences: 1

### filter_desired_device_types Returns Empty List for PrivateUse1
- failure_info: "RuntimeError not raised, assertRaisesRegex, filter_desired_device_types returns empty list, PrivateUse1 device_type mismatch, instantiate_device_type_tests only_for filter fails"
- observed_at: "common_device_type.py:filter_desired_device_types, triggered from test_testing.py:test_multiple_handling_of_same_param_error and test_dtypes_composition_invalid"
- failure_type: "framework"
- root_cause: "filter_desired_device_types (common_device_type.py) has two bugs when PrivateUse1 is active: (1) Bare-string iteration — when only_for/except_for is a bare string (e.g. 'npu:0') instead of a list, the list comprehension `for x in only_for` iterates over individual characters, producing ['n','p','u',':','0'] where no character equals the backend name 'npu', so no conversion to 'privateuse1' occurs; (2) Asymmetric normalization — the function normalizes only the input side (only_for/except_for) from backend_name to 'privateuse1', but does NOT normalize x.device_type in the filter lambda. After PrivateUse1TestBase.setUpClass() (line 707) mutates the base class device_type from 'privateuse1' to 'npu' (via `base.setUpClass()` called in _setUpClass at line 946), the normalized input 'privateuse1' can never match the mutated x.device_type 'npu'. These two bugs together cause filter_desired_device_types to return an empty list for any PrivateUse1-related only_for value, which means instantiate_device_type_tests skips all test instantiation and no RuntimeError is ever raised for redundant parameter checks."
- solution: "Fix filter_desired_device_types in common_device_type.py: (1) Guard against bare strings with `if isinstance(only_for, str): only_for = [only_for]`; (2) Introduce a _normalize_device_type helper that strips device index ('npu:0' → 'npu') and maps backend_name → 'privateuse1'; (3) Apply _normalize_device_type to BOTH sides — the input lists AND x.device_type in the filter lambdas — so the comparison is always between consistently normalized values."
- note: "The setUpClass mutation is triggered by _setUpClass (line 946) which calls `base.setUpClass()` directly on PrivateUse1TestBase rather than through the derived class, permanently changing the global base class device_type attribute. This affects any code that later reads device_type_test_bases entries."
- last_seen: "2026-03-25"
- occurrences: 1

## Searchable Keywords

- Memory: OOM, EL0004, 200000, 207018, memory exhausted, OOM observer, OutOfMemoryError, RuntimeError
- Hardware: 107010, FORCE STOP, device task abort, ECC, link error
- Distributed: HCCL, timeout, broadcast, all_reduce, 107020
- Framework: 107002, context empty, stream not in context, dispatcher not registered, PrivateUse1, low-level API, CUDA API, _cuda_*, _npu_*, functionalize_rng_ops, default_generators, filter_desired_device_types, setUpClass device_type mutation, asymmetric normalization
- Test: device detection, device mismatch, index_fill, bfloat16, CUDA dependency, deprecation warning, CudaNonDefaultStream, device_sleep, OOM notifies_oom, assertRaisesRegex RuntimeError not raised, instantiate_device_type_tests only_for, parametrize handled multiple times
- Operator: custom ops, not supported, fallback, expanded_weights, aclnnIm2col, GroupNorm, int4pack, weight quantization
- Environment: libhccl.so, libascendcl.so, ASCEND_OPP_PATH, CANN
- Accuracy: per_sample_grad, batch_threshold, numerical bias, gradient accuracy

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
