# Failure Showcase

Historical MindSpore failures and their solutions. Format:
```yaml
- failure_info: "[error keywords/context]"
- observed_at: "[file:function or test location where observed]"
- backend: "ascend|gpu|cpu|all"
- failure_type: "platform|scripts|framework|backend"
- root_cause: "[specific cause]"
- solution: "[actionable steps]"
- last_seen: "[timestamp]"
- occurrences: [count]
- issue_url: "[issue URL, comma-separated for multiple]"
```

## Common Failure Patterns

### Out of Memory (Ascend HBM)
- failure_info: "EL0004, 200000, 207018, RuntimeError, out of memory, device memory exhausted"
- observed_at: "N/A - seed entry"
- backend: "ascend"
- failure_type: "platform"
- root_cause: "Ascend HBM memory exhausted due to large tensors, batch size, or graph memory fragmentation"
- solution: "Reduce batch size; use gradient checkpointing (ms.nn.CellRecompute); call ms.hal.memory.empty_cache(); set max_device_memory in context"
- last_seen: "2026-03-12"
- occurrences: 1
- issue_url: "N/A"

### Out of Memory (GPU VRAM)
- failure_info: "CUDA out of memory, RuntimeError, CUDA error"
- observed_at: "N/A - seed entry"
- backend: "gpu"
- failure_type: "platform"
- root_cause: "GPU VRAM exhausted due to large model or batch size"
- solution: "Reduce batch size; use gradient checkpointing; set CUDA_VISIBLE_DEVICES to correct GPU; check nvidia-smi for other processes using VRAM"
- last_seen: "2026-03-12"
- occurrences: 1
- issue_url: "N/A"

### Missing CANN Environment
- failure_info: "libascendcl.so not found, ImportError, cannot find CANN, ASCEND_OPP_PATH"
- observed_at: "N/A - seed entry"
- backend: "ascend"
- failure_type: "scripts"
- root_cause: "CANN environment variables not set or CANN not installed"
- solution: "Source CANN environment: source /usr/local/Ascend/ascend-toolkit/set_env.sh; verify CANN installation with cat /usr/local/Ascend/ascend-toolkit/version"
- last_seen: "2026-03-12"
- occurrences: 1
- issue_url: "N/A"

### Device Target Mismatch
- failure_info: "RuntimeError, device_target, should be one of, Ascend GPU CPU, invalid device"
- observed_at: "N/A - seed entry"
- backend: "all"
- failure_type: "scripts"
- root_cause: "set_context device_target does not match available hardware or is misspelled"
- solution: "Check available hardware; use correct device_target string ('Ascend', 'GPU', 'CPU'); verify device availability with npu-smi info or nvidia-smi"
- last_seen: "2026-03-12"
- occurrences: 1
- issue_url: "N/A"

### Graph Compilation Error (Static Graph)
- failure_info: "RuntimeError, graph compile, type inference, abstract type, infer failed"
- observed_at: "N/A - seed entry"
- backend: "all"
- failure_type: "framework"
- root_cause: "MindSpore GRAPH_MODE cannot compile construct() due to unsupported Python syntax, dynamic control flow, or type inference failure"
- solution: "Switch to PYNATIVE_MODE for debugging; remove dynamic control flow from construct(); use @ms.jit_class for custom classes; check Supported Platforms for operators"
- last_seen: "2026-03-12"
- occurrences: 1
- issue_url: "N/A"

### Operator Not Supported on Backend
- failure_info: "RuntimeError, not supported, operator, backend, fallback"
- observed_at: "N/A - seed entry"
- backend: "all"
- failure_type: "framework"
- root_cause: "Operator/feature not implemented for target backend in current MindSpore version"
- solution: "Check operator's Supported Platforms; upgrade MindSpore/CANN version; use alternative operator; or use CPU fallback for specific ops"
- last_seen: "2026-03-12"
- occurrences: 1
- issue_url: "N/A"

### Context Empty (Ascend)
- failure_info: "107002, context is empty, aclrtSetContext, aclrtSetDevice, RuntimeError"
- observed_at: "N/A - seed entry"
- backend: "ascend"
- failure_type: "framework"
- root_cause: "NPU context not initialized before calling NPU operations"
- solution: "Ensure mindspore.set_context(device_target='Ascend', device_id=N) is called before any tensor operations; check initialization sequence"
- last_seen: "2026-03-12"
- occurrences: 1
- issue_url: "N/A"

### Device Heartbeat Lost (Ascend)
- failure_info: "507010, lost heartbeat, RuntimeError, task scheduler, device hang"
- observed_at: "N/A - seed entry"
- backend: "ascend"
- failure_type: "platform"
- root_cause: "NPU device task scheduler heartbeat lost, device may be unresponsive"
- solution: "Check device health with npu-smi info; device may need reset; check for thermal throttling or hardware failure"
- last_seen: "2026-03-12"
- occurrences: 1
- issue_url: "N/A"

### HBM ECC Error (Ascend)
- failure_info: "507054, HBM ECC, multi-bit, hardware error, memory fault"
- observed_at: "N/A - seed entry"
- backend: "ascend"
- failure_type: "platform"
- root_cause: "HBM memory hardware ECC fault on Ascend device"
- solution: "Hardware fault — contact hardware support; try different device; check npu-smi info for device health status"
- last_seen: "2026-03-12"
- occurrences: 1
- issue_url: "N/A"

### HCCL Communication Error (Ascend Distributed)
- failure_info: "HCCL, timeout, EI0002, EI0006, notify wait, socket build, distributed"
- observed_at: "N/A - seed entry"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "HCCL distributed communication failure — network timeout, rank configuration error, or HCCS link issue"
- solution: "Check network connectivity between devices; verify ranktable configuration; check HCCL_WHITELIST_DISABLE setting; ensure all ranks start correctly"
- last_seen: "2026-03-12"
- occurrences: 1
- issue_url: "N/A"

### NCCL Communication Error (GPU Distributed)
- failure_info: "NCCL error, timeout, unhandled system error, distributed, GPU"
- observed_at: "N/A - seed entry"
- backend: "gpu"
- failure_type: "backend"
- root_cause: "NCCL distributed communication failure — network issue, GPU topology problem, or version mismatch"
- solution: "Check GPU connectivity with nvidia-smi topo; set NCCL_DEBUG=INFO for detailed logs; verify NCCL version compatibility; check firewall settings"
- last_seen: "2026-03-12"
- occurrences: 1
- issue_url: "N/A"

### TBE Operator Compilation Error (Ascend)
- failure_info: "TBE, compile failed, E9xxxx, EBxxxx, operator compilation, UB overflow"
- observed_at: "N/A - seed entry"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "TBE operator compilation failed due to unsupported shape/dtype, UB memory overflow, or CANN version incompatibility"
- solution: "Check input shapes/dtypes match TBE operator constraints; check CANN version compatibility; try jit_compile=False in ascend_config; check CANN logs for detailed compilation error"
- last_seen: "2026-03-12"
- occurrences: 1
- issue_url: "N/A"

### Shape Inference Failure
- failure_info: "ValueError, shape, infer, expected, dimensions, rank mismatch"
- observed_at: "N/A - seed entry"
- backend: "all"
- failure_type: "scripts"
- root_cause: "Input tensor shape does not match operator requirements (wrong dimensions, rank mismatch)"
- solution: "Check input tensor shapes; verify reshape/transpose operations; use print(tensor.shape) to debug; check operator documentation for expected input shapes"
- last_seen: "2026-03-12"
- occurrences: 1
- issue_url: "N/A"

### AI Core Execution Timeout (Ascend)
- failure_info: "507014, AI Core timeout, AICORE_TIMEOUT, execution timeout"
- observed_at: "N/A - seed entry"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "AI Core operator execution timed out, possibly due to infinite loop in operator or hardware issue"
- solution: "Check CANN logs for specific operator; try reducing input size; check if operator has known timeout issues in current CANN version; contact support if persistent"
- last_seen: "2026-03-12"
- occurrences: 1
- issue_url: "N/A"

### Dtype Mismatch
- failure_info: "TypeError, dtype, expected, Float32, Float16, type not match, cast"
- observed_at: "N/A - seed entry"
- backend: "all"
- failure_type: "scripts"
- root_cause: "Tensor data types don't match operator requirements or between operands"
- solution: "Use ops.Cast() to convert types; check precision_mode in ascend_config for auto-casting behavior; verify model parameter dtypes match input dtypes"
- last_seen: "2026-03-12"
- occurrences: 1
- issue_url: "N/A"

### mint View Op in GRAPH_MODE / JIT
- failure_info: "RuntimeError, jit_view_unsupported, view, squeeze, flatten, reshape, GRAPH_MODE, graph compile"
- observed_at: "N/A - seed entry"
- backend: "all"
- failure_type: "framework"
- root_cause: "mint view operations (squeeze, unsqueeze, flatten, reshape, t, narrow, split, broadcast_to, permute, transpose) are decorated with @jit_view_unsupported and may fail or produce incorrect results in GRAPH_MODE / JIT compilation"
- solution: "Switch to PYNATIVE_MODE for debugging; use ops.* equivalents instead of mint view ops in GRAPH_MODE; or use .copy() after view ops to materialize the tensor"
- last_seen: "2026-03-12"
- occurrences: 1
- issue_url: "N/A"

### mint.equal() Return Type Confusion
- failure_info: "TypeError, AttributeError, mint.equal, bool, Tensor expected, item"
- observed_at: "N/A - seed entry"
- backend: "all"
- failure_type: "scripts"
- root_cause: "mint.equal() returns a Python bool (not a Tensor), unlike ops.equal() which returns a Tensor; code expecting a Tensor from equal() breaks"
- solution: "Use ops.equal() if you need a Tensor result; use mint.equal() only when a scalar bool comparison is intended; check return type docs"
- last_seen: "2026-03-12"
- occurrences: 1
- issue_url: "N/A"

### mint.item() on Multi-Element Tensor
- failure_info: "RuntimeError, cannot be converted to Scalar, mint.item, elements"
- observed_at: "N/A - seed entry"
- backend: "all"
- failure_type: "scripts"
- root_cause: "mint.item() requires a single-element Tensor; calling it on a Tensor with multiple elements raises RuntimeError"
- solution: "Ensure the Tensor has exactly 1 element before calling item(); use indexing (tensor[0]) or reduction (tensor.sum()) to get a single element first"
- last_seen: "2026-03-12"
- occurrences: 1
- issue_url: "N/A"

### mint Experimental API Removed After Upgrade
- failure_info: "AttributeError, ImportError, module mint has no attribute, API removed, experimental"
- observed_at: "N/A - seed entry"
- backend: "all"
- failure_type: "framework"
- root_cause: "Many mint APIs are marked 'experimental' and may be removed or renamed across MindSpore versions; code using these APIs breaks after version upgrade"
- solution: "Check MindSpore release notes for API changes; use stable ops.* equivalents as fallback; pin MindSpore version if API stability is required"
- last_seen: "2026-03-12"
- occurrences: 1
- issue_url: "N/A"

### mint.nn Layer Parameter Validation
- failure_info: "ValueError, TypeError, in_channels, out_channels, groups, Validator, mint.nn.Conv, mint.nn.BatchNorm"
- observed_at: "N/A - seed entry"
- backend: "all"
- failure_type: "scripts"
- root_cause: "mint.nn layers (Conv, BatchNorm, etc.) use strict parameter validation via Validator; invalid params like in_channels not divisible by groups raise ValueError"
- solution: "Check parameter constraints in mint.nn layer documentation; ensure in_channels % groups == 0 and out_channels % groups == 0 for Conv layers"
- last_seen: "2026-03-12"
- occurrences: 1
- issue_url: "N/A"

### mint.distributed init_process_group Failure
- failure_info: "RuntimeError, init_process_group, TCPStore, connection refused, distributed, mint.distributed"
- observed_at: "N/A - seed entry"
- backend: "all"
- failure_type: "scripts"
- root_cause: "mint.distributed.init_process_group failed due to TCPStore connection issues — master address/port unreachable, firewall blocking, or incorrect rank configuration"
- solution: "Verify MASTER_ADDR and MASTER_PORT environment variables; check firewall rules; ensure all ranks use consistent world_size; check if master process started before workers"
- last_seen: "2026-03-12"
- occurrences: 1
- issue_url: "N/A"

### mint.optim FusedAdamW Not Available
- failure_info: "RuntimeError, NotImplementedError, FusedAdamW, not supported, backend"
- observed_at: "N/A - seed entry"
- backend: "cpu"
- failure_type: "framework"
- root_cause: "mint.optim.FusedAdamW is a performance-optimized optimizer that may not be available on all backends (especially CPU)"
- solution: "Use mint.optim.AdamW as fallback on unsupported backends; FusedAdamW is primarily optimized for Ascend/GPU"
- last_seen: "2026-03-12"
- occurrences: 1
- issue_url: "N/A"

### gen_ops.py YAML Build Error
- failure_info: "gen_ops.py, KeyError, YAML, keys structure, py_method missing, function_doc"
- observed_at: "N/A - seed entry"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "ACLNN operator YAML definition has incorrect field hierarchy, missing py_method, or missing function_doc entry; gen_ops.py fails during code generation"
- solution: "Compare YAML structure with a working operator (e.g., add); ensure op_def + api_def + function_doc are consistent; check for Chinese characters in English YAML files (encoding issues)"
- last_seen: "2026-03-12"
- occurrences: 1
- issue_url: "N/A"

### GeneralInfer Dynamic Shape Crash
- failure_info: "RuntimeError, InferShape, dynamic shape, unknown, kShapeDimAny, GetScalarValue, has_value"
- observed_at: "N/A - seed entry"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "C++ GeneralInfer does not handle dynamic shape/rank fallback — GetScalarValue used without checking has_value(), or no fallback for unknown dimensions"
- solution: "Check has_value() before accessing scalar values; return kShapeDimAny for unknown dims; return kShapeRankAny for dynamic rank; use GetArrayValue with per-element IsValueUnknown check"
- last_seen: "2026-03-12"
- occurrences: 1
- issue_url: "N/A"

### PyBoost Parameter Conversion Failure
- failure_info: "RuntimeError, TypeError, PyBoost, tuple, vector, Optional, None, LAUNCH_ACLNN, parameter"
- observed_at: "N/A - seed entry"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "PyBoost customize fails to convert MindSpore parameters to ACLNN-expected types: tuple[int] not converted to std::vector<int64_t>, Optional None not handled, or string not converted to enum"
- solution: "Add proper parameter conversion in PyBoost customize: tuple→vector, None→empty tensor or default, str→enum mapping; compare with PTA source for exact conversion logic"
- last_seen: "2026-03-12"
- occurrences: 1
- issue_url: "N/A"

### KBK Kernel Registration / Namespace Error
- failure_info: "RuntimeError, undeclared, undefined, MS_ACLNN_KERNEL_FACTORY_REG, namespace, KBK, Graph kernel"
- observed_at: "N/A - seed entry"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "KBK kernel has namespace mismatch between header and implementation, or registration macro uses wrong class name"
- solution: "Align namespace in header and .cc files; verify MS_ACLNN_KERNEL_FACTORY_REG class name matches; check forward/backward are in separate files with separate registration"
- last_seen: "2026-03-12"
- occurrences: 1
- issue_url: "N/A"

### BPROP Input/Output Count Mismatch
- failure_info: "RuntimeError, bprop, gradient, input count, output count, REG_BPROP_BUILDER, backward"
- observed_at: "N/A - seed entry"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "BPROP builder has wrong input/output count — backward inputs should equal forward inputs + 2 (out + dout), backward outputs should equal forward input count"
- solution: "Backward inputs = forward inputs + out + dout; each forward input gets one gradient output; non-differentiable inputs return ib->OutZeros(x); multi-output forward: use TupleGetItem for out"
- last_seen: "2026-03-12"
- occurrences: 1
- issue_url: "N/A"

### BPROP Dynamic Value in Graph Mode
- failure_info: "RuntimeError, bprop, Conditional, ShapeCalc, dynamic, ValueAny, graph mode, backward compile"
- observed_at: "N/A - seed entry"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "BPROP builder uses C++ if/else on scalar values that may be unknown at compile time in graph mode; needs runtime branching via Conditional/ShapeCalc"
- solution: "Check BuildValue()->ContainsValueAny() for each scalar input in bprop; use ib->Conditional() for unknown scalars instead of C++ if; use DEF_PURE_SHAPE_CALC + ib->ShapeCalc() for dynamic shape dependencies; use IsDynamicRank check with dedicated dynamic path function"
- last_seen: "2026-03-12"
- occurrences: 1
- issue_url: "N/A"

### ACLNN Composite Op Missing Sub-Operator
- failure_info: "RuntimeError, aclnn, composite, sub-operator, missing, LAUNCH_ACLNN, call chain"
- observed_at: "N/A - seed entry"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "Composite operator (maps to multiple ACLNN sub-operators) fails because one or more sub-operators are not yet adapted in MindSpore — missing YAML/Infer/PyBoost/KBK for a sub-op"
- solution: "Analyze PTA C++ implementation for all EXEC_NPU_CMD/LAUNCH_ACLNN calls; check each sub-operator status in MindSpore (YAML + Infer + PyBoost + KBK); implement missing sub-ops first (leaf before composite)"
- last_seen: "2026-03-12"
- occurrences: 1
- issue_url: "N/A"

### View Op Strides Calculation Error
- failure_info: "RuntimeError, view, strides, transpose, reshape, zero-copy, silent corruption, incorrect output"
- observed_at: "N/A - seed entry"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "View operator's strides calculation function produces incorrect output shape/strides/offset; causes silent data corruption or incorrect results"
- solution: "Verify strides calculation against PyTorch TensorShape.cpp reference; check YAML has correct view: True / graph_view: True flags; verify REG_VIEW_STRIDES_CALC_FUN registration; test with non-contiguous inputs"
- last_seen: "2026-03-12"
- occurrences: 1
- issue_url: "N/A"

### PTA-MindSpore Parameter Mismatch
- failure_info: "RuntimeError, ValueError, parameter mismatch, PTA, forward backward, different names, hidden parameter"
- observed_at: "N/A - seed entry"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "MindSpore ACLNN adaptation has parameter alignment issues with PTA: forward/backward parameter name differences, hidden hardcoded parameters in backward, Optional None handling divergence, or output tensor count mismatch"
- solution: "Audit PTA source: check op_plugin_functions.yaml for exact signatures, derivatives.yaml for backward registration, C++ implementation for hidden defaults and None handling; record all discrepancies and align MindSpore implementation"
- last_seen: "2026-03-12"
- occurrences: 1
- issue_url: "N/A"

### aclnn_config.yaml Missing Mapping (Auto-Gen Path)
- failure_info: "RuntimeError, aclnn_config, mapping, auto-generated, dispatch, operator not found"
- observed_at: "N/A - seed entry"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "Auto-generated ACLNN operator (Path 1) fails because aclnn_config.yaml is missing the operator name mapping entry"
- solution: "Add operator mapping to aclnn_config.yaml; verify YAML op_def has dispatch.enable: True without explicit Ascend field; rebuild to trigger auto-generation"
- last_seen: "2026-03-12"
- occurrences: 1
- issue_url: "N/A"

## Searchable Keywords

- Memory: OOM, EL0004, 200000, 207018, memory exhausted, CUDA out of memory, bad_alloc
- Hardware: 507010, 507053, 507054, 507056, heartbeat, ECC, link error, device hang
- Distributed: HCCL, NCCL, timeout, all_reduce, broadcast, EI0002, EI0006, mint.distributed, init_process_group, TCPStore
- Framework: context empty, 107002, graph compile, type inference, infer failed
- Operator: TBE, AKG, compile failed, not supported, fallback, E9xxxx, EBxxxx
- Environment: CANN, libascendcl.so, ASCEND_OPP_PATH, CUDA, nvidia-smi
- Mode: GRAPH_MODE, PYNATIVE_MODE, construct, ms_function, jit
- Config: set_context, device_target, precision_mode, jit_compile
- Dtype: Float16, Float32, BFloat16, cast, dtype mismatch
- Shape: shape mismatch, rank, dimensions, reshape, unsqueeze
- Mint: mint, mint.nn, mint.nn.functional, mint.optim, mint.linalg, mint.distributed, mint.special
- Mint view: jit_view_unsupported, view, squeeze, flatten, reshape, t, narrow, split, broadcast_to
- Mint API: mint.equal, mint.item, experimental, FusedAdamW, Validator
- ACLNN adaptation: gen_ops.py, YAML, op_def, api_def, function_doc, aclnn_config
- ACLNN inference: GeneralInfer, InferShape, InferType, dynamic shape, kShapeDimAny, kShapeRankAny
- ACLNN execution: PyBoost, LAUNCH_ACLNN, KBK, MS_ACLNN_KERNEL_FACTORY_REG, RunOp, customize
- ACLNN backward: BPROP, REG_BPROP_BUILDER, Conditional, ShapeCalc, OutZeros, backward
- ACLNN composite: call chain, sub-operator, bprop_expander, Meta DSL, REGISTER_FUNCTION_OP
- ACLNN view: strides, view, graph_view, REG_VIEW_STRIDES_CALC_FUN, host kernel, zero-copy
- PTA alignment: PTA, parameter mismatch, derivatives.yaml, op_plugin_functions.yaml, hidden parameter

## Observed Failures
### aclnnAll/aclnnReduceSum Duplicate Dims in Axis
- failure_info: "EZ1001, aclnnAllGetWorkspaceSize, aclnnReduceSumGetWorkspaceSize, RuntimeError, duplicate dims, Dim appears multiple times"
- observed_at: "test_f_all.py:test_f_all_axis_list_int, test_f_sum.py:test_f_sum_tuple_dim_6d"
- backend: "ascend"
- failure_type: "scripts"
- root_cause: "axis list contains duplicate dimensions (e.g. [-1, -1, -1] normalizes to [5, 5, 5]); CANN aclnnAll/aclnnReduceSum API rejects duplicate dims with ACLNN_ERR_PARAM_INVALID"
- solution: "Convert test to exception case using pytest.raises(RuntimeError) to expect the error; or change axis to non-duplicate values like [0, 1, 2]; framework-side fix would be to deduplicate axis before passing to aclnn APIs"
- last_seen: "2026-03-13"
- occurrences: 2
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICWM5W"

### Segmentaion Fault in HCCL Communication
- failure_info: "Segmentation fault, segmentation fault, HCCL, communication operator, distributed"
- observed_at: "test_allgather_input_data_type_diff"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "HCCL refactoring introduced issue; incorrect interface used during exception reporting"
- solution: "Fixed in CANN 8.3.RC1.B080"
- last_seen: "2025-11-18"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICWG49"

### conv3dtranspose CANN Compatibility
- failure_info: "RuntimeError, ops.conv3dtranspose, function, ops.py, CANN update error"
- observed_at: "test_p_conv3dtranspose_dilation_int"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "conv3dtranspose operator error after CANN package update"
- solution: "Use fixed CANN package for verification"
- last_seen: "2025-10-22"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ID0JWL"

### mint.nn.functional.max_pool2d L2_DFX_PHASE1 Missing
- failure_info: "EZ9903, RuntimeError, ValueError, AttributeError, mint.nn.functional.max_pool2d, function"
- observed_at: "test_mint_n_f_max_pool2d_float32_4d_3x9x3x5_random"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "aclnn interface coding non-standard; missing L2_DFX_PHASE1 macro before L2 interface"
- solution: "Add L2_DFX_PHASE1 macro before aclnn L2 interface call"
- last_seen: "2025-10-21"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ID0KHV"

### tensor.sigmoid Complex Type Precision
- failure_info: "tensor.sigmoid, function, ops.py, KBK, complex64, complex128"
- observed_at: "test_t_sigmoid_complex64_6d_4x9x8x8x3x7_random"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "sigmoid operator uses AICPU implementation for complex64/complex128 types, calling eigen library; CANN upgraded eigen in mid-September, benchmark changed to TensorFlow"
- solution: "Operator SE suggests modifying benchmark and lowering precision standard; experiments show error can be reduced from 0.001 to 0.01"
- last_seen: "2025-10-10"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ID0EYH"

### EI0006 Conv2D MP Strategy Mismatch
- failure_info: "EI0006, RuntimeError, ops.py, timeout, BPROP, HCCL"
- observed_at: "test_semi_auto_parallel_conv2d_strategy_4x1x1x2"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "Conv2D operator with MP causes different graphs on rank0 and rank1; communication operator execution sequence doesn't match between ranks; HCCL doesn't validate second identical communication operator"
- solution: "CCB converted to requirement JDCIR20250102001561: support execution sequence validity check at compile time"
- last_seen: "2025-10-10"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ID0FXX"

### ops.ring_attention_update Performance Regression
- failure_info: "ops.ring_attention_update, function, ops.functional, ops.RingAttentionUpdateFactory, ops.py, performance regression"
- observed_at: "test_performance_f_ring_attention_update_float16_1x64x127x64_sbh_2k"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "CANN-side rtMemcpyAsync changed; host-side non-pin memory becomes synchronous copy, causing performance regression"
- solution: "Remove actual_seq_qlen_tensor memory allocation; aclnn interface accepts nullptr input"
- last_seen: "2025-10-09"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ID0KK0"

### MindSpore-CANN Compatibility Symbol Missing
- failure_info: "RuntimeError, compatibility, timeout, custom operator, symbol missing, ge::OpDesc::UpdateOutputDesc"
- observed_at: "test_highgrad_switch_layer_grad"
- backend: "ascend"
- failure_type: "framework"
- root_cause: "MindSpore and CANN packages incompatible; CANN missing symbol ge::OpDesc::UpdateOutputDesc(unsigned int, ge::GeTensorDesc const&), causing custom operator infer so load failure"
- solution: "Modify custom operator call interface to remove use of missing symbol; latest CANN package provides this symbol, no modification needed"
- last_seen: "2025-09-28"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICXYCV"

### TypeError ops.kaiser_window Validation Order
- failure_info: "TypeError, ops.kaiser_window, function, validation order, boundary value"
- observed_at: "test_f_kaiser_window_input_type_int_periodic_int_error"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "When first parameter hits boundary value with seed=1757876499, function returns tensor before validating second parameter"
- solution: "Move second parameter validation before first parameter boundary check"
- last_seen: "2025-09-28"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICYB7J"

### MatMulFastGelu Fusion Pass Condition
- failure_info: "RuntimeError, MatMulFastGelu, function, timeout, KBK, fusion pass"
- observed_at: "test_infer_fusion_matmulelemwise_x_float32"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "For non-ATB enabled scenarios, fusion should be supported; for ATB scenarios, fusion should be removed; fusion pass code was modified and doesn't meet this constraint"
- solution: "Fix fusion pass enable condition to meet requirements"
- last_seen: "2025-09-28"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICY9BT"

### mint.gather CANN Memory Issue
- failure_info: "mint.gather, Tensor.gather, nn.Cell, precision issue, memory access, CANN Gather"
- observed_at: "test_mint_f_gather_int32_6d_4x7x9x5x5x5_random"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "CANN late July modification to Gather underlying implementation; repeated execution with certain input shapes causes memory stepping or out-of-bounds behavior"
- solution: "Use 0925 b080 CANN package to resolve"
- last_seen: "2025-09-28"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICXCRP"

### ops.tensorscatterupdate Graph Mode Precision
- failure_info: "ops.tensorscatterupdate, function, ops.tensorscatterupdate, ops.py, dynamic shape, precision issue"
- observed_at: "test_dynamic_shape_p_tensorscatterupdate_6d_float32"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "Tensor added graph mode judgment, introducing precision issue"
- solution: "Remove related mode judgment"
- last_seen: "2025-09-28"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICR0CH"

### mint.nn.functional.silu BPROP TensorMove Elimination
- failure_info: "function, mint.nn.functional, mint.nn, mint.py, nn.functional, BPROP"
- observed_at: "test_mint_n_f_silu"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "SiLU node's bprop backward uses InplaceSiLU operator which modifies forward input; inplace backward processing inserts Load and TensorMove to get pre-modification value; UpdateState optimization pass incorrectly eliminates TensorMove as non-side-effect operator"
- solution: "Don't eliminate TensorMove in UpdateState related optimization passes"
- last_seen: "2025-09-26"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICUOZY"

### ops.margin_ranking_loss Implementation Mismatch
- failure_info: "ops.margin_ranking_loss, function, ops.py, implementation mismatch, maximum vs clamp_min"
- observed_at: "test_f_margin_ranking_loss_3d_float16"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "margin_ranking_loss operator implementation inconsistent; MindSpore uses maximum_(-target * (input1 - input2) + margin, 0), torch uses torch.clamp_min(-target * (input1 - input2) + margin, 0)"
- solution: "Align MindSpore implementation with torch: ops.clamp(-target * (input1 - input2) + margin, min=0)"
- last_seen: "2025-09-26"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICZCW0"

### EZ9999 ops.auto_generate.mla ATB Block Size
- failure_info: "EZ9999, RuntimeError, ValueError, ops.auto_generate.mla, function, ops.auto_generate"
- observed_at: "test_infer_mla_kernel_batch_1024_random_forward_04"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "ATB upgrade lost block_size validation; precision issue in TP1+MTP scenario"
- solution: "Inference operator will be provided through custom operator library; ops.auto_generate.mla in mindspore will no longer be used; 20250925 CCB review: issue converted to requirement in custom operator library"
- last_seen: "2025-09-25"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICX3CU"

### HCCL CCOOL RootInfo Cleanup
- failure_info: "RuntimeError, HCCL, CCOOL, rootinfo, cleanup, timeout"
- observed_at: "test_parallel_cross_cluster_002_polymorph_D910B_graph"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "Framework logic cleans up used rootinfo after hccl initialization; in CCOOL scenario, groups can initialize hccl separately, causing one AZ group to clean up outer ccool_group's rootinfo after successful initialization, preventing other AZ groups from broadcasting rootinfo"
- solution: "CCOOL broadcasts rootinfo for same-AZ hccl_group in funcGenerateRootInfo during device-side communication group initialization; no need for outer ccool_group rootinfo broadcast"
- last_seen: "2025-09-25"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICY3OU"

### IndexError NormalizeTupleIndex Error Message
- failure_info: "IndexError, front, tensor_index.cc, NormalizeTupleIndex, error message, missing space"
- observed_at: "test_ms_parser_tensor_index_abnormal_0016_polymorph_D910A_graph"
- backend: "ascend"
- failure_type: "framework"
- root_cause: "ccsrc/frontend/operator/composite/tensor_index.cc:451 NormalizeTupleIndex error message missing space"
- solution: "Add space to NormalizeTupleIndex error message"
- last_seen: "2025-09-25"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICZ8WE"

### LITE Model Output Shape Limit
- failure_info: "LITE, export single operator model, inference, shape limit exceeded"
- observed_at: "test_f_unsqueeze_float32_8d"
- backend: "ascend"
- failure_type: "framework"
- root_cause: "Model output shape exceeds limit"
- solution: "Raise shape dimension limit"
- last_seen: "2025-09-25"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICXLSE"

### mint.nn.functional.batch_norm Backward Dtype
- failure_info: "mint.nn.functional.batch_norm, function, mint.nn.functional, mint.nn, mint.nn_functional, mint.BatchNormMock"
- observed_at: "test_dynamic_shape_mint_n_f_batch_norm_dyn_shape_2"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "mint.nn.functional.batch_norm backward output d_weight, d_bias dtype is same as forward input x, not aligned with PyTorch; should be same as forward input weight, bias dtype per autodiff rules"
- solution: "Modify Bprop function output dtype to match forward input weight, bias dtype"
- last_seen: "2025-09-24"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICYJNM"

### vllm-mindspore Version Mismatch
- failure_info: "vllm, timeout, PyBoost, version mismatch, libmindspore_extension.so, libpyboost.so"
- observed_at: "test_vllm_mf_online_mtp_001"
- backend: "ascend"
- failure_type: "platform"
- root_cause: "CI machine builds vllm-mindspore whl with mindspore package from dependent_packages.yaml (9.15 daily); test machine uses latest mindspore daily (9.22); vllm-mindspore custom operators link mindspore dynamic library; 9.19 mindspore custom operator directory restructuring removed libmindspore_exten..."
- solution: "Update dependent_packages.yaml to depend on new mindspore; future versions will decouple, vllm-mindspore only depends on stable mindspore releases"
- last_seen: "2025-09-24"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICVK6M"

### ops.ctc_loss InferShape Value Dependency
- failure_info: "ValueError, AttributeError, ops.ctc_loss, function, ops.py, nn.Cell"
- observed_at: "test_f_ctc_loss_float32_3d_8x8x9_inf"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "GE modified CTCLossV2 infershape to value-dependent inference"
- solution: "Modify MindSpore infershape to value-dependent infershape"
- last_seen: "2025-09-23"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICX4S0"

### custom_ops Header File Compatibility
- failure_info: "custom_ops, timeout, custom operator, header file, ms_extension.h, custom_op_api.h"
- observed_at: "test_custom_cpp_function_muiti_op_tensor_hook"
- backend: "ascend"
- failure_type: "platform"
- root_cause: "Custom operator directory restructured, external header replaced with custom_op_api.h; old version headers need rename for compatibility but one file was missed"
- solution: "Rename custom_op_api.h to ms_extension.h, keep both versions with ms_extension/api.h and ms_extension/all.h"
- last_seen: "2025-09-23"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICVCUA"

### BPROP Morph Monad Parameter Cache
- failure_info: "BPROP, Morph, monad parameter, func graph cache, parameter mismatch"
- observed_at: "test_parallel_morph_custom_bprop_001"
- backend: "ascend"
- failure_type: "scripts"
- root_cause: "Morph pass parses Morph forward function and modifies resulting func graph by adding monad parameter; func graphs generated during parsing are cached; if Morph's bprop calls forward function, it hits cache but forward func graph has added monad parameter, causing bprop call parameter mismatch"
- solution: "When parsing forward function, get cached copy to modify instead of modifying cache itself"
- last_seen: "2025-09-19"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICWMVN"

### RuntimeError host_comm_lib_instance_ GE DryRun
- failure_info: "RuntimeError, host_comm_lib_instance_, PyBoost, HCCL, GE dryrun, host_collective"
- observed_at: "test_parallel_sharding_propagation_with_save_load_008"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "In GE scenario, dryrun mode creates real hccl communication domain but doesn't create host_collective; after initializing hccl communication group, need to cleanup unique_id which requires host_collective, causing failure in GE dryrun mode"
- solution: "Skip unique_id cleanup in non-host_collective scenarios"
- last_seen: "2025-09-19"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICT2XZ"

### HCCL destroy_process_group Async Send/Recv
- failure_info: "RuntimeError, HCCL, destroy_process_group, async send/recv, teardown, synchronization"
- observed_at: "test_hccl_mccl_mint_send_recv_008"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "pytest teardown actively calls destroy_process_group which destructs CPU communication resources; since send/recv are async and test only calls send/recv, Python script reaches teardown before actual communication operator tasks complete"
- solution: "Add synchronization in destroy_process_group to ensure operator tasks complete before destruction"
- last_seen: "2025-09-19"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICTQH2"

### KBK IsCompileSimulation Mock ACL
- failure_info: "KBK, IsCompileSimulation, mock acl, GE dryrun level1"
- observed_at: "test_parallel_sharding_propagation_with_save_load_008"
- backend: "ascend"
- failure_type: "platform"
- root_cause: "common::IsCompileSimulation() interface doesn't check if current mode is KBK, causing GE mode dryrun level1 to also fully mock ACL interfaces"
- solution: "Change to UseSimulationApi() interface so only KBK mode has mock ACL interface behavior"
- last_seen: "2025-09-19"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICT2XZ"

### DistributedMeta local_rank_id Device ID
- failure_info: "timeout, DistributedMeta, local_rank_id, device_id, communication domain uninitialized"
- observed_at: "test_custom_mem_pool_check_002"
- backend: "ascend"
- failure_type: "platform"
- root_cause: "When communication domain is uninitialized, DistributedMeta::GetInstance()->local_rank_id() always returns device_id=0, not matching actual configured device_id, causing memory allocation failure"
- solution: "Call ms_context->get_param(MS_CTX_DEVICE_ID) to get correct device_id"
- last_seen: "2025-09-19"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICVB8M"

### CPUDeviceAddress Ascend Operator Execution
- failure_info: "compiler, timeout, custom operator, CPUDeviceAddress, Ascend operator, device type mismatch"
- observed_at: "test_custom_cpp_function_muiti_op_tensor_hook"
- backend: "ascend"
- failure_type: "platform"
- root_cause: "Output Tensor created CPUDeviceAddress, using CPUDeviceAddress to execute Ascend operator fails"
- solution: "Only skip creating output address when output address matches current DeviceType"
- last_seen: "2025-09-19"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICVCUA"

### aclrtDestroyEvent Null Check
- failure_info: "aclrtDestroyEvent, null check, resource release order, destructor"
- observed_at: "test_ascend_resnet50_modelparallel_2p_8192_polymorph_D910B"
- backend: "ascend"
- failure_type: "scripts"
- root_cause: "aclrtDestroyEvent call missing null check; object destruction cannot guarantee resource release order; MindSpore needs explicit object release on exit"
- solution: "Add null check for aclrtDestroyEvent call; explicitly release objects when MindSpore exits"
- last_seen: "2025-09-19"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICV5CF"

### vllm ops.AllReduce group_size=1
- failure_info: "RuntimeError, vllm, ops.AllReduce, ops.ReduceOp, timeout, DP+EP"
- observed_at: "test_vllm_mf_server_dp_moe_ep_001"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "mindformers introduced; reproducible in DP+EP scenario with single machine"
- solution: "Adapt logic: skip communication when group_size equals 1"
- last_seen: "2025-09-19"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICVK6M"

### Segmentation fault Null Pointer Access
- failure_info: "Segmentation fault, null pointer access, other feature introduced"
- observed_at: "test_alltoallv_check_001"
- backend: "ascend"
- failure_type: "framework"
- root_cause: "Other feature introduced null pointer access"
- solution: "PR merged; issue no longer reproduces after merge"
- last_seen: "2025-09-19"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICSORQ"

### BN gamma 5HD Format AllGather
- failure_info: "RuntimeError, semi-auto parallel, BatchNorm, gamma, 5HD format, AllGather"
- observed_at: "test_semi_auto_parallel_batchnorm_strategy_set_001"
- backend: "ascend"
- failure_type: "framework"
- root_cause: "Previous graph's BN gamma weight format is 5HD, passed to next graph's AllGather; input format mismatch error"
- solution: "For 5HD with host shape 1D, don't pass special format"
- last_seen: "2025-09-19"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICR1XL"

### RuntimeError ops.Addrmsnormquant ATB Quant
- failure_info: "RuntimeError, ValueError, AttributeError, ops.Addrmsnormquant, function, ops.py"
- observed_at: "test_dynamic_shape_infer_add_rmsnorm_quant_dyn_shape_1"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "Addrmsnormquant test case: ATB version upgrade, Quant operator spec changed, doesn't support scale and offset as scalar Tensor; aclnnAscendQuant doesn't support offset type int8"
- solution: "Quant spec changed, need to use according to aclnn method (change notification sent)"
- last_seen: "2025-09-18"
- occurrences: 2
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICX3BP"

### EZ1001 GroupedMatmul Internal Input Count
- failure_info: "EZ1001, RuntimeError, ValueError, AttributeError, GroupedMatmul, function"
- observed_at: "test_infer_grouped_matmul_fp16_2d_2x16x16x8_random_forward"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "Internal operator GroupedMatmul registered input count doesn't match MindSpore passed count, causing operator selection to fail and fall back to aclnn"
- solution: "Fix MindSpore internal input index map, remove unneeded inputs so input count and type check pass, can normally select internal operator"
- last_seen: "2025-09-18"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICVR9S"

### RuntimeError mint.py View Tensor Inplace NoGrad
- failure_info: "RuntimeError, function, mint.py, view tensor, inplace, no_grad"
- observed_at: "test_mint_n_f_threshold__float32_2d_discontinuous_tensor"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "Test cases do inplace operation on view tensor in no_grad mode, then use it as function input for gradient calculation, causing error: A view of base is being rebase, which created in no_grad mode and inplace modified with grad mode enabled"
- solution: "Add logic to skip version and creation type check for this case"
- last_seen: "2025-09-18"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICWN6V"

### ValueError mint.nn.Linear Empty Tensor
- failure_info: "ValueError, mint.nn.Linear, function, mint.nn, mint.py, ops.composite"
- observed_at: "test_mint_n_f_linear_float32_1d_0_none_1d_0_none_0d"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "Reshape refactored, added validation for some empty Tensor scenarios"
- solution: "Modify Dense backward implementation logic"
- last_seen: "2025-09-18"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICWZB1"

### EZ1001 aclnnGroupedMatmul Empty Tensor Strides
- failure_info: "EZ1001, RuntimeError, aclnnGroupedMatmul, function, ops.py, empty tensor"
- observed_at: "test_f_gmm_group_type_0_bias_none_fp32_x_weight_empty"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "Non-issue: Reshape refactor aligned behavior with torch; empty Tensor reshape has all strides=1, Hisilicon can't detect if tensor is transposed via strides; aligned with Hisilicon: group_type=2 doesn't support empty Tensor for x, will document and validate empty Tensor"
- solution: "Test can use workaround for fix"
- last_seen: "2025-09-18"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICX6QM"

### mint.nn.PixelShuffle IsContiguous Bug
- failure_info: "mint.nn.PixelShuffle, mint.nn, nn.PixelShuffle, IsContiguous, empty tensor, bug"
- observed_at: "test_mint_n_pixelshuffle_float32_5d_0_none"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "IsContiguous function has bug, calculation result incorrect"
- solution: "Fix IsContiguous bug: empty Tensor must be contiguous"
- last_seen: "2025-09-17"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICX71Q"

### RuntimeError ops.DynamicNTK Empty Shape
- failure_info: "RuntimeError, ops.DynamicNTK, function, ops.py, inference operator, empty shape"
- observed_at: "test_infer_dynamic_shape_dynamicntk_dyn_shape_3"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "Test case operator input is empty shape, causing underlying operator tiling to fail"
- solution: "Modify single operator test case to avoid giving operator empty shape input"
- last_seen: "2025-09-11"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICX62E"

### mint.cumsum Float16 Precision Standard
- failure_info: "mint.cumsum, function, mint.py, aclnn, float16, precision standard"
- observed_at: "test_mint_f_cumsum_float16_1d_8_random"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "mint.cumsum uses aclnn operator; operator developers use dual benchmark for precision; test repository doesn't use this method"
- solution: "Test repository doesn't use dual benchmark method for precision; need to specify precision for this operator in float16, experiment shows loss=0.002 passes"
- last_seen: "2025-09-11"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICWZN8"

### RuntimeError mint.nn.LayerNorm Size Zero
- failure_info: "RuntimeError, mint.nn.LayerNorm, mint.nn, nn.LayerNorm, size=0, parameter store"
- observed_at: "test_mint_n_layer_norm_float32_1d_0_none"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "Parameter store skipped size=0 address allocation"
- solution: "size=0 input also needs address allocation flow; fixed in CPU flow switch PR"
- last_seen: "2025-09-10"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICSKA7"

### Custom Memory Pool 910A Memory Conflict
- failure_info: "interface, custom memory pool, 910A, 910B, memory allocation, physical memory"
- observed_at: "test_custom_mem_pool_check_001"
- backend: "ascend"
- failure_type: "framework"
- root_cause: "910A and 910B memory pool and allocation default behavior differs; 910A allocates all physical memory at initialization, preventing custom memory pool from getting memory; custom memory pool feature conflicts with 910A default memory allocation"
- solution: "Add interface description limiting this interface to 910B only"
- last_seen: "2025-09-08"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICVB8M"

### RuntimeError ops.Meshgrid View Tag
- failure_info: "RuntimeError, ValueError, AttributeError, ops.Meshgrid, function, ops.py"
- observed_at: "test_f_cartesian_prod"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "Meshgrid Dispatch but didn't implement aclnn kernel mod, depends on view tag to use aclOp; after removing view tag, directly throws exception"
- solution: "Don't modify operator selection logic for now, add back view tag; meshgrid is pure view operator concatenation implementation, add view tag to use basic type input for better performance"
- last_seen: "2025-09-04"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICVFDG"

### RuntimeError View Inplace Manager Null
- failure_info: "RuntimeError, ValueError, AttributeError, ops.abs, nn.Conv2d, nn.GraphCell"
- observed_at: "test_view_and_inplace_improve_020"
- backend: "ascend"
- failure_type: "framework"
- root_cause: "In view inplace scenario with free variable handling, need Renormalize to update abstract; didn't set manager for graph causing null manager error"
- solution: "Preserve logic for setting manager on graph"
- last_seen: "2025-09-03"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICVHH1"

### RuntimeError vllm ATB Quant Per-Tensor
- failure_info: "RuntimeError, vllm, timeout, ATB quant, per-tensor, aclnn operator"
- observed_at: "test_vllm_mf_online_mtp_001"
- backend: "ascend"
- failure_type: "platform"
- root_cause: "MindSpore upgrade caused ATB quant operator to not support per-tensor, changed to aclnn operator"
- solution: "Modify weight loading logic: select aclnn operator for quantization; vllm weight loading: input_zp cast to bf16, input_scale divide by 1"
- last_seen: "2025-09-02"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICVK6M"

### EZ1001 nn.focalloss Multi-Thread Exception
- failure_info: "EZ1001, RuntimeError, TypeError, nn.focalloss, function, ops.py"
- observed_at: "test_n_focalloss_predict_dtype_int_uint"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "Multi-thread test execution throws multiple exceptions, can't stop on first exception like single-thread"
- solution: "Coordinate with test team to change test case to single-thread"
- last_seen: "2025-09-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICR89S"

### ValueError interleaved_parallel AllToAll
- failure_info: "ValueError, interleaved_parallel, alltoall, rearrangement, restriction"
- observed_at: "test_parallel_online_get_strategy_metadata_004"
- backend: "ascend"
- failure_type: "platform"
- root_cause: "interleaved_parallel scenario doesn't support alltoall rearrangement but no restriction was added; historical issue"
- solution: "Disable alltoall rearrangement by default when using interleaved_parallel"
- last_seen: "2025-09-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICTA6T"

### mint.max Empty Tensor Dirty Memory
- failure_info: "mint.max, mint.min, empty tensor, dirty memory, max/min value"
- observed_at: "test_mint_f_max_overload1_int32_1d_0_none"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "Empty tensor has no max/min value, returned dirty memory"
- solution: "CCB review: this scenario is exception case, needs error handling"
- last_seen: "2025-08-27"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICSLDJ"

### ops.logcumsumexp Float16 Precision
- failure_info: "ops.logcumsumexp, function, ops.py, float16, precision standard, tensorflow benchmark"
- observed_at: "test_f_logcumsumexp_input_1d_float16_0"
- backend: "all"
- failure_type: "backend"
- root_cause: "logcumsumexp operator in fp16 exceeds 0.001 precision standard relative to fp32 benchmark; this is precision instability due to data type"
- solution: "logcumsumexp operator doesn't support fp16 backward in torch, tf-based benchmark meets 0.001 standard; adjust test precision standard from 0.001 to 0.005"
- last_seen: "2025-09-26"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICZDFQ"

### AVX/AVX512 vs ARM/SSE MatMul Precision Inconsistency
- failure_info: "avx, avx512, arm, sse, matmul, precision, col alignment, lstm"
- observed_at: "N/A"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "AVX-accelerated matmul operator requires 8-aligned col parameter; lstm operator calls matmul without 8-aligning col parameter, causing precision error"
- solution: "Add 8-alignment check for col parameter in MatVecMulAvxFp32 function"
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICFZQ3"

### JIT Strict Mode Recompute Cell TypeError
- failure_info: "TypeError, JIT, strict mode, Recompute, Cell, CellList, static graph"
- observed_at: "test_mindone_sdxl_common_fusion_check"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "Recompute implementation removed mode check, default added a Recompute sub-Cell; in CellList scenario, this sub-Cell was added to CellList, causing static graph to parse RecomputeCell with unsupported syntax"
- solution: "Add mode check back: don't create sub-Cell in static graph mode; CellList module owner should fix the bug of adding sub-Cell to CellList"
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICLQEI"

### LLaMA2 70B PTQ Multi-Batch Performance Regression
- failure_info: "pynative, reshape, performance, llama2, ptq, quantization, c8"
- observed_at: "test_ms_llama2_70b_quant_ptq_bf16_w8a8c8_infer_4p_0001"
- backend: "ascend"
- failure_type: "framework"
- root_cause: "Flatten operation generated extra pynative reshape operators, causing overall performance degradation; c8 optimization not effective due to flatten issues"
- solution: "Modify golden-stick to enable c8; use numpy implementation for pynative reshape operator"
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICMFUA"

### functional f_clip_by_norm List Tensor Device Access
- failure_info: "Tensor, Device, Host, data access, clip_by_norm"
- observed_at: "test_f_clip_by_norm_x_is_list_of_tensor_max_norm_2_norm_type_6_error_if_nonfinite_false"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "Tensor directly accessing Ascend data triggers exception; framework code had many illegal data access points, Tensor refactoring added mandatory validation requiring Device-to-Host copy for data access"
- solution: "Access data locations need to copy to CPU first; original code flow was incorrect, now has mandatory validation"
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICR235"

### PadV3 Dynamic Shape Memory Corruption
- failure_info: "cann, PadV3, dynamic shape, memory corruption, backward"
- observed_at: "test_nn_pad_nd"
- backend: "ascend"
- failure_type: "platform"
- root_cause: "CANN PadV3 backward has memory stomp behavior, corrupting forward benchmark output results"
- solution: "Trigger condition is strict, single operator cannot reproduce; CANN has plan to replace PadV3 but not scheduled this year"
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICR3XQ"

### Tensor move_to Ascend DeviceAddress Error
- failure_info: "Tensor, move_to, Ascend, DeviceAddress, validation"
- observed_at: "test_ms_tensor_device_003_polymorph_D910B_pynative"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "Tensor.move_to('Ascend') has incorrect Tensor Address; device address validation failed without move_to CPU"
- solution: "Create Tensor Ascend DeviceAddress correctly"
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICRCSO"

### Parallel Static Graph Consecutive Reshape Layout Mismatch
- failure_info: "Reshape, layout, parallel, static graph, redistribution"
- observed_at: "N/A"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "In static graph parallel mode with consecutive Reshape, redistribution operator insertion is incorrect; when Reshape input shape differs from in_layout shape, direct replacement causes issues"
- solution: "After deriving redistribution list in Reshape distributed operator, if in shape differs from in_layout shape and first redistribution operator is not Reshape, insert a Reshape operator at the beginning to convert to in_layout shape before redistribution"
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICS610"

### ops.expm1 Precision Issue in Small Value Range
- failure_info: "precision, expm1, Taylor expansion, small value range"
- observed_at: "test_f_expm1_x_0d_float32"
- backend: "ascend"
- failure_type: "platform"
- root_cause: "expm1 operator precision not meeting standard for certain inputs; uses Taylor expansion which is advantageous for values near 0, mainly improving small value range precision"
- solution: "CANN operator has scene limitations; use within small value range (-1e-7 to 1e-7); for non-small value range, use exp(x)-1 instead"
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICSFL8"

### CPU floor_mod fp16 Precision With Fixed Seed
- failure_info: "precision, floor_mod, fp16, seed, gradient, accumulation error"
- observed_at: "test_f_floor_mod_x_1d_float16_y_1d_float16"
- backend: "cpu"
- failure_type: "backend"
- root_cause: "CPU floor_mod operator has precision error with fp16 input at fixed seed=1754755021; small input y values cause gradient accumulation error in backward calculation; fp32 precision is correct at this seed"
- solution: "Modify test benchmark to convert input to fp32 for calculation then convert back to fp16"
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICT0YC"

### CPU addcdiv Precision With Small Input2 Values
- failure_info: "precision, addcdiv, inf, division by zero, input2"
- observed_at: "test_p_addcdiv_input_fp32_value_3"
- backend: "cpu"
- failure_type: "platform"
- root_cause: "Randomly generated test data too small; input2 parameter less than 1e-6 treated as division by zero, resulting in inf; pytorch has more precise implementation"
- solution: "Control input2 parameter generation range in test case to be greater than 1e-6"
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICTS23"

### CPU exp Precision Overflow With ReduceMean
- failure_info: "precision, exp, reducemean, overflow"
- observed_at: "test_dynamic_shape_p_exp_8d_float32"
- backend: "cpu"
- failure_type: "backend"
- root_cause: "exp operator has precision difference from tensorflow at fixed seed; original calculation method causes operator overflow"
- solution: "Modify monitoring method to use exp operator directly instead of reducemean"
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICTS70"

### Profiler Core Dump on CANN 8.3.RC1.B020
- failure_info: "CANN, profiler, core dump, RC1, B020"
- observed_at: "test_dvm_profiler"
- backend: "ascend"
- failure_type: "platform"
- root_cause: "CANN 8.3.RC1.B020 has known issue where profiler collection causes core dump in some scenarios"
- solution: "Not a MindSpore issue; CANN fixed in B021 and later versions"
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICTY36"

### Rprop Precision With Dense and CrossEntropyLoss
- failure_info: "precision, rprop, Dense, CrossEntropyLoss, gradient"
- observed_at: "test_n_rprop_forward_inputn_16_inputc_1024_outputc_16_basic"
- backend: "ascend"
- failure_type: "platform"
- root_cause: "Rprop operator precision occasionally fails; test case combines Net:Dense + loss:CrossEntropyLoss + optimizer:Rprop; Dense forward error causes CrossEntropyLoss backward gradient difference; CrossEntropyLoss gradient is very sensitive to input"
- solution: "Ensure Dense layer forward output error is small enough; current error mae_diff: 0.00001532 mre_diff: 0.00000198"
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICU6UD"

### Feature Value Detection Not Supporting Data Parallel
- failure_info: "precision, feature value detection, data parallel, mirror, gradient"
- observed_at: "test_feature_value_detect_resnet_data_para_open_npu_asd"
- backend: "ascend"
- failure_type: "framework"
- root_cause: "New feature value detection does not support data parallel mode; feature value sampling needs to get gradient before communication from mirror operator, which depends on parallel flow inserting mirror operator; auto parallel/semi-auto parallel modes go through framework parallel flow, but data parallel mode does not"
- solution: "Add warning prompt when using silent detection in non-auto parallel/semi-auto parallel mode"
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICU9KA"

### msadapter.autograd.backward Tensor DeepCopy Inconsistency
- failure_info: "msadapter, autograd, backward, tensor, deepcopy, gradient"
- observed_at: "test_msa_operations_check_017"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "msadapter.autograd.backward(r1, grad) execution error, no computable gradient; ms framework tensor deepcopy behavior differs from torch"
- solution: "Override tensor.deepcopy method to fix the inconsistency"
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICUA4L"

### Custom Operator View Inplace Null Pointer Error
- failure_info: "RuntimeError, view, inplace, RebaseVariable, null pointer, custom operator"
- observed_at: "test_msa_operations_check_023"
- backend: "ascend"
- failure_type: "scripts"
- root_cause: "Custom operator view + inplace scenario reports null pointer error; RebaseVariable converts grad_node to FuncBackwardNode type, but custom operator grad_node is not this type causing null pointer; after fixing null pointer, custom operator backward function doesn't run because inplace operation updates view tensor version before UpdateNextEdge"
- solution: "Move the logic out of RebaseVariable to specified location; modify custom operator implementation to do backward edge connection first, then run forward"
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICUA5P"

### as_strided bfloat16 Precision Issue
- failure_info: "precision, as_strided, bfloat16, torch_npu"
- observed_at: "test_t_as_strided_bfloat16_2d_8x8_random"
- backend: "ascend"
- failure_type: "platform"
- root_cause: "Tensor.as_strided operator bfloat16 test case has precision issue, doesn't meet default loss=0.004 standard; torch_npu backward output is binary consistent with ms backward"
- solution: "Set bfloat16 loss to 0.04; running 300 test cases without failure"
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICUE0Q"

### atexit Resource Release Order Causing Wild Pointer
- failure_info: "atexit, DeviceContext, CollectiveManager, wild pointer, core"
- observed_at: "test_msrun_tail_all_renamed_worker_log"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "Level1 test case only calls init interface then hangs core; in atexit hook, DeviceContext is released before CollectiveManager, causing wild pointer access when communication domain initialization is slow"
- solution: "Modify resource release order in atexit, put dependent modules later"
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICUH4S"

### Model Conversion ERROR Log for Unsupported Constant Folding
- failure_info: "ascend, model conversion, constant folding, error log"
- observed_at: "N/A"
- backend: "ascend"
- failure_type: "framework"
- root_cause: "Ascend model conversion scenario reports error log for operators not supported by constant folding pass, but doesn't affect conversion success; need to change log level"
- solution: "For ascend model conversion scenario, change bottom-level log level to warning for unsupported operators in constant folding pass; for cpu scenario, upper layer has error exit log, doesn't affect original logic"
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICUHZW"

### CPU ops.prod Precision With Small Values
- failure_info: "precision, ops.prod, prod, seed, division, small value"
- observed_at: "test_f_prod_input_x_2d_float16_1_true"
- backend: "cpu"
- failure_type: "backend"
- root_cause: "CPU ops.prod operator precision occasionally fails at fixed seed; forward has a very small value, then prod backward has division, dividing by this small value causes error to be amplified"
- solution: "Modify test case to clip input value lower bound"
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICUL5V"

### Custom Optimizer requires_grad False Still Computes Gradient
- failure_info: "grad_node, grad_fn, UpdateNextEdge, requires_grad"
- observed_at: "test_msa_interface_combination_check_004"
- backend: "ascend"
- failure_type: "framework"
- root_cause: "Leaf node originally needed gradient, then set to not need gradient, but still carries grad_node (grad_fn); UpdateNextEdge still connects during backward edge connection, causing gradient computed for leaf node that shouldn't need it"
- solution: "Modify UpdateNextEdge logic to skip this case during edge connection"
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICULDO"

### CPU Graph Mode AddN Single Input Error
- failure_info: "AddN, merge, single input, graph mode"
- observed_at: "test_p_addn_normal_input_n_1"
- backend: "cpu"
- failure_type: "framework"
- root_cause: "AddN operator with only one input reports error; frontend AddN merge pass doesn't properly handle single input case"
- solution: "For AddN that doesn't need ZeroFilter, continue to do single input simplification optimization"
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICUMFB"

### CPU Graph Mode BCEWithLogitsLoss Large Shape Precision
- failure_info: "precision, BCEWithLogitsLoss, shape, accumulation, float16"
- observed_at: "test_bce_with_logits_loss_input_4d_float16"
- backend: "cpu"
- failure_type: "framework"
- root_cause: "CPU graph mode BCEWithLogitsLoss operator precision fails at fixed seed; operator input shape too large, accumulation has thousandths precision difference"
- solution: "Modify test case to change loss from 1e-3 to 2e-3"
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICUMGM"

### Inplace Next UpdateState Null Pointer Error
- failure_info: "inplace, TensorMove, UpdateState, null pointer"
- observed_at: "test_view_and_inplace_nested_ctrl_if_for_while_and_if_while_for_and_dynamic_rank"
- backend: "ascend"
- failure_type: "framework"
- root_cause: "After backward differentiation inserts TensorMove, there exists scenario where Inplace input UpdateState and inplace_next_updatestate input UpdateStatus are inconsistent; after fix, random precision issue appears in backward"
- solution: "Only find UpdateState user using target Inplace node; don't eliminate TensorMove in UpdateState related optimization pass"
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICUOJI"

### CPU RandomChoiceWithMask VMap Performance Test
- failure_info: "RandomChoiceWithMask, vmap, performance"
- observed_at: "test_vmap_p_randomchoicewithmask_64"
- backend: "cpu"
- failure_type: "backend"
- root_cause: "RandomChoiceWithMask vmap performance test fails; SEG conclusion reached to remove CPU operator vmap performance test cases without customer demand"
- solution: "Remove CPU operator vmap performance test cases without customer demand"
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICUUSW"

### RMSNorm Quant Test Log Validation Error
- failure_info: "cmake, SUBMODULE, ID, log validation"
- observed_at: "test_rmsnorm_quant"
- backend: "ascend"
- failure_type: "framework"
- root_cause: "Test case log validation failed; during code refactoring, cmake missed SUBMODULE ID information, causing test case validation log info to not exist"
- solution: "Fix missing SUBMODULE ID information"
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICUVH9"

### Morph Custom BPROP Gradient Count Validation
- failure_info: "Morph, bprop, gradient count, validation"
- observed_at: "test_morph_custom_bprop_005"
- backend: "ascend"
- failure_type: "framework"
- root_cause: "When custom bprop returns incorrect number of gradients, error message is unclear; no legality validation when custom bprop returns wrong gradient count"
- solution: "Add legality validation when custom bprop returns incorrect gradient count"
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICUXJ4"

### MSLite and Torch_NPU Mixed Inference Resource Conflict
- failure_info: "mslite, torch_npu, acl, context, resource conflict"
- observed_at: "N/A"
- backend: "ascend"
- failure_type: "platform"
- root_cause: "MSLite inference and torch_npu mixed inference resource usage exception; mslite and torch_npu share acl context, torch_npu uses and releases context created by mslite, causing exception"
- solution: "Before executing ModelInfer related functions, save current state first, then execute, restore state on exit to avoid exception"
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICV0YS"

### IsCompileSimulation KBK Mode Mock ACL Issue
- failure_info: "IsCompileSimulation, kbk, mock acl, dryrun, strategy"
- observed_at: "test_parallel_sharding_propagation_with_save_load_008"
- backend: "ascend"
- failure_type: "platform"
- root_cause: "common::IsCompileSimulation() interface doesn't check whether current is kbk mode, causing ge mode dryrun level1 to also fully mock acl interface"
- solution: "Change to UseSimulationApi() interface, so mock acl interface behavior only happens in kbk mode"
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICV313"

### AST Nested Trace kwargs UnpackCall Error
- failure_info: "ast, kwargs, unpackcall, trace, nested"
- observed_at: "test_view_and_inplace_nested_ctrl_if_for_while_and_if_while_for_and_dynamic_rank"
- backend: "ascend"
- failure_type: "framework"
- root_cause: "AST nested trace scenario input kwargs error; when ast parses kwargs, it wraps a layer of unpackcall outside subgraph, causing trace node to only get formal parameters, unable to get python objects stored in previous flow"
- solution: "For AST nested trace scenario, let ast take over kwargs input"
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICV39M"

### AddRmsNormQuant Performance With Large Non-Reduce Axis
- failure_info: "AddRmsNorm, Quant, performance, Reduce, non-reduce axis"
- observed_at: "test_infer_add_rmsnorm_quant_bfloat16_performence_2"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "Test fails on 910B1/B2/B3, only passes on 910B4; when non-reduce axis is very large (330K in test case) and reduce axis is relatively small (4000), current fusion operator reduces performance of post-fusion Quant, causing fused kernel performance worse than unfused (8.6% degradation); real scenario DeepSeek has reduce axis around 7168"
- solution: "Performance monitoring focuses on real scenarios; in DeepSeek, reduce axis is 7168, non-reduce axis is generally within 1024"
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICV9D8"

### ColumnParallelLinear A16W4 GPTQ Quantization ValueError
- failure_info: "ValueError, gather_from_model_parallel_region, ColumnParallelLinear, GPTQ, quantization"
- observed_at: "test_ptq_moe_a16w4_gptq_columnparallel_other_param_fp32"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "Golden-stick quantization stage error; gather_from_model_parallel_region parameter error"
- solution: "Change gather_from_model_parallel_region parameter to output_parallel"
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICVB2B"

### aclrtDestroyEvent Null Check Error
- failure_info: "aclrtDestroyEvent, null check, event, destructor"
- observed_at: "test_ascend_resnet50_modelparallel_2p_8192_polymorph_D910B"
- backend: "ascend"
- failure_type: "platform"
- root_cause: "MindSpore exit, object destructor calls CANN aclrtDestroyEvent interface error; aclrtDestroyEvent call doesn't check null; object destructor cannot guarantee resource release order"
- solution: "Add null check before calling aclrtDestroyEvent; explicitly release objects during MindSpore exit"
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICVBE0"

### Dataset Independent Process NoSuchProcess Error
- failure_info: "lscpu, psutil, NoSuchProcess, subprocess"
- observed_at: "test_datasets_independent_process"
- backend: "cpu"
- failure_type: "backend"
- root_cause: "When running test case and starting process, found 'lscpu' and other subprocesses, eventually reports psutil.NoSuchProcess error"
- solution: "Add catch for psutil.NoSuchProcess error in test case to let program continue"
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICVSSG"

### msadapter Tensor requires_grad AttributeError
- failure_info: "AttributeError, msadapter, tensor, __init__, requires_grad"
- observed_at: "test_vllm_mf_online_mtp_001"
- backend: "ascend"
- failure_type: "framework"
- root_cause: "msadapter tensor.__init__() method has incorrect requires_grad assignment; vllm-mindspore weight loading has param requires_grad attribute as True, causing inplace operation failure"
- solution: "Fix msadapter requires_grad assignment to assign to self._requires_grad; set param requires_grad to False during model initialization in vllm-mindspore"
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICVT67"

### RepeatInterleave Not Supported on 910A
- failure_info: "RuntimeError, aclnn, aclnnRepeatInterleaveInt, 910A"
- observed_at: "test_repeat_interleave"
- backend: "ascend"
- failure_type: "platform"
- root_cause: "910A doesn't support RepeatInterleaveInt; performance optimization prioritizes aclnn, but 910A doesn't support aclnnRepeatInterleaveInt"
- solution: "Modify overload yaml, pynative/KBK both prioritize aclOp"
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICVU9J"

### CPU cummax fp32 vs fp64 Precision
- failure_info: "cummax, fp32, fp64, precision"
- observed_at: "test_p_cummax_op_normal_input"
- backend: "cpu"
- failure_type: "backend"
- root_cause: "Random seed generates random values with too small difference, exceeding fp32 precision range; fp64 can distinguish; MindSpore uses fp64 for calculation with correct result, while torch uses fp32 with result exceeding precision range"
- solution: "Modify test case torch calculation precision to be consistent with MindSpore version"
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICW0P0"

### DP+MOE_EP Group Not Created Error
- failure_info: "RuntimeError, mindformers, Group, MOE, EP"
- observed_at: "test_vllm_mf_server_dp_moe_ep_001"
- backend: "ascend"
- failure_type: "platform"
- root_cause: "DP+MOE_EP(all to all), 16 cards, parallel strategy data_parallel raises RuntimeError: Group is not created; mindformers introduced; reproducible in DP+EP scenario"
- solution: "Golden-stick adapts this logic, skip communication when group_size equals 1"
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICW0QI"

### ops.squeeze 0D Input Not Supported in Static Graph
- failure_info: "ops.squeeze, squeeze, 0D, static graph"
- observed_at: "test_f_squeeze_input_0d_float32_axis_0"
- backend: "all"
- failure_type: "backend"
- root_cause: "Static graph scenario ops/mint.squeeze doesn't support 0D input, inconsistent with dynamic graph and torch behavior"
- solution: "Modify squeeze infer implementation so static graph also supports this scenario; after modification, Ascend backend static graph kbk supports, but ge mode using GeOp still doesn't support"
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICW6ZF"

### ops.unbind Empty Output Error
- failure_info: "ops.unbind, unbind, pynative, view, empty output"
- observed_at: "test_f_unbind_int8_1d_0_none"
- backend: "all"
- failure_type: "backend"
- root_cause: "Output cannot be empty; pynative framework view operator has two flows: graph composition allows empty output, pyboost flow has strong constraint on output not allowing empty"
- solution: "Move constraint to pyboost flow call location; keep graph composition allowing empty output; pyboost flow to remove constraint after reviewing all view operators"
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICW75C"

### CPU FastGELU Large Input Underflow
- failure_info: "FastGELU, CPU, underflow, large input"
- observed_at: "test_n_fastgelu_input_5d_fp16"
- backend: "cpu"
- failure_type: "backend"
- root_cause: "FastGELU backward CPU operator, when computing with large input values, internal intermediate calculation variables use same data type as input, causing numerical underflow and result becoming 0"
- solution: "Use double type to store intermediate calculation results"
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICWB7G"

### TP+DP+MOE_TP LazyInline Position Error
- failure_info: "standalone, lazyinline, FlashAttention"
- observed_at: "test_vllm_mf_server_tp_dp_moe_tp_001"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "Inference flow frontend doesn't go through semi-auto parallel, obtained parallel mode is standalone; after removing standalone judgment branch, lazyinline tag position is incorrect"
- solution: "Revert the operation of removing standalone judgment logic for lazyinline"
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICWE7J"

### Attention Linear A8W8 Static Quantization Error
- failure_info: "golden-stick, qwen3, quantization, resize kernelmod"
- observed_at: "test_ptq_layers_moe_a8w8_llama2_2p"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "Original test case flow deprecated, changed to monitor qwen3 mcore quantization flow; golden-stick qwen3 quantized weight format doesn't match huggingface format, mcore cannot load correctly; mindformers qwen3 needs quantization inference support but not yet adapted"
- solution: "Golden-stick add weight name conversion logic, do hf name conversion when storing and loading weights; mindformers qwen3 adapt quantization inference"
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICWG4S"

### Recursive Programming DryRun 16384 Cards Limit
- failure_info: "recursive, programming, pp, dryrun, 1024 limit"
- observed_at: "test_dryrun_with_parallel_train_check_001"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "Recursive programming is operator-level parallel strategy search; typically for 10K card cluster with pp=16, operator-level strategy is usually cards=1024; originally added limit that cards must be less than 1024 when developing this feature"
- solution: "Currently this feature has no external users and no 10K card demand; feature has no evolution plan, suggest temporarily not converting to requirement, can hang first"
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICWID7"

### mint.optim.AdamW 910B Precision With Sparse Data
- failure_info: "precision, mint.optim.AdamW, seed, sparse data"
- observed_at: "test_mint_optim_adamw_dtype_float32"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "AdamW 910B occasional precision issue; input data not standardized, specific seed generates high sparsity data, amplifying framework implementation differences; loss function depends on data distribution, different seeds cause different input distributions leading to different error distributions and gradient starting points"
- solution: "Standardize input data to reduce sparsity; use fixed gradient, only test optimizer without mixing Loss and forward network fluctuations"
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICWLST"

### ops.topk Non-Stable Sort Index Difference
- failure_info: "ops.topk, aicpu, tf, stable sort"
- observed_at: "test_p_topk_forward_fp16_8_8732_k_int8732"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "Topk fp16 precision limited; original calculation comparison method changed to more reasonable comparison method, causing calculation result difference; original benchmark uses tf calculation result, using non-stable sort may cause topk result selected positions in original data to be different, leading to different indices; aicpu operator calculation result is non-stable sort, different from benchmark tf stable sort result, but operator final result is correct"
- solution: "Modify test case to use high-precision sorted result for loss comparison"
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICWLWZ"

### DryRun Semi Compile Fused Optimizer Error
- failure_info: "Tensor, flatten_weights, adam, fused optimizer"
- observed_at: "test_dry_run"
- backend: "ascend"
- failure_type: "framework"
- root_cause: "Tensor storage optimization feature, fully reverted flatten_weights interface feature; at that time confirmed no subsequent inclusion of adam optimizer operator fusion related logic, increased test case graph compilation time"
- solution: "Restore self.use_fused_opt related logic"
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICWMNZ"

### Pipeline Parallel LazyInline Required
- failure_info: "ValueError, lazy_inline, pipeline, AddN"
- observed_at: "test_parallel_dynamic_shape_with_features_003"
- backend: "ascend"
- failure_type: "platform"
- root_cause: "Custom dynamic_shape net+pp+profiler reports ValueError: For 'AddN', input shape must be same; this test case doesn't enable lazy_inline, current feature development for pipeline parallel only supports @lazy_inline"
- solution: "Add @lazy_inline tag to network to enable lazy_inline"
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICWMZR"

### MlaPreprocess NZ Format Conversion Error
- failure_info: "MlaPreprocess, NZ format, MS_INTERNAL_ENABLE_NZ_OPS, ND"
- observed_at: "test_infer_mla_preprocess_bfloat16_n100_blocknum9_blocksize128_cache_mode3_random_forward"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "MindSpore uses environment variable MS_INTERNAL_ENABLE_NZ_OPS to convert input ND format to NZ; in mla_preprocess operator, when cache_mode is nz+quantization or nz, key_cache and krope_cache need to be converted to nz format; ms_kernel_internal when processing atb operator, if format is nz, will do nd->nz conversion"
- solution: "If original test case didn't enable export MS_INTERNAL_ENABLE_NZ_OPS='MlaPreprocess', need to add this environment variable; remove framework-side conversion of these two inputs, let ms_kernels_internal handle internally"
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICX407"

### gather_elements bfloat16 Memory Error
- failure_info: "gather_elements, bfloat16, memcpy, grad_pytorch_impl"
- observed_at: "test_f_gather_elements_bf16_3d_random"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "Test case issue; grad_pytorch_impl function doesn't convert bfloat16 backward output to fp32, while numpy array has issues with bf16 support, causing memory problem when indexing in _count_unequal_element; forward_pytorch_impl and grad_mindspore_impl both have extra handling for bf16"
- solution: "Add bf16 conversion; operator backward has occasional bf16 precision issue, but already has conclusion meeting standard 2"
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICX4N5"

### Pipeline Parallel + Shard Not Fully Split + Dataset data_parallel, Sequence Parallel + Optimizer Parallel Combined, Dryrun Thousand-Card Compilation
- failure_info: "dataset_strategy, data_parallel, pipeline_parallel, sequence_parallel, optimizer_parallel, dryrun, null_pointer, OOM"
- observed_at: "test_dryrun_with_parallel_train_check_009"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "Pipeline parallel + shard not fully split + dataset data_parallel, sequence parallel + optimizer parallel combined features, dryrun simulating thousand-card compilation scenario causes null pointer error. When dataset_strategy is set to data_parallel, GBS=DP*single DP GBS, causing GBS to be too large, thus requesting 256G host memory in CreateZerosTensor, memory too large causing allocation failure returning null pointer, then CreateZeroseOutput function catches null pointer exception."
- solution: "Modify null pointer check, add detailed description of error cause and modification suggestions."
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICX5ZV"

### SOMAS ResNet50 Training, mpirun + ranktable, raise error: call hccl
- failure_info: "ranktable, hcom, mpirun, hccl"
- observed_at: "test_ms_somas_006_polymorph_D910B"
- backend: "ascend"
- failure_type: "platform"
- root_cause: "Error when exporting ranktable during mpirun. hcom interface cannot be called in mpirun under ranktable scenario."
- solution: "ranktable env and mpirun mixed running was never supported, add documentation to limit mpirun usage."
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICX7BN"

### Functional Programming Data Sink Mode ResNet50, Precision Issue
- failure_info: "precision, data_sink, sink_mode, loop"
- observed_at: "test_ms_cell_functional_programming_datasink_004"
- backend: "ascend"
- failure_type: "framework"
- root_cause: "Precision issue. data_sink function has incorrect handling of sink_mode, should use loop sink but didn't."
- solution: "Modify handling of loop variable, only when jit is not added and graph_mode, then judge as loop sink, set loop=1."
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICXCCX"

### BatchNorm Semi-Auto Parallel, Parameter Format Mismatch
- failure_info: "BN, gamma, format, 5HD, AllGather, semi_auto_parallel"
- observed_at: "test_semi_auto_parallel_batchnorm_strategy_set_001"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "GE subgraph passed parameter format error. Previous graph BN gamma weight format is 5HD, passed to next graph's AllGather, error due to inconsistent input format."
- solution: "For 5HD with host shape being 1D, do not pass special format."
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICXK4O"

### Gate Test Error: test_silent_detect.py::test_silent_detect_strikeout
- failure_info: "TCPStore, silent_detect, null_pointer, destructor"
- observed_at: "test_silent_detect"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "test_silent_detect_strikeout case null pointer error at end stage. Silent detection depends on TCPStore. At training end stage, TCPStore destructs first, then silent detection call causes error."
- solution: "Stop silent detection thread before TCPStore destructor."
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICXNBM"

### test_optimizer_parallel_semi_auto_matmul Loss Difference Too Large Between Two Executions
- failure_info: "SEG, O2, O0, precision, optimizer_parallel"
- observed_at: "test_optimizer_parallel_semi_auto_matmul"
- backend: "ascend"
- failure_type: "framework"
- root_cause: "Precision issue. According to SEG conclusion, O2 cases are not maintained, switch to O0. 910A does not support O0, issue returned."
- solution: "According to SEG conclusion, O2 cases are not maintained, switch to O0. 910A does not support O0, issue returned."
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICY1Z1"

### test_semi_auto_parallel_avgpool3d_16p_split_d_and_h_overlap Precision Issue
- failure_info: "precision, SEG, O2, O0, avgpool3d, semi_auto_parallel"
- observed_at: "test_semi_auto_parallel_avgpool3d_16p_split_d_and_h_overlap_less_slice"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "Precision issue. According to SEG conclusion, O2 cases are not maintained, switch to O0. 910A does not support O0, issue returned."
- solution: "According to SEG conclusion, O2 cases are not maintained, switch to O0. 910A does not support O0, issue returned."
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICY20T"

### ops.layer_norm Ascend Backend Intermittent Precision Issue
- failure_info: "precision, ops.layer_norm, backward, seed"
- observed_at: "test_f_layer_norm_float16_2d_3x8_random"
- backend: "ascend"
- failure_type: "platform"
- root_cause: "LayerNorm backward precision issue under specific seed. Test precision function did not correctly reflect result precision."
- solution: "Use CANN provided precision comparison function for result comparison, precision has no issue."
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICY3KT"

### CPU Backend floormod Operator Fixed Seed Precision Error
- failure_info: "precision, floormod, shape, reducesum, CPU, float32"
- observed_at: "test_p_floormod_left_shape_12x124x2_shape_1x1x1_dtype_float32"
- backend: "cpu"
- failure_type: "platform"
- root_cause: "floormod operator CPU backward gradient result has error. Test case input shape is large (12, 124, 2, 8, 4, 3, 6), floormod backward has reducesum operation with accumulation count up to 1,714,176 times. TensorFlow backward operator reference precision error is large (compared to numpy and pytorch)."
- solution: "Modify test case script, use torch instead of tf for reference data, and adjust error value. Adjust error value according to accumulation count standard."
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICY3O0"

### nn.BCELoss fp32 Intermittent Precision Issue
- failure_info: "precision, nn.BCELoss, tf, torch"
- observed_at: "test_n_bceloss_input_32x16x8x2x12x8_6d"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "Operator precision does not match torch. Operator implementation matches tf, inconsistent with torch, precision comparison is meaningless."
- solution: "Existing conclusion, modify test case, compare with numpy reference."
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICYB05"

### nn.tanh Ascend Intermittent Precision Issue
- failure_info: "precision, nn.tanh, seed, np, random"
- observed_at: "test_n_tanh_float16_2d_4x9_random"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "tanh operator precision not met under certain seed. At seed=1757853406, using np.random.randn generated output_grad has value <-2 (probability of random to >2 or <-2 range is very small), then used for backward out.backward(output_grad), finally nn.tanh backward gradient value grad_out=grad*output_grad, grad_out used for comparing output error. Random output_grad generated by test case has a large value causing gradient error to be amplified. Actual tanh backward gradient real value error is 0.0004, meeting set threshold 0.001."
- solution: "Limit np.random.randn generated output_grad size, for this case range is [-2,2]."
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICYB1I"

### mint.nn.functional.hardswish Intermittent Precision Issue
- failure_info: "precision, mint.nn.functional.hardswish, nn.functional, seed, np, random"
- observed_at: "test_dynamic_shape_mint_n_f_hardswish_dyn_rank_2"
- backend: "ascend"
- failure_type: "framework"
- root_cause: "hardswish operator precision not met under certain seed. At seed=1757835548, using np.random.randn generated input has value equal to critical value 3, taking left limit 3- to calculate gradient value causing precision error. Using dual reference comparison result pass."
- solution: "Limit np.random.randn random numbers not to generate critical point values, for this case critical point value is 3."
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICYB1K"

### ops.dynamicrnn Ascend Backend Intermittent Precision Issue
- failure_info: "precision, ops.dynamicrnn, ms, fp16, torch"
- observed_at: "test_p_dynamicrnn_cell_clip_2_neg"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "ops.dynamicrnn Ascend backend intermittent precision issue. ms runs with fp16, torch runs with fp32, this operator has about 10000 accumulations and multiplications, causing error accumulation."
- solution: "Modify test case loss standard according to operator group standard."
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICYB36"

### ops.gelu Ascend Backend Intermittent Precision Issue
- failure_info: "precision, ops.gelu"
- observed_at: "test_dynamic_shape_p_gelu_input_float16"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "ops.gelu Ascend backend intermittent precision issue. fp16 mindspore output result can align with fp32 mindspore/pytorch output result. fp16 pytorch output result has large difference with fp32 mindspore/pytorch output result only at 2.7344e-2."
- solution: "Extreme scenario, consider modifying test case."
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICYB3H"

### batchnormgradgrad Ascend Backend Intermittent Precision Issue
- failure_info: "precision, ops.batchnorm, float32"
- observed_at: "test_p_batchnormgradgrad_training_nchw_big_shape_fp16_succ"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "batchnormgradgrad Ascend backend intermittent precision issue. Operator internal calculation involves multiple float32 accumulations, accumulation error causes precision not met under certain data conditions. mindspore/ops/kernel/ascend/aicpu/aicpu_ops/cpu_kernel/ms_kernel/batch_norm_grad_grad.cc dscale calculation code has multiple nested loops."
- solution: "Modify test case, relax error value according to operator precision evaluation standard, for accumulation count over 200k times, can relax to 0.008. Reference document: https://wiki.huawei.com/domains/21427/wiki/40193/WIKI202509168307538"
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICYB5L"

### Pipeline(stages=4) + HSDP(shard_size=2), Simulating 16384 Cluster Card 0 Compilation, Graph Cycle Error
- failure_info: "require_grad, False, pipeline, HSDP, dryrun"
- observed_at: "test_dryrun_with_parallel_train_forward_check_002"
- backend: "ascend"
- failure_type: "framework"
- root_cause: "require_grad=False weight passed to optimizer causes error. Parallel processing flow skipped require_grad=False weight. require_grad=False weight still performs differentiation and optimization."
- solution: "require_grad parameter documentation correction. Change require_grad attribute to is_trainable (pending usability review meeting)."
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICYCS3"

### Quantization A16W8 llama Network Input fp32, No module named 'mindformers.parallel_core.inference.tensor_parallel.gemm_layers'
- failure_info: "mindformers, gemm_layers, grouped_layers, quantization, A16W8"
- observed_at: "test_ptq_a16w8_fp32_llama2_2p"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "No module named 'mindformers.parallel_core.inference.tensor_parallel.gemm_layers'. mindformers gemm_layers file renamed to grouped_layers, golden hoop needs adaptation."
- solution: "Replace gemm_layers with grouped_layers."
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICYF6W"

### MF Service qwen2_5 yaml Configuration Error, No Error Reported
- failure_info: "vllm, ms, mcore, yaml, configuration"
- observed_at: "test_vllm_mf_yaml_err_001"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "MF service qwen2_5 yaml configuration error, no error reported. vllm-ms service has all switched to mcore model (including qwen2_5), but mcore model does not support yaml configuration, so error. Version 2.3 removed trace function, causing frontend compilation degradation."
- solution: "Adapt test case, do not use yaml to configure mf model, use other methods."
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICYKPF"

### DP + MOE_TP, 16-Card Scenario, Parallel Strategy data_parallel, raise error: send_and_get_request Response [500]
- failure_info: "quantization, ascend, MOE_TP, data_parallel, send_and_get_request"
- observed_at: "test_vllm_mf_server_dp_moe_tp_001"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "DP + MOE_TP, 16-card scenario, parallel strategy set to data_parallel, raise error: send_and_get_request Response [500]. Quantization weight not configured --quantization ascend."
- solution: "Add configuration --quantization ascend to test case service command."
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICYO8B"

### Gate Test Error: test_capture_graph.py::test_dynamic_shape_for_capture_graph
- failure_info: "aclgraph, core_dump, graph_capture, superkernelactor"
- observed_at: "test_capture_graph"
- backend: "ascend"
- failure_type: "platform"
- root_cause: "aclgraph gate test case intermittent core dump. aclgraph related resources not cleaned up completely. aclgraph managed graph_capture instance and superkernelactor managed instance destructor order incorrect."
- solution: "Modify destructor order of two instances. Completely clean up all resources."
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICYQVQ"

### Ascend910B Backend, MindSpore Export Single Operator Model, Inference Precision Not Met
- failure_info: "precision, NAN, export, inference"
- observed_at: "test_f_ceil_float32_4d_7x7x4x7_random"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "MindSpore export some single operator cases have precision issues. Test case generated reference is unreasonable, output reference contains NAN. When using fp16 for computation, under random input, occasionally cannot align with reference and ratio is about 1/10000, but precision verification is very strict, causing failure. gumel_softmax operator lite currently not supported."
- solution: "Modify random input to fixed input to avoid intermittent precision failure caused by fp16 precision insufficiency. Remove unsupported operator test cases. Fix unreasonable reference."
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICYWM8"

### vllm raise error: ImportError: libmindspore_extension.so: cannot open shared object file
- failure_info: "ci, vllm, mindspore, libmindspore_extension, ImportError"
- observed_at: "test_vllm_mf_online_mtp_001"
- backend: "ascend"
- failure_type: "platform"
- root_cause: "vllm-mindspore error cannot find libmindspore_extension.so. ci machine when building vllm-mindspore whl package, used mindspore package specified in jenkins/test/config/dependent_packages.yaml (currently specified 9/15 daily package). Test machine when executing vllm-mindspore, used latest mindspore daily package (9/22). vllm-mindspore custom operator needs to link mindspore dynamic library. Based on old mindspore package built vllm."
- solution: "Update dependent_packages.yaml to depend on new mindspore. Future version decoupling, vllm-mindspore only depends on mindspore released stable version, such problems can be completely eliminated."
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICZ5J5"

### custom_ops CustomOpBuilder Custom Operator, gelu/add with MS Built-in Interface Network, Interaction with Hook raise Error
- failure_info: "custom_op_api, ms_extension, header_file"
- observed_at: "test_custom_cpp_function_muiti_op_tensor_hook"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "Custom operator test case compilation error, cannot find ms_extension.h header file. Custom operator directory restructured, external header file replaced with custom_op_api.h, old version header file needs to be compatible with old version through renaming, but missed this file."
- solution: "Rename custom_op_api.h to ms_extension.h, keep two versions together with ms_extension/api.h and ms_extension/all.h."
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICZ6NN"

### mint.sort 910A Intermittent Precision Issue
- failure_info: "precision, mint.sort, mint, sort, Float32, Float16, truncation_error"
- observed_at: "test_dynamic_shape_mint_f_sort_dyn_shape_2"
- backend: "ascend"
- failure_type: "platform"
- root_cause: "mint.sort operator intermittent precision issue on 910A. When input is Float32 on 910A, it will be converted to FLOAT16 for sorting, then converted back to FLOAT32, causing truncation error, leading to intermittent precision issue."
- solution: "Modify test case, use fp16 to construct input data, then convert to fp32 to pass to operator for testing."
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICZDJ6"

### CPU Backend ops.logsumexp Operator Fixed Seed Precision Error
- failure_info: "precision, ops.logsumexp, logsumexp, CPU, float16"
- observed_at: "test_f_logsumexp_float16_4d_4x5x5x9_random"
- backend: "cpu"
- failure_type: "backend"
- root_cause: "ops.logsumexp operator has precision error under float16 input type. Operator calculation has precision error. Verified: float16 type as input calling mindspore logsumexp operator vs float32 type as input calling torch logsumexp operator precision difference is generally smaller than float16 type as input calling torch logsumexp operator vs float32 type as input calling torch logsumexp operator precision difference."
- solution: "Recommend adjusting test case loss to 0.003 to avoid this precision error."
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICZDTV"

### Online MF Scenario Using Deepseek Model Enable MTP, Explicitly Specify V0 Architecture Startup, raise error: Got an unexpected keyword argument 'q_seq_lens_cpu'
- failure_info: "q_seq_lens_cpu, mtp, deepseek, vllm"
- observed_at: "test_vllm_mf_online_mtp_001"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "Online MF scenario using deepseek model enable MTP, explicitly specify V0 architecture startup, raise error Got an unexpected keyword argument 'q_seq_lens_cpu'. When adding q_seq_lens_cpu parameter, MTP model was not adapted."
- solution: "MTP model adapt q_seq_lens_cpu parameter."
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICZJ54"

### sparsesoftmaxcrossentropywithlogitsv2 910A Intermittent Precision Issue
- failure_info: "precision, ops.sparsesoftmaxcrossentropywithlogitsv, seed, CANN"
- observed_at: "test_p_sparsesoftmaxcrossentropywithlogitsv2_input_dtype_float32_int32"
- backend: "ascend"
- failure_type: "platform"
- root_cause: "sparsesoftmaxcrossentropywithlogitsv2 910A intermittent precision issue (fixed seed=1758650828 can reproduce). For random seed test cases, should use CANN operator precision standard 2.0."
- solution: "Test case should use CANN operator precision standard 2.0. Single reference comparison method verification passed. Recommend gate maintenance standard change to 0.0002."
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICZKLX"

### CCAE Single Node, CCAE_SAVE_INTERVAL=30, rank0 Training Sleep Until Communication Timeout Exit
- failure_info: "timeout, ccae, event, hccl"
- observed_at: "test_hccl_status_save_01"
- backend: "ascend"
- failure_type: "platform"
- root_cause: "CCAE single node, CCAE_SAVE_INTERVAL=30, rank0 training sleep until communication timeout exit, Python error: Aborted. CCAE thread needs to query event to determine its execution status during saving communication operator execution status. Event query may throw exception due to interface call failure. This exception is not caught and handled in CCAE thread, causing child thread to core. Old version CANN has no issue, new version CANN cores, should be CANN package modified interface behavior, triggering this issue."
- solution: "CCAE thread actively catches exception thrown by event interface query failure, while printing warning: no longer record communication operator execution status."
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ID0ALD"

### ops.resizebicubic Has RuntimeError and Verification Issues
- failure_info: "RuntimeError, ops.resizebicubic, aicpu, cann"
- observed_at: "test_p_resizebicubic_half_pixel_centers_true_fp64"
- backend: "ascend"
- failure_type: "platform"
- root_cause: "resizebicubic Ascend some test cases failed. Removed custom aicpu operator, called CANN operator, some behavior inconsistent causing test case failure."
- solution: "Fix performance degradation frontend graph optimization pass. Through supporting boost infer function remove trace function causing performance degradation."
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ID0D6D"

### Communication Operator ReduceScatter Test, 8-Card Input Type Inconsistent, Card 3/5/6/7 Execution Timeout
- failure_info: "timeout, hccl, reducescatter, ring, hd"
- observed_at: "test_reducescatter_input_f16_f32"
- backend: "ascend"
- failure_type: "platform"
- root_cause: "Single node 8-card abnormal test case, 4 cards reducescatter error exit, another 4 cards stuck not exit. 910A hccl for reducescatter operator algorithm changed from ring to hd. When using ring algorithm, hccl uses connection timeout (using HCCL_CONNECT_TIMEOUT, default 120s). When using hd algorithm, hccl uses execution timeout (using HCCL_EXEC_TIMEOUT, default 1836s). In test framework, timeout is 800s; so before algorithm modification 120s < 800s test case passed; after algorithm modification 1836s > 800s test case failed."
- solution: "Test case modify execution timeout to 3min, within 800s. export HCCL_EXEC_TIMEOUT=180"
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ID0LSD"

### profiler Collecting Profiler Data, Non-root User Did Not Collect disk and osrt Data
- failure_info: "mindspore, host_sys, disk, osrt, profiler, permission"
- observed_at: "test_func_profiler_exper_cfg_set_level1_activity_npu_cpu_text_host_sys_disk test_func_profiler_exper_cfg_set_level1_activity_npu_cpu_text_host_sys_osrt"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "Collecting profiler data, non-root user did not collect disk and osrt data. MindSpore host_sys class information, if normal user needs to collect disk and osrt, needs to configure all commands password-free, this does not meet security requirements. Therefore, host_sys disk and osrt after decision, do not support non-root user collection."
- solution: "Through detecting current user permission, switch configuration, implement disk and osrt class information not supporting non-root user collection."
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ID12XD"

### MindSpore Lite Test Case LoadMindIR Failed
- failure_info: "mindspore, buffer, size, MindIR, Lite"
- observed_at: "N/A"
- backend: "ascend"
- failure_type: "framework"
- root_cause: "MindSpore Lite loading MindIR model failed. When loading model, due to MindSpore internal interface modification, loaded model buffer is empty, causing get size failure."
- solution: "For data pointer empty scenario, through tensor directly get size, not through data bytesize method."
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ID1OTX"

### Gate Test Error: test_deepseekv3_pretrain.py::test_deepseekv3_cell_dp2mp2ep2pp2mb4gas1bs1_8p_1b1f_performance
- failure_info: "EOFERROR, deepseekv3, pretrain"
- observed_at: "test_deepseekv3_pretrain"
- backend: "ascend"
- failure_type: "platform"
- root_cause: "Gate test error test_deepseekv3_pretrain.py::test_deepseekv3_cell_dp2mp2ep2pp2mb4gas1bs1_8p_1b1f_performance etc 4 cases. After process ends has intermittent EOFERROR, causing process exit."
- solution: "CANN issue, CANN colleague has fixed (mainline 10/13, C23 branch 10/24), waiting for gate update."
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ID40EB"

### Gate Test Error: test_deepseekv3_pretrain.py Precision Issue Due to MF commit_id Change
- failure_info: "mf, commit_id, precision, deepseekv3"
- observed_at: "test_deepseekv3_pretrain"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "Gate test error test_deepseekv3_pretrain.py::test_deepseekv3_cell_dp2mp2ep2pp2mb4gas1bs1_8p. mf recently modified commit_id, causing precision change."
- solution: "Modify baseline, and give a reasonable precision fluctuation range 0.001."
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ID4RGP"

### Distributed Verification SSL Certificate, Data Parallel Execution Error: RuntimeError: Fetch parameter async host to device failed
- failure_info: "RuntimeError, SSL, distributed, Fetch parameter"
- observed_at: "N/A"
- backend: "ascend"
- failure_type: "platform"
- root_cause: "AllocMemAndCopyForParameter call error. Actually underlying CANN interface error. Environment issue."
- solution: "ci environment execution multiple times not reproduced, local stress test 200 times not reproduced."
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ID8NCT"

### Pipeline GeneratorDataset (Multi-thread) map Multi-process (spawn) dvpp + get_dataset_size + get_batch_size, Stuck Not Exit
- failure_info: "SIGSEGVHandler, OStreamWrite, GeneratorDataset, spawn, dvpp"
- observed_at: "test_func_spawn_with_generatordataset"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "Pipeline GeneratorDataset (multi-thread) map multi-process (spawn) dvpp + get_dataset_size + get_batch_size, stuck not exit. Child process is stuck in SIGSEGVHandler signal log module. Need to print log through OStreamWrite method."
- solution: "Child process is stuck in SIGSEGVHandler signal log module. Need to print log through OStreamWrite method."
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=IDB133"

### llama2 Quantization layer_policies Value PTQConfig, Error: aclnnApplyRotaryPosEmbGetWorkspaceSize call failed
- failure_info: "llama2, ApplyRotaryPosEmb, aclnn, quantization, PTQConfig"
- observed_at: "N/A"
- backend: "ascend"
- failure_type: "platform"
- root_cause: "llama2 quantization error: aclnnApplyRotaryPosEmbGetWorkspaceSize call failed, please check! llama2 floating point model, dynamic graph inference: ApplyRotaryPosEmb operator selected aclnn operator, but currently MindSpore only supports self-developed rope operator, not truly connected to aclnn rope operator."
- solution: "Before model inference, need to set ms.set_context(jit_config={'jit_level': 'O0', 'infer_boost': 'on'}), ensure dynamic graph can select to self-developed internal operator. Test engineer needs to modify test case construction."
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=IDBXVF"

### vllm_mindspore DP + MOE_TP + MOE_EP, 8-Card, Parallel Strategy data_parallel: 8, Error Call aclrtSetDevice repeated
- failure_info: "507033, set_context, infer_boost, on, aclrtSetDevice, EP, DP"
- observed_at: "N/A"
- backend: "ascend"
- failure_type: "platform"
- root_cause: "EP/DP parallel vllm-mindspore inference, error set_device repeated. set_context(infer_boost='on') logic modification, when setting internal_kernel_list, which judges whether 310p platform function, will call aclinit, default will set card 0, need set_device before set_context(infer_boost='on')."
- solution: "Short-term solution: modify calling set_internal_kernel_list function to IsEnableInferBoost function when infer_boost is True (same as original call). Final solution: when actually calling internal_kernel related, such as loading internal_kernel.so."
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=IDC64M"

### Bert Network Randomly Add ms_function, 8-Card Training, Error: DropoutDoMask-op0 op dtype is not same
- failure_info: "GE, cast, DropoutDoMask, dtype, ms_function"
- observed_at: "test_ms_jit_network_002_mindir_infer"
- backend: "ascend"
- failure_type: "platform"
- root_cause: "Bert network randomly add ms_function, 8-card training, error DropoutDoMask-op0 op dtype is not same, type1:DT_FLOAT, type2:DT_FLOAT16. GE mistakenly optimized cast operator away."
- solution: "GE fix. Reference DTS2025120121138."
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=IDC6NE"

### maskrcnn_mobilenetv1 Network Training, CPU Environment, Model Mixed Precision Configuration O1, Loss Explodes
- failure_info: "precision, RPN, self, loss, mixed_precision, O1, CPU"
- observed_at: "test_ms_mixed_precision_o1_maskrcnn_mobilenetv1_cpu"
- backend: "cpu"
- failure_type: "platform"
- root_cause: "maskrcnn_mobilenetv1 network training, CPU environment, Model mixed precision configuration O1, loss explodes. Operating on same RPN object, self.loss etc variables are refreshed by Inplace causing impact. Actual expectation is modifying loss does not affect self.loss."
- solution: "Test case adaptation: loss = loss + loss_total; loss_print += (loss_total, loss_cls_item, loss_reg_item); clsloss = clsloss + loss_cls_item; regloss = regloss + loss_reg_item."
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=IDC6VE"

### Python3.12 x86 Machine, tf Upgrade to 2.20, protobuf Also Upgraded, MS protobuf Too Old, Incompatible, Causing coredump
- failure_info: "mindspore, protobuf, tensorflow2, python3.12, coredump"
- observed_at: "N/A"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "Python3.12 version MindSpore and TensorFlow 2.20 have compatibility issue causing coredump. MindSpore protobuf version and TensorFlow 2.20 protobuf version incompatible."
- solution: "2025/12/18 SEG conclusion, MindSpore does not promise to support running with TensorFlow. Reached temporary solution with test team, all environments use TensorFlow 2.18 version. In problematic x86 environment, do not maintain Python3.12 version MindSpore test cases, only run Python3.10. Python3.12 version in aarch64 environment using TensorFlow 2.18 did not have issue, normal maintenance."
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=IDD29D"

### Optimizer Parallel + Data Parallel, 8p, Auto Mixed Precision, Error: Sync data from device failed
- failure_info: "precision, CANN, DTS2512040063545, optimizer_parallel, mixed_precision"
- observed_at: "N/A"
- backend: "ascend"
- failure_type: "platform"
- root_cause: "Memory copy access page table failed, causing copy error. CANN package upgrade caused issue, DTS2512040063545."
- solution: "Based on CANN 8.5.0.B100 local verification, test case pass."
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=IDDKSJ"

### Parallel layernorm Network, Training Result Compared with Single Card, Error Very Large
- failure_info: "layernormgrad, PTA, precision, parallel"
- observed_at: "N/A"
- backend: "ascend"
- failure_type: "platform"
- root_cause: "Parallel layernorm network, training result compared with single card, error very large. layernormgrad operator precision incorrect, test provided environment and operator group provided environment results inconsistent, operator group provided environment precision correct consistent with PTA, test provided environment precision incorrect."
- solution: "Update MindSpore package and CANN package to re-verify, use CANN 8.5.0.B089 or above version to verify."
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=IDDLAT"

### Semi-Auto Parallel, Supported Shard Configuration, Supported Layout, in_strategy Configured Operator (layernorm), self_define_shard=True, Calculation Result is nan
- failure_info: "layernormgrad, nan, semi_auto_parallel, self_define_shard"
- observed_at: "N/A"
- backend: "ascend"
- failure_type: "backend"
- root_cause: "Semi-auto parallel, supported shard configuration, supported layout, in_strategy configured operator (layernorm), self_define_shard=True, calculation result is nan. layernormgrad operator precision incorrect."
- solution: "Same issue as IDDLAT, this issue closed, transfer to IDDLAT to track."
- last_seen: "2025-12-01"
- occurrences: 1
- issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=IDE5GI"

## Adding New Failures

When analyzing a new failure:

1. Extract key information: error code, keywords, context, backend
2. Identify failure type (platform/scripts/framework/backend)
3. Determine root cause through orientation analysis
4. Document solution (verified or proposed)
5. Update this file with YAML format above
6. Include approximate timestamp and occurrence count
7. Add issue_url if available (issue link, or "N/A" for seed entries)

### failure_info Guidelines

**Language: English ONLY. No Chinese characters allowed anywhere in this file.**

**Include:** error codes, error keywords, operators, operation patterns, exception types
**Exclude:** test/function names, file names, variable names, specific parameter values, Chinese text

**Examples:**
- OK: "507010, lost heartbeat, RuntimeError, device hang"
- BAD: "test_matmul_npu_fp16_case3 failed"
- OK: "HCCL timeout, EI0002, distributed"
- BAD: "rank_table_file=/path/to/config.json"

**Purpose:** Enable semantic pattern matching; use `observed_at` for specific instances.

### issue_url Guidelines

**Format:** `https://e.gitee.com/mind_spore/dashboard?issue=ISSUE_ID`

**Rules:**
- For Observed Failures: always include the Gitee issue URL
- For Common Failure Patterns (seed entries): use "N/A"
- Multiple issues: comma-separated URLs

**Examples:**
- Single issue: `issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICWM5W"`
- Multiple issues: `issue_url: "https://e.gitee.com/mind_spore/dashboard?issue=ICWM5W, https://e.gitee.com/mind_spore/dashboard?issue=ICWG49"`
- Seed entry: `issue_url: "N/A"`

**Purpose:** Enable traceability to original issue for full context and discussion.
