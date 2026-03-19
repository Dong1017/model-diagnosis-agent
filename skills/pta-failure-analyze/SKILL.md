---
name: pta-failure-analyze
description: Analyzes torch_npu errors and provides fixing advice through a 3-stage workflow: Stage 1 finds similar problems via error codes and failure showcase; Stage 2 analyzes failures using platform→scripts→torch_npu framework→CANN orientation; Stage 3 accumulates analysis experience. Use when users encounter torch_npu runtime errors, NPU device failures, ACL/CANN error codes (ERRxxxxx, 100xxx, EL0004), memory failures, distributed training errors (HCCL), operator execution failures, or numerical accuracy issues.
---

# PTA (PyTorch Ascend) Failure Analyzer

## Stage 1: Find Similar Problem

### 1. Check Torch_npu Error Code

Parse error message for `ERR<SubModule><ErrorCode>` format:
- `ERR000xx` - PTA (PyTorch Ascend) framework errors
- `ERR010xx` - OPS (operator) errors
- `ERR020xx` - DIST (distributed) errors
- `ERR030xx` - GRAPH (graph) errors
- `ERR040xx` - PROF (profiler) errors

If error code found:
1. Extract SubModule and ErrorCode
2. Check [Error Codes reference](references/error-codes.md) for known solutions
3. If direct match → provide solution
4. If partial match → proceed to Stage 2

### 2. Check Failure Showcase

Search failure showcase for matching patterns using keyword categories:
- **Memory:** OOM, EL0004, 200000, 207018, OutOfMemoryError
- **Hardware:** 107010, FORCE STOP, ECC, link error
- **Distributed:** HCCL, timeout, broadcast, all_reduce, 107020
- **Framework:** 107002, context empty, dispatcher not registered, PrivateUse1, low-level API, default_generators
- **Test Framework:** device detection, device mismatch, index_fill, CUDA dependency, deprecation warning, CudaNonDefaultStream
- **Operator:** custom ops, expanded_weights, aclnnIm2col, int4pack
- **Environment:** libhccl.so, libascendcl.so, ASCEND_OPP_PATH
- **Accuracy:** per_sample_grad, batch_threshold, numerical bias

Search strategy:
- Error keywords (from categories above)
- Call stack patterns (operator names, functions)
- Device/module combinations (Conv1d+InstanceNorm, distributed+HCCL, etc.)

Reference: See [Searchable Keywords](references/failure-showcase.md#searchable-keywords) for complete category mapping.

If matching failure found:
1. Show historical failure info
2. Provide previously successful solution
3. Ask user: "Does this solution work for your issue?"
4. If yes → END Stage 1
5. If no → proceed to Stage 2

## Stage 2: Analyze Failure

**Failure Orientation Strategy:** Platform → Scripts → torch_npu Framework → CANN

### Step 1: Collect Evidence

For each orientation layer, collect:
- Platform: HW type, driver version, CANN version
- Scripts: User code patterns, library calls, configurations
- torch_npu Framework: Operator calls, parameters, debug logs
- CANN: Error codes from ACL/HCCL/GE API returns, CANN logs (if available)

### Step 2: Orient and Diagnose

Apply orientation strategy in order:

**Platform Level:**
- Check NPU device health with `npu-smi info`
- Verify hardware compatibility (device vs CANN version)
- Check for hardware errors (107010 task abort, ECC errors)
- If hardware issue → Provide hardware-specific fix → Validate with user

**Script Level:**
- Analyze user code for misuse (wrong device, wrong dtype, shape mismatches)
- Check environment variables (CANN paths, log levels)
- Review script patterns (repeated initialization, improper cleanup)
- **Test Framework Specific Checks:**
  - Device detection: Verify tests check for 'npu' not 'cuda' in device_type guards
  - CUDA dependencies: Check for imports of torch.testing._internal.common_cuda
  - Deprecation warnings: Verify CUDA-specific warnings (torch.cuda.amp.custom_fwd) aren't expected on NPU
  - RNG handling: Check for default_generators access in functionalized RNG ops (empty on CPU-only builds)
  - Low-level API usage: Detect torch._C._cuda_* calls that should be torch_npu._C._npu_*
- If script issue → Provide code fix → Validate with user

**torch_npu Framework Level:**
- Check operator registration and availability
- Verify parameter validation (types, shapes, constraints)
- Review torch_npu API usage (correct order, proper handles)
- **Dispatcher Registration Checks:**
  - For "PrivateUse1 not registered" errors: Check if operator has NPU dispatcher entry in npu_native_functions.yaml
  - Verify if operator exists in separate namespace (torch_npu.npu_* vs aten::_*)
  - Example: torch_npu.npu_convert_weight_to_int4pack vs aten::_convert_weight_to_int4pack
- **Low-Level API Gap Analysis:**
  - Identify missing APIs: torch._C._cuda_* vs torch_npu._C._npu_* equivalents
  - Common gaps: _cuda_setStream, _cuda_setDevice, _cuda_sleep, _cuda_attach_out_of_memory_observer
  - Check if CANN lacks equivalent kernel (e.g., device-side sleep like cudaSleep)
  - **Call Chain Comparison Method (CUDA vs NPU):**
    1. Trace the CUDA call chain: User code → torch.* API → torch._C._cuda_* internal API → CUDA runtime → Device kernel (source in PyTorch aten/src/ATen/native/cuda/)
    2. Trace the NPU call chain: User code → torch_npu.npu.* API → torch_npu._C._npu_* internal API → ACL runtime → CANN kernel (source in torch_npu/csrc/aten/)
    3. Compare at each layer: API availability, Parameter mapping, Kernel presence
    4. Use code search tools: Grep in aten/src/ATen/native/cuda/ for CUDA, Grep in csrc/aten/ for NPU
    5. Check for format/signness differences: Int4pack: CUDA uses unsigned [0,15], NPU uses signed [-8,7]
- Read [Torch_npu Operators reference](references/torch-npu-operators.md) for API details
- If framework issue → Provide framework fix → Validate with user

**CANN Level:**
- Parse CANN error codes (100xxx series) using [Error Codes reference](references/error-codes.md)
- Check CANN logs: `/var/log/npu/slog/*/device-*/plog/`
- Compare with PyTorch operator expectations via [PyTorch Operators reference](references/pytorch-operators.md)
- **Operator-Specific Behavioral Issues:**
  - aclnnIm2col optimization: Triggers numerical differences for per_sample_grad when batch_size >= 32
    - Root cause: PyTorch's call_for_per_sample_grads switches to unfold-based algorithm at THRESHOLD=32
    - NPU uses CANN's aclnnIm2col optimization which changes floating-point operation order
    - Solution: Use batch_size < 32 for testing, or increase tolerances (atol=1e-2, rtol=1e-3)
  - Format Mismatches: Check for signedness and layout differences vs CUDA
    - Example: int4pack has different signedness [-8,7] vs [0,15] and input layouts
    - Value range differences: CUDA uses unsigned [0,15]; NPU uses signed [-8,7]
  - Expanded Weights: Per-sample gradients may produce numerical differences in InstanceNorm, GroupNorm, LayerNorm
- If CANN issue → Provide CANN-specific fix → Validate with user

**Show Fix Advice:**
```
Analysis: [Failure type identified]
Root Cause: [Specific cause]
Solution: [Actionable steps]
```

### Step 3: Validate and Iterate

After providing fix:
1. Ask user: "Did this advice resolve your issue?"
   - Yes → Proceed to Stage 3
   - No → Collect additional evidence, re-orient at current or deeper level
2. Never end Stage 2 unless user accepts fix

## Stage 3: Accumulate Experience

### Step 1: Report Analysis Summary

Review analysis and extract key points:
- **Failure Info:** Error message, context, environment
- **Failure Type:** Platform/Scripts/Framework/CANN
- **Root Cause:** Specific issue identified
- **Solution:** Steps that resolved the issue

Report to user as structured summary.

### Step 2: Update Failure Showcase

Decide if this failure should be added to showcase:
- Is this a new failure pattern not captured before?
- Is the solution distinct from existing entries?

If yes, write to [Failure Showcase reference](references/failure-showcase.md):
```yaml
- failure_info: "[error keywords/context]"
  observed_at: "[file:function or test location where observed]"
  failure_type: "platform|scripts|framework|cann"
  root_cause: "[specific cause]"
  solution: "[actionable steps]"
  last_seen: "[timestamp]"
  occurrences: [count]
```

## Quick References

- [Error Codes](references/error-codes.md) - Complete error code mappings
- [Failure Showcase](references/failure-showcase.md) - Historical failures and solutions
  - Searchable Keywords - Keyword categories for pattern matching (Memory, Hardware, Distributed, Framework, Test, Operator, Environment, Accuracy)
- [Torch_npu Operators](references/torch-npu-operators.md) - Operator registration and API details
- [PyTorch Operators](references/pytorch-operators.md) - PyTorch operator specifications

## Diagnostic Commands

```bash
# Device and Platform
npu-smi info                                   # Device status
cat /usr/local/Ascend/ascend-toolkit/version   # CANN version
python -c "import torch_npu; print(torch_npu.__version__)"

# CANN Logs
tail -f /var/log/npu/slog/*/device-*/plog/*.log # CANN logs

# Test Framework Diagnostics
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('NPU available:', torch.npu.is_available())"
python -c "import torch; print('Default generators:', torch._C._get_default_generators())"
python -c "import torch; print(torch.ops.aten._convert_weight_to_int4pack)"
python -c "import torch_npu; print(hasattr(torch_npu, 'npu_convert_weight_to_int4pack'))"
```

## Environment Variables

- `ASCEND_OPP_PATH` - Operator compiler path
- `ASCEND_GLOBAL_LOG_LEVEL` - Log level (0-4)
- `TORCH_NPU_COMPACT_ERROR_OUTPUT=1` - Compact error output
- `DISABLED_TESTS_FILE` - Skip unsupported tests
- `ASCEND_LAUNCH_BLOCKING=1` - Synchronous execution for debugging

## Test Framework Detection Patterns

Common test framework issues to watch for:
- **Device type guards:** `if self.device_type == 'cuda'` should be `'npu'`
- **CUDA imports:** `torch.testing._internal.common_cuda` may contain CUDA-specific lazy evaluation
- **Low-level API gaps:** `torch._C._cuda_*` → `torch_npu._C._npu_*`
- **Deprecation warnings:** CUDA-specific warnings don't exist on NPU
- **RNG state access:** `torch.cuda.default_generators` is empty on CPU-only builds