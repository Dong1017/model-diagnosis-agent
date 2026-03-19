---
name: pta-failure-analyze
description: "PyTorch Ascend (torch_npu) failure analyzer for NPU backends. Use whenever users report ANY torch_npu error, crash, or unexpected behavior — including error codes (ERRxxxxx, CANN 100xxx-500xxx, EL0004, E[x]9999), Python exceptions, NPU device issues (OOM, heartbeat, ECC), operator execution failures, distributed training errors (HCCL), numerical accuracy issues, op-plugin adaptation problems, or torch_npu test failures. Also use when users paste error logs, stack traces, or CANN debug output."
---

# PTA (PyTorch Ascend) Failure Analyzer

## Stage 0: Gather Context

Before diving into diagnosis, ensure you have the minimum context. If any of these are missing from the user's report, ask for them upfront — it saves multiple round-trips later.

| Info | Why It Matters | How to Get |
|------|---------------|------------|
| torch_npu version | API/behavior changes across versions | `python -c "import torch_npu; print(torch_npu.__version__)"` |
| PyTorch version | torch_npu must match specific PyTorch versions | `python -c "import torch; print(torch.__version__)"` |
| CANN version | Version mismatch is the #1 silent failure source | `cat /usr/local/Ascend/ascend-toolkit/version` |
| NPU device info | Hardware type affects supported features | `npu-smi info` |
| Complete error message | Truncated logs hide the real root cause | Ask for full traceback if only a snippet is provided |

If the user provides a clear error with sufficient context, skip directly to Stage 1.

## Stage 1: Find Similar Problem

### 1. Check Failure Showcase (check history FIRST)

Before analyzing the error in detail, search [Failure Showcase](references/failure-showcase.md) for matching patterns:
- Error keywords (OOM, timeout, ECC, link error, compile failed, etc.)
- Call stack patterns (operator names, functions)
- Device/module combinations (Conv1d+InstanceNorm, distributed+HCCL, etc.)
- Error code patterns (ERRxxxxx, 107010, EL0004, etc.)

If matching failure found:
1. Show historical failure info
2. Provide previously successful solution
3. Ask user: "Does this solution work for your issue?"
4. If yes → END Stage 1 (skip error code analysis entirely)
5. If no → continue to step 2 (Identify Error Pattern)

If no match found → continue to step 2.

### 2. Identify Error Pattern

Parse error message to determine error source. torch_npu errors come in several forms:

**Torch_npu Error Codes (`ERR<SubModule><ErrorCode>`):**
- `ERR000xx` — PTA (PyTorch Ascend) framework errors
- `ERR010xx` — OPS (operator) errors
- `ERR020xx` — DIST (distributed) errors
- `ERR030xx` — GRAPH (graph) errors
- `ERR040xx` — PROF (profiler) errors

**Python Exceptions (from torch_npu or PyTorch):**
- `RuntimeError` — device errors, operator execution failures, tensor mismatches
- `ValueError` — parameter validation, shape mismatches, invalid configurations
- `TypeError` — wrong data types, incompatible tensor types
- `NotImplementedError` — operator not supported on NPU backend

**CANN Error Codes (Ascend backend, numeric):**
- `1xxxxx` — environment/logic errors (user-fixable)
- `2xxxxx` — resource exhaustion (memory, streams, devices)
- `3xxxxx` — business exceptions (queue full/empty, storage limits)
- `5xxxxx` — internal software/hardware errors (need support)

**CANN Inner Error Codes (Ascend backend, alphanumeric):**
- `E[5-B]xxxx` — AICORE errors (TBE compilation, operator spec issues)
- `E3xxxx` — AICPU kernel execution failures
- `EExxxx` — Runtime errors (HBM OOM, task failure)
- `EI/EJxxxx` — HCCL/HCCP distributed errors
- `EKxxxx` — Profiling errors
- `EZxxxx` — AICORE execution failures (rtStreamSynchronize, model execution)

If error code found:
1. Extract error code/exception type and context
2. Check [Error Codes reference](references/error-codes.md) for known solutions
3. If direct match → provide solution, then ask user: "Did this resolve your issue?" (same verification loop as Stage 2 Step 3 — do not skip validation just because the match looks obvious)
4. If partial match → proceed to Stage 2

## Stage 2: Analyze Failure

**Failure Orientation Strategy:** Platform → Scripts → torch_npu Framework → CANN

### Quick Route (skip levels when evidence is clear)

Based on Stage 1 error pattern, jump directly to the most likely level:

| Stage 1 Finding | Start At | Rationale |
|-----------------|----------|-----------|
| CANN error code (1xxxxx-5xxxxx) | CANN | Error originates from CANN runtime |
| CANN inner code (E[x]xxxx) | CANN | TBE/AICPU/HCCL internal error |
| Hardware keywords (heartbeat, ECC, link error) | Platform | Hardware fault signals |
| ERR000xx (PTA framework errors) | torch_npu Framework | Core framework parameter/type errors |
| ERR010xx (operator errors) | torch_npu Framework | Operator execution failures |
| ERR020xx (distributed errors) | torch_npu Framework | HCCL communication errors |
| ERR030xx (graph errors) | torch_npu Framework | Graph compilation errors |
| Python exception only (ValueError, TypeError) | Scripts | Likely user code or API misuse |
| Precision mismatch / numerical accuracy | CANN | Operator implementation difference vs PyTorch |
| Operator not supported / not found | torch_npu Framework | Operator availability issue |
| `npu_native_functions.yaml` / op-plugin build error | torch_npu Framework | Operator registration or build issue |
| No clear signal | Platform (top-down) | Default: scan from top |

**Important distinction — Framework vs CANN for operator issues:**
- **Framework**: Operator does not exist, is not registered for NPU, or cannot be found → "Operator Not Supported" is a framework-level availability issue
- **CANN**: Operator exists but produces wrong results, has precision issues, or behaves differently from PyTorch (e.g., different gradient computation, incorrect strides, wrong backward dtype) → these are operator **implementation** issues classified as CANN, even when the fix involves modifying Python-level or C++ code in op-plugin

When jumping to a specific level, still collect basic evidence from upper levels (version info, device status) to rule out environmental factors.

### Step 1: Collect Evidence

For each orientation layer, collect:
- Platform: NPU device type, driver version, CANN version, firmware version
- Scripts: User code patterns, library calls, configurations, environment variables
- torch_npu Framework: Operator calls, parameters, registration status, debug logs
- CANN: Error codes from ACL/HCCL/GE API returns, CANN logs (if available)

### Step 2: Orient and Diagnose

Apply orientation strategy (start at the level identified by Quick Route, then expand if needed):

**Platform Level:**
- Check NPU device health with `npu-smi info`
- **Version compatibility check** (very common failure source):
  - torch_npu ↔ PyTorch: each torch_npu version requires a specific PyTorch version. Check:
    ```bash
    python -c "import torch; print(torch.__version__)"
    python -c "import torch_npu; print(torch_npu.__version__)"
    ```
    Cross-reference with torch_npu release notes for supported PyTorch versions.
  - torch_npu ↔ CANN: verify CANN version matches torch_npu requirements
  - Driver version: `npu-smi info` must meet minimum requirements
  - If version mismatch detected → this is likely the root cause; advise version alignment before further diagnosis
- Check for hardware errors:
  - 107010 device task abort / FORCE STOP / heartbeat lost
  - HBM_MULTIBIT_ECC_ERROR — HBM memory ECC error
  - DEVICE_MEM_ERROR / UCE ERROR — Uncorrectable memory error
  - LINK_ERROR — Network/hardware link issue
- If hardware issue → Provide hardware-specific fix → Validate with user

**Script Level:**
- Analyze user code for misuse:
  - Wrong device placement (tensor on CPU when NPU expected, or vice versa)
  - Shape/dtype mismatches in network definition
  - Cross-device tensor operations
- Check environment variables (`ASCEND_OPP_PATH`, `LD_LIBRARY_PATH`, CANN paths)
- Review script patterns (repeated initialization, improper resource cleanup)
- **torch_npu test framework cases:**
  - Tests may use `DISABLED_TESTS_FILE` to skip unsupported operators
  - Device type checks may incorrectly check for `'cuda'` instead of `'npu'`
  - When reproducing test failures, ensure correct CANN environment is sourced
- If script issue → Provide code fix → Validate with user

**torch_npu Framework Level:**
- Check operator registration and availability:
  - Search for operator in `npu_native_functions.yaml`: `grep -r "<op_name>" --include="*.yaml" <torch_npu_repo>`
  - Search for implementation in op-plugin: `grep -r "<OpName>" --include="*.cpp" <op_plugin_repo>`
- Verify parameter validation (types, shapes, constraints)
- Review torch_npu API usage (correct order, proper handles)
- Read [Torch_npu Operators reference](references/torch-npu-operators.md) for operator registration and implementation details
- **Source code search (when source is available):**
  - Registration YAML: search for `npu_native_functions.yaml` in torch_npu repo
  - OpAPI implementations: search for `*KernelNpuOpApi.cpp` in op-plugin repo
  - ACL implementations: search for `*KernelNpu.cpp` in op-plugin repo
  - Version support: search for `op_plugin_functions.yaml` in op-plugin repo
- If framework issue → Provide framework fix → Validate with user

**CANN Level:**

Read the matching section in [Backend Diagnosis Reference](references/backend-diagnosis.md) for detailed Ascend diagnosis steps.

- Parse CANN error codes (100xxx-500xxx series) using [Error Codes reference](references/error-codes.md)
- Check CANN logs: `/var/log/npu/slog/*/device-*/plog/`
- Cross-reference with [CANN API Reference](references/cann-api-reference.md) for aclnn API constraints and op-plugin adaptation flow
- Compare with PyTorch operator expectations via [PyTorch Operators reference](references/pytorch-operators.md):
  - Signature, precision, edge cases
  - NaN handling, in-place ops, autograd behavior
  - Per-sample gradients, tolerance differences
- **Source code search for CANN-level operator issues (when source is available):**
  - PyTorch native operator specs: search for `native_functions.yaml` in PyTorch repo
  - NPU operator implementations: search for `*KernelNpuOpApi.cpp` in op-plugin repo
  - Operator fallback: check if CPU fallback is available when NPU operator has issues
- If CANN issue → Provide CANN-specific fix → Validate with user

**Show Fix Advice:**

When root cause is **confirmed**:
```
Analysis: [Failure type identified]
Root Cause: [Specific cause]
Solution: [Actionable steps]
```

When root cause is **uncertain** — provide further location means instead of guessing:
```
Analysis: [Failure type identified]
Root Cause: Unable to confirm — further debugging required
Further Location:
  [Specific debug steps with env vars, log commands, or debug patches]
```

### Step 3: Validate and Iterate

After showing fix advice, ask user: "Did this resolve your issue?"

Handle response:
- **User confirms fixed** → Proceed to Stage 3
- **User says not fixed** → Loop back to Step 2 with new evidence
- **User provides debug output** → Analyze, refine, show updated advice, ask again
- **User asks to implement fix** → Execute the fix, then **re-ask** validation question (see below)
- **User changes topic** → Ask about original issue first, then handle new topic

**Why verification matters — do not skip this step:**

Writing an unverified fix into failure-showcase.md creates false knowledge that misleads all future diagnoses. A proposed fix may look correct but fail in the user's actual environment due to version differences, edge cases, or misidentified root causes. The cost of recording a wrong solution far exceeds the cost of one extra confirmation round-trip.

Therefore, after showing fix advice or executing a code change at the user's request, always re-ask the validation question: "Change applied. Please verify in your test environment — did this resolve the issue?" Only proceed to Stage 3 (Accumulate Experience) after the user explicitly confirms resolution (e.g., "resolved", "passed", "fixed", "works now"). Until then:
- Do not write or update failure-showcase.md
- Do not declare the diagnosis complete
- Do not move on to a different problem
- If the user ignores the question, ask again

## Stage 3: Accumulate Experience

### Step 1: Report Analysis Summary

Review analysis and extract key points:
- **Failure Info:** Error message, context, environment
- **Failure Type:** Platform/Scripts/Framework/CANN
- **Root Cause:** Specific issue identified
- **Solution:** Steps that resolved the issue

Report to user as structured summary.

### Step 2: Update Failure Showcase

First, search [Failure Showcase reference](references/failure-showcase.md) for matching existing entries.

**Matching criteria** — an existing entry "matches" if ANY of these hold:
- Same error code (e.g., both have `107010` or `ERR01002`)
- Same exception type AND similar root cause (e.g., both are `RuntimeError` from operator precision mismatch)
- Overlapping error keywords in `failure_info` (≥ 2 significant keywords match)

**If matching entry exists** — update it in place (do NOT create a duplicate):
- Enrich `failure_info` / `root_cause` / `solution` with any new details learned
- Refresh `last_seen` to current date
- Increment `occurrences` count
- **Fill `observed_at`** if it was empty — always record at least one concrete observation location

**If no matching entry exists** — create a new entry (all fields required, `observed_at` must NOT be left empty):
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

- [Error Codes](references/error-codes.md) — Complete error code mappings (torch_npu ERR + CANN + ACLNN + CANN Inner)
- [Failure Showcase](references/failure-showcase.md) — Historical failures and solutions
- [Torch_npu Operators](references/torch-npu-operators.md) — Operator registration, implementation, and debugging
- [PyTorch Operators](references/pytorch-operators.md) — PyTorch native operator specifications for comparison
- [Backend Diagnosis](references/backend-diagnosis.md) — Ascend diagnosis, torch_npu logs, op-plugin code location, further location techniques
- [CANN API Reference](references/cann-api-reference.md) — ACLNN two-phase interface, op-plugin adaptation flow

## Source Code Analysis

When diagnosis requires reading source code to confirm root cause or locate the fix point, ask the user to provide the relevant repository directory. Do not assume source code is available — always ask first.

**When to request source code:**
- Operator implementation issue → ask user to provide the op-plugin repository path to inspect the operator implementation
- Operator registration issue → ask user to provide the torch_npu repository path to check operator registration
- PyTorch behavior comparison → ask user to provide the PyTorch repository path to compare native implementation

**Repositories and their roles:**

| Repository | When to request | What to look for |
|-----------|----------------|-----------------|
| torch_npu | Framework errors, operator registration, device management | `npu_native_functions.yaml`, `NPUException.h`, `_error_code.py` |
| op-plugin | NPU operator implementations, CANN API adaptation, backward rules | `*KernelNpuOpApi.cpp`, `op_plugin_functions.yaml`, `derivatives.yaml` |
| PyTorch | Comparing native behavior, checking operator specs | `native_functions.yaml`, ATen native implementations |

**After user provides the path, use these search patterns:**

| Looking for | Search command |
|------------|---------------|
| Operator registration | `find <path> -name "npu_native_functions.yaml" -exec grep "<op_name>" {} +` |
| Operator implementation | `grep -r "<OpName>" <path> --include="*KernelNpuOpApi.cpp"` |
| ACLNN API binding | `grep -r "aclnn<OpName>" <path> --include="*.cpp"` |
| Version support | `find <path> -name "op_plugin_functions.yaml" -exec grep -A 5 "<func_name>" {} +` |
| Backward rule | `find <path> -name "derivatives.yaml" -exec grep -A 5 "<op_name>" {} +` |
| PyTorch native spec | `find <path> -name "native_functions.yaml" -exec grep -A 5 "<op_name>" {} +` |

## Diagnostic Commands

```bash
# NPU Device
npu-smi info                                     # NPU device status
npu-smi info -t usages                           # NPU utilization details
npu-smi info -t memory                           # NPU memory details

# Version Info
python -c "import torch; print(torch.__version__)"                # PyTorch version
python -c "import torch_npu; print(torch_npu.__version__)"        # torch_npu version
cat /usr/local/Ascend/ascend-toolkit/version                      # CANN version
python -c "import torch; print(torch.npu.is_available())"         # NPU availability

# CANN Logs
tail -f /var/log/npu/slog/*/device-*/plog/*.log  # CANN device logs (real-time)
ls /var/log/npu/slog/*/device-*/plog/             # List available CANN log files

# System
free -h                                           # Memory status
ulimit -a                                         # System limits
```

## Environment Variables

- `ASCEND_OPP_PATH` — Operator compiler path (Ascend)
- `ASCEND_GLOBAL_LOG_LEVEL` — CANN log level (0=debug, 1=info, 2=warning, 3=error, 4=null)
- `ASCEND_SLOG_PRINT_TO_STDOUT` — Print CANN slog to stdout (1=enable, 0=disable)
- `ASCEND_LAUNCH_BLOCKING` — Synchronous execution for debugging (1=enable) — makes stack traces accurate
- `TORCH_NPU_COMPACT_ERROR_OUTPUT` — Compact error output (1=enable)
- `TASK_QUEUE_ENABLE` — Task queue optimization (1=enable, 0=disable for debugging)
- `HCCL_CONNECT_TIMEOUT` — HCCL connection timeout in seconds (default 120)
- `HCCL_EXEC_TIMEOUT` — HCCL execution timeout in seconds (default 1800)
- `DISABLED_TESTS_FILE` — File listing tests to skip (for test framework)
- `PYTORCH_NPU_ALLOC_CONF` — NPU memory allocator configuration
