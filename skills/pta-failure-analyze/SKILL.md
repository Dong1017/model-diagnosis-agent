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

Search failure showcase for matching patterns:
- Error keywords (OOM, timeout, ECC, link error, etc.)
- Call stack patterns (operator names, functions)
- Device/module combinations (Conv1d+InstanceNorm, distributed+HCCL, etc.)

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
- If script issue → Provide code fix → Validate with user

**torch_npu Framework Level:**
- Check operator registration and availability
- Verify parameter validation (types, shapes, constraints)
- Review torch_npu API usage (correct order, proper handles)
- Read [Torch_npu Operators reference](references/torch-npu-operators.md) for API details
- If framework issue → Provide framework fix → Validate with user

**CANN Level:**
- Parse CANN error codes (100xxx series) using [Error Codes reference](references/error-codes.md)
- Check CANN logs: `/var/log/npu/slog/*/device-*/plog/`
- Compare with PyTorch operator expectations via [PyTorch Operators reference](references/pytorch-operators.md)
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
- [Torch_npu Operators](references/torch-npu-operators.md) - Operator registration and API details
- [PyTorch Operators](references/pytorch-operators.md) - PyTorch operator specifications

## Diagnostic Commands

```bash
npu-smi info                                   # Device status
cat /usr/local/Ascend/ascend-toolkit/version   # CANN version
tail -f /var/log/npu/slog/*/device-*/plog/*.log # CANN logs
python -c "import torch_npu; print(torch_npu.__version__)"
```

## Environment Variables

- `ASCEND_OPP_PATH` - Operator compiler path
- `ASCEND_GLOBAL_LOG_LEVEL` - Log level (0-4)
- `TORCH_NPU_COMPACT_ERROR_OUTPUT=1` - Compact error output
- `DISABLED_TESTS_FILE` - Skip unsupported tests