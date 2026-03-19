# Backend Diagnosis Reference

Detailed diagnosis steps for torch_npu on Ascend NPU. Read the section matching the diagnostic level identified by the Quick Route in SKILL.md.

## Table of Contents

- [Ascend Backend (CANN)](#ascend-backend-cann)
- [torch_npu Framework Diagnosis](#torch_npu-framework-diagnosis)
- [op-plugin Code Location](#op-plugin-code-location)
- [torch_npu Log Analysis](#torch_npu-log-analysis)
- [Further Location Techniques](#further-location-techniques)

## Ascend Backend (CANN)

- Parse CANN error codes using [Error Codes reference](error-codes.md)
- Check CANN logs: `/var/log/npu/slog/*/device-*/plog/`
- Check TBE/AKG compilation errors
- Cross-reference with [CANN API Reference](cann-api-reference.md) for aclnn API constraints

### CANN Error Code Diagnosis Flow

```
Error code → Category → Action
  1xxxxx   → Environment/Logic  → User-fixable: check params, init, file paths
  2xxxxx   → Resource            → OOM, feature unsupported: check memory, versions
  3xxxxx   → Business            → Queue/storage limits: check configs
  5xxxxx   → Internal            → Hardware/software fault: collect logs, contact support
  161xxx   → ACLNN Parameter     → Check tensor allocations, dtype, nullptr
  361xxx   → ACLNN Runtime       → Check CANN logs for internal detail
  561xxx   → ACLNN Internal      → Shape inference, tiling, kernel package issues
  E[x]xxxx → CANN Inner          → TBE/AICPU/HCCL/Runtime internal errors
```

### Common CANN Failure Points

| Symptom | Error Code | Cause | Solution |
|---------|-----------|-------|----------|
| Device OOM | 207018, EL0004, 200000 | HBM memory exhausted | Reduce batch size, `torch_npu.npu.empty_cache()`, gradient checkpointing |
| Device task abort | 107010, 507010 | Heartbeat lost / hardware fault | `npu-smi info`, device reset, hardware check |
| Context empty | 107002 | NPU context not initialized | Ensure `torch.npu` device is set before operations |
| Feature not supported | 207000, 200006 | CANN version incompatible | Upgrade CANN, check version compatibility |
| Kernel not found | 561003, 561112 | Operator package not installed | Install correct CANN operator packages |
| OPP path missing | 561107, 148049 | ASCEND_OPP_PATH not set | `source /usr/local/Ascend/ascend-toolkit/set_env.sh` |
| AI Core timeout | 507014, 507017 | Kernel execution hung | Check operator inputs, may exceed hardware limits |
| HBM ECC error | 507054 | Memory hardware fault | Hardware replacement required |
| HCCL timeout | EI0002, 107020 | Distributed communication stalled | Check network, HCCL config, process health |

### Precision / Numerical Accuracy Issues

When NPU operator produces different results from PyTorch CPU/CUDA:

1. **Compare operator signatures**: Check [PyTorch Operators reference](pytorch-operators.md) for native specs
2. **Check tolerance**: NPU may use different internal precision (fp16 accumulation, etc.)
3. **Test with `ASCEND_LAUNCH_BLOCKING=1`**: Ensures synchronous execution for accurate error location
4. **Dump intermediate tensors**: Compare NPU vs CPU for each operation step
5. **Check op-plugin implementation**: If source is available, search for the operator's `*KernelNpuOpApi.cpp` to see if there's a known precision difference

## torch_npu Framework Diagnosis

### Operator Registration Issues

When an operator is "not found" or "not supported":

1. Check if operator is registered (search in torch_npu source if available):
   ```bash
   find <torch_npu_repo> -name "npu_native_functions.yaml" -exec grep "<op_name>" {} +
   ```
2. Check if implementation exists (search in op-plugin source if available):
   ```bash
   grep -r "<OpName>" <op_plugin_repo> --include="*KernelNpuOpApi.cpp"
   grep -r "<OpName>" <op_plugin_repo> --include="*KernelNpu.cpp"
   ```
3. Check version support (search in op-plugin config if available):
   ```bash
   find <op_plugin_repo> -name "op_plugin_functions.yaml" -exec grep -A 5 "<op_name>" {} +
   ```

### Parameter Validation Errors

torch_npu uses `TORCH_CHECK` with structured error codes:

| Error Code | Meaning | Common Cause |
|-----------|---------|-------------|
| `ERR00001` / `ERR01001` | Invalid parameter | Wrong input shape, size, or value |
| `ERR00002` / `ERR01002` | Invalid type | Wrong tensor dtype, device type mismatch |
| `ERR00003` | Invalid value | Out-of-range configuration |
| `ERR00007` | Feature not supported | Operator/feature unavailable on NPU |
| `ERR00100` | ACL API call failed | Underlying CANN error, check CANN logs |
| `ERR00200` | HCCL API call failed | Distributed communication failure |
| `ERR02xxx` | Distributed errors | HCCL init, rank, world_size issues |
| `ERR03xxx` | Graph errors | Graph compilation, torchscript issues |

### Device Error Handling

torch_npu detects hardware errors via keywords in CANN error messages:

| Keyword | Meaning | Action |
|---------|---------|--------|
| `DEVICE_TASK_ABORT` | Device hardware failure | Check `npu-smi info`, may need reset |
| `DEVICE_MEM_ERROR` / `UCE ERROR` | Uncorrectable memory error | Hardware fault |
| `DEVICE_HBM_ECC_ERROR` | HBM ECC error | Hardware fault |
| `HCCS_LINK_ERROR` | Inter-chip link error | Check physical connections |
| `HCCL_OP_RETRY_FAILED` | HCCL operation retry exhausted | Network/topology issue |
| `SUSPECT_REMOTE_ERROR` | Remote device suspected error | Check all devices in cluster |

### Test Framework Issues

When torch_npu test cases fail:

| Pattern | Cause | Fix |
|---------|-------|-----|
| `DISABLED_TESTS_FILE` skip | Test known unsupported on NPU | Expected behavior, check skip list |
| Device check `'cuda'` vs `'npu'` | Test early return condition wrong | Change `self.device_type == 'cuda'` to `'npu'` |
| `SupportedDevices` skip | Device type not in allowed list | Check `Ascend910B`/`Ascend910A` match |
| `SkipIfNotGteCANNVersion` | CANN version too old | Upgrade CANN |
| `privateuse1` detection | torch_npu registers as `privateuse1` | Use `torch.npu.is_available()` |

## op-plugin Code Location

### Operator Implementation Patterns

op-plugin provides two implementation styles for NPU operators:

**Key file naming conventions (search by pattern, don't hardcode paths):**
- OpAPI implementations: `*KernelNpuOpApi.cpp`
- ACL implementations: `*KernelNpu.cpp`
- Operator registry: `op_plugin_functions.yaml`
- Backward rules: `derivatives.yaml`
- Two-phase execution macro: `op_api_common.h` (contains `EXEC_NPU_CMD`)
- Shape inference helpers: `KernelNpuOutputSize.h`

### OpAPI Pattern (Preferred, aclnn-based)

```cpp
// namespace: op_api
// Filename pattern: XxxKernelNpuOpApi.cpp

at::Tensor xxx_npu(const at::Tensor &self, ...) {
    // 1. Prepare output tensor
    at::Tensor result = npu_preparation::apply_tensor_without_format(self);
    // 2. Call aclnn via EXEC_NPU_CMD (two-phase internally)
    EXEC_NPU_CMD(aclnnXxx, self, ..., result);
    return result;
}
```

### ACL OpCommand Pattern (Legacy)

```cpp
// namespace: acl_op
// Filename pattern: XxxKernelNpu.cpp

at::Tensor xxx_npu(const at::Tensor &self, ...) {
    at_npu::native::OpCommand cmd;
    cmd.Name("Xxx")
       .Input(self)
       .Output(result)
       .Run();
    return result;
}
```

### Compatibility Fallback

`DO_COMPATIBILITY` macro falls back from aclnn to aclops when aclnn kernel is unavailable:
```cpp
DO_COMPATIBILITY(aclnnXxx, acl_op::xxx(self, other, alpha));
```

### Key Search Patterns

When source code is available, use these patterns to locate relevant files (replace `<repo>` with actual repo path):

| Looking for | Search command |
|------------|---------------|
| Operator implementation | `grep -r "<OpName>" <op_plugin_repo> --include="*.cpp"` |
| ACLNN API call | `grep -r "aclnn<OpName>" <op_plugin_repo> --include="*KernelNpuOpApi.cpp"` |
| Version support | `find <op_plugin_repo> -name "op_plugin_functions.yaml" -exec grep -A 5 "<func_name>" {} +` |
| Backward rule | `find <op_plugin_repo> -name "derivatives.yaml" -exec grep -A 5 "<op_name>" {} +` |
| Output shape logic | `grep -r "<OpName>" <op_plugin_repo> --include="KernelNpuOutputSize.h"` |
| Registration | `find <torch_npu_repo> -name "npu_native_functions.yaml" -exec grep "<op_name>" {} +` |

## torch_npu Log Analysis

### CANN Log Files

CANN device logs contain operator-level error details:
```bash
# Real-time monitoring
tail -f /var/log/npu/slog/*/device-*/plog/*.log

# Search for errors
grep -i "error\|fail\|exception\|abort" /var/log/npu/slog/*/device-*/plog/*.log | head -50

# Search for specific error codes
grep -i "aclnn\|acl_error\|ret=" /var/log/npu/slog/*/device-*/plog/*.log | tail -30

# Search for CANN inner errors
grep -i "EE\|EI\|EJ\|EZ\|EL" /var/log/npu/slog/*/device-*/plog/*.log | head -20
```

### torch_npu Python-level Logging

torch_npu patches `sys.excepthook` to handle NPU errors:
- Regex detection: `ERR\d{5}` pattern
- Timeout handling: on error code `107020`, sleeps before exit
- Compact output: controlled by `TORCH_NPU_COMPACT_ERROR_OUTPUT`

### OOM Snapshot

When NPU OOM occurs, torch_npu can capture memory snapshots:
```bash
export OOM_SNAPSHOT_ENABLE=1        # Enable OOM snapshots
export OOM_SNAPSHOT_PATH=./oom/     # Output directory
```
Generates `oom_snapshot_{pid}_{timestamp}.pickle` files for analysis.

### Asynchronous Execution Warning

NPU uses asynchronous execution by default. When errors occur:
```
WARNING: Since the operator is called asynchronously, the stacktrace may be inaccurate
```

To get accurate stack traces:
```bash
export ASCEND_LAUNCH_BLOCKING=1     # Force synchronous execution
```

## Further Location Techniques

When the root cause cannot be confirmed from available evidence, use these techniques to narrow down the issue.

### CANN Debug Logs

Enable CANN debug logs to stdout:
```bash
export ASCEND_GLOBAL_LOG_LEVEL=0       # 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR
export ASCEND_SLOG_PRINT_TO_STDOUT=1   # Print CANN logs to stdout
```
Then ask user to re-run and provide the debug log output.

**When user provides CANN debug logs:** Do NOT read the full log (can be extremely large). Instead, search for key patterns first:
```bash
grep -i "error\|fail\|exception\|abort" cann_debug.log | head -50
grep -i "aclnn\|acl_error\|ret=" cann_debug.log | tail -30
grep -i "EE\|EI\|EJ\|EH\|EP" cann_debug.log | head -20
```
Start from error/failure lines, then read surrounding context only as needed.

### Operator Debug Mode

Enable CANN operator debug output:
```bash
export ACL_OP_DEBUG_LEVEL=3            # 0-4, higher = more detail
export ACL_DEBUG_DIR=./acl_debug/      # Debug output directory
```

### HCCL Debug

For distributed training issues:
```bash
export HCCL_DESYNC_DEBUG=1             # Enable HCCL desync detection
export HCCL_CONNECT_TIMEOUT=300        # Increase timeout (default 120)
export HCCL_EXEC_TIMEOUT=3600          # Increase exec timeout (default 1800)
```

### Debug Patch for torch_npu / op-plugin

When code-level investigation is needed, provide a debug patch:

1. Generate the patch using `git diff` or `diff -u` format
2. Verify: `git apply --check debug.patch`
3. The patch should be minimal — only add logging/debug statements at suspected failure points
4. Include clear instructions: which repo/branch, how to apply, what output to look for

Debug patch output template:
```
Debug Patch (apply to [torch_npu/op-plugin] [branch]):

[git diff / unified diff content]

Apply: git apply debug.patch
Verify: git apply --check debug.patch
Expected output to look for: [description of key log lines]
```

### PTA Source Audit (For Alignment Issues)

When NPU operator behavior differs from PyTorch, check three key files:

| File | Filename Pattern | Extract |
|------|-----------------|---------|
| Function signature | `op_plugin_functions.yaml` | Parameter names, types, defaults |
| Backward registration | `derivatives.yaml` | Differentiable inputs, grad function |
| C++ implementation | `XxxKernelNpuOpApi.cpp` | Actual aclnn calls, parameter preprocessing |

Common torch_npu-vs-PyTorch discrepancies:
- Forward/backward parameter name differences
- Hidden hardcoded parameters in backward (e.g., `deterministic=true`)
- Optional None handling differences (empty tensor vs null)
- Output tensor count/shape differences
- Scalar-to-tensor conversion on device

## See Also

- [Error Codes](error-codes.md) — Error code lookup tables (torch_npu ERR + CANN + ACLNN)
- [CANN API Reference](cann-api-reference.md) — ACLNN API constraints, adaptation flow
- [Torch_npu Operators](torch-npu-operators.md) — Operator registration, implementation, debugging
- [PyTorch Operators](pytorch-operators.md) — PyTorch native operator specifications
- [Failure Showcase](failure-showcase.md) — Historical failures and solutions
