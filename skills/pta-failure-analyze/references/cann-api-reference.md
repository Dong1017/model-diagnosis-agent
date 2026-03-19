# CANN API Reference & op-plugin Adaptation Guide

## Table of Contents

- [1. ACLNN Two-Phase Interface Pattern](#aclnn-two-phase-interface-pattern)
- [2. ACLNN Error Handling](#aclnn-error-handling)
- [3. op-plugin Adaptation Flow](#op-plugin-adaptation-flow)
- [4. Common op-plugin Adaptation Issues](#common-op-plugin-adaptation-issues)
- [5. Operator Compatibility and Fallback](#operator-compatibility-and-fallback)
- [6. Diagnostic Checklist](#diagnostic-checklist)

## ACLNN Two-Phase Interface Pattern

All aclnn operators follow a two-phase execution pattern:

**Phase 1: GetWorkspaceSize** — Computes required workspace memory and creates an executor.
```c
aclnnStatus aclnnXxxGetWorkspaceSize(/* inputs, params, outputs */, uint64_t *workspaceSize, aclOpExecutor **executor);
```

**Phase 2: Execute** — Runs the operator on the specified stream.
```c
aclnnStatus aclnnXxx(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
```

In op-plugin, this two-phase pattern is encapsulated by the `EXEC_NPU_CMD` macro:
```cpp
EXEC_NPU_CMD(aclnnAdd, self, other, alpha, result);
// Internally:
// 1. Resolves aclnnAddGetWorkspaceSize and aclnnAdd via GetOpApiFuncAddr
// 2. Calls GetWorkspaceSize → allocates workspace
// 3. Calls Execute on current NPU stream
// 4. NPU_CHECK_ERROR on return code
```

### API Categories

| Source | Category | Example APIs |
|--------|----------|-------------|
| ops-math | Basic math | aclnnAbs, aclnnAdd, aclnnSub, aclnnMul, aclnnDiv, aclnnMatmul |
| ops-nn | Neural network | aclnnSoftmax, aclnnBatchNorm, aclnnLayerNorm, aclnnDropout |
| ops-transformer | Attention/MoE | aclnnFlashAttentionScore, aclnnMoeDistributeDispatch, aclnnFFNV2 |
| ops-cv | Computer vision | aclnnRoiAlign, aclnnNms, aclnnDeformableConv2d |

API libraries are loaded at runtime from `libopapi.so`, `libopapi_math.so`, `libopapi_nn.so`, etc. via `GetOpApiFuncAddr`.

## ACLNN Error Handling

When an aclnn API fails, use `aclGetRecentErrMsg()` to get detailed error information. In torch_npu, this is done automatically by `NPU_CHECK_ERROR` and `C10_NPU_SHOW_ERR_MSG()`.

### Common ACLNN Failure Patterns

| Symptom | Error Code | Cause | Solution |
|---------|-----------|-------|----------|
| nullptr parameter | 161001 | Uninitialized tensor passed | Check all tensor allocations |
| dtype mismatch | 161002 | Input types don't satisfy promotion rules | Cast tensors to compatible types |
| Runtime exception | 361001 | NPU runtime call failed internally | Check CANN logs for details |
| Shape inference error | 561001 | Output shape cannot be computed | Check input shapes are valid |
| Tiling error | 561002 | Kernel tiling computation failed | Check tensor dimensions, may exceed hardware limits |
| Kernel not found | 561003 | Operator binary package not installed | Install correct CANN operator packages |
| OPP path missing | 561107 | ASCEND_OPP_PATH not set | `source /usr/local/Ascend/ascend-toolkit/set_env.sh` |
| Kernel package missing | 561112 | Operator binary kernel not loaded | Check CANN installation completeness |

## op-plugin Adaptation Flow

### Two Implementation Paths

op-plugin provides two implementation styles, with OpAPI (aclnn) being the preferred modern path:

```
PyTorch operator call
    ↓
torch_npu dispatch (npu_native_functions.yaml)
    ↓
┌─────────────────────────────────────┐
│ op-plugin implementation            │
│                                     │
│  Path A: OpAPI (aclnn)              │
│  ├── EXEC_NPU_CMD(aclnnXxx, ...)    │
│  └── Two-phase execution            │
│                                     │
│  Path B: ACL OpCommand (legacy)     │
│  ├── OpCommand().Name("Xxx")...     │
│  └── .Input().Output().Run()        │
└─────────────────────────────────────┘
    ↓
CANN kernel execution on NPU
```

### Path A: OpAPI (aclnn) — Preferred

**Filename pattern:** `XxxKernelNpuOpApi.cpp`
**Namespace:** `op_api`

Implementation steps:
1. **Prepare output tensor**: `npu_preparation::apply_tensor_without_format(...)` or `apply_tensor_with_format(...)`
2. **Handle scalars**: Convert CPU scalars to device if needed via `self_tensor_to_device()`
3. **Call aclnn**: `EXEC_NPU_CMD(aclnnXxx, input1, input2, ..., output)`
4. **Return result**

### Path B: ACL OpCommand — Legacy

**Filename pattern:** `XxxKernelNpu.cpp`
**Namespace:** `acl_op`

Implementation steps:
1. **Prepare output tensor**
2. **Build command**: `OpCommand cmd; cmd.Name("Xxx").Input(self).Output(result).Run()`
3. **Return result**

### Configuration Registry

**`op_plugin_functions.yaml`** — Central operator registry (search: `find <op_plugin_repo> -name "op_plugin_functions.yaml"`):
```yaml
- func: add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
  acl_op: all_version        # Supported in ACL OpCommand path
  op_api: all_version        # Supported in aclnn OpAPI path
  gen_opapi:                  # Auto-generation config
    out:
      size: self              # Output shape same as self
      dtype: self             # Output dtype same as self
    exec: aclnnAdd            # ACLNN API to call
```

**`derivatives.yaml`** — Backward/autograd rules (search: `find <op_plugin_repo> -name "derivatives.yaml"`):
```yaml
- name: npu_add_layer_norm(Tensor x1, ...) -> (Tensor, Tensor, Tensor, Tensor)
  output_differentiability: [true, false, false, true]
  x1, x2, gamma, beta: npu_add_layer_norm_backward(grads[0], ...)
```

### Operator Implementation Layers

Each op-plugin operator involves multiple potential failure points:

```
op_plugin_functions.yaml  (registration + version support)
    ↓
npu_native_functions.yaml (dispatch registration in torch_npu)
    ↓
XxxKernelNpuOpApi.cpp     (OpAPI kernel implementation)
  ├── Output preparation   (shape inference, memory allocation)
  ├── Input preprocessing  (scalar handling, dtype promotion, format)
  ├── EXEC_NPU_CMD         (aclnn two-phase execution)
  └── NPU_CHECK_ERROR      (error checking)
    ↓
derivatives.yaml          (backward graph)
```

## Common op-plugin Adaptation Issues

### Parameter Handling Issues

| Error Pattern | Cause | Fix |
|--------------|-------|-----|
| Scalar on wrong device | CPU scalar passed where NPU tensor expected | Use `self_tensor_to_device()` or extract via `.item()` |
| dtype mismatch after promotion | `at::native::result_type` differs from aclnn expectation | Add explicit `npu_dtype_cast()` |
| Format mismatch | Tensor in wrong format (ND vs NZ vs NCHW) | Check `FormatHelper::IsOpInputBaseFormat`, use `apply_tensor_with_format` |
| Bool tensor handling | Bool not directly supported by some aclnn ops | Cast to `uint8`/`byte` first |

### Output Shape Issues

| Error Pattern | Cause | Fix |
|--------------|-------|-----|
| Shape inference error | `KernelNpuOutputSize.h` helper returns wrong shape | Check `broadcast_ops_npu_output_size`, `reduce_ops_npu_output_size` |
| Empty tensor crash | Zero-sized dimension not handled | Add size-0 check before calling aclnn |
| Broadcast failure | Input shapes not broadcastable | Validate shapes before operation |

### Backward / Autograd Issues

| Error Pattern | Cause | Fix |
|--------------|-------|-----|
| Gradient shape mismatch | Backward produces wrong shape | Check `derivatives.yaml` input/output spec |
| Non-differentiable input | Missing zero gradient for non-differentiable inputs | Check `output_differentiability` in derivatives.yaml |
| Custom backward crash | NPU custom backward function error | Compare with PyTorch native backward behavior |

### Compatibility Fallback Issues

The `DO_COMPATIBILITY` macro falls back from aclnn to aclops:
```cpp
DO_COMPATIBILITY(aclnnXxx, acl_op::xxx(self, other, alpha));
```

If aclnn kernel is not available (older CANN version), it silently falls back to aclops path, which may have different behavior or precision.

## Operator Compatibility and Fallback

### Checking Operator Availability

```python
import torch_npu

# Check if specific aclnn kernel is available at runtime
# In C++: check_aclnn_kernel_available("aclnnXxx")
```

### CPU Fallback

When an NPU operator is unavailable or buggy, CPU fallback can be used:
```python
# Move tensor to CPU, compute, move back
cpu_result = op(tensor.cpu())
npu_result = cpu_result.to('npu')
```

### Version-Specific Support

`op_plugin_functions.yaml` specifies version support per operator:
```yaml
- func: some_op(...)
  acl_op: [v2.1, v2.2, v2.3]    # Only these versions
  op_api: all_version             # All versions
```

If an operator is missing for the current PyTorch version, it won't be registered, causing "not implemented" errors.

## Diagnostic Checklist

When encountering an operator-related error in torch_npu:

### Quick Triage

1. **Is it a torch_npu ERR code?** Check [Error Codes](error-codes.md) for ERRxxxxx
2. **Is it an ACLNN error code?** Check for 161xxx/361xxx/561xxx codes
3. **Is it a CANN runtime error?** Check for 107xxx/207xxx/507xxx codes
4. **Is it a precision/accuracy issue?** Compare with PyTorch CPU output
5. **Is it a missing operator?** Check `op_plugin_functions.yaml` for version support
6. **Is it a backward error?** Check `derivatives.yaml`
7. **Is it format-related?** Check tensor storage format (ND/NZ/NCHW)

### Collecting Evidence

```bash
# CANN version
cat /usr/local/Ascend/ascend-toolkit/version

# CANN device logs (operator-level errors)
tail -f /var/log/npu/slog/*/device-*/plog/*.log

# Recent error message
# C++: aclGetRecentErrMsg() (called automatically by NPU_CHECK_ERROR)
# Python: via torch_npu exception propagation

# Check operator binary packages
ls /usr/local/Ascend/ascend-toolkit/latest/opp/

# Search for operator in op-plugin source (when available)
grep -r "<OpName>" <op_plugin_repo> --include="*.cpp"

# Search for ACLNN binding
grep -r "aclnn<OpName>" <op_plugin_repo> --include="*KernelNpuOpApi.cpp"

# Check version support
find <op_plugin_repo> -name "op_plugin_functions.yaml" -exec grep -A 5 "<func_name>" {} +

# Check backward rule
find <op_plugin_repo> -name "derivatives.yaml" -exec grep -A 5 "<op_name>" {} +
```

## See Also

- [Error Codes](error-codes.md) — ACLNN error code table (161xxx/361xxx/561xxx) and full CANN code mappings
- [Backend Diagnosis](backend-diagnosis.md) — Step-by-step Ascend diagnosis and further location techniques
- [Torch_npu Operators](torch-npu-operators.md) — Operator registration, API details, debugging
- [PyTorch Operators](pytorch-operators.md) — PyTorch native operator specifications for comparison
- [Failure Showcase](failure-showcase.md) — Historical failures indexed by error keywords
