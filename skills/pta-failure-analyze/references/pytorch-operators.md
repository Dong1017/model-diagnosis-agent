# PyTorch Operators Reference

## Purpose

This reference provides PyTorch operator specifications for comparison with NPU implementations. Use when:
- Verifying operator behavior differences between CPU/CUDA and NPU
- Checking PyTorch's expected input/output specifications
- Understanding operator constraints and edge cases

## PyTorch Operator Sources

### 1. PyTorch Native Functions Definition

**File:** `torch/atives/native/native_functions.yaml` (upstream PyTorch)

Defines all PyTorch native operators with signatures, autograd functions, and types.

### 2. PyTorch ATen Operators

**Location:** `aten/src/ATen/native/`

Contains reference CPU implementations for comparison.

### 3. PyTorch CUDA Implementations

**Location:** `aten/src/ATen/native/cuda/`

Contains GPU reference implementations.

## Common Operator Specifications

### InstanceNorm

**Signature:**
```python
torch.nn.functional.instance_norm(
    input,
    running_mean=None,
    running_var=None,
    weight=None,
    bias=None,
    use_input_stats=True,
    momentum=0.1,
    eps=1e-5,
    training=False
)
```

**Constraints:**
- Input shape: (N, C, L) for 1d, (N, C, H, W) for 2d, (N, C, D, H, W) for 3d
- Normalizes across spatial dimensions per channel
- Affine parameters (weight, bias) applied if provided

### GroupNorm

**Signature:**
```python
torch.nn.functional.group_norm(
    input,
    num_groups,
    weight=None,
    bias=None,
    eps=1e-5
)
```

**Constraints:**
- Number of channels must be divisible by num_groups
- Normalizes within groups across spatial and batch dimensions

### LayerNorm

**Signature:**
```python
torch.nn.functional.layer_norm(
    input,
    normalized_shape,
    weight=None,
    bias=None,
    eps=1e-5
)
```

**Constraints:**
- normalized_shape: dimensions to normalize over (from the end)
- Normalizes per sample across specified dimensions

### Convolution

**Signature:**
```python
torch.nn.functional.conv1d/2d/3d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1
)
```

**Constraints:**
- Input/weight shapes must match convolution formula
- groups: number of blocked connections from input to output channels
- Dilation: spacing between kernel elements

### Linear

**Signature:**
```python
torch.nn.functional.linear(
    input,
    weight,
    bias=None
)
```

**Constraints:**
- Input: (N, in_features)
- Weight: (out_features, in_features)
- Bias: (out_features)

## Finding PyTorch Operator Specs

### 1. Check operator documentation

```python
import torch
help(torch.nn.functional.<operator_name>)
```

### 2. Check native functions (upstream PyTorch)

```bash
# Find in native_functions.yaml
grep "<operator_name>" torch/atives/native/native_functions.yaml
```

### 3. Check C++ implementation

```bash
# Find in ATen native (CPU reference)
grep -r "<operator_name>" aten/src/ATen/native/

# Find in ATen CUDA (GPU reference)
grep -r "<operator_name>" aten/src/ATen/native/cuda/
```

## Operator Comparison

### Expected Behavior vs Actual Behavior

When analyzing NPU operator issues:

1. **Check signature compatibility**
   - Input type and shape
   - Parameter types and constraints
   - Output shape and dtype

2. **Check numerical precision**
   - PyTorch uses FP32 by default
   - NPU may use FP16 internally
   - Compare tolerances (atol/rtol)

3. **Check edge cases**
   - Zero-size tensors
   - Empty tensors
   - Boundary values

### Common Differences

| Aspect | PyTorch CPU/CUDA | NPU |
|--------|-----------------|-----|
| Precision | FP32 default | May use FP16 internally |
| NaN handling | IEEE 754 | May differ in edge cases |
| In-place ops | Immediate | May be queued/async |
| Autograd graph | Explicit gradients | May use different grad computation |

## Numerical Accuracy Considerations

### Per-Sample Gradients (Expanded Weights)

When using expanded weights (per-sample gradients), note:

- PyTorch computes per-sample gradients by iterating
- NPU may batch compute with different accumulation
- Result may differ numerically but should be logically equivalent

NPU tolerance recommendation for affected operators:
```python
atol, rtol = 1e-2, 1e-3  # For tests on NPU
atol, rtol = 1e-4, 5e-5   # For tests on CPU/CUDA
```

## Operator Fallback

When NPU doesn't support an operator:

1. **Check if CPU fallback is acceptable**
   - Small tensors: OK
   - Large tensors: Performance penalty

2. **Check if alternative NPU op exists**
   - `npu_fmax` instead of `fmax`
   - Different parameter name

3. **Check if newer torch_npu/CANN version adds support**
   - Check release notes
   - Update if available

When an operator falls back:
```
CAUTION: The operator '<op_name>' is not currently supported on the NPU 
backend and will fall back to run on the CPU. This may have performance implications.
```

## Validation Checklist

When comparing PyTorch vs NPU operator:

- [ ] Input shapes match expected signatures
- [ ] Input dtypes match requirements
- [ ] Parameter values within valid ranges
- [ ] Output shapes match expected
- [ ] Output dtypes match expected
- [ ] Numerical values within acceptable tolerance
- [ ] Autograd gradients computed correctly
- [ ] Edge cases handled properly
