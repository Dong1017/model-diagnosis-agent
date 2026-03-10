# Torch_npu Operators Reference

## Overview

torch_npu registers operators through the C++ code generation system using YAML configuration files.

## Operator Registration

### Native Functions Configuration

**File:** `torch_npu/csrc/aten/npu_native_functions.yaml`

NPU backend operators are explicitly registered here:

```yaml
backend: NPU
cpp_namespace: at_npu::native

supported:
  - _local_scalar_dense
  - clone
    op_api: True
  - copy_
    op_api: True
  - copy_memory_
  - empty.memory_format
  ...
```

### Key Registration Attributes

- `op_api: True` - Indicates direct operator API usage (not autograd wrapper)
- `device_check: NoCheck` - Skip device verification
- `func:` alias - Maps Python function to C++ implementation

## Finding Operator Implementation

### Step 1: Check if operator is supported

Search `npu_native_functions.yaml` for the operator name.

If operator is listed: → Proceed to Step 2

If operator is NOT listed: → Should fall back to CPU or not supported

### Step 2: Find C++ implementation

NPU operators are typically in `torch_npu/csrc/aten/native/`

Common patterns:
- `CopyMemoryKernel.cpp` - Copy operations
- `NPUTensorNatives.cpp` - Basic tensor operations
- `SmallKernel.cpp` - Small utility kernels

### Step 3: Check op-plugin (if needed)

Some operators are provided by op-plugin for version-specific implementations.

**Config locations:** `op-plugin/op_plugin/config/v2r7/operator-api.yaml` (version-specific)

## Common Operator Categories

### Basic Tensor Operations
- `clone`, `copy_`, `copy_` - Memory copy/clone
- `empty`, `full`, `zeros`, `ones` - Tensor creation
- `resize_`, `reshape`, `squeeze` - Shape operations

### Mathematical Operations
- `add`, `sub`, `mul`, `div` - Arithmetic
- `matmul`, `mm`, `dot` - Matrix operations
- `sum`, `mean`, `max`, `min` - Reduction operations

### Neural Network Operations
- `conv1d`, `conv2d`, `conv3d` - Convolutions
- `batch_norm`, `instance_norm`, `layer_norm` - Normalization
- `linear`, `embedding` - Linear layers

### Activation Operations
- `relu`, `sigmoid`, `tanh` - Activations
- `softmax`, `log_softmax` - Softmax operations

## Operator Support Check

At runtime, torch_npu checks operator availability:

1. **Python level:** Check if `torch.<opfcn>.npu()` is available
2. **C++ level:** `NPU_CHECK_ERROR` macros wrap ACL API calls
3. **Fallback:** If not supported, may fallback to CPU with warning

Example unsupported operator warning:
```
CAUTION: The operator 'aten::fmax.out' is not currently supported on the NPU 
backend and will fall back to run on the CPU. This may have performance implications.
```

## Error Handling in Operators

### Parameter Validation

```cpp
AT_ASSERT(condition message, OPS_ERROR(ErrCode::PARAM));
```

Typical validation checks:
- Tensor is NPU tensor: `torch_npu::utils::is_npu(tensor)`
- Matching dtypes: `src.dtype() == self.dtype()`
- Matching devices: `device.index() == self.device().index()`
- Storage offset checks: `storage_offset() == 0`

### ACL Error Checking

```cpp
NPU_CHECK_ERROR(aclrtGetCurrentContext(&context));
```

Wraps CANN API calls and converts errors to torch_npu format.

## Code Generation

The `torchnpugen` package reads YAML configs and generates:

1. **Registration files:** `RegisterNPU.cpp`, `NPUNativeFunctions.h`
2. **Autograd bindings:** Forward/backward function wrappers
3. **Dispatch code:** Routing tensors to NPU backend

## Searching for Specific Operators

### Grep patterns

Find NPU operator C++ implementation:
```bash
# Search for function name in C++ source
grep -r "NPUNativeFunctions::<func>" torch_npu/csrc/aten/

# Search for operator in native files
grep -r "<func>" torch_npu/csrc/aten/native/
```

Find operator in YAML:
```bash
grep "<func>" torch_npu/csrc/aten/npu_native_functions.yaml
```

### Common Implementation Locations

| Operation Type | File Pattern | Example |
|---------------|-------------|---------|
| Copy | `CopyMemory*` | `CopyMemoryKernel.cpp` |
| Reduction | `*Reduce*.cpp` | `ReduceOps.cpp` |
| Activation | `*Activation*` | `ActivationOps.cpp` |
| Convolution | `*Conv*` | `Convolution.cpp` |
| Normalization | `*Norm*.cpp` | `Normalization.cpp` |

## Debugging Operator Issues

### 1. Check operator is supported
```python
import torch
import torch_npu

# Test if op exists
t = torch.rand(10).npu()
try:
    result = t.abs()  # Replace with your op
    print("Supported!")
except:
    print("Not supported or error occurred")
```

### 2. Check CANN logs for operator errors
```bash
# View latest CANN device logs
tail -f /var/log/npu/slog/*/device-*/plog/$(ls -t /var/log/npu/slog/*/device-*/plog/ | head -1)
```

### 3. Enable synchronous execution for debugging
```bash
export ASCEND_LAUNCH_BLOCKING=1
```
This makes errors appear at the correct call site.

### 4. Check explicit operator API vs autograd
Some operations use `op_api: True` for direct API calls (no autograd).
