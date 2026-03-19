# Error Codes Reference

## Torch_npu Error Codes Format: `ERR<SubModule><ErrorCode>`

### SubModule IDs
| Code | Name | Description |
|------|------|-------------|
| 00 | PTA | PyTorch Ascend - Core framework errors |
| 01 | OPS | Operations - Operator execution errors |
| 02 | DIST | Distributed - HCCL communication errors |
| 03 | GRAPH | Graph - Graph compilation errors |
| 04 | PROF | Profiler - Profiling errors |
| 99 | UNKNOWN | Unknown module |

### Error Codes
| Code | Name | Description | Example |
|------|------|-------------|---------|
| 001 | PARAM | Invalid parameter | ERR00001, ERR01001 |
| 002 | TYPE | Invalid type | ERR00002, ERR01002 |
| 003 | VALUE | Invalid value | ERR00003 |
| 004 | PTR | Invalid pointer (nullptr) | ERR00004 |
| 005 | INTERNAL | Internal error | ERR00005 |
| 006 | MEMORY | Memory error | ERR00006 |
| 007 | NOT_SUPPORT | Feature not supported | ERR00007 |
| 008 | NOT_FOUND | Resource not found | ERR00008 |
| 009 | UNAVAIL | Resource unavailable | ERR00009 |
| 010 | SYSCALL | System call failed | ERR00010 |
| 011 | TIMEOUT | Timeout error | ERR00011 |
| 012 | PERMISSION | Permission error | ERR00012 |
| 100 | ACL | Call ACL API failed | ERR00100 |
| 200 | HCCL | Call HCCL API failed | ERR00200 |
| 300 | GE | Call GE API failed | ERR00300 |
| 999 | EXCEPT | Unhandled exception | ERR00999 |

### Error Code Source Files
- Python definitions: search for `_error_code.py` in torch_npu source
- C++ definitions: search for `NPUException.h` in torch_npu source
- C++ formatting macros: `PTA_ERROR()`, `OPS_ERROR()`, `DIST_ERROR()`, `GRAPH_ERROR()`, `PROF_ERROR()`

## Python Exception Types

torch_npu raises standard Python exceptions with structured error messages containing ERR codes.

| Exception | Common Causes |
|-----------|--------------|
| `RuntimeError` | Device errors, operator execution failures, ACL API failures, tensor mismatches |
| `ValueError` | Invalid parameter values, shape mismatches, invalid configurations |
| `TypeError` | Wrong data types, incompatible tensor types, device type mismatch |
| `NotImplementedError` | Operator not supported on NPU backend (privateuse1) |
| `ImportError` | CANN libraries not found (libhccl.so, libascendcl.so), version mismatch |

## CANN/ACL Error Codes (Ascend Backend)

Error code rules:
- `1xxxxx` — Environment/logic errors, user-fixable
- `2xxxxx` — Resource exhaustion (memory, streams, devices)
- `3xxxxx` — Business exceptions (queue full/empty, storage)
- `5xxxxx` — Internal software/hardware errors, need technical support

### ACL General Errors (100000-148053)
| Code | Name | Description | Solution |
|------|------|-------------|----------|
| 100000 | ACL_ERROR_INVALID_PARAM | Parameter verification failed | Check input parameters |
| 100001 | ACL_ERROR_UNINITIALIZE | Not initialized | Ensure aclInit called before other APIs |
| 100002 | ACL_ERROR_REPEAT_INITIALIZE | Repeated initialization | Avoid duplicate init calls |
| 100003-100006 | File errors | Invalid/failed/parse file errors | Check file exists, permissions, content |
| 100020 | ACL_ERROR_OP_TYPE_NOT_MATCH | Operator type mismatch | Check operator type |
| 100021 | ACL_ERROR_OP_INPUT_NOT_MATCH | Operator input mismatch | Check operator inputs |
| 100024 | ACL_ERROR_OP_NOT_FOUND | Operator not found | Check if operator type is supported |
| 100026 | ACL_ERROR_UNSUPPORTED_DATA_TYPE | Unsupported data type | Check dtype support |
| 100027 | ACL_ERROR_FORMAT_NOT_MATCH | Format mismatch | Check tensor format |
| 148049 | ACL_ERROR_INVALID_OPP_PATH | ASCEND_OPP_PATH not set or invalid | Set ASCEND_OPP_PATH to opp install path |

### ACL Resource Errors (200000-200007)
| Code | Name | Description | Solution |
|------|------|-------------|----------|
| 200000 | ACL_ERROR_BAD_ALLOC | Memory allocation failed (OOM) | Check available memory, reduce batch size |
| 200001 | ACL_ERROR_API_NOT_SUPPORT | API not supported | Check API compatibility |
| 200002 | ACL_ERROR_INVALID_DEVICE | Invalid device | Check device exists |
| 200006 | ACL_ERROR_FEATURE_UNSUPPORTED | Feature not supported | Check CANN version compatibility |

### ACL Internal Errors (500000-500005)
| Code | Name | Description | Solution |
|------|------|-------------|----------|
| 500000 | ACL_ERROR_INTERNAL_ERROR | Unknown internal error | Collect logs, contact support |
| 500001 | ACL_ERROR_FAILURE | Internal error | Collect logs, contact support |
| 500002 | ACL_ERROR_GE_FAILURE | Graph Engine error | Collect logs, contact support |
| 500003 | ACL_ERROR_RT_FAILURE | Runtime error | Collect logs, contact support |
| 500004 | ACL_ERROR_DRV_FAILURE | Driver error | Collect logs, contact support |

### Runtime Parameter/Context Errors (107xxx)
| Code | Name | Description | Solution |
|------|------|-------------|----------|
| 107000 | ACL_ERROR_RT_PARAM_INVALID | Parameter verification failed | Check input parameters |
| 107001 | ACL_ERROR_RT_INVALID_DEVICEID | Invalid device ID | Check device ID is valid |
| 107002 | ACL_ERROR_RT_CONTEXT_NULL | Context is empty | Call aclrtSetCurrentContext or aclrtSetDevice |
| 107003 | ACL_ERROR_RT_STREAM_CONTEXT | Stream not in current context | Check stream-context association |
| 107019 | ACL_ERROR_RT_WAIT_TIMEOUT | Wait timeout | Retry the operation |
| 107020 | ACL_ERROR_RT_TASK_TIMEOUT | Task execution timeout | Check business logic or adjust timeout |
| 107022 | ACL_ERROR_RT_DEVICE_TASK_ABORT | Device task abort conflict | Wait for abort to finish before other ops |

### Runtime Resource Errors (207xxx)
| Code | Name | Description | Solution |
|------|------|-------------|----------|
| 207000 | ACL_ERROR_RT_FEATURE_NOT_SUPPORT | Feature not supported | Check CANN logs |
| 207001 | ACL_ERROR_RT_MEMORY_ALLOCATION | Memory allocation failed | Check storage space |
| 207003 | ACL_ERROR_RT_AICORE_OVER_FLOW | AI Core overflow | Check operator for overflow, use dump data |
| 207004 | ACL_ERROR_RT_NO_DEVICE | Device unavailable | Check device is running |
| 207018 | ACL_ERROR_RT_DEVIDE_OOM | Device memory exhausted | Check device memory, optimize usage |

### Runtime Internal/Hardware Errors (507xxx)
| Code | Name | Description | Solution |
|------|------|-------------|----------|
| 507010 | ACL_ERROR_RT_LOST_HEARTBEAT | Task scheduler heartbeat lost | Contact support |
| 507011 | ACL_ERROR_RT_MODEL_EXECUTE | Model execution failed | Contact support |
| 507014 | ACL_ERROR_RT_AICORE_TIMEOUT | AI Core execution timeout | Contact support |
| 507015 | ACL_ERROR_RT_AICORE_EXCEPTION | AI Core execution exception | Contact support |
| 507017 | ACL_ERROR_RT_AICPU_TIMEOUT | AI CPU execution timeout | Contact support |
| 507034 | ACL_ERROR_RT_VECTOR_CORE_TIMEOUT | Vector core execution timeout | Contact support |
| 507053 | ACL_ERROR_RT_DEVICE_MEM_ERROR | Memory UCE error | Use aclrtGetMemUceInfo to fix |
| 507054 | ACL_ERROR_RT_HBM_MULTI_BIT_ECC_ERROR | HBM ECC error | Hardware fault, contact support |
| 507056 | ACL_ERROR_RT_LINK_ERROR | Inter-device communication link error | Retry, check communication link |

### GE (Graph Engine) Errors (145xxx-545xxx)
| Code | Name | Description | Solution |
|------|------|-------------|----------|
| 145000 | ACL_ERROR_GE_PARAM_INVALID | Parameter verification failed | Check input parameters |
| 145019 | ACL_ERROR_GE_PLGMGR_PATH_INVALID | Invalid so file/path | Check LD_LIBRARY_PATH |
| 145020 | ACL_ERROR_GE_FORMAT_INVALID | Invalid format | Check tensor format |
| 145021 | ACL_ERROR_GE_SHAPE_INVALID | Invalid shape | Check tensor shape |
| 145022 | ACL_ERROR_GE_DATATYPE_INVALID | Invalid data type | Check tensor dtype |
| 245000 | ACL_ERROR_GE_MEMORY_ALLOCATION | Memory allocation failed | Check available memory |
| 545000 | ACL_ERROR_GE_INTERNAL_ERROR | Unknown internal error | Contact support |
| 545601 | ACL_ERROR_GE_MODEL_EXECUTE_TIMEOUT | Model execution timeout | Contact support |

## ACLNN Error Codes (Ascend Operator API)

ACLNN errors occur during the two-phase interface: `aclnnXxxGetWorkspaceSize` → `aclnnXxx`.

### General Status Codes
| Code | Name | Description |
|------|------|-------------|
| 0 | ACLNN_SUCCESS | Success |
| 161001 | ACLNN_ERR_PARAM_NULLPTR | Illegal nullptr in parameter |
| 161002 | ACLNN_ERR_PARAM_INVALID | Parameter validation error (dtype mismatch, etc.) |
| 361001 | ACLNN_ERR_RUNTIME_ERROR | Internal NPU runtime call exception |

### Internal Error Codes (561xxx)
| Code | Name | Description |
|------|------|-------------|
| 561000 | ACLNN_ERR_INNER | General internal exception |
| 561001 | ACLNN_ERR_INNER_INFERSHAPE_ERROR | Output shape inference error |
| 561002 | ACLNN_ERR_INNER_TILING_ERROR | NPU kernel tiling error |
| 561003 | ACLNN_ERR_INNER_FIND_KERNEL_ERROR | Kernel not found (binary package may not be installed) |
| 561101 | ACLNN_ERR_INNER_CREATE_EXECUTOR | Failed to create aclOpExecutor |
| 561107 | ACLNN_ERR_INNER_OPP_PATH_NOT_FOUND | ASCEND_OPP_PATH not configured |
| 561108 | ACLNN_ERR_INNER_LOAD_JSON_FAILED | Failed to load operator kernel JSON |
| 561112 | ACLNN_ERR_INNER_OPP_KERNEL_PKG_NOT_FOUND | Operator binary kernel package not found |

## CANN Inner Error Codes (Alphanumeric)

Format: `E[module-prefix][error-code]`

| Prefix | Module | Description |
|--------|--------|-------------|
| E5-E8 | AICORE checksum | Checksum errors during AICORE compilation |
| E9-EB | AICORE TBE | TBE compilation, operator spec issues |
| E80xxx | AICORE ops | Specific operator errors (StridedSliceGradD, ReduceSum, etc.) |
| EB0xxx | AICORE | Transpose spec unsupported, UB memory overflow |
| EZ9999 | AICORE execution | rtStreamSynchronize failure, model execution failed |
| E39999 | AICPU | AICPU kernel execution failure |
| EE9999 | Runtime | HBM OOM, task failure, halMemAlloc failure |
| EE1001 | Runtime | Invalid device ID |
| EI0002 | HCCL | Notify wait timeout |
| EI0004 | HCCL | Invalid ranktable |
| EI0005 | HCCL | Inconsistent comm parameters (AllReduce count mismatch, etc.) |
| EI0006 | HCCL | Socket build timeout |
| EJ0001 | HCCP | HCCP init failure |
| EK0001 | Profiling | Invalid profiling path/parameters |
| EL0004 | CANN | OOM error |

## Hardware Error Detection

Keywords detected by torch_npu's error handler (`NPUException.h`):

| Keyword | Meaning | Error Code |
|---------|---------|-----------|
| `DEVICE_TASK_ABORT` / `device task abort` | Device hardware failure | 107010 |
| `DEVICE_HBM_ECC_ERROR` / `hbm Multi-bit ECC error` | HBM memory ECC error | 507054 |
| `DEVICE_MEM_ERROR` / `UCE ERROR` | Uncorrectable memory error | 507053 |
| `HCCS_LINK_ERROR` / `network link error` | Network/hardware link issue | 507056 |
| `LOST_HEARTBEAT` | Task scheduler heartbeat lost | 507010 |
| `AICORE_EXCEPTION` | AI Core execution exception | 507015 |
| `HCCL_OP_RETRY_FAILED` | HCCL operation retry exhausted | — |
| `SUSPECT_REMOTE_ERROR` | Remote device suspected error | — |
| `SUSPECT_DEVICE_MEM_ERROR` | Suspected memory error | — |

### HCCL Errors
| Pattern | Description | Solution |
|---------|-------------|----------|
| HCCL timeout | Network communication timeout | Check network connectivity, verify HCCL config |
| HCCL link error | HCCS link issue | Check physical connections, card topology |
| EI0002 | Notify wait timeout | Check if all ranks are alive and progressing |
| EI0005 | Inconsistent comm parameters | Verify all ranks use same arguments for collective ops |
| EI0006 | Socket build timeout | Check firewall, network connectivity |
| EJ0001 | HCCP init failure | Check HCCL rank table, device configuration |

## Error Trigger Sources

1. **torch_npu Framework** — ERRxxxxx codes from parameter validation, type checking, API calls
2. **CANN Runtime** — 1xxxxx/2xxxxx/5xxxxx codes from ACL/HCCL/GE API returns
3. **CANN Operators** — ACLNN 161xxx/361xxx/561xxx codes from operator execution
4. **CANN Inner** — E[x]xxxx codes from TBE/AICPU/HCCL internal modules
5. **Hardware/Driver** — 507xxx codes from device hardware conditions

## Error Location

- Application logs: stdout/stderr
- torch_npu error handler: auto-detects `ERR\d{5}` pattern, formats error output
- CANN device logs: `/var/log/npu/slog/*/device-*/plog/`
- OOM snapshots: `OOM_SNAPSHOT_PATH` directory (when `OOM_SNAPSHOT_ENABLE=1`)
- CANN error messages: `aclGetRecentErrMsg()` (called automatically by `NPU_CHECK_ERROR`)
- Historical patterns: [Failure Showcase](failure-showcase.md)

## See Also

- [CANN API Reference](cann-api-reference.md) — ACLNN API constraints, op-plugin adaptation flow
- [Backend Diagnosis](backend-diagnosis.md) — Step-by-step diagnosis and further location techniques
- [Torch_npu Operators](torch-npu-operators.md) — Operator registration, implementation, debugging
- [Failure Showcase](failure-showcase.md) — Historical failures indexed by error codes and keywords
