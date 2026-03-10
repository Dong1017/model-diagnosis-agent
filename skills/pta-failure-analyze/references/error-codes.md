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

### Error Codes
| Code | Description | Example |
|------|-------------|---------|
| 001 | Invalid parameter | ERR00001, ERR01001 |
| 002 | Invalid type | ERR00002 |
| 003 | Invalid value | ERR00003 |
| 004 | Invalid pointer | ERR00004 |
| 005 | Internal error | ERR00005 |
| 006 | Memory error | ERR00006 |
| 007 | Feature not supported | ERR00007 |
| 008 | Resource not found | ERR00008 |
| 009 | Resource unavailable | ERR00009 |
| 010 | System call failed | ERR00010 |
| 011 | Timeout error | ERR00011 |
| 012 | Permission error | ERR00012 |
| 100 | Call ACL API failed | ERR00100 |
| 200 | Call HCCL API failed | ERR00200 |
| 300 | Call GE API failed | ERR00300 |

## CANN/ACL Error Codes (100000+ Series)

### General ACL Errors (100000-200000)
| Code | Error | Solution |
|------|-------|---------|
| 100000 | Parameter verification failed | Check if input parameters are correct |
| 100001 | ACL uninitialized | Call acl.init before other interfaces |
| 100002 | Repeated initialization/loading | Avoid duplicate initialization calls |
| 100003-100006 | File errors (invalid/failed/parsing) | Check file exists and has correct permissions |
| 200000 | Failed to apply for memory (OOM) | Check available memory, reduce batch size |
| 200001 | Interface does not support this function | Check if interface is supported |
| 200006 | Feature not supported | Check CANN logs, contact support |

### Runtime Errors (107000-507901)
| Code | Error | Solution |
|------|-------|---------|
| 107002 | Context is empty | Check if acl.rt.set_context or acl.rt.set_device is called |
| 107003 | Stream not in current context | Check stream context matches current context |
| 107010 | Device task abort (FORCE STOP) | Hardware issue, may need device reset |
| 107019 | Task execution timed out | Re-execute the interface |
| 207000 | Feature not supported | Check CANN logs for error information |
| 207001 | Failed to apply for memory | Check remaining storage space |
| 207018 | Memory on device exhausted | Check device memory usage, optimize memory usage |
| EL0004 | CANN OOM error | Reduce batch size, clear cache |

### HCCL Errors (handled implicitly in distributed operations)
| Code | Error | Solution |
|------|-------|---------|
| HCCL timeout | Network communication issue | Check network connectivity, verify HCCL config |
| HCCL link error | HCCS link issue | Check physical connections, card topology |

## Hardware Error Detection

Keywords indicating hardware errors:
- `DEVICE_TASK_ABORT` or `device task abort` (107010)
- `HBM_MULTIBIT_ECC_ERROR` - HBM memory ECC error
- `DEVICE_MEM_ERROR` or `UCE ERROR` - Uncorrectable memory error
- `LINK_ERROR` or `network link error` - Network/hardware link issue
- `reason=device task abort` - Modern format
- `reason=hbm Multi-bit ECC error` - HBM ECC error format

## CANN Inner Errors

**Format:** `E[1-9A-Z]9999`

Indicates internal CANN errors. Check CANN logs in `/var/log/npu/slog/*/device-*/plog/` for detailed information.

## Error Trigger Sources

1. **Torch_npu Framework** - ERRxxxxx codes from parameter validation, type checking
2. **CANN Runtime** - 100000+ codes from ACL/HCCL/GE API returns
3. **Hardware/Driver** - 107xxx codes from device hardware conditions

## Error Location

- Application logs: stdout/stderr
- CANN device logs: `/var/log/npu/slog/*/device-*/plog/`
- Historical event marking: Available in failure showcase
