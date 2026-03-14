# Memory Consistency Issue Cases

Historical NPU vs GPU memory consistency issues and their root cause analysis.

Format:
```yaml
- dts_number: "[DTS defect number]"
  description: "[brief issue description]"
  aclnn_interface: "[related aclnn interface or note]"
  root_cause: "[root cause analysis]"
  solution: "[resolution strategy]"
  category: "[issue classification]"
```

## Cases

### DTS2025072527150 - aclnnInplaceNormal
- dts_number: "DTS2025072527150"
- description: "【ACLNN】【内存占用】torch.randn接口内存占用是torch_gpu的3倍"
- aclnn_interface: "aclnnInplaceNormal"
- root_cause: "aclnn接口内部插了一次cast"
- solution: "转需求，补充非cast路线"
- category: "cast导致显存占用升高"

### DTS2026013114653 - 缺少对应的aclnn接口
- dts_number: "DTS2026013114653"
- description: "【A3】【PT2.6】torch.linalg.solve内存一致性不达标，和gpu的差距在5%以上"
- aclnn_interface: "缺少对应的aclnn接口"
- root_cause: "aten::_linalg_solve_ex.result以及aten::linalg_solve_backward缺少对应的npu后端实现"
- solution: "转需求，开发对应aclnn算子"
- category: "缺少aclnn算子"