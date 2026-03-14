#!/usr/bin/env python3
"""
将 NPU 显存测试脚本转换为对应的 GPU 版本。

转换规则（从 npu/gpu 标杆脚本对比提取）:
  1. 删除 import torch_npu
  2. torch_npu.npu.xxx  → torch.cuda.xxx
  3. torch.npu.xxx      → torch.cuda.xxx
  4. device("npu")      → device("cuda")
  5. device='npu'        → device='cuda'
  6. pta_reserved_GB    → gpu_reserved_GB
  7. pta_activated_GB   → gpu_activated_GB

用法:
    python convert_npu_to_gpu.py <npu脚本路径> [-o <输出路径>]

示例:
    python convert_npu_to_gpu.py torchapi_id0299_nanmean.py
    python convert_npu_to_gpu.py torchapi_id0299_nanmean.py -o torchapi_id0299_nanmean_gpu.py
"""

import argparse
import os
import re
import sys


TRANSFORMATIONS = [
    # (pattern, replacement, description)
    # 顺序重要：先处理更长/更具体的模式，避免短模式误匹配

    # 1. 删除 torch_npu 导入行
    (r'^import torch_npu\s*\n', '', 'remove import torch_npu'),
    (r'^from torch_npu.*\n', '', 'remove from torch_npu import'),

    # 2. torch_npu.npu.xxx → torch.cuda.xxx (必须在 torch.npu 之前)
    (r'torch_npu\.npu\.', 'torch.cuda.', 'torch_npu.npu.* → torch.cuda.*'),

    # 3. torch.npu.xxx → torch.cuda.xxx
    (r'torch\.npu\.', 'torch.cuda.', 'torch.npu.* → torch.cuda.*'),

    # 4. device 字符串替换
    (r'device\s*\(\s*["\']npu["\']\s*\)', 'device("cuda")', 'device("npu") → device("cuda")'),
    (r"device\s*=\s*['\"]npu['\"]", "device='cuda'", "device='npu' → device='cuda'"),

    # 5. 变量名替换
    (r'pta_reserved_GB', 'gpu_reserved_GB', 'pta_reserved_GB → gpu_reserved_GB'),
    (r'pta_activated_GB', 'gpu_activated_GB', 'pta_activated_GB → gpu_activated_GB'),

    # 6. JSON key 替换（引号内的）
    (r'"pta_reserved_GB"', '"gpu_reserved_GB"', '"pta_reserved_GB" → "gpu_reserved_GB"'),
    (r'"pta_activated_GB"', '"gpu_activated_GB"', '"pta_activated_GB" → "gpu_activated_GB"'),
]


def convert_npu_to_gpu(content):
    """应用所有转换规则，返回 (转换后内容, 应用的规则列表)。"""
    applied = []
    result = content
    for pattern, replacement, desc in TRANSFORMATIONS:
        new_result = re.sub(pattern, replacement, result, flags=re.MULTILINE)
        if new_result != result:
            applied.append(desc)
            result = new_result
    return result, applied


def default_output_path(input_path):
    """input.py → input_gpu.py，如果已有 _npu 后缀则替换为 _gpu。"""
    base, ext = os.path.splitext(input_path)
    if base.endswith('_npu'):
        return base[:-4] + '_gpu' + ext
    return base + '_gpu' + ext


def main():
    parser = argparse.ArgumentParser(
        description='将 NPU 显存测试脚本转换为 GPU 版本',
    )
    parser.add_argument('input', help='NPU 测试脚本路径')
    parser.add_argument('-o', '--output', help='输出 GPU 脚本路径 (默认: <input>_gpu.py)')
    parser.add_argument('--diff', action='store_true', help='仅显示会应用的转换规则，不写文件')
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f'错误: 文件不存在: {args.input}', file=sys.stderr)
        sys.exit(1)

    with open(args.input, 'r', encoding='utf-8') as f:
        content = f.read()

    converted, applied = convert_npu_to_gpu(content)

    if not applied:
        print('未检测到需要转换的 NPU 特征，请确认输入文件是 NPU 脚本。')
        return

    print(f'应用了 {len(applied)} 条转换规则:')
    for rule in applied:
        print(f'  * {rule}')

    if args.diff:
        return

    output_path = args.output or default_output_path(args.input)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(converted)

    print(f'已生成 GPU 脚本: {output_path}')


if __name__ == '__main__':
    main()
