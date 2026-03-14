#!/usr/bin/env python3
"""
NPU plog memory filter - extract memory-relevant lines for debugging.

Usage:
    python filter_plog_memory.py <plog_file> [-o output] [-v] [--context N]

Examples:
    python filter_plog_memory.py mem_plog-xxx.log
    python filter_plog_memory.py mem_plog-xxx.log -o filtered.log -v --context 1
"""

import re
import sys
import argparse
from collections import defaultdict, OrderedDict


FILTER_RULES = OrderedDict([
    ("workspace_alloc", {
        "desc": "workspace alloc size",
        "pattern": r'unsafe_empty_workspace.*Alloc workspace',
    }),
    ("workspace_size_computed", {
        "desc": "workspace computed size",
        "pattern": r'workspaceSize_:',
    }),
    ("op_get_workspace", {
        "desc": "aclnn GetWorkspaceSize done",
        "pattern": r'aclnn\w+GetWorkspaceSize\]\[\d+\].*Leaving function',
    }),
    ("op_run", {
        "desc": "aclnn op Run",
        "pattern": r'"Op aclnn\w+ Run',
    }),
    ("op_exec_handle", {
        "desc": "aclnn op exec with handle",
        "pattern": r'"Exec Op aclnn\w+',
    }),
    ("pta_malloc", {
        "desc": "PTA malloc (malloc/cached/allocated)",
        "pattern": r'PTA CachingAllocator malloc:',
    }),
    ("pta_free", {
        "desc": "PTA free (free/cached/allocated)",
        "pattern": r'PTA CachingAllocator free:',
    }),
    ("pta_acl_malloc", {
        "desc": "PTA acl_malloc",
        "pattern": r'pta_memory acl_malloc:',
    }),
    ("dev_malloc", {
        "desc": "Driver DevMalloc (size+addr)",
        "pattern": r'DevMalloc:.*size=',
    }),
    ("dev_free", {
        "desc": "Driver DevFree",
        "pattern": r'DevFree:',
    }),
    ("graph_tensor_mem", {
        "desc": "kernel graph tensor memory",
        "pattern": r'PrintTensors.*memory size',
    }),
    ("graph_node_output", {
        "desc": "kernel graph node (output->child)",
        "pattern": r'PrintGraph.*node\[.*\].*output.*child-nodes',
    }),
    ("graph_node_input", {
        "desc": "kernel graph node (input<-parent)",
        "pattern": r'PrintGraph.*node\[.*\].*input.*father-nodes',
    }),
    ("op_params_tensor", {
        "desc": "op params (tensor shape/dtype/ptr)",
        "pattern": r'Entering function params:.*aclTensor\(',
    }),
    ("cast_enter", {
        "desc": "Cast type conversion",
        "pattern": r'CastAiCpu.*Entering function CastAiCpu',
    }),
    ("contiguous_enter", {
        "desc": "Contiguous copy",
        "pattern": r'Contiguous\]\[\d+\].*Entering function Contiguous',
    }),
    ("viewcopy_enter", {
        "desc": "ViewCopy",
        "pattern": r'ViewCopy\]\[\d+\].*Entering function ViewCopy',
    }),
    ("op_entry_normal", {
        "desc": "normal_ entry",
        "pattern": r'"normal_ exec',
    }),
    ("op_entry_pta_marker", {
        "desc": "PTA op entry marker",
        "pattern": r'\[PTA\].*exec with jit compile',
    }),
])

VERBOSE_RULES = OrderedDict([
    ("aclrt_malloc_align", {
        "desc": "aclrtMallocAlign32 (aligned size)",
        "pattern": r'aclrtMallocAlign32Impl:.*size\s*=',
    }),
    ("dev_mem_alloc_online", {
        "desc": "DevMemAllocOnline (type=1024, user tensor)",
        "pattern": r'DevMemAllocOnline:.*type=0,\s*size=',
    }),
    ("graph_kernel_add", {
        "desc": "kernel graph AddKernelNode",
        "pattern": r'AddKernelNode.*Add kernel nodes',
    }),
    ("graph_sort", {
        "desc": "kernel graph sorting",
        "pattern": r'SortKernelTensor.*Kernel graph before sorting',
    }),
    ("contiguous_leave", {
        "desc": "Contiguous done",
        "pattern": r'Contiguous\]\[\d+\].*Leaving function Contiguous',
    }),
    ("cast_leave", {
        "desc": "Cast done",
        "pattern": r'CastAiCpu.*Leaving function CastAiCpu',
    }),
    ("viewcopy_leave", {
        "desc": "ViewCopy done",
        "pattern": r'ViewCopy\]\[\d+\].*Leaving function ViewCopy',
    }),
    ("mem_allocator_offset", {
        "desc": "MaxAllocator offset layout",
        "pattern": r'MaxAllocator tensor index:.*offset:',
    }),
    ("op_get_workspace_enter", {
        "desc": "aclnn GetWorkspaceSize start",
        "pattern": r'aclnn\w+GetWorkspaceSize\]\[\d+\].*Entering function',
    }),
    ("make_sure_queue_empty", {
        "desc": "TaskQueue flush",
        "pattern": r'MakeSureQueueEmpty|Begin to makesure taskqueue empty',
    }),
])


def compile_rules(rules):
    compiled = []
    for name, rule in rules.items():
        compiled.append((name, rule["desc"], re.compile(rule["pattern"])))
    return compiled


def format_bytes(size_bytes):
    if size_bytes >= 1 << 30:
        return f"{size_bytes / (1 << 30):.3f} GiB"
    elif size_bytes >= 1 << 20:
        return f"{size_bytes / (1 << 20):.2f} MiB"
    elif size_bytes >= 1 << 10:
        return f"{size_bytes / (1 << 10):.1f} KiB"
    return f"{size_bytes} B"


def extract_workspace_size(line):
    m = re.search(r'Alloc workspace (\d+) bytes', line)
    if m:
        return int(m.group(1))
    return None


def extract_dev_malloc_size(line):
    m = re.search(r'DevMalloc:.*size=(\d+)', line)
    if m:
        return int(m.group(1))
    return None


def extract_pta_stats(line):
    m = re.search(
        r'malloc\s*=\s*(\d+),\s*cached\s*=\s*(\d+),\s*allocated\s*=\s*(\d+)',
        line,
    )
    if not m:
        m = re.search(
            r'free\s*=\s*(\d+),\s*cached\s*=\s*(\d+),\s*allocated\s*=\s*(\d+)',
            line,
        )
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return None


def extract_op_name(line):
    m = re.search(r'OpName:\[(aclnn\w+?)(?:_\d+)?\]', line)
    if m:
        return m.group(1)
    m = re.search(r'Op (aclnn\w+) Run', line)
    if m:
        return m.group(1)
    m = re.search(r'Exec Op (aclnn\w+)', line)
    if m:
        return m.group(1)
    return None


def infer_nearest_op_name(all_lines, idx, *, lookback=80, lookahead=10):
    """在 idx 附近推断这条记录最可能归属的 aclnn 算子。

    优先向上回溯（一般 Alloc workspace 会出现在 Exec/Run/WorkspaceSize 附近）。
    如果回溯找不到，再小范围向前查找。
    """
    # 1) look back
    start = max(0, idx - lookback)
    for j in range(idx, start - 1, -1):
        op = extract_op_name(all_lines[j])
        if op:
            return op
    # 2) look ahead (rare ordering)
    end = min(len(all_lines) - 1, idx + lookahead)
    for j in range(idx + 1, end + 1):
        op = extract_op_name(all_lines[j])
        if op:
            return op
    return None


def main():
    parser = argparse.ArgumentParser(
        description="NPU plog memory filter",
    )
    parser.add_argument("logfile", help="plog log file path")
    parser.add_argument("-o", "--output", help="output file (default: stdout)")
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="include more auxiliary info",
    )
    parser.add_argument(
        "--context", type=int, default=0, metavar="N",
        help="context lines around each match (default: 0)",
    )
    args = parser.parse_args()

    rules = compile_rules(FILTER_RULES)
    if args.verbose:
        rules += compile_rules(VERBOSE_RULES)

    try:
        with open(args.logfile, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: file not found: {args.logfile}", file=sys.stderr)
        sys.exit(1)

    total_lines = len(all_lines)
    matched_indices = set()
    category_hits = defaultdict(int)

    for idx, line in enumerate(all_lines):
        for rule_name, rule_desc, pattern in rules:
            if pattern.search(line):
                matched_indices.add(idx)
                category_hits[rule_name] += 1
                break

    if args.context > 0:
        expanded = set()
        for idx in matched_indices:
            for offset in range(-args.context, args.context + 1):
                new_idx = idx + offset
                if 0 <= new_idx < total_lines:
                    expanded.add(new_idx)
        matched_indices = expanded

    sorted_indices = sorted(matched_indices)

    workspace_sizes = []
    workspace_ops = []
    dev_malloc_sizes = []
    peak_allocated = 0
    peak_cached = 0
    ops_seen = []

    for idx in sorted_indices:
        line = all_lines[idx]
        ws = extract_workspace_size(line)
        if ws is not None:
            workspace_sizes.append(ws)
            op = infer_nearest_op_name(all_lines, idx)
            workspace_ops.append(op or "UNKNOWN")
        dm = extract_dev_malloc_size(line)
        if dm is not None:
            dev_malloc_sizes.append(dm)
        stats = extract_pta_stats(line)
        if stats:
            _, cached, allocated = stats
            peak_allocated = max(peak_allocated, allocated)
            peak_cached = max(peak_cached, cached)
        op = extract_op_name(line)
        if op and (not ops_seen or ops_seen[-1] != op):
            ops_seen.append(op)

    out_file = (
        open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    )

    try:
        sep = "=" * 80
        out_file.write(f"{sep}\n")
        out_file.write(
            f"  NPU Memory Log Filter  |  {total_lines} -> "
            f"{len(sorted_indices)} lines"
            f"  ({len(sorted_indices)/max(total_lines,1)*100:.1f}%)\n"
        )
        out_file.write(f"  Source: {args.logfile}\n")
        out_file.write(f"{sep}\n\n")

        out_file.write("[Summary]\n")
        if ops_seen:
            out_file.write(f"  Op chain: {' -> '.join(ops_seen)}\n")
        if workspace_sizes:
            out_file.write(f"  Workspace allocs: {len(workspace_sizes)}\n")
            for i, (ws, op) in enumerate(zip(workspace_sizes, workspace_ops), 1):
                out_file.write(
                    f"    #{i}: {ws:>15,} bytes  ({format_bytes(ws)})"
                    f"  |  op: {op}\n"
                )
            out_file.write(
                f"    Total: {sum(workspace_sizes):>13,} bytes"
                f"  ({format_bytes(sum(workspace_sizes))})\n"
            )
        if dev_malloc_sizes:
            out_file.write(f"  DevMalloc calls: {len(dev_malloc_sizes)}\n")
            for i, dm in enumerate(dev_malloc_sizes, 1):
                out_file.write(
                    f"    #{i}: {dm:>15,} bytes  ({format_bytes(dm)})\n"
                )
            out_file.write(
                f"    Total: {sum(dev_malloc_sizes):>13,} bytes"
                f"  ({format_bytes(sum(dev_malloc_sizes))})\n"
            )
        if peak_allocated > 0:
            out_file.write(
                f"  PTA peak allocated: {peak_allocated:>13,} bytes"
                f"  ({format_bytes(peak_allocated)})\n"
            )
            out_file.write(
                f"  PTA peak cached:    {peak_cached:>13,} bytes"
                f"  ({format_bytes(peak_cached)})\n"
            )

        out_file.write(f"\n  Hits by category:\n")
        all_rule_names = list(FILTER_RULES.keys())
        if args.verbose:
            all_rule_names += list(VERBOSE_RULES.keys())
        all_rules_dict = {**FILTER_RULES, **VERBOSE_RULES}
        for rule_name in all_rule_names:
            if rule_name in category_hits:
                desc = all_rules_dict[rule_name]["desc"]
                out_file.write(
                    f"    {desc:<40s}  {category_hits[rule_name]:>4d} lines\n"
                )

        out_file.write(f"\n{sep}\n")
        out_file.write("[Filtered Log]\n")
        out_file.write(f"{sep}\n\n")

        prev_idx = -2
        for idx in sorted_indices:
            if prev_idx >= 0 and idx - prev_idx > 1:
                gap = idx - prev_idx - 1
                out_file.write(
                    f"{'... (skip ' + str(gap) + ' lines) ...':^87s}\n"
                )
            line = all_lines[idx].rstrip("\n\r")
            out_file.write(f"{idx+1:>7d}| {line}\n")
            prev_idx = idx

        out_file.write(f"\n{sep}\n")
        ratio = total_lines / max(len(sorted_indices), 1)
        out_file.write(
            f"  Output {len(sorted_indices)} lines"
            f" (from {total_lines}, ratio {ratio:.0f}:1)\n"
        )
        out_file.write(f"{sep}\n")

    finally:
        if out_file is not sys.stdout:
            out_file.close()
            print(
                f"Written: {args.output}  ({len(sorted_indices)} lines)",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
