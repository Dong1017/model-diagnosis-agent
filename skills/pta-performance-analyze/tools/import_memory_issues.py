#!/usr/bin/env python3
"""
从 xlsx 问题记录表导入显存一致性问题到 memory_consistency_issue_cases.md

读取 xlsx 中 B/C/D/F/G/H 六列，去重后追加到 md 文件末尾。
使用 Python 标准库解析 xlsx（无需 openpyxl）。

用法:
    python import_memory_issues.py <xlsx文件路径> [--md <md文件路径>] [--dry-run]

示例:
    python import_memory_issues.py ../references/api_consistency_issue_list.xlsx
    python import_memory_issues.py new_issues.xlsx --dry-run
"""

import argparse
import os
import re
import sys
import zipfile
import xml.etree.ElementTree as ET

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MD_PATH = os.path.join(SCRIPT_DIR, "..", "references", "memory_consistency_issue_cases.md")

NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

TARGET_COLS = ["B", "C", "D", "F", "G", "H"]
COL_LABELS = {
    "B": "dts_number",
    "C": "description",
    "D": "aclnn_interface",
    "F": "root_cause",
    "G": "solution",
    "H": "category",
}


def parse_xlsx(xlsx_path):
    """用标准库解析 xlsx，按 DTS 单号分组返回。

    返回: [{
        "B": "DTS...", "C": "描述...",
        "sub_issues": [{"D": "...", "F": "...", "G": "...", "H": "..."}, ...]
    }, ...]
    同一 DTS 的多行会合并为一条，共享 B/C，D/F/G/H 归入 sub_issues 列表。
    连续的空 B 行继承上一条 DTS 的 B/C。
    """
    with zipfile.ZipFile(xlsx_path) as zf:
        shared_strings = _parse_shared_strings(zf)
        raw_rows = _parse_sheet_rows(zf, shared_strings)
    return _group_by_dts(raw_rows)


def _parse_shared_strings(zf):
    tag_t = f"{{{NS['s']}}}t"
    strings = []
    try:
        tree = ET.parse(zf.open("xl/sharedStrings.xml"))
    except KeyError:
        return strings
    for si in tree.findall(".//s:si", NS):
        parts = [t.text for t in si.iter(tag_t) if t.text]
        strings.append("".join(parts))
    return strings


def _parse_sheet_rows(zf, shared_strings):
    """返回原始行列表（保留空 B 行，用于后续合并）。"""
    tree = ET.parse(zf.open("xl/worksheets/sheet1.xml"))
    rows = tree.findall(".//s:sheetData/s:row", NS)
    results = []
    for i, row in enumerate(rows):
        if i == 0:
            continue
        cell_map = {}
        for cell in row.findall("s:c", NS):
            ref = cell.get("r")
            col = "".join(ch for ch in ref if ch.isalpha())
            if col not in TARGET_COLS:
                continue
            cell_map[col] = _cell_value(cell, shared_strings)
        has_any_content = any(cell_map.get(c, "").strip() for c in TARGET_COLS)
        if not has_any_content:
            continue
        results.append(cell_map)
    return results


SUB_ISSUE_COLS = ["D", "F", "G", "H"]


def _group_by_dts(raw_rows):
    """按 DTS 单号分组，空 B 行继承上一条的 B/C。"""
    from collections import OrderedDict
    groups = OrderedDict()
    last_dts = None

    for row in raw_rows:
        dts = row.get("B", "").strip()
        desc = row.get("C", "").strip()

        if dts:
            last_dts = dts
            if dts not in groups:
                groups[dts] = {"B": dts, "C": desc, "sub_issues": []}
            elif desc and not groups[dts]["C"]:
                groups[dts]["C"] = desc
        elif last_dts:
            dts = last_dts
        else:
            continue

        sub = {c: row.get(c, "").strip() for c in SUB_ISSUE_COLS}
        if any(sub.values()):
            groups[dts]["sub_issues"].append(sub)

    return list(groups.values())


def _cell_value(cell, shared_strings):
    v_elem = cell.find("s:v", NS)
    if v_elem is None or v_elem.text is None:
        return ""
    if cell.get("t") == "s":
        idx = int(v_elem.text)
        return shared_strings[idx] if idx < len(shared_strings) else ""
    return v_elem.text


def load_existing_dts_numbers(md_path):
    """从已有 md 文件中提取所有已记录的 DTS 单号（兼容新旧标题格式）。"""
    if not os.path.isfile(md_path):
        return set()
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()
    by_field = re.findall(r'^- dts_number:\s*"(DTS\w+)"', content, re.MULTILINE)
    by_heading = re.findall(r'^###\s+(DTS\w+)', content, re.MULTILINE)
    return set(by_field) | set(by_heading)


def format_entries(group):
    """将分组数据格式化为 md 条目列表。

    每个 sub_issue 独立成一条完整条目，共享 dts_number 和 description。
    group: {"B": "DTS...", "C": "...", "sub_issues": [{D,F,G,H}, ...]}
    返回: [str, ...]  每个元素是一个完整的 ### 条目
    """
    dts = group["B"]
    desc = group["C"] or "N/A"
    subs = group["sub_issues"] or [{}]

    entries = []
    for sub in subs:
        aclnn = sub.get("D", "").strip() or "N/A"
        lines = [
            f"### {dts} - {aclnn}",
            f'- dts_number: "{dts}"',
            f'- description: "{desc}"',
        ]
        for col in SUB_ISSUE_COLS:
            label = COL_LABELS[col]
            value = sub.get(col, "").strip() or "N/A"
            lines.append(f'- {label}: "{value}"')
        entries.append("\n".join(lines))

    return entries


MD_HEADER = """# Memory Consistency Issue Cases

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

"""


def main():
    parser = argparse.ArgumentParser(
        description="从 xlsx 导入显存一致性问题到 md 文件",
    )
    parser.add_argument("xlsx", help="xlsx 问题记录表路径")
    parser.add_argument(
        "--md", default=DEFAULT_MD_PATH,
        help=f"目标 md 文件路径 (默认: references/memory_consistency_issue_cases.md)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="仅预览将要写入的内容，不实际修改文件",
    )
    args = parser.parse_args()

    md_path = os.path.abspath(args.md)

    if not os.path.isfile(args.xlsx):
        print(f"错误: xlsx 文件不存在: {args.xlsx}", file=sys.stderr)
        sys.exit(1)

    groups = parse_xlsx(args.xlsx)
    if not groups:
        print("xlsx 中未找到有效数据行（B 列非空的行）。")
        return

    total_subs = sum(len(g["sub_issues"]) for g in groups)
    print(f"xlsx 中共读取到 {len(groups)} 个 DTS（{total_subs} 条子问题）。")

    existing_dts = load_existing_dts_numbers(md_path)
    if existing_dts:
        print(f"md 文件中已存在 {len(existing_dts)} 个 DTS。")

    new_groups = [g for g in groups if g["B"] not in existing_dts]
    skipped = len(groups) - len(new_groups)
    if skipped > 0:
        print(f"跳过 {skipped} 个已存在的 DTS。")

    if not new_groups:
        print("没有新记录需要写入。")
        return

    entries = []
    for g in new_groups:
        entries.extend(format_entries(g))
    new_content = "\n\n".join(entries) + "\n"

    if args.dry_run:
        print(f"\n--- 将要追加的 {len(new_groups)} 个 DTS (dry-run) ---\n")
        print(new_content)
        return

    if not os.path.isfile(md_path):
        os.makedirs(os.path.dirname(md_path), exist_ok=True)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(MD_HEADER)
            f.write(new_content)
        print(f"已创建 md 文件并写入 {len(new_groups)} 个 DTS: {md_path}")
    else:
        with open(md_path, "a", encoding="utf-8") as f:
            f.write("\n" + new_content)
        print(f"已追加 {len(new_groups)} 个 DTS 到: {md_path}")


if __name__ == "__main__":
    main()
