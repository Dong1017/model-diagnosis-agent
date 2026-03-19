#!/usr/bin/env python3
"""One-click regression eval pipeline: execute → grade → aggregate.

Combines run_regression_eval.py, grade_regression.py, and
benchmark aggregation into a single invocation.

Usage:
    python tools/run_pipeline.py --sample 5 --verbose
    python tools/run_pipeline.py --ids reg_003,reg_005 --lang en
    python tools/run_pipeline.py --all --filter-difficulty L1
"""

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
SKILL_DIR = TOOLS_DIR.parent

sys.path.insert(0, str(TOOLS_DIR))

from run_regression_eval import (  # noqa: E402
    SKILL_MD,
    _detect_claude_model,
    check_claude_cli,
    find_project_root,
    run_single_eval,
    save_eval_result,
    select_evals,
    setup_workspace,
)
from grade_regression import grade_workspace, load_evals_map  # noqa: E402

from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: E402


def _calculate_stats(values):
    if not values:
        return {"mean": 0.0, "stddev": 0.0, "min": 0.0, "max": 0.0}
    n = len(values)
    mean = sum(values) / n
    stddev = math.sqrt(sum((x - mean) ** 2 for x in values) / (n - 1)) if n > 1 else 0.0
    return {
        "mean": round(mean, 4), "stddev": round(stddev, 4),
        "min": round(min(values), 4), "max": round(max(values), 4),
    }


def aggregate_workspace(workspace: Path, skill_name: str) -> dict:
    """Lightweight aggregation that reads grading.json files in workspace."""
    run_meta_path = workspace / "run_metadata.json"
    run_meta = {}
    if run_meta_path.exists():
        with open(run_meta_path, encoding="utf-8") as f:
            run_meta = json.load(f)

    runs = []
    for eval_dir in sorted(workspace.glob("eval-*")):
        meta_path = eval_dir / "eval_metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path, encoding="utf-8") as f:
                    eval_id = json.load(f).get("eval_id", eval_dir.name)
            except Exception:
                eval_id = eval_dir.name
        else:
            eval_id = eval_dir.name.replace("eval-", "")

        for config_dir in sorted(eval_dir.iterdir()):
            if not config_dir.is_dir():
                continue
            for run_dir in sorted(config_dir.glob("run-*")):
                grading_file = run_dir / "grading.json"
                if not grading_file.exists():
                    continue
                with open(grading_file, encoding="utf-8") as f:
                    grading = json.load(f)
                run_number = int(run_dir.name.split("-")[1])
                summary = grading.get("summary", {})
                runs.append({
                    "eval_id": eval_id,
                    "configuration": config_dir.name,
                    "run_number": run_number,
                    "result": {
                        "pass_rate": summary.get("pass_rate", 0),
                        "passed": summary.get("passed", 0),
                        "failed": summary.get("failed", 0),
                        "total": summary.get("total", 0),
                        "time_seconds": 0.0, "tokens": 0,
                        "tool_calls": 0, "errors": 0,
                    },
                    "expectations": grading.get("expectations", []),
                    "notes": [],
                })

    pass_rates = [r["result"]["pass_rate"] for r in runs]
    eval_ids = sorted(set(r["eval_id"] for r in runs))
    configs = sorted(set(r["configuration"] for r in runs))

    run_summary = {}
    for cfg in configs:
        cfg_rates = [r["result"]["pass_rate"] for r in runs if r["configuration"] == cfg]
        run_summary[cfg] = {"pass_rate": _calculate_stats(cfg_rates)}

    if len(configs) >= 2:
        delta = (run_summary[configs[0]]["pass_rate"]["mean"]
                 - run_summary[configs[1]]["pass_rate"]["mean"])
    elif configs:
        delta = run_summary[configs[0]]["pass_rate"]["mean"]
    else:
        delta = 0.0
    run_summary["delta"] = {"pass_rate": f"{delta:+.2f}"}

    return {
        "metadata": {
            "skill_name": skill_name,
            "skill_path": str(SKILL_DIR),
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model": run_meta.get("model", "unknown"),
            "lang": run_meta.get("lang", "unknown"),
            "evals_run": eval_ids,
            "total_completed": run_meta.get("completed", len(runs)),
            "total_failed": run_meta.get("failed", 0),
        },
        "runs": runs,
        "run_summary": run_summary,
    }


def _load_eval_grading_details(workspace: Path) -> dict:
    """Load full grading details from each eval's grading.json."""
    details = {}
    for eval_dir in sorted(workspace.glob("eval-*")):
        eval_id = eval_dir.name.replace("eval-", "")
        for config_dir in sorted(eval_dir.iterdir()):
            if not config_dir.is_dir():
                continue
            for run_dir in sorted(config_dir.glob("run-*")):
                grading_file = run_dir / "grading.json"
                if grading_file.exists():
                    with open(grading_file, encoding="utf-8") as f:
                        details[eval_id] = json.load(f)
    return details


def _load_eval_sources(workspace: Path) -> dict:
    """Load eval source titles from eval_metadata.json."""
    sources = {}
    for eval_dir in sorted(workspace.glob("eval-*")):
        eval_id = eval_dir.name.replace("eval-", "")
        meta_path = eval_dir / "eval_metadata.json"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            sources[eval_id] = {
                "title": meta.get("source", {}).get("showcase_title", ""),
                "difficulty": meta.get("metadata", {}).get("difficulty", ""),
                "failure_type": meta.get("metadata", {}).get("failure_type", ""),
                "lang": meta.get("lang", ""),
                "elapsed": meta.get("elapsed_seconds", 0),
            }
    return sources


def generate_markdown(benchmark: dict, workspace: Path) -> str:
    meta = benchmark["metadata"]
    rs = benchmark["run_summary"]
    configs = [k for k in rs if k != "delta"]

    grading_details = _load_eval_grading_details(workspace)
    eval_sources = _load_eval_sources(workspace)

    lines = []

    # --- Header ---
    lines.append(f"# Regression Benchmark Report")
    lines.append("")
    lines.append(f"**Skill**: `{meta['skill_name']}`")
    lines.append(f"**Date**: {meta['timestamp']}")
    lines.append(f"**Model**: `{meta.get('model', 'unknown')}`")
    lines.append(f"**Language**: {meta.get('lang', 'unknown')}")
    lines.append(f"**Evals Run**: {meta.get('total_completed', len(benchmark['runs']))}"
                 f" completed, {meta.get('total_failed', 0)} failed")
    lines.append("")

    # --- Overall Pass Rate ---
    if configs:
        cfg = configs[0]
        pr = rs[cfg]["pass_rate"]
        lines.append(f"## Overall Result")
        lines.append("")
        total_pass = sum(1 for r in benchmark["runs"]
                         if all(e["passed"] for e in r["expectations"]))
        total_runs = len(benchmark["runs"])
        lines.append(f"**Pass Rate**: **{pr['mean']*100:.1f}%** "
                      f"({total_pass}/{total_runs} evals fully passed)")
        lines.append("")

    # --- Grading Criteria ---
    lines.append("## Grading Criteria (5-Layer Assertions)")
    lines.append("")
    lines.append("Each eval is graded across 5 layers. An eval passes if its weighted score >= 0.70.")
    lines.append("")
    lines.append("| Layer | Weight | What It Checks | Method |")
    lines.append("|-------|--------|---------------|--------|")
    lines.append("| **L1**: Error Identification | 20% | "
                 "Agent correctly identifies key error codes/keywords and backend | "
                 "Keyword match (any/all/min-N) + backend string match |")
    lines.append("| **L2**: Classification | 15% | "
                 "Agent classifies the failure type (platform/scripts/framework/cann) | "
                 "Enum string match in output |")
    lines.append("| **L3**: Root Cause Analysis | 30% | "
                 "Agent identifies the correct root cause with key technical terms | "
                 "Keyword partial match (min N of M keywords) |")
    lines.append("| **L4**: Solution Quality | 25% | "
                 "Agent provides actionable solution with correct commands/steps | "
                 "Keyword partial match (min N of M keywords) |")
    lines.append("| **L5**: Process Compliance | 10% | "
                 "Agent references failure-showcase and asks user to verify | "
                 "String match for showcase ref + validation question |")
    lines.append("")

    # --- Layer Averages ---
    lines.append("## Layer Score Averages")
    lines.append("")
    layer_avgs = {}
    if benchmark["runs"]:
        layer_sums = {"L1": 0, "L2": 0, "L3": 0, "L4": 0, "L5": 0}
        for eid, detail in grading_details.items():
            rd = detail.get("regression_detail", {})
            for lk in layer_sums:
                layer_sums[lk] += rd.get("layers", {}).get(lk, 0)
        n = len(grading_details) or 1
        layer_avgs = {k: v / n for k, v in layer_sums.items()}

    lines.append("| Layer | Avg Score | Status |")
    lines.append("|-------|-----------|--------|")
    for lk in ["L1", "L2", "L3", "L4", "L5"]:
        avg = layer_avgs.get(lk, 0)
        status = "OK" if avg >= 0.8 else ("WARN" if avg >= 0.5 else "LOW")
        bar = "█" * int(avg * 10) + "░" * (10 - int(avg * 10))
        lines.append(f"| {lk} | {avg:.2f} {bar} | {status} |")
    lines.append("")

    # --- Per-Eval Results Table ---
    lines.append("## Per-Eval Results")
    lines.append("")
    lines.append("| Eval ID | Source | Difficulty | Type | L1 | L2 | L3 | L4 | L5 | Score | Status | Time |")
    lines.append("|---------|--------|-----------|------|----|----|----|----|-----|-------|--------|------|")
    for run in benchmark["runs"]:
        eid = run["eval_id"]
        src = eval_sources.get(eid, {})
        title = src.get("title", "")[:30]
        diff = src.get("difficulty", "")
        ftype = src.get("failure_type", "")
        elapsed = src.get("elapsed", 0)

        exps = run["expectations"]
        layer_cells = []
        for exp in exps:
            layer_cells.append("PASS" if exp["passed"] else "**FAIL**")
        while len(layer_cells) < 5:
            layer_cells.append("-")

        detail = grading_details.get(eid, {})
        total_score = detail.get("regression_detail", {}).get("total_score", 0)
        is_pass = all(e["passed"] for e in exps)
        status = "PASS" if is_pass else "**FAIL**"

        lines.append(
            f"| {eid} | {title} | {diff} | {ftype} | "
            f"{' | '.join(layer_cells)} | {total_score:.2f} | {status} | {elapsed:.0f}s |"
        )
    lines.append("")

    # --- Failed Layer Details ---
    failed_evals = []
    for run in benchmark["runs"]:
        eid = run["eval_id"]
        failed_layers = [e for e in run["expectations"] if not e["passed"]]
        if failed_layers:
            failed_evals.append((eid, failed_layers))

    if failed_evals:
        lines.append("## Failed Layer Details")
        lines.append("")
        for eid, failed_layers in failed_evals:
            src = eval_sources.get(eid, {})
            lines.append(f"### {eid}: {src.get('title', '')}")
            lines.append("")
            for fl in failed_layers:
                lines.append(f"- **{fl['text']}**")
                lines.append(f"  - Evidence: {fl['evidence']}")
            lines.append("")

    # --- Footer ---
    lines.append("---")
    lines.append("")
    lines.append(f"*Generated by `run_pipeline.py` at {meta['timestamp']}*")

    return "\n".join(lines)


def run_phase_execute(args, evals_data, skill_name, skill_description,
                      skill_content, workspace):
    """Phase 1: Execute evals via claude -p."""
    check_claude_cli()
    all_evals = evals_data["evals"]
    selected = select_evals(all_evals, args)
    if not selected:
        print("No evals matched selection criteria.", file=sys.stderr)
        sys.exit(1)

    project_root = find_project_root()
    langs = ["zh", "en"] if args.lang == "both" else [args.lang]
    tasks = [(e, lang) for e in selected for lang in langs]

    print(f"\n{'='*60}", file=sys.stderr)
    print(f" Phase 1: EXECUTE  ({len(tasks)} tasks = "
          f"{len(selected)} evals x {len(langs)} lang)", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    completed = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        future_map = {}
        for eval_entry, lang in tasks:
            prompt_text = eval_entry["prompt"]
            if isinstance(prompt_text, dict):
                prompt_text = prompt_text.get(lang, prompt_text.get("en", ""))
            future = executor.submit(
                run_single_eval,
                eval_entry["id"],
                prompt_text,
                skill_name,
                skill_description,
                skill_content,
                str(SKILL_DIR),
                args.timeout,
                str(project_root),
                args.model,
            )
            future_map[future] = (eval_entry, lang)

        for future in as_completed(future_map):
            eval_entry, lang = future_map[future]
            try:
                result = future.result()
                save_eval_result(workspace, eval_entry, result, lang)
                completed += 1
                status = "OK" if not result["error"] else f"ERR: {result['error']}"
                if args.verbose:
                    print(f"  [{completed}/{len(tasks)}] {eval_entry['id']} ({lang}): "
                          f"{status} ({result['elapsed_seconds']}s)", file=sys.stderr)
            except Exception as e:
                failed += 1
                print(f"  FAILED {eval_entry['id']} ({lang}): {e}", file=sys.stderr)

    run_meta = {
        "skill_name": skill_name,
        "evals_file": str(args.evals),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": args.model or _detect_claude_model() or "unknown",
        "lang": args.lang,
        "total_tasks": len(tasks), "completed": completed, "failed": failed,
    }
    (workspace / "run_metadata.json").write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  Execute done: {completed}/{len(tasks)} completed, "
          f"{failed} failed", file=sys.stderr)
    return completed


def run_phase_grade(args, workspace, evals_path):
    """Phase 2: Grade all transcripts."""
    print(f"\n{'='*60}", file=sys.stderr)
    print(f" Phase 2: GRADE", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    evals_map = load_evals_map(evals_path)
    summary = grade_workspace(workspace, evals_map, args.verbose)
    if not summary:
        print("  No evals graded.", file=sys.stderr)
        return None

    summary_path = workspace / "grading_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  Graded: {summary['graded']}/{summary['total_runs']}", file=sys.stderr)
    print(f"  Passed: {summary['passed']}/{summary['graded']} "
          f"({summary['pass_rate']*100:.1f}%)", file=sys.stderr)
    if summary.get("layer_averages"):
        avgs = summary["layer_averages"]
        print(f"  Layer avgs: " + "  ".join(
            f"{k}={v:.2f}" for k, v in avgs.items()), file=sys.stderr)
    return summary


def run_phase_aggregate(workspace, skill_name):
    """Phase 3: Aggregate into benchmark."""
    print(f"\n{'='*60}", file=sys.stderr)
    print(f" Phase 3: AGGREGATE", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    benchmark = aggregate_workspace(workspace, skill_name)

    bench_json = workspace / "benchmark.json"
    bench_json.write_text(
        json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    bench_md = workspace / "benchmark.md"
    bench_md.write_text(generate_markdown(benchmark, workspace), encoding="utf-8")

    print(f"  Generated: {bench_json}", file=sys.stderr)
    print(f"  Generated: {bench_md}", file=sys.stderr)
    return benchmark


def main():
    parser = argparse.ArgumentParser(
        description="One-click regression eval pipeline: execute → grade → aggregate"
    )
    sel = parser.add_argument_group("eval selection (pick one)")
    sel.add_argument("--ids", default=None, help="Comma-separated eval IDs")
    sel.add_argument("--all", action="store_true", help="Run all evals")
    sel.add_argument("--sample", type=int, default=None, help="Random sample N evals")

    flt = parser.add_argument_group("filters")
    flt.add_argument("--filter-type", choices=["seed", "observed"], default=None)
    flt.add_argument("--filter-difficulty", choices=["L1", "L2"], default=None)

    exe = parser.add_argument_group("execution")
    exe.add_argument("--evals", default=str(SKILL_DIR / "evals" / "regression-evals.json"),
                     help="Path to regression-evals.json")
    exe.add_argument("--lang", choices=["zh", "en", "both"], default="zh")
    exe.add_argument("--num-workers", type=int, default=4)
    exe.add_argument("--timeout", type=int, default=600, help="Seconds per eval")
    exe.add_argument("--model", default=None, help="Claude model override")
    exe.add_argument("--workspace", type=Path, default=None,
                     help="Workspace base dir (default: <skill>/workspace)")

    ctl = parser.add_argument_group("pipeline control")
    ctl.add_argument("--skip-execute", action="store_true",
                     help="Skip execution, grade an existing workspace")
    ctl.add_argument("--existing-workspace", type=Path, default=None,
                     help="Grade+aggregate an existing workspace (implies --skip-execute)")
    ctl.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if args.existing_workspace:
        args.skip_execute = True
        workspace = args.existing_workspace
        if not workspace.exists():
            print(f"Error: workspace not found: {workspace}", file=sys.stderr)
            sys.exit(1)
    else:
        if not args.skip_execute and not args.ids and not args.all and not args.sample:
            parser.error("Specify --ids, --all, or --sample N (or --existing-workspace)")
        workspace_base = args.workspace or (SKILL_DIR / "workspace")
        workspace = setup_workspace(workspace_base)

    evals_path = Path(args.evals)
    if not evals_path.exists():
        print(f"Error: {evals_path} not found", file=sys.stderr)
        sys.exit(1)

    skill_name = "pta-failure-analyze"
    skill_description = ""
    skill_content = ""
    if SKILL_MD.exists():
        skill_content = SKILL_MD.read_text(encoding="utf-8")
        lines = skill_content.split("\n")
        if lines[0].strip() == "---":
            for line in lines[1:]:
                if line.strip() == "---":
                    break
                if line.startswith("name:"):
                    skill_name = line.split(":", 1)[1].strip().strip("\"'")
                elif line.startswith("description:"):
                    skill_description = line.split(":", 1)[1].strip().strip("\"'")

    print(f"Workspace: {workspace}", file=sys.stderr)

    if not args.skip_execute:
        with open(evals_path, encoding="utf-8") as f:
            evals_data = json.load(f)
        run_phase_execute(args, evals_data, skill_name, skill_description,
                          skill_content, workspace)

    summary = run_phase_grade(args, workspace, evals_path)

    benchmark = run_phase_aggregate(workspace, skill_name)

    print(f"\n{'='*60}", file=sys.stderr)
    print(f" PIPELINE COMPLETE", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"  Workspace : {workspace}", file=sys.stderr)
    if summary:
        print(f"  Pass rate : {summary['pass_rate']*100:.1f}% "
              f"({summary['passed']}/{summary['graded']})", file=sys.stderr)
    print(f"  Reports   : benchmark.json, benchmark.md, grading_summary.json",
          file=sys.stderr)
    print(f"\n  cat \"{workspace / 'benchmark.md'}\"", file=sys.stderr)

    output = {
        "workspace": str(workspace),
        "grading_summary": summary,
        "benchmark_json": str(workspace / "benchmark.json"),
        "benchmark_md": str(workspace / "benchmark.md"),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
