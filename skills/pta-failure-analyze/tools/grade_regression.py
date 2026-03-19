#!/usr/bin/env python3
"""Grade regression eval transcripts against 5-layer assertions.

Reads transcripts from a workspace directory and regression-evals.json,
applies deterministic code-based grading, and outputs grading.json files
compatible with the skill-creator aggregate_benchmark.py / generate_review.py.

Usage:
    python tools/grade_regression.py --workspace workspace/regression-run-2026-03-18 --evals evals/regression-evals.json
    python tools/grade_regression.py --workspace workspace/regression-run-2026-03-18 --evals evals/regression-evals.json --verbose
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


BACKEND_SYNONYMS = {
    "ascend": ["ascend", "npu", "910", "davinci", "cann"],
}


def grade_layer1(transcript_lower: str, assertions: dict) -> dict:
    """Layer 1: Error identification — keyword match + backend match."""
    l1 = assertions["layer1_identification"]
    keywords = l1["error_keywords"]
    found_kw = [kw for kw in keywords if kw.lower() in transcript_lower]

    match_mode = l1["error_keywords_match"]
    if "_" in str(match_mode):
        min_match = int(str(match_mode).split("_")[1])
    elif match_mode == "all":
        min_match = len(keywords)
    else:
        min_match = 1

    kw_pass = len(found_kw) >= min_match

    backend_expected = l1["backend_expected"]
    backend_aliases = l1.get("backend_aliases", [])
    candidates = [backend_expected.lower()] + [a.lower() for a in backend_aliases]
    candidates += BACKEND_SYNONYMS.get(backend_expected.lower(), [])
    candidates = list(dict.fromkeys(candidates))
    backend_pass = any(c in transcript_lower for c in candidates)

    score = (1.0 if kw_pass else 0.0) * 0.7 + (1.0 if backend_pass else 0.0) * 0.3

    return {
        "score": round(score, 3),
        "keyword_pass": kw_pass,
        "keywords_found": found_kw,
        "keywords_expected": keywords,
        "min_match": min_match,
        "backend_pass": backend_pass,
        "backend_expected": backend_expected,
    }


def grade_layer2(transcript_lower: str, assertions: dict) -> dict:
    """Layer 2: Classification — failure_type presence (with alias support)."""
    l2 = assertions["layer2_classification"]
    expected_type = l2["failure_type_expected"]
    aliases = [a.lower() for a in l2.get("failure_type_aliases", [])]
    candidates = [expected_type.lower()] + aliases
    type_pass = any(c in transcript_lower for c in candidates)
    matched = next((c for c in candidates if c in transcript_lower), None)

    quick_route = l2.get("quick_route_level", "")
    route_pass = quick_route.lower() in transcript_lower if quick_route else True

    score = 1.0 if type_pass else 0.0

    return {
        "score": round(score, 3),
        "failure_type_pass": type_pass,
        "failure_type_expected": expected_type,
        "failure_type_matched": matched,
        "quick_route_pass": route_pass,
    }


def grade_layer3(transcript_lower: str, assertions: dict) -> dict:
    """Layer 3: Root cause — keyword partial match."""
    l3 = assertions["layer3_root_cause"]
    keywords = l3["keywords"]
    min_match = l3["keywords_match_min"]
    found = [kw for kw in keywords if kw.lower() in transcript_lower]

    score = min(1.0, len(found) / max(min_match, 1))

    return {
        "score": round(score, 3),
        "keywords_found": found,
        "keywords_expected": keywords,
        "min_match": min_match,
        "semantic_rubric": l3.get("semantic_rubric", ""),
    }


def grade_layer4(transcript_lower: str, assertions: dict) -> dict:
    """Layer 4: Solution — keyword partial match."""
    l4 = assertions["layer4_solution"]
    keywords = l4["keywords"]
    min_match = l4["keywords_match_min"]
    found = [kw for kw in keywords if kw.lower() in transcript_lower]

    score = min(1.0, len(found) / max(min_match, 1))

    return {
        "score": round(score, 3),
        "keywords_found": found,
        "keywords_expected": keywords,
        "min_match": min_match,
        "semantic_rubric": l4.get("semantic_rubric", ""),
    }


def grade_layer5(transcript: str, transcript_lower: str) -> dict:
    """Layer 5: Process compliance — showcase reference + validation question."""
    showcase_keywords = [
        "failure showcase", "failure-showcase", "failure_showcase",
        "showcase", "similar problem", "known issue", "historical",
        "common pattern", "common error", "common failure",
        "frequently seen", "typical error", "typical issue",
        "previously seen", "encountered before", "known pattern",
        "reference", "error pattern",
    ]
    showcase_ref = any(kw in transcript_lower for kw in showcase_keywords)

    validation_keywords = [
        "resolve your issue", "did this resolve", "does this help",
        "verify in your environment", "confirm", "let me know",
        "try this", "please test", "check if", "did this fix",
        "verification", "please try", "try the above",
        "work without", "work correctly",
        "resolve the", "solve the issue", "fix the",
        "still see", "still get", "still occur",
    ]
    validation_asked = any(kw in transcript_lower for kw in validation_keywords)

    score = (0.5 if showcase_ref else 0.0) + (0.5 if validation_asked else 0.0)

    return {
        "score": round(score, 3),
        "showcase_referenced": showcase_ref,
        "validation_asked": validation_asked,
    }


def grade_eval(transcript: str, eval_entry: dict) -> dict:
    """Grade a single eval transcript against its assertions.

    Returns a grading result with:
    - Per-layer scores and details
    - Weighted total score
    - Pass/fail determination
    - grading.json-compatible expectations array
    """
    assertions = eval_entry["assertions"]
    scoring = eval_entry["scoring"]
    transcript_lower = transcript.lower()

    l1 = grade_layer1(transcript_lower, assertions)
    l2 = grade_layer2(transcript_lower, assertions)
    l3 = grade_layer3(transcript_lower, assertions)
    l4 = grade_layer4(transcript_lower, assertions)
    l5 = grade_layer5(transcript, transcript_lower)

    layers = {"L1": l1["score"], "L2": l2["score"], "L3": l3["score"],
              "L4": l4["score"], "L5": l5["score"]}

    total_score = (
        layers["L1"] * scoring["layer1_weight"]
        + layers["L2"] * scoring["layer2_weight"]
        + layers["L3"] * scoring["layer3_weight"]
        + layers["L4"] * scoring["layer4_weight"]
        + layers["L5"] * scoring["layer5_weight"]
    )
    total_score = round(total_score, 4)
    passed = total_score >= scoring["pass_threshold"]

    expectations = [
        {
            "text": f"L1: Error identification (keywords={l1['min_match']}+ of {len(l1['keywords_expected'])}, backend={l1['backend_expected']})",
            "passed": l1["score"] >= 0.7,
            "evidence": f"Keywords found: {l1['keywords_found']}; backend_pass={l1['backend_pass']}",
        },
        {
            "text": f"L2: Classification (failure_type={l2['failure_type_expected']})",
            "passed": l2["score"] >= 0.5,
            "evidence": f"failure_type_pass={l2['failure_type_pass']}; quick_route_pass={l2['quick_route_pass']}",
        },
        {
            "text": f"L3: Root cause ({l3['min_match']}+ keywords)",
            "passed": l3["score"] >= 0.8,
            "evidence": f"Keywords found: {l3['keywords_found']} of {l3['keywords_expected']}",
        },
        {
            "text": f"L4: Solution ({l4['min_match']}+ keywords)",
            "passed": l4["score"] >= 0.8,
            "evidence": f"Keywords found: {l4['keywords_found']} of {l4['keywords_expected']}",
        },
        {
            "text": "L5: Process compliance (showcase ref + validation question)",
            "passed": l5["score"] >= 0.5,
            "evidence": f"showcase_ref={l5['showcase_referenced']}; validation_asked={l5['validation_asked']}",
        },
    ]

    exp_passed = sum(1 for e in expectations if e["passed"])
    exp_total = len(expectations)

    return {
        "expectations": expectations,
        "summary": {
            "passed": exp_passed,
            "failed": exp_total - exp_passed,
            "total": exp_total,
            "pass_rate": round(exp_passed / exp_total, 4) if exp_total else 0,
        },
        "regression_detail": {
            "layers": layers,
            "layer_details": {
                "L1": l1, "L2": l2, "L3": l3, "L4": l4, "L5": l5,
            },
            "total_score": total_score,
            "pass_threshold": scoring["pass_threshold"],
            "passed": passed,
        },
    }


def load_evals_map(evals_path: Path) -> dict:
    """Load evals and return a dict keyed by eval_id."""
    with open(evals_path, encoding="utf-8") as f:
        data = json.load(f)
    return {e["id"]: e for e in data["evals"]}


def _extract_agent_response(full_text: str) -> str:
    """Extract only the '## Agent Response' section from a transcript.md.

    Only the agent response should be graded to avoid false-positive
    keyword matches from the prompt itself.
    """
    marker = "## Agent Response"
    idx = full_text.find(marker)
    if idx != -1:
        return full_text[idx + len(marker):]
    return full_text


def find_transcript(eval_run_dir: Path) -> Optional[str]:
    """Find and read the agent response from a transcript in an eval run dir."""
    transcript_path = eval_run_dir / "transcript.md"
    if transcript_path.exists():
        full = transcript_path.read_text(encoding="utf-8")
        return _extract_agent_response(full)

    for md_file in eval_run_dir.glob("*.md"):
        full = md_file.read_text(encoding="utf-8")
        return _extract_agent_response(full)

    return None


def grade_workspace(workspace: Path, evals_map: dict, verbose: bool = False) -> dict:
    """Grade all transcripts in a workspace directory."""
    eval_dirs = sorted(workspace.glob("eval-*"))
    if not eval_dirs:
        print(f"No eval-* directories found in {workspace}", file=sys.stderr)
        return {}

    total = 0
    graded = 0
    passed_count = 0
    layer_totals = {"L1": 0.0, "L2": 0.0, "L3": 0.0, "L4": 0.0, "L5": 0.0}
    results_summary = []

    for eval_dir in eval_dirs:
        eval_id = eval_dir.name.replace("eval-", "")

        if eval_id not in evals_map:
            if verbose:
                print(f"  SKIP {eval_id}: not found in evals file", file=sys.stderr)
            continue

        eval_entry = evals_map[eval_id]

        for config_dir in sorted(eval_dir.iterdir()):
            if not config_dir.is_dir():
                continue
            for run_dir in sorted(config_dir.glob("run-*")):
                total += 1
                transcript = find_transcript(run_dir)
                if not transcript:
                    if verbose:
                        print(f"  SKIP {eval_id}/{config_dir.name}/{run_dir.name}: no transcript", file=sys.stderr)
                    continue

                grading = grade_eval(transcript, eval_entry)
                graded += 1

                grading_path = run_dir / "grading.json"
                grading_path.write_text(
                    json.dumps(grading, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                detail = grading["regression_detail"]
                is_passed = detail["passed"]
                if is_passed:
                    passed_count += 1

                for layer_key in layer_totals:
                    layer_totals[layer_key] += detail["layers"][layer_key]

                result_line = {
                    "eval_id": eval_id,
                    "total_score": detail["total_score"],
                    "passed": is_passed,
                    "layers": detail["layers"],
                }
                results_summary.append(result_line)

                if verbose:
                    status = "PASS" if is_passed else "FAIL"
                    score = detail["total_score"]
                    layers_str = " ".join(
                        f"L{i}={detail['layers'][f'L{i}']:.2f}" for i in range(1, 6)
                    )
                    print(
                        f"  [{status}] {eval_id}: {score:.3f} ({layers_str})",
                        file=sys.stderr,
                    )

    layer_avgs = {}
    if graded > 0:
        layer_avgs = {k: round(v / graded, 4) for k, v in layer_totals.items()}

    summary = {
        "total_runs": total,
        "graded": graded,
        "passed": passed_count,
        "failed": graded - passed_count,
        "pass_rate": round(passed_count / graded, 4) if graded else 0,
        "layer_averages": layer_avgs,
        "results": results_summary,
    }

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Grade regression eval transcripts against 5-layer assertions"
    )
    parser.add_argument("--workspace", required=True, type=Path,
                        help="Workspace directory containing eval-* dirs")
    parser.add_argument("--evals", required=True, type=Path,
                        help="Path to regression-evals.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not args.workspace.exists():
        print(f"Error: workspace not found: {args.workspace}", file=sys.stderr)
        sys.exit(1)

    if not args.evals.exists():
        print(f"Error: evals file not found: {args.evals}", file=sys.stderr)
        sys.exit(1)

    evals_map = load_evals_map(args.evals)
    print(f"Loaded {len(evals_map)} eval definitions", file=sys.stderr)

    summary = grade_workspace(args.workspace, evals_map, args.verbose)

    if not summary:
        print("No evals graded", file=sys.stderr)
        sys.exit(1)

    summary_path = args.workspace / "grading_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nGrading complete:", file=sys.stderr)
    print(f"  Graded: {summary['graded']}/{summary['total_runs']}", file=sys.stderr)
    print(f"  Passed: {summary['passed']}/{summary['graded']} "
          f"({summary['pass_rate']*100:.1f}%)", file=sys.stderr)
    if summary.get("layer_averages"):
        print(f"  Layer averages:", file=sys.stderr)
        for layer, avg in summary["layer_averages"].items():
            print(f"    {layer}: {avg:.3f}", file=sys.stderr)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
