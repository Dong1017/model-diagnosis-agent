#!/usr/bin/env python3
"""Run regression evaluation for pta-failure-analyze skill.

Executes regression evals by sending each prompt to Claude via `claude -p`
with the skill injected, then collects the full transcript for grading.

Usage:
    python tools/run_regression_eval.py --evals evals/regression-evals.json --all
    python tools/run_regression_eval.py --evals evals/regression-evals.json --ids reg_003,reg_005
    python tools/run_regression_eval.py --evals evals/regression-evals.json --sample 5 --lang zh
"""

import argparse
import json
import os
import platform
import random
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

IS_WINDOWS = platform.system() == "Windows"


SKILL_DIR = Path(__file__).resolve().parent.parent
SKILL_MD = SKILL_DIR / "SKILL.md"


def _resolve_claude_path() -> str:
    """Resolve the full path to the claude CLI.

    On Windows, npm installs global packages as .cmd wrappers.
    subprocess.Popen can't execute .cmd without shell=True,
    so we resolve the full path via shutil.which() once.
    """
    path = shutil.which("claude")
    if path is None:
        return ""
    return path


CLAUDE_PATH = _resolve_claude_path()


def check_claude_cli():
    """Check if claude CLI is available on PATH. Exit with clear message if not."""
    if not CLAUDE_PATH:
        print(
            "Error: 'claude' CLI not found.\n"
            "\n"
            "The executor requires Claude Code CLI (claude -p) to run evals.\n"
            "Install it via:  npm install -g @anthropic-ai/claude-code\n"
            "Docs: https://docs.anthropic.com/en/docs/claude-cli\n"
            "\n"
            "Alternatively, use Cursor Task subagents to execute evals manually,\n"
            "then grade with:  python tools/grade_regression.py --workspace <path> ...",
            file=sys.stderr,
        )
        sys.exit(1)

EXECUTOR_PROMPT_TEMPLATE = """\
You are a torch_npu / PyTorch Ascend failure diagnosis assistant. Follow the diagnostic workflow below.

{skill_content}

---
USER PROBLEM:

{eval_prompt}

---

Instructions:
1. Follow Stage 0 → Stage 1 → Stage 2 from the skill above
2. If you need reference files, try reading them from: {skill_dir}/references/
3. Search failure-showcase.md at: {skill_dir}/references/failure-showcase.md
4. Do NOT execute Stage 3 (do not modify any files)
5. Provide complete analysis with: failure type, root cause, and solution
6. End with a validation question asking the user to verify
"""


def _drain_pipe(pipe, chunks: list):
    """Read all data from a pipe into chunks list. Used as thread target."""
    while True:
        data = pipe.read(8192)
        if not data:
            break
        chunks.append(data)


def _kill_process_tree(pid: int):
    """Kill a process and all its children. Works on both Windows and Unix."""
    if IS_WINDOWS:
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    else:
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def _parse_claude_output(stdout_text: str) -> str:
    """Parse claude -p output and extract the assistant's response.

    Supports two output formats:
    - Stream-JSON mode (--output-format stream-json): newline-delimited JSON events
    - JSON mode (--output-format json): single JSON object with "result" key
    - Plain text fallback
    """
    stdout_text = stdout_text.strip()
    if not stdout_text:
        return ""

    lines = stdout_text.split("\n")
    if len(lines) > 1 or (len(lines) == 1 and not stdout_text.startswith("{")):
        transcript = ""
        parsed_any = False
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            parsed_any = True

            if event.get("type") == "assistant":
                message = event.get("message", {})
                for block in message.get("content", []):
                    if block.get("type") == "text":
                        transcript += block.get("text", "")
                    elif block.get("type") == "tool_use":
                        tool_name = block.get("name", "")
                        tool_input = block.get("input", {})
                        transcript += (
                            f"\n[Tool Call: {tool_name}"
                            f"({json.dumps(tool_input, ensure_ascii=False)[:200]})]\n"
                        )
            elif event.get("type") == "result":
                result_text = event.get("result", "")
                if result_text:
                    transcript += "\n" + result_text

        if parsed_any and transcript:
            return transcript

    try:
        data = json.loads(stdout_text)
        if isinstance(data, dict):
            result = data.get("result", "")
            if isinstance(result, str) and result:
                return result
        return ""
    except json.JSONDecodeError:
        pass

    return stdout_text


def find_project_root() -> Path:
    """Find the project root by walking up from cwd looking for .claude/."""
    current = Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / ".claude").is_dir():
            return parent
    return current


def _detect_claude_model() -> str:
    """Detect the configured Claude model from env vars and settings file.

    Priority: ANTHROPIC_MODEL env var > ~/.claude/settings.json env.ANTHROPIC_MODEL
    """
    model = os.environ.get("ANTHROPIC_MODEL", "")
    if model:
        return model

    settings_path = Path.home() / ".claude" / "settings.json"
    try:
        if settings_path.exists():
            data = json.loads(settings_path.read_text(encoding="utf-8"))
            model = data.get("env", {}).get("ANTHROPIC_MODEL", "")
            if model:
                return model
    except Exception:
        pass

    return ""


def run_single_eval(
    eval_id: str,
    prompt: str,
    skill_name: str,
    skill_description: str,
    skill_content: str,
    skill_dir: str,
    timeout: int,
    project_root: str,
    model: Optional[str] = None,
) -> dict:
    """Run a single eval and return the full transcript."""
    unique_id = uuid.uuid4().hex[:8]
    clean_name = f"{skill_name}-skill-{unique_id}"
    project_commands_dir = Path(project_root) / ".claude" / "commands"
    command_file = project_commands_dir / f"{clean_name}.md"

    executor_prompt = EXECUTOR_PROMPT_TEMPLATE.format(
        skill_content=skill_content,
        eval_prompt=prompt,
        skill_dir=skill_dir,
    )

    start_time = time.time()
    transcript = ""
    error_msg = ""

    try:
        project_commands_dir.mkdir(parents=True, exist_ok=True)
        indented_desc = "\n  ".join(skill_description.split("\n"))
        command_content = (
            f"---\n"
            f"description: |\n"
            f"  {indented_desc}\n"
            f"---\n\n"
            f"# {skill_name}\n\n"
            f"This skill handles: {skill_description}\n"
        )
        command_file.write_text(command_content, encoding="utf-8")

        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

        cmd = [CLAUDE_PATH, "--output-format", "json", "--max-turns", "15"]
        if model:
            cmd.extend(["--model", model])

        popen_kwargs = dict(
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=project_root,
            env=env,
        )
        if IS_WINDOWS:
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

        process = subprocess.Popen(cmd, **popen_kwargs)

        try:
            stdout_bytes, _ = process.communicate(
                input=executor_prompt.encode("utf-8"),
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            _kill_process_tree(process.pid)
            stdout_bytes, _ = process.communicate(timeout=10)
            error_msg = f"Timeout after {timeout}s (partial output captured)"

        stdout_text = stdout_bytes.decode("utf-8", errors="replace")
        transcript = _parse_claude_output(stdout_text)

        if not transcript and error_msg:
            error_msg = f"Timeout after {timeout}s"
        elif not transcript and stdout_text.strip():
            transcript = stdout_text.strip()

    except Exception as e:
        error_msg = str(e)
    finally:
        if command_file.exists():
            command_file.unlink()

    elapsed = time.time() - start_time

    return {
        "eval_id": eval_id,
        "transcript": transcript,
        "error": error_msg,
        "elapsed_seconds": round(elapsed, 1),
    }


def select_evals(evals: list[dict], args) -> list[dict]:
    """Select evals based on CLI arguments."""
    selected = evals

    if args.ids:
        id_set = set(args.ids.split(","))
        selected = [e for e in selected if e["id"] in id_set]
        missing = id_set - {e["id"] for e in selected}
        if missing:
            print(f"Warning: eval IDs not found: {missing}", file=sys.stderr)

    if args.filter_type:
        selected = [e for e in selected if e["source"]["entry_type"] == args.filter_type]

    if hasattr(args, "filter_backend") and args.filter_backend:
        selected = [e for e in selected if e.get("metadata", {}).get("backend") == args.filter_backend]

    if args.filter_difficulty:
        selected = [e for e in selected if e["metadata"]["difficulty"] == args.filter_difficulty]

    if args.sample and args.sample < len(selected):
        selected = random.sample(selected, args.sample)

    return selected


def setup_workspace(base_dir: Path) -> Path:
    """Create workspace directory for this run."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    workspace = base_dir / f"regression-run-{timestamp}"
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def save_eval_result(workspace: Path, eval_entry: dict, result: dict, lang: str):
    """Save eval result to workspace in the expected directory layout."""
    eval_id = eval_entry["id"]
    eval_top = workspace / f"eval-{eval_id}"
    run_dir = eval_top / "with_skill" / "run-1"
    run_dir.mkdir(parents=True, exist_ok=True)

    outputs_dir = run_dir / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    prompt_text = eval_entry["prompt"]
    if isinstance(prompt_text, dict):
        prompt_text = prompt_text.get(lang, prompt_text.get("en", ""))

    transcript_path = run_dir / "transcript.md"
    transcript_content = f"# Eval: {eval_id}\n\n"
    transcript_content += f"**Source:** {eval_entry['source']['showcase_title']}\n"
    transcript_content += f"**Lang:** {lang}\n"
    transcript_content += f"**Elapsed:** {result['elapsed_seconds']}s\n\n"
    if result["error"]:
        transcript_content += f"**Error:** {result['error']}\n\n"
    transcript_content += "---\n\n"
    transcript_content += f"## Prompt\n\n{prompt_text}\n\n"
    transcript_content += "---\n\n"
    transcript_content += f"## Agent Response\n\n{result['transcript']}\n"
    transcript_path.write_text(transcript_content, encoding="utf-8")

    metadata = {
        "eval_id": eval_id,
        "source": eval_entry["source"],
        "lang": lang,
        "environment": eval_entry.get("environment", {}),
        "metadata": eval_entry.get("metadata", {}),
        "elapsed_seconds": result["elapsed_seconds"],
        "error": result["error"],
    }
    metadata_json = json.dumps(metadata, ensure_ascii=False, indent=2)
    (outputs_dir / "eval_metadata.json").write_text(metadata_json, encoding="utf-8")
    (eval_top / "eval_metadata.json").write_text(metadata_json, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Run regression evaluation for pta-failure-analyze skill"
    )
    parser.add_argument("--evals", required=True, help="Path to regression-evals.json")
    parser.add_argument("--ids", default=None, help="Comma-separated eval IDs (e.g. reg_003,reg_005)")
    parser.add_argument("--all", action="store_true", help="Run all evals")
    parser.add_argument("--sample", type=int, default=None, help="Random sample N evals")
    parser.add_argument("--filter-type", choices=["seed", "observed"], default=None)
    parser.add_argument("--filter-difficulty", choices=["L1", "L2"], default=None)
    parser.add_argument("--lang", choices=["zh", "en", "both"], default="zh", help="Prompt language")
    parser.add_argument("--num-workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout per eval (seconds)")
    parser.add_argument("--model", default=None, help="Model for claude -p")
    parser.add_argument("--workspace", type=Path, default=None,
                        help="Workspace base dir (default: <skill_dir>/workspace)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not args.ids and not args.all and not args.sample:
        parser.error("Specify --ids, --all, or --sample N")

    check_claude_cli()

    evals_path = Path(args.evals)
    if not evals_path.exists():
        print(f"Error: {evals_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(evals_path, encoding="utf-8") as f:
        evals_data = json.load(f)
    all_evals = evals_data["evals"]

    selected = select_evals(all_evals, args)
    if not selected:
        print("No evals matched the selection criteria", file=sys.stderr)
        sys.exit(1)

    if not SKILL_MD.exists():
        print(f"Error: SKILL.md not found at {SKILL_MD}", file=sys.stderr)
        sys.exit(1)

    skill_content = SKILL_MD.read_text(encoding="utf-8")
    lines = skill_content.split("\n")
    skill_name = ""
    skill_description = ""
    if lines[0].strip() == "---":
        for i, line in enumerate(lines[1:], 1):
            if line.strip() == "---":
                break
            if line.startswith("name:"):
                skill_name = line.split(":", 1)[1].strip().strip('"').strip("'")
            elif line.startswith("description:"):
                skill_description = line.split(":", 1)[1].strip().strip('"').strip("'")

    project_root = find_project_root()
    workspace_base = args.workspace or (SKILL_DIR / "workspace")
    workspace = setup_workspace(workspace_base)

    langs = ["zh", "en"] if args.lang == "both" else [args.lang]

    tasks = []
    for eval_entry in selected:
        for lang in langs:
            tasks.append((eval_entry, lang))

    print(f"Running {len(tasks)} eval tasks ({len(selected)} evals x {len(langs)} lang(s))", file=sys.stderr)
    print(f"Workspace: {workspace}", file=sys.stderr)
    print(f"Workers: {args.num_workers}, Timeout: {args.timeout}s", file=sys.stderr)

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
                    print(
                        f"  [{completed}/{len(tasks)}] {eval_entry['id']} ({lang}): "
                        f"{status} ({result['elapsed_seconds']}s)",
                        file=sys.stderr,
                    )
            except Exception as e:
                failed += 1
                print(f"  FAILED {eval_entry['id']} ({lang}): {e}", file=sys.stderr)

    claude_version = _detect_claude_model()

    run_meta = {
        "skill_name": skill_name,
        "evals_file": str(evals_path),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": args.model or claude_version or "unknown",
        "selection": {
            "ids": args.ids,
            "sample": args.sample,
            "filter_type": args.filter_type,
            "filter_difficulty": args.filter_difficulty,
            "lang": args.lang,
        },
        "total_tasks": len(tasks),
        "completed": completed,
        "failed": failed,
    }
    (workspace / "run_metadata.json").write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nDone: {completed}/{len(tasks)} completed, {failed} failed", file=sys.stderr)
    print(f"Workspace: {workspace}", file=sys.stderr)
    print(json.dumps({"workspace": str(workspace), **run_meta}, indent=2))


if __name__ == "__main__":
    main()
