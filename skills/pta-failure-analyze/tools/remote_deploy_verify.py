#!/usr/bin/env python3
"""Remote deploy & verify tool for torch_npu development.

Syncs local code changes to a remote Ascend server, builds, runs tests,
and reports results — all via structured JSON output for LLM consumption.

Usage:
    python tools/remote_deploy_verify.py collect --local-root /path/to/torch_npu
    python tools/remote_deploy_verify.py preflight --config deploy_context.json
    python tools/remote_deploy_verify.py sync --config deploy_context.json [--dry-run] [--files ...]
    python tools/remote_deploy_verify.py build --config deploy_context.json [--mode MODE]
    python tools/remote_deploy_verify.py install --config deploy_context.json [--wheel PATH]
    python tools/remote_deploy_verify.py run --config deploy_context.json [--command CMD]
    python tools/remote_deploy_verify.py exec --config deploy_context.json "ls -la"
    python tools/remote_deploy_verify.py sync-run --config deploy_context.json
    python tools/remote_deploy_verify.py report --config deploy_context.json
    python tools/remote_deploy_verify.py rollback --config deploy_context.json

Docker Build Mode:
    To build inside a Docker container, add 'docker' config to deploy_context.json:
    {
      "build": {
        "python_version": "3.10",
        "docker": {
          "image": "manylinux-builder:v1",
          "mounts": ["/home:/home", "/root:/root"],
          "container_name": "pta_compile_temp",
          "auto_cleanup": true
        }
      }
    }
"""

import argparse
import fnmatch
import hashlib
import io
import json
import os
import posixpath
import re
import shlex
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

TOOLS_DIR = Path(__file__).resolve().parent
SKILL_DIR = TOOLS_DIR.parent
DEFAULT_LOG_DIR = SKILL_DIR / "deploy_logs"

# ---------------------------------------------------------------------------
# Module classification rules (based on torch_npu source layout)
# ---------------------------------------------------------------------------

MODULE_RULES = [
    ("third_party/op-plugin/op_plugin/ops/opapi/*.cpp", "op-plugin", "operator_impl_opapi"),
    ("third_party/op-plugin/op_plugin/ops/aclops/*.cpp", "op-plugin", "operator_impl_aclop"),
    ("third_party/op-plugin/op_plugin/ops/custom/*.cpp", "op-plugin", "operator_impl_custom"),
    ("third_party/op-plugin/op_plugin/ops/official/*.cpp", "op-plugin", "operator_impl_official"),
    ("third_party/op-plugin/op_plugin/ops/atb/*.cpp", "op-plugin", "operator_impl_atb"),
    ("third_party/op-plugin/op_plugin/utils/*", "op-plugin", "op_utils"),
    ("third_party/op-plugin/op_plugin/config/*", "op-plugin-config", "op_config"),
    ("third_party/op-plugin/gencode.sh", "op-plugin-config", "codegen_script"),
    ("third_party/op-plugin/codegen/*", "op-plugin-config", "codegen"),
    ("third_party/op-plugin/op_plugin/python/*", "op-plugin-py", "op_python"),
    ("torch_npu/csrc/*", "torch_npu", "framework_cpp"),
    ("torch_npu/npu/*.py", "torch_npu-py", "framework_py"),
    ("codegen/*", "torch_npu", "codegen"),
    ("generate_code.sh", "torch_npu", "codegen_script"),
    ("CMakeLists.txt", "torch_npu", "build_config"),
    ("setup.py", "torch_npu", "build_config"),
    ("test/*", "test", "test_case"),
]

BUILD_MODE_MAP = {
    frozenset(["test"]): "skip",
    frozenset(["op-plugin-py"]): "skip",
    frozenset(["op-plugin-py", "test"]): "skip",
    frozenset(["torch_npu-py"]): "skip",
    frozenset(["torch_npu-py", "test"]): "skip",
    frozenset(["op-plugin"]): "incremental",
    frozenset(["op-plugin", "test"]): "incremental",
    frozenset(["op-plugin", "op-plugin-py"]): "incremental",
    frozenset(["op-plugin", "op-plugin-py", "test"]): "incremental",
    frozenset(["op-plugin-config"]): "full",
    frozenset(["torch_npu"]): "incremental",
    frozenset(["torch_npu", "test"]): "incremental",
    frozenset(["torch_npu", "op-plugin"]): "incremental",
    frozenset(["torch_npu", "op-plugin", "test"]): "incremental",
}

BUILD_COMMANDS = {
    "incremental": "cd {build_dir} && make -j$(nproc)",
    "full": "bash ci/build.sh --python={python_version}",
    "develop": "python{python_version} setup.py develop",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _json_out(data: dict):
    print(json.dumps(data, ensure_ascii=False, indent=2))


def _load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_config(path: str, config: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _ensure_log_dir(config: dict) -> Path:
    log_dir = Path(config.get("log_dir", str(DEFAULT_LOG_DIR)))
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


DEFAULT_TIMEOUTS = {
    "exec": 60,
    "build": 1800,
    "install": 300,
    "run": 300,
    "preflight": 60,
}


def _get_timeout(config: dict, operation: str, arg_timeout: int = None) -> int:
    """Get timeout for an operation with precedence: arg > config > default."""
    if arg_timeout is not None:
        return arg_timeout
    config_timeouts = config.get("timeouts", {})
    if operation in config_timeouts:
        return config_timeouts[operation]
    return DEFAULT_TIMEOUTS.get(operation, 60)


def _get_context(config: dict, context_type: str) -> dict:
    """Get context configuration for compile or verify.

    Supports two config formats:
    1. New format (separated contexts):
       {
         "contexts": {
           "compile": { "type": "docker", ... },
           "verify": { "type": "bare", ... }
         }
       }
    2. Legacy format (single exec_context):
       {
         "exec_context": { "type": "bare", ... }
       }

    For build operations, use 'compile' context.
    For run/exec operations, use 'verify' context.
    """
    contexts = config.get("contexts", {})
    if context_type in contexts:
        return contexts[context_type]
    legacy_ctx = config.get("exec_context", {})
    return legacy_ctx


def _classify_file(filepath: str) -> tuple:
    """Classify a changed file into (module, file_type) using MODULE_RULES."""
    for pattern, module, ftype in MODULE_RULES:
        if fnmatch.fnmatch(filepath, pattern):
            return module, ftype
        parts = filepath.split("/")
        for i in range(len(parts)):
            sub = "/".join(parts[i:])
            if fnmatch.fnmatch(sub, pattern):
                return module, ftype
    if filepath.endswith(".py"):
        return "python", "script"
    return "other", "unknown"


def _get_ssh_base(config: dict) -> list:
    """Build base SSH command list from config."""
    remote = config["remote"]
    cmd = ["ssh"]
    port = remote.get("port", 22)
    if port != 22:
        cmd += ["-p", str(port)]
    cmd += ["-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10"]
    cmd.append(f"{remote['user']}@{remote['host']}")
    return cmd


def _build_remote_command(config: dict, user_cmd: str, context_type: str = "verify") -> str:
    """Wrap user_cmd with exec_context (bare / docker / custom).

    Args:
        config: Configuration dict
        user_cmd: Command to execute on remote
        context_type: 'compile' or 'verify' (default: verify)
    """
    ctx = _get_context(config, context_type)
    remote_root = config["paths"]["remote_root"]

    parts = []
    env_setup = ctx.get("env_setup")
    if env_setup:
        parts.append(env_setup)
    parts.append(f"cd {remote_root}")
    parts.append(user_cmd)
    inner_cmd = " && ".join(parts)

    ctx_type = ctx.get("type", "bare")

    if ctx_type == "docker":
        container = ctx.get("docker_container", "")
        opts = ctx.get("docker_exec_opts", "") or ""
        escaped = inner_cmd.replace("'", "'\\''")
        return f"docker exec {opts} {container} bash -c '{escaped}'".strip()

    if ctx_type == "custom":
        prefix = ctx.get("shell_prefix", "") or ""
        return f"{prefix} {inner_cmd}".strip()

    return inner_cmd


def _run_ssh(config: dict, remote_cmd: str, timeout: int = 600) -> dict:
    """Execute a command on the remote host via SSH. Returns structured result.

    Uses explicit UTF-8 encoding to avoid Windows GBK codec errors when
    the remote server (Linux) returns UTF-8 output.
    """
    ssh_base = _get_ssh_base(config)
    full_cmd = ssh_base + [remote_cmd]

    start = time.time()
    try:
        proc = subprocess.run(
            full_cmd,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        elapsed = time.time() - start
        return {
            "exit_code": proc.returncode,
            "stdout": proc.stdout or "",
            "stderr": proc.stderr or "",
            "elapsed_seconds": round(elapsed, 1),
            "timed_out": False,
        }
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        return {
            "exit_code": -1,
            "stdout": "",
            "stderr": f"Command timed out after {timeout}s",
            "elapsed_seconds": round(elapsed, 1),
            "timed_out": True,
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            "exit_code": -1,
            "stdout": "",
            "stderr": str(e),
            "elapsed_seconds": round(elapsed, 1),
            "timed_out": False,
        }


def _resolve_sync_path(changed_file: str, config: dict) -> str:
    """Map a local file path to its remote destination path.

    Always produces POSIX paths (forward slashes) because the remote
    target is Linux, even when this script runs on Windows.
    """
    normalized = changed_file.replace("\\", "/")
    remote_root = config["paths"]["remote_root"]
    op_plugin_sub = config.get("paths", {}).get(
        "op_plugin_submodule_path", "third_party/op-plugin"
    )

    extra_mappings = config.get("sync", {}).get("extra_mappings", [])
    for mapping in extra_mappings:
        local_pat = mapping.get("local_pattern", "")
        if fnmatch.fnmatch(normalized, local_pat) or normalized.startswith(local_pat.rstrip("*")):
            remote_base = mapping["remote_base"]
            rel = normalized
            prefix = local_pat.rstrip("*").rstrip("/")
            if prefix and normalized.startswith(prefix + "/"):
                rel = normalized[len(prefix) + 1:]
            elif prefix and normalized.startswith(prefix):
                rel = normalized[len(prefix):]
            return posixpath.join(remote_base, rel)

    if "third_party/op-plugin" in normalized:
        return posixpath.join(remote_root, normalized)
    if normalized.startswith("op_plugin/"):
        return posixpath.join(remote_root, op_plugin_sub, normalized)
    return posixpath.join(remote_root, normalized)


def _get_sync_target_root(config: dict) -> str:
    """Determine the SSH sync target path (may differ for docker volume mounts)."""
    ctx = config.get("exec_context", {})
    if ctx.get("type") == "docker" and ctx.get("sync_target") == "host_volume":
        return ctx.get("host_mount_path", config["paths"]["remote_root"])
    return config["paths"]["remote_root"]


def _parse_build_errors(log: str) -> list:
    pattern = re.compile(
        r"^(?P<file>[^\s:]+):(?P<line>\d+):\d+: error: (?P<message>.+)$",
        re.MULTILINE,
    )
    errors = []
    for m in pattern.finditer(log):
        errors.append({
            "file": m.group("file"),
            "line": int(m.group("line")),
            "message": m.group("message"),
        })
    return errors


def _parse_build_warnings(log: str) -> list:
    pattern = re.compile(
        r"^(?P<file>[^\s:]+):(?P<line>\d+):\d+: warning: (?P<message>.+)$",
        re.MULTILINE,
    )
    return [m.group(0) for m in pattern.finditer(log)][:20]


def _parse_pytest_output(output: str) -> dict:
    """Parse the pytest summary line into a structured dict.

    Handles all standard pytest summary fields in any order, e.g.:
        = 1 failed, 2064 deselected in 10.61s =
        = 5 passed, 3 skipped, 1 xfailed, 2 warnings in 4.32s =
    """
    summary_re = re.compile(r"=+\s*(.*?)\s*in\s*[\d.]+s\s*=+")
    m = summary_re.search(output)
    if not m:
        return {"passed": 0, "failed": 0, "error": 0, "total": 0, "parse_failed": True}

    summary_text = m.group(1)
    item_re = re.compile(
        r"(\d+)\s+(passed|failed|errors?|warnings?|deselected|skipped|xfailed|xpassed|no tests ran)"
    )
    result: dict = {"passed": 0, "failed": 0, "error": 0}
    for im in item_re.finditer(summary_text):
        count = int(im.group(1))
        key = im.group(2)
        if key in ("error", "errors"):
            result["error"] = count
        elif key.startswith("warning"):
            result["warnings"] = count
        else:
            result[key] = count

    result["total"] = result.get("passed", 0) + result.get("failed", 0) + result.get("error", 0)
    return result


def _extract_failure_detail(output: str) -> dict:
    """Extract failure details from pytest output."""
    lines = output.splitlines()
    failure_lines = []
    in_failure = False
    for line in lines:
        if "FAILED" in line or "ERRORS" in line or "_ _ _ _" in line:
            in_failure = True
        if in_failure:
            failure_lines.append(line)
        if len(failure_lines) > 30:
            break

    error_type = ""
    message = ""
    for line in failure_lines:
        if "Error" in line and ": " in line:
            parts = line.strip().split(": ", 1)
            error_type = parts[0].split(".")[-1] if "." in parts[0] else parts[0]
            message = parts[1] if len(parts) > 1 else ""
            break

    return {
        "error_type": error_type.strip(),
        "message": message.strip(),
        "traceback_tail": "\n".join(failure_lines[-15:]) if failure_lines else "",
    }


def _file_md5(filepath: str) -> str:
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_collect(args):
    """Collect local git changes and classify them.

    Also generates a patch file for consistent sync to remote.
    """
    local_root = Path(args.local_root).resolve()
    if not local_root.exists():
        _json_out({"status": "error", "message": f"Local root not found: {local_root}"})
        return

    try:
        unstaged = subprocess.check_output(
            ["git", "diff", "--name-only"],
            cwd=str(local_root), text=True
        ).strip().splitlines()
    except Exception:
        unstaged = []

    try:
        staged = subprocess.check_output(
            ["git", "diff", "--name-only", "--staged"],
            cwd=str(local_root), text=True
        ).strip().splitlines()
    except Exception:
        staged = []

    all_files = sorted(set(f for f in unstaged + staged if f))

    patch_content = ""
    patch_file = None
    if all_files:
        try:
            diff = subprocess.check_output(
                ["git", "diff"], cwd=str(local_root), text=True
            )
            staged_diff = subprocess.check_output(
                ["git", "diff", "--staged"], cwd=str(local_root), text=True
            )
            patch_content = diff + staged_diff

            log_dir = _ensure_log_dir({"log_dir": str(DEFAULT_LOG_DIR)})
            patch_file = log_dir / f"changes_{_timestamp()}.patch"
            patch_file.write_text(patch_content, encoding="utf-8")
        except Exception as e:
            patch_content = ""
            patch_file = None

    if not all_files:
        _json_out({
            "status": "ok",
            "changed_files": [],
            "summary": {
                "total_files": 0,
                "modules_affected": [],
                "has_cpp_changes": False,
                "has_test_only_changes": False,
            },
            "recommendations": {
                "sync_strategy": "none",
                "sync_reason": "no changes detected",
                "build_mode": "skip",
                "build_reason": "no changes",
                "needs_compile": False,
            },
            "patch_file": None,
        })
        return

    classified = []
    modules = set()
    has_cpp = False

    for f in all_files:
        module, ftype = _classify_file(f)
        classified.append({"path": f, "module": module, "type": ftype})
        modules.add(module)
        if f.endswith((".cpp", ".h", ".hpp", ".cc", ".cxx")):
            has_cpp = True

    modules_frozen = frozenset(modules)
    build_mode = BUILD_MODE_MAP.get(modules_frozen, "full")
    needs_compile = build_mode not in ("skip",)

    sync_strategy = "scp" if len(all_files) <= 3 else "rsync"

    build_reason_map = {
        "skip": "only Python/test files changed",
        "incremental": "C++ source changed, cmake cache can be reused",
        "full": "config/codegen changed or mixed modules, need full rebuild",
    }

    _json_out({
        "status": "ok",
        "changed_files": classified,
        "summary": {
            "total_files": len(all_files),
            "modules_affected": sorted(modules),
            "has_cpp_changes": has_cpp,
            "has_test_only_changes": modules == {"test"},
        },
        "recommendations": {
            "sync_strategy": sync_strategy,
            "sync_reason": f"changed_files={len(all_files)}, {'<= 3 → scp' if len(all_files) <= 3 else '> 3 → rsync'}",
            "build_mode": build_mode,
            "build_reason": build_reason_map.get(build_mode, "fallback to full"),
            "needs_compile": needs_compile,
            "needs_codegen": "op-plugin-config" in modules,
        },
        "patch_file": str(patch_file) if patch_file else None,
    })


def cmd_preflight(args):
    """Run pre-flight checks on the remote server."""
    config = _load_config(args.config)
    timeout = _get_timeout(config, "preflight", args.timeout)
    ctx = _get_context(config, "verify")
    ctx_type = ctx.get("type", "bare")
    checks = {}
    warnings = []

    result = _run_ssh(config, "echo OK", timeout=15)
    checks["ssh_connection"] = {
        "passed": result["exit_code"] == 0 and "OK" in result["stdout"],
        "detail": f"connected in {result['elapsed_seconds']}s" if result["exit_code"] == 0 else result["stderr"][:200],
    }

    if not checks["ssh_connection"]["passed"]:
        _json_out({
            "status": "ok",
            "exec_context_type": ctx_type,
            "checks": checks,
            "warnings": [
                "SSH connection failed. Cannot proceed.",
                "This tool only supports SSH key-based authentication (password login is not supported).",
                "Please ensure: (1) SSH key pair is generated (ssh-keygen), "
                "(2) public key is deployed to the remote server (~/.ssh/authorized_keys), "
                "(3) key file permissions are correct (chmod 600 ~/.ssh/id_rsa).",
            ],
            "ready": False,
            "blocker": "ssh_connection",
        })
        return

    if ctx_type == "docker":
        container = ctx.get("docker_container", "")
        result = _run_ssh(config, f"docker inspect --format='{{{{.State.Running}}}}' {container}", timeout=15)
        running = "true" in result["stdout"].lower()
        checks["docker_container"] = {
            "passed": running,
            "detail": f"container '{container}' is running" if running else f"container '{container}' not running or not found",
        }
        if not running:
            _json_out({
                "status": "ok",
                "exec_context_type": ctx_type,
                "checks": checks,
                "warnings": [f"Docker container '{container}' is not running."],
                "ready": False,
                "blocker": "docker_container",
            })
            return

    path_cmd = _build_remote_command(config, f"test -d . && echo PATH_OK")
    result = _run_ssh(config, path_cmd, timeout=15)
    checks["remote_path"] = {
        "passed": "PATH_OK" in result["stdout"],
        "detail": f"{config['paths']['remote_root']} exists" if "PATH_OK" in result["stdout"] else "path not found",
    }

    git_cmd = _build_remote_command(config, "git status --porcelain 2>/dev/null | head -5")
    result = _run_ssh(config, git_cmd, timeout=15)
    git_clean = result["exit_code"] == 0 and result["stdout"].strip() == ""
    checks["remote_git_clean"] = {
        "passed": git_clean,
        "detail": "clean working tree" if git_clean else f"{len(result['stdout'].strip().splitlines())} uncommitted changes on remote",
    }
    if not git_clean:
        warnings.append("Remote repo has uncommitted changes. Recommend: git stash before sync.")

    py_ver = config.get("build", {}).get("python_version", "3")
    py_cmd = _build_remote_command(config, f"python{py_ver} --version 2>&1")
    result = _run_ssh(config, py_cmd, timeout=15)
    checks["remote_python"] = {
        "passed": result["exit_code"] == 0,
        "detail": result["stdout"].strip() or result["stderr"].strip()[:100],
    }

    cann_cmd = _build_remote_command(config, "cat /usr/local/Ascend/ascend-toolkit/version 2>/dev/null || echo NOT_FOUND")
    result = _run_ssh(config, cann_cmd, timeout=15)
    has_cann = "NOT_FOUND" not in result["stdout"]
    checks["remote_cann"] = {
        "passed": has_cann,
        "detail": result["stdout"].strip()[:100] if has_cann else "CANN not found at default path",
    }

    cache_cmd = _build_remote_command(config, "test -f build/CMakeCache.txt && echo CACHE_OK || echo NO_CACHE")
    result = _run_ssh(config, cache_cmd, timeout=15)
    has_cache = "CACHE_OK" in result["stdout"]
    checks["cmake_cache"] = {
        "passed": has_cache,
        "detail": "cmake cache exists (incremental build possible)" if has_cache else "no cmake cache (will need full build)",
    }

    docker_config = config.get("build", {}).get("docker", None)
    if docker_config:
        docker_image = docker_config.get("image", "manylinux-builder:v1")
        result = _run_ssh(config, f"docker images -q {docker_image}", timeout=15)
        has_image = bool(result["stdout"].strip())
        checks["docker_image"] = {
            "passed": has_image,
            "detail": f"Docker image '{docker_image}' exists" if has_image else f"Docker image '{docker_image}' not found",
        }
        if not has_image:
            warnings.append(f"Docker image '{docker_image}' not found. Build may fail.")

        result = _run_ssh(config, "docker --version", timeout=15)
        checks["docker_available"] = {
            "passed": result["exit_code"] == 0,
            "detail": result["stdout"].strip()[:50] if result["exit_code"] == 0 else "Docker not available",
        }

    remote_root = config["paths"]["remote_root"]
    disk_cmd = f"df -BG {remote_root} 2>/dev/null | tail -1 | awk '{{print $4}}'"
    result = _run_ssh(config, disk_cmd, timeout=15)
    avail_gb = 0
    if result["exit_code"] == 0 and result["stdout"].strip():
        try:
            avail_str = result["stdout"].strip().replace("G", "")
            avail_gb = int(avail_str)
        except ValueError:
            pass
    min_required_gb = 20
    checks["disk_space"] = {
        "passed": avail_gb >= min_required_gb,
        "detail": f"{avail_gb}GB available (need >= {min_required_gb}GB for build)",
    }
    if avail_gb < min_required_gb:
        warnings.append(f"Low disk space: only {avail_gb}GB available, recommend at least {min_required_gb}GB")

    all_critical = all(
        v["passed"] for k, v in checks.items()
        if k in ("ssh_connection", "remote_path", "remote_python")
    )
    blocker = ""
    if not all_critical:
        for k in ("ssh_connection", "remote_path", "remote_python"):
            if not checks.get(k, {}).get("passed", True):
                blocker = k
                break

    _json_out({
        "status": "ok",
        "exec_context_type": ctx_type,
        "checks": checks,
        "warnings": warnings,
        "ready": all_critical and not blocker,
        "blocker": blocker,
    })


def cmd_sync(args):
    """Sync local changed files to the remote server.

    Supports three strategies:
    - patch: Apply git diff as patch (recommended for consistency)
    - scp: Direct file copy for small changes
    - rsync: Bulk sync for many files

    The patch strategy ensures remote code matches local exactly.
    Tracks new files and original state for proper rollback.
    """
    config = _load_config(args.config)
    local_root = Path(config["paths"]["local_root"]).resolve()
    sync_target_root = _get_sync_target_root(config)
    strategy = args.strategy or config.get("sync", {}).get("strategy", "patch")

    try:
        unstaged = subprocess.check_output(
            ["git", "diff", "--name-only"],
            cwd=str(local_root), text=True
        ).strip().splitlines()
    except Exception:
        unstaged = []
    try:
        staged = subprocess.check_output(
            ["git", "diff", "--name-only", "--staged"],
            cwd=str(local_root), text=True
        ).strip().splitlines()
    except Exception:
        staged = []

    all_files = sorted(set(f.replace("\\", "/") for f in unstaged + staged if f))

    explicit_remote: dict[str, str] = {}

    extra_files_cli = getattr(args, "files", None) or []
    for ef in extra_files_cli:
        if ":" in ef and not ef[0] == "/":
            local_p, remote_p = ef.split(":", 1)
            norm = local_p.replace("\\", "/")
            if norm not in all_files:
                all_files.append(norm)
            explicit_remote[norm] = remote_p
        else:
            norm = ef.replace("\\", "/")
            if norm not in all_files:
                all_files.append(norm)

    for ef_entry in config.get("sync", {}).get("extra_files", []):
        local_p = ef_entry.get("local", "").replace("\\", "/")
        remote_p = ef_entry.get("remote", "")
        if local_p:
            if local_p not in all_files:
                all_files.append(local_p)
            if remote_p:
                explicit_remote[local_p] = remote_p

    all_files = sorted(set(all_files))

    def _remote_for(f: str) -> str:
        if f in explicit_remote:
            return explicit_remote[f]
        return _resolve_sync_path(f, config).replace(
            config["paths"]["remote_root"], sync_target_root
        )

    if not all_files:
        _json_out({"status": "ok", "strategy_used": "none", "files_synced": [], "all_verified": True, "message": "No files to sync"})
        return

    if strategy == "auto":
        strategy = "patch"

    remote = config["remote"]
    ssh_target = f"{remote['user']}@{remote['host']}"
    port = remote.get("port", 22)

    if args.dry_run:
        planned = []
        for f in all_files:
            remote_path = _remote_for(f)
            planned.append({"local": f, "remote": remote_path})

        _json_out({
            "status": "dry_run",
            "strategy": strategy,
            "files_planned": planned,
            "total_files": len(planned),
            "commands_preview": _build_sync_commands_preview(
                strategy, all_files, local_root, sync_target_root, ssh_target, port, config
            ),
        })
        return

    backup_info = {}
    remote_root = config["paths"]["remote_root"]

    result = _run_ssh(config, f"cd {remote_root} && git status --porcelain", timeout=30)
    original_dirty = result["stdout"].strip() if result["exit_code"] == 0 else ""

    result = _run_ssh(config, f"cd {remote_root} && git rev-parse HEAD 2>/dev/null || echo UNKNOWN", timeout=15)
    original_commit = result["stdout"].strip() if result["exit_code"] == 0 else "UNKNOWN"

    result = _run_ssh(config, f"cd {remote_root} && git ls-files --others --exclude-standard", timeout=30)
    original_untracked = result["stdout"].strip().splitlines() if result["exit_code"] == 0 else []

    synced_files = []
    new_files = []
    modified_files = []

    if strategy == "patch":
        patch_file = tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False)
        try:
            diff = subprocess.check_output(
                ["git", "diff"], cwd=str(local_root), text=True
            )
            staged_diff = subprocess.check_output(
                ["git", "diff", "--staged"], cwd=str(local_root), text=True
            )
            patch_content = diff + staged_diff
            patch_file.write(patch_content)
            patch_file.close()

            remote_patch = f"/tmp/sync_{_timestamp()}.patch"
            scp_cmd = ["scp"]
            if port != 22:
                scp_cmd += ["-P", str(port)]
            scp_cmd += [patch_file.name, f"{ssh_target}:{remote_patch}"]
            subprocess.run(scp_cmd, capture_output=True, text=True, timeout=60)

            apply_result = _run_ssh(config, f"cd {remote_root} && git apply --check {remote_patch} 2>&1", timeout=30)
            if apply_result["exit_code"] != 0 or "error" in apply_result["stdout"].lower():
                _run_ssh(config, f"cd {remote_root} && git checkout -- . && git clean -fd", timeout=60)
                apply_result = _run_ssh(config, f"cd {remote_root} && git apply {remote_patch}", timeout=30)

            for f in all_files:
                remote_path = _remote_for(f)
                rel_path = remote_path.replace(remote_root + "/", "") if remote_root + "/" in remote_path else f
                is_new = rel_path in original_untracked or not any(
                    _run_ssh(config, f"cd {remote_root} && git ls-files {shlex.quote(rel_path)}", timeout=15)["stdout"].strip()
                    for _ in [None]
                )
                synced_files.append({
                    "local": f,
                    "remote": remote_path,
                    "success": True,
                    "error": "",
                    "is_new": is_new,
                })
                if is_new:
                    new_files.append(remote_path)
                else:
                    modified_files.append(remote_path)

            backup_info = {
                "method": "git_patch",
                "patch_path": remote_patch,
                "original_commit": original_commit,
                "original_dirty": original_dirty[:500] if original_dirty else "",
            }
        finally:
            os.unlink(patch_file.name)

    elif strategy == "scp":
        for f in all_files:
            local_path = str(local_root / f) if f in explicit_remote else str(local_root / f)
            if f in explicit_remote:
                local_abs = Path(f).resolve() if Path(f).is_absolute() else (local_root / f).resolve()
                local_path = str(local_abs)
            remote_path = _remote_for(f)

            check_cmd = _build_remote_command(config, f"test -f {shlex.quote(remote_path)} && echo EXISTS || echo NEW")
            check_result = _run_ssh(config, check_cmd, timeout=15)
            is_new = "NEW" in check_result["stdout"]

            mkdir_cmd = _build_remote_command(config, f"mkdir -p $(dirname {shlex.quote(remote_path)})")
            _run_ssh(config, mkdir_cmd, timeout=15)

            scp_cmd = ["scp"]
            if port != 22:
                scp_cmd += ["-P", str(port)]
            scp_cmd += [local_path, f"{ssh_target}:{remote_path}"]
            proc = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=120)
            synced_files.append({
                "local": f,
                "remote": remote_path,
                "success": proc.returncode == 0,
                "error": proc.stderr.strip()[:200] if proc.returncode != 0 else "",
                "is_new": is_new,
            })
            if is_new:
                new_files.append(remote_path)
            else:
                modified_files.append(remote_path)

    elif strategy == "rsync":
        filelist = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
        try:
            for f in all_files:
                filelist.write(f + "\n")
            filelist.close()

            rsync_cmd = [
                "rsync", "-avz", "--relative",
                "--files-from", filelist.name,
                str(local_root) + "/",
                f"{ssh_target}:{sync_target_root}/",
            ]
            if port != 22:
                rsync_cmd += ["-e", f"ssh -p {port}"]

            proc = subprocess.run(rsync_cmd, capture_output=True, text=True, timeout=300)
            for f in all_files:
                remote_path = _remote_for(f)
                rel_path = remote_path.replace(remote_root + "/", "") if remote_root + "/" in remote_path else f
                is_new = rel_path in original_untracked
                synced_files.append({
                    "local": f,
                    "remote": remote_path,
                    "success": proc.returncode == 0,
                    "error": proc.stderr.strip()[:200] if proc.returncode != 0 else "",
                    "is_new": is_new,
                })
                if is_new:
                    new_files.append(remote_path)
                else:
                    modified_files.append(remote_path)
        finally:
            os.unlink(filelist.name)

    all_ok = all(s["success"] for s in synced_files)

    sync_state = {
        "timestamp": _timestamp(),
        "strategy": strategy,
        "files": [s["local"] for s in synced_files],
        "new_files": new_files,
        "modified_files": modified_files,
        "backup": backup_info,
        "original_commit": original_commit,
        "original_dirty": original_dirty[:500] if original_dirty else "",
        "original_untracked": original_untracked,
    }
    state_path = _ensure_log_dir(config) / "sync_state.json"
    _save_config(str(state_path), sync_state)

    _json_out({
        "status": "ok" if all_ok else "error",
        "strategy_used": strategy,
        "files_synced": synced_files,
        "new_files": new_files,
        "remote_backup": backup_info,
        "all_verified": all_ok,
        "sync_state_file": str(state_path),
    })


def _build_sync_commands_preview(strategy, files, local_root, sync_root, ssh_target, port, config):
    """Build preview of commands that would be executed."""
    previews = []
    if strategy == "scp":
        for f in files[:5]:
            local_path = str(local_root / f)
            remote_path = _resolve_sync_path(f, config).replace(
                config["paths"]["remote_root"], sync_root
            )
            p = f"-P {port} " if port != 22 else ""
            previews.append(f"scp {p}{local_path} {ssh_target}:{remote_path}")
        if len(files) > 5:
            previews.append(f"... and {len(files) - 5} more files")
    elif strategy == "rsync":
        p = f'-e "ssh -p {port}" ' if port != 22 else ""
        previews.append(f"rsync -avz --relative --files-from=<filelist> {p}{local_root}/ {ssh_target}:{sync_root}/")
    elif strategy == "patch":
        previews.append(f"git diff > fix.patch && scp fix.patch {ssh_target}:/tmp/")
        previews.append(f"ssh {ssh_target} 'cd {sync_root} && git apply /tmp/fix.patch'")
    return previews


def _build_docker_command(config: dict, container: str, inner_cmd: str) -> str:
    """Build a command that executes inside a Docker container."""
    escaped = inner_cmd.replace("'", "'\\''")
    return f"docker exec {container} bash -c '{escaped}'"


def cmd_build(args):
    """Run remote build.

    Supports two modes:
    1. Bare/SSH mode: Build directly in exec_context (existing behavior)
    2. Docker mode: Build inside a Docker container, with optional auto-cleanup

    Docker mode is triggered when build.docker config is present:
    {
      "build": {
        "docker": {
          "image": "manylinux-builder:v1",
          "mounts": ["/home:/home", "/root:/root"],
          "container_name": "pta_compile_temp",
          "auto_cleanup": true
        }
      }
    }
    """
    config = _load_config(args.config)
    log_dir = _ensure_log_dir(config)
    mode = args.mode or config.get("build", {}).get("mode", "auto")
    py_ver = config.get("build", {}).get("python_version", "3.9")
    timeout = _get_timeout(config, "build", args.timeout)
    remote_root = config["paths"]["remote_root"]

    docker_config = config.get("build", {}).get("docker", None)
    use_docker = docker_config is not None

    if mode == "custom":
        build_cmd = config["build"]["command"]
    elif mode == "auto":
        collect_state_path = log_dir / "collect_state.json"
        if collect_state_path.exists():
            with open(collect_state_path) as f:
                collect_data = json.load(f)
            recommended = collect_data.get("recommendations", {}).get("build_mode", "full")
        else:
            recommended = "full"

        if recommended == "incremental" and not use_docker:
            check_cmd = _build_remote_command(config, "test -f build/CMakeCache.txt && echo HAS_CACHE")
            result = _run_ssh(config, check_cmd, timeout=15)
            if "HAS_CACHE" not in result["stdout"]:
                recommended = "full"

        mode = recommended
        if mode == "skip":
            _json_out({
                "status": "ok",
                "build_mode": "skip",
                "success": True,
                "message": "No C++ compilation needed",
                "elapsed_seconds": 0,
            })
            return

        template = BUILD_COMMANDS.get(mode, BUILD_COMMANDS["full"])
        build_cmd = template.format(
            python_version=py_ver,
            build_dir=f"{remote_root}/build",
        )
    elif mode in BUILD_COMMANDS:
        template = BUILD_COMMANDS[mode]
        build_cmd = template.format(
            python_version=py_ver,
            build_dir=f"{remote_root}/build",
        )
    else:
        build_cmd = config.get("build", {}).get("command", f"bash ci/build.sh --python={py_ver}")

    needs_codegen = False
    collect_state_path = log_dir / "collect_state.json"
    if collect_state_path.exists():
        with open(collect_state_path) as f:
            cdata = json.load(f)
        needs_codegen = cdata.get("recommendations", {}).get("needs_codegen", False)

    if needs_codegen and mode != "full":
        codegen_cmd = f"bash generate_code.sh python{py_ver} $(cat version.txt | head -1 | tr -d '\\n')"
        build_cmd = f"{codegen_cmd} && {build_cmd}"

    container_name = None
    container_created = False

    if use_docker:
        docker_image = docker_config.get("image", "manylinux-builder:v1")
        container_name = docker_config.get("container_name", f"pta_compile_{_timestamp()}")
        mounts = docker_config.get("mounts", ["/home:/home", "/root:/root"])
        auto_cleanup = docker_config.get("auto_cleanup", True)

        mount_args = " ".join(f"-v {m}" for m in mounts)
        create_cmd = f"docker run -d -it {mount_args} --name {container_name} {docker_image} bash"

        print(f"Creating Docker container '{container_name}' from image '{docker_image}'...", file=sys.stderr)
        create_result = _run_ssh(config, create_cmd, timeout=60)
        if create_result["exit_code"] != 0:
            _json_out({
                "status": "error",
                "phase": "docker_create",
                "message": f"Failed to create Docker container: {create_result['stderr'][:300]}",
            })
            return
        container_created = True

        inner_build_cmd = f"cd {remote_root} && rm -rf build dist && {build_cmd}"
        full_remote_cmd = _build_docker_command(config, container_name, inner_build_cmd)
    else:
        full_remote_cmd = _build_remote_command(config, build_cmd)

    print(f"Executing remote build (mode={mode}, docker={use_docker})...", file=sys.stderr)
    result = _run_ssh(config, full_remote_cmd, timeout=timeout)

    combined_output = result["stdout"] + "\n" + result["stderr"]
    log_file = log_dir / f"build_{_timestamp()}.log"
    log_file.write_text(combined_output, encoding="utf-8")

    errors = _parse_build_errors(combined_output)
    warnings = _parse_build_warnings(combined_output)
    output_lines = combined_output.strip().splitlines()
    log_tail = output_lines[-10:] if output_lines else []

    success = result["exit_code"] == 0

    if use_docker and container_created and auto_cleanup:
        print(f"Cleaning up Docker container '{container_name}'...", file=sys.stderr)
        cleanup_cmd = f"docker rm -f {container_name}"
        _run_ssh(config, cleanup_cmd, timeout=30)

    error_summary = ""
    if errors:
        files_with_errors = set(e["file"] for e in errors)
        error_summary = f"{len(errors)} error(s) in {', '.join(files_with_errors)}"

    output = {
        "status": "ok" if success else "error",
        "build_mode": mode,
        "exec_context_type": "docker" if use_docker else config.get("exec_context", {}).get("type", "bare"),
        "full_command": full_remote_cmd[:500],
        "success": success,
        "elapsed_seconds": result["elapsed_seconds"],
        "log_file": str(log_file),
        "log_tail": log_tail,
        "errors": errors[:20],
        "warnings": warnings[:10],
    }
    if use_docker:
        output["docker"] = {
            "image": docker_image,
            "container": container_name,
            "auto_cleaned": auto_cleanup and container_created,
        }
    if error_summary:
        output["error_summary"] = error_summary
    if result["timed_out"]:
        output["timed_out"] = True

    _json_out(output)


def cmd_run(args):
    """Run remote verification command.

    Supports auto-rollback when verification passes:
    - If --auto-rollback is set and verdict is "pass", triggers rollback automatically
    - Useful for one-shot verification without leaving changes on remote
    """
    config = _load_config(args.config)
    log_dir = _ensure_log_dir(config)
    verify = config.get("verify", {})
    command = args.command or verify.get("command")
    timeout = _get_timeout(config, "run", args.timeout or verify.get("timeout"))
    auto_rollback = getattr(args, "auto_rollback", False) or verify.get("auto_rollback", False)

    if not command:
        _json_out({"status": "error", "message": "No verification command specified"})
        return

    env_vars = verify.get("env_vars", {})
    env_prefix = " ".join(f"{k}={v}" for k, v in env_vars.items())
    full_verify_cmd = f"{env_prefix} {command}".strip() if env_prefix else command

    full_remote_cmd = _build_remote_command(config, full_verify_cmd)

    print(f"Running remote verification...", file=sys.stderr)
    result = _run_ssh(config, full_remote_cmd, timeout=timeout)

    combined = result["stdout"] + "\n" + result["stderr"]
    log_file = log_dir / f"run_{_timestamp()}.log"
    log_file.write_text(combined, encoding="utf-8")

    pytest_summary = _parse_pytest_output(combined)
    output_lines = combined.strip().splitlines()

    verdict = "pass" if result["exit_code"] == 0 else "fail"
    if result["timed_out"]:
        verdict = "timeout"

    tail_lines = config.get("verify", {}).get("output_tail_lines", 15)
    if verdict in ("fail", "timeout"):
        tail_lines = max(tail_lines, 30)

    output = {
        "status": "ok",
        "command": full_verify_cmd,
        "exit_code": result["exit_code"],
        "elapsed_seconds": result["elapsed_seconds"],
        "log_file": str(log_file),
        "pytest_summary": pytest_summary,
        "output_tail": output_lines[-tail_lines:] if output_lines else [],
        "verdict": verdict,
    }

    if verdict == "fail":
        output["failure_detail"] = _extract_failure_detail(combined)
    if result["timed_out"]:
        output["timed_out"] = True

    if auto_rollback and verdict == "pass":
        print("Verification passed, triggering auto-rollback...", file=sys.stderr)
        rollback_args = type("Args", (), {"config": args.config})()
        rollback_buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = rollback_buf
        try:
            cmd_rollback(rollback_args)
        finally:
            sys.stdout = old_stdout
        try:
            rollback_result = json.loads(rollback_buf.getvalue())
            output["auto_rollback"] = rollback_result
        except json.JSONDecodeError:
            output["auto_rollback"] = {"status": "error", "message": "Failed to parse rollback output"}

    _json_out(output)


def cmd_sync_run(args):
    """Sync local changes then immediately run verification (combined workflow).

    Supports auto-rollback when verification passes.
    """
    config = _load_config(args.config)
    log_dir = _ensure_log_dir(config)
    auto_rollback = getattr(args, "auto_rollback", False) or config.get("verify", {}).get("auto_rollback", False)

    class _NS:
        pass
    sync_args = _NS()
    sync_args.config = args.config
    sync_args.strategy = args.strategy
    sync_args.dry_run = False
    sync_args.files = getattr(args, "files", None)

    sync_buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = sync_buf
    try:
        cmd_sync(sync_args)
    finally:
        sys.stdout = old_stdout
    sync_output_str = sync_buf.getvalue()
    try:
        sync_result = json.loads(sync_output_str)
    except json.JSONDecodeError:
        _json_out({"status": "error", "phase": "sync", "message": "Failed to parse sync output", "raw": sync_output_str[:500]})
        return

    if sync_result.get("status") == "error":
        _json_out({"status": "error", "phase": "sync", "sync_result": sync_result})
        return

    verify = config.get("verify", {})
    command = args.command or verify.get("command")
    timeout = args.timeout or verify.get("timeout", 600)

    if not command:
        _json_out({"status": "error", "phase": "run", "message": "No verification command specified", "sync_result": sync_result})
        return

    env_vars = verify.get("env_vars", {})
    env_prefix = " ".join(f"{k}={v}" for k, v in env_vars.items())
    full_verify_cmd = f"{env_prefix} {command}".strip() if env_prefix else command
    full_remote_cmd = _build_remote_command(config, full_verify_cmd)

    print("Running remote verification...", file=sys.stderr)
    result = _run_ssh(config, full_remote_cmd, timeout=timeout)

    combined = result["stdout"] + "\n" + result["stderr"]
    log_file = log_dir / f"syncrun_{_timestamp()}.log"
    log_file.write_text(combined, encoding="utf-8")

    pytest_summary = _parse_pytest_output(combined)
    output_lines = combined.strip().splitlines()

    verdict = "pass" if result["exit_code"] == 0 else "fail"
    if result["timed_out"]:
        verdict = "timeout"

    tail_lines = config.get("verify", {}).get("output_tail_lines", 15)
    if verdict == "fail":
        tail_lines = max(tail_lines, 30)

    output = {
        "status": "ok",
        "sync_result": {
            "strategy_used": sync_result.get("strategy_used"),
            "files_synced": len(sync_result.get("files_synced", [])),
            "all_verified": sync_result.get("all_verified"),
        },
        "run_result": {
            "command": full_verify_cmd,
            "exit_code": result["exit_code"],
            "elapsed_seconds": result["elapsed_seconds"],
            "log_file": str(log_file),
            "pytest_summary": pytest_summary,
            "output_tail": output_lines[-tail_lines:] if output_lines else [],
            "verdict": verdict,
            "failure_detail": _extract_failure_detail(combined) if verdict == "fail" else {},
            "timed_out": result["timed_out"],
        },
    }

    if auto_rollback and verdict == "pass":
        print("Verification passed, triggering auto-rollback...", file=sys.stderr)
        rollback_args = type("Args", (), {"config": args.config})()
        rollback_buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = rollback_buf
        try:
            cmd_rollback(rollback_args)
        finally:
            sys.stdout = old_stdout
        try:
            rollback_result = json.loads(rollback_buf.getvalue())
            output["auto_rollback"] = rollback_result
        except json.JSONDecodeError:
            output["auto_rollback"] = {"status": "error", "message": "Failed to parse rollback output"}

    _json_out(output)


def cmd_install(args):
    """Install the built wheel package on the remote host.

    This command finds the wheel file in dist/ directory and installs it
    using pip. Typically used after 'build' command completes.

    Usage:
        python tools/remote_deploy_verify.py install --config deploy_context.json
        python tools/remote_deploy_verify.py install --config deploy_context.json --wheel dist/torch_npu-*.whl
    """
    config = _load_config(args.config)
    remote_root = config["paths"]["remote_root"]
    timeout = _get_timeout(config, "install", args.timeout)

    if args.wheel:
        wheel_path = args.wheel
    else:
        find_cmd = f"ls {remote_root}/dist/*.whl 2>/dev/null | head -1"
        find_result = _run_ssh(config, find_cmd, timeout=30)
        if find_result["exit_code"] != 0 or not find_result["stdout"].strip():
            _json_out({
                "status": "error",
                "message": "No wheel file found in dist/ directory. Run 'build' first.",
            })
            return
        wheel_path = find_result["stdout"].strip()

    wheel_name = wheel_path.split("/")[-1]

    install_cmd = f"pip install {wheel_path} --force-reinstall"
    full_remote_cmd = _build_remote_command(config, install_cmd)

    print(f"Installing wheel: {wheel_name}...", file=sys.stderr)
    result = _run_ssh(config, full_remote_cmd, timeout=timeout)

    success = result["exit_code"] == 0

    output_lines = (result["stdout"] + "\n" + result["stderr"]).strip().splitlines()

    _json_out({
        "status": "ok" if success else "error",
        "wheel_file": wheel_path,
        "wheel_name": wheel_name,
        "success": success,
        "elapsed_seconds": result["elapsed_seconds"],
        "output_tail": output_lines[-15:] if output_lines else [],
        "error": result["stderr"][:500] if not success else "",
    })


def cmd_exec(args):
    """Execute an arbitrary command on the remote server (lightweight, no log file)."""
    config = _load_config(args.config)
    timeout = _get_timeout(config, "exec", args.timeout)
    remote_cmd = _build_remote_command(config, args.remote_command)
    result = _run_ssh(config, remote_cmd, timeout=timeout)
    _json_out({
        "status": "ok",
        "command": args.remote_command,
        "exit_code": result["exit_code"],
        "stdout": result["stdout"],
        "stderr": result["stderr"],
        "elapsed_seconds": result["elapsed_seconds"],
        "timed_out": result["timed_out"],
    })


def cmd_report(args):
    """Generate a summary report from all phase outputs."""
    config = _load_config(args.config)
    log_dir = _ensure_log_dir(config)
    remote = config["remote"]

    report = {
        "server": remote["host"],
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    # Read sync state
    sync_state_file = log_dir / "sync_state.json"
    if sync_state_file.exists():
        with open(sync_state_file) as f:
            sync_state = json.load(f)
        report["sync"] = {
            "strategy": sync_state.get("strategy", "unknown"),
            "files": len(sync_state.get("files", [])),
        }

    # Read latest build log summary
    build_logs = sorted(log_dir.glob("build_*.log"), reverse=True)
    if build_logs:
        content = build_logs[0].read_text(encoding="utf-8", errors="replace")
        errors = _parse_build_errors(content)
        report["build"] = {
            "success": len(errors) == 0,
            "log_file": str(build_logs[0]),
            "errors": len(errors),
        }

    # Read latest run log summary
    run_logs = sorted(log_dir.glob("run_*.log"), reverse=True)
    if run_logs:
        content = run_logs[0].read_text(encoding="utf-8", errors="replace")
        pytest_summary = _parse_pytest_output(content)
        report["test"] = {
            "verdict": "pass" if pytest_summary.get("failed", 0) == 0 and pytest_summary.get("error", 0) == 0 and not pytest_summary.get("parse_failed") else "fail",
            **pytest_summary,
        }

    # Determine conclusion
    build_ok = report.get("build", {}).get("success", True)
    test_ok = report.get("test", {}).get("verdict") == "pass"
    if build_ok and test_ok:
        conclusion = "fix_verified"
    elif not build_ok:
        conclusion = "build_failed"
    else:
        conclusion = "test_failed"
    report["conclusion"] = conclusion

    # Build human-readable report
    lines = ["Remote Deploy & Verify Report:"]
    lines.append(f"- Server: {report['server']}")
    if "sync" in report:
        lines.append(f"- Sync: {report['sync']['strategy']}, {report['sync']['files']} files")
    if "build" in report:
        status = "success" if report["build"]["success"] else f"failed ({report['build']['errors']} errors)"
        lines.append(f"- Build: {status}")
    if "test" in report:
        t = report["test"]
        lines.append(f"- Test: {t.get('passed', 0)} passed, {t.get('failed', 0)} failed")
        lines.append(f"- Verdict: {t.get('verdict', 'unknown')}")
    conclusions_zh = {
        "fix_verified": "Fix verified on remote",
        "build_failed": "Build failed, fix needs correction",
        "test_failed": "Test failed, issue not resolved",
    }
    lines.append(f"- Conclusion: {conclusions_zh.get(conclusion, conclusion)}")

    _json_out({
        "status": "ok",
        "report": report,
        "report_text": "\n".join(lines),
    })


def cmd_rollback(args):
    """Rollback remote changes using the recorded sync state.

    Performs complete restoration:
    1. Delete new files created during sync
    2. Restore modified files via git checkout
    3. Reset to original commit if needed
    4. Clear sync state file
    """
    config = _load_config(args.config)
    log_dir = _ensure_log_dir(config)
    remote_root = config["paths"]["remote_root"]

    sync_state_file = log_dir / "sync_state.json"
    if not sync_state_file.exists():
        _json_out({"status": "ok", "message": "No sync state found. Remote is already clean."})
        return

    with open(sync_state_file) as f:
        sync_state = json.load(f)

    new_files = sync_state.get("new_files", [])
    modified_files = sync_state.get("modified_files", [])
    original_commit = sync_state.get("original_commit", "")
    original_dirty = sync_state.get("original_dirty", "")
    original_untracked = sync_state.get("original_untracked", [])

    results = {
        "new_files_deleted": [],
        "modified_files_restored": [],
        "errors": [],
    }

    for nf in new_files:
        rm_cmd = f"rm -f {shlex.quote(nf)}"
        rm_result = _run_ssh(config, rm_cmd, timeout=15)
        if rm_result["exit_code"] == 0:
            results["new_files_deleted"].append(nf)
        else:
            results["errors"].append(f"Failed to delete {nf}: {rm_result['stderr'][:100]}")

    if modified_files:
        checkout_cmd = f"cd {remote_root} && git checkout -- " + " ".join(
            shlex.quote(f.replace(remote_root + "/", "")) for f in modified_files
        )
        checkout_result = _run_ssh(config, checkout_cmd, timeout=60)
        if checkout_result["exit_code"] == 0:
            results["modified_files_restored"] = modified_files
        else:
            results["errors"].append(f"Git checkout failed: {checkout_result['stderr'][:200]}")

    if original_commit and original_commit != "UNKNOWN":
        result = _run_ssh(config, f"cd {remote_root} && git status --porcelain", timeout=30)
        current_dirty = result["stdout"].strip() if result["exit_code"] == 0 else ""
        if not current_dirty:
            result = _run_ssh(config, f"cd {remote_root} && git rev-parse HEAD", timeout=15)
            current_commit = result["stdout"].strip() if result["exit_code"] == 0 else ""
            if current_commit != original_commit and not original_dirty:
                reset_result = _run_ssh(config, f"cd {remote_root} && git reset --hard {original_commit}", timeout=30)
                if reset_result["exit_code"] != 0:
                    results["errors"].append(f"Git reset failed: {reset_result['stderr'][:200]}")

    patch_path = sync_state.get("backup", {}).get("patch_path", "")
    if patch_path:
        _run_ssh(config, f"rm -f {patch_path}", timeout=10)

    sync_state_file.unlink(missing_ok=True)

    success = len(results["errors"]) == 0
    _json_out({
        "status": "ok" if success else "partial",
        "rollback_summary": {
            "new_files_deleted": len(results["new_files_deleted"]),
            "modified_files_restored": len(results["modified_files_restored"]),
            "errors": results["errors"],
        },
        "details": results,
        "sync_state_cleared": True,
    })


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Remote deploy & verify for torch_npu development"
    )
    sub = parser.add_subparsers(dest="subcmd", help="Sub-commands")

    # collect
    p_collect = sub.add_parser("collect", help="Collect local git changes and classify")
    p_collect.add_argument("--local-root", required=True, help="Path to torch_npu local repo")

    # preflight
    p_pre = sub.add_parser("preflight", help="Run pre-flight checks on remote")
    p_pre.add_argument("--config", required=True, help="Path to deploy_context.json")
    p_pre.add_argument("--timeout", type=int, default=None, help="Preflight timeout in seconds")

    # sync
    p_sync = sub.add_parser("sync", help="Sync local changes to remote")
    p_sync.add_argument("--config", required=True, help="Path to deploy_context.json")
    p_sync.add_argument("--strategy", choices=["auto", "scp", "rsync", "patch"], default=None)
    p_sync.add_argument("--dry-run", action="store_true", help="Preview only, do not execute")
    p_sync.add_argument("--files", nargs="*", default=None, help="Extra files to sync (path or local:remote pairs)")

    # build
    p_build = sub.add_parser("build", help="Run remote build")
    p_build.add_argument("--config", required=True, help="Path to deploy_context.json")
    p_build.add_argument("--mode", choices=["auto", "incremental", "full", "develop", "custom", "skip"], default=None)
    p_build.add_argument("--timeout", type=int, default=None, help="Build timeout in seconds")

    # install
    p_install = sub.add_parser("install", help="Install built wheel package on remote host")
    p_install.add_argument("--config", required=True, help="Path to deploy_context.json")
    p_install.add_argument("--wheel", default=None, help="Path to wheel file (auto-detect from dist/ if not specified)")
    p_install.add_argument("--timeout", type=int, default=None, help="Install timeout in seconds")

    # run
    p_run = sub.add_parser("run", help="Run remote verification")
    p_run.add_argument("--config", required=True, help="Path to deploy_context.json")
    p_run.add_argument("--command", default=None, help="Override verification command")
    p_run.add_argument("--timeout", type=int, default=None, help="Run timeout in seconds")
    p_run.add_argument("--auto-rollback", action="store_true", help="Auto-rollback after verification passes")

    # exec
    p_exec = sub.add_parser("exec", help="Execute arbitrary command on remote")
    p_exec.add_argument("--config", required=True, help="Path to deploy_context.json")
    p_exec.add_argument("remote_command", help="Command to run on remote server")
    p_exec.add_argument("--timeout", type=int, default=None, help="Timeout in seconds (default 60)")

    # sync-run
    p_sr = sub.add_parser("sync-run", help="Sync then immediately run verification")
    p_sr.add_argument("--config", required=True, help="Path to deploy_context.json")
    p_sr.add_argument("--strategy", choices=["auto", "scp", "rsync", "patch"], default=None)
    p_sr.add_argument("--files", nargs="*", default=None, help="Extra files to sync")
    p_sr.add_argument("--command", default=None, help="Override verification command")
    p_sr.add_argument("--timeout", type=int, default=None, help="Run timeout in seconds")
    p_sr.add_argument("--auto-rollback", action="store_true", help="Auto-rollback after verification passes")

    # report
    p_report = sub.add_parser("report", help="Generate summary report")
    p_report.add_argument("--config", required=True, help="Path to deploy_context.json")

    # rollback
    p_rollback = sub.add_parser("rollback", help="Rollback remote changes")
    p_rollback.add_argument("--config", required=True, help="Path to deploy_context.json")

    args = parser.parse_args()

    if not args.subcmd:
        parser.print_help()
        sys.exit(1)

    cmd_map = {
        "collect": cmd_collect,
        "preflight": cmd_preflight,
        "sync": cmd_sync,
        "build": cmd_build,
        "install": cmd_install,
        "run": cmd_run,
        "exec": cmd_exec,
        "sync-run": cmd_sync_run,
        "report": cmd_report,
        "rollback": cmd_rollback,
    }

    cmd_map[args.subcmd](args)


if __name__ == "__main__":
    main()
