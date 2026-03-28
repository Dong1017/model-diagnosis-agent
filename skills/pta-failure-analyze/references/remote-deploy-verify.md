# Remote Deploy & Verify Reference

Optionally triggered from Stage 4 to help the user sync fixed code to a remote Ascend server, compile, and run verification. All deterministic operations are handled by `tools/remote_deploy_verify.py`; the agent drives user interaction, interprets failures, and synthesizes reports from JSON outputs.

> **Path resolution**: All `tools/` paths below are relative to the skill directory (the folder containing `SKILL.md`). Before executing any command, resolve to the **absolute path**. For example, if the skill directory is `/repo/skills/pta-failure-analyze`, use `/repo/skills/pta-failure-analyze/tools/remote_deploy_verify.py` instead of `tools/remote_deploy_verify.py`.

## Table of Contents

- [When to Use](#when-to-use)
- [Workflow Overview](#workflow-overview)
- [Phase 1: Collect Local Changes](#phase-1-collect-local-changes)
- [Phase 2: Gather Deploy Context](#phase-2-gather-deploy-context)
- [Phase 3: Preflight Check](#phase-3-preflight-check)
- [Phase 4: Sync Code](#phase-4-sync-code)
- [Phase 5: Build](#phase-5-build)
- [Phase 5.5: Install](#phase-55-install-new)
- [Phase 6: Run Verification](#phase-6-run-verification)
- [Quick Path: sync-run](#quick-path-sync-run)
- [Utility: exec](#utility-exec)
- [Rollback](#rollback)
- [Advanced Config](#advanced-config)
- [Troubleshooting](#troubleshooting)

## When to Use

- User modified torch_npu / op-plugin / test files locally and needs to verify on a remote Ascend server.
- User explicitly requests help with deployment verification in Stage 4.
- Do NOT use when the user can run tests locally or has already verified.

## Workflow Overview

```
collect → gather context → preflight → sync (dry-run → confirm → execute)
  → build (if C++ changed) → install → run → summarize → return to Stage 4
```

**Docker build path** (recommended for torch_npu):

```
build (with docker config) → install (on host) → run
```

**Fast path** (test-only changes, iterative debugging):

```
sync-run   (= sync + run in one step, skipping build)
```

**Lightweight remote inspection**:

```
exec "python -c 'import torch; print(torch.__version__)'"
```

## Phase 1: Collect Local Changes

Run the script to analyze local git changes:

```bash
python tools/remote_deploy_verify.py collect --local-root <path_to_torch_npu>
```

Read the JSON output. Present the user with:
- Changed file list and types (cpp / config / test / python)
- Recommended build mode
- Whether C++ compilation is needed

## Phase 2: Gather Deploy Context

**First, check if a `deploy_context.json` already exists** in the working directory or skill's `deploy_logs/` directory. If found:
1. Read the file and present the key configuration to the user in a readable summary, for example:
   - Server: `dev@10.0.1.100:22`
   - Execution: bare / docker (`container_name`)
   - Remote path: `/home/dev/torch_npu`
   - Python: 3.9
   - Verify command: `python -m pytest test/test_xxx.py -xvs`
2. Ask the user: "Found existing deployment configuration. Do you want to **reuse it**, **modify specific fields**, or **create a new one**?"
3. If reuse → proceed directly to Phase 3 with existing config.
4. If modify → ask which fields to change, update the JSON, and proceed.
5. If create new → follow the full collection flow below.

If no existing config is found, collect deployment information from the user via `AskQuestion` or dialogue:

> **Important**: This tool only supports **SSH key-based authentication**. Password login is not supported. Before proceeding, confirm the user has SSH key access to the remote server. If not, guide them to set up key-based authentication first:
> ```bash
> ssh-keygen -t rsa  # generate key pair (skip if already exists)
> ssh-copy-id user@remote_host  # deploy public key to remote server
> ```

1. **Remote server**: host, user, port
2. **Execution context**: bare (direct SSH) / docker (container name + optional host mount path)
3. **Remote project path**: torch_npu root on remote (or container path)
4. **Pre-execution setup commands**: Ask the user: "Is there any custom command that needs to run before each SSH session on the remote server?" Common examples:
   - `source /home/xxx/env.sh` — custom environment variables
   - `source /usr/local/Ascend/ascend-toolkit/set_env.sh` — CANN toolkit
   - `conda activate torch_npu_dev` — conda environment
   - `export LD_LIBRARY_PATH=...` — library paths
   - Multiple commands can be combined: `source /home/xxx/env.sh && conda activate myenv`
   
   This is stored in `exec_context.env_setup` and will be executed automatically before every remote command (build, run, preflight checks, etc.).
5. **Python version**: e.g. 3.9, 3.10
6. **Verification command**: test command or script to run

Write the collected information to `deploy_context.json` directly as JSON:

```json
{
  "remote": { "host": "10.0.1.100", "user": "dev", "port": 22 },
  "paths": { "local_root": "/home/user/torch_npu", "remote_root": "/home/dev/torch_npu" },
  "exec_context": {
    "type": "bare",
    "env_setup": "source /home/dev/env.sh && conda activate torch_npu_dev && source /usr/local/Ascend/ascend-toolkit/set_env.sh"
  },
  "build": { "python_version": "3.9" },
  "verify": {
    "command": "python -m pytest test/test_xxx.py -xvs",
    "timeout": 600,
    "env_vars": {},
    "output_tail_lines": 15
  }
}
```

## Phase 3: Preflight Check

```bash
python tools/remote_deploy_verify.py preflight --config deploy_context.json
```

Check the JSON output:
- If `ready` is `true` → proceed to sync.
- If `ready` is `false` → show `blocker` and `warnings` to user, ask them to fix, then retry.

**Interpreting preflight results:**

| Check | What it verifies | If failed |
|-------|------------------|-----------|
| `ssh_connection` | SSH key auth works | Guide user through `ssh-copy-id` |
| `remote_path` | Project directory exists on remote | Check `remote_root` in config |
| `remote_python` | Python version available | Verify conda env / python path |
| `remote_cann` | CANN toolkit installed | Check `set_env.sh` in `env_setup` |
| `remote_git_clean` | No uncommitted changes on remote | Suggest `git stash` on remote |
| `cmake_cache` | CMakeCache.txt exists (for incremental builds) | Will need `--mode full` |

## Phase 4: Sync Code

First do a dry run for user confirmation:

```bash
python tools/remote_deploy_verify.py sync --config deploy_context.json --dry-run
```

Show the planned file list to the user. After confirmation:

```bash
python tools/remote_deploy_verify.py sync --config deploy_context.json
```

**Sync strategies** (default: `patch`):
- `patch`: Generate git diff patch and apply on remote (recommended for consistency)
- `scp`: Direct file copy for small changes (≤3 files)
- `rsync`: Bulk sync for many files

The `patch` strategy ensures remote code matches local exactly via `git diff`. It tracks:
- **New files**: Files created during sync (will be deleted on rollback)
- **Modified files**: Existing files changed (will be restored via `git checkout`)
- **Original state**: Commit hash, dirty status, untracked files (for complete restoration)

Verify `all_ok` is `true` in the output. Sync state is saved to `deploy_logs/sync_state.json`.

### Syncing extra files (outside git diff)

Use `--files` to add files beyond git changes:

```bash
# Sync additional files by relative path (uses _resolve_sync_path)
python tools/remote_deploy_verify.py sync --config deploy_context.json --files path/to/file.py

# Sync with explicit local:remote mapping (for non-standard locations)
python tools/remote_deploy_verify.py sync --config deploy_context.json \
  --files "code/pytorch/torch/testing/_internal/common_device_type.py:/home/user/.conda/envs/pt/lib/python3.9/site-packages/torch/testing/_internal/common_device_type.py"
```

Or configure persistent extra files in `deploy_context.json` (see [Advanced Config](#advanced-config)).

## Phase 5: Build

Skip if `collect` output shows `needs_compile: false`.

Build mode is chosen by the agent based on `collect` output:
- `skip`: only test/python files changed
- `incremental`: C++ changed + cmake cache exists
- `full`: config changed or no cmake cache
- `develop`: quick iterative cycle
- `custom`: uses `build.command` from config

```bash
python tools/remote_deploy_verify.py build --config deploy_context.json --mode incremental
```

Check the JSON output:
- If `success` is `true` → proceed to run.
- If `success` is `false` → read the `errors` array and `log_tail`, analyze the build failure. Determine if the error is from the user's fix or a pre-existing issue. Suggest corrections, then loop back to sync after the user applies the fix.

## Phase 6: Run Verification

```bash
python tools/remote_deploy_verify.py run --config deploy_context.json
```

Override the verification command on-the-fly:

```bash
python tools/remote_deploy_verify.py run --config deploy_context.json \
  --command "python -m pytest test/test_foo.py -k test_bar -xvs"
```

> **Important**: Never run `import torch_npu` from within the torch_npu source directory. This causes circular import errors. The verification command should always execute from a different directory (e.g., `cd /home/dev/workspace && python -m pytest ...`).

Check the JSON output:
- If `verdict` is `"pass"` → summarize results and return to Stage 4.
- If `verdict` is `"fail"` → read `output_tail` and analyze: is the original issue still present, or is this a new regression? Provide guidance.
- If `verdict` is `"timeout"` → suggest increasing timeout or simplifying the test.

**`output_tail` behavior**: By default, 15 lines of output are shown. On failure/timeout, this automatically increases to at least 30 lines. Configure via `verify.output_tail_lines` in `deploy_context.json`.

After verification, synthesize a summary from all phase JSON outputs (collect, preflight, sync, build, run). Then return to Stage 4 main flow:
- If verification passed → confirm fix and close.
- If verification failed → carry new evidence back into diagnosis loop.

## Quick Path: sync-run

For iterative test-fix-test cycles (no C++ changes), use `sync-run` to combine sync and run into one step:

```bash
python tools/remote_deploy_verify.py sync-run --config deploy_context.json
```

This is equivalent to running `sync` then `run` sequentially, but produces a single combined JSON output. Supports the same `--strategy`, `--files`, `--command`, and `--timeout` flags.

**When to use `sync-run`:**
- Test-only or Python-only changes (no build step needed)
- Iterative debugging where you want fast feedback
- Agent-driven loops where you modify a file and immediately want to see results

## Utility: exec

Run any command on the remote server without logging or pytest parsing:

```bash
python tools/remote_deploy_verify.py exec --config deploy_context.json "pip show torch_npu"
python tools/remote_deploy_verify.py exec --config deploy_context.json "cat /proc/driver/npu/version"
python tools/remote_deploy_verify.py exec --config deploy_context.json "python -c 'import torch; print(torch.__version__)'"
```

Useful for:
- Quick environment checks during diagnosis
- Inspecting remote file contents or directory structure
- Verifying package versions before syncing

## Rollback

If the remote changes need to be reverted:

```bash
python tools/remote_deploy_verify.py rollback --config deploy_context.json
```

**Rollback performs complete restoration:**
1. **Delete new files** created during sync (tracked in `sync_state.json`)
2. **Restore modified files** via `git checkout --`
3. **Reset to original commit** if needed (when remote was clean before sync)
4. **Clear sync state file** after successful rollback

**Auto-rollback**: Trigger automatic rollback after successful verification:

```bash
# Via CLI flag
python tools/remote_deploy_verify.py run --config deploy_context.json --auto-rollback

# Or in config file
{
  "verify": {
    "auto_rollback": true
  }
}
```

When `auto_rollback` is enabled and `verdict` is `"pass"`, rollback executes automatically. This is useful for:
- One-shot verification without leaving changes on remote
- CI/CD pipelines that need clean remote state
- Iterative testing where each run should start fresh

## Phase 5.5: Install (New)

After building, install the wheel package on the remote host:

```bash
python tools/remote_deploy_verify.py install --config deploy_context.json
```

Or specify a wheel file explicitly:

```bash
python tools/remote_deploy_verify.py install --config deploy_context.json --wheel dist/torch_npu-2.7.1-cp310-cp310-linux_aarch64.whl
```

JSON output:
```json
{
  "status": "ok",
  "wheel_file": "/home/dev/torch_npu/dist/torch_npu-2.7.1-cp310-cp310-manylinux_2_28_aarch64.whl",
  "wheel_name": "torch_npu-2.7.1-cp310-cp310-manylinux_2_28_aarch64.whl",
  "success": true,
  "elapsed_seconds": 43.1
}
```

## Advanced Config

### Docker Build Mode

For torch_npu, the recommended build flow is: **compile in Docker container → install on host**. Enable Docker build mode by adding a `docker` config:

```json
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
```

| Field | Description |
|-------|-------------|
| `image` | Docker image to use for compilation |
| `mounts` | Volume mounts (host:container format) |
| `container_name` | Container name (auto-generated if not specified) |
| `auto_cleanup` | Remove container after build completes (default: true) |

When Docker mode is enabled, the `build` command will:
1. Create a new container from the specified image
2. Execute the build command inside the container
3. Clean up the container (if `auto_cleanup: true`)
4. The wheel file remains in the host's `dist/` directory (via volume mount)

### Extra sync mappings

For files that need to be synced to non-standard remote locations (e.g., site-packages), add `sync.extra_mappings` or `sync.extra_files` to `deploy_context.json`:

```json
{
  "sync": {
    "extra_mappings": [
      {
        "local_pattern": "code/pytorch/torch/**",
        "remote_base": "/home/user/.conda/envs/pt/lib/python3.9/site-packages/torch"
      }
    ],
    "extra_files": [
      {
        "local": "code/pytorch/torch/testing/_internal/common_device_type.py",
        "remote": "/home/user/.conda/envs/pt/lib/python3.9/site-packages/torch/testing/_internal/common_device_type.py"
      }
    ]
  }
}
```

- **`extra_mappings`**: Pattern-based. Any file matching `local_pattern` is synced under `remote_base` with the pattern prefix stripped.
- **`extra_files`**: Explicit pairs. Each entry maps one local file to one remote path.

Both are merged with git-diff files during `sync` and `sync-run`.

### Output tail configuration

Control how many lines of output are shown in results:

```json
{
  "verify": {
    "output_tail_lines": 20
  }
}
```

On failure/timeout, this is automatically increased to `max(configured, 30)`.

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| SSH connection failed | Network / key / port issue | This tool only supports key-based auth (no password). Check: `ssh -vvv user@host`, verify `~/.ssh/id_rsa` exists and has `chmod 600`, confirm public key is in remote `~/.ssh/authorized_keys`. Run `ssh-copy-id user@host` to deploy key. |
| rsync permission denied | Remote directory permissions | Check remote dir owner |
| Files synced to wrong path | Local/remote root mismatch | Verify `local_root` and `remote_root` correspondence. Use `sync --dry-run` to preview. |
| Windows backslashes in paths | Running on Windows | Paths are auto-normalized to POSIX format. If issues persist, use forward slashes in config. |
| Build can't find dependencies | Remote env not set up | Confirm `source set_env.sh`, check CANN/Python paths |
| Build OK but old error at runtime | Build artifacts not installed | Use `--mode develop` or re-`pip install -e .` |
| New .cpp files not picked up | cmake glob not refreshed | Use `--mode full` to re-run cmake configure |
| pytest summary shows `parse_failed` | Non-standard output format | Check raw output in log file. The parser handles `passed/failed/error/deselected/skipped/xfailed/xpassed/warnings`. |
| Need to sync to site-packages | Framework files outside torch_npu tree | Use `sync.extra_files` or `sync.extra_mappings` in config, or `--files local:remote` on CLI |
| `import torch_npu` fails with circular import | Running in torch_npu source directory | **Never import torch_npu from within the source directory**. Change to a different directory (e.g., `cd /tmp`) before importing. This is a known limitation — the local source takes precedence over the installed package. |
| `import torch_npu` causes core dump | Build environment mismatch | Compile in Docker container on the remote server, then install and verify on the remote server host (not inside container). Direct host compilation may cause runtime issues. |

## See Also

- [Backend Diagnosis](backend-diagnosis.md) — Detailed diagnosis steps for CANN/torch_npu layers
- [Error Codes](error-codes.md) — Error code lookup tables
