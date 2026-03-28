# pta-failure-analyze Tools

Tools for regression evaluation and remote deploy & verify for the `pta-failure-analyze` skill.

## Prerequisites

- Python 3.10+
- Claude Code CLI (`npm install -g @anthropic-ai/claude-code`) — for eval tools
- SSH client — for remote deploy & verify

## Quick Start

### One-click pipeline (recommended)

```bash
# Sample 5 random evals, execute + grade + aggregate
python tools/run_pipeline.py --sample 5 --verbose

# Run specific evals
python tools/run_pipeline.py --ids reg_003,reg_005 --lang en --verbose

# Run all L1 evals
python tools/run_pipeline.py --all --filter-difficulty L1 --verbose

# Run all evals in both languages
python tools/run_pipeline.py --all --lang both --verbose
```

### Step-by-step execution

```bash
# Step 1: Execute evals → collect transcripts
python tools/run_regression_eval.py \
    --evals evals/regression-evals.json \
    --sample 5 --lang zh --verbose

# Step 2: Grade transcripts → grading.json per eval
python tools/grade_regression.py \
    --workspace workspace/regression-run-2026-03-18_120000 \
    --evals evals/regression-evals.json --verbose

# Step 3: Re-grade an existing workspace via pipeline
python tools/run_pipeline.py \
    --existing-workspace workspace/regression-run-2026-03-18_120000 --verbose
```

## Tools

### `run_regression_eval.py`

Executes regression evals by sending each prompt to Claude via `claude -p` with the skill injected.

| Flag | Description |
|------|-------------|
| `--evals` | Path to regression-evals.json (required) |
| `--ids` | Comma-separated eval IDs |
| `--all` | Run all evals |
| `--sample N` | Random sample N evals |
| `--lang zh\|en\|both` | Prompt language (default: zh) |
| `--filter-type seed\|observed` | Filter by entry type |
| `--filter-difficulty L1\|L2` | Filter by difficulty |
| `--num-workers N` | Parallel workers (default: 4) |
| `--timeout N` | Timeout per eval in seconds (default: 600) |
| `--model` | Claude model override |
| `--workspace` | Custom workspace base directory |

### `grade_regression.py`

Grades transcripts against 5-layer assertions using deterministic code-based matching.

| Layer | What it checks | Method |
|-------|---------------|--------|
| L1 | Error identification | Keyword match + backend match |
| L2 | Classification | failure_type enum match |
| L3 | Root cause | Keyword partial match (min N of M) |
| L4 | Solution | Keyword partial match (min N of M) |
| L5 | Process compliance | Showcase reference + validation question |

| Flag | Description |
|------|-------------|
| `--workspace` | Workspace directory with eval-* dirs (required) |
| `--evals` | Path to regression-evals.json (required) |
| `--verbose` | Show per-eval grading details |

### `run_pipeline.py`

One-click pipeline: execute → grade → aggregate. Combines both tools above plus benchmark generation.

Additional flags beyond `run_regression_eval.py`:

| Flag | Description |
|------|-------------|
| `--skip-execute` | Skip execution, only grade + aggregate |
| `--existing-workspace` | Grade an existing workspace (implies --skip-execute) |

## Output Structure

```
workspace/
└── regression-run-{timestamp}/
    ├── run_metadata.json           # Execution metadata
    ├── grading_summary.json        # Aggregate grading results
    ├── benchmark.json              # Benchmark data
    ├── benchmark.md                # Human-readable benchmark report
    └── eval-{id}/
        ├── eval_metadata.json      # Eval case metadata
        └── with_skill/
            └── run-1/
                ├── transcript.md   # Full prompt + agent response
                ├── grading.json    # 5-layer grading result
                └── outputs/
                    └── eval_metadata.json
```

## Remote Deploy & Verify

`remote_deploy_verify.py` helps sync local code changes to a remote Ascend server, compile, and run verification tests. It is optionally triggered by the agent during Stage 4 (Validate and Close). See [remote-deploy-verify reference](../references/remote-deploy-verify.md) for the full workflow.

All subcommands output structured JSON to stdout for LLM consumption. The agent synthesizes reports from JSON outputs directly — no dedicated report subcommand needed.

### Quick Start

```bash
# 1. Analyze local changes
python tools/remote_deploy_verify.py collect --local-root /path/to/torch_npu

# 2. Preflight check (requires deploy_context.json)
python tools/remote_deploy_verify.py preflight --config deploy_context.json

# 3. Sync with dry-run preview, then execute
python tools/remote_deploy_verify.py sync --config deploy_context.json --dry-run
python tools/remote_deploy_verify.py sync --config deploy_context.json

# 4. Build remotely
python tools/remote_deploy_verify.py build --config deploy_context.json --mode incremental

# 5. Run verification
python tools/remote_deploy_verify.py run --config deploy_context.json

# 6. Rollback if needed
python tools/remote_deploy_verify.py rollback --config deploy_context.json
```

### `remote_deploy_verify.py` Subcommands

| Subcommand | Description |
|------------|-------------|
| `collect` | Analyze local git changes, classify files (cpp/config/test/python), recommend build mode |
| `preflight` | Single-SSH check: connectivity, remote path, Python, CANN, cmake cache, Docker |
| `sync` | Sync changed files to remote via rsync (supports `--dry-run`); auto-stashes remote state |
| `build` | Run remote compilation (`--mode`: incremental / full / develop / custom / skip) |
| `run` | Execute remote verification command, return exit code + output tail |
| `rollback` | Revert remote changes via `git checkout` + `git stash pop` |

### `deploy_context.json` Schema

```json
{
  "remote": { "host": "10.0.1.100", "user": "dev", "port": 22 },
  "paths": {
    "local_root": "/home/user/torch_npu",
    "remote_root": "/home/dev/torch_npu"
  },
  "exec_context": {
    "type": "bare | docker",
    "docker_container": null,
    "env_setup": "source /usr/local/Ascend/ascend-toolkit/set_env.sh",
    "host_mount_path": null
  },
  "build": {
    "python_version": "3.9",
    "command": null
  },
  "verify": {
    "command": "python -m pytest test/test_xxx.py::test_func -xvs",
    "timeout": 600,
    "env_vars": { "ASCEND_LAUNCH_BLOCKING": "1" }
  }
}
```

### Execution Context Types

| Type | When to use | How it works |
|------|-------------|-------------|
| `bare` | Direct SSH to physical/virtual machine | `ssh → env_setup → cd remote_root → cmd` |
| `docker` | Build environment is in a Docker container | `ssh → docker exec container bash -c 'env_setup && cd && cmd'` |

### Build Modes

| Mode | Command | When |
|------|---------|------|
| `skip` | _(none)_ | Only Python / test files changed |
| `incremental` | `make -j$(nproc)` in build dir | C++ source changed, cmake cache exists |
| `full` | `bash ci/build.sh --python=X.Y` | Config changed, no cmake cache, or first build |
| `develop` | `python setup.py develop` | Quick iterative development |
| `custom` | `build.command` from config | Special build requirements |

### File Classification

The `collect` subcommand classifies changed files into 4 types:

| Type | Matches | Build recommendation |
|------|---------|---------------------|
| `cpp` | `.cpp`, `.h`, `.hpp`, `.cc`, `.cxx` | incremental |
| `config` | `config/`, `CMakeLists.txt`, `setup.py`, `codegen/`, `gencode.sh`, `generate_code.sh` | full |
| `test` | `test/` prefix | skip |
| `python` | Everything else | skip |

---

## Eval Data

| File | Contents |
|------|---------|
| `evals/regression-evals.json` | 23 regression evals (19 L1 + 4 L2) with 5-layer assertions |
| `evals/evals.json` | 5 capability evals for boundary testing |

All prompts are bilingual (en/zh). Use `--lang` to select which language to test.
