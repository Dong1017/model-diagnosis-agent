# pta-failure-analyze Eval Tools

Tools for running, grading, and aggregating regression evals for the `pta-failure-analyze` skill.

## Prerequisites

- Python 3.10+
- Claude Code CLI (`npm install -g @anthropic-ai/claude-code`)

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

## Eval Data

| File | Contents |
|------|---------|
| `evals/regression-evals.json` | 23 regression evals (19 L1 + 4 L2) with 5-layer assertions |
| `evals/evals.json` | 5 capability evals for boundary testing |

All prompts are bilingual (en/zh). Use `--lang` to select which language to test.
