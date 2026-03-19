# Regression Eval Execution Plan

How to actually run regression eval cases and collect results for `pta-failure-analyze`.

## 1. Existing Infrastructure

| Component | Location | Capability | Reuse for pta |
|-----------|----------|-----------|---------------|
| `run_regression_eval.py` | `ms-failure-analyze/tools/` | Execute evals via `claude -p` with skill injected | **Copy + adapt** prompt template |
| `grade_regression.py` | `ms-failure-analyze/tools/` | 5-layer deterministic code-based grading | **Direct reuse** (~99%) |
| `run_pipeline.py` | `ms-failure-analyze/tools/` | One-click execute → grade → aggregate | **Copy + adapt** skill_name |
| `grader.md` | `.cursor/skills/skill-creator/agents/` | LLM Grader for semantic checks | **Reusable** for L3/L4 |
| `aggregate_benchmark.py` | `.cursor/skills/skill-creator/scripts/` | Aggregate grading results | **Reusable** |
| `generate_review.py` | `.cursor/skills/skill-creator/eval-viewer/` | HTML result viewer | **Reusable** |

## 2. Execution Architecture

```
                    regression-evals.json
                           │
                    ┌──────▼─────────┐
                    │  Eval Runner   │  ← Per-eval execution
                    │  (orchestrator)│
                    └──────┬─────────┘
                           │
              ┌────────────┼─────────────┐
              ▼            ▼             ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ Executor │  │ Executor │  │ Executor │  ← Parallel workers
        │ (+ skill)│  │ (+ skill)│  │ (+ skill)│
        └────┬─────┘  └────┬─────┘  └────┬─────┘
             │             │             │
             ▼             ▼             ▼
        transcript    transcript    transcript
             │             │             │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │Code-based│  │Code-based│  │Code-based│  ← Deterministic grading
        │ Grader   │  │ Grader   │  │ Grader   │
        └────┬─────┘  └────┬─────┘  └────┬─────┘
             │             │             │
             ▼             ▼             ▼
        grading.json  grading.json  grading.json
             │             │             │
             └─────────────┼─────────────┘
                           ▼
                    ┌─────────────┐
                    │  Aggregator │
                    │  + Reporter │
                    └──────┬──────┘
                           ▼
                    benchmark.json
                    + benchmark.md
```

## 3. Phased Execution Strategy

### Phase 1: Manual Execution + Manual Grading (Immediate, 0 development)

**Applicable**: First 5-10 evals trial run to verify eval quality.

**Steps:**

1. **Prepare**: Open Cursor, ensure `pta-failure-analyze` skill is configured.

2. **Run**: Paste each regression eval's `prompt` into Cursor conversation:
   ```
   I'm training a model on Ascend 910B and getting a memory error:
   RuntimeError: Device memory exhausted. error code: 207018
   EL0004: device memory not enough
   Environment: PyTorch 2.1.0, torch_npu 2.1.0, CANN 8.0.RC3, Python 3.9
   ```

3. **Collect transcript**: Save complete conversation as markdown file.

4. **Manual grading**: Check against assertions layer by layer:
   - Layer 1: Does output contain `207018` / `OOM` / `EL0004`? Identified as ascend?
   - Layer 2: Classified as `platform` failure?
   - Layer 3: Root cause mentions HBM memory exhaustion?
   - Layer 4: Solution includes reduce batch size / gradient checkpointing?
   - Layer 5: Referenced failure-showcase? Asked user to verify?

5. **Record results**: Fill in pass/fail table.

**Pros**: Zero development cost, immediately executable, validates eval quality.
**Cons**: Not scalable for full eval set.

### Phase 2: Semi-automated Execution + Auto Code-based Grading (Requires minor development)

**Applicable**: Full ~21 regression evals batch execution.

**Components to develop/adapt:**

#### Component 1: Eval Executor (adapted from ms-failure-analyze)

Copy `tools/run_regression_eval.py` and modify the `EXECUTOR_PROMPT_TEMPLATE`:

```
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
```

#### Component 2: Code-based Grader (direct reuse)

Copy `tools/grade_regression.py` — the 5-layer grading logic is entirely generic:
- Layer 1: keyword match + backend match
- Layer 2: failure_type enum match
- Layer 3: root_cause keyword partial match
- Layer 4: solution keyword partial match
- Layer 5: showcase reference + validation question

Only minor keyword adjustment may be needed in Layer 5 (validation question patterns).

#### Component 3: Pipeline (adapted from ms-failure-analyze)

Copy `tools/run_pipeline.py` and change:
- `skill_name` default to `pta-failure-analyze`
- Evals path default to `evals/regression-evals.json`

#### Component 4: Aggregator (inlined in run_pipeline.py)

Already included in `run_pipeline.py` — generates `benchmark.json` and `benchmark.md`.

### Phase 3: Full Automated CI Pipeline (Long-term goal)

**Applicable**: Auto-run full regression evals after each skill update.

```
[skill update / PR submit]
         │
         ▼
[CI trigger: run regression evals]
         │
    ┌────▼──────┐
    │ claude -p │  ← Batch execute via Claude CLI
    │ + skill   │
    └────┬──────┘
         │
    ┌────▼─────────┐
    │ code grader  │  ← Python script auto-grading
    │ + LLM grader │
    └────┬─────────┘
         │
    ┌────▼───────┐
    │ benchmark  │  ← Generate report + compare with last run
    │ + report   │
    └────┬───────┘
         │
    ┌────▼──────────┐
    │ CI gate:      │
    │ regression    │  ← pass rate < threshold → block
    │ check         │
    └───────────────┘
```

## 4. Recommended Execution Path

```
              Now                      1 week later          2-3 weeks later
               │                        │                    │
Phase 1 ───────┤                        │                    │
Manual 5-10    │                        │                    │
Verify eval    │                        │                    │
quality        │                        │                    │
               │                        │                    │
Phase 2 ────────────────────────────────┤                    │
Adapt tools from ms                     │                    │
Batch execute ~21 evals                 │                    │
Generate first benchmark                │                    │
               │                        │                    │
Phase 3 ─────────────────────────────────────────────────────┤
Integrate to CI                                              │
Full automated regression gate                               │
```

## 5. Key Considerations

### Environment Isolation

Each eval's executor must start from a clean state:
- **No shared conversation context**: Each executor is independent
- **No shared failure-showcase modifications**: Evals should only test through Stage 2 (provide solution + ask verification), not enter Stage 3 (write showcase)

### Non-determinism Handling

Same eval may produce different results across runs:
- Phase 1 (manual): 1 run per eval, for eval quality validation
- Phase 2 (batch): Key evals run 3 times, use pass^3
- Phase 3 (CI): 1 run per eval (cost consideration), but track historical trends

### Cost Estimates

| Phase | Eval count | Trials each | Total runs | Est. tokens/run | Total tokens |
|-------|-----------|------------|-----------|----------------|-------------|
| Phase 1 | 5 | 1 | 5 | ~8K | ~40K |
| Phase 2 | 21 | 1 | 21 | ~8K | ~168K |
| Phase 2 (3 trials) | 21 | 3 | 63 | ~8K | ~504K |
| Phase 3 (CI per run) | 21 | 1 | 21 | ~8K | ~168K |

## 6. Output File Structure

```
skills/pta-failure-analyze/
├── tools/
│   ├── README.md                    ← Usage documentation
│   ├── run_regression_eval.py       ← Adapted from ms (prompt template change)
│   ├── grade_regression.py          ← Direct copy from ms (generic)
│   └── run_pipeline.py              ← Adapted from ms (skill_name change)
├── evals/
│   ├── evals.json                   ← Capability evals
│   └── regression-evals.json        ← Regression evals (~21 entries)
└── workspace/                       ← Generated at runtime
    └── regression-run-{timestamp}/
        ├── eval-reg_001/
        │   └── with_skill/
        │       └── run-1/
        │           ├── transcript.md
        │           └── grading.json
        ├── grading_summary.json
        ├── benchmark.json
        └── benchmark.md
```

## 7. Deliverables Checklist

| Phase | Deliverable | Format |
|-------|------------|--------|
| Phase 1 | 5 manual grading results + eval quality feedback | markdown table |
| Phase 2 | `tools/run_regression_eval.py` (adapted) | Python script |
| Phase 2 | `tools/grade_regression.py` (reused) | Python script |
| Phase 2 | `tools/run_pipeline.py` (adapted) | Python script |
| Phase 2 | ~21 transcripts | markdown files |
| Phase 2 | ~21 grading.json files | JSON |
| Phase 2 | Regression benchmark baseline | benchmark.json + benchmark.md |
| Phase 3 | CI configuration | GitHub Actions YAML |
