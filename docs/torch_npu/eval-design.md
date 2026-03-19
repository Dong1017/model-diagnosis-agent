# pta-failure-analyze Eval Design

Based on [Anthropic: Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents), adapted for the `pta-failure-analyze` skill — a diagnostic conversational Agent for torch_npu/Ascend NPU failures.

## 1. Current Status

The `pta-failure-analyze` skill currently has no eval infrastructure. The `failure-showcase.md` contains 19 seed entries that can serve as the basis for regression eval generation.

## 2. Evaluation Dimensions (6 Dimensions)

This skill is a **diagnostic conversational Agent** — it maintains state (diagnosis stage), uses tools (reads reference files), takes actions (updates knowledge base), and the interaction itself is an evaluation target.

### Dimension 1: Diagnostic Accuracy

> Core question: "Did the Agent correctly identify the root cause?"

This is the most important **outcome evaluation**.

| Check Item | Grader Type | Example |
|-----------|------------|--------|
| Error code / exception type identified correctly | Code-based (keyword match) | Output contains `107010` and classifies as "heartbeat lost" |
| Root cause located correctly | Model-based (rubric scoring) | Root cause description semantically matches reference answer |
| Failure layer located correctly | Code-based (enum match) | Output failure_type is correct within `platform/scripts/framework/cann` |
| Quick Route jump is reasonable | Model-based | Jumped to a reasonable level based on error characteristics |

**Partial credit design:**

| Completion | Score |
|-----------|-------|
| Identified correct error code but root cause analysis wrong | 60% |
| Error code and root cause correct but solution not actionable | 80% |
| All correct | 100% |

### Dimension 2: Process Compliance

> Core question: "Did the Agent follow the Stage 0→1→2→3 diagnostic workflow?"

Evaluation of the transcript (process trace). Following Anthropic's advice — **evaluate outcomes not paths** — we don't require rigid step ordering but check whether key behaviors appeared.

| Key Behavior | Grader Type | Detection Method |
|-------------|------------|-----------------|
| Stage 0: Proactively asks for missing info | Code-based (transcript) | When prompt lacks version/device info, output contains follow-up question |
| Stage 1: Checks Failure Showcase first | Code-based (tool_calls) | Read `failure-showcase.md` before analysis |
| Stage 2: Provides structured fix advice | Code-based (format check) | Output contains `Analysis/Root Cause/Solution` structure |
| Stage 2→3: Requests user verification before concluding | Code-based (string match) | Output contains validation question |
| Stage 3: Does not write to showcase before user confirmation | Code-based (state_check) | `failure-showcase.md` not modified before confirmation |

### Dimension 3: Knowledge Retrieval Effectiveness

> Core question: "Did the Agent effectively use the reference knowledge base?"

| Check Item | Grader Type | Description |
|-----------|------------|-------------|
| Reuses Failure Showcase match directly | Code-based | Should not re-analyze from scratch when historical case exists |
| Error Codes lookup is accurate | Code-based | Error code explanation matches `error-codes.md` |
| References correct reference files | Code-based (tool_calls) | Read corresponding reference sections based on error type |
| No hallucinated information | Model-based (groundedness) | Explanations can be traced back to reference files |

### Dimension 4: Solution Quality

> Core question: "Is the proposed solution actionable and accurate?"

| Check Item | Grader Type | Description |
|-----------|------------|-------------|
| Solution is actionable | Model-based (rubric) | Not vague "check config" but specific commands/code |
| Solution matches root cause | Model-based | Solution actually addresses the identified root cause |
| Diagnostic commands are correct | Code-based | `npu-smi` / env variable commands have correct syntax |
| Provides further location steps when uncertain | Model-based | When root cause unclear, gives CANN debug log steps rather than guessing |

### Dimension 5: Interaction Quality

> Anthropic specifically notes that for conversational Agents, the interaction itself is an evaluation target.

| Check Item | Grader Type | Description |
|-----------|------------|-------------|
| Reasonable number of turns | Code-based (transcript) | Should not exceed N turns for initial diagnosis |
| Information density | Model-based | Not redundant, not missing key info |
| Iterates when user says "not fixed" | Model-based (multi-turn eval) | Does not repeat same solution, adjusts with new evidence |
| Handles topic change properly | Model-based | Confirms original issue status first, then handles new topic |

### Dimension 6: Knowledge Accumulation Correctness

> Core question: "Is the experience written to Failure Showcase correct?"

This is a **state_check (environment state check)** dimension.

| Check Item | Grader Type | Description |
|-----------|------------|-------------|
| Only writes after user confirmation | Code-based (state_check) | Check showcase file modification timing |
| New entry format is correct | Code-based (schema validation) | All required fields present (failure_info, observed_at, etc.) |
| Duplicate entries are merged not duplicated | Code-based | Updates occurrences on matching entry, does not create copy |
| Written content matches diagnosis conclusion | Model-based | showcase root_cause/solution matches conversation conclusion |

## 3. Capability Eval vs Regression Eval

### Regression Eval (target pass rate ≈ 100%)

**Source**: Verified resolved Failure Showcase entries.

Each historical case is naturally a regression test case — use its `failure_info` as prompt and `root_cause + solution` as expected outcome. The current `failure-showcase.md` has 19 seed entries that can be converted to regression evals.

**Purpose**: Ensure skill updates don't "forget" known diagnosis capabilities.

### Capability Eval (initial pass rate should be low)

Design goal is to explore skill capability boundaries:

| Difficulty | Example Scenario |
|-----------|-----------------|
| L1 - Direct match | Failure Showcase has an exact matching known issue |
| L2 - Pattern inference | Error code is known but specific scenario is new (e.g., new operator's 561003) |
| L3 - Cross-layer | Surface error is CANN but root cause is version incompatibility (Platform layer) |
| L4 - Ambiguous/incomplete | User only gives truncated log, needs multi-turn probing to locate |
| L5 - Novel problem | Error pattern not in Showcase or Error Codes |

As capability eval pass rates improve, high-pass tasks can "graduate" to regression evals.

## 4. Balanced Problem Sets

Anthropic emphasizes **testing both should and shouldn't directions** to avoid one-sided optimization.

| Balance Dimension | "Should" Direction | "Shouldn't" Direction |
|------------------|-------------------|----------------------|
| Match situation | Showcase has match → reuse directly | Showcase has no match → should not force-fit wrong case |
| Information completeness | Info complete → diagnose directly | Info insufficient → should ask, not guess |
| Root cause certainty | Root cause clear → give solution | Root cause unclear → should give further location steps, not guess |
| User confirmation | User confirms fix → write to showcase | User has not confirmed → should not write |
| Stage skip | Info sufficient → can skip Stage 0 | Should not skip Stage 0 when info is insufficient |

### Failure Type Coverage Matrix

Ensure eval set has reasonable distribution across this matrix (pta-failure-analyze is Ascend-only):

| | Platform | Scripts | Framework | CANN |
|--|----------|---------|-----------|------|
| **Ascend** | HW fault (107010/ECC/OOM) | env config, device check | operator registration, privateuse1 | CANN/ACLNN error codes, precision |

## 5. Grader Selection Strategy

Following Anthropic's principle: **use deterministic grader when possible, LLM grader only when necessary**.

### Priority 1: Code-based (deterministic, fast, cheap, reproducible)

- Error code identification (keyword match)
- Failure type classification (enum match)
- Process key behaviors (tool_calls / string match)
- Showcase write format (schema validation)
- Showcase write timing (state_check)
- Output structured format (regex/template match)

### Priority 2: Model-based (LLM Judge, flexible, handles open-ended output)

- Root cause semantic correctness (rubric scoring)
- Solution actionability (rubric scoring)
- Interaction quality (rubric scoring)
- Groundedness check (reference verification)

### Priority 3: Human (expert review, gold standard calibration)

- Calibrate LLM Judge rubrics
- Periodic spot-check of complex diagnosis quality
- Verify L4/L5 difficulty eval task design reasonableness

## 6. Non-determinism Handling

Due to model output randomness, multiple trials are needed for reliable signals:

| Metric | Meaning | Applicable Scenario |
|--------|---------|-------------------|
| **pass@k** | Probability of at least 1 success in k attempts | Diagnostic scenarios — success if can diagnose at all |
| **pass^k** | Probability of all k attempts succeeding | Process compliance — should comply every time |

Recommendations:
- Diagnostic accuracy: use **pass@3** (at least 1 correct in 3 attempts)
- Process compliance: use **pass^3** (all 3 must comply)
- Regression eval: use **pass^3** (known issues must pass stably)

## 7. Improvement Roadmap

### Phase 1: Quick Start (1-2 days)

- [ ] Auto-generate ~19 regression evals from Failure Showcase seed entries
- [ ] Add code-based assertions (keyword match + enum match) to each eval
- [ ] Design 4-5 capability evals covering different failure types

### Phase 2: Dimension Expansion (3-5 days)

- [ ] Redesign eval graders per 6-dimension system
- [ ] Add multi-turn interaction evals (probing, iterative diagnosis, topic switching)
- [ ] Add "shouldn't" direction evals (shouldn't guess, shouldn't skip probing, shouldn't prematurely write showcase)
- [ ] Design partial credit scoring rules

### Phase 3: LLM Judge Calibration (5-7 days)

- [ ] Write rubric prompts for root cause / solution / interaction quality LLM scoring
- [ ] Calibrate LLM Judge with 10-20 human review results
- [ ] Compare LLM Judge vs human review consistency (target > 85% agreement)

### Phase 4: Continuous Operation

- [ ] After each new diagnosis, add verified cases to eval set
- [ ] Monitor capability eval saturation, add higher difficulty tasks
- [ ] Periodically "graduate" high-pass capability evals to regression evals
- [ ] Periodic human spot-check of transcripts to verify grader effectiveness
