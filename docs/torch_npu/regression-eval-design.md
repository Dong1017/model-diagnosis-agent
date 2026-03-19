# Regression Eval Generation Design

Design for auto/semi-auto generation of regression evals from `failure-showcase.md` historical cases.

## 1. Data Source Analysis

### Entry Count and Distribution

`failure-showcase.md` contains **19 entries**, all seed entries in the `## Common Failure Patterns` section:

| Classification | Count | Characteristics |
|---------------|-------|----------------|
| Seed entries | 19 | Generic error patterns, `observed_at` is empty or "N/A - seed entry" |
| Observed entries | 0 | No real observed failures yet (section exists but empty) |

### Failure Type Distribution

| Failure Type | Count | Examples |
|-------------|-------|---------|
| cann | 8 | InstanceNorm precision, CANN inner error, HCCL timeout, kernel not found, scalar mismatch, overflow, format error, DO_COMPATIBILITY |
| framework | 4 | Delayed execution warning, context empty, feature not supported, stream context, operator not registered |
| platform | 3 | OOM, device task abort, version mismatch |
| scripts | 2 | Missing CANN environment, test framework device detection |

### Entry Fields

```
- failure_info:   Error keywords/context description
- observed_at:    Observation location (test case name or "N/A - seed entry")
- failure_type:   platform | scripts | framework | cann
- root_cause:     Root cause description
- solution:       Actionable steps
- last_seen:      Last observation date
- occurrences:    Occurrence count
```

## 2. Entry Classification and Applicability

### Classification Criteria

| Class | Criteria | Eval Applicability | Handling |
|-------|---------|-------------------|---------|
| **A: High quality** | root_cause is specific + solution is actionable + failure_info has clear error features | High | Generate eval directly |
| **B: Usable** | root_cause is somewhat specific but solution is brief, or failure_info keywords not distinctive enough | Medium | Need manual prompt detail supplement |
| **C: Low quality** | root_cause is vague (e.g., "PR merged"), solution is "Fixed" | Low | Skip, or generate basic "can identify failure_type" eval only |

### Screening Results for Current 19 Entries

| Class | Count | Examples |
|-------|-------|---------|
| A | ~14 | OOM (clear error codes + specific solution), Device Task Abort (107010 + hardware diagnosis), Kernel Not Found (561003 + installation steps) |
| B | ~4 | CANN Inner Error (generic E[x]9999), DO_COMPATIBILITY fallback (subtle issue) |
| C | ~1 | Delayed Execution Warning (informational, not really a failure to diagnose) |

**First batch target**: Generate ~14 L1 evals from A-class + ~4 L2 variants from select A-class entries + ~3 B-class evals = **~21 evals total**.

## 3. Eval Generation Strategy

### Seed Entry → Synthesized User Report Prompt

Seed entries have keyword-style `failure_info`, not real error logs. Keywords must be **expanded into realistic user reports**.

**Template structure:**

```
I'm running torch_npu on {device} and encountering the following error:

{synthesized_error_log}

Environment: PyTorch {pytorch_version}, torch_npu {torch_npu_version}, CANN {cann_version}, Python {python_version}

{optional_context}
```

**Synthesis rules:**

| Field | Synthesis Method |
|-------|-----------------|
| `device` | "Ascend 910B" (all entries are Ascend-only) |
| `pytorch_version` | Use "2.1.0" or similar recent version |
| `torch_npu_version` | Use matching torch_npu version |
| `cann_version` | Use "8.0.RC3" or from entry's last_seen context |
| `synthesized_error_log` | Assemble from failure_info keywords: error code + exception type + key descriptors |
| `optional_context` | Reverse-infer from root_cause what user might describe |

**Example:**

Seed entry `Out of Memory (OOM)`:
```
failure_info: "EL0004, FAIL_TO_ALLOCATE_MEMORY, 200000, 207018, out of memory"
```

Synthesized prompt:
```
I'm training a model on Ascend 910B and getting a memory error:

RuntimeError: Device memory exhausted.
[ERROR] RUNTIME: memory allocation failed, error code: 207018
EL0004: device memory not enough

Batch size is 64, model has ~7B parameters.

Environment: PyTorch 2.1.0, torch_npu 2.1.0, CANN 8.0.RC3, Python 3.9
```

## 4. Assertion (Grader) Design

Each regression eval contains **layered assertions**, from easy to hard, supporting partial credit.

### Assertion Layers

```
Layer 1: Error Identification (must pass)
  ├── error_keywords:  Output contains key error codes/keywords
  └── backend_match:   Correctly identifies backend (ascend)

Layer 2: Classification (should pass)
  ├── failure_type:    Correctly classifies to platform/scripts/framework/cann
  └── quick_route:     Reasonable Quick Route jump level

Layer 3: Root Cause Analysis (core validation)
  ├── root_cause_keywords: Output contains root cause core keywords
  └── root_cause_semantic: Root cause description semantically correct (model-based)

Layer 4: Solution (completeness validation)
  ├── solution_keywords:  Output contains solution key actions
  └── solution_actionable: Solution is actionable (model-based)

Layer 5: Process Compliance (behavior validation)
  ├── showcase_referenced: Referenced failure-showcase matching entry
  └── user_validation:     Asked user to verify (contains validation question)
```

### Scoring Weights

| Layer | Weight | Description |
|-------|--------|-------------|
| Layer 1: Error Identification | 20% | Basic threshold, all wrong = 0 |
| Layer 2: Classification | 15% | |
| Layer 3: Root Cause Analysis | 30% | Core value |
| Layer 4: Solution | 25% | |
| Layer 5: Process Compliance | 10% | |

### Assertion Extraction Rules

Auto-extract assertions from each showcase entry:

| Assertion Field | Source | Extraction Method |
|----------------|--------|------------------|
| `error_keywords` | `failure_info` | Extract error codes (6-digit numbers, E[A-Z]xxxx, ERRxxxxx) and exception types |
| `backend_expected` | Always "ascend" for pta | Fixed value |
| `failure_type_expected` | `failure_type` | Direct value |
| `root_cause_keywords` | `root_cause` | Extract core noun phrases, top 3-5 keywords |
| `solution_keywords` | `solution` | Extract action keywords (commands, function names, config changes) |

## 5. Eval Entry Structure

### JSON Schema

```json
{
  "id": "reg_001",
  "source": {
    "showcase_title": "Out of Memory (OOM)",
    "showcase_section": "Common Failure Patterns",
    "entry_type": "seed"
  },
  "prompt": {
    "en": "I'm training a model on Ascend 910B and getting a memory error...",
    "zh": "I'm training a model on Ascend 910B and getting a memory error..."
  },
  "environment": {
    "pytorch_version": "2.1.0",
    "torch_npu_version": "2.1.0",
    "backend": "ascend",
    "cann_version": "8.0.RC3"
  },
  "assertions": {
    "layer1_identification": {
      "error_keywords": ["207018", "OOM", "out of memory", "EL0004"],
      "error_keywords_match": "any",
      "backend_expected": "ascend"
    },
    "layer2_classification": {
      "failure_type_expected": "platform",
      "quick_route_level": "Platform"
    },
    "layer3_root_cause": {
      "keywords": ["HBM", "memory exhausted", "batch size", "device memory"],
      "keywords_match_min": 2,
      "semantic_rubric": "Root cause should mention device memory (HBM) exhaustion related to model size, batch size, or memory management"
    },
    "layer4_solution": {
      "keywords": ["batch size", "gradient checkpointing", "empty_cache"],
      "keywords_match_min": 2,
      "semantic_rubric": "Solution should include at least one concrete memory reduction strategy"
    },
    "layer5_process": {
      "showcase_referenced": true,
      "user_validation_asked": true
    }
  },
  "scoring": {
    "layer1_weight": 0.20,
    "layer2_weight": 0.15,
    "layer3_weight": 0.30,
    "layer4_weight": 0.25,
    "layer5_weight": 0.10,
    "pass_threshold": 0.70
  },
  "metadata": {
    "difficulty": "L1",
    "backend": "ascend",
    "failure_type": "platform",
    "generated_from": "failure-showcase.md",
    "created_at": "2026-03-18"
  }
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier, `reg_` prefix + sequence number |
| `source` | object | Traces back to specific showcase entry |
| `prompt` | object | Synthesized user report (en/zh) |
| `environment` | object | Simulated environment info, for grader reference |
| `assertions.layer*` | object | Layered assertions, code-based + model-based hybrid |
| `scoring` | object | Layer weights and pass threshold |
| `metadata.difficulty` | string | L1 (direct match) / L2 (requires inference) |

## 6. Difficulty Levels

Even for regression evals, difficulty varies:

| Difficulty | Definition | Prompt Characteristics | Expected Pass Rate |
|-----------|-----------|----------------------|-------------------|
| **L1** | Prompt directly contains showcase entry's core error code/keywords | Error info is explicit, environment complete | ≥95% |
| **L2** | Prompt uses synonymous descriptions or partial info, requires Agent inference | Error info has variants, or partial environment info missing | ≥80% |

### L1 vs L2 Generation Differences

Using `Device Task Abort (FORCE STOP)` as example:

**L1 prompt** (directly contains 107010):
```
Training process errored:
RuntimeError: NPU function error: FORCE STOP
retCode=107010, device task abort, reason=device task abort
Environment: PyTorch 2.1.0, torch_npu 2.1.0, CANN 8.0.RC3, Ascend 910B
```

**L2 prompt** (describes symptoms without direct error code):
```
Training a large model on Ascend, ran a few steps then it hung. After a while
got a device hang error, logs mention something about task abort and heartbeat.
Environment: PyTorch 2.1.0, CANN 8.0.RC3
```

**Recommended ratio**: 1 L1 eval per showcase entry. Additional ~4 L2 variants from high-quality A-class entries.

## 7. Generation Flow

### Step 1: Parse failure-showcase.md

```
Input: failure-showcase.md
Output: Structured entry list [{ title, failure_info, failure_type, root_cause, solution, ... }]
```

Parse rules:
- Lines starting with `### ` are entry titles
- `- field_name: "value"` format extracts fields

### Step 2: Quality Screening

| Rule | Condition | Result |
|------|----------|--------|
| root_cause length < 20 chars | e.g., "PR merged" | → C class |
| solution only contains "Fixed" / "Fixed in xxx" | No actionable steps | → C class |
| root_cause contains specific error code or technical term | e.g., "aclnn rejects duplicate dims" | → A class |
| failure_info contains matchable error code | e.g., "561003", "107010" | → A class |
| Other | | → B class |

### Step 3: Prompt Synthesis

Apply prompt template based on entry type:

| Entry Type | Template Elements |
|-----------|------------------|
| Seed (all Ascend) | Device "Ascend 910B", CANN version, torch_npu version |
| Observed (with test name) | "torch_npu test {test_name} failed on Ascend" + environment variables |
| Observed (no test name) | Treat as seed template with observed_at context |

### Step 4: Assertion Extraction

Auto-extract code-based assertions from entry fields:

```
From failure_info → error_keywords (regex: \d{6}, E[A-Z]\d{4}, ERR\d{5}, exception class names)
From backend     → backend_expected (always "ascend" for pta)
From failure_type → failure_type_expected
From root_cause  → root_cause_keywords (rule-based top 3-5 noun phrases)
From solution    → solution_keywords (action verbs and tool names)
```

### Step 5: Assemble Eval JSON

Combine prompt + assertions + scoring + metadata into final eval entry.

### Step 6: Human Review

**Must review:** B-class entry prompts, assertion keyword coverage, L2 variant prompt quality.
**Can skip:** A-class L1 evals (auto-generated quality is high), structural fields.

## 8. Output Estimates

| Class | Count | Generation Method |
|-------|-------|------------------|
| A-class L1 eval | ~14 | Fully automated |
| A-class L2 variant | ~4 | Semi-automated (manual prompt fuzzing) |
| B-class L1 eval (reviewed) | ~3 | Semi-automated (manual prompt supplementation) |
| **Total first batch** | **~21** | |

### Coverage Matrix

First batch ~21 evals distribution target:

| | Platform | Scripts | Framework | CANN | Total |
|--|----------|---------|-----------|------|-------|
| **Ascend** | ~3 | ~2 | ~5 | ~11 | ~21 |

## 9. Eval File Structure

```
skills/pta-failure-analyze/
  evals/
    evals.json                  ← Capability evals (new)
    regression-evals.json       ← Regression evals (this design's output)
```

`regression-evals.json` uses the extended schema defined in this design (with layered assertions and scoring).

## 10. Execution Plan

| Phase | Task | Estimated Effort |
|-------|------|-----------------|
| Phase 1 | Parse failure-showcase.md → structured entry list | 0.5 day |
| Phase 2 | Quality screening (A/B/C classification) | 0.5 day |
| Phase 3 | A-class auto-generate ~14 L1 evals | 0.5 day |
| Phase 4 | Select ~4 A-class for L2 variants + manual adjust | 0.5 day |
| Phase 5 | B-class manual supplement + generate ~3 evals | 0.5 day |
| Phase 6 | Full review + trial run + corrections | 0.5 day |
| **Total** | **~21 regression evals** | **~3 days** |
