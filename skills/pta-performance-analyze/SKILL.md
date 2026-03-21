---
name: pta-performance-analyze
description: Use when users report NPU memory usage exceeding GPU, memory consistency issues, operator-level memory profiling, workspace overhead, aclnn memory allocation problems, or ask to diagnose/compare torch_npu vs torch_gpu single-operator memory consumption.
---

# PTA Memory Consistency Analyzer

Diagnoses why a torch_npu operator uses more device memory than the equivalent torch_gpu operator, and provides actionable fixes.

**Trigger keywords:** "memory overconsumption", "NPU memory higher than GPU", "memory consistency", "plog abnormal allocation", "diagnose operator memory", "run memory comparison"

## Pipeline Overview

```
Stage 1: Prepare & Collect ──→ Stage 2: Analyze Root Cause ──→ Stage 3: Validate & Accumulate
  execute_mem_test.py            plog + known issues + source     user confirmation + case file
  (auto: locate/convert/run)     code analysis
```

**Entry point routing:**

| User provides | Start from |
|---------------|------------|
| Script path or directory (no results yet) | Stage 1 |
| No explicit path | Stage 1 (use current working directory as input) |
| Both `mem_results_*.md` AND `filtered_plog_*.log` | Stage 2 |
| Only one of the above (incomplete) | Stage 1 (regenerate pair) |

**Always:** If the user provides a `pytorch_npu` source path at any point, record it for Stage 2 Step 2.

**Required output:** data evidence (plog values, memory comparison tables) + root cause attribution (torch_npu vs CANN) + actionable fix suggestions or code changes.

## Stage 1: Prepare Environment & Collect Data

Run the automation script. **Do NOT read the script's source code** — just execute and inspect output.

```bash
python <this_skill_directory>/tools/execute_mem_test.py <file_or_directory_path_provided_by_user>
```

**Interpret output:**
- `[SUCCESS]` → extract `api_name` and file paths from output, proceed to Check Results below
- `[SCRIPT_ERROR]` → the user's NPU/GPU test script has a functional or runtime issue. Show the error to the user, then ask:
  **"The test script has an issue. Would you like me to: (A) diagnose and fix it, or (B) skip the fix and proceed with whatever results are available?"**
  - User chooses A → read script source code, fix, re-run `execute_mem_test.py`
  - User chooses B → check if `mem_results_<api_name>.md` + `filtered_plog_<api_name>.log` already exist in the script directory. If both exist → proceed to Check Results. If missing → inform user and re-offer choice A.
- `[ERROR]` (without `SCRIPT_`) → a preset validation error (wrong path, bad filename format, etc.). Show error to user and **STOP** — do NOT continue to Stage 2

> **HARD GATE**: You MUST have `mem_results_<api_name>.md` + `filtered_plog_<api_name>.log` before entering Stage 2. Do NOT skip to known issue lookup based on the API name alone.

### Check Results

Read `mem_results_<api_name>.md`. Key metrics:

| Metric | Description |
|--------|-------------|
| **Memory Benchmark: torch.xxx** (heading) | Extract torch.xxx as `TARGET_API` |
| **total_driver_GB** | Actual driver-level memory delta |
| **reserved_GB** | PTA CachingAllocator reserved (NPU) / gpu_reserved_GB (GPU) |
| **activated_GB** | PTA CachingAllocator peak allocated (**most important**) |
| **Ratio** | NPU/GPU — above 1.05x warrants investigation |

**Early exit check** on activated_GB ratio:
- **≤ 1.05** → inform user memory is normal, ask to verify test script correctness, **skip Stage 2 & 3**
- **> 1.05** → proceed to Stage 2

## Stage 2: Analyze Root Cause

### Step 1: Analyze Plog & Match Known Issues

This step performs plog analysis and known-issue lookup together **before** presenting any conclusions to the user.

#### 1a. Plog Analysis

**Inputs (all required):**

| Source | What to extract |
|--------|----------------|
| `mem_results_<api_name>.md` — Key Code | torch API call scenario, input shape/dtype → estimate expected memory baseline |
| `mem_results_<api_name>.md` — Table | NPU vs GPU metrics (total_driver_GB, reserved_GB, activated_GB, ratio) |
| `filtered_plog_<api_name>.log` — `[Summary]` | `Workspace allocs`: `#N: ... bytes ... \| op: aclnnXXX` |
| `filtered_plog_<api_name>.log` — Events | `PTA CachingAllocator malloc/free`, `DevMalloc`, `workspaceSize_:N` → identify peak allocated |

**Goal:** Pinpoint which NPU-side aclnn ops account for extra memory or consume memory **beyond what the API semantically needs**. For each suspected interface, record: aclnn interface name + estimated memory impact + suspected root cause.

**Common patterns:**

| Pattern | Plog signature | Root cause |
|---------|---------------|------------|
| Internal Cast | Cast node in workspace (e.g. FP32→FP16) | aclnn op does dtype conversion internally |
| Large workspace | `workspaceSize_` >> input size | aclnn algorithm needs large scratch buffer |
| Redundant Contiguous | Multiple `Contiguous` calls per op | Non-contiguous tensor triggers extra copy |

**After analysis, classify the plog outcome as one of:**

| Outcome | Definition | Characteristic |
|---------|-----------|----------------|
| **Target attribution** | Overconsumption traced to aclnn ops of `TARGET_API` itself | Proceed with normal analysis |
| **Non-target attribution** | Overconsumption source is identifiable but NOT from `TARGET_API` (e.g. input data preprocessing, implicit Cast) — can be cross-verified against test script's Key Code | Root cause is clear, but the issue is NOT in the target API. MUST be flagged to user in conclusions |
| **Inconclusive** | Plog does not show expected memory footprint for `TARGET_API`, or memory allocation source is unclear/unattributable — cannot be cross-verified against test script | Do NOT ask for more logs. Mark as "Inconclusive from plog" and continue to 1b/1c — source code analysis (Step 2) can often resolve what plog alone cannot |

Both **Non-target attribution** and **Inconclusive** MUST be explicitly reported to the user in all downstream conclusions (1c, Step 3).

#### 1b. Known-Issue Lookup

For each suspected aclnn interface from 1a (skip this sub-step if 1a found no suspected interfaces), **use Grep** (do NOT read the full file) to search [memory_consistency_issue_cases.md](references/memory_consistency_issue_cases.md) by interface name.

**Name extraction:** From plog entries like `aclnn[OpName]_[Num]_[InnerKernel]`, extract base name `aclnn[OpName]` (e.g. `aclnnInplaceNormal_1_CastAiCore` → `aclnnInplaceNormal`). Deduplicate before searching.

**For each suspected interface, classify:**

| Search result | Classification |
|---------------|---------------|
| Known issue found, root cause corroborates plog analysis | **Closed-loop** — no further analysis needed |
| Known issue found, root cause does NOT corroborate | **Needs source analysis** |
| No matching case, but plog clearly indicates CANN-side issue (e.g. oversized workspace, internal Cast) | **Closed-loop (CANN)** — root cause attributable to CANN, no torch_npu source analysis needed |
| No matching case, root cause unclear from plog alone | **Needs source analysis** |

#### 1c. Brief Progress Presentation

Briefly inform the user of analysis progress and preliminary direction. Keep it concise — **detailed conclusions are deferred to Step 3**.

**Content (plain text, no strict template):**
1. Plog analysis outcome — state which category from 1a applies:
   - **Target attribution**: list suspected aclnn op names + abnormal memory observations
   - **Non-target attribution**: identify the actual source (e.g. preprocessing, implicit Cast) and flag that `TARGET_API` is not the cause
   - **Inconclusive**: you MUST explicitly tell the user "Plog analysis was inconclusive"
2. Known-issue match status (which interfaces matched / not matched)
3. Preliminary judgment: closed-loop vs needs deeper analysis

**Then decide next action:**

| Situation | Action |
|-----------|--------|
| All key interfaces are closed-loop or closed-loop (CANN) | Skip to Step 3 (Present Findings) |
| Some interfaces need source analysis OR plog was inconclusive | Ask: **"Would you like me to perform source code analysis for deeper investigation?"** |

- User says yes → proceed to Step 2
- User says no → skip to Step 3

### Step 2: Source Code Analysis

Perform source code analysis for every suspected interface that was NOT closed-loop in Step 1.

**Prerequisite: pytorch_npu source path**
- If the user has already provided the `pytorch_npu` source path (e.g. at the start of the conversation), use it directly.
- If not, ask the user: **"Source code analysis requires the pytorch_npu source tree. Please provide the local path to your pytorch_npu repository (e.g. `d:\open_source\pytorch_npu`)."**
- If the user cannot provide it → **skip source code analysis entirely**, present only the conclusions available so far (plog analysis + any known issue matches), and warn the user: "Without pytorch_npu source, the analysis is limited to plog-level evidence and known issue matching. Root cause confirmation and code-level fixes cannot be provided. It is recommended to re-run analysis with source access for definitive conclusions."

**If source is available, proceed:**

1. **Search pytorch_npu source** (`pytorch_npu/third_party/op-plugin/`) for:
   - The operator's kernel implementation (e.g. `NansumKernelNpuOpApi.cpp`)
   - The dispatch path (check `op_plugin_functions.yaml` for registration)
   - Whether a composite decomposition is used (no dedicated kernel → PyTorch decomposes into sub-ops)

2. **Check if an optimization exists in compiled mode but not eager mode:**
   - Search `torch_npu/_inductor/` for the same pattern
   - If compiled mode has optimization but eager doesn't → the fix is to port the optimization to eager

3. **Determine fix location:**
   - **torch_npu code issue** → describe code-level defect, propose code fix if possible; otherwise suggest fix approach
   - **CANN issue** (workspace too large, internal Cast) → describe workspace/cast behavior, recommend filing ISSUE or feature request to CANN team

### Step 3: Present Findings

> **MANDATORY**: This step MUST be executed before entering Stage 3. Do NOT skip it and do NOT wait for user permission.

Collect ALL conclusions (closed-loop from Step 1 + source analysis from Step 2 if performed) and report as the format below:

```
## Analysis: {TARGET_API} Memory Overconsumption

### Memory Overview
- NPU peak activated: X GiB | GPU baseline: Y GiB | Ratio: X/Y

### Root Causes
For each suspect:
- **Interface**: aclnnXxx | **Owner**: torch_npu / CANN | **Impact**: N GiB
- **Evidence**: plog analysis summary
- **Known issue**: ISSUE-xxx (corroborated) / None
- **Fix**: description or code change

### Code Changes (if applicable)
- File: path → Change: description

### Summary
[1-2 sentences: what is the primary cause of the NPU–GPU memory gap. Do NOT repeat details already in Root Causes.]
[If Non-target attribution: explicitly state that the issue lies in auxiliary ops, not the target API being tested.]
[If Inconclusive (even after source analysis): explicitly state that root cause could not be fully determined, and say sorry to user.]
```

## Stage 3: Validate & Accumulate

### Step 1: Ask User for Validation

Ask: **"Does the above analysis resolve your issue?"**

- **User confirms resolved** → proceed to Step 2
- **User says no or provides additional info** → return to Stage 2 Step 2/3 with new evidence, continue deep analysis, then re-ask
- Never end Stage 3 Step 1 until user confirms resolution

### Step 2: Accumulate Experience

#### 2a. Ask for ISSUE Number

Ask user: **"Please provide the ISSUE number for this issue (if available)."**

- If user provides ISSUE → use it
- If user has none → generate an internal ID: `INT-YYYYMMDD-NNN`

#### 2b. Write to Cases File

Append to [memory_consistency_issue_cases.md](references/memory_consistency_issue_cases.md):

```yaml
### <dts_number>
- dts_number: "<dts_number>"
- description: "<brief issue description>"
- aclnn_interface: "<related aclnn interface>"
- root_cause: "<root cause analysis>"
- solution: "<resolution strategy>"
- category: "<issue classification>"
```

Category values:
- `cast导致显存占用升高` — internal Cast in aclnn op
- `缺少aclnn算子` — missing dedicated aclnn kernel
- `worst-case预分配` — pre-allocating for worst case (e.g. nonzero)
- `workspace过大` — aclnn workspace disproportionate to input
- `torch_npu逻辑缺陷` — missing optimization in torch_npu eager path
- `其他` — other

## Quick References

- [execute_mem_test.py](tools/execute_mem_test.py) — All-in-one: script locating → GPU conversion → remote testing → result validation
- [Memory Issue Cases](references/memory_consistency_issue_cases.md) — Historical issues and solutions for known-issue matching (Stage 2 Step 1b)

## Diagnostic Commands

```bash
# Check NPU memory
npu-smi info

# Check CANN version
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg

# torch_npu version
python -c "import torch_npu; print(torch_npu.__version__)"
```
