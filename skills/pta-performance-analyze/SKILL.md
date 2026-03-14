---
name: pta-performance-analyze
description: Diagnoses torch_npu single-operator memory overconsumption vs torch_gpu through a 4-stage workflow: Stage 1 prepares NPU/GPU test scripts; Stage 2 runs remote benchmarks and collects plog; Stage 3 analyzes plog timeline, matches known issues, and inspects source code; Stage 4 validates with user and accumulates experience. Use when users report NPU memory usage exceeding GPU, memory consistency issues, operator-level memory profiling, workspace overhead, or aclnn memory allocation problems.
---

# PTA Memory Consistency Analyzer

Diagnoses why a torch_npu operator uses more device memory than the equivalent torch_gpu operator, and provides actionable fixes.

## When to Use This Skill

- When the user asks to "diagnose/analyze an operator's memory usage", "NPU memory is higher than GPU", "memory consistency not met", "torch_npu xxx interface memory overconsumption", "run a memory comparison", "plog shows abnormal memory allocation", etc. — follow this skill's stages from the beginning.
- When the user provides an NPU memory test script and asks for GPU comparison — start from Stage 1 Step 2.
- When the user already has NPU and GPU memory data (e.g. `mem_results.md`) and asks for root cause analysis — start from Stage 3.
- When the user provides a plog file and asks for memory bottleneck analysis — start from Stage 3 Step 2.
- If the user provides a `pytorch_npu` / `torch_npu` source code path at any point, record it for Stage 3 Step 3 source code analysis.
- Output must include: **data evidence (plog values, memory comparison tables), root cause attribution (torch_npu vs CANN), and actionable fix suggestions or code changes**.

## Stage 1: Prepare Test Scripts

### 1. Identify Target

Locate the NPU memory test script:
- If the user explicitly provides a script path, use it directly
- If no script is specified, scan the current working directory for `torchapi_*.py` files:
  - If exactly one match → use it automatically
  - If multiple matches → list them and ask the user which one to use
  - If no match → ask the user for the script path
- The script must contain `TARGET_API = "torch.xxx"` and a `memory_decorator` that captures `total_driver_GB`, `pta_reserved_GB`, `pta_activated_GB`
- Extract the target API name from the script's `TARGET_API` variable

### 2. Generate GPU Script

Use [convert_npu_to_gpu.py](tools/convert_npu_to_gpu.py) to convert the NPU script to GPU:

```bash
python tools/convert_npu_to_gpu.py <npu_script.py>
```

> **HARD GATE**: Stage 1 Step 2 MUST NOT be skipped. You MUST have **both** the NPU script and the converted GPU script ready before entering Stage 2. If GPU script generation fails, fix it first instead of proceeding with remote execution.

## Stage 2: Remote Execution & Data Collection

### 1. Run Remote Benchmarks

**Do not ask the user in this stage.** First, try to reuse existing artifacts in the NPU script directory:
- If `mem_results.md` exists **and** there is at least one `filtered_plog_*.log` file → reuse them and proceed to Stage 3
- Otherwise → run remote benchmarks immediately (this stage)

Use [run_remote_mem_test.py](tools/run_remote_mem_test.py) to execute tests on Ascend and GPU servers in parallel:

```bash
python tools/run_remote_mem_test.py <npu_script> <gpu_script> [options]
```

Options:
- `--skip-gpu` — skip GPU test (use with `--gpu-json` for manual GPU data)
- `--gpu-json <file_or_json>` — provide GPU results manually (JSON file or inline)
- `--skip-ascend` — skip Ascend test
- `--api-name torch.xxx` — override API name

Server config is read from [servers.json](servers.json). The script:
1. Connects to servers via SSH (SSH_ASKPASS method)
2. Sources `env.sh` if present (conda + CANN environment)
3. Creates `<remote_dir>/<api_name>/` directory
4. Uploads scripts, sets `ASCEND_GLOBAL_LOG_LEVEL=0` and `ASCEND_PROCESS_LOG_PATH`
5. Runs test, captures PID and JSON output
6. Finds plog files matching PID, runs [filter_plog_memory.py](tools/filter_plog_memory.py) on them
7. Downloads filtered plog to local

Outputs (in NPU script's directory):
- `mem_results.md` — comparison report with NPU vs GPU table
- `filtered_plog_<api_name>.log` — filtered plog for analysis

### 2. Check Results

Read `mem_results.md`. Key metrics:
- **total_driver_GB**: actual driver-level memory delta
- **reserved_GB**: PTA CachingAllocator reserved
- **activated_GB**: PTA CachingAllocator peak allocated (most important)
- **Ratio**: NPU/GPU — anything above 1.2x warrants investigation

> **HARD GATE**: Stage 2 MUST be completed before entering Stage 3.
> The agent MUST have actual memory results (`mem_results.md`) and filtered plog
> (`filtered_plog_*.log`) from remote NPU script execution. Do NOT skip Stage 2
> and jump to known issue lookup based on the API name alone.

## Stage 3: Analyze Root Cause

### Step 1: Analyze Plog Timeline (Preliminary)

Read the filtered plog and `mem_results.md`, reconstruct the memory timeline:

**Track these events chronologically:**
- `PTA CachingAllocator malloc:` — memory allocated (with malloc size, cached, allocated)
- `PTA CachingAllocator free:` — memory freed
- `pta_memory acl_malloc:` — new block from driver
- `DevMalloc:` — driver-level allocation
- `Alloc workspace N bytes` — operator workspace allocation
- `workspaceSize_:N` — workspace size computed by aclnn

**Identify the peak:**
1. Find the maximum `allocated` value across all malloc/free events
2. Identify which operation(s) are live at that peak
3. Calculate how much each live allocation contributes

**Common memory overconsumption patterns:**

| Pattern | Signature in plog | Root cause |
|---------|------------------|------------|
| Internal Cast | workspace contains Cast node (e.g. FP32→FP16) | aclnn op does dtype conversion internally |
| Worst-case nonzero | `aclnnNonzeroV2` output shape `[ndim, numel()]` | Bool indexing pre-allocates max possible indices |
| Large workspace | `workspaceSize_` >> input size | aclnn algorithm needs large scratch buffer |
| Redundant Contiguous | Multiple `Contiguous` calls per op | Non-contiguous tensor triggers extra copy |

**Output a preliminary analysis** listing each suspected aclnn interface and its estimated memory impact. Present this to the user and ask:

**"Based on the plog analysis, the following aclnn interfaces are suspected of abnormal memory allocation: ... Would you like me to proceed with deeper analysis? If not, we can go directly to experience accumulation (Stage 4 Step 2)."**

- **User says no** → skip to Stage 4 Step 2
- **User says yes** → proceed to Step 2

### Step 2: Match Known Issues

For each suspected aclnn interface identified in Step 1, search [memory_consistency_issue_cases.md](references/memory_consistency_issue_cases.md) for known issues.

**aclnn interface name extraction rules:**
- Plog entries follow patterns like `aclnn[OpName]_[Num]_[InnerKernel]` or `aclnn[OpName]`
- Extract the base name `aclnn[OpName]` for lookup — e.g. from `aclnnInplaceNormal_1_CastAiCore` extract `aclnnInplaceNormal`
- Deduplicate: search each base name only once even if it appears in multiple plog entries

**Matching rules (strict):**
- The `aclnn_interface` field in the cases file must match the extracted base name exactly

**For each suspected interface:**
1. Search the cases file for an exact match on the base `aclnn[OpName]`
2. **If a known issue is found:**
   - Check whether its documented root cause corroborates your Step 1 plog analysis conclusion
   - If corroborated → mark this interface's conclusion as "closed-loop with known issue" (cache it)
   - If NOT corroborated → this interface's source code analysis in Step 3 **cannot be skipped**
3. **If no known issue is found** → this interface's source code analysis in Step 3 **cannot be skipped**

### Step 3: Source Code Analysis

Perform source code analysis for every suspected interface that was NOT closed-loop in Step 2.

**Prerequisite: pytorch_npu source path**
- If the user has already provided the `pytorch_npu` source path (e.g. at the start of the conversation or in Stage 1), use it directly.
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
   - **torch_npu code issue** → propose code fix, modify the source file
   - **CANN issue** (workspace too large, internal Cast) → report root cause and recommend DTS

### Step 4: Present Findings

Collect ALL conclusions — both "closed-loop with known issue" results from Step 2 and source code analysis results from Step 3 — and report to user in structured format:

```
## Analysis: <API name> Memory Overconsumption

### Memory Timeline
- Peak: X GiB at <operation>
- GPU baseline: Y GiB
- Ratio: X/Y

### Root Causes Found
1. [aclnn interface] — [torch_npu / CANN] — [N GiB impact]
   - Evidence: [plog analysis summary]
   - Known issue: [DTS number] (corroborated) / None
   - Source analysis: [findings if applicable]
   - Fix: [description]

2. [aclnn interface] — ...

### Code Changes (if applicable)
- File: <path>
- Change: <description>
```

## Stage 4: Validate & Accumulate

### Step 1: Ask User for Validation

Ask: **"以上分析是否解决了你的问题？"**

- **User says "是" or confirms resolved** → proceed to Step 2
- **User says "否" or provides additional info** → return to Stage 3 Step 2/3 with new evidence, continue deep analysis, then re-ask
- Never end Stage 4 Step 1 until user confirms resolution

### Step 2: Accumulate Experience

#### 2a. Ask for DTS Number

Ask user: **"请提供该问题的 DTS 编号（如果有的话）"**

- If user provides DTS → use it
- If user says "没有" or "不知道" → generate an internal ID: `INT-YYYYMMDD-NNN`

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

- [Memory Issue Cases](references/memory_consistency_issue_cases.md) — Historical issues and solutions
- [convert_npu_to_gpu.py](tools/convert_npu_to_gpu.py) — NPU→GPU script converter
- [run_remote_mem_test.py](tools/run_remote_mem_test.py) — Remote benchmark runner
- [filter_plog_memory.py](tools/filter_plog_memory.py) — Plog memory filter

## Key Environment Variables

```bash
export ASCEND_GLOBAL_LOG_LEVEL=0        # Enable debug plog
export ASCEND_PROCESS_LOG_PATH=<dir>    # Redirect plog output
```

## Diagnostic Commands

```bash
# Check NPU memory
npu-smi info

# Check CANN version
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg

# torch_npu version
python -c "import torch_npu; print(torch_npu.__version__)"
```
