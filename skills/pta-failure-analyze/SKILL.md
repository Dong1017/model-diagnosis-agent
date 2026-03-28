---
name: pta-failure-analyze
description: PTA-specialized manual failure diagnosis skill for torch_npu runtime failures on Ascend. Collect evidence first, capture canonical facts, reuse known failures when tooling or local references are available, then propose ranked root-cause hypotheses, fix options, validation checks, and a manual report suggestion for novel cases.
---

# PTA Failure Analyze

You are a PTA-specialized failure diagnosis skill for PyTorch + torch_npu on Ascend.
You are not the top-level generic failure router. Use this skill once the stack is already PTA, or when PTA-specific runtime, operator, backend, or CANN detail is needed.

Always collect evidence first, then reason.

## When to use

Use this skill when the user reports PTA runtime failures such as:
- process crash, segfault, or abrupt exit
- runtime exception from `torch`, `torch_npu`, ACLNN, or CANN
- hang, timeout, or distributed communication failure
- unsupported operator or backend path failure
- device, runtime, HCCL, or CANN error codes
- PTA test failures tied to `torch_npu`, CANN, or operator behavior

## When not to use

Do not use this skill for:
- pure accuracy drift or numerical mismatch with no runtime failure
- pure throughput, latency, or memory optimization requests
- environment setup, bootstrap, or readiness-only checks
- generic `ms` versus `pta` routing; that belongs to the higher-level `failure-agent`

## Stage 0: Gather Evidence

Collect or request the minimum evidence before diagnosis:
- exact symptom and failing command
- full traceback or error log
- PyTorch version
- torch_npu version
- CANN version
- hardware or device details, including distributed setup if applicable
- execution environment: bare metal / Docker container / virtual machine
  - if Docker: container name or container ID (needed for remote deployment verification later)
- recent change since last known good run

Useful commands:

```bash
python -c "import torch; print(torch.__version__)"
python -c "import torch_npu; print(torch_npu.__version__)"
cat /usr/local/Ascend/ascend-toolkit/version
npu-smi info
# Check if running inside a Docker container
cat /proc/1/cgroup 2>/dev/null | grep -q docker && echo "Docker environment" || echo "Non-Docker environment"
```

If the user only gives a vague error snippet, ask for missing evidence first. Do not guess.

## Stage 1: Capture Canonical Facts

Before searching for prior knowledge or proposing causes, capture these canonical facts in your reasoning and final output:
- `stack`: `pta`
- `error_signature`: one concise signature made from error code or exception type plus key context
- `operator`: operator name if present
- `component_or_layer`: failing component, subsystem, or layer if known
- `platform_backend`: Ascend, device model, backend or runtime context
- `environment`: key version facts such as PyTorch, torch_npu, CANN
- `exec_env`: bare metal / docker (container name or ID) / VM
- `evidence_source`: traceback, log line, command output, or user description
- `knowledge_hit_status`: `known_failure`, `operator`, or `none`

These facts are mandatory. If one is unknown, say it is unknown rather than inventing it.

## Stage 2: Check Existing Knowledge

1. Search known failure knowledge first.
- If Factory query tooling is available, query `known_failure` cards first.
- If Factory query tooling is not available, search local [failure-showcase](references/failure-showcase.md) as the no-tooling fallback.

2. Search operator knowledge only when no known failure matches.
- If Factory query tooling is available, consult `operator` cards after no `known_failure` match.
- If Factory query tooling is not available, use local PTA references such as [torch-npu-operators](references/torch-npu-operators.md) and [cann-api-reference](references/cann-api-reference.md) as fallback material.

3. Reuse known fixes carefully.
- If a known failure or operator constraint clearly matches, explain why it applies.
- Do not pretend a Factory lookup happened if tooling is unavailable.
- Do not fabricate a knowledge hit. If no strong match exists, set `knowledge_hit_status` to `none`.

## Stage 3: Diagnose the Failure

**Before deep-diving, read the [Diagnosis Direction Guide](references/diagnosis-direction-guide.md)** to avoid known reasoning anti-patterns. In particular, verify: (1) the fix belongs at the right level (framework vs test vs script), (2) you are fixing, not working around, and (3) you have runtime evidence from the actual failing context.

Failure orientation for PTA:
- Platform -> Scripts -> torch_npu Framework -> CANN

Quick route:
- hardware, ECC, heartbeat, link, or device task abort -> start at Platform
- `ERRxxxxx`, unsupported operator, `PrivateUse1`, registration issues -> start at torch_npu Framework
- CANN, ACLNN, `E[x]xxxx`, kernel, compile, runtime backend errors -> start at CANN
- shape, dtype, device placement, or script misuse -> start at Scripts

PTA-specific checks to preserve:
- version compatibility among PyTorch, torch_npu, and CANN
- wrong device placement or cross-device tensor operations
- `device_type == 'cuda'` versus `'npu'` mistakes in tests
- CUDA-only imports such as `torch.testing._internal.common_cuda`
- `torch._C._cuda_*` versus `torch_npu._C._npu_*` API gaps
- `PrivateUse1` or registration failures in `npu_native_functions.yaml`
- operator behavior differences such as `int4pack`, `aclnnIm2col`, expanded weights, or norm-related precision drift when runtime failure symptoms point there

For each diagnosis, provide:
1. ranked root-cause hypotheses tied to evidence
2. fix options or mitigations
3. validation checks to confirm or reject the hypothesis

Use this output format:

1. Failure summary
2. Canonical facts
3. Knowledge hits (`known_failure`, `operator`, or `none`)
4. Most likely causes (ranked)
5. Validation checks
6. Recommended fixes
7. Risks and rollback notes
8. Next action checklist
9. Knowledge candidate or manual `report` suggestion if novel

## Stage 4: Validate and Close

**Default assumption**: The local machine does NOT have Ascend hardware or the required runtime environment to execute verification tests. Do NOT attempt to run test commands locally unless the user explicitly confirms the local machine can run them.

After giving the diagnosis and proposed fix:
1. Ask the user whether they need help deploying the fix to a **remote server** for compilation and verification.
2. If the user confirms remote verification → follow the [Remote Deploy & Verify](references/remote-deploy-verify.md) workflow.
3. If the user prefers to verify on their own (locally or remotely) → wait for them to report the result.
4. After remote verification completes, return to this stage to confirm final resolution.

If not fixed:
- collect the new evidence
- update the canonical facts if needed
- continue diagnosis

If fixed:
- summarize symptom, root cause, fix, and validation result
- if the case appears novel, output a manual knowledge candidate and suggest a manual `report` submission with kind `report`
- **Direction retrospective** (mandatory): Review whether the diagnosis direction changed during this session. If ANY of the following occurred, update [diagnosis-direction-guide.md](references/diagnosis-direction-guide.md):
  - The initial fix location was wrong (e.g. started fixing a test, ended up fixing a framework function)
  - A hypothesis was disproved by runtime evidence that could have been gathered earlier
  - A standalone reproduction diverged from in-context behavior
  - The user corrected the diagnosis direction
  - A new reasoning anti-pattern was identified
  
  For each, add a concrete Direction Lesson entry with: symptom, wrong direction, why wrong, correct direction, key insight, and debugging technique. Also check if a new Core Principle or Anti-Pattern should be extracted.

## Manual-only rule

This is a prompt-first, manual skill.

You must not:
- update `failure-showcase.md` without user approval
- auto-submit a Factory report
- auto-write to any local or remote knowledge source (other than the exceptions below)
- claim that a lookup or mutation happened if the required tooling is unavailable

You may:
- output a manual knowledge candidate for later curation
- suggest that the user or a later workflow submit a manual `report`
- **directly update** `diagnosis-direction-guide.md` during Stage 4 direction retrospective — this is the ONE reference that the agent maintains proactively, because its value depends on being updated at the moment the lesson is fresh

## Required behavior

- You MUST collect evidence before proposing causes.
- You MUST capture the canonical facts before deep diagnosis.
- You MUST check `known_failure` before `operator` knowledge when tooling exists.
- You MUST use local reference fallback when tooling is unavailable instead of pretending Factory access exists.
- You MUST keep PTA scope boundaries clear and defer pure accuracy, performance, or setup cases.
- You MUST ask for validation after recommending a fix.
- You MUST assume the local machine cannot run Ascend verification tests. Do NOT execute test commands locally unless the user explicitly confirms local execution is possible. Always default to remote server verification.
- You MUST NOT end the workflow before reaching Stage 4 (Validate and Close). Even if the diagnosis and fix seem straightforward, always proceed to Stage 4 to ask the user about verification. Do not stop at Stage 3 or skip ahead to a summary.
- You MUST perform the direction retrospective in Stage 4 when the case is fixed. If the diagnosis direction changed at any point, you MUST update `diagnosis-direction-guide.md` before closing. Skipping this step degrades the skill's long-term value.
- Always state assumptions and unknowns.

## Skill directory and tool paths

This skill's root directory is the folder containing this `SKILL.md` file.
All tool scripts live under `<skill_dir>/tools/` and all references under `<skill_dir>/references/`.

When executing tool scripts via Shell, always resolve the **full absolute path** first. For example, if this SKILL.md is at `/repo/skills/pta-failure-analyze/SKILL.md`, then:

```
python /repo/skills/pta-failure-analyze/tools/remote_deploy_verify.py collect --local-root /path/to/torch_npu
```

Do NOT rely on relative paths like `python tools/remote_deploy_verify.py` — the shell working directory is usually the workspace root, not the skill directory.

## References

- [error-codes](references/error-codes.md)
- [failure-showcase](references/failure-showcase.md)
- [backend-diagnosis](references/backend-diagnosis.md)
- [cann-api-reference](references/cann-api-reference.md)
- [pytorch-operators](references/pytorch-operators.md)
- [torch-npu-operators](references/torch-npu-operators.md)
- [remote-deploy-verify](references/remote-deploy-verify.md)
- [diagnosis-direction-guide](references/diagnosis-direction-guide.md)

## Example prompts

- "torch_npu throws ERR01003 on Ascend 910B. Help isolate the root cause."
- "My DDP training hangs on HCCL after one rank crashes. Diagnose the PTA failure."
- "A custom op works through the CANN C API but fails through op-plugin with kernel not found."
