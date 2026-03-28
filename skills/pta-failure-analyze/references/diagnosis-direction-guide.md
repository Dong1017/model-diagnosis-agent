# Diagnosis Direction Guide

Accumulated lessons on **how to approach PTA failure diagnosis** — specifically, common missteps in analysis direction, and principles for staying on the correct path. This is NOT a list of failures; it is a list of **reasoning anti-patterns** and **direction-correction rules** learned from real cases.

Read this reference at the START of Stage 3 (Diagnose the Failure) to avoid known directional mistakes.

## Table of Contents

- [Core Principles](#core-principles)
- [Direction Anti-Patterns](#direction-anti-patterns)
- [Direction Lessons (Case-Based)](#direction-lessons-case-based)
- [Decision Checkpoints](#decision-checkpoints)

---

## Core Principles

### P1: Fix at the root, not at the symptom site

When a test fails, the failure site (the line that throws) is almost never the right place to fix. Trace the execution path **backwards** to find the first point where behavior diverges from expectation. Fixing at the symptom site creates workarounds that mask the real bug and break other callers.

**Self-check**: "Am I fixing the code that **produces** the wrong behavior, or the code that **observes** the wrong behavior?"

### P2: Understand the full call chain before proposing a fix

Before writing any fix, trace the complete call chain from the entry point to the failure site. Identify every function involved, and what each one assumes about its inputs. A fix that doesn't account for the full chain will either miss the real cause or break something upstream.

**Self-check**: "Can I draw the full call chain from entry to failure on paper?"

### P3: When a framework function misbehaves, fix the framework function

If a PyTorch/torch_npu framework utility (e.g. `filter_desired_device_types`, `instantiate_device_type_tests`) behaves incorrectly for NPU, the fix belongs in that utility — not in every caller. Patching individual callers is a whack-a-mole approach that will fail for the next caller.

**Self-check**: "Am I patching the caller or fixing the callee? If I'm patching the caller, why can't I fix the callee?"

### P4: Verify assumptions with runtime evidence before deep-diving

Before spending time on a hypothesis, add a single targeted debug probe (print, assert, or standalone script) to verify or refute it. A 30-second probe is worth more than 30 minutes of code reading.

**Self-check**: "What is the ONE print statement that would confirm or kill my current hypothesis?"

### P5: When the same function is used by multiple tests, the bug is in the function

If multiple tests exhibit the same failure pattern through the same utility function, the root cause is in the shared utility, not in each individual test. Collect the affected tests first, then fix the common path once.

**Self-check**: "Are there other callers of this function that would have the same problem?"

---

## Direction Anti-Patterns

### AP1: Bypassing instead of fixing

**What it looks like**: Replacing the failing call with a different API that avoids the broken path (e.g. replacing `instantiate_device_type_tests` with direct `instantiate_test` calls).

**Why it's wrong**: The broken function remains broken for all other callers. The "fix" is actually a workaround that adds NPU-specific divergence from upstream.

**Correct direction**: Analyze WHY the function fails for NPU and fix it there. The test code should stay as close to upstream as possible.

### AP2: Fixing the test when the framework is wrong

**What it looks like**: Adding NPU-specific logic, wrappers, or imports to individual test files when the real problem is in PyTorch's test infrastructure (`common_device_type.py`, `common_utils.py`, etc.).

**Why it's wrong**: Creates maintenance burden — every new test that hits the same framework bug needs the same workaround.

**Correct direction**: Fix the framework utility, keep the test simple.

### AP3: Assuming the class attribute is immutable

**What it looks like**: Reasoning about `PrivateUse1TestBase.device_type` based on its class definition (`= "privateuse1"`) without checking whether `setUpClass` or other runtime code has mutated it.

**Why it's wrong**: Python class attributes are mutable. `setUpClass` explicitly modifies `cls.device_type` at runtime, and this can affect the base class (not just derived classes) depending on how `setUpClass` is called.

**Correct direction**: Always verify class attribute values at the actual point of use with a runtime probe, not by reading the class definition.

### AP4: Testing in isolation then assuming in-context behavior is the same

**What it looks like**: Writing a standalone debug script that works, then assuming the same code path will work inside the actual test runner.

**Why it's wrong**: The test runner has prior state (e.g. `setUpClass` has already run, class attributes are mutated, globals are modified). A standalone script starts from a clean state.

**Correct direction**: When a standalone script and the in-context test disagree, the difference IS the clue. List what state changes exist between "standalone" and "in-context" — the answer is in that delta.

### AP5: Looking at one side of a comparison when both sides matter

**What it looks like**: Focusing on normalizing the input (`only_for`) but not checking what value `x.device_type` actually holds at comparison time.

**Why it's wrong**: A comparison has two sides. Normalizing only one side is guaranteed to break when the other side doesn't match expectations.

**Correct direction**: When diagnosing a failed comparison/filter, print BOTH sides. Verify that the normalization is symmetric.

---

## Direction Lessons (Case-Based)

### DL1: filter_desired_device_types empty-list return (2026-03-25)

**Symptom**: `assertRaisesRegex(RuntimeError, "handled multiple times")` fails — RuntimeError never raised.

**Wrong direction taken**: Attempted to bypass `instantiate_device_type_tests` entirely by calling `instantiate_test` directly with `CPUTestBase`. This "worked" but was a workaround, not a fix.

**Why direction was wrong**: The real bug was inside `filter_desired_device_types`, which returned an empty list. Bypassing the caller doesn't fix the callee — every other test calling `instantiate_device_type_tests` with PrivateUse1 `only_for` would hit the same bug.

**Correct direction**: Analyze `filter_desired_device_types` directly. The function had two bugs: (1) bare-string input iterated as characters; (2) normalization was applied to input side but not to `x.device_type`, which had been mutated by `setUpClass`. Fixing the function itself with symmetric normalization resolved the issue for all callers.

**Key insight**: When a utility function produces an unexpected empty result, the fix belongs in that function. Don't route around it.

**Debugging technique that finally worked**: Add a runtime probe INSIDE the test context (not in a standalone script) to print `bases=[]`, which immediately revealed the filter was returning empty. Then compare standalone-vs-in-context to identify the state delta (`setUpClass` mutation).

---

## Decision Checkpoints

Use these checkpoints during Stage 3 diagnosis to catch directional mistakes early.

### Checkpoint 1: Where does the fix belong?

```
Is the failing behavior in:
  (a) Test code specific to this test?      → Fix in test
  (b) A shared utility/framework function?  → Fix in the utility
  (c) torch_npu adaptation layer?           → Fix in torch_npu
  (d) PyTorch upstream code?                → Fix in PyTorch source

Rule: If (b), (c), or (d), do NOT patch individual tests.
```

### Checkpoint 2: Am I working around or fixing?

```
Does my proposed change:
  (a) Make the broken function work correctly?  → Good: this is a fix
  (b) Avoid calling the broken function?        → Bad: this is a workaround
  (c) Add NPU-specific code to a test?          → Suspect: check if a framework fix is possible first

Rule: Prefer (a). Only accept (b) or (c) if (a) is proven infeasible.
```

### Checkpoint 3: Have I verified with runtime evidence?

```
Before proposing a fix, have I:
  [ ] Confirmed the hypothesis with a probe in the ACTUAL failing context (not standalone)?
  [ ] Printed both sides of any failing comparison?
  [ ] Checked for state mutations (setUpClass, module-level side effects, global variable changes)?

Rule: No fix without runtime evidence from the actual failure context.
```

### Checkpoint 4: Standalone vs in-context divergence

```
If a standalone reproduction works but the actual test fails:
  → The difference is runtime state. Check:
    [ ] Has setUpClass/setUp modified class attributes?
    [ ] Have module-level instantiate_* calls changed global state?
    [ ] Are there __pycache__/.pyc files masking changes?
    [ ] Is sys.path or import order different?

Rule: The delta between standalone and in-context IS the root cause.
```

---

## Adding New Lessons

When a diagnosis goes in the wrong direction, capture it here:

1. **Symptom**: What the failure looked like
2. **Wrong direction taken**: What was attempted and why it seemed right
3. **Why direction was wrong**: The reasoning flaw
4. **Correct direction**: What should have been done
5. **Key insight**: One-sentence takeaway
6. **Debugging technique**: What probe/method finally cracked it

Keep entries concrete and case-based. Abstract principles go in [Core Principles](#core-principles). Anti-patterns go in [Direction Anti-Patterns](#direction-anti-patterns).
