---
name: simulate
description: Draft a runnable PEPS-tVMC simulation script from a user's physics proposal. Builds context, clarifies requirements, checks feasibility, produces a validated script with explanation and run instructions. Use when the user wants to simulate a quantum system, asks "can this code do X", or wants help setting up a simulation.
---

# Simulate

Draft a complete, validated simulation script from a user's physics proposal.

## Step 1: Build context

1. Call MCP `tool_list_models`, `tool_list_operators`, `tool_list_examples` to understand what the codebase can do.
2. Call MCP `tool_query_experience` with topics relevant to the user's request.
3. List PDFs in `/notes/` and suggest which papers are relevant to the user's physics. Ask which to read. Invoke `/read-notes` for the selected papers.
4. Read `EXPERIENCE.md` directly for any additional practitioner guidance.

## Step 2: Clarify requirements

Ask the user one question at a time. Prefer multiple choice. Cover:

1. **Hamiltonian:** What terms? User describes in physics language. Map each term to available operators. If a term doesn't map cleanly, ask for clarification.
2. **Lattice:** Shape and boundary conditions (this codebase only supports OBC).
3. **Task:** Ground state (imaginary-time SR), real-time dynamics (TDVP), or quench?
4. **Observables:** What to measure? Energy is automatic. What else? (magnetization, correlators, plaquette expectation values, etc.)
5. **Parameters:** Suggest defaults from EXPERIENCE.md and paper benchmarks. Confirm: bond dimension, n_samples, n_chains, dt, diag_shift, solver.
6. **Two-stage workflow?** If dynamics, does the user need a ground-state preparation step first?

## Step 3: Check feasibility

Call MCP `tool_check_feasibility` with the structured config built from Step 2.

- **Feasible:** Proceed to Step 4.
- **Partially feasible:** Explain what works and what's missing. Describe what code would need to be added. Ask if the user wants to proceed with what's available or try a closest alternative.
- **Not feasible:** Explain the gap clearly. Call MCP `tool_find_closest_example` to suggest the closest thing the codebase CAN do. Ask if the user wants to pivot.

## Step 4: Draft script

1. Call MCP `tool_find_closest_example` to find the best template.
2. Read that example script to understand its structure.
3. Adapt it to the user's requirements:
   - New Hamiltonian terms if needed
   - Different lattice size, observables, parameters
   - Place the script in the appropriate `examples/` subdirectory
4. For two-stage workflows: draft both ground_state.py and dynamics.py.
5. The script MUST:
   - `from vmc import config` before any JAX imports
   - Use `runner.run()` for the main loop
   - Use `DEFAULT_METRICS_CONFIG` from runner
   - Use `resolve_solver` from runner for solver choice
   - Follow the `sys.path.insert` pattern from existing examples
   - Include a docstring explaining what physics it simulates

## Step 5: Write explanation and run instructions

Alongside the script, produce:

1. **Explanation:** For each physics-to-code mapping, explain the choice:
   - Why this model family (PEPS vs GIPEPS vs BlockadePEPS)
   - How each Hamiltonian term maps to an operator class
   - Why these parameter values (cite paper benchmarks or EXPERIENCE.md)
   - Any coordinate conventions or sign conventions to be aware of
2. **Run instructions:** Exact commands to execute, resume, extend:
   ```
   # Run
   uv run python examples/path/to/script.py --n-steps 200

   # Resume if interrupted
   uv run python examples/path/to/script.py --resume

   # Output location
   data/run_dir/latest.json    # series data + config
   data/run_dir/latest/        # orbax checkpoint

   # Visualize results
   # (use the /visualize skill)
   ```
3. For two-stage workflows, show both commands in sequence with the `--state` handoff.

## Step 6: Review

Dispatch **two review agents** (spawn as subagents):

### Physics Review Agent

```
You are reviewing a simulation script for the PEPS-tVMC codebase.

Script: [PATH]
User's request: [DESCRIPTION]
Relevant paper(s): [PAPER_PATHS]
EXPERIENCE.md: EXPERIENCE.md

Check:
1. Hamiltonian terms match the user's description? Coefficients/signs correct?
2. Model family correct for this physics?
3. Observables measure what the user asked? Coordinate trap: user convention
   is likely bottom-up (paper), but PlaquetteOperator uses top-down (internal).
   Verify open_to_internal_plaquette() is applied correctly if applicable.
4. Parameters reasonable for this lattice size? (Check EXPERIENCE.md + papers)
5. Two-stage handoff correct if applicable?

Verify by reading code, not trusting explanation.
Report: Approved | Issues Found [with line references]
```

### Code Review Agent

```
You are reviewing a simulation script for code correctness.

Script: [PATH]
CLAUDE.md: CLAUDE.md
EXPERIENCE.md: EXPERIENCE.md

Check:
1. Runner API used correctly? All required args provided?
2. Imports valid? Model family compatible with operator type?
3. TDVPDriver args correct (time_unit, integrator, preconditioner)?
4. `from vmc import config` comes before any JAX imports?
5. Follows CLAUDE.md conventions?

Report: Approved | Issues Found [with line references]
```

Fix issues from either reviewer. Re-dispatch until both approve (max 3 iterations).

## Step 7: Smoke test

Call MCP `tool_smoke_test` with the script path and tiny parameter overrides.

For two-stage workflows: smoke test ground state first, then pass its run_dir to dynamics via `chain_state`.

If smoke test fails:
1. Read the traceback
2. Diagnose the issue
3. Fix the script
4. Re-run smoke test (max 3 retries)

## Step 8: Deliver

Present to the user:
1. The validated script (with file path)
2. The explanation of physics choices
3. Exact run instructions
4. Suggestion: "Use /visualize to plot results after the run completes"
