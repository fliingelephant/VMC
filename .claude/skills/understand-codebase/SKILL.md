---
name: understand-codebase
description: Answer questions about the PEPS-tVMC codebase architecture, capabilities, and APIs. For developers wanting to understand or contribute, and physicists wanting to know what simulations are possible. Use when the user asks "how does X work", "what can this code do", "how do I add Y", or explores the codebase.
---

# Understand Codebase

Answer questions about the PEPS-tVMC codebase — architecture, capabilities, APIs, and physics.

## Step 1: Identify the question type

- **Capability question** ("what gauge groups are supported?", "can this do fermions?") → Use MCP discovery + compatibility tools
- **Architecture question** ("how does contraction work?", "what's the cache-turnover pattern?") → Read relevant source code + CLAUDE.md
- **API question** ("how do I add a new operator?", "how does the workflow module work?") → Read source code + examples
- **Physics question** ("what's the difference between SR and minSR?", "why gauge removal?") → Read papers via `/read-notes` + EXPERIENCE.md

## Step 2: Gather context

Based on question type:

1. **For capabilities:** Call MCP `tool_list_models`, `tool_list_operators`, `tool_list_strategies`, `tool_list_solvers` as needed. Call `tool_check_compatibility` or `tool_check_feasibility` if the user is asking about a specific simulation.

2. **For architecture:** Read CLAUDE.md for the architecture overview (core framework, three PEPS families, operators, QGT, preconditioners, drivers). Read the specific source files mentioned. Key files:
   - `src/vmc/core/` — MC sampler (make_mc_sampler, vmap, lax.scan)
   - `src/vmc/peps/common/contraction.py` — MPO-to-boundary contractions
   - `src/vmc/peps/common/energy.py` — local energy + derivative computation
   - `src/vmc/drivers/tdvp.py` — TDVPDriver (the main driver)
   - `src/vmc/preconditioners/` — SR, QGT solving
   - `src/vmc/qgt/` — Jacobian, SlicedJacobian, small-o trick

3. **For APIs:** Read the relevant module and an example script that uses it. Call MCP `tool_list_examples` to find relevant examples.

4. **For physics:** Call MCP `tool_query_experience` for practitioner advice. Invoke `/read-notes` if papers are needed.

## Step 3: Answer

- Reference specific code locations (file:line)
- For capability questions: be precise about what IS and ISN'T supported
- For architecture questions: explain the design pattern and why it exists
- For API questions: show a concrete code example from an existing script
- For physics questions: cite the paper and explain in the context of this codebase

If the question reveals a gap in the documentation or EXPERIENCE.md, suggest adding an entry.
