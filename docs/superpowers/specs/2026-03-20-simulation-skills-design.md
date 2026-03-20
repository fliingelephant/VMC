# Simulation Skills & MCP Server Design

Skills and MCP tools for helping users understand the PEPS-tVMC codebase, check whether it can run their proposed simulations, draft runnable scripts, and visualize results.

## Motivation

The codebase has a steep learning curve: three model families, multiple operator types, contraction strategies, solver choices, and domain-specific conventions. New users — both physicists proposing simulations and developers exploring the architecture — need guided assistance to map their intent to working code.

## Architecture

Three layers with clean separation of concerns:

| Layer | Responsibility | Examples |
|-------|---------------|----------|
| **MCP server** (`vmc-mcp`) | Structured queries, mechanical actions | list_models, check_compatibility, smoke_test, plot_convergence |
| **Skills** | Conversational orchestration, judgment | simulate flow, understand-codebase Q&A, visualize interpretation |
| **Docs** | Persistent knowledge | EXPERIENCE.md (practitioner wisdom), CLAUDE.md (rules) |

### MCP Server: `vmc-mcp`

Single Python server using the MCP SDK, importing from `vmc` directly for live introspection. Tools organized by namespace.

```
tools/vmc-mcp/
  server.py          # MCP server entry point
  discovery.py       # model/operator/strategy introspection
  experience.py      # query EXPERIENCE.md
  compatibility.py   # compatibility matrix + feasibility checks
  visualization.py   # plotting and animation
  runner_tools.py    # smoke test, checkpoint metadata
```

#### Discovery Tools

| Tool | Input | Output |
|------|-------|--------|
| `list_models()` | — | List of model classes with their config parameters, physical dimensions, supported gauge groups |
| `list_operators()` | — | List of operator types with their signatures, what sites they act on, which model families support them |
| `list_strategies()` | — | Contraction strategies with parameters and trade-offs |
| `list_solvers()` | — | Solver functions with characteristics (GPU-friendliness, robustness) |
| `list_examples()` | — | Available example scripts with physics description, model family, lattice sizes, paper references |
| `find_closest_example(description)` | Physics description dict | Best-matching example with path and what would need to change |

Discovery tools introspect the actual codebase at runtime — scanning for operator subclasses, reading model configs, checking what's importable. This avoids stale documentation. Note: `list_operators()` must scan both `vmc.operators.local_terms` (standard operators) and `vmc.peps.gi.local_terms` (GI-specific terms like `MatterMassTerm`, `HiggsLinkTerm`, etc.), as they live in different modules.

#### Compatibility Tools

A curated compatibility matrix encoding domain knowledge that pure introspection cannot discover:

| Tool | Input | Output |
|------|-------|--------|
| `check_compatibility(model, operator_types)` | Model class name, list of operator type names | Compatible (yes/no), reason, required model family if different |
| `check_feasibility(simulation_config)` | Full structured config (model, terms, lattice, observables) | Feasible / partially feasible / not feasible, with explanation |

The compatibility matrix encodes which **term types** are valid with which **model family** (note: `GILocalHamiltonian` is an alias for `LocalHamiltonian` — the container class is the same; what matters is the terms inside):
- GIPEPS terms: `PlaquetteOperator`, `LinkDiagonalTerm`, `MatterMassTerm`, `HorizontalMatterHoppingTerm`, `VerticalMatterHoppingTerm`, `HorizontalHiggsLinkTerm`, `VerticalHiggsLinkTerm` (from `vmc.peps.gi.local_terms`)
- Standard PEPS terms: `OneSiteOperator`, `DiagonalOperator`, `HorizontalTwoSiteOperator`, `VerticalTwoSiteOperator` (from `vmc.operators`)
- `PlaquetteOperator` works with both GIPEPS and standard PEPS, requires lattice >= 2x2
- `BlockadePEPS` requires blockade-constrained sampling, uses standard PEPS operator types
- Z_N gauge theories require GIPEPS with matching `N`
- Higgs link terms require `conserve_particle_number=False`; matter hopping terms do not
- minSR (`SampleSpace`) is preferred for GIPEPS with large per-site parameters; SR (`ParameterSpace`) is fine for standard PEPS

#### Experience Tools

| Tool | Input | Output |
|------|-------|--------|
| `query_experience(topic)` | Topic string (e.g., "GPU contraction", "Z2 convergence") | Relevant entries from EXPERIENCE.md |

#### Visualization Tools

| Tool | Input | Output |
|------|-------|--------|
| `plot_convergence(run_dir, keys=None)` | Runner output directory, optional series keys | Saves figure, returns path. Default: energy + all observables vs time |
| `plot_heatmap(run_dir, step, observable_prefix)` | Runner output, step index, prefix like "P_" | 2D spatial heatmap reconstructed from flat observable series |
| `animate(run_dir, observable_prefix, fps=5)` | Runner output, prefix, frame rate | GIF animation of spatial observable over time |

#### Runner Tools

| Tool | Input | Output |
|------|-------|--------|
| `smoke_test(script_path, overrides)` | Script path, dict of CLI arg overrides (tiny params) | pass/fail + stdout/stderr. Runs with `cwd` set to the script's parent directory so `sys.path` / `from runner import` resolves correctly. Generated scripts must be placed inside the `examples/` tree. Cleans up all generated output. For two-stage workflows, chains ground state (2 steps) → passes its `run_dir` as `--state` to dynamics (2 steps). |
| `read_checkpoint_metadata(run_dir)` | Runner output directory | Parsed latest.json metadata (config, series summary, step/time) |

### Skills

#### `simulate` (user-facing, top-level)

Full orchestration flow for drafting a simulation script.

**Trigger:** User describes a simulation they want to run ("I want to simulate...", "Can this code do...", "How would I set up...")

**Flow:**

1. **Build context**
   - Call MCP `list_models()`, `list_operators()`, `list_examples()`
   - Call MCP `query_experience()` for relevant topics
   - Read paper titles from `/notes/`, suggest relevant ones to user
   - Invoke `read-notes` skill for papers the user selects

2. **Clarify requirements** (one question at a time)
   - Hamiltonian: user describes in physics terms, agent maps to operators
   - Lattice geometry and size
   - Ground state vs real-time dynamics vs quench
   - Observables to track
   - Solver/preconditioner preferences (suggest defaults from experience)
   - Bond dimension, samples, chains (suggest from paper benchmarks)
   - For two-stage workflows (ground state → dynamics): clarify both stages

3. **Check feasibility**
   - Call MCP `check_feasibility(structured_config)`
   - If not feasible: explain the gap, suggest closest alternative via MCP `find_closest_example()`, ask if user wants to proceed with alternative
   - If partially feasible: explain what works and what's missing, describe what code would need to be added

4. **Draft script**
   - Call MCP `find_closest_example()` to get the best template
   - Agent reads that example script as reference
   - Agent adapts it to user's requirements (new Hamiltonian terms, different lattice, different observables)
   - For two-stage workflows: draft both scripts
   - Write: script file + explanation of physics-to-code mapping + run instructions (how to run, resume, extend, and where output goes)

5. **Review** (two separate review agents)
   - **Dispatch Physics Review Agent:**
     - Reads the drafted script and the relevant paper(s)
     - Checks: do Hamiltonian terms match the user's description? Are coefficients/signs correct? Are observables measuring what the user asked for? Are parameters reasonable for this lattice size?
     - Returns: Approved | Issues Found
   - **Dispatch Code Review Agent:**
     - Reads the drafted script, CLAUDE.md, EXPERIENCE.md
     - Checks: does it use the runner API correctly? Are imports valid? Is the model family correct? Are operator types compatible?
     - Returns: Approved | Issues Found
   - Fix issues, re-dispatch, max 3 iterations per reviewer

6. **Validate**
   - Call MCP `smoke_test(script_path, tiny_overrides)`
   - For two-stage workflows: run ground state (2 steps) → dynamics (2 steps)
   - If smoke test fails: agent reads traceback, diagnoses, fixes script, retries (max 3)
   - MCP cleans up all smoke test output

7. **Deliver**
   - Present: validated script, explanation of all physics choices and why, exact run commands, description of output format, suggestion to use `visualize` skill for results

#### `understand-codebase` (user-facing, also used by simulate)

For developers and physicists who want to understand the codebase without necessarily drafting a simulation.

**Trigger:** Architecture questions ("how does contraction work?"), capability questions ("what gauge groups are supported?"), API questions ("how do I add a new operator?")

**Flow:**
1. Call MCP discovery tools for structured facts
2. Read EXPERIENCE.md for practitioner context
3. Optionally explore source code for deeper questions (read specific files)
4. Answer with references to specific code locations

#### `visualize` (user-facing)

Intelligent visualization of simulation results.

**Trigger:** "Plot the results", "Show me the energy convergence", "Make a heatmap of the plaquettes", "Animate the vison propagation"

**Flow:**
1. User points to a run_dir or latest.json
2. Call MCP `read_checkpoint_metadata(run_dir)` to understand what data is available
3. Suggest what to plot based on available observables and physics context (e.g., "This is a vison dynamics run — want to see plaquette heatmaps at key timesteps?")
4. Ask user what they want
5. Call MCP visualization tools to produce figures/GIFs
6. **Dispatch visualization review agent:** checks axes labels, data interpretation, whether the plot answers the user's question
7. Present figures with interpretation (e.g., "Energy drift is 0.1% — the simulation is stable" or "Energy hasn't converged — consider running more steps")

### EXPERIENCE.md

A living document of practitioner wisdom, queryable via MCP. Organized by topic with tagged entries.

Location: project root `EXPERIENCE.md`

Format:
```markdown
## Contraction Strategy

- **Always use Variational on GPU.** ZipUp involves SVD which is not well batched on GPU. Variational uses iterative sweeps that parallelize better.
- **ZipUp is fine on CPU** for small systems (L <= 6) where SVD cost is manageable.

## Bond Dimension

- **Z2 LGT converges well with D_k=2** for lattice sizes up to 32x32 (Wu & Liu 2025, Table I).
- **For Z2 Higgs, D_k=2 is sufficient** for both deconfined and Higgs phases (Wu & Nys 2026).
- **Start with D_k=2, increase if energy variance is large.**

## Solver Choice

- **Cholesky is default.** Faster and more parallelizable on GPU than SVD.
- **SVD is more robust** for ill-conditioned QGT. Use when Cholesky gives NaN.
- **CG (conjugate gradient)** for very large parameter counts where direct solve is too expensive.

## Solver Space (SR vs minSR)

- **For GIPEPS with large per-site parameters, use minSR** (`SampleSpace`). Avoids materializing the full Jacobian. The z2_vison_higgs examples default to minSR.
- **For standard PEPS, SR** (`ParameterSpace`) is fine — N_p is typically smaller than N_s.
- **When N_s > N_p - N_gv - 2, use minSR.** This is the crossover point (Wu & Nys 2026, Sec. III.C).

## Sampling

- **n_samples=10240, n_chains=1024** is a good starting point for production.
- **For testing/debugging, use n_samples=64, n_chains=8.**

## Convergence

- **FS_norm_squared should decrease** during imaginary-time optimization. If it plateaus, the state is near a local minimum.
- **Energy drift < 0.5%** over the full trajectory indicates stable real-time evolution.
- **TDVP residual 1e-9 to 1e-25** is normal and indicates the TDVP equation is being solved accurately (Wu & Nys 2026, Fig. 6).
```

### Review Agent Prompts

#### Physics Review Agent

```
You are reviewing a simulation script for the PEPS-tVMC codebase.

**Script to review:** [SCRIPT_PATH]
**User's simulation request:** [USER_DESCRIPTION]
**Relevant paper(s):** [PAPER_PATHS]
**EXPERIENCE.md:** [EXPERIENCE_PATH]

## What to Check

1. **Hamiltonian correctness:** Do the operator terms match what the user described? Are coefficients and signs correct? Compare against the paper if applicable.
2. **Model family:** Is PEPS vs GIPEPS vs BlockadePEPS correct for this physics?
3. **Observables:** Do they measure what the user asked for? **Coordinate trap:** the user's plaquette convention is likely bottom-up (open-data / paper convention), but `PlaquetteOperator` uses top-down (internal). Verify `open_to_internal_plaquette()` is applied correctly.
4. **Parameters:** Are bond dimension, samples, dt, diag_shift reasonable for this lattice size? Check against EXPERIENCE.md and paper benchmarks.
5. **Workflow:** For two-stage runs, is the handoff correct (ground state checkpoint → dynamics load)?

## CRITICAL: Verify by Reading Code

Do NOT trust any accompanying explanation. Read the actual script and compare against the paper and user request.

Report:
- Approved (if physics is correct)
- Issues Found: [specific issues with line references]
```

#### Code Review Agent

```
You are reviewing a simulation script for code correctness.

**Script to review:** [SCRIPT_PATH]
**CLAUDE.md:** [CLAUDE_PATH]
**EXPERIENCE.md:** [EXPERIENCE_PATH]

## What to Check

1. **Runner API:** Does the script use runner.run() correctly? Are all required args provided?
2. **Imports:** Are all imports valid? Is the model family compatible with the operator type?
3. **Driver construction:** Are TDVPDriver args correct (time_unit, integrator, preconditioner)?
4. **Type compatibility:** GIPEPS needs GILocalHamiltonian, standard PEPS needs LocalHamiltonian.
5. **Import ordering:** `from vmc import config` must come before any JAX imports (enables float64). This is a common silent bug.
6. **Style:** Follows CLAUDE.md conventions?

Report:
- Approved (if code is correct)
- Issues Found: [specific issues with line references]
```

#### Visualization Review Agent

```
You are reviewing a visualization produced from PEPS-tVMC simulation data.

**Figure/GIF path:** [OUTPUT_PATH]
**Run directory:** [RUN_DIR]
**User's request:** [USER_DESCRIPTION]

## What to Check

1. **Data correctness:** Does the plot show the data the user asked for?
2. **Axes and labels:** Are they correct and readable?
3. **Interpretation:** Would a physicist looking at this figure understand what it shows?
4. **Anomalies:** Any obvious artifacts (NaN, sudden jumps, wrong scale)?

Report:
- Approved
- Issues Found: [specifics]
```

## Implementation Order

1. **EXPERIENCE.md** — create with initial practitioner knowledge
2. **MCP server** — discovery tools, compatibility matrix, experience query, visualization, smoke test
3. **Skills** — simulate, understand-codebase, visualize
4. **Review agent prompts** — physics reviewer, code reviewer, visualization reviewer

## File Structure

```
EXPERIENCE.md                              # project root
tools/vmc-mcp/
  server.py                                # MCP entry point
  discovery.py                             # list_models, list_operators, etc.
  compatibility.py                         # compatibility matrix, check_feasibility
  experience.py                            # query EXPERIENCE.md
  visualization.py                         # plot_convergence, plot_heatmap, animate
  runner_tools.py                          # smoke_test, read_checkpoint_metadata
.claude/skills/
  simulate.md                              # top-level simulation skill
  understand-codebase.md                   # developer/physicist Q&A skill
  visualize.md                             # visualization skill
  prompts/
    physics-reviewer.md                    # physics review agent prompt
    code-reviewer.md                       # code review agent prompt
    visualization-reviewer.md             # visualization review agent prompt
```
