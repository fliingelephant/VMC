---
name: visualize
description: Visualize PEPS-tVMC simulation results from runner output. Produces convergence plots, 2D spatial heatmaps, and time-evolution GIF animations with physical interpretation. Use when the user wants to plot results, see convergence, make heatmaps, or animate dynamics.
---

# Visualize

Produce and interpret visualizations from PEPS-tVMC simulation output.

## Step 1: Load data

Ask the user for the `run_dir` path (the directory containing `latest.json` and `latest/`).

Call MCP `tool_read_checkpoint_metadata` to understand what data is available:
- What step and time the simulation reached
- What observables were tracked
- The simulation config (model, lattice size, physics parameters)

## Step 2: Suggest visualizations

Based on the available data, suggest what to plot. Use the physics context:

- **Energy convergence** — always available. For imaginary-time runs, check if energy has converged (variance decreasing). For real-time, check energy conservation (drift).
- **Observable convergence** — if named observables are present (magnetization, plaquette values, correlators), suggest plotting them.
- **2D spatial heatmaps** — if observable names match a grid pattern (e.g., `P_0_0_mean`, `P_0_1_mean`, ...), suggest a plaquette heatmap at a specific time.
- **Animation** — if spatial observables exist over multiple time steps, suggest a GIF animation (especially useful for vison propagation).

Ask the user what they want. Multiple choices OK.

## Step 3: Generate visualizations

For each requested visualization:

- **Convergence:** Call MCP `tool_plot_convergence(run_dir, keys)`. If the user asked for specific observables, pass them as keys.
- **Heatmap:** Call MCP `tool_plot_heatmap(run_dir, step, prefix)`. Ask which time step if not specified. Suggest the last step, or key moments (e.g., "step 0 and final step for comparison").
- **Animation:** Call MCP `tool_animate(run_dir, prefix, fps)`.

## Step 4: Interpret results

After generating each visualization, provide physical interpretation:

- **Energy convergence (imaginary time):** "Energy decreased from X to Y over N steps. Variance is Z — [converged / still decreasing / plateau suggests local minimum]."
- **Energy conservation (real time):** "Energy drift is X% — [stable / concerning]. Reference: EXPERIENCE.md says < 0.5% is good."
- **Plaquette heatmap:** "Plaquettes are [uniform / showing VBS pattern / localized excitation]. For vison dynamics, the bright spots indicate vison positions."
- **Animation:** "The vison [spreads / remains localized], consistent with [deconfined / confined] phase."

Call MCP `tool_query_experience("convergence")` or relevant topics for reference values.

## Step 5: Review

Dispatch a **visualization review agent** (spawn as subagent):

```
You are reviewing a visualization from PEPS-tVMC simulation data.

Figure/GIF: [PATH]
Run directory: [RUN_DIR]
User's request: [DESCRIPTION]

Check:
1. Does the plot show the data the user asked for?
2. Are axes and labels correct and readable?
3. Would a physicist understand what this shows?
4. Any obvious artifacts (NaN, sudden jumps, wrong scale)?

Report: Approved | Issues Found [specifics]
```

Fix issues if found. Present the final visualization with interpretation.
