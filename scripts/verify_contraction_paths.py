"""Verify hand-tuned contraction paths against opt_einsum optimal paths.

Uses D=4 (PEPS bond dim) and Dc=16 (boundary MPS bond dim = D^2) as
representative bulk dimensions for a standard PEPS with truncation.

Checks both standalone einsums and implicit multi-step sequences where
sequential operations could share intermediates.
"""
from __future__ import annotations

import numpy as np

try:
    import opt_einsum
except ImportError:
    raise ImportError("opt_einsum is required (JAX dependency)")

# ── Representative dimensions (bulk) ────────────────────────────────
D = 4       # PEPS bond dimension (up, down, mL, mR)
Dc = 16     # Boundary MPS bond dimension (= D^2 with truncation)
p = 2       # Physical dimension


def compare_paths(name, subscript, shapes, hand_path):
    """Compare hand-tuned path vs optimal path for one einsum."""
    operands = [np.empty(s) for s in shapes]

    _, info_hand = opt_einsum.contract_path(
        subscript, *operands, optimize=hand_path,
    )
    path_opt, info_opt = opt_einsum.contract_path(
        subscript, *operands, optimize="optimal",
    )

    print(f"\n{'─'*70}")
    print(f"  {name}")
    print(f"  {subscript}  shapes={shapes}")
    print(f"  Hand  path={hand_path}")
    print(f"        cost={info_hand.opt_cost:>12,}  "
          f"largest_intermediate={info_hand.largest_intermediate:>8,}")
    print(f"  Opt   path={list(path_opt)}")
    print(f"        cost={info_opt.opt_cost:>12,}  "
          f"largest_intermediate={info_opt.largest_intermediate:>8,}")

    if info_hand.opt_cost > info_opt.opt_cost:
        ratio = info_hand.opt_cost / info_opt.opt_cost
        print(f"  ** SUBOPTIMAL: hand is {ratio:.2f}x optimal **")
    elif info_hand.opt_cost == info_opt.opt_cost:
        print(f"  OK (optimal)")
    else:
        # Shouldn't happen — might indicate different cost model
        print(f"  OK (hand cheaper than 'optimal'?)")

    return info_hand.opt_cost, info_opt.opt_cost


# ════════════════════════════════════════════════════════════════════
# PART 1: Standalone einsums
# ════════════════════════════════════════════════════════════════════
print("=" * 70)
print("PART 1: STANDALONE EINSUMS")
print("=" * 70)

# ── 1-row environments ──────────────────────────────────────────────

# Index dimensions for 1-row contractions:
#   Boundary MPS tensors: (Dc, D, Dc)
#     top_env[col]:    (a=Dc, u=D, b=Dc)   u = PEPS up-bond (boundary "phys")
#     bottom_env[col]: (e=Dc, v=D, f=Dc)   v = PEPS down-bond
#   MPO tensor: (D, D, D, D)
#     mpo[col]:        (c=D, d=D, u=D, v=D) = (mL, mR, up, down)
#   Left/right env: (Dc, D, Dc)
#     left_env:        (a=Dc, c=D, e=Dc)    = (top_left, mpo_left, bottom_left)
#     right_env:       (b=Dc, d=D, f=Dc)    = (top_right, mpo_right, bottom_right)

compare_paths(
    "_compute_right_envs (1-row)",
    "aub,cduv,evf,bdf->ace",
    shapes=[
        (Dc, D, Dc),   # top_env[c+1]
        (D, D, D, D),   # mpo[c+1]
        (Dc, D, Dc),   # bottom_env[c+1]
        (Dc, D, Dc),   # right_envs[c+1]
    ],
    hand_path=[(0, 3), (0, 2), (0, 1)],
)

compare_paths(
    "_update_left_env_1row",
    "ace,aub,cduv,evf->bdf",
    shapes=[
        (Dc, D, Dc),   # left_env
        (Dc, D, Dc),   # top_env[col]
        (D, D, D, D),   # mpo[col]
        (Dc, D, Dc),   # bottom_env[col]
    ],
    hand_path=[(0, 1), (0, 2), (0, 1)],
)

compare_paths(
    "_compute_single_gradient",
    "ace,aub,evf,bdf->ucvd",
    shapes=[
        (Dc, D, Dc),   # left_env
        (Dc, D, Dc),   # top_env[col]
        (Dc, D, Dc),   # bottom_env[col]
        (Dc, D, Dc),   # right_env[col]
    ],
    hand_path=[(0, 1), (0, 1), (0, 1)],
)

compare_paths(
    "transition_amplitude (5-tensor → scalar)",
    "ace,aub,cduv,evf,bdf->",
    shapes=[
        (Dc, D, Dc),   # left_env
        (Dc, D, Dc),   # top_env[col]
        (D, D, D, D),   # mpo[col]
        (Dc, D, Dc),   # bottom_env[col]
        (Dc, D, Dc),   # right_env[col]
    ],
    hand_path=[(0, 1), (1, 2), (1, 2), (0, 1)],
)

# ── Two-site operator einsums (1-row) ──────────────────────────────

# HorizontalTwoSiteOperator: 8-tensor einsum
# Index dimensions:
#   a=Dc (top left), b=Dc (top mid), g=Dc (top right)
#   e=Dc (bot left), f=Dc (bot mid), i=Dc (bot right)
#   c=D  (mpo left), r=D (mpo mid), x=D (mpo right)
#   u=D  (up₀), d=D (down₀), v=D (up₁), w=D (down₁)
#   p=2  (phys₀), q=2 (phys₁)

compare_paths(
    "_eval_term(HorizontalTwoSiteOperator)",
    "ace,aub,edf,pudcr,qvwrx,bvg,fwi,gxi->pq",
    shapes=[
        (Dc, D, Dc),           # left_env:       (a, c, e)
        (Dc, D, Dc),           # top_env[col]:   (a, u, b)
        (Dc, D, Dc),           # bottom_env[col]:(e, d, f)
        (p, D, D, D, D),       # PEPS[col]:      (p, u, d, c, r)
        (p, D, D, D, D),       # PEPS[col+1]:    (q, v, w, r, x)
        (Dc, D, Dc),           # top_env[col+1]: (b, v, g)
        (Dc, D, Dc),           # bot_env[col+1]: (f, w, i)
        (Dc, D, Dc),           # right_env[col+1]:(g, x, i)
    ],
    hand_path=[(0, 1), (1, 6), (0, 5), (1, 3), (1, 2), (1, 2), (0, 1)],
)

# ── 2-row environments ──────────────────────────────────────────────

# Index dimensions for 2-row contractions:
#   2-row left_env:  (a=Dc, l=D, m=D, g=Dc) = (top_L, mpo0_L, mpo1_L, bot_L)
#   2-row right_env: (b=Dc, r=D, n=D, f=Dc) = (top_R, mpo0_R, mpo1_R, bot_R)
#   top_env:         (a=Dc, u=D, b=Dc)
#   bottom_env_next: (g=Dc, w=D, f=Dc)
#   mpo row0:        (l=D, r=D, u=D, v=D)
#   mpo row1:        (m=D, n=D, v=D, w=D)  (note: up=v connects row0 down to row1 up)
#   PEPS[row,col]:   (p=2, u=D, v=D, l=D, r=D)
#   PEPS[row+1,col]: (q=2, v=D, w=D, m=D, n=D)

compare_paths(
    "_compute_right_envs_2row",
    "aub,lruv,xyvw,ewf,bryf->alxe",
    shapes=[
        (Dc, D, Dc),       # top_env[c+1]
        (D, D, D, D),       # mpo_row0[c+1]
        (D, D, D, D),       # mpo_row1[c+1]
        (Dc, D, Dc),       # bottom_env_next[c+1]
        (Dc, D, D, Dc),    # right_envs_2row[c+1]
    ],
    hand_path=[(0, 4), (0, 3), (0, 2), (0, 1)],
)

compare_paths(
    "_update_left_env_2row",
    "alxe,aub,lruv,xyvw,ewf->bryf",
    shapes=[
        (Dc, D, D, Dc),    # left_env_2row
        (Dc, D, Dc),       # top_env[col]
        (D, D, D, D),       # mpo_row0[col]
        (D, D, D, D),       # mpo_row1[col]
        (Dc, D, Dc),       # bottom_env_next[col]
    ],
    hand_path=[(0, 1), (0, 3), (0, 2), (0, 1)],
)

# VerticalTwoSiteOperator: 6-tensor einsum
compare_paths(
    "_eval_term(VerticalTwoSiteOperator)",
    "almg,aub,puvlr,qvwmn,gwf,brnf->pq",
    shapes=[
        (Dc, D, D, Dc),        # left_env_2row:   (a, l, m, g)
        (Dc, D, Dc),           # top_env[col]:    (a, u, b)
        (p, D, D, D, D),       # PEPS[row,col]:   (p, u, v, l, r)
        (p, D, D, D, D),       # PEPS[row+1,col]: (q, v, w, m, n)
        (Dc, D, Dc),           # bot_env_next[col]:(g, w, f)
        (Dc, D, D, Dc),        # right_env_2row:  (b, r, n, f)
    ],
    hand_path=[(0, 1), (2, 3), (0, 2), (1, 2), (0, 1)],
)

# ════════════════════════════════════════════════════════════════════
# PART 2: Implicit multi-step sequences
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 2: IMPLICIT SEQUENCES (shared intermediates)")
print("=" * 70)

# ── Sequence A: gradient + left_env_update at same column ──────────
# Both share: left_env(ace) × top_env(aub) → intermediate(ceub)
# Currently computed independently in each function.

print(f"\n{'─'*70}")
print("  Sequence A: gradient + left_env_update (same column)")
print("  Current: two independent 4-tensor einsums")
print("  Proposed: share left_env × top_env intermediate")

# Current costs (from PART 1 analysis)
shapes_left = (Dc, D, Dc)
shapes_top = (Dc, D, Dc)
shapes_bot = (Dc, D, Dc)
shapes_right = (Dc, D, Dc)
shapes_mpo = (D, D, D, D)

ops_grad = [np.empty(s) for s in [shapes_left, shapes_top, shapes_bot, shapes_right]]
_, info_grad = opt_einsum.contract_path(
    "ace,aub,evf,bdf->ucvd", *ops_grad, optimize="optimal",
)

ops_left = [np.empty(s) for s in [shapes_left, shapes_top, shapes_mpo, shapes_bot]]
_, info_left = opt_einsum.contract_path(
    "ace,aub,cduv,evf->bdf", *ops_left, optimize="optimal",
)
current_cost = info_grad.opt_cost + info_left.opt_cost

# Proposed: pre-compute intermediate, then two 3-tensor einsums
shapes_inter = (D, Dc, D, Dc)  # ceub
ops_pre = [np.empty(shapes_left), np.empty(shapes_top)]
_, info_pre = opt_einsum.contract_path("ace,aub->ceub", *ops_pre, optimize="optimal")

ops_grad2 = [np.empty(shapes_inter), np.empty(shapes_bot), np.empty(shapes_right)]
_, info_grad2 = opt_einsum.contract_path("ceub,evf,bdf->ucvd", *ops_grad2, optimize="optimal")

ops_left2 = [np.empty(shapes_inter), np.empty(shapes_mpo), np.empty(shapes_bot)]
_, info_left2 = opt_einsum.contract_path("ceub,cduv,evf->bdf", *ops_left2, optimize="optimal")

proposed_cost = info_pre.opt_cost + info_grad2.opt_cost + info_left2.opt_cost

print(f"  Current:  gradient={info_grad.opt_cost:>10,} + "
      f"left_update={info_left.opt_cost:>10,} = {current_cost:>10,}")
print(f"  Proposed: precompute={info_pre.opt_cost:>10,} + "
      f"gradient={info_grad2.opt_cost:>10,} + "
      f"left_update={info_left2.opt_cost:>10,} = {proposed_cost:>10,}")
print(f"  Savings per column: {current_cost - proposed_cost:,} "
      f"({100*(current_cost - proposed_cost)/current_cost:.1f}%)")
print(f"  Note: XLA CSE may already eliminate this redundancy at compile time.")

# ── Sequence B: gradient + left_env_update + HorizTwoSite at same col ──
# The horizontal term also starts with left_env × top_env.
print(f"\n{'─'*70}")
print("  Sequence B: gradient + left_env_update + HorizTwoSite (same column)")
print("  All three start with left_env(ace) × top_env(aub) → ceub")

shapes_htso = [
    (Dc, D, Dc), (Dc, D, Dc), (Dc, D, Dc),
    (p, D, D, D, D), (p, D, D, D, D),
    (Dc, D, Dc), (Dc, D, Dc), (Dc, D, Dc),
]
ops_htso = [np.empty(s) for s in shapes_htso]
_, info_htso = opt_einsum.contract_path(
    "ace,aub,edf,pudcr,qvwrx,bvg,fwi,gxi->pq",
    *ops_htso, optimize="optimal",
)
# Check: does the optimal path for the 8-tensor einsum also start with ace×aub?
path_htso_opt, _ = opt_einsum.contract_path(
    "ace,aub,edf,pudcr,qvwrx,bvg,fwi,gxi->pq",
    *ops_htso, optimize="optimal",
)
htso_starts_with_ace_aub = path_htso_opt[0] == (0, 1)

total_without_sharing = info_grad.opt_cost + info_left.opt_cost + info_htso.opt_cost
# With sharing: save one precompute per additional operation that uses ceub
# The 8-tensor einsum embeds the precompute in step 1
shared_savings = info_pre.opt_cost  # one precompute saved if gradient shares with htso
if htso_starts_with_ace_aub:
    shared_savings += info_pre.opt_cost  # another if left_update shares with htso

print(f"  Gradient cost:       {info_grad.opt_cost:>10,}")
print(f"  Left_update cost:    {info_left.opt_cost:>10,}")
print(f"  HorizTwoSite cost:   {info_htso.opt_cost:>10,}")
print(f"  Total (independent): {total_without_sharing:>10,}")
print(f"  Optimal 8-tensor path starts with ace×aub (0,1): {htso_starts_with_ace_aub}")
print(f"  Potential savings from sharing ceub: "
      f"up to {shared_savings:,} per column")

# ── Sequence C: Full column iteration cost model ────────────────────
print(f"\n{'─'*70}")
print("  Sequence C: Full backward-sweep cost per row (1-row terms)")
print("  = right_envs + n_cols × (gradient + left_update + terms)")

# Right env computation: n_cols-1 steps of _compute_right_envs
ops_renv = [np.empty(s) for s in [
    (Dc, D, Dc), (D, D, D, D), (Dc, D, Dc), (Dc, D, Dc),
]]
_, info_renv = opt_einsum.contract_path(
    "aub,cduv,evf,bdf->ace", *ops_renv, optimize="optimal",
)

n_cols = 12
renv_cost = (n_cols - 1) * info_renv.opt_cost
grad_cost = n_cols * info_grad.opt_cost
left_cost = n_cols * info_left.opt_cost
# Assume ~1 horizontal term per column on average (varies by Hamiltonian)
htso_cost = n_cols * info_htso.opt_cost
total_row = renv_cost + grad_cost + left_cost + htso_cost

print(f"  n_cols = {n_cols}")
print(f"  Right envs:  {n_cols-1} × {info_renv.opt_cost:>10,} = {renv_cost:>12,}")
print(f"  Gradients:   {n_cols} × {info_grad.opt_cost:>10,} = {grad_cost:>12,}")
print(f"  Left updates:{n_cols} × {info_left.opt_cost:>10,} = {left_cost:>12,}")
print(f"  Horiz terms: {n_cols} × {info_htso.opt_cost:>10,} = {htso_cost:>12,}")
print(f"  Total per row: {total_row:>12,}")
print(f"  Total per sweep (12 rows): {12 * total_row:>12,}")

# ════════════════════════════════════════════════════════════════════
# PART 3: Summary
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
