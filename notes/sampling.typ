#import "@preview/physica:0.9.4": *

#set page(margin: (x: 1in, y: 1in))
#set text(size: 11pt)
#set heading(numbering: "1.")

#align(center)[
  #text(size: 16pt, weight: "bold")[
    Sampling of Independent Sets
  ]
  #text(size: 11pt)[Summary]
  #text(size: 10pt)[January 6, 2026]
]

#block(
  inset: 8pt,
  stroke: (paint: luma(200), thickness: 0.6pt),
  radius: 4pt,
)[
  *One-line goal.* Aim to sample independent sets $n in cal(I)(G)$ from the
  Born distribution $pi_theta(n) prop abs(psi_theta(n))^2$ to obtain
  lower-variance, lower-autocorrelation estimators in VMC for Rydberg blockade
  Hamiltonians.
]

= Background: Rydberg blockade as an independent-set constraint

In the strong blockade regime, a Rydberg array with binary occupation variables
$n_i in {0,1}$ admits only configurations in which no two excited atoms violate
the blockade radius. Defining a graph $G = (V, E)$ where $(i, j) in E$ when atoms
$i, j$ are within the blockade distance, the allowed configurations coincide with
independent sets:

#align(center)[
  $cal(I)(G) = { n in {0,1}^{|V|}: n_i n_j = 0 forall (i, j) in E }. $
]

This is a common mapping used to encode (maximum) independent set optimization
on Rydberg hardware.

VMC optimization of a variational state $psi_theta(n)$ requires samples from

#align(center)[
  $pi_theta(n) = frac(abs(psi_theta(n))^2, Z_theta),
  quad n in cal(I)(G),
  quad Z_theta = sum_(n in cal(I)(G)) abs(psi_theta(n))^2.$
]

The primary practical issue is constructing samplers that (i) respect the hard
constraint, (ii) mix across near-degenerate patterns (e.g., ordered/stripe
phases), and (iii) remain computationally practical given the expense of
evaluating $psi_theta$.

= Why sampling is hard in practice

Independent-set spaces become rugged when the target distribution is strongly
biased toward high-density configurations or when multiple symmetry-related
patterns compete. Common failure modes include:

- *Low acceptance at high density:* most empty vertices are blocked, so
  insertions are rarely valid.
- *Collective slow modes:* changing between stripe or checkerboard-like patterns
  requires moving many excitations coherently; single-site flips diffuse.
- *Multimodality:* distinct basins separated by entropic barriers lead to
  metastability and long autocorrelation times.

= Sampling independent sets: a practical catalogue

== A. Local, constraint-aware updates (baseline, inexpensive)

*A1. Insert-delete Metropolis (Glauber-like) with fast constraint checks.*
Maintain (i) adjacency lists $cal(N)(i)$ and (ii) a blocked count
$b_i = sum_(j in cal(N)(i)) n_j$. Then insertion at $i$ is allowed iff $b_i = 0$.

For a proposal $n -> n'$, accept with

#align(center)[
  $alpha = min(1,
    frac(abs(psi_theta(n'))^2, abs(psi_theta(n))^2)
    dot frac(q(n | n'), q(n' | n))
  ).$
]

*A2. Single-site heat-bath (Gibbs) updates (often better).* At a chosen site $i$,
consider the two configurations differing only in $n_i$. If $b_i > 0$, the
$n_i = 1$ option has zero weight. Otherwise, sample

#align(center)[
  $Pr(n_i = 1 | n_(-i)) = frac(abs(psi_theta(n^(1)))^2,
    abs(psi_theta(n^(0)))^2 + abs(psi_theta(n^(1)))^2).$
]

This is rejection-free and can reduce wasted evaluations when Metropolis
acceptance is low.

*A3. Locally balanced / informed site selection.* Bias the choice of update site
toward flippable locations or those with large expected amplitude ratio. Correct
any asymmetry with the proposal ratio. This is a relatively low-engineering way
to raise acceptance.

== B. Swap / pivot moves (high impact near maximal independent sets)

When insertions are rare, add moves that keep density approximately fixed:

- Pick an empty vertex $v$ blocked by exactly one occupied neighbor $u$.
- Propose a swap $n' = n - {u} + {v}$.

This can allow efficient surface diffusion along near-maximal independent sets
and is often helpful in MIS-like regimes.

== C. Patch / strip (block) updates to destroy domain walls

To accelerate collective rearrangements, propose updates that rewrite an entire
region $P subset V$ while keeping the boundary fixed.

*Patch proposal template.*

+ Choose a patch $P$ (e.g., a $k x k$ square in a 2D array or a stripe).
+ Remove occupations inside $P$, keep $partial P$ fixed.
+ Propose a new independent-set pattern in $P$ given the boundary using a fast
  proposal $q_P$:
  - dynamic programming / transfer-matrix for narrow stripes,
  - precomputed enumeration for very small patches.
+ Accept with MH using the global $abs(psi_theta)^2$ ratio and $q_P$ ratio.

*Inspiration from classical hard-core gases.* Rejection-free strip updates that
evaporate all particles in a strip and reoccupy it efficiently sample
low-entropy/high-density regimes in hard-core lattice gases; the same move
geometry is often useful as a proposal inside VMC.

== D. Rejection-free kinetic scheduling (n-fold way / BKL)

When invalid or rejected proposals dominate, use rejection-free event selection.
Define a set of allowed move types $m$ with rates $r_m(n)$ that satisfy (global
or detailed) balance with respect to $pi_theta$. Sample the next move according
to rates and update the configuration without rejection. This can improve
efficiency when acceptance is small.

== E. Tempering / population strategies for multimodality

*Parallel tempering in a Born exponent.* Run replicas targeting

#align(center)[
  $pi_(theta, beta)(n) prop abs(psi_theta(n))^(2 beta),
  quad 0 < beta <= 1.$
]

and occasionally swap neighboring $beta$-replicas with replica-exchange
acceptance. Smaller $beta$ flattens the landscape and helps the chain traverse
between modes.

== F. Learned global proposals (flows or other generative models) with MH correction

When the wavefunction is not directly sampleable, train a separate proposal model
$q_phi(n)$ over $cal(I)(G)$ (e.g., a flow-based generator). Use independence
Metropolis-Hastings:

#align(center)[
  $alpha = min(1,
    frac(abs(psi_theta(n'))^2 q_phi(n), abs(psi_theta(n))^2 q_phi(n'))
  ).$
]

Mixing local kernels with occasional global proposals can reduce autocorrelation
when $abs(psi_theta)^2$ is multi-modal.

= A recommended sampler "stack" for Rydberg-VMC

A practical starting point is a mixture kernel combining complementary moves:

- *Every sweep:* single-site Gibbs (or insert-delete MH) using flippable-site
  lists.
- *Every few sweeps:* swap/pivot moves to improve mobility at high density.
- *Periodically:* patch or stripe proposals (MH-corrected) to heal domain walls.
- *If bimodal or hysteretic:* parallel tempering in $beta$ on
  $abs(psi)^(2 beta)$.
- *Optional:* learned global proposals (flows or other generative models) mixed
  into the kernel.

#block(
  inset: 8pt,
  stroke: (paint: luma(200), thickness: 0.6pt),
  radius: 4pt,
)[
  *Rule of thumb.* A move is often worthwhile if it reduces variance per unit
  cost. A patch move that is 10-20x more expensive than a local move can still be
  a net win if it reduces integrated autocorrelation times by 100x.
]

= Implementation notes (engineering details that matter)

*Data structures.* For a geometric blockade graph (unit-disk-like), store
adjacency lists and maintain blocked counts $b_i$. Maintain dynamic sets:

- occupied sites ${i: n_i = 1}$,
- insertable sites ${i: n_i = 0, b_i = 0}$,
- swappable pairs $(u -> v)$ where $u$ is occupied, $v$ is empty, and $v$ is
  blocked by exactly one neighbor $u$.

These can be updated in $O(deg(i))$ per accepted local move.

*Amplitude ratios.* Most MH or Gibbs decisions only require
$log(abs(psi(n'))) - log(abs(psi(n)))$. For local moves, consider caching or
incremental evaluation whenever the ansatz supports it.

*Diagnostics.* Track integrated autocorrelation times $tau_"int"$ for key
observables (excitation number, structure factors, energy), and report effective
sample size (ESS) per wall-clock second. Tune mixture weights to maximize
ESS/sec.
