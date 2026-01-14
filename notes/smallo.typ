#import "@preview/physica:0.9.4": *

= Small-o tricks

For PEPS, the parameter index $alpha$ factorizes as
$
  alpha equiv (mono(x), p, l r d u)
$
where $mono(x)$ is the site index, $p$ is the physical index and $l r d u$ are the virtual indices.

There is a fact:
$
  O[mono(x)](s)_(p, l r d u) = 0 "if" p != s(mono(x))
$

One thus only keeps non-zero elements in Jacobians $o[mono(x)](s)_(l r d u) = O[mono(x)](s)_(s(mono(x)), l r d u)$. The point is in how to use these small-os' to reconstruct $O^dagger O$ or $O O^dagger$ and choose different implementations for a better tradeoff between memory and efficiency.

== SR
$
  sum_s O^dagger_(alpha s) O_(s alpha^prime) &= sum_s O^dagger [mono(x)](s)_(p, l r d u) O[mono(x^prime)](s)_(p^prime, l^prime r^prime d^prime u^prime)\
  &= sum_s O^dagger [mono(x)](s)_(p, l r d u) O[mono(x^prime)](s)_(p^prime, l^prime r^prime d^prime u^prime) bold(1)[p = s(mono(x))] bold(1)[p^prime = s (mono(x^prime))]\
  &= sum_s underbrace(o^dagger [mono(x)](s)_(l r d u) bold(1)[p = s(mono(x))], tilde(o)^dagger_"SR") quad underbrace(o[mono(x^prime)](s)_(l^prime r^prime d^prime u^prime) bold(1)[p^prime = s (mono(x^prime))], tilde(o)_"SR")\
$

== minSR
$
    sum_alpha O_(s alpha) O^dagger_(alpha s^prime) &= sum_mono(x) sum_p sum_(l r d u) O[mono(x)](s)_(p, l r d u) O^dagger [mono(x)](s^prime)_(p, l r d u)\
    &= sum_mono(x) sum_p sum_(l r d u) O[mono(x)](s)_(p, l r d u) O^dagger [mono(x)](s^prime)_(p, l r d u) bold(1)[p = s(mono(x))] bold(1)[p = s^prime (mono(x))] \
    &= sum_p sum_mono(x) sum_(l r d u) underbrace(o[mono(x)](s)_(l r d u) bold(1)[p = s(mono(x))], tilde(o)_"minSR") quad underbrace(o^dagger [mono(x)](s^prime)_(l r d u) bold(1)[p = s^prime (mono(x))], tilde(o)^dagger_"minSR") \
$
which is exactly the small-o trick in minSR, with a mere difference that one has no need to store a $N_s times N_p$ sample tensor. But maybe in practice one might prefer calculating a $N_s times N_p$ sample tensor for a better dense GEMM.

=== enumeration order
For the small-o trick, one enumerates over $p$ first, which replace . One can also consider other enumeration orders. replace $sum_p$ first: One $(N_s, N_p) times (N_p, N_s)$ GEMM with:
- $sum_p$ first: $arrow$ $dim(p)$-many $(N_s, N_p/dim(p)) times (N_p/dim(p), N_s)$ ones
- $sum_x$ first: $arrow$ $n_"site"$-many $(N_s, N_p/n_"site") times (N_p/n_"site", N_s)$ ones

I think one should follow the principle that:
1. Gradients should be recorded as block-sparse.
2. If you have enough memory, enumerate over the index with the smallest dimension (e.g. $dim(p)=2$ for qubits) for large dense GEMMs.

== LGT
For gauge-invariant PEPS, the gradients are extremly sparse:
$
  O[mono(x)](s)_(p, l r d u) = 0 "if" p != s(mono(x)) "and" l r d u in.not "virtual space specified by "
$

== PEPS with nearest blockade
Require all virtual spaces to be a direct sum: $V_0 plus.o V_1$.

$ A_(p,l r d u) != 0 "for" $

#table(
  columns: (auto, auto, auto, auto, auto),
  align: center,
  table.header(
    text(9pt)[$p$],
    text(9pt)[$u$],
    text(9pt)[$l$],
    text(9pt)[$d$],
    text(9pt)[$r$]
  ),
)

The number of parameters is $"O"()$