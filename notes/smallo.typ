#import "@preview/physica:0.9.4": *

In the PEPS setting, the parameter index $alpha$ naturally factorizes as
$
  alpha equiv (mono(x), p, l r d u)
$
where $mono(x)$ is the site index, $p$ is the physical index and $l r d u$ are the virtual indices.

There is a fact:
$
  O[mono(x)](s)_(p, l r d u) = 0 "if" p != s(mono(x))
$
    
One thus only keeps non-zero elements in Jacobians $o[mono(x)](s)_(l r d u) = O[mono(x)](s)_(s(mono(x)), l r d u)$ to improve minSR:

$
    sum_alpha O_(s alpha) O^dagger_(alpha s^prime) &= sum_mono(x) sum_p sum_(l r d u) O[mono(x)](s)_(p, l r d u) O^dagger [mono(x)](s^prime)_(p, l r d u)\
    &=^"Fact" sum_mono(x) sum_p sum_(l r d u) O[mono(x)](s)_(p, l r d u) O^dagger [mono(x)](s^prime)_(p, l r d u) bold(1)[p = s(mono(x))] bold(1)[p = s^prime (mono(x))] \
    &= sum_mono(x) sum_p sum_(l r d u) underbrace(O[mono(x)](s)_(s(mono(x)), l r d u), o[mono(x)](s)_(l r d u)) underbrace(O^dagger [mono(x)](s^prime)_(s^prime (mono(x)), l r d u), o^dagger [mono(x)](s^prime)_(l r d u)) bold(1)[p = s(mono(x))] bold(1)[p = s^prime (mono(x))] \
    &= sum_p sum_mono(x) sum_(l r d u) underbrace(o[mono(x)](s)_(l r d u) bold(1)[p = s(mono(x))], tilde(o)) underbrace(o^dagger [mono(x)](s^prime)_(l r d u) bold(1)[p = s^prime (mono(x))], tilde(o)^dagger) \
$
which is exactly the small-o trick in minSR, with a mere difference that one has no need to store a $N_s times N_p$ sample tensor. But maybe in practice one might prefer calculating a $N_s times N_p$ sample tensor for a better dense GEMM.

For the small-o trick, one enumerates over $p$. One can also consider other enumeration orders.

=
For LGT cases, the fact mentioned above becomes:
$
  O[mono(x)](s)_(p, l r d u) = 0 "if" p != s(mono(x)) "and" l r d u in.not "virtual space specified by "
$