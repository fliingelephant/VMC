#import "@preview/physica:0.9.4": *

For any operator $A$ (e.g., Hamiltonian $H$ or partial derivative operator $partial_theta$), the quantum expectation reads:
$
expval(A) &:= braket(psi, A, psi) / braket(psi, psi) \
&= (sum_s braket(psi, s) braket(s, A, psi)) / braket(psi, psi) &("insert complete basis") \
&= (sum_s braket(psi, s) #text(fill: red)[$braket(s, psi)$] braket(s, A, psi)) / (braket(psi, psi) #text(fill: red)[$braket(s, psi)$]) quad&("multiply and divide by" braket(s, psi)) \
&= sum_s (|braket(s, psi)|^2) / (braket(psi, psi)) times braket(s, A, psi) / braket(s, psi)  \
&=: sum_s p(s) A_"loc"(s) &("Born probability")
$

This recasts the quantum expectation as a classical average over Born probability $p(s)$ with local estimator $A_"loc" (s) := braket(s, A, psi) / braket(s, psi)$.

If $braket(s, psi) = 0$ for certain configuration $s$, local estimator diverges:
$
  underbrace(p(s), 0) times underbrace(A_"loc"(s), infinity) = underbrace((braket(psi, s) braket(s, A, psi)) / braket(psi, psi), "finite") 
$
The product remains finite, but MC sampling cannot handle this: we would need to sample an infinite value with zero probability. This arises from dividing by $braket(s, psi)=0$ in the derivation above.

== MC Evaluation of Hamiltonian

For Hamiltonian $H$, the energy expectation is:
$
  E = expval(H) = sum_s p(s) E_"loc"(s), quad E_"loc"(s) := braket(s, H, psi) / braket(s, psi) = sum_(s') H_(s s') braket(s', psi) / braket(s, psi)
$

*MC estimator*: Draw i.i.d. samples $s_1, ..., s_N tilde p(s) = |braket(s, psi)|^2 \/ Z$, then:
$
  hat(E) = 1/N sum_(i=1)^N E_"loc"(s_i)
$

This estimator is *unbiased*: $EE[hat(E)] = E$. At exact nodes where $braket(s, psi) = 0$:
- $p(s) = 0$, so nodes are never sampled
- The contribution $braket(psi, s) braket(s, H, psi) = 0$ vanishes in the true expectation

However, *near* nodes where $|braket(s, psi)| approx 0$:
- $p(s) approx 0$: rare samples
- $E_"loc"(s)$: potentially large

This causes *high variance*, not bias. Rare samples with large $E_"loc"$ slow convergence.