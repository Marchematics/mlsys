# General smooth-loss response–utility theory (PAMI addition A3)

Status: derivation complete and numerically validated (see
`results/derived/R102_response_theory/`). Scope limits are stated explicitly at
the end; nothing here is claimed beyond the assumptions.

## 1. Setup

Let `f_A(x) in F subset R^T` be the prediction of a predictor that consumes
the anchor context `S_0` and the selected context subset `A`, where `T` is the
prediction dimension (a horizon for regression, the number of classes for
logits). Let `y` be the target, `ell: F x Y -> R` the task loss, and

```
d_{A,j}(x) = f_{A u {j}}(x) - f_A(x)
```

the **predictor response** to adding candidate `j` at state `A`. The marginal
utility is

```
m(j | A) = E[ ell(f_A(x), y) - ell(f_{A u {j}}(x), y) ] - lambda c_j .
```

(Observation-level version: `m~(j|A) = ell(f_A(x),y) - ell(f_{A u {j}}(x),y) - lambda c_j`.)

The predictor is frozen while utilities are generated; all expectations are
over the episode distribution at fixed state.

## 2. Proposition 1 (response expansion and two-sided bound)

**Assumption A1 (local smoothness).** For every episode in the support,
`f -> ell(f, y)` is twice continuously differentiable on the segment
`[f_A(x), f_{A u {j}}(x)]`.

**Assumption A2 (L-smoothness).** There is `L < inf` with
`|| grad^2 ell(z, y) ||_2 <= L` for every `z` on that segment (equivalently
`grad ell(., y)` is `L`-Lipschitz there).

**Claim.** Under A1,

```
m(j|A) = - E[ grad ell(f_A, y)^T d_{A,j} ] + R_j - lambda c_j ,
R_j = - 1/2 E[ d_{A,j}^T grad^2 ell(xi, y) d_{A,j} ] ,        (1)
```

with `xi` on the segment, and under A2 additionally

```
| m(j|A) + E[grad ell(f_A,y)^T d_{A,j}] + lambda c_j | <= (L/2) E||d_{A,j}||^2 .   (2)
```

Equivalently the sandwich

```
- E[grad^T d] - (L/2) E||d||^2 - lambda c_j  <=  m(j|A)
                                             <=  - E[grad^T d] + (L/2) E||d||^2 - lambda c_j .  (3)
```

*Proof.* Taylor's theorem with Lagrange remainder applied to
`t -> ell(f_A + t d, y)` on `[0,1]` gives
`ell(f_A + d, y) = ell(f_A,y) + grad ell(f_A,y)^T d + 1/2 d^T grad^2 ell(xi,y) d`.
Rearranging and taking expectations gives (1). Under A2,
`|d^T grad^2 ell(xi,y) d| <= L ||d||^2`, hence (2) and (3). QED

**Reading.** The leading term is the alignment between the loss gradient at
the current prediction and the direction in which the candidate moves the
predictor. The remainder is *second order in the response magnitude*: small
responses make the first-order term accurate, and a candidate whose response is
large carries an error of at most `(L/2)||d||^2`.

## 3. Proposition 2 (exact identity for Bregman losses)

Let `ell(f,y) = D_phi(y, f) = phi(y) - phi(f) - grad phi(f)^T (y - f)` with
`phi` convex and differentiable. This covers squared loss (`phi(z) = ||z||^2`),
the Poisson/deviance losses, and other exponential-family negative
log-likelihoods in their natural parametrisation.

**Claim.**

```
m(j|A) = E[ (grad phi(f_{A u {j}}) - grad phi(f_A))^T (y - f_A) ]
         - E[ D_phi(f_A, f_{A u {j}}) ] - lambda c_j .        (4)
```

*Proof.* The Bregman three-point identity states, for any `u,v,w`,
`D_phi(u,v) - D_phi(u,w) = (grad phi(w) - grad phi(v))^T (u - v) - D_phi(v,w)`.
(Expand both sides; the `phi` terms cancel.) Apply it with `u = y`,
`v = f_A`, `w = f_{A u {j}} = f_A + d`, then take expectations. QED

**Reading.** For this whole family the identity is *exact*, with no smoothness
constant. The first term is the inner product between the *change of the loss
gradient* and the *current residual*; the second term `-D_phi(f_A, f_A + d) <= 0`
is a movement penalty measured in the Bregman geometry. The squared-loss
identity used by the main paper is the special case `phi(z)=||z||^2`,
`grad phi(z) = 2z`:

```
m(j|A) = E[ 2 d_{A,j} (y - f_A) - d_{A,j}^2 ] - lambda c_j .
```

If `phi` is `mu`-strongly convex and `L`-smooth then
`(mu/2)||d||^2 <= D_phi(f_A, f_A + d) <= (L/2)||d||^2`, which turns (4) into a
two-sided bound phrased entirely in the response magnitude.

## 4. Proposition 3 (cross-entropy corollary, explicit constants)

Let `f in R^C` be logits, `y` a one-hot target, and
`ell(f,y) = -f_y + log sum_k exp(f_k)` the softmax cross-entropy with
`p = softmax(f)`.

**Lemma (curvature of softmax cross-entropy).** The Hessian
`grad^2 ell = diag(p) - p p^T` is positive semidefinite with spectral norm at
most `1/2`.

*Proof.* `H 1 = 0`, so `v^T H v = Var_p(v)` for every `v`, where `Var_p` treats
`v_1..v_C` as the values of a random variable with probabilities `p_1..p_C`
(adding a constant to all coordinates leaves the quadratic form unchanged).
Let `M = max_k v_k`, `m = min_k v_k`, `mu_p = sum_k p_k v_k`. Bhatia–Davis
gives `Var_p(v) <= (M - mu_p)(mu_p - m) <= (M - m)^2 / 4`.
Replace `v` by `v - mean(v) 1` (which does not change `H v` and preserves the
range bound): the recentred vector is zero-sum, hence `M >= 0 >= m` and
`||v||^2 >= M^2 + m^2`. Therefore
`(M - m)^2 = (M + |m|)^2 <= 2(M^2 + m^2) <= 2||v||^2`, and
`v^T H v <= ||v||^2 / 2`. The bound is attained by `p = (1/2,1/2,0,...)` and
`v = (1,-1,0,...)/sqrt(2)`. QED

Applying Proposition 1 with `L = 1/2`, and using
`-grad ell(f_A,y) = e_y - p_A`, gives the label-explicit sandwich

```
<e_y - p_A, d_{A,j}> - (1/4) E||d_{A,j}||^2 - lambda c_j
    <= m(j|A) <= <e_y - p_A, d_{A,j}> - lambda c_j .          (5)
```

The upper bound holds because the second-order term is a penalty for any
positive semidefinite Hessian; the lower bound is the quantitative version of
"a large response can cost at most a quarter of its squared size".

## 5. Proposition 4 (why response features are the right ones)

For squared loss, insert the tower property into the exact identity:

```
E[ m~(j|A) | x, A ] = 2 d_{A,j}(x)^T ( mu(x) - f_A(x) ) - ||d_{A,j}(x)||^2 - lambda c_j ,
mu(x) = E[y | x, A] .
```

**Claim.** Conditioned on the state, the Bayes-optimal marginal-utility
predictor is *linear in the response vector* plus a state-independent quadratic
penalty. Writing `g_A(x) = 2(mu(x) - f_A(x))` for the state-dependent
coefficient,

```
E[m~(j|A) | x, A] = <g_A(x), d_{A,j}(x)> - ||d_{A,j}(x)||^2 - lambda c_j .   (6)
```

*Consequences.* (i) A model that is affine in the response, with coefficients
that depend on the state, is correctly specified for the squared-loss case —
this is exactly the parameterisation used by R-MUR (state features plus
response features in one head). (ii) Any score that ignores the response
(`d`-free) can only capture the state average and provably misses the
`<g_A, d>` interaction, which is the term the paper measures as structural
headroom. (iii) The curvature term `||d||^2` is a known, response-only
correction, which explains why response *magnitude* features carry utility
information even before the state is taken into account.

For a general differentiable loss the same argument holds with
`grad ell(f_A,y)` in place of `2(f_A - y)`; the residual is no longer a
sufficient statistic, so the state must also be observed, which is why the
router head receives both state and response features.

## 6. Scope and honest limits

* The bounds are *conditional on the state* and require the loss gradient at
  the current prediction. At routing time the label `y` is unobserved, so the
  router estimates the expectation of the first-order term from training
  episodes. The theory explains the feature design and the second-order error
  of a response-based surrogate; it is not a label-free guarantee.
* Proposition 1 is local: the constants hold on the segment between the two
  predictions actually visited, which is all the routing decision needs.
* Proposition 2 requires the loss to be a Bregman divergence in the
  prediction. Softmax cross-entropy with unconstrained logits is handled by
  Proposition 3 rather than by Proposition 2.
* The `O(||d||^2)` reading is a bound, not an equality: the true remainder can
  be much smaller, and it vanishes exactly when the loss is affine in
  `f` along the response direction.
* Nothing in this section asserts that a learned router attains the bound; the
  empirical sections report the realised recovery of oracle headroom.

## 7. Numerical validation (what is checked, and where)

| Check | Object | Script / artifact |
| --- | --- | --- |
| Exact squared-loss identity, machine precision | traffic nonlinear expert | `tests/test_neural_expert.py`, `results/derived/R102_response_theory/` |
| Sandwich (5) for cross-entropy, bound tightness | frozen in-context classifier | `results/derived/R102_response_theory/validation_ce.json` |
| Response sufficiency: R^2 of state-only vs state+response | both families | `results/derived/R102_response_theory/` |
| Second-order remainder scaling with `||d||^2` | traffic expert | `results/derived/R102_response_theory/` |
