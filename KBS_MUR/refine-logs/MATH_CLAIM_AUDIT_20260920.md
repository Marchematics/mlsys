# Mathematical claim audit (2026-09-20)

## One-step routing regret
- Statement: with uniform utility error `epsilon` at state `A`, the true
  marginal gap between the predicted argmax and the true argmax is at most
  `2 epsilon`.
- Status: correct for the stated fixed-state assumption.
- Boundary: local statement; no terminal-set control without further structure.

## Conditional set-level guarantee
- Statement: if `F` is normalized, monotone, and submodular, and utility error is
  at most `epsilon` at every state visited by the approximate greedy policy,
  then `F(A_MUR) >= (1 - (1 - 1/B)^B) F(A*) - 2 B epsilon`.
- Status: proof added to `paper/supplement.tex`.
- Boundary: the result is conditional on monotone submodularity and the uniform
  per-state error bound. The main routing problem permits negative and
  non-submodular marginals.

## Calibration wording
- The interval uses a split-conformal symmetric residual quantile.
- The finite-sample correction is `ceil((n+1)(1-alpha))/n` with the higher
  quantile convention.
- Status: the text states marginal-calibration language and avoids claims of
  simultaneous coverage over adaptive selection trajectories.
- Boundary: coverage requires compatibility between calibration and deployment
  state distributions.

## Evaluation quantity
- `H_state = (G_OG - G_OS) / G_OG`.
- Status: an oracle evaluation quantity; not part of the router objective and
  not an assumption of the method.
