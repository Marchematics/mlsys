# Electricity experiment code review

## Review status

The secondary review found two blocking issues before real-data deployment:

1. Candidate weeks could be sampled after the query week, and the episode did
   not retain client/week metadata for a split audit.
2. Result provenance lacked cache hash, source-module hashes, deterministic run
   metadata, and split-overlap counts.

## Fixes applied

- Candidate contexts are now sampled strictly before the anchor/query week.
- Activity filtering uses only the training time window.
- Each batch retains target client, anchor week, candidate weeks, and query
  position.
- Every run writes split-audit counts, cache and source hashes, a unique run
  instance ID, deterministic settings, and all split windows.
- Base-only and pool-all references are included in the result JSON.
- A test now asserts that candidate weeks are historical and distinct from the
  anchor week.
- The pair sampler rejects invalid budget values before subset sampling.

## Non-blocking notes

- Ridge before/after utility is computed against the held-out load target.
- Empty-set packing is zero-state and does not read the target.
- The current Electricity smoke is a code/data sanity result. Its predictive
  metrics are inconclusive and do not pass Gate D.

## Verification

`python -m pytest -q` passes 13 tests after the fixes.

## Re-review

The blocking issues are resolved. The second review confirmed train-window
activity filtering, historical candidate weeks, split metadata, ridge label
correctness, empty-set packing, deterministic provenance, and base/pool
references. No blocking issue remains for the Electricity smoke deployment.
