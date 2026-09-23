# Submitted companion method manuscript — frozen

This directory contains the already submitted **IPM companion method paper**
(previously developed under the KBS working directory). Treat the submitted
manuscript and its numerical evidence as frozen.

The active new manuscript is `../TKDE_MUR/`.

## Separation from TKDE

The submitted companion establishes the basic R-MUR method under its
multi-target traffic protocol. The TKDE paper may share the problem notation and
the basic routing primitive, with explicit disclosure, but it must not simply
repeat the submitted paper's main numerical tables or engineering story.

TKDE instead owns the material developed after and outside that submitted
package:

- general response--utility theory;
- theory-shaped heads;
- identifiability and observability measurements;
- strong nonlinear backbones and predictor-strength/adaptation sweeps;
- candidate-library scaling and shortlist-removal tests;
- pool-geometry/novelty mechanism and label-free adaptation;
- the second task family;
- recent-selector extensions and alternative-policy audits;
- compute and deployment-pilot validation.

## Frozen evidence

- `paper/`: submitted companion manuscript source.
- `results/raw/`, `results/derived/`: evidence used by the submitted paper.
- `scripts/`: table builders and analyses for that frozen submission.

Do not fold new TKDE-only experiments back into this manuscript. The old
`paper_risk/` direction remains inactive.
