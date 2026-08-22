# Results

Generated experiment outputs belong here and are ignored by Git except for this file and `preregistration.md`.

Future result bundles should include the exact configuration, seed, environment/policy specification, metrics, uncertainty summaries, and decision-gate outcome. Do not commit large checkpoints or raw generated datasets by default.

## Phase 0A - implementation check (2026-08-21): PASS

Full numbers in `phase0a/results.json` (regenerate with `.venv/bin/python scripts/run_phase0a.py`; 3 bring-up seeds, test-split metrics).

| Dataset | Deterministic: |pred mean|| cycle err | MDN NLL | coverage | precision | invalid mass | sample cycle err |
|---|---|---|---|---|---|---|---|
| e1_symmetric | 0.007 | 0.624 | -2.761 | 1.000 | 0.887 | 0.114 | 0.038 |
| e1_asymmetric | 0.304 | 0.522 | -2.842 | 0.991 | 0.887 | 0.114 | 0.038 |
| e1_single_magnitude | 0.019 | 1.000 | -2.761 | 1.000 | 0.887 | 0.114 | 0.050 |
| e1_noisy | 0.015 | 0.624 | -2.757 | 0.999 | 0.886 | 0.115 | 0.038 |
| e10_control | 0.750 | 0.002 | -3.454 | 1.000 | 0.887 | 0.114 | 0.038 |

Pass criteria all hold:

- Deterministic MSE lands on conditional means: predictions near zero under symmetric +-a, near the weighted mean at 0.7/0.3 asymmetry, and its known-forward cycle error equals E[a^2] exactly, as theory predicts.
- The MDN represents both valid modes on every dataset (coverage >= 0.99) and sampled actions beat point predictions roughly 16x on forward cycle consistency.
- E10 no-regression: deterministic error 0.0012; density heads stay sharp.

Documented observations:

- MDN training has three discrete basins (both modes / one magnitude bucket / off-mode drift); entry is init- and learning-rate dependent. Five restarts with validation-NLL selection reliably reach the correct basin and are part of the runner.
- Heteroscedastic Gaussian places ~93 percent of mass between modes on ambiguous datasets, matching the variance-inflation failure signature in the experiment plan.

Implementation notes:

- Conditioning for transition-conditioned heads is state-plus-displacement; raw endpoint pairs bury the displacement signal after standardization because states drift ~125 units per episode.
- Mode metrics take row-wise mode vectors and normalize shapes internally; earlier cross-row broadcasting produced silently wrong values near 0.5 and was fixed with an oracle-mixture check.

## Phase 1A - decisive-core bring-up (2026-08-21): 3 seeds, no gate decision

Numbers in `phase1a/results.json` (regenerate with `.venv/bin/python scripts/run_phase1a.py`). Gate decisions wait for the five-seed extension per the staged seed policy; these are bring-up observations.

Endpoint components: probe R2, latent-planning success, and post-hoc cycle error from an identically trained MDN head on each arm's frozen latents (prereg amendment 2).

Bring-up findings:

1. Deterministic IDM stabilizes latent planning on E1: planning deltas +0.001/+0.497/+0.893 across seeds versus an unstable no-IDM arm ({1.00/0.50/0.11}). Direction consistent in all three seeds; t=+1.80 at n=3.
2. The generic auxiliary-task control does not reproduce that gain (deltas -0.36/+0.47/+0.89), so the deterministic-IDM effect is action-content-specific rather than generic supervision.
3. Jointly trained MDN heads destabilize E1 planning (all three seeds negative, mean delta -0.36) and one E10 seed diverged entirely. This mirrors the Phase 0A basin behavior and currently argues against the project's headline hypothesis at this training recipe.
4. Coverage regularization does its stated job - effective rank rises from ~1.0 to ~9.2 while every other arm sits near rank 1 - but planning degrades on both environments: variance without controllable structure.
5. Probe R2 saturates at 1.000 for all arms because the controllable state is one-dimensional and linearly decodable even from rank-1 latents; the probe component is non-discriminative on this environment family and its standardized values are unstable when baseline SD approaches zero.
6. Possible H8 no-regression violations on E10 (gauss, mdn, coverage planning deltas consistently negative) require the five-seed extension before any claim.

Next: five-seed extension for gate evaluation; MDN-arm stability investigation (restart selection inside backbone training) if the extension confirms finding 3.
