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
