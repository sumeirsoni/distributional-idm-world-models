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

## Phase 1A - decisive core, five-seed gate evaluation (2026-08-21): Gate 2 does not advance

Analysis in `phase1a/gate_analysis.json` (regenerate with `.venv/bin/python scripts/analyze_phase1a.py`): 48 paired tests (six arms x four endpoints x two datasets), Holm-corrected, advance rule = corrected p < 0.05 AND standardized effect >= +0.2.

Outcome: **no arm advances**. No comparison clears both bars; none is significant after correction at n=5.

Key numbers:

- e1_symmetric planning: det_idm +0.33 (d=+0.86, p_holm=1.00), aux_task +0.40 (d=+1.05, p_holm=1.00), mdn_idm -0.24 (d=-0.63).
- e10_control planning: gauss_idm -0.42 (d=-1.23, p_holm=1.00), coverage -0.42 (d=-1.23), hybrid -0.38 (d=-1.11); aux_task +0.31 (d=+0.92).

Interpretation under the pre-registered rules:

1. The bring-up impression that deterministic-IDM gains are action-content-specific did not survive: at five seeds the random-feature auxiliary control improves E1 planning by a similar amount (d +1.05 vs +0.86), and neither is significant. The stabilization signal, if real, is attributable to added supervision rather than action content specifically.
2. The headline hypothesis - explicitly multimodal inverse density shaping representations better than unimodal controls - is unsupported at this scale and recipe. Jointly trained MDN heads trend negative on E1 planning and show instability (one diverged encoder seed).
3. Distributional and coverage arms show consistent, non-significant planning degradations on the one-to-one control environment (E10), a candidate H8 no-regression violation worth monitoring rather than claiming.
4. Probe R2 remains saturated at ceiling for all arms (scalar state), contributing no discriminative power.

Per the project brief's acceptable-outcomes clause, a controlled null result is a valid terminal artifact: the environment suite, metric protocol, oracle-validated metrics, and this analysis constitute the deliverable unless follow-ups justify continuing.

Registered follow-ups that could change the picture (each requires its own dated amendment before running):

- Predictor-conditioned inverse-head ablation (amendment 4): tests whether training the head on the planner's own input distribution closes the train-inference gap.
- MDN-arm stability treatment (restart selection inside backbone training) before re-comparing multimodal arms.
