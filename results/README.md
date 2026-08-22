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

## Phase 1A - stability-treated rerun per amendment 6 (2026-08-21)

After the amendment 5 null, amendment 6 added gradient clipping (max norm 1) and best-of-three joint-training restarts selected by validation latent-planning success, uniformly across all arms; evaluation also sanitizes broken heads to bounded O(1) errors instead of astronomical ones. Full rerun at five seeds.

Standardized planning deltas versus no-IDM (positive = better):

| arm | e1_symmetric | e10_control |
|---|---|---|
| det_idm | +0.53 | +1.16 |
| gauss_idm | +1.29 | +2.59 |
| mdn_idm | +0.70 | +0.81 |
| aux_task | +0.68 | -0.06 |
| coverage | +0.19 | -0.99 |
| hybrid | -0.40 | -0.87 |

Findings:

1. The amendment 5 conclusion that jointly trained MDN arms destabilize was substantially an optimization artifact: with clipping and restart selection the MDN arm moved from -0.24 to +0.70 standardized planning on E1 and shows no divergence events.
2. Every action-supervised arm now trends positive on both environments. Heteroscedastic Gaussian is strongest throughout (E10 d=+2.59).
3. On the one-to-one control, the generic auxiliary control does nothing (-0.06) while every action-supervised arm improves - tentative evidence that action supervision carries something beyond generic or state-retention supervision. Not significant after correction; flagged for the five-seed-plus regime.
4. Coverage and hybrid still raise effective rank (~12) yet degrade planning on E10: geometric variance without controllable structure remains harmful.
5. Formal gate outcome under the amendment 6 protocol: still no arm advances - the only Holm-significant comparison is coverage's probe R2 on E10 (0.9993 versus 0.9998), the anticipated negligible-but-significant case with a negative direction.

The project stands at: directional support for action-prediction regularization (strongest for unimodal uncertainty), no statistically decisive separation at this scale, and two registered follow-ups (predictor-conditioned head; extended seeds or reduced test family) requiring amendments before running.

## Phase 1A - predictor-conditioned ablation per amendment 4 (2026-08-21)

Ran `mdn_pred`: the MDN action head conditions on `[f(s_t), P(f(s_t), a_t)]` - the forward predictor's own output - instead of the encoded actual next observation, so head gradients reach both encoder and predictor (teacher-forced evaluation on held-out data).

Result: **negative on planning in both environments** - e1_symmetric d=-1.04 (mean 0.178, one diverged seed), e10_control d=-1.64 (mean 0.342), against endpoint-conditioned mdn_idm's +0.70/+0.81. Training the action head in the planner's own input distribution does not close the train-inference gap; it makes representation quality worse. The plausible mechanism: the head's likelihood gradients now act directly on the predictor weights, fighting the world-model objective over the same parameters, whereas endpoint conditioning lets the two losses specialize (predictor learns next-state geometry, head learns action decoding from it).

Consequence for the train-inference-consistency argument: rejected in this form. If planner-proposal quality is the goal, PRISM-style post-hoc heads on frozen latents remain the supported route.

Updated arm ordering on latent-planning success: gauss_idm (+1.29/+2.59) > mdn_idm (+0.70/+0.81) > det_idm (+0.53/+1.16) >> mdn_pred (-1.04/-1.64). No comparison is Holm-significant at n=5; formal gate outcome unchanged.

## Phase 1A - five-seed gate evaluation pre-amendment-6 (2026-08-21): Gate 2 did not advance

Superseded numerically by the amendment 6 rerun above; retained as the registered history of that decision.

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
