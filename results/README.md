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

## Phase 1A - amendment 8 sigma-channel decomposition (2026-08-21): statistic-channel view rejected

Ran `sigma_only` (Gaussian NLL with mean pinned to zero; only the predicted standard deviation trains) against the existing arms, five seeds.

E1 planning deltas: gauss_idm +1.29, mdn_idm +0.70, det_idm +0.53, **sigma_only -0.06**. Paired gauss-minus-sigma differences positive in all five seeds (+0.44/+0.29/+0.48/+0.26/+0.31).

The pre-registered prediction - that the uncertainty channel alone reproduces most of the Gaussian arm's benefit - is rejected. Combined with mu-only (= det_idm) also being weak, neither channel alone suffices: **the regularization benefit lives in the joint Gaussian NLL**, and notably the two losses converge to nearly identical optima on E1 once the free mean approaches zero, so the divergence must be dynamical (how coupled two-channel optimization shapes gradients during training) rather than a property of either target statistic. The mechanism remains genuinely open after eight amendments; what is now firmly established is the decomposition result itself.

Current full ordering on latent planning (both environments): gauss_idm > mdn_idm > det_idm > no_idm, with every multimodal/alternative construction tried (flow, cycle, predictor-conditioned, imagination-space) at or below baseline.

## Phase 1A - amendment 10 synthetic-structure artifact tests (2026-08-21)

Prompted by the concern that E1's zero-mean, permanently-symmetric ambiguity drove the head-family conclusions. Ran {no_idm, det_idm, gauss_idm, sigma_only, mdn_idm} on two new datasets, five seeds under amendment 6 protocol.

**e2_asymmetric** (sign probabilities 0.7/0.3; conditional means now carry information):

| arm | planning d vs no-IDM |
|---|---|
| det_idm | +1.05 |
| gauss_idm | +1.23 |
| sigma_only | +0.82 |
| mdn_idm | +0.39 |

- H-sym CONFIRMED: det_idm's weakness was substantially the zero-mean artifact (+0.53 on E1 versus +1.05 here).
- H-pin CONFIRMED: sigma_only flips from -0.06 to +0.82 once a pinned-zero mean is wrong rather than optimal.
- The Gaussian's advantage over deterministic shrinks from a wide gap to +1.23 versus +1.05: with live means, head families converge.

**e1_mixed** (|a|=1 transitions deterministic, |a|=0.5 transitions bimodal), stratified by displacement bucket:

| arm | ambiguous bucket (Δs=0.25) | deterministic bucket (Δs=1.00) | overall d |
|---|---|---|---|
| no_idm | 0.488 | 0.846 | +0.00 |
| det_idm | 0.629 | 0.909 | +0.44 |
| gauss_idm | **0.197** | **0.998** | **-0.30** |
| mdn_idm | 0.295 | 0.858 | -0.39 |
| sigma_only | 0.296 | 0.819 | -0.47 |

- H-perm REJECTED with an inversion: action-arm gains concentrate on the DETERMINISTIC bucket, not the ambiguous one. Most strikingly, gauss_idm drives ambiguous-bucket planning below the untrained baseline (0.197 versus 0.488) while perfecting the deterministic bucket (0.998).

Interpretation: under partial ambiguity, heteroscedastic NLL training trades ambiguous-region competence for deterministic-region precision - the opposite of what uncertainty-aware framing predicts. Combined with amendment 10(a), the study's earlier headline ordering is confirmed to be an artifact of permanent symmetric ambiguity: real-regime conclusions require environments where informative means and partial ambiguity coexist, which none of E1/E2/E10 provide alone. The e1_mixed generator and stratified metrics are retained as the template for that next environment generation.

## Phase 1A - e1_mixed training-dynamics instrumentation (2026-08-21)

Traces in `phase1a/e1_mixed_loss_traces.json` (regenerate with `.venv/bin/python scripts/instrument_mixed.py`): per-epoch validation NLL split by displacement bucket plus ambiguous-bucket planning success, three arms x three seeds, single runs (no restart selection).

Pattern classification against the three pre-stated hypotheses:

1. **gauss: interference confirmed (pattern 1).** Ambiguous-bucket NLL improves early then degrades in two of three seeds (seed1: 0.18 -> 0.69; seed2: -0.04 -> 13.8, a captured live divergence) exactly while deterministic-bucket NLL plummets (-0.2 -> -3.3, sigma approaching the floor). Deterministic rows offer unbounded likelihood gains; ambiguous rows offer bounded ones; the shared encoder reallocates accordingly. Active forgetting, not inability.
2. **det: saturates at its floor (pattern 2, benign).** Ambiguous-bucket MSE converges to 0.25-0.30 - precisely the irreducible variance of a constant-mean predictor on +-0.5 modes - with no degradation. The head cannot express better, but nothing gets worse.
3. **mdn: no interference (contradicts pattern expectations).** Ambiguous NLL improves monotonically to the best levels of any arm (-0.27) with zero degradation. Yet its ambiguous-bucket planning still ends below the no-IDM baseline (0.30 versus 0.49), so for MDN a residual slice of pattern 3 remains: density fit improves faster than planner-relevant geometry.

Cross-cutting finding (corrected): the no-IDM baseline outperforms every NLL-family action-supervised arm on ambiguous-bucket planning - gauss (0.197), sigma (0.296), mdn (0.295) all sit below 0.488 - while det_idm, whose MSE gradients weight all rows equally, improves both buckets (+0.14 ambiguous). Earlier phrasing of this finding overclaimed by including det_idm and by comparing restart-selected runner numbers against single-run instrument trajectories; the apples-to-apples statement is about the NLL family's confidence weighting. Mechanism hypothesis: Gaussian NLL scales each row's gradient by 1/sigma^2, amplifying confident-region rows and suppressing uncertain-region rows, so shared geometry drifts to serve deterministic rows; MDN softens this via component responsibilities (intermediate harm); MSE has no such asymmetry (no harm). Falsifiable prediction: detaching sigma in the weighting term of the gaussian loss should remove the per-row amplification and eliminate the ambiguous-bucket degradation without hurting the deterministic bucket.

This resolves the mechanism question raised after amendment 10 with direct evidence: gradient-volume imbalance between low-entropy and high-entropy rows drives the competence reallocation. It also sharpens the flow-matching prediction made earlier: because FM regression targets are drawn per row rather than concentrated on consistent rows, it should exhibit less bucket imbalance than NLL heads - now a falsifiable prediction rather than speculation.

## Phase 1A - amendment 11 spike-budget test (2026-08-21): damping confirmed, budget-theft refuted

Two joint-training variants on e1_mixed/e10_control, five seeds: gauss_bnnl (beta-NLL, detached sg(sigma)^0.5 multiplier, PRISM-faithful form) and gauss_clampnll (per-sample NLL clamped at 10 nats).

Ambiguous-bucket planning on e1_mixed: gauss_bnnl 0.490 versus gauss_idm 0.197 - beta-NLL fully eliminates the ambiguous-bucket degradation while retaining most deterministic gains (0.868 versus 0.998). Clamping made it worse (0.132), refuting the budget-stealing variant of the hypothesis.

Refined conclusion: the operative channel is RELATIVE GRADIENT VOLUME between bucket types. Plain NLL lets deterministic-row gradients dominate roughly sigma_amb^2/sigma_det^2-fold once sigma collapses; beta-NLL multiplies each row by detached sqrt(sigma), compressing exactly that asymmetry. This also explains why PRISM can train beta-NLL safely: the reweighting is the point, not incidental. Data incident during this amendment: filtered runs had been overwriting results.json (sigma_only rows, e2_asymmetric, and original e1_mixed baselines temporarily lost); all runs are deterministically reproducible and were regenerated, and the runner now merges instead of overwrites under filters.

Caveat: traces are single-init runs; restart-selected checkpoints could differ, though the interference signature appeared in all three gauss seeds independently.

## Phase 1A - amendment 7 multimodal-repair factorial (2026-08-21): all three hypotheses rejected

Ran `flow_idm` (24-knot monotone-spline flow head, exact NLL), `mdn_cycle` (MDN plus reparameterized-sample cycle loss through the forward predictor, weight 1.0), and `flow_cycle` (both), five seeds each under the amendment 6 protocol.

Planning deltas versus no-IDM:

| arm | e1_symmetric | e10_control |
|---|---|---|
| gauss_idm (reference best) | +1.29 | +2.59 |
| flow_idm | +0.05 | -0.43 |
| mdn_cycle | +0.27 | -0.00 |
| flow_cycle | +0.17 | -0.51 |

Findings:

1. H-A rejected: replacing the MDN with a degeneracy-free spline flow does not restore regularization pressure; flow_idm matches no-IDM within noise.
2. H-B rejected: the cycle term alone yields marginal movement (+0.27/-0.00).
3. H-C rejected: flow_cycle is worse than flow_idm on E10 and produced multiple fully diverged encoders (sanitized probes of exactly 0); Newton-carried sampling gradients through a jointly trained predictor are unstable at this scale.
4. The transmission-line theory stated earlier in this file - "the gap exists; only optimization degeneracy blocks it" - is itself falsified by (1): with degeneracies removed the gap still fails to transmit.
5. Refined mechanism hypothesis going forward: what matters is not distributional expressivity but whether the head's loss *requires computing an informative statistic of the transition* - the heteroscedastic variance channel demands displacement information even when conditional means saturate, while shape-fitting channels are self-absorbing regardless of parameterization. Call it the statistic-channel view, registered here as the replacement for the absorption/pressure framing.

Cost note: flow_cycle averaged ~900 s per seed-dataset (40-step bisection sampling inside every update); the arm is also the least stable, so it is not recommended for scaling.

Standing conclusion after amendments 4 through 7: heteroscedastic Gaussian NLL is the strongest representation regularizer among eleven tested action-head variants; multimodal density quality and regularization value remain anti-correlated in every construction tried.

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
