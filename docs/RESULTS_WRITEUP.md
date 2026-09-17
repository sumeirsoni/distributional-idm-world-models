# Results writeup

## Bottom line

This study does not support multimodal inverse dynamics as a reliable anti-collapse mechanism for a learned world-model encoder.

The multimodal heads learned the conditional action distributions in the implementation checks. They still failed to produce better planner-relevant representations in end-to-end training. A flexible inverse head can explain the full action distribution on its own, so the encoder does not have to make its transition latents more informative than the head needs. That is the main negative result.

The result is specific to this training setup and these synthetic environments. It is not a theorem that every probabilistic inverse model is useless. Heteroscedastic Gaussian NLL was often the strongest action-supervised arm, but its benefit did not survive every change in the ambiguity structure, and no arm passed the preregistered gate.

## What was tested

The project compared a no-IDM world model with deterministic, heteroscedastic Gaussian, mixture-density, coverage, hybrid, and auxiliary-task objectives. The primary comparison used five seeds on a symmetric quadratic ambiguity environment and a one-to-one control environment. The gate required a positive effect of at least 0.2 and Holm-corrected significance across the preregistered endpoints.

Phase 0A passed its implementation checks. The deterministic head learned conditional means, the mixture-density head covered both valid modes, and sampled actions improved forward-cycle consistency. The negative representation result is therefore not explained by an inverse head that cannot fit the synthetic conditionals.

## Main results

The amendment 6 rerun used gradient clipping and best-of-three joint-training restarts for every arm. Standardized planning deltas versus the no-IDM arm were:

| Arm | E1 symmetric | E10 control |
| --- | ---: | ---: |
| Deterministic IDM | +0.53 | +1.16 |
| Gaussian IDM | +1.29 | +2.59 |
| MDN IDM | +0.70 | +0.81 |
| Auxiliary task | +0.68 | -0.06 |
| Coverage | +0.19 | -0.99 |
| Hybrid | -0.40 | -0.87 |

The Gaussian arm ranked first in this rerun, but no arm advanced through the formal gate after Holm correction. The probe metric also saturated near one on the scalar-state tasks, so planning and cycle metrics carried most of the useful signal.

The multimodal result was weaker than its density fit. The MDN improved after the stability treatment, but it remained behind the Gaussian arm. Replacing the MDN with a monotone-spline flow did not restore the missing representation benefit. Adding a sampled cycle loss did not restore it either. The predictor-conditioned MDN ablation was negative on planning in both environments.

## Why the original conclusion needs a qualification

The first version of the conclusion was too broad. On the symmetric E1 task, deterministic IDM, Gaussian NLL, and MDN NLL all improved planning relative to no IDM, with Gaussian NLL highest. The amendment 10 artifact tests showed why that result is not enough.

On the asymmetric E2 task, where conditional means carry action information, the planning effects were deterministic +1.05, Gaussian +1.23, sigma-only +0.82, and MDN +0.39. The deterministic and Gaussian arms were closer than they were on E1. This rejects the idea that the symmetric zero-mean setup alone proves a general advantage for uncertainty modeling.

On E1 mixed, the magnitude-1 transitions were deterministic while the magnitude-0.5 transitions stayed ambiguous. The Gaussian arm reached 0.998 planning success on the deterministic bucket but only 0.197 on the ambiguous bucket, below the no-IDM baseline of 0.488. The MDN showed the same direction at a smaller magnitude. A density head can fit an ambiguous conditional while the encoder loses competence exactly where ambiguity remains.

The training traces point to relative gradient volume as the mechanism. Plain Gaussian NLL gives low-variance, deterministic rows much larger gradients as the predicted variance shrinks. Those rows then dominate the shared encoder update, while ambiguous rows receive less influence. A beta-NLL variant that damped this per-row weighting restored ambiguous-bucket planning to 0.490 and kept deterministic-bucket planning at 0.868. Clamping the per-sample NLL made the ambiguous bucket worse at 0.132, so the evidence does not support the simpler claim that a few large spikes steal the clipping budget.

The safer conclusion is therefore:

> A high-capacity multimodal inverse head is not a reliable anti-collapse signal by itself. It can absorb action ambiguity in the decoder without forcing the encoder to preserve transition information that a planner needs. In this study, the useful signal came from how the objective weighted transition rows during joint optimization, not from the head's ability to represent a richer action distribution.

## What the experiments rule out

- Better multimodal density expressivity alone did not restore representation pressure. The spline-flow and MDN repair arms were negative or near baseline.
- A cycle loss on sampled actions did not solve the problem. The flow-plus-cycle arm was also unstable at this scale.
- Training the inverse head on the predictor's own output did not close the train-inference gap. The predictor-conditioned MDN made planning worse.
- Generic latent coverage is not a substitute for controllable structure. Coverage increased effective rank but degraded planning on E10.
- The uncertainty channel alone did not explain the Gaussian result. The sigma-only arm was weak on E1, and the mean-only arm is the deterministic baseline.

## What remains open

The study identifies a useful failure mode, not a final recipe. The next test should use environments that combine informative conditional means with partial ambiguity and should compare row-balanced objectives under the same restart and planning-selection protocol. Beta-NLL is a promising candidate because it changes the relative gradient weighting directly, but its representation benefit still needs a preregistered comparison.

The current evidence is limited to small synthetic state-vector environments, five-seed comparisons, and a few single-init training traces. The result should not be generalized to pixels, larger world models, or offline planner proposals without new experiments.

## Reproduction and source data

The chronological record, amendment decisions, exact metrics, and generated-result paths are in [`results/README.md`](../results/README.md). The committed gate analysis is [`results/phase1a/gate_analysis.json`](../results/phase1a/gate_analysis.json). Large generated result bundles and checkpoints remain ignored by design.

Run the unit tests and linter with:

```sh
.venv/bin/python -m pytest -q
.venv/bin/ruff check src scripts tests
```

Regenerate the phase analyses with the commands documented in `results/README.md` after producing the corresponding result bundle.
