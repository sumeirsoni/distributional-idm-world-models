# Multimodal inverse dynamics does not reliably prevent latent collapse

This repository tests whether a conditional action-density loss can make a world-model encoder preserve more useful transition information when several actions produce the same observed transition.

## Result

In these experiments, a multimodal inverse-dynamics head is not a reliable anti-collapse mechanism.

An MDN or a flow head can fit the full conditional action distribution without forcing the encoder to make its latents more informative for planning. The decoder absorbs the ambiguity. That is the central negative result.

The result is specific to this training setup and these synthetic environments. It does not show that every probabilistic inverse model fails. Heteroscedastic Gaussian NLL was often the strongest action-supervised objective, but no arm passed the preregistered gate.

## What we ran

The primary comparison used five seeds on two environments:

- E1 symmetric: $s_{t+1}=s_t+a_t^2$, with both signs valid for each nonzero displacement.
- E10 control: a one-to-one nonnegative action map.

The study compared no IDM, deterministic IDM, Gaussian IDM, MDN IDM, generic auxiliary supervision, latent-coverage regularization, and a hybrid objective. Follow-up amendments tested spline flows, sampled cycle losses, predictor-conditioned decoding, asymmetric action probabilities, partial ambiguity, and beta-NLL reweighting.

Phase 0A passed its implementation checks. The deterministic head learned conditional means. The MDN covered both valid modes, and sampled actions improved forward-cycle consistency. The later negative representation result is not explained by a head that cannot fit the action distribution.

## Main numbers

The stability-treated rerun used gradient clipping and three joint-training restarts per seed. Values below are standardized planning deltas versus the no-IDM arm.

| Arm | E1 symmetric | E10 control |
| --- | ---: | ---: |
| Deterministic IDM | +0.53 | +1.16 |
| Gaussian IDM | +1.29 | +2.59 |
| MDN IDM | +0.70 | +0.81 |
| Auxiliary task | +0.68 | -0.06 |
| Coverage | +0.19 | -0.99 |
| Hybrid | -0.40 | -0.87 |

Gaussian NLL ranked first in this rerun. The formal gate still rejected every arm after Holm correction. The probe metric was near ceiling on these scalar-state tasks, so planning and cycle metrics were more informative.

## What happened under partial ambiguity

The symmetric task hides an important failure mode. In E1 mixed, the magnitude-1 transitions were deterministic while the magnitude-0.5 transitions stayed ambiguous.

The Gaussian arm reached 0.998 planning success on the deterministic bucket but only 0.197 on the ambiguous bucket. The no-IDM baseline reached 0.488 on that bucket. The MDN showed the same pattern at a smaller magnitude.

Training traces point to relative gradient volume. As Gaussian NLL drives the predicted variance down on deterministic rows, those rows produce much larger gradients than ambiguous rows. The shared encoder then favors the easy, low-variance part of the data.

A beta-NLL variant reduced that weighting gap. It raised ambiguous-bucket planning to 0.490 while keeping deterministic-bucket planning at 0.868. Clamping each per-sample NLL made the ambiguous bucket worse at 0.132. The evidence therefore favors a gradient-allocation explanation over the simpler claim that a few large loss spikes steal the clipping budget.

## What did not fix the problem

- Replacing the MDN with a monotone-spline flow did not restore representation pressure.
- Adding a sampled cycle loss did not restore it and made the flow variant unstable.
- Conditioning the MDN on the predictor's output made planning worse.
- Latent-coverage regularization increased effective rank but degraded planning on E10.
- Training only the Gaussian uncertainty channel did not reproduce the full Gaussian result.

## Interpretation

The useful signal is not the ability to represent a richer action distribution by itself. A high-capacity decoder can model multimodal actions while leaving the encoder free to discard transition details that matter to the planner.

The stronger design target is a loss whose encoder gradients stay useful on ambiguous transitions. Beta-NLL is a candidate because it changes per-row gradient weighting directly. It needs a new preregistered comparison before we treat it as a solution.

## Limits

The evidence comes from small synthetic state-vector environments, five-seed comparisons, and a few single-init training traces. It does not establish the same result for pixel observations, larger world models, or offline planner proposals.

## Repository map

- [`docs/RESULTS_WRITEUP.md`](docs/RESULTS_WRITEUP.md) has the longer interpretation and caveats.
- [`results/README.md`](results/README.md) has the chronological experiment record.
- [`docs/EXPERIMENT_PLAN.md`](docs/EXPERIMENT_PLAN.md) has the preregistration and decision rules.
- [`src/distributional_idm/`](src/distributional_idm/) contains the environments, models, losses, training code, and evaluation code.
- [`tests/`](tests/) contains implementation and amendment regression tests.

Run the checks with:

```sh
.venv/bin/python -m pytest -q
.venv/bin/ruff check src scripts tests
```
