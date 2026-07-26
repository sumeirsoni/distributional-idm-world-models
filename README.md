# Distributional Inverse-Dynamics Regularization for Action-Ambiguous World Models

This project investigates whether a conditional action-density objective can serve as a useful representation regularizer for world models and JEPA-like predictive encoders when multiple actions are compatible with the same observed transition.

## Motivation

A deterministic continuous-action inverse-dynamics model trained with mean squared error predicts the conditional mean action. Under a multimodal inverse map, that mean can fall between valid modes and may produce a different transition. A distributional inverse-dynamics model instead learns

$$
q_\phi(a_t\mid z_t,z_{t+1}),
$$

allowing it to represent multiple compatible actions. The main research question is whether this more faithful inverse objective improves learned representations and downstream prediction, planning, or control—not merely action-density fit.

The project also distinguishes this proposal from broad latent-distribution regularizers: latent coverage objectives discourage collapse, while distributional inverse dynamics specifies action-related representation content. Hybrid objectives are part of the planned comparison.

## Current status

> **Planning/scaffold only; awaiting user review.**

No substantive experiments, datasets, dependencies, or training pipelines have been implemented. Named-method and paper-specific claims remain subject to primary-source verification.

## Planned first step

Begin with falsification-first synthetic environments:

- quadratic ambiguity: $s_{t+1}=s_t+a_t^2$;
- redundant linear action maps with a nontrivial null space;
- controlled saturation, partial-observability, history, exogenous-state, and behavior-policy variants.

Simple discretized and mixture-density models should pass explicit decision gates before adding flows, diffusion models, energy-based models, or larger environments.

## Repository guide

- [`PROJECT_BRIEF.md`](PROJECT_BRIEF.md): self-contained research handoff and scope.
- [`docs/RESEARCH_QUESTIONS.md`](docs/RESEARCH_QUESTIONS.md): hypotheses, falsifiers, novelty boundary, and unresolved questions.
- [`docs/EXPERIMENT_PLAN.md`](docs/EXPERIMENT_PLAN.md): phased experiment matrix, metrics, ablations, decision gates, and expected failure signatures.
- `src/distributional_idm/`: placeholder package structure only.
- `tests/`, `configs/`, `scripts/`, `results/`: placeholders for future approved work.

## Non-goals at this stage

- claiming that stochastic inverse dynamics is itself novel;
- claiming that a distributional decoder makes a non-injective inverse map invertible;
- implementing major experiments before review;
- installing dependencies or downloading datasets;
- making unsupported literature claims.
