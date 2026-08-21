# Distributional Inverse-Dynamics Regularization for Action-Ambiguous World Models

This project investigates whether a conditional action-density objective can serve as a useful representation regularizer for world models and JEPA-like predictive encoders when multiple actions are compatible with the same observed transition.

## Motivation

A deterministic continuous-action inverse-dynamics model trained with mean squared error predicts the conditional mean action. Under a multimodal inverse map, that mean can fall between valid modes and may produce a different transition. A distributional inverse-dynamics model instead learns

$$
q_\phi(a_t\mid z_t,z_{t+1}),
$$

allowing it to represent multiple compatible actions. The main research question is whether this more faithful inverse objective improves learned representations and downstream prediction, planning, or control—not merely action-density fit.

The project also distinguishes this proposal from broad latent-distribution regularizers: latent coverage objectives discourage collapse, while distributional inverse dynamics specifies action-related representation content. Hybrid objectives are part of the planned comparison.

[INTACT](https://arxiv.org/abs/2607.26056) is the closest verified precedent for end-to-end action-likelihood supervision. It trains one shared intent-to-action predictor on a local physical intent $z_{t+1}-z_t$ and a detached future-goal intent $\operatorname{sg}(z_g)-z_t$, then supports direct diagonal-Gaussian mean execution with optional local search. The two intent families need not be pointwise equal; their shared interpretation is defined through the conditional action law on supported conditions. INTACT therefore narrows novelty claims about probabilistic action regularization and shared local/goal prediction, but it does not test whether explicitly multimodal densities improve representations or avoid invalid between-mode actions under structural inverse ambiguity.

[PRISM](https://arxiv.org/abs/2606.07974) addresses a different stage: it trains a probabilistic action-sequence prior on frozen JEPA features and uses the prior to guide MPC sampling. This project will treat INTACT-style representation and direct-execution effects separately from PRISM-style planner-proposal effects.

Delta-JEPA trains a displacement-conditioned deterministic inverse objective inside a JEPA world model, and PLDM combines inverse dynamics with VICReg-style regularization. Deterministic IDM regularization is therefore established precedent; the open question is whether the density family matters for representations and downstream control.

## Current status

> **Planning/scaffold only; awaiting user review.**

No substantive experiments, datasets, dependencies, or training pipelines have been implemented. The INTACT, PRISM, Delta-JEPA, and LeJEPA/SIGReg descriptions are verified against primary sources. LeWorldModel and PLDM were identified as further adjacent work and require full-text review.

## Planned first step

Begin with falsification-first synthetic environments:

- quadratic ambiguity: $s_{t+1}=s_t+a_t^2$;
- redundant linear action maps with a nontrivial null space;
- controlled saturation, partial-observability, history, exogenous-state, and behavior-policy variants.

Simple discretized, fixed scalar-variance Gaussian, heteroscedastic Gaussian, and mixture-density models should first pass explicit transition-ambiguity gates and then demonstrate a transition-only representation benefit. Only after that gate will the plan add an INTACT-style shared local/goal Gaussian control, actor-sharing and gradient-routing ablations, support-overlap diagnostics, and actor-disabled planning before considering higher-capacity densities, larger environments, or PRISM-style downstream planner guidance.

The experiment plan defines a decisive core: environments E1 and E10, seven objective arms with frozen-head controls where meaningful, three seeds in bring-up extended to five before gate decisions, all governed by a pre-registered primary endpoint. All remaining phases are contingent branches. A null representation result is an acceptable terminal outcome with its own deliverable: the environment suite, the metric protocol, and the controlled analysis.

## Repository guide

- [`PROJECT_BRIEF.md`](PROJECT_BRIEF.md): self-contained research handoff and scope.
- [`docs/RESEARCH_QUESTIONS.md`](docs/RESEARCH_QUESTIONS.md): hypotheses, falsifiers, novelty boundary, and unresolved questions.
- [`docs/EXPERIMENT_PLAN.md`](docs/EXPERIMENT_PLAN.md): pre-registration protocol, decisive-core experiment, contingent branches, decision gates, and reporting requirements.
- [`docs/ARCHIVE.md`](docs/ARCHIVE.md): deferred environments, baselines, metrics, and phase designs, with verbatim text preserved at commit `8498e45`.
- `src/distributional_idm/`: placeholder package structure only.
- `tests/`, `configs/`, `scripts/`, `results/`: placeholders for future approved work.

## Non-goals at this stage

- claiming that stochastic inverse dynamics is itself novel;
- claiming that a distributional decoder makes a non-injective inverse map invertible;
- implementing major experiments before review;
- installing dependencies or downloading datasets;
- making unsupported literature claims.
