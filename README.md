# Distributional Inverse-Dynamics Regularization for Action-Ambiguous World Models

This project investigates whether a conditional action-density objective can serve as a useful representation regularizer for world models and JEPA-like predictive encoders when multiple actions are compatible with the same observed transition.

## Motivation

A deterministic continuous-action inverse-dynamics model trained with mean squared error predicts the conditional mean action. Under a multimodal inverse map, that mean can fall between valid modes and may produce a different transition. A distributional inverse-dynamics model instead learns

$$
q_\phi(a_t\mid z_t,z_{t+1}),
$$

allowing it to represent multiple compatible actions. The main research question is whether this more faithful inverse objective improves learned representations and downstream prediction, planning, or control, not merely action-density fit.

The project also distinguishes this proposal from broad latent-distribution regularizers: latent coverage objectives discourage collapse, while distributional inverse dynamics specifies action-related representation content. Hybrid objectives are part of the planned comparison.

[INTACT](https://arxiv.org/abs/2607.26056) is the closest verified precedent for end-to-end action-likelihood supervision. It trains one shared intent-to-action predictor on a local physical intent $z_{t+1}-z_t$ and a detached future-goal intent $\operatorname{sg}(z_g)-z_t$, then supports direct diagonal-Gaussian mean execution with optional local search. The two intent families need not be pointwise equal; their shared interpretation is defined through the conditional action law on supported conditions. INTACT therefore narrows novelty claims about probabilistic action regularization and shared local/goal prediction, but it does not test whether explicitly multimodal densities improve representations or avoid invalid between-mode actions under structural inverse ambiguity.

[PRISM](https://arxiv.org/abs/2606.07974) addresses a different stage: it trains a probabilistic action-sequence prior on frozen JEPA features and uses the prior to guide MPC sampling. This project will treat INTACT-style representation and direct-execution effects separately from PRISM-style planner-proposal effects.

Delta-JEPA trains a displacement-conditioned deterministic inverse objective inside a JEPA world model, and PLDM combines inverse dynamics with VICReg-style regularization. Deterministic IDM regularization is therefore established precedent; the open question is whether the density family matters for representations and downstream control.

## Current status

Phase 0A implementation checks pass, and the Phase 1A experiment suite has been run on the registered synthetic environments. The main result is negative for the original multimodal anti-collapse hypothesis: mixture-density and spline-flow inverse heads fit action distributions but do not reliably improve planner-relevant representations. Heteroscedastic Gaussian NLL is the strongest action-supervised arm in the main rerun, but no arm passes the preregistered gate after correction.

Read [`docs/RESULTS_WRITEUP.md`](docs/RESULTS_WRITEUP.md) for the evidence, mechanism analysis, and limits. Read [`results/README.md`](results/README.md) for the chronological experiment record.

## Completed experiment program

The falsification-first suite tested:

- symmetric quadratic ambiguity: $s_{t+1}=s_t+a_t^2$;
- asymmetric action signs with probabilities 0.7 and 0.3;
- partial ambiguity, where one displacement bucket is deterministic and another remains bimodal;
- one-to-one nonnegative control.

Phase 0A checked deterministic, fixed-variance Gaussian, heteroscedastic Gaussian, and mixture-density heads. Phase 1A then compared end-to-end objectives, multimodal-head repairs, predictor-conditioned decoding, synthetic ambiguity variants, and NLL gradient reweighting.

The decisive core used environments E1 and E10, seven objective arms with frozen-head controls where meaningful, and five seeds for gate decisions. Later registered amendments tested multimodal-head repairs, asymmetric and partially ambiguous data, and NLL gradient reweighting. The controlled null result is the current terminal artifact for this experiment program.

## Repository guide

- [`docs/RESULTS_WRITEUP.md`](docs/RESULTS_WRITEUP.md): concise interpretation of the completed experiments and their limits.
- [`PROJECT_BRIEF.md`](PROJECT_BRIEF.md): self-contained research handoff and scope.
- [`docs/RESEARCH_QUESTIONS.md`](docs/RESEARCH_QUESTIONS.md): hypotheses, falsifiers, novelty boundary, and unresolved questions.
- [`docs/EXPERIMENT_PLAN.md`](docs/EXPERIMENT_PLAN.md): pre-registration protocol, decisive-core experiment, contingent branches, decision gates, and reporting requirements.
- [`docs/ARCHIVE.md`](docs/ARCHIVE.md): deferred environments, baselines, metrics, and phase designs, with verbatim text preserved at commit `8498e45`.
- `src/distributional_idm/`: environment, model, objective, training, and evaluation code.
- `tests/`: implementation and amendment regression tests.
- `scripts/`: experiment runners and analysis scripts.
- `results/`: preregistration, committed gate analysis, and a guide to generated outputs.

## Non-goals at this stage

- claiming that stochastic inverse dynamics is itself novel;
- claiming that a distributional decoder makes a non-injective inverse map invertible;
- implementing major experiments before review;
- installing dependencies or downloading datasets;
- making unsupported literature claims.
