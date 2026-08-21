# Research Questions

## Framing

The proposed contribution is not stochastic action prediction or end-to-end action NLL in isolation. [INTACT](https://arxiv.org/abs/2607.26056) already trains a JEPA-style representation with paired physical-transition and future-goal action likelihood through a shared predictor. The incremental question is whether **explicitly multimodal inverse-density regularization** improves representation content and valid action-mode coverage beyond deterministic and unimodal Gaussian controls under structural action ambiguity.

The INTACT, PRISM, Delta-JEPA, and LeJEPA/SIGReg descriptions are verified against primary sources as of August 2026. Two further adjacent works require full-text review before comparative claims: [LeWorldModel](https://arxiv.org/abs/2603.19312), which applies SIGReg-style regularization end-to-end in a JEPA world model, and [PLDM](https://arxiv.org/abs/2502.14819), which combines predictive learning, VICReg-style regularization, and inverse dynamics. The hypotheses below do not depend on those two attributions.

## Primary research question

When the inverse map from an observed transition to an action is multimodal or non-identifiable, does an explicitly multimodal conditional action-density objective produce more useful representations and more valid actions than deterministic MSE and an INTACT-style unimodal Gaussian, after separating actor-sharing, support, execution, and planner effects?

## Core hypotheses and falsifiers

### H1 — Deterministic MSE produces invalid averages under action ambiguity

For multimodal conditionals $p(a_t\mid z_t,z_{t+1})$, a deterministic MSE decoder converges toward the conditional mean, which can lie outside the valid action modes.

**Predictions**

- In $s_{t+1}=s_t+a_t^2$ with symmetric $\pm a$, MSE predicts near zero.
- Mean predictions have poor forward cycle consistency.
- MSE cannot distinguish conditionals with identical means but different modal structure.

**Falsifiers**

- The controlled datasets do not produce the expected conditional-mean solution after optimization and capacity checks.
- The mean action is not measurably worse than modal predictions on transition consistency.

### H2 — Simple conditional-density models represent ambiguity better

A discretized categorical model or mixture density network captures multiple valid actions more faithfully than deterministic regression.

**Predictions**

- Better held-out conditional scoring and calibration.
- Higher valid-mode coverage and precision.
- Samples produce transitions closer to the target than the MSE mean.

**Falsifiers**

- Simple density models consistently collapse to one mode or fail calibration despite controlled optimization.
- Improved likelihood does not correspond to valid action samples or cycle consistency.

### H3 - Expressive density support adds representation value beyond unimodal action NLL

When the inverse map has separated modes, an explicitly multimodal density should retain more useful controllable structure than deterministic IDM or an INTACT-style diagonal Gaussian. When compatible actions instead lie on a connected lower-dimensional set, the relevant extension is support-aware or manifold-capable density modeling rather than multimodality by itself. Both comparisons must also include goal-only behavior cloning and a frozen probabilistic head.

**Predictions**

- Better controllable-state probes, forward prediction, or downstream control under known ambiguity.
- Better mode precision, mode coverage, calibration, and known-forward validity than a unimodal Gaussian that inflates variance over invalid actions.
- Some gains remain when the learned actor is disabled and the representation is evaluated with probes or vanilla planning.
- Gains are strongest on structurally ambiguous environments and shrink on effectively one-to-one controls.
- Improvements survive capacity-matched decoder and actor-sharing comparisons.
- On environments where the encoder can resolve aliasing, deterministic IDM matches or beats distributional IDM on controllable-state probes. This is the pressure-strength signature defined in the project brief.
- On structurally ambiguous environments, deterministic IDM encodes more behavior-policy cues, measured by shortcut probes, and retains less exogenous information than distributional IDM. This is the calibrated-acceptance signature.

**Falsifiers**

- Only density likelihood improves; encoder probes, actor-disabled planning, and downstream outcomes do not.
- Gains disappear when decoder capacity, parameter count, and local/goal actor topology are controlled.
- An INTACT-style Gaussian matches the multimodal model on ambiguity-specific validity and representation metrics.
- Exogenous task-relevant information degrades enough to offset controllable-state gains.
- Neither mechanism signature appears in any environment pair, which means decoder family is representationally irrelevant at the tested scale regardless of density-fit gains.

### H4 — Broad latent coverage and action-conditioned density are complementary

A broad anti-collapse or latent-coverage objective and a distributional IDM address different failure modes.

**Predictions**

- Coverage-only regularization maintains latent variance/rank but can encode nuisance factors.
- Distributional IDM emphasizes controllable factors but can underrepresent exogenous factors.
- A hybrid can improve both collapse metrics and task-relevant probes when weighted appropriately.

**Falsifiers**

- The two objectives are empirically redundant across controlled factors and downstream tasks.
- The hybrid introduces optimization conflict without a measurable representation benefit.

### H5 — Transition-conditioned likelihood gain detects behavior-policy shortcuts

The held-out score

$$
\log q(a_t\mid z_t,z_{t+1})-\log\pi(a_t\mid z_t)
$$

measures whether transition information improves action prediction beyond state-only policy correlations, provided both models are properly and separately fitted.

**Predictions**

- Gains shrink when the next representation is permuted or removed.
- Gains are larger when transitions contain action information not predictable from state alone.
- State-only performance changes with behavior-policy correlation, while genuine transition contribution is more stable.

**Falsifiers**

- The gain persists after transition permutation.
- The result is explained by unequal model capacity or a deliberately degraded baseline.
- The score is unstable across held-out splits or policy shifts.

### H6 - A carefully designed CMI-inspired training variant reduces shortcuts

Folded into H5 as a contingent Phase 2 design. The evaluation-only score from H5 comes first; any training variant (frozen baseline, alternating optimization, or a derived variational bound) activates only if the diagnostic reliably detects permutation and policy shortcuts. Its predictions and falsifiers are archived.

### H7 — History resolves only observational, not structural, ambiguity

For partial observability where past observations identify the hidden state, a history-conditioned IDM should reduce uncertainty. For structurally many-to-one action effects, history should not recover the realized action.

**Predictions**

- Conditional entropy or multimodality decreases with history in resolvable POMDP variants.
- It remains in quadratic symmetry or action-null-space variants where action identity is absent from the full observed history.

**Falsifiers**

- History appears to resolve structural non-identifiability, suggesting data leakage or an unintended policy cue.

### H8 — Distributional modeling should not regress on one-to-one tasks

On tasks with an effectively deterministic inverse map, a simple distributional model should match deterministic IDM performance within a pre-registered equivalence margin on held-out likelihood, representation probes, and downstream metrics, at matched compute and tuning budgets.

**Falsifiers**

- Persistent degradation in likelihood, representation quality, or downstream performance on ordinary one-to-one dynamics.

### H9 - Representation gains and planner-proposal gains are separable

[PRISM](https://arxiv.org/abs/2606.07974) shows that a learned state-and-goal-conditioned Gaussian action-sequence prior can improve MPPI sampling while the JEPA encoder and world model remain frozen. The proposed project should therefore distinguish improvements caused by representation regularization from improvements caused by a better inference-time action proposal.

**Predictions**

- A PRISM-style proposal can improve planning for more than one representation objective.
- If distributional-IDM training improves the representation, some gain should remain under vanilla planning without a learned proposal.
- Crossing representation objectives with vanilla and learned proposals will reveal whether the effects are additive, redundant, or interactive.
- Multimodal proposals beat unimodal proposals and vanilla planning only under tight candidate budgets, long horizons that require sequence-level densities, or cost landscapes with between-mode local optima. Under generous budgets with rescoring, unimodal proposals match multimodal ones.
- Density-calibration metrics predict proposal quality weakly; action-validity metrics predict it strongly.

**Falsifiers**

- Apparent distributional-IDM planning gains disappear entirely when every representation is evaluated with the same planner proposal.
- Starting from the same checkpoint and using a matched head, end-to-end distributional-IDM training fails to improve representation probes or vanilla-planner performance over the frozen-encoder control.

### H10 - Shared local/goal action semantics are separable from density multimodality

An INTACT-style predictor can couple a local physical intent

$$
m_t^{\mathrm{local}}=z_{t+1}-z_t
$$

with a detached future-goal intent

$$
m_t^{\mathrm{goal}}=\operatorname{sg}(z_g)-z_t
$$

through a shared action law without requiring the two displacements to be pointwise equal. Actor sharing, local-to-goal support overlap, density family, and encoder gradient routing are independent scientific factors. This hypothesis activates with contingent Phase 1B, after the transition-only representation gate.

**Predictions**

- Paired local-plus-goal training outperforms goal-only training when physical-successor supervision adds more than behavior cloning.
- A fully shared predictor outperforms parameter-matched independent predictors when common action semantics are useful.
- Multimodal-density gains can appear with or without full actor sharing and are largest when the conditional action law has separated valid modes.
- Actor-disabled planning retains some gain when paired action losses improved the representation or forward model.
- Goal-conditioned performance degrades as deployment intents move outside supported local or demonstrated-goal regions.

**Falsifiers**

- Sharing gains disappear under parameter-matched, condition-token, or independent-capacity controls.
- Goal-only training matches the full paired objective.
- All improvement vanishes when the actor is disabled.
- Goal shuffling, previous-action shuffling, or episode-disjoint evaluation does not reduce apparent transfer.
- Improvements occur only for goal intents nearly identical to observed one-step intents.
- A multimodal head improves density fit but not valid action modes, representation probes, or downstream behavior.

## Secondary research questions

1. Which ambiguity types matter most: finite modes, continuous null spaces, saturation plateaus, policy mixtures, or partial observability?
2. When is endpoint conditioning preferable to displacement-only or state-plus-displacement conditioning?
3. Does higher action-density capacity improve the encoder or merely absorb ambiguity in the decoder?
4. How much behavior-policy diversity is required to distinguish environment ambiguity from policy artifacts?
5. Can controllable and exogenous task-relevant state coexist without a hybrid coverage objective?
6. Are gains caused by changing the representation, by using a learned action interface, by using a planner proposal, or by their interactions?
7. When is a learned heteroscedastic Gaussian sufficient, and when is explicit multimodal structure necessary?
8. Can Guarded local search hide invalid between-mode mass or density-family errors?

The remaining secondary questions - dimensionality effects across density families, cycle-consistency as a utility predictor, forward-model sensitivity, actor-sharing comparisons, gradient routing, previous-action conditioning, goal construction, INTACT-PRISM crossing - are archived with their contingent branches.

## Novelty boundary

### Potentially defensible contribution, subject to literature review

- Explicit multimodal inverse-density comparisons inside transition-only and shared local/goal action operators.
- Benchmarks that isolate structural inverse-action ambiguity, local-to-goal support overlap, and policy shortcuts.
- An evaluation protocol connecting calibrated action-mode quality to representation content, direct execution, actor-disabled planning, and downstream utility.
- A factorial separation of representation gradients, actor-sharing topology, density family, execution rule, and planner-proposal guidance.
- A carefully controlled conditional-information or held-out likelihood-improvement variant that compares transition-conditioned and state-only action models.
- Evidence about when distributional IDM and broad latent-coverage objectives are complementary.

INTACT is direct adjacent work for end-to-end action likelihood, shared physical/deployment intent prediction, and search-free or locally verified control. Its diagonal Gaussian does not resolve the project's explicit multimodality question. PRISM is direct adjacent work for probabilistic action heads and uncertainty-aware planning guidance on frozen representations. Delta-JEPA establishes displacement-conditioned deterministic inverse dynamics as a JEPA representation regularizer, and PLDM combines inverse dynamics with VICReg-style regularization, so the deterministic-IDM-as-regularizer precedent is established. LeWorldModel supplies a verified SIGReg-style coverage baseline for end-to-end JEPA world models. These baselines intervene at different stages and must not be treated as interchangeable.

### Claims not to make

- “Stochastic inverse dynamics is novel.”
- “The method makes inverse dynamics invertible.”
- “Injected noise is sufficient for multimodal prediction.”
- “Simple density heads beating deterministic regression under multimodality is a new finding.”
- “MSE assumes the real data are Gaussian.”
- “Conditional entropy is the irreducible MSE.”
- “Better action likelihood guarantees better representations.”
- “Displacement-only decoding always prevents shortcuts.”
- “A naïve difference of trainable log-likelihoods is automatically a mutual-information bound.”
- “This is the first probabilistic action head attached to a JEPA world model.”
- “This is the first end-to-end action NLL used to shape a JEPA representation.”
- “This is the first shared local/goal intent-to-action predictor.”
- “A shared actor proves local and goal latent displacements are equal or globally isomorphic.”
- “INTACT demonstrates explicit multimodal action-density learning.”
- “Search-free execution of a Gaussian mean is valid under multimodal inverse dynamics.”
- “This is the first use of state-dependent action uncertainty to improve world-model planning.”
- “Actor-disabled planning alone proves that the representation contains the correct action modes.”
- “INTACT and PRISM are interchangeable probabilistic-action baselines.”
- “Planning success proves that the learned action density represents all valid modes.”

## Open design questions requiring review

1. **Base architecture:** Which minimal predictive encoder or JEPA-like objective should receive the first regularizer integration?
2. **Initial data protocol:** Fixed generated datasets, online interaction, or both?
3. **Action scale:** Start with one-dimensional finite modes, then what dimension and nullity for redundant actions?
4. **Coverage baseline:** Which verified latent-distribution regularizer should be used?
5. **CMI-inspired design:** Evaluation-only ratio, frozen baseline, alternating optimization, or explicit variational bound?
6. **Downstream criterion:** Probes first, planning with known dynamics, or learned-model control?
7. **Density comparison:** How should normalized-likelihood and implicit/sample-based models be compared fairly?
8. **Success threshold:** What minimum downstream gain justifies added density-model complexity?
9. **Planner role:** Should the inverse density remain training-only, or also serve as a proposal or realizability signal during planning?
10. **Support gate:** Should local/goal support overlap remain diagnostic or become a numerical advancement criterion?

The remaining design questions - forward-verifier trust, source-review priorities, PRISM-comparison timing, goal construction, previous-action inclusion, Direct execution rules, actor topology controls, and Pure-CEM timing - are archived; they block contingent branches, not the decisive core.
