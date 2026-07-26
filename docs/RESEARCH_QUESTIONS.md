# Research Questions

## Framing

The proposed contribution is not stochastic action prediction in isolation. It is the use and controlled evaluation of a properly trained **distributional inverse-dynamics model as a representation regularizer** for world models or JEPA-like predictive encoders under action ambiguity.

Paper-specific descriptions of SIGReg, LeJEPA, Delta-JEPA, or related prior work remain **pending primary-source verification**. The hypotheses below are phrased so that the core experiments do not depend on those attributions.

## Primary research question

When the inverse map from an observed transition to an action is multimodal or non-identifiable, does replacing deterministic continuous-action MSE reconstruction with a conditional action-density objective produce representations that are more useful for forward prediction, planning, and control?

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

### H3 — Distributional IDM regularization improves representation content

When gradients from a proper conditional-density objective reach the encoder, the representation retains more controllable information than with deterministic IDM or no IDM.

**Predictions**

- Better controllable-state probes.
- Better learned forward prediction and downstream control/planning.
- Gains remain after capacity-matched decoder comparisons.

**Falsifiers**

- Only decoder likelihood improves; encoder probes and downstream outcomes do not.
- Gains disappear when decoder capacity and parameter count are controlled.
- Exogenous task-relevant information degrades enough to offset controllable-state gains.

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

### H6 — A carefully designed CMI-inspired training variant reduces shortcuts

A frozen-baseline, alternating, or explicitly derived variational objective can encourage extra transition information without allowing the baseline to be gamed.

**Predictions**

- Greater transition sensitivity at fixed state.
- Better held-out likelihood gain over the state-only model.
- No degradation of the baseline caused solely by adversarial encoder updates.

**Falsifiers**

- Training raises the score by worsening $\pi(a\mid z_t)$.
- Optimization becomes unstable or damages the base world-model objective.
- Gains vanish under held-out policy or transition-permutation tests.

### H7 — History resolves only observational, not structural, ambiguity

For partial observability where past observations identify the hidden state, a history-conditioned IDM should reduce uncertainty. For structurally many-to-one action effects, history should not recover the realized action.

**Predictions**

- Conditional entropy or multimodality decreases with history in resolvable POMDP variants.
- It remains in quadratic symmetry or action-null-space variants where action identity is absent from the full observed history.

**Falsifiers**

- History appears to resolve structural non-identifiability, suggesting data leakage or an unintended policy cue.

### H8 — Distributional modeling should not regress on one-to-one tasks

On tasks with an effectively deterministic inverse map, a simple distributional model should match deterministic IDM performance within reasonable compute and optimization tolerance.

**Falsifiers**

- Persistent degradation in likelihood, representation quality, or downstream performance on ordinary one-to-one dynamics.

## Secondary research questions

1. Which ambiguity types matter most: finite modes, continuous null spaces, saturation plateaus, policy mixtures, or partial observability?
2. When is endpoint conditioning preferable to displacement-only or state-plus-displacement conditioning?
3. How does action dimensionality affect discretized, MDN, flow, diffusion, and energy-based models?
4. Does higher action-density capacity improve the encoder or merely absorb ambiguity in the decoder?
5. Which encoder gradient paths and stop-gradient placements are necessary for stable representation learning?
6. How should bounded, constrained, or mixed discrete-continuous action spaces be modeled?
7. Can forward cycle consistency predict downstream utility, or is it only a local validity metric?
8. How much behavior-policy diversity is required to distinguish environment ambiguity from policy artifacts?
9. Can controllable and exogenous task-relevant state coexist without a hybrid coverage objective?
10. How sensitive are conclusions to known versus learned forward dynamics used for cycle evaluation?

## Novelty boundary

### Potentially defensible contribution, subject to literature review

- A world-model/JEPA regularizer based on a proper conditional action density rather than deterministic continuous-action reconstruction.
- Benchmarks that isolate several kinds of inverse-action ambiguity and policy shortcuts.
- An evaluation protocol connecting action-density quality to representation content and downstream utility.
- A carefully controlled conditional-information or held-out likelihood-improvement variant that compares transition-conditioned and state-only action models.
- Evidence about when distributional IDM and broad latent-coverage objectives are complementary.

### Claims not to make

- “Stochastic inverse dynamics is novel.”
- “The method makes inverse dynamics invertible.”
- “Injected noise is sufficient for multimodal prediction.”
- “MSE assumes the real data are Gaussian.”
- “Conditional entropy is the irreducible MSE.”
- “Better action likelihood guarantees better representations.”
- “Displacement-only decoding always prevents shortcuts.”
- “A naïve difference of trainable log-likelihoods is automatically a mutual-information bound.”

## Open design questions requiring review

1. **Base architecture:** Which minimal predictive encoder or JEPA-like objective should receive the first regularizer integration?
2. **Initial data protocol:** Fixed generated datasets, online interaction, or both?
3. **Action scale:** Start with one-dimensional finite modes, then what dimension and nullity for redundant actions?
4. **Coverage baseline:** Which verified latent-distribution regularizer should be used?
5. **CMI-inspired design:** Evaluation-only ratio, frozen baseline, alternating optimization, or explicit variational bound?
6. **Downstream criterion:** Probes first, planning with known dynamics, or learned-model control?
7. **Density comparison:** How should normalized-likelihood and implicit/sample-based models be compared fairly?
8. **Forward verifier:** When may a learned forward model be trusted for cycle metrics?
9. **Source review:** Which named methods and adjacent inverse-model literature are highest priority for primary-source verification?
10. **Success threshold:** What minimum downstream gain justifies added density-model complexity?
