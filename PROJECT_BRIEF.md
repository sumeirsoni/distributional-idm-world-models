# Project Brief: Distributional Inverse-Dynamics Regularization for Action-Ambiguous World Models

## Purpose of this document

This document is a research handoff for investigating whether a **distributional inverse-dynamics objective** can improve the representations learned by world models and JEPA-like predictive encoders when observed transitions are compatible with multiple actions.

The project is intentionally at the planning and scaffold stage. The immediate goal is not to assume the proposal works, but to establish a falsification-first program that separates three issues that are often blurred together:

1. representation collapse and broad latent coverage;
2. action-related representation content;
3. ambiguity or multimodality in the inverse map from transitions to actions.

As of August 2026, the INTACT, PRISM, Delta-JEPA, and LeJEPA/SIGReg descriptions in this document have been checked against their primary sources. That review identified two further adjacent works that require full-text reading before any comparative claim: LeWorldModel ([arXiv:2603.19312](https://arxiv.org/abs/2603.19312)), a SIGReg-regularized end-to-end JEPA world model, and PLDM ([arXiv:2502.14819](https://arxiv.org/abs/2502.14819)), which combines predictive learning, VICReg-style regularization, and inverse dynamics. Remaining paper-specific claims must be verified before publication.

## Working title

**Distributional Inverse-Dynamics Regularization for Action-Ambiguous World Models**

## Executive summary

A deterministic continuous-action inverse-dynamics model (IDM) commonly predicts an action from two consecutive representations and is trained with mean squared error:

$$
\mathcal{L}_{\mathrm{MSE-IDM}}
= \left\|a_t-g_\phi(z_t,z_{t+1})\right\|_2^2.
$$

At the population optimum, this decoder returns the conditional mean

$$
g_\phi^*(z_t,z_{t+1})
= \mathbb{E}[A_t\mid Z_t=z_t,Z_{t+1}=z_{t+1}].
$$

When the inverse map is multimodal, the conditional mean can lie between valid modes and may itself be physically invalid. The proposed replacement is a conditional action density:

$$
q_\phi(a_t\mid z_t,z_{t+1}),
\qquad
\mathcal{L}_{\mathrm{dist-IDM}}
= -\log q_\phi(a_t\mid z_t,z_{t+1}).
$$

This density should be used as a regularizer alongside the world model's normal predictive or alignment objective. The research question is not merely whether the density model predicts actions better. It is whether its training signal encourages representations that retain useful controllable structure and improve prediction, planning, or control under action ambiguity.

[INTACT](https://arxiv.org/abs/2607.26056) already demonstrates end-to-end action NLL in a JEPA-style world model, paired physical-transition and future-goal action losses, a shared local/goal action predictor, and direct probabilistic action execution. The defensible incremental question is therefore narrower: whether an explicitly multimodal inverse density improves representation content, valid action-mode coverage, and downstream behavior beyond deterministic and INTACT-style unimodal Gaussian controls when the inverse map is structurally non-identifiable.

The central caveat is equally important: a distributional IDM can represent uncertainty or non-identifiability honestly, but it cannot recover action information that is absent from the observations. If the same observed transition is genuinely produced by either $+1$ or $-1$, the correct model may be a 50/50 conditional distribution; no decoder can identify which action occurred without additional information.

## Competing mechanisms behind any representation effect

All proper action-prediction objectives reward representations that preserve action-relevant transition information. They differ in how strongly they reward it and in what they do when ambiguity is irreducible. The program pre-registers two opposing mechanisms instead of assuming the intended direction:

1. **Pressure strength.** Deterministic MSE leaves excess loss proportional to squared mode distance whenever the encoder merges states that need different actions, and separating those latents removes the loss without bound. A density head fitted to merged latents reaches its Bayes optimum, and separation recovers only the bounded mixture entropy. Deterministic regression may therefore exert stronger anti-aliasing pressure on the encoder.
2. **Calibrated acceptance.** When ambiguity is structural, no representation can remove it. The deterministic objective cannot reach its optimum, and its residual gradients reward spurious cues that correlate with action identity, such as behavior-policy traces. A properly fitted conditional density accepts the irreducible entropy and stops there. Distributional objectives may therefore produce fewer shortcuts and retain more exogenous information.

The two mechanisms make different predictions:

- Where the encoder can resolve aliasing, mechanism 1 predicts deterministic-IDM representations beat distributional-IDM representations on controllable-state probes.
- Where ambiguity is structural, mechanism 2 predicts deterministic-IDM encodes more behavior-policy and spurious action cues, measured by shortcut probes, and retains less exogenous information.
- If neither signature appears in any environment pair, decoder family is representationally irrelevant at the tested scale, and the representation claim fails regardless of density-fit gains.

The frozen-head control separates decoder-side fit from encoder-side structure throughout.

## 1. Two different meanings of “distributional”

### 1.1 Latent distribution regularization

Latent distribution regularizers—such as covariance, variance, whitening, spectral, uniformity, or other aggregate-geometry constraints—shape the distribution of representations. Their purpose is generally to prevent representational collapse or pathological concentration.

Abstractly, such a term may constrain an aggregate distribution or batch statistic:

$$
\mathcal{R}_{\mathrm{latent}}(\{z_i\}_{i=1}^B).
$$

This says that representations must vary or occupy a suitably rich geometry. It does **not** by itself specify which factors of variation should be retained. Nuisance variables can satisfy a variance or rank constraint just as well as controllable or task-relevant state.

References to SIGReg- or LeJEPA-style constraints should therefore be treated as examples of broad latent-distribution regularization, subject to source verification of the exact formulations and claimed effects.

### 1.2 Distributional inverse dynamics

A distributional IDM models a conditional density over actions:

$$
q_\phi(a_t\mid z_t,z_{t+1}).
$$

Its role is semantic rather than merely geometric: it pressures the representations to preserve information useful for describing which actions are compatible with a transition. It also permits more than one plausible action for a single represented transition.

These are distinct uses of “distributional”:

- **latent distribution regularization** asks whether the representation population has adequate variation or coverage;
- **distributional inverse dynamics** asks whether a transition-conditioned action distribution is represented correctly.

They are complementary, not interchangeable. A useful hybrid may combine both.

## 2. Why deterministic continuous-action MSE is insufficient

### 2.1 Population optimum

Let $X=(Z_t,Z_{t+1})$. For a deterministic predictor $g(X)$, the population squared-error risk is

$$
\mathcal{R}(g)=\mathbb{E}\left[\|A-g(X)\|_2^2\right].
$$

Conditioning on $X=x$, the minimizer is

$$
g^*(x)=\mathbb{E}[A\mid X=x].
$$

The irreducible minimum squared error is governed by conditional second moments—equivalently the conditional variance/covariance—not by conditional entropy. This distinction matters: entropy describes uncertainty in the full conditional distribution, whereas MSE only elicits its mean.

MSE does **not** require the data-generating conditional to be Gaussian in order to estimate the conditional mean. A Gaussian observation model with fixed variance is one probabilistic interpretation of squared error, but non-Gaussian data do not invalidate the conditional-mean result. The actual defect is that the mean is an inadequate summary of a multimodal conditional.

### 2.2 Canonical ambiguity example

Consider

$$
s_{t+1}=s_t+a_t^2.
$$

For a transition with $s_{t+1}-s_t=1$, both $a_t=+1$ and $a_t=-1$ are valid. Under a symmetric behavior policy,

$$
p(a_t\mid s_t,s_{t+1})
=\tfrac12\delta_{-1}+\tfrac12\delta_{+1}.
$$

The MSE-optimal prediction is zero:

$$
\mathbb{E}[A_t\mid s_t,s_{t+1}]=0.
$$

But action zero produces no state change, so the predicted action is not merely uncertain—it is inconsistent with the observed transition.

### 2.3 Scope of the criticism

For discrete actions, a softmax classifier trained with cross-entropy already represents a conditional categorical distribution. It can allocate probability to multiple valid actions. The sharp criticism therefore applies primarily to **deterministic continuous-action regression**, though discrete models can still suffer shortcut learning, poor calibration, insufficient conditioning, or representation collapse.

## 3. Delta-conditioned decoding and its limits

An ordinary endpoint-conditioned inverse decoder

$$
g_\phi(z_t,z_{t+1})
$$

may exploit shortcuts. For example, if the behavior policy strongly correlates state with action, the decoder can predict $a_t$ mostly from $z_t$ and make little use of the represented transition.

A displacement-conditioned decoder instead uses

$$
g_\phi(\Delta z_t),
\qquad
\Delta z_t=z_{t+1}-z_t.
$$

The intended pressure is for the latent transition itself to reveal action-related information. This addresses a **conditioning or shortcut-learning problem**. It does not, by itself, address multimodality.

If a displacement-conditioned model still emits one continuous action and is trained with MSE, its population prediction remains a conditional mean:

$$
g^*(\Delta z_t)=\mathbb{E}[A_t\mid \Delta Z_t=\Delta z_t].
$$

It therefore cannot represent multiple actions that produce the same displacement.

Displacement-only decoding also introduces a separate possible failure. In many systems, action effects depend on state:

$$
s_{t+1}=F(s_t,a_t),
$$

and the same action can produce different displacements in different states, or the same displacement can require different actions. Removing $z_t$ may therefore make inverse prediction ill-posed. Endpoint, displacement-only, and state-plus-displacement conditioning should be compared empirically rather than assuming one is universally preferable.

The name and exact formulation of “Delta-JEPA,” along with claims about its motivation or empirical performance, require primary-source verification.

## 4. Primary proposed objective

Let an encoder produce

$$
z_t=f_\theta(o_t),
\qquad
z_{t+1}=f_\theta(o_{t+1}),
$$

or the analogous online/target representations used by the chosen world-model architecture. Replace deterministic action reconstruction with a proper conditional density objective:

$$
\mathcal{L}_{\mathrm{dist-IDM}}(\theta,\phi)
=\mathbb{E}_{(o_t,a_t,o_{t+1})}
\left[-\log q_\phi(a_t\mid z_t,z_{t+1})\right].
$$

Combine this with the world model's ordinary prediction, alignment, or reconstruction loss:

$$
\mathcal{L}_{\mathrm{total}}
=\mathcal{L}_{\mathrm{WM}}
+\lambda_{\mathrm{IDM}}\mathcal{L}_{\mathrm{dist-IDM}}
+\lambda_{\mathrm{coverage}}\mathcal{R}_{\mathrm{latent}},
$$

where the broad latent-coverage term is optional in the primary comparison but important as a hybrid baseline.

Tune $\lambda_{\mathrm{IDM}}$ and $\lambda_{\mathrm{coverage}}$ on validation splits only, and report sensitivity curves. Objective families differ in gradient scale, so untuned or test-tuned weights confound family comparisons.

The encoder-gradient path must be explicit. A highly capable decoder can sometimes fit behavior-policy regularities while ignoring one or both conditioning representations. Diagnostics must confirm that the conditional prediction changes when $z_{t+1}$, $\Delta z_t$, or other transition information is changed.

### 4.1 Noise injection is not enough

Adding a random input $\epsilon$ to a deterministic decoder,

$$
a=g_\phi(z_t,z_{t+1},\epsilon),
$$

does not automatically define or train a correct conditional distribution. If it is still optimized by pointwise MSE against one observed action, the model may ignore the noise or regress toward a mean.

The model needs an objective that is proper for distributions, such as:

- exact or approximate conditional log-likelihood;
- score matching;
- flow matching;
- a well-defined energy objective with valid negative sampling or sampling-based training;
- another proper scoring rule suitable for the selected action representation.

The density family and objective must be evaluated together.

### 4.2 Objective comparability and PRISM-style beta-NLL

[PRISM](https://arxiv.org/abs/2606.07974) trains a heteroscedastic diagonal-Gaussian action-sequence head with beta-NLL after freezing its JEPA world model. Its reported objective uses $\beta=0.5$ and an elementwise detached variance weight:

$$
\mathcal{L}_{\beta\text{-NLL}}
=
\operatorname{sg}\left((\sigma^2)^\beta\right)
\left[
\frac{(a^\star-\mu)^2}{2\sigma^2}
+\log \sigma
\right].
$$

A PRISM-faithful ablation must preserve the detached multiplier, elementwise reduction over normalized actions, $\beta=0.5$, and the documented standard-deviation floor. This is relevant as an optimization and uncertainty-modeling baseline, but it should not be treated as interchangeable with ordinary conditional Gaussian maximum likelihood.

The clean primary unimodal-density baseline for this project should use ordinary conditional Gaussian NLL. The beta-NLL variant should be an explicitly labeled objective ablation. All Gaussian variants should be evaluated with the same held-out ordinary NLL, continuous calibration, sharpness, and action-validity metrics rather than by comparing non-commensurate training losses.

### 4.3 INTACT-style paired local and goal action laws

[INTACT](https://arxiv.org/abs/2607.26056) trains one shared predictor on a local physical intent $z_{t+1}-z_t$ and a detached future-goal intent $\operatorname{sg}(z_g)-z_t$, with asymmetric gradient routing: gradients flow through the current representation and the physical-successor occurrence, while the future-goal occurrence is a stop-gradient anchor. The paired operator, its routing controls, its actor-sharing topologies, and its execution rules are deferred to contingent Phase 1B; the full specification is archived. The transition-only density remains the primary ambiguity probe.

## 5. Shortcut-resistant conditional-information variant

A transition-conditioned model can predict actions well without using the transition if the behavior policy makes actions predictable from state alone. Compare $q_\phi(a_t\mid z_t,z_{t+1})$ with a properly fitted state-only baseline $\pi_\psi(a_t\mid z_t)$ through the held-out score

$$
\mathbb{E}\left[\log q_\phi(a_t\mid z_t,z_{t+1})-\log \pi_\psi(a_t\mid z_t)\right],
$$

whose population analogue is the conditional mutual information $I(A_t;Z_{t+1}\mid Z_t)$.

Maximizing this difference jointly over all parameters is unsound: the baseline can be made deliberately worse to inflate the score, encoder updates can degrade $z_t$ for the baseline rather than improve transition information, and model misspecification affects the two terms differently. The recommendation is conservative: use the score as an evaluation diagnostic with separately and properly fitted models, and defer any training variant to contingent Phase 2. The required diagnostics - permutation tests, capacity matching, policy shifts, and baseline-degradation monitoring - are archived with the Phase 2 design.

## 6. Candidate conditional model families

The project should increase model complexity only when simpler models establish the phenomenon.

1. **Discretized continuous-action bins with cross-entropy.** Easy to inspect and calibrate; resolution grows poorly with action dimension and introduces binning error.
2. **Fixed scalar-variance conditional Gaussian.** An equivalence and implementation sanity check: with one fixed isotropic scalar variance, Gaussian NLL is globally scaled MSE plus a constant, so it has the same optimum and gradient direction unless the action support is transformed. A fixed unequal diagonal covariance instead corresponds to dimension-weighted MSE and must be paired with that weighted-MSE control.
3. **Heteroscedastic diagonal Gaussian.** Predicts both mean and state-dependent variance. This tests whether uncertainty alone is useful before adding multimodal expressivity, but it can cover separated modes by inflating variance and assigning mass to invalid actions.
4. **Mixture density networks (MDNs).** A practical first continuous baseline with explicit modes; vulnerable to component collapse, variance pathologies, and sensitivity to the number of components. Apply variance floors or clamping during joint encoder training, not only in standalone density fitting.
5. **Winner-takes-all or best-of-$K$.** A simple multimodal prediction baseline; may cover modes without yielding a normalized calibrated density.
6. **Conditional normalizing flows.** Exact likelihood for invertible continuous transforms; potentially expressive but more difficult for disconnected or lower-dimensional supports.
7. **Energy-based models (EBMs).** Flexible unnormalized conditionals; training and Langevin sampling can be expensive or unstable, and likelihood comparison is difficult.
8. **Conditional diffusion or flow matching over actions.** Flexible sampling for complex conditionals; substantially greater computational and evaluation complexity.
9. **History-conditioned inverse models.** Model
   $$
   q(a_t\mid z_{t-k:t+1})
   $$
   to test whether temporal context resolves ambiguity caused by partial observability. This is an ambiguity-resolving baseline, not merely a more expressive density family.

For multidimensional bounded actions, support handling must be explicit—for example, truncated distributions, transformed unconstrained variables, or bins limited to valid action ranges.

## 7. Scientific caveats and scope boundaries

### 7.1 Honest ambiguity is not action recovery

If action identity is absent from the observed transition, a stochastic model can represent the compatible set or probabilities but cannot infer the realized action. Success should be defined as calibrated mode representation and useful encoder pressure, not impossible reconstruction.

### 7.2 Flexible decoders can ignore conditioning

A density decoder may collapse to a marginal or state-only behavior-policy model:

$$
q(a_t\mid z_t,z_{t+1})\approx p(a_t)
\quad\text{or}\quad
q(a_t\mid z_t,z_{t+1})\approx p(a_t\mid z_t).
$$

Conditional-vs-state-only likelihood, permutation tests, and transition sensitivity are mandatory.

### 7.3 The controllability gap

IDM objectives reward representations that retain controllable information. General-purpose world models also require exogenous or uncontrollable but task-relevant dynamics: other agents, moving obstacles, weather, latent goals, or external events.

A distributional IDM does not solve this gap. It can even bias capacity away from exogenous factors if over-weighted. Include probes and downstream tasks for both controllable and exogenous state, and compare against a broad anti-collapse/coverage regularizer and a hybrid objective.

### 7.4 Sources of multimodality must be separated

Multimodal inverse conditionals may arise from:

- actuator redundancy or null action dimensions;
- action saturation;
- symmetries in the transition function;
- partial observability;
- temporal aliasing that history can resolve;
- stochastic environment dynamics;
- the data-collection or behavior policy;
- mixtures of policies or demonstrators.

Experiments should distinguish environment-level non-identifiability from behavior-policy artifacts. Policy diversity must be varied explicitly.

### 7.5 Novelty boundary

Do not claim that:

- stochastic inverse dynamics is itself novel;
- a distributional decoder makes a non-injective inverse map invertible;
- noise injection alone solves multimodal regression;
- better action likelihood necessarily implies better world-model representations;
- endpoint or displacement conditioning universally prevents shortcuts.

The proposed contribution must be framed and evaluated as a controlled study of **explicitly multimodal inverse-density regularization under structural action ambiguity**. The study must isolate density expressivity, representation gradients, local/goal actor sharing, execution rules, and planner proposals. It must not claim the first end-to-end action NLL for a JEPA, the first shared local/goal action predictor, global latent-action isomorphism, or evidence that INTACT already handles explicit multimodality.

### 7.6 Adjacent work: INTACT

[INTACT: Isomorphic Intent-to-Action Learning for Search-Free World Models](https://arxiv.org/abs/2607.26056) is the closest precedent for representation-shaping action likelihood and amortized intent-to-action control. It jointly trains a JEPA-style encoder, forward predictor, and shared diagonal-Gaussian action predictor using both local physical intents and detached future-goal intents. Its Direct controller executes the Gaussian mean, while its Guarded variant performs limited local verification.

INTACT demonstrates that action-likelihood supervision can improve actor-disabled forward planning, so broad claims that probabilistic action NLL can shape a world-model representation are no longer distinctive. It does not compare a Gaussian with an explicit multimodal density under controlled non-identifiability, and its mean can lie between separated valid action modes. This project therefore treats INTACT's shared local/goal Gaussian as a required control and asks whether calibrated multimodal structure adds representation or control value beyond it.

### 7.7 Adjacent work: PRISM

[PRISM: PRior-guided Imagination Sampling in world Models](https://arxiv.org/abs/2606.07974) is relevant adjacent work, but it intervenes at a different stage. PRISM freezes a JEPA world-model encoder and predictor, then trains a small goal-conditioned head that models a diagonal-Gaussian distribution over future action chunks:

$$
p_\phi(a_{t:t+HB-1}\mid z_t,z_g).
$$

At deployment, the learned mean and variance are fused with the proposal distribution of MPPI or CEM to improve candidate sampling. PRISM therefore studies a probabilistic planner proposal on fixed representations. This project studies a transition-conditioned inverse density whose gradients are intended to shape the representation during world-model training:

$$
q_\phi(a_t\mid z_t,z_{t+1}).
$$

PRISM narrows broad novelty claims about probabilistic action heads, state-dependent action uncertainty, and uncertainty-aware action sampling in JEPA world models. It does not establish that a distributional inverse objective improves representation learning under action ambiguity.

PRISM also illustrates why planning success is not sufficient evidence of density correctness. Its unimodal Gaussian can assign probability between distinct behavior modes, while MPPI can still recover useful actions by evaluating and refining sampled candidates with the world model. This project must therefore retain direct calibration, mode-coverage, mode-precision, and known-forward cycle metrics in addition to planning outcomes.

| Dimension | Proposed distributional IDM | INTACT | PRISM |
|---|---|---|---|
| Primary conditioning | Observed transition, with paired local/goal extension | Shared local and future-goal intents plus previous action | Current and goal latents |
| Density target | Usually one-step action | Action block under diagonal Gaussian | Action sequence or chunk under diagonal Gaussian |
| Encoder training | Action-density gradients are a controlled factor | End-to-end action and world losses | Frozen encoder and world model |
| Multimodality focus | Central research question | No explicit mixture comparison | Unimodal Gaussian proposal |
| Default execution | Training regularizer first; execution is factorialized | Direct mean with optional local verification | MPC proposal guidance |
| Scientific role | Representation under inverse ambiguity | Amortized intent-to-action control | More efficient planner sampling |

## 8. Falsification-first experimental program

## Phase 0: controlled synthetic environments

### A. Quadratic ambiguity

$$
s_{t+1}=s_t+a_t^2.
$$

Begin with one-dimensional symmetric actions so $+a$ and $-a$ are observationally identical. Control the action distribution so that experiments include:

- symmetric modes with identical probability;
- asymmetric mode probabilities;
- conditionals with the same mean but different modal structure;
- low-noise and noisy observations;
- one-to-one control variants for regression testing.

Expected deterministic-MSE failure: prediction between modes, poor forward cycle consistency, and inability to distinguish conditionals sharing a mean.

### B and C. Deferred ambiguity variants

Redundant linear maps with a nontrivial null space, action saturation, partial observability with history controls, exogenous variables, state-dependent action effects, and behavior-policy diversity sweeps are deferred to contingent branches; their constructions and expected failure signatures are archived. Two of them activate first because they carry the competing-mechanism signatures: the exogenous-variable environment (E8) tests calibrated acceptance, and the policy-shortcut environment (E11) tests pressure strength.

## 9. Baselines

The decisive-core comparison set is:

- no IDM;
- deterministic endpoint-conditioned MSE IDM;
- heteroscedastic diagonal-Gaussian IDM trained with ordinary NLL;
- MDN distributional IDM;
- generic auxiliary-task control: an equally parameterized head predicting a non-action target, such as temporal distance or random features, at matched weight and schedule, to separate auxiliary-supervision effects from action-content effects;
- latent coverage regularizer (one verified SIGReg-style method);
- hybrid distributional IDM plus latent coverage regularization;
- frozen-encoder probabilistic-head control to isolate decoder-local gains from representation changes;
- state-only action predictor $\pi(a\mid z_t)$.

Match decoder capacities and parameter counts across arms, and report frozen versus end-to-end encoder training explicitly. Deferred baselines - displacement-conditioned MSE, discretized bins, fixed scalar-variance Gaussian, PRISM-style beta-NLL, best-of-K, history conditioning, INTACT-style actor topologies, higher-capacity densities, and planner proposals - are archived and enter only when a gate demands them.

## 10. Measurements

### 10.1 Conditional-density quality

- Held-out conditional negative log-likelihood when tractable.
- Held-out likelihood improvement over $\pi(a\mid z_t)$.
- Mode coverage: fraction of valid modes represented.
- Mode precision: fraction of predicted mass or samples that correspond to valid modes.
- Invalid probability mass between known valid modes.
- Ability to distinguish conditionals with identical means but different modal structure.

For models without tractable normalized likelihood, use clearly labeled alternative proper scores or sample-based metrics; do not compare surrogate training losses as if they were commensurate NLLs. Calibration, coverage, sharpness, support-overlap, and action-law metrics are archived until their phases activate.

### 10.2 Forward cycle consistency

Sample candidate actions from the inverse model and pass them through the known synthetic transition or a separately evaluated learned forward model. Measure whether they land near the target next state or representation:

$$
\epsilon_{\mathrm{cycle}}
= d\big(F(s_t,\tilde a_t),s_{t+1}\big)
\quad\text{or}\quad
 d\big(\hat F(z_t,\tilde a_t),z_{t+1}\big).
$$

Report this as a metric first. Do not assume a learned forward model is a trustworthy verifier: it may share representation errors, extrapolate poorly, or accept invalid actions.

### 10.3 Representation quality

- Collapse indicators: per-dimension variance, covariance spectrum, effective rank, and singular values.
- Linear and nonlinear probes for controllable state.
- Linear and nonlinear probes for exogenous task-relevant state.
- Forward prediction error.
- Sensitivity and invariance tests tied to known generative factors.
- Representation performance under action ambiguity and under ordinary one-to-one dynamics.

### 10.4 Downstream utility

- Planning or model-predictive-control performance on one small task, as an endpoint component.
- Success or return versus planner candidate count.
- Actor-disabled planning delta for the same action-supervised checkpoint.
- Robustness and no-regression checks on E10.

Planning experiments must separate representation quality from action-interface quality and proposal quality. A planning gain that appears only with a learned proposal should not be attributed to representation regularization. The full representation-by-proposal factorial, including co-trained actor and mixture-fusion arms, is archived until MPC experiments begin.

### 10.5 Shortcut diagnostics

- Conditional versus state-only held-out likelihood.
- Transition permutation or shuffling tests.
- Fixed-state transition sensitivity.
- Performance under shifts in the behavior policy.

## 11. Ablations

Core ablations for the decisive experiment:

- density family: deterministic mean, learned heteroscedastic variance, and multimodal density;
- $\lambda_{\mathrm{IDM}}$ and $\lambda_{\mathrm{coverage}}$, tuned on validation splits only with sensitivity curves reported;
- frozen versus end-to-end heads initialized from the same world-model checkpoint, with every gradient-receiving component enumerated;
- MDN component count, with variance floors or clamping during joint encoder training;
- behavior-policy diversity and state-action correlation (E11);
- exogenous-state capacity and task relevance (E8);
- observation noise.

The extended ablation checklist - conditioning variants, actor topologies, routing, execution rules, planner proposals, and data sweeps - is archived and activates with its contingent branch.

## 12. Decision gates

### Gate 1: implementation sanity check

Under symmetric $\pm a$ actions, the conditional-mean behavior of MSE and the mode capture of a correctly optimized density model follow from standard results. Passing Gate 1 is therefore expected. The gate exists to catch optimization, capacity, and metric bugs before they contaminate later phases, not to test a scientific hypothesis.

**Debugging trigger:** deterministic MSE does not exhibit the expected invalid-mean behavior under the designed data, or the simple distributional model cannot reliably capture modes despite adequate optimization and capacity. Treat either outcome as an implementation defect to locate and fix.

### Gate 2: establish representation benefit

Proceed only if the pre-registered primary endpoint improves beyond its multiplicity-corrected threshold and the improvement survives the frozen-head control. Secondary metrics generate hypotheses; they never justify advancement alone. Report which competing-mechanism signature appeared, or that neither did.

**Falsifier:** the density model predicts actions better but the pre-registered endpoint shows no improvement, or exogenous information degrades enough to erase any benefit.

### Gate 3: establish shared-operator or multimodal value

Advance the paired local/goal study only if at least one matched claim survives: local plus goal beats goal only; full sharing beats parameter-matched independent actors; action-supervised checkpoints improve actor-disabled evaluation; or an explicit multimodal density improves valid-mode or downstream outcomes beyond the shared Gaussian. Support-overlap analysis must rule out uncontrolled goal extrapolation.

**Falsifier:** Direct success is explained entirely by goal-conditioned behavior cloning, actor capacity, episode retrieval, local search, or a unimodal mean that remains invalid between modes.

### Gate 4: test shortcut resistance

Proceed only if the CMI-inspired or likelihood-improvement variant reduces reliance on state-policy shortcuts without destabilizing training or gaming the baseline.

**Falsifier:** held-out transition-conditioned improvement is illusory, disappears under policy shift or permutation tests, or comes from making the baseline worse.

### Gate 5: justify model complexity

Add a conditional flow, diffusion model, or EBM only if simpler categorical or MDN models reveal a clear density-support limitation that added expressivity is likely to address.

**Falsifier:** simple models already saturate the relevant metrics, or higher-capacity models improve density fit without representation or downstream gains.

### Gate 6: justify environment scaling

Move to a larger controlled environment when the core ambiguity, representation, shortcut, and shared-operator conclusions survive their matched controls. Scaling does not require a higher-capacity density if a simple model remains adequate.

**Falsifier:** the larger environment removes the diagnostics needed to distinguish density, representation, actor, and planner effects.

## 13. Recommended initial implementation order after approval

1. Implement E1 and E10 generators with analytically known ambiguity, and fix train/validation/test protocols with policy-diversity controls.
2. Implement the core baselines and the density, mode, cycle-consistency, collapse, probe, and shortcut metrics.
3. Pass the Phase 0A implementation check.
4. Run the decisive core (Phase 1A) under the pre-registered endpoint.
5. Interpret the competing-mechanism signatures; advance through the gates, redirect the design, or accept the null-result deliverable.
6. Activate contingent branches only as their gates open; add PRISM-style proposal guidance only when downstream MPC experiments begin.

## 14. Questions requiring user review before implementation

Questions 1 through 5 block the start of implementation. Questions 6 through 8 block only contingent branches; answer them at branch activation.

### Blocking

1. What do the E1 and E10 observations look like, and which encoder reads them?
   - **Recommended default:** Plain state vectors, read by a small MLP online/target encoder. Skip benchmark architectures; they add cost without changing the question. Keep the encoder weak on purpose. A constant latent with a compensating predictor already satisfies the forward loss, so the no-IDM arm can collapse, and Gate 2 needs that failure to exist. If the IDM arms prevent the collapse while the generic auxiliary-task arm does not, the cause is action content rather than extra supervision.
2. Where does the training data come from?
   - **Recommended default:** Generate fixed datasets up front. That keeps runs reproducible and pins the conditional action laws exactly. Record dataset size, episode structure, and behavior-policy draws in `results/preregistration.md` before training. Add online rollouts later, only after the core results are stable.
3. Which regularizer plays the SIGReg/LeJEPA coverage role?
   - **Recommended default:** VICReg's variance plus covariance terms. They are simple, and they keep pushing while embeddings are nearly collapsed. SIGReg would match LeWM, but its test statistic loses gradient as embeddings approach collapse, and near-collapse is exactly where this arm has to work. Use one, not both.
4. Who owns the endpoint numbers, and what will compute cost?
   - **Recommended default:** The experiment plan states the endpoint, seed count, equivalence margins, and multiplicity correction. Review them there, once. Compute costs about 70 runs across the decisive core, from seven arms times two training modes times five seeds. Frozen-head runs are cheap; end-to-end arms dominate the bill.
5. Does the staged ladder conflict with the composite endpoint?
   - **Recommended default:** Not if the planning component uses known dynamics. The decisive core then computes all three endpoint parts at once. Probes still get reported first, and learned-model control stays outside the endpoint until simpler evaluations show a benefit.

### Needed at branch activation

6. Should the first CMI-inspired experiment be evaluation-only, frozen-baseline training, or a derived variational bound?
   - **Recommended default:** Evaluation-only held-out likelihood difference between separately and properly fitted transition-conditioned and state-only models. Escalate to training variants only if the diagnostic reliably detects permutation and policy shortcuts. Blocks Phase 2.
7. Should the learned inverse density remain training-only, or also serve as a planner proposal or realizability signal?
   - **Recommended default:** Establish the training-only representation effect first. Evaluate Direct execution, realizability filtering, and planner-proposal use later as separately labeled interventions. Blocks Phases 1B and 3P.
8. In what order should contingent branches activate after the decisive core?
   - **Recommended default:** E8 and E11 first, because they carry the mechanism signatures; then Phase 1B, Phase 2, Phase 3, Phase 3P, and Phase 4 in gate order.

Questions about redundant-map dimensionality, paired-operator goal construction, previous-action conditioning, mode-aware Direct rules, actor-sharing controls, and support-overlap gating are archived with their recommended defaults; review them when their branch opens.

## Acceptable outcomes including null results

The most likely single outcome at Gate 2 is a null or small representation effect, because every proper action-prediction objective creates encoder incentives in the same direction and differs mainly in magnitude. Treat that outcome as publishable rather than failed. The deliverable would be the synthetic environment suite, the metric protocol, and a controlled analysis of when decoder expressivity does and does not shape world-model representations. Define the primary endpoint before implementation so a null result is interpretable rather than inconclusive.

## 15. Current project boundary

This repository should remain a planning and scaffold artifact until the user reviews this handoff. No expensive jobs, dataset downloads, dependency installation, substantive experiment implementation, or unsupported literature claims should be added before approval.
