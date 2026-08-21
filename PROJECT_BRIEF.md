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

[INTACT](https://arxiv.org/abs/2607.26056) trains one intent-to-action predictor on two condition families:

$$
m_t^{\mathrm{local}}=z_{t+1}-z_t,
\qquad
m_t^{\mathrm{goal}}=\operatorname{sg}(z_g)-z_t.
$$

Its shared predictor conditions on the common input grammar

$$
G_\eta\!\left(z_t,m_t,a_{t-1}\right),
$$

implemented with current-state, intent, state-intent interaction, and previous-action features. The paired action objective is conceptually

$$
\lambda_{\mathrm{local}}
\mathbb{E}\!\left[-\log p_\eta(a_t\mid z_t,m_t^{\mathrm{local}},a_{t-1})\right]
+
\lambda_{\mathrm{goal}}
\mathbb{E}\!\left[-\log p_\eta(a_t\mid z_t,m_t^{\mathrm{goal}},a_{t-1})\right].
$$

The local and goal displacements are not assumed to be numerically equal or globally aligned. The relevant claim is that supported conditions can share an action-law interpretation. Proper NLL determines behavior only on sampled support, so experiments must measure local-to-goal intent overlap, performance by support-overlap quantile, and degradation on extrapolative goals rather than treating shared parameters as proof of global isomorphism.

The INTACT-style default gradient route is asymmetric:

- gradients flow through the current representation $z_t$;
- the local physical-successor occurrence $z_{t+1}$ remains attached;
- the future-goal occurrence $z_g$ is a stop-gradient anchor;
- the shared action predictor receives both losses;
- every encoder, target encoder, predictor, previous-action encoder, shared trunk, and branch-specific head must be enumerated if present.

Required routing and coupling controls include joint goal-endpoint gradients, both target endpoints detached, a frozen encoder/action-head-only arm, local-only and goal-only losses, a fully shared actor, a condition-token shared actor, a shared trunk with separate outputs, parameter-matched independent actors, and an independent-actor capacity control. A latent detached as a future goal in one call may still receive gradients when it appears as a current state or physical successor elsewhere.

The transition-only density remains the primary ambiguity probe. The paired local/goal operator is an additional baseline and deployment-aligned extension. Hold its conditioning grammar fixed while comparing a diagonal Gaussian with discretized or mixture-based multimodal densities. Cross operator design and density family independently so a benefit from local/goal sharing is not confused with a benefit from multimodal expressivity.

For execution, report distinct rules rather than one aggregate control score:

1. **Direct mean:** execute the conditional mean with no candidate search.
2. **Direct mode:** use a predeclared rule for a multimodal model, either numerically maximize the full density or select a component representative by peak density. Do not treat the highest-weight component mean as the global mixture mode. For a single Gaussian, mean and mode coincide.
3. **Direct sample:** draw a calibrated sample under a fixed sampling budget.
4. **Guarded/local verification:** preserve the Direct plan as a reference and run limited local search around it, retaining the globally best candidate.
5. **Pure CEM/execution actor disabled:** remove the action predictor from execution and plan through the learned representation and forward model.

Execution actor disabled and training action objective disabled are different controls. The first tests whether action-supervised training improved the representation or forward model; the second removes action-supervision gradients during training.

## 5. Shortcut-resistant conditional-information variant

### 5.1 Motivation

A transition-conditioned model can predict actions well without learning useful transition content if the behavior policy already makes actions predictable from the current state. Introduce a state-only baseline:

$$
\pi_\psi(a_t\mid z_t),
$$

and compare it with

$$
q_\phi(a_t\mid z_t,z_{t+1}).
$$

For a sample, define the likelihood-ratio-style score

$$
r_t
=\log q_\phi(a_t\mid z_t,z_{t+1})
-\log \pi_\psi(a_t\mid z_t).
$$

In expectation under the true data distribution, the analogous difference between true conditional log densities is related to conditional mutual information:

$$
I(A_t;Z_{t+1}\mid Z_t)
=\mathbb{E}\left[
\log p(a_t\mid z_t,z_{t+1})
-\log p(a_t\mid z_t)
\right].
$$

The intended signal is the extra action information supplied by the next representation beyond what is already predictable from the current representation.

### 5.2 Why a naïve trainable difference is unsafe

Simply maximizing

$$
\log q_\phi(a_t\mid z_t,z_{t+1})-
\log \pi_\psi(a_t\mid z_t)
$$

jointly over all parameters is not automatically a valid mutual-information estimator or lower bound. In particular:

- the baseline can be made deliberately worse, artificially increasing the difference;
- encoder updates can degrade $z_t$ for the baseline rather than improve transition information;
- density-model misspecification gives different approximation errors to the two terms;
- unrestricted scaling or poorly normalized energies may make scores incomparable;
- using the same data for fitting and evaluation can overstate likelihood gains.

### 5.3 Candidate sound designs

At least one of the following should be specified before implementation:

1. **Separate fitting with a frozen baseline.** Fit $\pi_\psi(a\mid z_t)$ to convergence or on alternating phases, then stop gradients through its parameters—and potentially through its encoder input for the ratio regularizer—while optimizing the transition-conditioned improvement.
2. **Alternating best responses.** Train each density to minimize its own proper held-out or training likelihood, with no incentive for the baseline to become worse, and use a carefully controlled encoder objective.
3. **Variational conditional-MI bound.** Derive and optimize an explicit bound whose assumptions, negative distribution, and gradient paths are documented.
4. **Evaluation-only ratio.** Use held-out likelihood improvement over a separately trained state-only baseline as a diagnostic first, without claiming it as the training objective.

The initial recommendation is conservative: begin with option 4 as a metric, then test option 1 or a derived variational design only after the basic distributional IDM succeeds.

### 5.4 Required diagnostics

- Held-out conditional NLL for both $q$ and $\pi$.
- Improvement $\mathbb{E}[\log q-\log\pi]$ on held-out data.
- Transition permutation: pair $z_t$ with an incorrect $z_{t+1}$ and measure degradation.
- Conditioning ablations: remove, mask, or replace $z_{t+1}$ or $\Delta z_t$.
- Prediction sensitivity to transition changes at fixed $z_t$.
- Baseline capacity matching and capacity sweeps.
- Monitoring for degradation of the baseline caused by encoder updates.
- Comparisons across behavior policies with different state-action correlations and action diversity.
- Future-goal shuffling and local-successor permutation as separate tests.
- Previous-action removal and shuffling when $a_{t-1}$ is part of the conditioning grammar.
- Episode-disjoint splits that exclude evaluation episodes from normalization, optimization, and checkpoint selection.
- Goal-only behavior-cloning and matched independent-actor controls.
- Support-stratified evaluation to determine whether goal transfer is confined to near-local intents.
- Actor-disabled planning to separate representation changes from action-head execution gains.

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

### B. Redundant linear action map

$$
s_{t+1}=s_t+B a_t,
$$

where $B$ has a nontrivial null space. For a displacement $d$, all actions satisfying $Ba=d$ are compatible, yielding an affine set rather than a finite collection of modes.

This tests whether a model can represent continuous ambiguity and whether common density families behave poorly when the valid action distribution lies near a lower-dimensional manifold. The behavior policy determines how probability is distributed along the null space and must be reported. For continuous families, include a constrained parameterization that predicts a minimum-norm solution plus null-space coordinates, so full-dimensional support limitations are not attributed to diagonal-covariance artifacts alone.

### C. Controlled ambiguity variants

1. **Action saturation:** multiple command magnitudes map to the same clipped effect.
2. **Partial observability:** observations alias distinct latent states; add history to determine whether ambiguity is resolvable.
3. **Exogenous variables:** include dynamics irrelevant to action prediction but relevant to future prediction or planning.
4. **State-dependent action effects:** test endpoint, displacement-only, and state-plus-displacement conditioning.
5. **Behavior-policy diversity:** vary action entropy, state-action correlation, multimodal policy structure, and mixtures of policies.

## 9. Baselines

The minimum comparison set is:

- deterministic endpoint-conditioned MSE IDM;
- displacement-conditioned MSE IDM (Delta-JEPA-style, pending source verification);
- state-only action predictor $\pi(a\mid z_t)$;
- categorical/discretized distributional IDM;
- fixed scalar-variance conditional Gaussian IDM;
- heteroscedastic diagonal-Gaussian IDM trained with ordinary Gaussian NLL;
- PRISM-style beta-NLL as an objective ablation for the heteroscedastic Gaussian;
- MDN distributional IDM;
- best-of-$K$ or winner-takes-all baseline;
- history-conditioned IDM;
- latent distribution regularization such as SIGReg-style coverage (exact method pending source verification);
- hybrid distributional IDM plus latent coverage regularization;
- frozen-encoder probabilistic-head control to isolate decoder-local gains from representation changes;
- generic auxiliary-task control: an equally parameterized head predicting a non-action target, such as temporal distance or random features, at matched weight and schedule, to separate auxiliary-supervision effects from action-content effects;
- INTACT-style local-only, goal-only, and paired local/goal diagonal-Gaussian actors;
- fully shared, condition-token shared, shared-trunk/separate-output, parameter-matched independent, and independent-capacity actor controls;
- paired local/goal MDN or discretized multimodal control after the transition-only ambiguity gate;
- Pure CEM on action-supervised checkpoints as an execution-actor-disabled representation control;
- one higher-capacity flow, diffusion, or EBM only after simple models pass the decision gates;
- PRISM-style planner proposal only in downstream MPC experiments, not as a Phase 0 substitute for density-quality evaluation.

Where possible, compare decoder capacities and parameter counts to reduce the chance that gains are attributed solely to a larger auxiliary model. Report frozen versus end-to-end encoder training explicitly.

## 10. Measurements

### 10.1 Conditional-density quality

- Held-out conditional negative log-likelihood when tractable.
- Calibration appropriate to the action representation.
- For continuous Gaussian outputs: marginal interval coverage, joint region or action-chunk coverage where applicable, sharpness, and probability-integral-transform or rank calibration when valid.
- Held-out likelihood improvement over $\pi(a\mid z_t)$.
- Mode coverage: fraction of valid modes represented.
- Mode precision: fraction of predicted mass or samples that correspond to valid modes.
- Probability-mass calibration across modes.
- Ability to distinguish conditionals with identical means but different modal structure.
- Invalid probability mass between known valid modes.
- Local-to-goal intent support overlap and performance by overlap quantile.
- Action-law agreement for matched local physical and deployment-goal intents.
- Validity of Direct mean, Direct mode, and sampled actions reported separately.

For models without tractable normalized likelihood, use clearly labeled alternative proper scores or sample-based metrics; do not compare surrogate training losses as if they were commensurate NLLs.

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

- Planning or model-predictive-control performance.
- Success or return versus planner candidate count.
- Candidate count and world-model evaluations required to reach a fixed success threshold.
- Planning wall-clock time, density-head overhead, and memory where materially different.
- Explicit proposal ablations: vanilla planner; learned-mean warm start with the planner's default variance; product-of-Gaussians fusion with a fitted global prior variance; and product-of-Gaussians fusion with learned state-dependent variance.
- Policy learning sample efficiency where appropriate.
- Goal-reaching or trajectory-quality metrics.
- Direct mean, mode-aware Direct, calibrated-sample, Guarded/local-verification, and Pure-CEM execution results.
- Actor-disabled planning delta for the same action-supervised checkpoint.
- Robustness and no-regression checks on tasks with an effectively deterministic inverse map.

Planning experiments must separate representation quality from action-interface quality and proposal quality. Cross each surviving representation-training objective with vanilla planning and, where practical, a PRISM-style mean-and-variance proposal. A planning gain that appears only with the learned proposal should not be attributed to representation regularization.

### 10.5 Shortcut diagnostics

- Conditional versus state-only held-out likelihood.
- Transition permutation or shuffling tests.
- Fixed-state transition sensitivity.
- Conditioning-input ablations.
- Performance under shifts in the behavior policy.
- Goal-shuffle and previous-action-shuffle degradation.
- Episode-disjoint anti-retrieval performance.
- Performance stratified by local/goal support overlap.
- Shared-versus-independent actor differences under matched parameter budgets.

## 11. Ablations

At minimum, vary:

- density family;
- deterministic mean, fixed isotropic scalar variance, learned heteroscedastic variance, and multimodal density;
- ordinary Gaussian NLL versus PRISM-style beta-NLL for the unimodal Gaussian baseline;
- number of bins, mixture components, or samples $K$;
- $\lambda_{\mathrm{IDM}}$ and $\lambda_{\mathrm{coverage}}$;
- endpoint, displacement-only, state-plus-displacement, history, local physical intent, and detached future-goal intent conditioning;
- local-only, goal-only, and paired local/goal action losses;
- fully shared, condition-token shared, shared-trunk/separate-output, parameter-matched independent, and independent-capacity actor topologies;
- physical-successor attachment, future-goal detachment, joint goal gradients, and both-targets-detached routing;
- previous action present, absent, and shuffled;
- full future goals versus intermediate future waypoints;
- Direct mean, Direct mode, Direct sample, Guarded/local verification, and Pure CEM;
- frozen versus end-to-end heads initialized from the same world-model checkpoint, with encoder, predictor, and other trainable components explicitly held constant or enumerated;
- state-only baseline capacity, fitting schedule, and freezing strategy;
- behavior-policy diversity and state-action correlation;
- observation noise and partial observability;
- exogenous-state capacity and task relevance;
- forward-cycle metric using a known versus learned forward model;
- vanilla planner, mean-only warm start, and PRISM-style mean-and-variance proposal in downstream planning;
- planner candidate count, iteration count, covariance rule, and action-prior scale.

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

1. Implement deterministic synthetic environment generators with analytically known ambiguity.
2. Establish fixed train/validation/test data protocols with policy-diversity controls.
3. Implement deterministic MSE, state-only, discretized, fixed scalar-variance Gaussian, heteroscedastic Gaussian, and small MDN baselines.
4. Implement density, calibration, mode, cycle-consistency, collapse, and probe metrics.
5. Reproduce Gate 1 before integrating the regularizer into a world model.
6. Add a minimal predictive encoder/world model and compare no-IDM, deterministic-IDM, distributional-IDM, coverage-only, and hybrid objectives, including a frozen-encoder probabilistic-head control.
7. After the transition-only representation gate, add INTACT-style local-only, goal-only, and paired local/goal Gaussian controls with explicit gradient routing and actor-sharing ablations.
8. Cross the best paired operator with a simple multimodal density, then compare mean, mode, sample, Guarded, and Pure-CEM execution while retaining actor-disabled representation tests.
9. Treat the likelihood-ratio/CMI-inspired quantity as an evaluation diagnostic before making it a training objective.
10. Advance through the remaining decision gates before increasing model or environment complexity.
11. Add PRISM-style proposal guidance only when downstream MPC experiments begin, and cross it with the surviving representation objectives to isolate planner-side gains.

## 14. Questions requiring user review before implementation

1. Which base world-model or JEPA-like architecture should be the first integration target?
   - **Recommended default:** A minimal online/target predictive encoder that preserves the essential JEPA gradient structure without introducing benchmark-scale architectural complexity.
2. Should Phase 0 begin with purely generated datasets, online rollouts, or both?
   - **Recommended default:** Begin with fixed generated datasets for reproducibility and exact control of conditional action laws. Add online rollouts only after the core invalid-mean and mode-capture results are stable.
3. What action dimensionality is sufficient for the first redundant-map experiment?
   - **Recommended default:** Start with $d_a=2$, $d_s=1$, rank $1$, and nullity $1$ so the compatible action set is directly visualizable. Follow with a higher-dimensional rank/nullity sweep after the basic case passes.
4. Which broad latent coverage regularizer should represent the SIGReg/LeJEPA-style baseline after source verification?
   - **Recommended default:** Use one verified, simple variance/covariance or spectral anti-collapse regularizer. Avoid combining multiple coverage mechanisms in the first comparison.
5. Should the first CMI-inspired experiment be evaluation-only, frozen-baseline training, or a derived variational bound?
   - **Recommended default:** Begin with an evaluation-only held-out likelihood difference between separately and properly fitted transition-conditioned and state-only models. Test frozen or alternating training only if the diagnostic reliably detects permutation and policy shortcuts.
6. What constitutes the first downstream task: representation probes, planning with known dynamics, or learned-model control?
   - **Recommended default:** Use a staged ladder: frozen representation probes first, planning with known dynamics second, and learned-model control only after the simpler evaluations show a benefit.
7. Should the learned inverse density remain training-only, or should it also be evaluated as a planner proposal or realizability signal?
   - **Recommended default:** Establish the training-only representation effect first. Evaluate Direct execution, realizability filtering, and planner-proposal use later as separately labeled interventions.
8. Should the first paired local/goal operator use full future goals, intermediate waypoints, or both?
   - **Recommended default:** Use controlled temporal offsets that include one-step or short waypoints and full future goals. Report results separately by goal horizon and support-overlap quantile.
9. Should previous-action conditioning be included in the first controlled INTACT-style comparison?
   - **Recommended default:** Make the model without $a_{t-1}$ the primary controlled condition. Include present, absent, shuffled, and previous-action-only variants to distinguish legitimate temporal information from policy persistence.
10. Which mode-aware Direct rule should be primary for an MDN or discretized density?
    - **Recommended default:** Use a predeclared deterministic mode-aware rule: the highest-probability bin for a discretized model and either the numerical mixture mode or a component representative selected by peak density for an MDN. Report the conditional mean, calibrated sampling, and limited forward verification as separate arms.
11. Which actor-sharing and gradient-routing variants are mandatory before scaling?
    - **Recommended default:** Require the fully shared default route, local-only, goal-only, parameter-matched independent actors, an independent-capacity control, future-goal-attached routing, both-targets-detached routing, and a frozen encoder/predictor head-only control. Add partial-sharing variants if fully shared and independent actors differ meaningfully.
12. Should local/goal support overlap be a diagnostic only or a decision-gate threshold?
    - **Recommended default:** Initially require support-overlap stratification rather than a single hard threshold. Any advancing result must report performance by overlap quantile, goal horizon, demonstrated versus recombined goals, and an explicitly extrapolative split.
13. Which literature claims and named methods should be included after a primary-source review?
    - **Recommended default:** Prioritize the latent-coverage baseline, deterministic displacement-conditioned IDM, multimodal inverse-dynamics and action-conditioned representation work, and goal-conditioned or hindsight behavior-cloning precedents. Review higher-capacity action densities only after a simple density family reaches a diagnosed limitation.
14. Which pre-registered primary endpoint, seed count, and equivalence margins should govern phase advancement?
    - **Recommended default:** Adopt the decisive-core endpoint defined in the experiment plan unless review changes it.

## Acceptable outcomes including null results

The most likely single outcome at Gate 2 is a null or small representation effect, because every proper action-prediction objective creates encoder incentives in the same direction and differs mainly in magnitude. Treat that outcome as publishable rather than failed. The deliverable would be the synthetic environment suite, the metric protocol, and a controlled analysis of when decoder expressivity does and does not shape world-model representations. Define the primary endpoint before implementation so a null result is interpretable rather than inconclusive.

## 15. Current project boundary

This repository should remain a planning and scaffold artifact until the user reviews this handoff. No expensive jobs, dataset downloads, dependency installation, substantive experiment implementation, or unsupported literature claims should be added before approval.
