# Project Brief: Distributional Inverse-Dynamics Regularization for Action-Ambiguous World Models

## Purpose of this document

This document is a research handoff for investigating whether a **distributional inverse-dynamics objective** can improve the representations learned by world models and JEPA-like predictive encoders when observed transitions are compatible with multiple actions.

The project is intentionally at the planning and scaffold stage. The immediate goal is not to assume the proposal works, but to establish a falsification-first program that separates three issues that are often blurred together:

1. representation collapse and broad latent coverage;
2. action-related representation content;
3. ambiguity or multimodality in the inverse map from transitions to actions.

Paper-specific terminology, attributions, and claims about prior results—especially those associated with SIGReg, LeJEPA, Delta-JEPA, and related methods—must be verified against primary sources before publication or strong comparative claims. The conceptual distinctions and mathematical arguments below do not depend on those unverified historical claims.

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

The central caveat is equally important: a distributional IDM can represent uncertainty or non-identifiability honestly, but it cannot recover action information that is absent from the observations. If the same observed transition is genuinely produced by either $+1$ or $-1$, the correct model may be a 50/50 conditional distribution; no decoder can identify which action occurred without additional information.

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

## 6. Candidate conditional model families

The project should increase model complexity only when simpler models establish the phenomenon.

1. **Discretized continuous-action bins with cross-entropy.** Easy to inspect and calibrate; resolution grows poorly with action dimension and introduces binning error.
2. **Fixed scalar-variance conditional Gaussian.** An equivalence and implementation sanity check: with one fixed isotropic scalar variance, Gaussian NLL is globally scaled MSE plus a constant, so it has the same optimum and gradient direction unless the action support is transformed. A fixed unequal diagonal covariance instead corresponds to dimension-weighted MSE and must be paired with that weighted-MSE control.
3. **Heteroscedastic diagonal Gaussian.** Predicts both mean and state-dependent variance. This tests whether uncertainty alone is useful before adding multimodal expressivity, but it can cover separated modes by inflating variance and assigning mass to invalid actions.
4. **Mixture density networks (MDNs).** A practical first continuous baseline with explicit modes; vulnerable to component collapse, variance pathologies, and sensitivity to the number of components.
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

The proposed contribution must be framed and evaluated as the **use of a properly trained distributional IDM as a representation regularizer for world models or JEPA-like systems**, with explicit treatment of action ambiguity, shortcut-resistant variants, controlled benchmarks, and downstream representation outcomes. The exact novelty relative to prior work requires a dedicated literature review using primary sources.

### 7.6 Adjacent work: PRISM

[PRISM: PRior-guided Imagination Sampling in world Models](https://arxiv.org/abs/2606.07974) is relevant adjacent work, but it intervenes at a different stage. PRISM freezes a JEPA world-model encoder and predictor, then trains a small goal-conditioned head that models a diagonal-Gaussian distribution over future action chunks:

$$
p_\phi(a_{t:t+HB}\mid z_t,z_g).
$$

At deployment, the learned mean and variance are fused with the proposal distribution of MPPI or CEM to improve candidate sampling. PRISM therefore studies a probabilistic planner proposal on fixed representations. This project studies a transition-conditioned inverse density whose gradients are intended to shape the representation during world-model training:

$$
q_\phi(a_t\mid z_t,z_{t+1}).
$$

PRISM narrows broad novelty claims about probabilistic action heads, state-dependent action uncertainty, and uncertainty-aware action sampling in JEPA world models. It does not establish that a distributional inverse objective improves representation learning under action ambiguity.

PRISM also illustrates why planning success is not sufficient evidence of density correctness. Its unimodal Gaussian can assign probability between distinct behavior modes, while MPPI can still recover useful actions by evaluating and refining sampled candidates with the world model. This project must therefore retain direct calibration, mode-coverage, mode-precision, and known-forward cycle metrics in addition to planning outcomes.

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

This tests whether a model can represent continuous ambiguity and whether common density families behave poorly when the valid action distribution lies near a lower-dimensional manifold. The behavior policy determines how probability is distributed along the null space and must be reported.

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
- Robustness and no-regression checks on tasks with an effectively deterministic inverse map.

Planning experiments must separate representation quality from proposal quality. Cross each surviving representation-training objective with vanilla planning and, where practical, a PRISM-style mean-and-variance proposal. A planning gain that appears only with the learned proposal should not be attributed to representation regularization.

### 10.5 Shortcut diagnostics

- Conditional versus state-only held-out likelihood.
- Transition permutation or shuffling tests.
- Fixed-state transition sensitivity.
- Conditioning-input ablations.
- Performance under shifts in the behavior policy.

## 11. Ablations

At minimum, vary:

- density family;
- deterministic mean, fixed isotropic scalar variance, learned heteroscedastic variance, and multimodal density;
- ordinary Gaussian NLL versus PRISM-style beta-NLL for the unimodal Gaussian baseline;
- number of bins, mixture components, or samples $K$;
- $\lambda_{\mathrm{IDM}}$ and $\lambda_{\mathrm{coverage}}$;
- endpoint, displacement-only, state-plus-displacement, and history conditioning;
- frozen versus end-to-end heads initialized from the same world-model checkpoint, with encoder, predictor, and other trainable components explicitly held constant or enumerated;
- state-only baseline capacity, fitting schedule, and freezing strategy;
- behavior-policy diversity and state-action correlation;
- observation noise and partial observability;
- exogenous-state capacity and task relevance;
- forward-cycle metric using a known versus learned forward model;
- vanilla planner, mean-only warm start, and PRISM-style mean-and-variance proposal in downstream planning;
- planner candidate count, iteration count, covariance rule, and action-prior scale.

## 12. Decision gates

### Gate 1: verify the basic failure and remedy

Proceed only if deterministic MSE fails in the predicted way on controlled ambiguity and a simple discretized model or MDN captures the valid modes with better calibration and cycle consistency.

**Falsifier:** deterministic MSE does not exhibit the expected invalid-mean behavior under the designed data, or the simple distributional model cannot reliably capture modes despite adequate optimization and capacity.

### Gate 2: establish representation benefit

Proceed only if distributional-IDM regularization improves representation or downstream metrics, not merely action likelihood.

**Falsifier:** the density model predicts actions better but encoder probes, prediction, planning, and control do not improve—or exogenous information degrades enough to erase any benefit.

### Gate 3: test shortcut resistance

Proceed only if the CMI-inspired or likelihood-improvement variant reduces reliance on state-policy shortcuts without destabilizing training or gaming the baseline.

**Falsifier:** held-out transition-conditioned improvement is illusory, disappears under policy shift or permutation tests, or comes from making the baseline worse.

### Gate 4: justify model complexity

Add a conditional flow, diffusion model, or EBM—and only then a larger environment—if simpler categorical or MDN models reveal a clear limitation that added expressivity is likely to address.

**Falsifier:** simple models already saturate the relevant metrics, or higher-capacity models improve density fit without representation or downstream gains.

## 13. Recommended initial implementation order after approval

1. Implement deterministic synthetic environment generators with analytically known ambiguity.
2. Establish fixed train/validation/test data protocols with policy-diversity controls.
3. Implement deterministic MSE, state-only, discretized, fixed scalar-variance Gaussian, heteroscedastic Gaussian, and small MDN baselines.
4. Implement density, calibration, mode, cycle-consistency, collapse, and probe metrics.
5. Reproduce Gate 1 before integrating the regularizer into a world model.
6. Add a minimal predictive encoder/world model and compare no-IDM, deterministic-IDM, distributional-IDM, coverage-only, and hybrid objectives, including a frozen-encoder probabilistic-head control.
7. Treat the likelihood-ratio/CMI-inspired quantity as an evaluation diagnostic before making it a training objective.
8. Advance through the remaining decision gates before increasing model or environment complexity.
9. Add PRISM-style proposal guidance only when downstream MPC experiments begin, and cross it with the surviving representation objectives to isolate planner-side gains.

## 14. Questions requiring user review before implementation

1. Which base world-model or JEPA-like architecture should be the first integration target?
2. Should Phase 0 begin with purely generated datasets, online rollouts, or both?
3. What action dimensionality is sufficient for the first redundant-map experiment?
4. Which broad latent coverage regularizer should represent the SIGReg/LeJEPA-style baseline after source verification?
5. Should the first CMI-inspired experiment be evaluation-only, frozen-baseline training, or a derived variational bound?
6. What constitutes the first downstream task: representation probes, planning with known dynamics, or learned-model control?
7. Should the learned inverse density remain training-only, or should it also be evaluated as a planner proposal or realizability signal?
8. Which literature claims and named methods should be included after a primary-source review?

## 15. Current project boundary

This repository should remain a planning and scaffold artifact until the user reviews this handoff. No expensive jobs, dataset downloads, dependency installation, substantive experiment implementation, or unsupported literature claims should be added before approval.
