# Experiment Plan

## Status and operating principle

This is a planning document only. No experiments have been implemented.

The program is **falsification-first**: begin with analytically understandable environments and simple density families; advance only when predefined decision gates are passed. INTACT, PRISM, Delta-JEPA, and LeJEPA/SIGReg specifications are verified against primary sources; LeWorldModel and PLDM require full-text review before exact implementation.

## Pre-registration requirements

Before any Phase 1 run, record the following in `results/preregistration.md`: the primary composite endpoint, the seed count, the equivalence margins, and the multiplicity correction. A phase advances only when its pre-registered endpoint clears the corrected threshold. Secondary metrics generate hypotheses; they never justify advancement on their own.

Recommended defaults, subject to review:

- Primary endpoint: the mean of controllable-state probe accuracy, known-forward cycle validity of sampled actions, and one small planning success rate, each normalized against the no-IDM arm of the same checkpoint.
- Seeds: at least five per cell; report seed-level intervals.
- Multiplicity: Holm correction across the pre-declared endpoint components and the cells of the decisive core.
- Equivalence margin for H8: declare a numeric band before the one-to-one control runs.

## Decisive core

The decisive experiment is Phase 1A restricted to environments E1 and E10, four IDM arms (none, deterministic, heteroscedastic Gaussian, MDN), frozen versus end-to-end encoder training, and the seed count above. Phases 0B through 0C, 1B, 2, 3, 3P, and 4 are contingent branches. Each starts only if the decisive core produces a representation effect worth pursuing or a diagnostic failure that redirects the design. Phase 0A remains an implementation check for generators, metrics, and training loops.

## Phase overview

| Phase | Goal | Environments | Models | Exit condition |
|---|---|---|---|---|
| 0A | Demonstrate invalid conditional means | Quadratic ambiguity | Endpoint MSE, displacement MSE, state-only, discretized, MDN | Simple density model captures both modes and beats MSE on modal/cycle metrics |
| 0B | Test continuous action non-identifiability | Rank-deficient linear action map | Same, plus best-of-K | Model represents policy-weighted action sets or samples valid actions |
| 0C | Separate ambiguity sources | Saturation, partial observability, exogenous state, state-dependent effects, policy variants | Add history and conditioning ablations | Failure signatures match the intended ambiguity mechanism |
| 1A | Test transition-only representation regularization | Minimal predictive encoder/world model on Phase 0 environments | No IDM, deterministic IDM, distributional IDM, coverage-only, hybrid | Representation or downstream gain beyond density fit |
| 1B | Test INTACT-style local/goal sharing and routing | Goal-support and multimodal-junction variants | Local-only, goal-only, shared and independent actors, Gaussian and simple multimodal densities | Sharing, routing, or multimodality adds value beyond behavior cloning and actor capacity |
| 2 | Test shortcut-resistant objective | Policy-correlated variants | State-only baseline, evaluation ratio, then controlled training variant | Reduced shortcut use without baseline gaming or instability |
| 3 | Test justified added capacity | Only environments where simple models are limiting | One of flow, diffusion/flow matching, or EBM | Added capacity improves relevant representation/downstream outcomes |
| 3P | Separate representation and proposal gains | First approved MPC environment | Vanilla planner, mean-only warm start, PRISM-style mean-and-variance proposal crossed with surviving representations | Planner-side and representation-side effects are identifiable |
| 4 | Scale cautiously | One larger controlled environment selected after review | Best surviving objectives | Synthetic conclusions reproduce without obscuring diagnostics |

## Environment matrix

| ID | Transition structure | Ambiguity source | Controlled variables | Main purpose | Expected failure signature |
|---|---|---|---|---|---|
| E1 | $s'=s+a^2$ | Symmetry, two modes | mode balance, magnitude, observation noise | Canonical invalid-mean test | MSE predicts near zero for symmetric $\pm a$; poor cycle consistency |
| E2 | Quadratic with asymmetric policy | Same environment, policy-weighted modes | probability of positive/negative mode | Test calibrated modal probabilities | Density should match weights; MSE tracks weighted mean, often between modes |
| E3 | Constructed conditionals sharing a mean | Different modal structure | spacing, weights, noise | Test information lost by mean prediction | Deterministic models cannot distinguish conditionals |
| E4 | $s'=s+Ba$, rank-deficient $B$ | Null-space redundancy | action dimension, rank, nullity, policy on null space | Continuous compatible-action set | Point estimator chooses one summary; density may struggle with manifold-like support |
| E5 | Clipped/saturated actuator | Many commands, same effect | clipping threshold, command distribution | Plateau ambiguity | MSE predicts interior command; density should represent saturated range/policy weighting |
| E6 | Partially observed state | State aliasing | observation map, history length | Test resolvable uncertainty | Endpoint model multimodal; sufficient history reduces ambiguity |
| E7 | Structural ambiguity plus history | Non-injective action effect | history length and policy | Negative control for history | History must not recover absent action identity |
| E8 | Exogenous dynamic variable | Uncontrollable but predictive state | exogenous strength, encoder capacity | Test controllability gap | IDM-heavy representation drops exogenous factor; coverage/hybrid may retain it |
| E9 | State-dependent action effect | Inverse depends on state | dynamics family, state distribution | Test conditioning choice | Displacement-only decoder underperforms endpoint/state-plus-delta |
| E10 | One-to-one inverse control | No intended ambiguity | noise, policy diversity | No-regression benchmark | Distributional method should match deterministic baseline without spurious modes |
| E11 | Policy shortcut variant | Strong $s$-to-$a$ correlation | policy entropy, mixtures, shifts, previous-action cues | Test transition and goal use | Endpoint or goal model remains strong after condition shuffling; ratio exposes low added information |
| E12 | Goal-support overlap continuum | Future goals vary from locally supported to extrapolative | goal horizon, waypoint spacing, local-intent density | Test local-to-goal support claims | Direct performance and calibration degrade as goal intents leave supported regions |
| E13 | Multimodal goal junction | Same current/goal condition admits distinct valid routes or contacts | route balance, obstacle geometry, behavior mixture | Test goal-conditioned invalid means | Gaussian mean lies between valid modes; mode-aware multimodal execution remains valid |

## Baseline matrix

| Baseline | Output/objective | Question answered | Key caveat |
|---|---|---|---|
| Deterministic endpoint IDM | $g(z_t,z_{t+1})$, MSE | Does standard regression average modes? | Can exploit state-policy shortcuts |
| Deterministic displacement IDM | $g(z_{t+1}-z_t)$, MSE | Does displacement conditioning reduce shortcuts? | Still predicts a mean; may be ill-posed under state dependence |
| State-only action model | $\pi(a_t\mid z_t)$ | How much action prediction comes from policy correlation? | Must be capacity-matched and properly fitted |
| Discretized distributional IDM | bins plus cross-entropy | Can a simple normalized model capture modes? | Binning error; scales poorly with dimension |
| Fixed scalar-variance Gaussian IDM | conditional Gaussian NLL with one fixed isotropic scalar variance | Does the implementation reproduce the expected MSE equivalence? | Globally scaled MSE plus a constant; unequal fixed diagonal variance instead requires a weighted-MSE control |
| Heteroscedastic Gaussian IDM | conditional Gaussian NLL with learned diagonal variance | Is state-dependent uncertainty sufficient? | Can cover separated modes by inflating variance over invalid actions |
| PRISM-style beta-NLL Gaussian | $\operatorname{sg}((\sigma^2)^{0.5})$ times elementwise diagonal-Gaussian NLL on normalized actions | Does PRISM's detached beta-NLL weighting change optimization or uncertainty usefulness? | Preserve $\beta=0.5$, reduction, action normalization, and standard-deviation floor; evaluate with ordinary held-out NLL |
| Mixture density network | conditional mixture likelihood | Can a simple continuous density capture modes? | Component collapse and variance instability |
| Best-of-K / winner-takes-all | multiple candidates, min/best loss | Does multi-hypothesis prediction suffice? | Usually not a calibrated normalized density |
| History-conditioned IDM | $q(a_t\mid z_{t-k:t+1})$ | Is ambiguity resolvable with temporal context? | Extra capacity and policy cues can confound results |
| Latent coverage regularizer | variance/rank/geometry term | Is broad anti-collapse pressure enough? | Nuisance factors can satisfy it; exact named method pending source review |
| Hybrid | distributional IDM plus coverage | Are semantic action pressure and broad coverage complementary? | Requires weight and gradient-conflict ablations |
| Frozen-encoder probabilistic head | same density head with no encoder updates | Are gains decoder-local or representational? | Can improve action prediction without changing representation quality |
| Generic auxiliary-task head | equally parameterized head predicting a non-action target such as temporal distance or random features, at matched weight | Does any auxiliary supervision help, independent of action content? | Must match parameter count, weight, and schedule with the IDM arms |
| INTACT-style local-only Gaussian | $p(a_t\mid z_t,z_{t+1}-z_t,a_{t-1})$ | What does physical inverse likelihood contribute? | No deployment-goal support |
| INTACT-style goal-only Gaussian | $p(a_t\mid z_t,\operatorname{sg}(z_g)-z_t,a_{t-1})$ | Is performance goal-conditioned behavior cloning? | Can exploit policy, retrieval, or previous-action shortcuts |
| Fully shared local/goal Gaussian | one diagonal-Gaussian actor for both intent families | Does shared action semantics help? | Unimodal mean can be invalid between modes |
| Fully shared local/goal MDN | one mixture actor for both intent families | Does explicit multimodality add value within the shared operator? | Component collapse and mode-selection rules require care |
| Independent local/goal actors | parameter-matched and capacity-control variants | Is sharing useful beyond parameter count? | Input grammar and optimization budgets must match |
| Alternative sharing topology | condition-token shared or shared trunk with separate outputs | Which parameters actually need to be common? | Partial sharing changes gradient interaction and capacity |
| Gradient-routing control | goal attached, both targets detached, or encoder frozen | Does asymmetric routing matter? | Must enumerate every latent occurrence and trainable component |
| Pure CEM on action-supervised checkpoint | execution actor disabled | Did action losses improve the representation or forward model? | Search budget and terminal cost must be matched |
| Higher-capacity density | flow, diffusion/flow matching, or EBM | Are simple-family limitations blocking progress? | Allowed only after the simple-model gates; fairness and compute are harder |
| PRISM-style planner proposal | $p(a_{t:t+HB-1}\mid z_t,z_g)$ fused with MPC proposal | Does a learned inference-time proposal improve planning independently of representation training? | Downstream planning baseline only; not an inverse-density or Phase 0 substitute |

## Conditioning matrix

Every applicable decoder should be compared under:

1. endpoint: $(z_t,z_{t+1})$;
2. displacement only: $z_{t+1}-z_t$;
3. state plus displacement: $(z_t,z_{t+1}-z_t)$;
4. local physical intent: $(z_t,z_{t+1}-z_t,a_{t-1})$, with the physical successor attached by default;
5. detached future-goal intent: $(z_t,\operatorname{sg}(z_g)-z_t,a_{t-1})$;
6. paired local/goal calls using an identical input grammar;
7. history: $z_{t-k:t+1}$, where relevant;
8. state only: $z_t$, as the behavior-policy baseline.

Within the paired study, vary previous action as present, absent, or shuffled; compare full future goals with intermediate future waypoints; and record each condition's local/goal support-overlap bucket. A fully shared claim requires the same grammar, trunk, output parameters, density family, and action transform for both calls. For every condition, record parameter count, receptive information, exact encoder-gradient paths, and every stop-gradient occurrence.

## Metric matrix

| Category | Metric | Interpretation | Important limitation |
|---|---|---|---|
| Density | Held-out conditional NLL | Proper fit when normalized likelihood is available | Not directly available for all EBMs/diffusion objectives |
| Density | Held-out $\mathbb{E}[\log q-\log\pi]$ | Added predictive information from transition | Not automatically a CMI bound with fitted approximations |
| Calibration | Bin/mode probability calibration | Whether predicted probabilities match frequencies | Requires careful grouping and sufficient samples |
| Calibration | Marginal interval coverage and sharpness | Whether continuous per-coordinate uncertainty is calibrated and concentrated | Marginal coverage does not establish joint action validity |
| Calibration | Joint region or action-chunk coverage | Whether multivariate predictions cover the realized action object | Requires a clearly defined region and enough held-out samples |
| Calibration | PIT or rank calibration | Distributional calibration for continuous outputs where applicable | Interpretation depends on dimensionality and conditional grouping |
| Modes | Mode coverage | Fraction of valid modes represented | Needs environment-specific mode definition |
| Modes | Mode precision | Fraction of predictions/mass near valid modes | Threshold-dependent |
| Structure | Same-mean discrimination | Detect modal differences hidden by the mean | Synthetic/controlled diagnostic |
| Cycle | Known-forward transition error | Whether sampled actions reproduce target transitions | Validity metric, not proof of action identity |
| Cycle | Learned-forward transition error | Approximate cycle consistency without known dynamics | Learned verifier can share errors or extrapolate poorly |
| Collapse | Per-dimension variance, covariance spectrum, effective rank | Broad representation health | High variance/rank can still be nuisance-driven |
| Probes | Controllable-state linear/nonlinear probes | Action-relevant representation content | Probe capacity affects conclusions |
| Probes | Exogenous-state linear/nonlinear probes | Retention of uncontrollable task-relevant content | Must control encoder capacity |
| Prediction | Forward prediction error | World-model quality | May not correlate with planning under distribution shift |
| Downstream | Planning/control return or success | Practical utility | Higher variance and environment cost |
| Planning | Success or return versus candidate count | Proposal sample efficiency | Must match planner iterations and model-evaluation budgets |
| Planning | World-model evaluations per success | Computational efficiency of planning | Sensitive to success threshold and environment difficulty |
| Planning | Wall-clock and proposal-head overhead | Whether sampling gains translate to practical efficiency | Hardware and implementation dependent |
| Planning | Mean-only warm start versus global-variance and heteroscedastic products | Operational value of learned uncertainty and fusion | Each arm must define its mean, covariance, and fusion rule |
| Shortcut | Transition and goal permutation degradation | Whether the decoder uses the supplied physical or deployment condition | Permutation distribution must remain meaningful |
| Shortcut | Fixed-state transition sensitivity | Whether conditional changes with transition | Sensitivity alone does not establish correctness |
| Support | Local-to-goal intent overlap | Whether deployment conditions resemble trained local or demonstrated-goal conditions | Metric choice is representation-dependent and does not prove semantic equivalence |
| Support | Performance by overlap quantile | Whether success is confined to supported intents | Requires a predeclared overlap estimator |
| Action law | Agreement on matched local/goal intents | Whether paired conditions induce compatible action densities | Agreement can be wrong if both branches learn a shortcut |
| Modes | Invalid between-mode mass | Probability assigned away from known valid modes | Requires analytically or operationally defined valid regions |
| Execution | Mean, mode, and sample validity | Consequence of the action-selection rule | Must not aggregate rules into one score |
| Representation | Actor-disabled planning delta | Whether action-supervised training improved representation or prediction | Pure-CEM budget and terminal cost must match |
| Shortcut | Goal and previous-action shuffle degradation | Reliance on goal and short-history inputs | Shuffles can create unrealistic combinations |
| Shortcut | Episode-disjoint anti-retrieval performance | Whether results survive strict held-out episodes and normalization | Does not rule out all policy memorization |
| Optimization | Local/goal gradient cosine and shared-parameter fraction | Interaction between paired losses | Cosine is meaningless without reporting which parameters are shared |
| Robustness | One-to-one no-regression delta | Cost on ordinary inverse dynamics | Must match compute and tuning budgets |

## Phase 0A: quadratic ambiguity

### Data conditions

- Symmetric $+a/-a$ with equal probability.
- Asymmetric mode probabilities.
- Multiple magnitudes.
- Conditionals constructed to share a mean but differ in modal structure.
- Observation-noise sweep.
- One-to-one control variant, such as a restricted nonnegative action range.

### Required models

- endpoint MSE;
- displacement MSE;
- state-only predictor;
- discretized categorical IDM;
- fixed scalar-variance Gaussian IDM;
- heteroscedastic diagonal-Gaussian IDM;
- small MDN;
- optional PRISM-style beta-NLL Gaussian objective ablation;
- optional best-of-K after the core comparison.

### Pass criteria

These criteria follow from standard results, so treat this phase as an implementation check whose outputs are working generators, metrics, and training loops.

- MSE exhibits the predicted conditional-mean behavior.
- At least one simple distributional model represents both valid modes.
- Density samples improve valid-mode and known-forward cycle metrics.
- Results hold on held-out transitions and are not an artifact of training-set memorization.

### Expected failure signatures

- **MSE invalid average:** low action MSE relative to other point predictions but poor transition validity.
- **Mode collapse:** MDN assigns nearly all mass to one symmetric mode.
- **Variance inflation:** MDN covers modes using one broad component that places substantial mass on invalid actions.
- **Bin aliasing:** coarse discretization merges modes or assigns mass between them.
- **Conditioning collapse:** distributional model predicts nearly identical densities after transition permutation.

## Phase 0B: redundant action map

### Construction

Choose $B\in\mathbb{R}^{d_s\times d_a}$ with $\operatorname{rank}(B)<d_a$. Generate actions from explicitly documented policies that control probability along the null space. For continuous families, include a constrained parameterization that predicts a minimum-norm solution plus null-space coordinates, so full-dimensional support limitations are not attributed to diagonal-covariance artifacts alone.

### Required sweeps

- action dimension $d_a$;
- rank and nullity of $B$;
- bounded versus approximately unbounded policy support;
- unimodal versus multimodal policy distribution along the null space;
- observation noise.

### Pass criteria

- Samples satisfy $Ba\approx s'-s$.
- The predicted distribution reflects the behavior policy along compatible null directions.
- Evaluation distinguishes transition validity from recovery of the exact logged action.

### Expected failure signatures

- **Manifold mismatch:** full-dimensional density wastes mass away from the compatible affine set.
- **Likelihood/validity mismatch:** good NLL but sampled actions violate the transition due to variance off the manifold.
- **Policy memorization:** model captures null-space policy while ignoring displacement.

## Phase 0C: ambiguity-source isolation

### Saturation

Vary clipping thresholds and command distributions. Determine whether predicted mass covers the compatible command interval rather than only a mean command.

### Partial observability and history

Construct one variant where history reveals the hidden state and one where action identity remains structurally absent. History should reduce uncertainty only in the first.

### Exogenous dynamics

Add an independently moving variable needed for future prediction or planning but irrelevant to action. Measure whether increasing IDM weight removes it from the representation.

### State-dependent effects

Construct dynamics where displacement alone is insufficient. Compare endpoint, delta-only, and state-plus-delta models.

### Policy shortcuts

Vary state-action correlation and evaluate under policy shift. Use state-only likelihood, transition permutation, and fixed-state conditional sensitivity.

## Phase 1A: transition-only representation regularization

### Minimal objective comparison

For the same encoder/world-model backbone, compare:

1. world-model objective only;
2. plus deterministic endpoint IDM;
3. plus deterministic displacement IDM;
4. plus discretized distributional IDM;
5. plus fixed scalar-variance Gaussian IDM;
6. plus heteroscedastic Gaussian IDM;
7. plus MDN distributional IDM;
8. plus latent coverage regularizer;
9. plus distributional IDM and latent coverage;
10. plus history-conditioned IDM where relevant.
11. plus a generic auxiliary-task head with a non-action target at matched weight.

For each probabilistic head, initialize the frozen and end-to-end arms from the same world-model checkpoint and use identical data, head architecture, initialization protocol, optimizer budget, and evaluation splits. In the frozen control, freeze both encoder and world-model predictor and train only the action head. In the end-to-end arm, enumerate every component that receives gradients. Report the regularizer weights and whether gradients reach online encoder, target encoder, predictor, and action decoder.

### Backbone gradient routing

Specify before coding how IDM gradients enter the minimal backbone: whether the action head reads online or target representations, whether $z_{t+1}$ occurrences receive IDM gradients when the forward predictor detaches targets, and which components each loss updates. Record the same enumeration that Phase 1B requires for INTACT-style routing.

### Required outcome

A method advances only if it improves at least one representation/downstream target without unacceptable degradation in:

- exogenous-state retention;
- base forward prediction;
- one-to-one control tasks;
- training stability;
- compute relative to the demonstrated benefit.

## Phase 1B: paired local/goal action laws

Run this phase only after a transition-only objective passes the representation gate. Use future observations from episode-disjoint demonstrations as deployment goals and preserve INTACT's default asymmetric route: current state attached, local physical successor attached, and future-goal occurrence detached.

### Staged factorial

Avoid an uncontrolled full Cartesian product. Run four stages:

1. **Action supervision:** none, local only, goal only, and paired local plus goal using a diagonal Gaussian.
2. **Actor coupling:** fully shared, condition-token shared, shared trunk with separate outputs, parameter-matched independent actors, and an independent-actor capacity control under the best routing.
3. **Density family:** shared and matched-independent Gaussian versus a discretized or MDN multimodal density, holding the conditioning grammar, action transform, data, optimizer budget, and evaluation splits fixed.
4. **Execution:** compare Direct mean, mode-aware Direct, calibrated sampling, Guarded/local verification, and Pure CEM on surviving checkpoints.

### Gradient-routing controls

For every run, record gradients to the online/current encoder, local physical-successor occurrence, future-goal occurrence, target encoder, world-model predictor, previous-action encoder, shared trunk, and any branch-specific output. Required controls are:

- physical successor attached and future goal detached;
- future goal attached;
- both target endpoints detached;
- encoder and predictor frozen;
- local loss only;
- goal loss only.

A latent detached as a goal in one call may still receive gradients in another role elsewhere. Do not describe this as globally freezing the goal encoder.

### Required outcomes

Advance only if at least one matched claim survives:

- paired local plus goal outperforms goal only;
- full sharing outperforms parameter-matched independent actors;
- an action-supervised checkpoint improves probes or Pure-CEM planning with the execution actor disabled;
- an explicit multimodal density improves valid-mode mass, calibration, cycle validity, or downstream behavior beyond the shared Gaussian.

In every advancing case, support-stratified results must also rule out uncontrolled extrapolation as the explanation.

Direct success alone is insufficient if behavior cloning, actor capacity, episode retrieval, local search, or invalid Gaussian means remain plausible explanations.

## Phase 2: shortcut-resistant conditional-information study

### Stage 2.1 — evaluation only

Fit $q(a\mid z_t,z_{t+1})$ and $\pi(a\mid z_t)$ using proper objectives and evaluate their log-likelihood difference on held-out data. Capacity-match where possible and include sweeps.

### Stage 2.2 — frozen/separate baseline

Only after Stage 2.1 works, test a baseline that is fitted separately and frozen or stop-gradient-controlled during encoder regularization. Document exactly which parameters receive each gradient.

### Stage 2.3 — variational design, optional

Derive an explicit conditional-MI bound or other principled contrastive objective before coding it. State assumptions and prove or cite the bound using verified primary sources.

### Abort conditions

- The baseline likelihood degrades as the score rises.
- Transition permutation does not reduce the score.
- Capacity mismatch explains the result.
- Encoder training destabilizes the world-model objective.

## Phase 3: one justified higher-capacity model

Select exactly one based on a diagnosed limitation:

- **normalizing flow** if exact likelihood and smooth full-dimensional support are central;
- **diffusion/flow matching** if disconnected or complex high-dimensional samples are needed and likelihood is secondary;
- **EBM** if flexible support is essential and sampling/training diagnostics are acceptable.

Do not add one solely because it is more expressive. Re-run the same metrics and require representation/downstream improvement, not just a better surrogate loss.

## Direct and locally verified intent-to-action execution

Report action-operator execution separately from PRISM-style planner proposals.

1. **Direct mean:** execute the conditional mean with zero sampled candidates and no terminal-cost search.
2. **Mode-aware Direct:** use one predeclared rule for a multimodal model, either numerical maximization of the full density or a component representative selected by peak density. Report highest-weight-component selection separately because it need not equal the global mixture mode. For a single diagonal Gaussian, mean and mode coincide.
3. **Direct sample:** draw from the learned density using a fixed sample budget and report both success and invalid-action rates.
4. **Guarded/local verification:** initialize limited local CEM around the Direct plan, retain the Direct plan as a reference candidate, and return the globally best candidate rather than only the last-iteration result. Report candidate count, iterations, and initial covariance. A paper-faithful INTACT reference is $128\times3$ candidates with initial standard deviation $0.25$, but this is a reference setting rather than a universal default.
5. **Pure CEM/execution actor disabled:** disable the learned action predictor and search raw actions through the representation and forward model. Match terminal cost, normalization, candidate count, iteration count, and world-model evaluation budget where the comparison requires it.

Broad actor-on CEM is not the same as Guarded execution and must be labeled separately. Also distinguish execution actor disabled from training action objective disabled: the former evaluates an action-supervised checkpoint without its actor, while the latter trains without action-supervision gradients.

## Downstream planning: representation-proposal factorial

PRISM is a planner-side intervention rather than a representation-training objective. INTACT-style Direct, mode-aware Direct, Guarded, and Pure-CEM arms belong to the action-operator execution block above and must not be inserted as proposal levels in this factorial. When the project reaches MPC experiments, run the full Cartesian product of every surviving representation checkpoint and every required proposal arm:

| Factor | Required levels |
|---|---|
| Representation checkpoint | Every surviving objective as a separate checkpoint, including base, latent coverage, each deterministic-IDM variant, each surviving distributional-IDM family, and each surviving hybrid; do not collapse distinct objectives into one level |
| Proposal arm | Vanilla MPPI, mean-only warm start, global-variance product, heteroscedastic product, co-trained actor prior, and optional mixture-prior fusion |

This factor definition, rather than a short illustrative subset of cells, is the required factorial. Any level removed for cost or feasibility must be declared before evaluation and reported as a coverage limitation.

For each representation checkpoint, freeze its encoder and predictor, then train a fresh goal-conditioned action-chunk prior with the same architecture, initialization protocol, demonstrations, action normalization, split, optimizer budget, and stopping rule. Never reuse one prior across incompatible latent spaces.

Define proposal arms explicitly:

1. **Vanilla MPPI:** use the planner's default mean and variance without a learned prior.
2. **Mean-only warm start:** initialize the planner mean from the learned prior mean and retain the planner's default fixed variance.
3. **Global-variance product:** fuse the learned mean and a fitted global prior variance with the planner Gaussian using the same product-of-Gaussians equations as PRISM.
4. **Heteroscedastic product:** fuse the learned mean and state-dependent predicted variance with the planner Gaussian.
5. **Co-trained actor prior:** use the surviving co-trained multimodal action head as the proposal source. This tests the train-inference consistency argument directly against post-hoc priors. It requires goal-intent conditioning, so it applies to Phase 1B checkpoints and inherits support-overlap stratification.
6. **Mixture-prior fusion, optional:** form the product of the planner Gaussian with each mixture component and reweight components in closed form. Label separately from Gaussian products.

Hold candidate count, planner iterations, latent cost, action normalization, and world-model evaluation budget constant. Run the primary factorial with MPPI, whose fused covariance is fixed during its optimization iterations. Treat CEM, which adaptively refits covariance, as a separate planner-family experiment rather than mixing it into the same cells.

Interpretation rules:

- gains under vanilla planning are evidence for representation or model improvements rather than proposal guidance;
- gains shared across all representations under PRISM-style guidance are planner-side gains;
- an interaction may indicate that the learned proposal benefits more from a particular representation;
- gains unique to the co-trained actor support the train-inference consistency argument, while gains shared with post-hoc priors are proposal-side;
- planning success alone does not establish calibrated mode coverage because world-model rescoring can correct an imperfect unimodal proposal.

## Ablation checklist

### Model family

- bins and bin resolution;
- fixed isotropic scalar variance versus learned diagonal variance;
- heteroscedastic Gaussian versus MDN;
- MDN component count and covariance parameterization;
- variance floors or clamping during joint encoder training;
- best-of-K candidate count;
- selected high-capacity model settings, if reached.

### Objective

- ordinary Gaussian NLL versus PRISM-style beta-NLL with $\beta=0.5$, detached variance weighting, elementwise reduction, normalized actions, and matched variance floor;
- distributional-IDM weight;
- latent-coverage weight;
- warm-up and update schedule;
- weight tuning on validation splits only, with sensitivity curves reported;
- target/online stop-gradient placement;
- physical-successor attachment, future-goal detachment, joint goal gradients, and both-targets-detached routing;
- local-only, goal-only, and paired local/goal losses;
- fully shared, condition-token shared, shared-trunk/separate-output, parameter-matched independent, and independent-capacity actors;
- frozen-encoder head versus end-to-end encoder regularization;
- separate versus joint decoder fitting.

### Conditioning

- endpoint;
- displacement only;
- state plus displacement;
- local physical versus detached future-goal intent;
- full future goal versus intermediate waypoint;
- previous action present, absent, and shuffled;
- local/goal support-overlap bucket;
- history length;
- masked or permuted next representation;
- shuffled future goal.

### State-only baseline

- capacity;
- fitting schedule;
- frozen versus alternating;
- encoder-gradient policy;
- held-out evaluation split.

### Data

- behavior-policy entropy and diversity;
- state-action correlation;
- policy mixtures and shifts;
- observation noise;
- action dimension and nullity;
- partial observability and history sufficiency.

### Representation

- encoder dimension and capacity;
- controllable/exogenous capacity competition;
- latent coverage objective choice;
- known versus learned forward cycle evaluation.

### Action-operator execution

- Direct mean versus mode-aware Direct versus calibrated sample;
- Guarded local verification versus Pure CEM;
- preservation of the Direct reference candidate;
- global-best versus last-iteration candidate selection;
- candidate count, iteration count, and initial covariance;
- execution actor enabled versus disabled;
- full goals versus intermediate waypoints;
- goal shuffle, episode-disjoint splits, and latent line/chord regularization as a negative control.

### Planner proposal, downstream only

- vanilla MPPI versus mean-only warm start versus global-variance product versus heteroscedastic product;
- planner default variance, fitted global prior variance, and learned state-dependent prior variance;
- candidate count and planner iteration count;
- MPPI fixed-covariance behavior versus a separately analyzed adaptive-covariance planner such as CEM;
- action-prior scaling and variance floor;
- predicted-variance permutation or replacement with a global variance.

## Decision gates

### Gate 1 - implementation sanity check

Passing this gate is expected: the predicted MSE behavior and simple-model mode capture follow from standard results. The gate catches optimization, capacity, and metric bugs early; it does not test a scientific hypothesis.

**Debugging trigger:** the synthetic task does not isolate invalid means, or simple density training is not reliable. Locate and fix the implementation defect before continuing.

### Gate 2 — representation value

**Advance if:** the pre-registered primary endpoint improves beyond its multiplicity-corrected threshold and the improvement survives the frozen-head control. Report which competing-mechanism signature appeared, or that neither did.

**Stop or narrow the claim if:** gains remain decoder-local, or only secondary metrics move.

### Gate 3 - shared operator and execution value

**Advance if:** at least one matched result survives: paired local plus goal beats goal only; full sharing beats parameter-matched independent actors; action-supervised checkpoints improve actor-disabled probes or Pure-CEM planning; or a multimodal density improves valid-mode or downstream outcomes beyond the shared Gaussian. Declare which claim is primary before running the phase; the others become secondary. Support-overlap analysis must rule out uncontrolled goal extrapolation.

**Stop or narrow the claim if:** Direct success is explained by behavior cloning, actor capacity, episode retrieval, local search, or an invalid Gaussian mean.

### Gate 4 - shortcut resistance

**Advance if:** the controlled likelihood-improvement/CMI-inspired design reduces state-policy shortcut use without gaming the baseline or destabilizing training.

**Stop or retain as metric only if:** the training objective is unsound or unstable.

### Gate 5 - added density complexity

**Advance if:** a specific failure of bins/MDNs is diagnosed and a higher-capacity family improves the relevant representation or downstream metric.

**Do not advance if:** added capacity only improves its own training surrogate.

### Gate 6 - environment scaling

**Advance if:** the core ambiguity, representation, shortcut, and shared-operator conclusions survive matched controls and can be tested in a larger environment without losing diagnostic visibility. A higher-capacity density is not required if a simple model remains adequate.

**Do not advance if:** scale makes density, representation, actor, and planner effects inseparable.

## Reporting requirements

Every experimental report should include:

- environment equations and ambiguity source;
- behavior-policy distribution;
- action support and constraints;
- train/validation/test split construction;
- model capacity and parameter counts;
- objective signs, weights, normalization, and gradient paths;
- held-out density and shortcut diagnostics;
- mode and cycle metrics;
- controllable and exogenous representation probes;
- applicable downstream results and one-to-one no-regression checks;
- frozen versus end-to-end encoder setting;
- random-seed variation and uncertainty intervals;
- pre-registered primary endpoint value, seed count, and multiplicity correction used for the phase decision;
- weight-tuning protocol and split provenance;
- sampling temperature and budget for sample-based cycle metrics, declared before evaluation;
- known failure cases and decision-gate outcome;
- local and future-goal endpoint construction;
- local/goal intent-support statistics and stratified outcomes;
- previous-action conditioning and shuffle result;
- exact shared-parameter topology and shared-parameter fraction;
- gradient destination for every latent occurrence and trainable component;
- density family, action support transform, and variance constraints;
- Direct action-selection rule and whether the execution actor is enabled;
- Guarded reference-candidate, covariance, and global-best rules where applicable;
- episode-disjoint split and normalization provenance;
- goal, successor, and previous-action shuffle outcomes.

Reports that include downstream MPC should additionally include:

- planner family and proposal arm;
- candidate count, iteration count, covariance rule, and model-evaluation budget;
- prior-training data, architecture, optimizer budget, action normalization, and stopping rule;
- planning wall-clock, density-head overhead, and matched-compute comparisons.

## Items requiring approval before implementation

- initial world-model backbone;
- generated-data versus online-rollout protocol;
- first redundant action-map dimension and nullity;
- verified latent-coverage baseline;
- first downstream task;
- treatment of the likelihood-ratio score as metric or training objective;
- whether the learned inverse density remains training-only or is also used in planning;
- phase and environment for the PRISM-style proposal comparison;
- full future goals, intermediate waypoints, or both for the first paired operator;
- whether previous-action conditioning belongs in the first controlled comparison;
- primary mode-aware Direct rule for multimodal densities;
- mandatory actor-sharing and gradient-routing controls;
- diagnostic-only versus gating use of local/goal support overlap;
- timing of Pure-CEM actor-disabled evaluation;
- numerical pass/fail thresholds for each decision gate;
- the pre-registered primary endpoint, seed counts, equivalence margins, and multiplicity correction;
- the decisive-core scope and the order of contingent branches.
