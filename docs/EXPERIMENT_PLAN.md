# Experiment Plan

## Status and operating principle

This is a planning document only. No experiments have been implemented.

The program is **falsification-first**: begin with analytically understandable environments and simple density families; advance only when predefined decision gates are passed. Named-method baselines require primary-source verification before their exact implementation is specified.

## Phase overview

| Phase | Goal | Environments | Models | Exit condition |
|---|---|---|---|---|
| 0A | Demonstrate invalid conditional means | Quadratic ambiguity | Endpoint MSE, displacement MSE, state-only, discretized, MDN | Simple density model captures both modes and beats MSE on modal/cycle metrics |
| 0B | Test continuous action non-identifiability | Rank-deficient linear action map | Same, plus best-of-K | Model represents policy-weighted action sets or samples valid actions |
| 0C | Separate ambiguity sources | Saturation, partial observability, exogenous state, state-dependent effects, policy variants | Add history and conditioning ablations | Failure signatures match the intended ambiguity mechanism |
| 1 | Test representation regularization | Minimal predictive encoder/world model on Phase 0 environments | No IDM, deterministic IDM, distributional IDM, coverage-only, hybrid | Representation or downstream gain beyond density fit |
| 2 | Test shortcut-resistant objective | Policy-correlated variants | State-only baseline, evaluation ratio, then controlled training variant | Reduced shortcut use without baseline gaming or instability |
| 3 | Test justified added capacity | Only environments where simple models are limiting | One of flow, diffusion/flow matching, or EBM | Added capacity improves relevant representation/downstream outcomes |
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
| E11 | Policy shortcut variant | Strong $s$-to-$a$ correlation | policy entropy, mixtures, shifts | Test transition use | Endpoint model predicts well after transition permutation; ratio exposes low added information |

## Baseline matrix

| Baseline | Output/objective | Question answered | Key caveat |
|---|---|---|---|
| Deterministic endpoint IDM | $g(z_t,z_{t+1})$, MSE | Does standard regression average modes? | Can exploit state-policy shortcuts |
| Deterministic displacement IDM | $g(z_{t+1}-z_t)$, MSE | Does displacement conditioning reduce shortcuts? | Still predicts a mean; may be ill-posed under state dependence |
| State-only action model | $\pi(a_t\mid z_t)$ | How much action prediction comes from policy correlation? | Must be capacity-matched and properly fitted |
| Discretized distributional IDM | bins plus cross-entropy | Can a simple normalized model capture modes? | Binning error; scales poorly with dimension |
| Mixture density network | conditional mixture likelihood | Can a simple continuous density capture modes? | Component collapse and variance instability |
| Best-of-K / winner-takes-all | multiple candidates, min/best loss | Does multi-hypothesis prediction suffice? | Usually not a calibrated normalized density |
| History-conditioned IDM | $q(a_t\mid z_{t-k:t+1})$ | Is ambiguity resolvable with temporal context? | Extra capacity and policy cues can confound results |
| Latent coverage regularizer | variance/rank/geometry term | Is broad anti-collapse pressure enough? | Nuisance factors can satisfy it; exact named method pending source review |
| Hybrid | distributional IDM plus coverage | Are semantic action pressure and broad coverage complementary? | Requires weight and gradient-conflict ablations |
| Higher-capacity density | flow, diffusion/flow matching, or EBM | Are simple-family limitations blocking progress? | Allowed only after Gate 3; fairness and compute are harder |

## Conditioning matrix

Every applicable decoder should be compared under:

1. endpoint: $(z_t,z_{t+1})$;
2. displacement only: $z_{t+1}-z_t$;
3. state plus displacement: $(z_t,z_{t+1}-z_t)$;
4. history: $z_{t-k:t+1}$, where relevant;
5. state only: $z_t$, as the behavior-policy baseline.

For each, record parameter count, receptive information, encoder-gradient paths, and any target-encoder stop-gradient.

## Metric matrix

| Category | Metric | Interpretation | Important limitation |
|---|---|---|---|
| Density | Held-out conditional NLL | Proper fit when normalized likelihood is available | Not directly available for all EBMs/diffusion objectives |
| Density | Held-out $\mathbb{E}[\log q-\log\pi]$ | Added predictive information from transition | Not automatically a CMI bound with fitted approximations |
| Calibration | Bin/mode probability calibration | Whether predicted probabilities match frequencies | Requires careful grouping and sufficient samples |
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
| Shortcut | Transition permutation degradation | Whether decoder uses next-state information | Permutation distribution must remain meaningful |
| Shortcut | Fixed-state transition sensitivity | Whether conditional changes with transition | Sensitivity alone does not establish correctness |
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
- small MDN;
- optional best-of-K after the core comparison.

### Pass criteria

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

Choose $B\in\mathbb{R}^{d_s\times d_a}$ with $\operatorname{rank}(B)<d_a$. Generate actions from explicitly documented policies that control probability along the null space.

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

## Phase 1: representation regularization

### Minimal objective comparison

For the same encoder/world-model backbone, compare:

1. world-model objective only;
2. plus deterministic endpoint IDM;
3. plus deterministic displacement IDM;
4. plus discretized distributional IDM;
5. plus MDN distributional IDM;
6. plus latent coverage regularizer;
7. plus distributional IDM and latent coverage;
8. plus history-conditioned IDM where relevant.

Use matched tuning budgets. Report the regularizer weights and whether gradients reach online encoder, target encoder, predictor, and action decoder.

### Required outcome

A method advances only if it improves at least one representation/downstream target without unacceptable degradation in:

- exogenous-state retention;
- base forward prediction;
- one-to-one control tasks;
- training stability;
- compute relative to the demonstrated benefit.

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

## Ablation checklist

### Model family

- bins and bin resolution;
- MDN component count and covariance parameterization;
- best-of-K candidate count;
- selected high-capacity model settings, if reached.

### Objective

- distributional-IDM weight;
- latent-coverage weight;
- warm-up and update schedule;
- target/online stop-gradient placement;
- separate versus joint decoder fitting.

### Conditioning

- endpoint;
- displacement only;
- state plus displacement;
- history length;
- masked or permuted next representation.

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

## Decision gates

### Gate 1 — mode capture

**Advance if:** deterministic MSE fails as predicted and a simple discretized model or MDN captures valid modes with better calibration and cycle consistency.

**Stop or redesign if:** the synthetic task does not isolate invalid means, or simple density training is not reliable.

### Gate 2 — representation value

**Advance if:** distributional regularization improves representation probes, prediction, planning, or control—not only action likelihood.

**Stop or narrow the claim if:** gains remain decoder-local.

### Gate 3 — shortcut resistance

**Advance if:** the controlled likelihood-improvement/CMI-inspired design reduces state-policy shortcut use without gaming the baseline or destabilizing training.

**Stop or retain as metric only if:** the training objective is unsound or unstable.

### Gate 4 — added complexity

**Advance if:** a specific failure of bins/MDNs is diagnosed and a higher-capacity family improves the relevant representation or downstream metric.

**Do not advance if:** added capacity only improves its own training surrogate.

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
- downstream results and one-to-one no-regression checks;
- random-seed variation and uncertainty intervals;
- known failure cases and decision-gate outcome.

## Items requiring approval before implementation

- initial world-model backbone;
- generated-data versus online-rollout protocol;
- first redundant action-map dimension and nullity;
- verified latent-coverage baseline;
- first downstream task;
- treatment of the likelihood-ratio score as metric or training objective;
- numerical pass/fail thresholds for each decision gate.
