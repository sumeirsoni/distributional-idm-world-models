# Archive

Deferred designs from the aggressive prune. Each entry summarizes what was cut and why it is deferred; verbatim text for every item is preserved at commit `8498e45` (`git show 8498e45:docs/EXPERIMENT_PLAN.md`, and likewise for the other files). Reactivate an entry only through its decision gate.

## Environments (from the environment matrix)

| ID | Structure | Ambiguity source | Expected failure signature |
|---|---|---|---|
| E2 | Quadratic with asymmetric policy | Policy-weighted modes | Density should match weights; MSE tracks weighted mean |
| E3 | Constructed conditionals sharing a mean | Different modal structure | Deterministic models cannot distinguish conditionals |
| E4 | $s'=s+Ba$, rank-deficient $B$ | Null-space redundancy | Point estimator picks one summary; densities waste mass off the affine set. Use a constrained parameterization (minimum-norm solution plus null-space coordinates) so diagonal-covariance artifacts are not blamed on the density concept |
| E5 | Clipped/saturated actuator | Many commands, same effect | MSE predicts interior command; density should cover the compatible interval |
| E6 | Partially observed state | State aliasing | Endpoint model multimodal; sufficient history reduces ambiguity |
| E7 | Structural ambiguity plus history | Non-injective action effect | Negative control: history must not recover absent action identity |
| E9 | State-dependent action effect | Inverse depends on state | Displacement-only decoder underperforms endpoint and state-plus-delta |
| E12 | Goal-support overlap continuum | Locally supported to extrapolative goals | Direct performance degrades as goal intents leave supported regions |
| E13 | Multimodal goal junction | Distinct valid routes per goal pair | Gaussian mean lies between valid modes |

## Deferred baselines

- Displacement-conditioned MSE IDM (Delta-JEPA-style): tests whether displacement conditioning reduces shortcuts; still predicts a mean.
- Discretized bins plus cross-entropy: simple normalized model; binning error grows with dimension.
- Fixed scalar-variance Gaussian: globally scaled MSE plus a constant; implementation equivalence check.
- PRISM-style beta-NLL Gaussian: $\operatorname{sg}((\sigma^2)^{0.5})$ times elementwise diagonal-Gaussian NLL on normalized actions, $\beta=0.5$, with the documented variance floor; evaluate with ordinary held-out NLL.
- Best-of-K / winner-takes-all: multi-hypothesis prediction; usually not a calibrated density.
- History-conditioned IDM $q(a_t\mid z_{t-k:t+1})$: ambiguity-resolving baseline for POMDP variants.
- INTACT-style local-only, goal-only, fully shared, condition-token shared, shared-trunk, parameter-matched independent, and capacity-control actors.
- Paired local/goal MDN or discretized multimodal actor.
- Gradient-routing controls: goal attached, both targets detached, encoder frozen, local-only, goal-only.
- Pure CEM on action-supervised checkpoints (execution actor disabled).
- Higher-capacity densities: normalizing flow (exact likelihood), diffusion/flow matching (complex supports), EBM (flexible support).
- PRISM-style planner proposal arms: vanilla MPPI, mean-only warm start, global-variance product, heteroscedastic product, co-trained actor prior, optional mixture-prior fusion (product of planner Gaussian with each component, reweighted in closed form).

## Deferred conditioning matrix

Endpoint $(z_t,z_{t+1})$; displacement only; state plus displacement; local physical intent $(z_t,z_{t+1}-z_t,a_{t-1})$; detached future-goal intent $(z_t,\operatorname{sg}(z_g)-z_t,a_{t-1})$; paired calls under identical grammar; history; state only. Vary previous action as present, absent, or shuffled; compare full goals with intermediate waypoints; record support-overlap buckets.

## Deferred metrics

Mode-probability calibration; marginal interval coverage and sharpness; joint region or chunk coverage; PIT/rank calibration; same-mean discrimination; learned-forward cycle error; exogenous-state probes beyond E8; invalid between-mode mass thresholds; local-to-goal intent overlap and performance by overlap quantile; action-law agreement on matched intents; Direct mean/mode/sample validity split; goal-shuffle and previous-action-shuffle degradation; episode-disjoint anti-retrieval; shared-versus-independent actor deltas; gradient cosine and shared-parameter fraction; planning success versus candidate count; world-model evaluations per success; wall-clock and proposal-head overhead.

## Phase 0B: redundant action map (E4)

Sweep action dimension, rank, nullity, bounded versus unbounded policy support, unimodal versus multimodal policy along the null space, and observation noise. Pass criteria: samples satisfy $Ba\approx s'-s$; predicted distribution reflects behavior policy along null directions; evaluation separates transition validity from exact-action recovery. Failure signatures: manifold mismatch, likelihood/validity mismatch, policy memorization.

## Phase 0C: ambiguity-source isolation

Saturation: vary clipping thresholds and command distributions. Partial observability: one history-resolvable variant, one structurally ambiguous negative control. Exogenous dynamics: measure whether IDM weight removes a predictive but uncontrollable variable. State-dependent effects: compare endpoint, delta-only, and state-plus-delta conditioning. Policy shortcuts: vary state-action correlation; use state-only likelihood, permutation, and fixed-state sensitivity.

## Phase 1B: paired local/goal action laws

Run only after Gate 2. Use episode-disjoint demonstration goals and INTACT's default asymmetric route: current state attached, physical successor attached, future goal detached.

Staged factorial:

1. Action supervision: none, local only, goal only, paired, using a diagonal Gaussian.
2. Actor coupling: fully shared, condition-token shared, shared trunk with separate outputs, parameter-matched independent, capacity control.
3. Density family: Gaussian versus discretized or MDN multimodal, grammar and budgets held fixed.
4. Execution: Direct mean, mode-aware Direct, calibrated sampling, Guarded/local verification, Pure CEM.

Gradient-routing controls: successor attached and goal detached; goal attached; both endpoints detached; encoder and predictor frozen; local loss only; goal loss only. A latent detached as a goal in one call may receive gradients in another role.

Execution rules: Direct mean executes the conditional mean; mode-aware Direct uses one predeclared rule (numerical mixture mode or peak-density component representative; highest-weight component reported separately); Direct sample uses a fixed budget; Guarded runs limited local CEM around the Direct plan and keeps the globally best candidate (INTACT reference setting: $128\times3$ candidates, initial standard deviation $0.25$); Pure CEM disables the execution actor with matched budgets. Distinguish execution-actor-disabled from training-objective-disabled.

Required outcomes and support-overlap stratification: see Gate 3.

## Phase 2: shortcut-resistant conditional-information study

Stage 2.1, evaluation only: fit $q(a\mid z_t,z_{t+1})$ and $\pi(a\mid z_t)$ properly and separately; evaluate the held-out log-likelihood difference with capacity matching. Stage 2.2, frozen or separately fitted baseline during encoder regularization, with documented gradient paths. Stage 2.3, optional variational conditional-MI bound derived and cited before coding. Abort if the baseline degrades as the score rises, permutation does not reduce the score, capacity mismatch explains the result, or encoder training destabilizes the world model.

## Phase 3: one justified higher-capacity model

Select exactly one family for a diagnosed limitation: flow for exact likelihood with smooth full-dimensional support; diffusion/flow matching for disconnected or complex high-dimensional samples; EBM for flexible support when sampling diagnostics are acceptable. Re-run the same metrics; require representation or downstream improvement, not just a better surrogate.

## Phase 3P: representation-proposal factorial

Full Cartesian product of surviving representation checkpoints with proposal arms: vanilla MPPI; mean-only warm start; global-variance product; heteroscedastic product; co-trained actor prior (Phase 1B checkpoints only; inherits support-overlap stratification); optional mixture-prior fusion. Freeze each checkpoint's encoder and predictor and train a fresh goal-conditioned action-chunk prior under matched architecture, data, normalization, budget, and stopping. Hold candidate count, iterations, latent cost, and world-model evaluation budget constant. Primary factorial uses MPPI with fixed fused covariance; treat adaptive-covariance planners separately.

Interpretation: gains under vanilla planning indicate representation or model improvements; gains shared across representations under learned proposals are planner-side; interaction indicates representation-specific proposal benefit; gains unique to the co-trained actor support train-inference consistency; planning success alone never establishes calibrated mode coverage.

## Extended ablation checklist

Model family: bins and resolution; MDN component count and covariance parameterization; best-of-K count; selected high-capacity settings. Objective: beta-NLL ablation; distributional-IDM weight; coverage weight; warm-up schedule; target/online stop-gradient placement; routing variants; previous action present, absent, shuffled; full goals versus waypoints. Conditioning: all eight variants in the conditioning matrix. State-only baseline: capacity, fitting schedule, freezing strategy, encoder-gradient policy, splits. Data: policy entropy and diversity, state-action correlation, mixtures and shifts, observation noise, action dimension and nullity, partial observability. Representation: encoder dimension, controllable/exogenous capacity competition, coverage objective choice, known versus learned forward cycle. Execution: all five rules, reference-candidate preservation, global-best selection, candidate counts and covariances. Planner: proposal arms, prior variances, candidate and iteration counts, MPPI versus CEM families, prior scaling and floors, predicted-variance permutation.

## Extended reporting requirements

Local and future-goal endpoint construction; intent-support statistics and stratified outcomes; previous-action conditioning results; exact shared-parameter topology and fraction; gradient destination for every latent occurrence and trainable component; density family, action transform, and variance constraints; Direct action-selection rule and execution-actor status; Guarded reference-candidate, covariance, and global-best rules; episode-disjoint split and normalization provenance; shuffle outcomes; full MPC reporting list (planner family, proposal arm, counts, prior-training protocol, wall-clock, overhead).

## Archived from PROJECT_BRIEF.md

### Full INTACT-style paired operator specification (former section 4.3)

The shared predictor conditions on the grammar $G_\eta(z_t,m_t,a_{t-1})$ with current-state, intent, state-intent interaction, and previous-action features. The paired objective is $\lambda_{\mathrm{local}}\mathbb{E}[-\log p_\eta(a_t\mid z_t,m^{\mathrm{local}}_t,a_{t-1})]+\lambda_{\mathrm{goal}}\mathbb{E}[-\log p_\eta(a_t\mid z_t,m^{\mathrm{goal}}_t,a_{t-1})]$. Local and goal displacements need not be numerically equal; sharing is justified through the conditional action law on supported conditions, so measure intent overlap, performance by overlap quantile, and extrapolative-goal degradation rather than treating shared parameters as global isomorphism. Default route: gradients through $z_t$ and the local successor occurrence; future-goal occurrence detached; every encoder, target encoder, predictor, previous-action encoder, trunk, and head enumerated. Required coupling controls: joint goal-endpoint gradients, both endpoints detached, frozen encoder/head-only arm, local-only and goal-only losses, fully shared actor, condition-token actor, shared-trunk/separate-output, parameter-matched independent, independent-capacity. Cross operator design with density family independently. Report Direct mean, Direct mode (predeclared rule; highest-weight component mean is not the global mixture mode), Direct sample, Guarded/local verification retaining the globally best candidate, and Pure CEM separately. Execution-actor-disabled and training-objective-disabled are different controls.

### CMI variant details (former sections 5.2 to 5.4)

Why a naive trainable difference is unsafe: the baseline can be made deliberately worse; encoder updates can degrade $z_t$ for the baseline; misspecification errors differ between terms; unrestricted scaling makes scores incomparable; fitting and evaluating on the same data overstates gains. Sound designs: separate fitting with a frozen baseline; alternating best responses with controlled encoder objectives; an explicit variational conditional-MI bound with documented assumptions and gradient paths; evaluation-only ratio as a diagnostic first. Required diagnostics: held-out NLL for both models; improvement $\mathbb{E}[\log q-\log\pi]$ on held-out data; transition permutation; conditioning ablations; fixed-state sensitivity; baseline capacity sweeps; baseline-degradation monitoring; behavior-policy sweeps; goal and successor shuffles; previous-action removal and shuffling; episode-disjoint splits; goal-only behavior-cloning controls; support-stratified evaluation; actor-disabled planning.

### Baseline list items deferred from section 9

Displacement-conditioned MSE; state-only predictor details; discretized categorical; fixed scalar-variance Gaussian; PRISM-style beta-NLL ablation; best-of-K; history-conditioned IDM; INTACT-style actor variants; higher-capacity densities; PRISM-style planner proposal.

### Measurement items deferred from section 10

Calibration appropriate to the action representation; interval coverage, sharpness, PIT where valid; probability-mass calibration across modes; same-mean discrimination; learned-forward caveats; local-to-goal overlap quantiles; action-law agreement; Direct validity splits; planning-versus-candidate-count curves; wall-clock and overhead; explicit proposal ablations; policy-learning sample efficiency; goal-reaching and trajectory quality; actor-disabled planning delta; extended shortcut diagnostics.

### Ablation items deferred from section 11

See "Extended ablation checklist" above; identical content.

### Implementation-order steps superseded by the decisive core (former section 13)

Steps covering redundant-map first experiments, INTACT-style integration ordering, CMI training variants, PRISM proposal timing, and environment scaling remain valid as branch designs; the decisive core replaces them as the main line.

### Review questions archived from section 14

Action dimensionality for the first redundant-map experiment ($d_a=2$, $d_s=1$, rank 1, nullity 1 recommended); paired-operator goal construction (one-step or short waypoints plus full goals, reported by horizon and overlap quantile); previous-action conditioning in the first INTACT comparison (absent as primary; present, absent, shuffled, previous-action-only variants); mode-aware Direct rule (highest-probability bin; numerical mixture mode or peak-density component); mandatory actor-sharing and routing controls before scaling; support overlap as diagnostic versus gate (stratify first, no hard threshold); literature-review priorities (coverage baseline, displacement-conditioned IDM, multimodal inverse-dynamics work, hindsight/goal-conditioned BC precedents).

## Archived from docs/RESEARCH_QUESTIONS.md

### H6 detail (folded into H5)

Predictions: greater transition sensitivity at fixed state; better held-out gain over the state-only model; no baseline degradation caused solely by adversarial encoder updates. Falsifiers: training raises the score by worsening $\pi(a\mid z_t)$; optimization destabilizes the base objective; gains vanish under held-out policy or permutation tests. The training variant remains a contingent Phase 2 design.

### Secondary research questions removed from the main list

How action dimensionality affects discretized, MDN, flow, diffusion, and energy-based models; can forward cycle consistency predict downstream utility; how sensitive conclusions are to known versus learned forward dynamics; does a fully shared local/goal predictor outperform independent predictors after matching; which latent occurrences should receive encoder gradients; does previous-action conditioning resolve temporal ambiguity or introduce shortcuts; should the first goal branch use full goals, waypoints, or both; can Guarded local search hide invalid between-mode mass; how should the INTACT-style operator be crossed with PRISM-style guidance without conflation.

### Open design questions removed from the main list

Forward-verifier trust conditions; source-review priorities; PRISM-comparison phase timing; goal construction for the first paired operator; previous-action inclusion; primary mode-aware Direct rule; mandatory actor topology controls; Pure-CEM evaluation timing.

Content survives in this file; nothing was deleted without a digest here or preservation at commit `8498e45`.
