# Experiment Plan

## Status and operating principle

This is a planning document only. No experiments have been implemented.

The program is deliberately small. One decisive experiment carries the project; every other phase is a contingent branch. Deferred environments, baselines, metrics, and phase designs are summarized in [`ARCHIVE.md`](ARCHIVE.md), and their full text is preserved at commit `8498e45`. Advance only when the decision gates pass. INTACT, PRISM, Delta-JEPA, and LeJEPA/SIGReg specifications are verified against primary sources; LeWorldModel and PLDM require full-text review before exact implementation.

## Pre-registration requirements

Before any Phase 1 run, record the following in `results/preregistration.md`: the primary composite endpoint, the seed count, the equivalence margins, and the multiplicity correction. A phase advances only when its pre-registered endpoint clears the corrected threshold. Secondary metrics generate hypotheses; they never justify advancement on their own.

Recommended defaults, subject to review:

- Primary endpoint: the mean of controllable-state probe accuracy, known-forward cycle validity of sampled actions, and one small planning success rate under known dynamics, each normalized against the no-IDM arm of the same checkpoint.
- Seeds: at least five per cell; report seed-level intervals.
- Multiplicity: Holm correction across the pre-declared endpoint components and the cells of the decisive core.
- Equivalence margin for H8: declare a numeric band before the one-to-one control runs. H8 is the no-regression hypothesis in `RESEARCH_QUESTIONS.md`.

## Decisive core

The decisive experiment is Phase 1A restricted to environments E1 and E10, with seven arms (world model only; deterministic IDM; heteroscedastic Gaussian IDM; MDN; generic auxiliary task; latent coverage; hybrid) and frozen-head variants for the four action-head arms, at the seed count above. Phases 0B through 0C, 1B, 2, 3, 3P, and 4 are contingent branches. Each starts only if the decisive core produces a representation effect worth pursuing or a diagnostic failure that redirects the design. Phase 0A remains an implementation check for generators, metrics, and training loops.

## Environments

| ID | Structure | Role |
|---|---|---|
| E1 | $s_{t+1}=s_t+a_t^2$ with symmetric $\pm a$ actions | Structural ambiguity; canonical invalid-mean test |
| E10 | One-to-one inverse map | No-regression control |

Two contingent environments carry the competing-mechanism signatures:

- **E8 (exogenous variable):** tests calibrated acceptance. Prediction: deterministic IDM drops exogenous task-relevant state; distributional IDM retains it.
- **E11 (policy shortcut):** tests pressure strength and shortcut probes under strong state-action correlation.

Environments E2-E7, E9, E12, and E13, with their controlled variables and expected failure signatures, are archived.

## Baselines

Decisive-core set:

1. no IDM;
2. deterministic endpoint-conditioned MSE IDM;
3. heteroscedastic diagonal-Gaussian IDM trained with ordinary NLL;
4. MDN distributional IDM;
5. generic auxiliary-task head predicting a non-action target at matched weight;
6. latent coverage regularizer (VICReg-style variance plus covariance terms by default; see section 14 of the brief);
7. hybrid distributional IDM plus coverage;
8. frozen-encoder probabilistic head;
9. state-only predictor $\pi(a_t\mid z_t)$.

Match decoder capacity and parameter counts across arms. Deferred baselines - displacement-conditioned MSE, discretized bins, fixed scalar-variance Gaussian, best-of-K, history conditioning, PRISM-style beta-NLL, INTACT-style actor topologies, higher-capacity densities, and planner proposals - are defined in the archive and enter only when a gate demands them.

## Metrics

Core metrics:

- Endpoint components: controllable-state probes, known-forward cycle validity of sampled actions, and small-planning success rate under known dynamics.
- Held-out conditional NLL where tractable, and held-out $\mathbb{E}[\log q-\log\pi]$ against the properly fitted state-only baseline.
- Mode coverage, mode precision, and invalid between-mode mass; E1 modes are analytic.
- Collapse indicators: per-dimension variance, covariance spectrum, effective rank.
- Shortcut probes: transition permutation, fixed-state sensitivity, behavior-policy shift.
- One-to-one no-regression delta on E10.

Declare sampling temperature and budget for sample-based cycle metrics before evaluation. Calibration, PIT, support-overlap, action-law, and planning-efficiency metrics are archived until their phases activate.

## Phase 0A: implementation check on E1

### Data conditions

- Symmetric $+a/-a$ with equal probability.
- Asymmetric mode probabilities.
- Multiple magnitudes.
- Conditionals constructed to share a mean but differ in modal structure.
- Observation-noise sweep.
- One-to-one control variant with a restricted nonnegative action range (E10).

### Required models

Endpoint MSE, state-only predictor, heteroscedastic diagonal-Gaussian IDM, and small MDN. The fixed scalar-variance Gaussian is an optional equivalence check of the implementation.

### Pass criteria

These criteria follow from standard results, so treat this phase as an implementation check whose outputs are working generators, metrics, and training loops.

- MSE exhibits the predicted conditional-mean behavior.
- At least one simple distributional model represents both valid modes.
- Density samples improve valid-mode and known-forward cycle metrics.
- Results hold on held-out transitions.

### Debugging signatures

- **MSE invalid average:** low action MSE but poor transition validity. Expected, not a bug.
- **Mode collapse:** MDN assigns nearly all mass to one symmetric mode.
- **Variance inflation:** one broad component covers both modes with mass on invalid actions.
- **Conditioning collapse:** nearly identical densities after transition permutation.

## Phase 1A: decisive representation experiment

### Arms

For the same encoder/world-model backbone on E1 and E10:

1. world-model objective only;
2. plus deterministic endpoint IDM;
3. plus heteroscedastic Gaussian IDM;
4. plus MDN distributional IDM;
5. plus generic auxiliary-task head with a non-action target at matched weight;
6. plus latent coverage regularizer;
7. plus distributional IDM and coverage.

Each arm runs end-to-end. Arms 2 through 5 additionally run a frozen-head variant: initialize from the same world-model checkpoint, freeze encoder and predictor, and train only the action head. The coverage and hybrid arms have no meaningful frozen variant, because their regularizers act on encoder gradients. Use identical data, head architecture, initialization protocol, optimizer budget, and evaluation splits across variants. In every end-to-end arm, enumerate every component that receives gradients. Tune weights on validation splits only and report sensitivity curves.

### Backbone gradient routing

Specify before coding how IDM gradients enter the minimal backbone: whether the action head reads online or target representations, whether $z_{t+1}$ occurrences receive IDM gradients when the forward predictor detaches targets, and which components each loss updates.

### Required outcome

An arm advances only if the pre-registered endpoint improves beyond its multiplicity-corrected threshold without unacceptable degradation in exogenous-state retention, base forward prediction, E10 performance, stability, or compute. Report which competing-mechanism signature appeared, or that neither did.

## Contingent branches

Each branch has an activation condition; its full design is archived.

| Branch | Activates when | Purpose |
|---|---|---|
| 0B/0C | Core ambiguity results need refinement | Redundant maps, saturation, partial observability, state-dependent effects |
| 1B | Phase 1A passes Gate 2 | INTACT-style local/goal sharing, routing, and multimodal density within the shared operator |
| 2 | Shortcut signatures appear | Shortcut-resistant conditional-information objective |
| 3 | Simple density families hit a diagnosed limit | One higher-capacity density family |
| 3P | MPC experiments begin | Representation-versus-proposal factorial, co-trained actor prior, mixture fusion |
| 4 | Gates 1 through 5 conclusions hold | One larger controlled environment |

## Decision gates

### Gate 1 - implementation sanity check

Passing this gate is expected: the predicted MSE behavior and simple-model mode capture follow from standard results. The gate catches optimization, capacity, and metric bugs early; it does not test a scientific hypothesis.

**Debugging trigger:** the synthetic task does not isolate invalid means, or simple density training is not reliable. Locate and fix the implementation defect before continuing.

### Gate 2 - representation value

**Advance if:** the pre-registered primary endpoint improves beyond its multiplicity-corrected threshold and the improvement survives the frozen-head control. Report which competing-mechanism signature appeared, or that neither did.

**Stop or narrow the claim if:** gains remain decoder-local, or only secondary metrics move.

### Gate 3 - shared operator and execution value

Gate 3 governs contingent Phase 1B. The archive defines Direct execution, Pure CEM, the actor topologies, and support-overlap analysis referenced here.

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
- train/validation/test split construction;
- model capacity and parameter counts;
- objective signs, weights, normalization, gradient paths, and weight-tuning protocol;
- held-out density and shortcut diagnostics;
- mode and cycle metrics with declared sampling budgets;
- controllable and exogenous representation probes;
- applicable downstream results and the E10 no-regression delta;
- frozen versus end-to-end setting;
- random-seed variation and uncertainty intervals;
- pre-registered endpoint value, seed count, and multiplicity correction used for the phase decision;
- known failure cases and decision-gate outcome.

Reports that include downstream MPC add planner family, proposal arm, candidate and iteration counts, prior-training protocol, and wall-clock overhead; see the archive for the extended reporting list.

## Items requiring approval before implementation

- initial world-model backbone;
- generated-data versus online-rollout protocol;
- the pre-registered primary endpoint, seed counts, equivalence margins, and multiplicity correction;
- verified latent-coverage baseline method;
- numerical pass/fail thresholds for each gate;
- the order of contingent branches after the decisive core.
