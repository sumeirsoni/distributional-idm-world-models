# Pre-registration: Decisive Core (Phase 1A)

Status: frozen for gate decisions. Phase 0A may proceed on these values; any change before Phase 1A runs requires a dated amendment below.

## Environments

- E1: $s_{t+1} = s_t + a_t^2$, actions drawn from $\{\pm 0.5, \pm 1\}$ uniformly unless a Phase 0A data condition says otherwise. Conditional action laws are two-point symmetric per displacement bucket; asymmetric and same-mean variants follow the brief's section 8A data conditions.
- E10: one-to-one inverse control with a restricted nonnegative action range.
- Episodes: 200 steps. Splits: 500 train / 50 validation / 50 test episodes, episode-disjoint.

## Backbone

- State-vector observations, small MLP online/target encoder (latent dim 16).
- Target encoder updated by EMA. No normalization or weight-decay settings chosen to prevent collapse; the no-IDM arm is expected to be able to collapse.
- Forward predictor conditioned on $(z_t, a_t)$. Adam, batch 256. Exact layer sizes and learning rates are recorded in code before the first Phase 1A run and count as part of this registration.

## Arms

End-to-end, all seven: world model only; + deterministic endpoint IDM; + heteroscedastic Gaussian IDM; + MDN IDM; + generic auxiliary task (non-action target); + VICReg variance+covariance coverage terms; + distributional IDM plus coverage.

Frozen-head variants, four only: deterministic, Gaussian, MDN, auxiliary task. Same checkpoint initialization, data, head architecture, optimizer budget as their end-to-end twins.

Coverage weights and $\lambda_{\mathrm{IDM}}$ are tuned on validation splits only; sensitivity curves are reported.

## Seeds

- Bring-up and smoke tests: 3 seeds per configuration.
- Gate decisions: all 11 configurations extended to 5 seeds. This extension rule is pre-registered here.

Pair arms by shared random-number streams where architectures allow.

## Primary endpoint

Mean of three components, each standardized against the paired no-IDM arm:

1. controllable-state probe accuracy;
2. known-forward cycle validity of sampled inverse actions;
3. latent-planning success: roll learned predictor from $z_t$, score candidates against encoded goal, success within tolerance. A true-dynamics state-space planner is an ungated reference ceiling.

Decision test: Holm-corrected paired t-tests across seeds, corrected p < 0.05 AND standardized effect >= 0.2, on each component and the composite mean.

H8 equivalence band (E10): within 5 percent relative of deterministic-IDM on held-out NLL and probe accuracy, within 2 percentage points on latent-planning success. Band subject to one recalibration after Phase 0A scale checks, before any Phase 1A run.

Multiplicity: Holm across pre-declared components and cells.

## Gate decisions this registration governs

- Gate 2 only. Gates 3 through 6 govern contingent branches and get their own registrations when those branches activate.

## Amendment log

| Date | Change | Reason |
|---|---|---|
| 2026-08-21 | Initial registration | Blocking defaults approved in review session |
| 2026-08-21 | Amendment 1: gradient routing fixed - action heads read ONLINE representations with both endpoints attached; forward-predictor targets stay detached via EMA target encoder. Fixed update budget for all arms (no validation-based stopping, which rewards collapse). | Registered before first Phase 1A run |
| 2026-08-21 | Amendment 2: cycle-validity component measured through an identically trained post-hoc MDN head on each arm's frozen latents, because the no-IDM arm has no inverse head to sample. Every arm receives the same post-hoc protocol; arms with attached heads additionally report their own head's metrics. | Spec gap found during implementation; registered before gate decisions |
| 2026-08-21 | Amendment 3: standardized deltas flip sign for error components so positive always means better; per-seed rows persisted for paired tests; probe R2 noted as saturated on scalar-state environments and therefore non-discriminative there. | Found in bring-up data before any gate decision |
| 2026-08-21 | Amendment 4: pre-declared Phase 1A follow-up ablation - predictor-conditioned inverse head. The registered arms condition the action head on encoded endpoints [f(s_t), f(s_{t+1})]; this ablation instead conditions on [f(s_t), P(f(s_t), a_t)], the forward predictor's own output, so head gradients reach both encoder and predictor. Motivation: the latent planner consumes predictor outputs at inference, and a head trained on those outputs tests the train-inference consistency argument directly. Run only after the five-seed gate decision; never mixed into the primary comparison. | User-requested design; registered as an ablation before implementation |
