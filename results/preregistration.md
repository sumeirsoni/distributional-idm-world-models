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
| 2026-08-21 | Amendment 5: Gate 2 evaluated at five seeds per the staged policy. Outcome: no arm advances (48 tests, none Holm-significant, none clearing d >= +0.2). Notable non-significant signals recorded in results/README.md. Per the acceptable-outcomes clause the controlled null stands unless a registered follow-up (amendment 4 ablation, MDN stability treatment) is approved with its own amendment before running. | Gate decision executed as registered |
| 2026-08-21 | Amendment 6: stability treatment before any re-comparison of arms. (a) Gradient-norm clipping (max norm 1.0) applied to every arm's backbone update. (b) Joint-training restart selection: every arm runs three independent backbone initializations and the one with the highest validation-split latent-planning success is kept; validation world-model loss is never used because it rewards collapse. Selection uses the validation split only. Rationale: planning success is effectively binary (healthy versus degenerate latent geometry) and arm comparisons at five seeds measured collapse-escape rates under a single-init lottery; best-of-three makes that lottery explicit and uniform across arms. This amendment supersedes the single-run protocol for any Phase 1A numbers reported after its date; the five-seed Gate 2 outcome of amendment 5 stands as recorded. | Approved by user before rerun |
| 2026-08-21 | Amendment 7: multimodal-head repair factorial. Three new arms at five seeds under the amendment 6 protocol, lambda matched to existing density arms (0.3): (a) flow_idm - endpoint-conditioned rational-quadratic-spline flow head, exact held-out NLL, no cycle term; tests whether removing component-bookkeeping degeneracies alone restores regularization pressure; (b) mdn_cycle - the existing endpoint-conditioned MDN plus an auxiliary cycle loss: reparameterized samples from the head are pushed through the forward predictor and regressed toward the detached target encoding of the actual next state (weight 1.0); tests whether restoring row-specific pressure alone suffices; (c) flow_cycle - both together. Cycle gradients reach encoder, head, and predictor (predictor action-responsiveness is an intended side effect, Delta-JEPA-style). Hypotheses: H-A absorption is fixable by de-degenerizing the head (a advances); H-B pressure is restorable independently of head family (b advances); H-C both required (only c advances). | Registered after user confirmation, before implementation |
| 2026-08-21 | Amendment 8: direct test of the statistic-channel hypothesis via channel decomposition. New arm sigma_only: Gaussian NLL with the mean pinned to zero, training only the predicted standard deviation. The full Gaussian arm decomposes into a mu channel (blind on E1, where every conditional mean is zero) and a sigma channel (live: the optimal standard deviation is 0.5 or 1.0 depending on displacement bucket, so setting it requires decoding displacement from latents). Pre-registered predictions: statistic-channel view predicts sigma_only reproduces most of gauss_idm's planning benefit on E1; an interaction-based alternative predicts sigma_only lands far below gauss_idm. Disclosure: pinning mu to zero is suboptimal on E10 where conditional means carry information, so E10 is secondary; the decisive read is E1. Lambda 0.3 matched to gauss_idm. | Registered after user confirmation, before implementation |
| 2026-08-21 | Amendment 9: amendment 8 outcome recorded. sigma_only produced E1 planning d=-0.06 versus gauss_idm's +1.29; paired gauss-minus-sigma deltas were positive in all five seeds (mean +0.36). The pre-registered statistic-channel prediction is rejected: the uncertainty channel alone provides none of the regularization benefit. Established instead: the benefit requires the JOINT Gaussian NLL - neither channel alone reproduces it (mu-only = det_idm was also weak). Note the near-degeneracy of the comparison: on E1 with converged mu approaching zero, the sigma-only loss converges in form toward the full loss; the divergence of outcomes under near-identical optima indicates the effect is dynamical (coupled two-channel optimization) rather than a property of either target statistic. Mechanism remains open. | Gate-style falsification executed as registered |
| 2026-08-21 | Amendment 10: synthetic-structure artifact tests, prompted by review suspicion that E1's zero-mean symmetric permanent ambiguity drives the head-family ordering. Two new datasets at five seeds under the amendment 6 protocol, arms {no_idm, det_idm, gauss_idm, sigma_only, mdn_idm}: (a) e2_asymmetric - conditional sign probabilities 0.7/0.3, so conditional MEANS now carry information (plus-or-minus 0.3·magnitude) and a pinned-zero mean becomes actively wrong; (b) e1_mixed - new per-magnitude sign forcing: |a|=1 transitions are deterministic (+1 only), |a|=0.5 transitions stay bimodal, so ambiguity is partial and localized. Planning success additionally stratified by displacement bucket on e1_mixed. Pre-registered artifact hypotheses: H-sym - if zero-mean symmetry caused det_idm's weakness, det improves substantially on e2_asymmetric relative to its E1 delta; H-pin - if pinned-zero mu is the reason sigma_only failed, sigma_only turns harmful on e2_asymmetric; H-perm - if permanent ambiguity inflated uncertainty-channel value, action-arm gains concentrate on the ambiguous bucket in e1_mixed. Failure of a prediction falsifies the corresponding artifact explanation. | Registered after user raised the concern, before implementation |
| 2026-08-21 | Amendment 11: spike-budget mechanism test on e1_mixed. Refined mechanism statement superseding the earlier confidence-weighting phrasing: deterministic-bucket rows near the variance floor generate transient NLL spikes whose 1/sigma^2 scaling consumes the global clipping budget, zeroing ambiguous-bucket contributions in those steps. Two joint-training arms at five seeds: (a) gauss_bnnl - beta-NLL with detached sg(sigma) multiplier, beta=0.5, PRISM-faithful form; damps spike magnitude as sigma approaches the floor; (b) gauss_clampnll - ordinary Gaussian NLL with each per-sample NLL contribution clamped at 10 nats; caps any single row's budget theft without touching normal gradients. Pre-registered predictions under the spike-budget hypothesis: both arms reduce or eliminate ambiguous-bucket degradation relative to gauss_idm (0.197 versus no-IDM 0.488) while keeping deterministic-bucket gains; if clamping works but beta-NLL does not, budget theft is the operative channel; if neither works, the reallocation is driven by expected-gradient asymmetry rather than spikes. Note: PRISM applies beta-NLL offline on frozen latents; this amendment tests it jointly, which is the regime beta-NLL was designed for. | Registered after discussion, before implementation |
