"""Phase 1A decisive-core runner.

Arms x datasets x seeds per the frozen pre-registration. Bring-up uses three
seeds; gate decisions extend to five. Endpoint components are standardized
against the paired no-IDM arm of the same seed. Every arm additionally gets
an identically trained post-hoc MDN head on its frozen latents so that
cycle validity and inverse NLL are defined for all arms including no-IDM.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from distributional_idm.envs.quadratic import QuadraticConfig, generate_dataset
from distributional_idm.evaluation.planning import (
    latent_planning_success,
    posthoc_cycle_validity,
    posthoc_head_nll,
)
from distributional_idm.evaluation.probes import encode_dataset, latent_health, ridge_probe_r2
from distributional_idm.models.flow import FlowIDM
from distributional_idm.models.idm import (
    MDNIDM,
    DeterministicIDM,
    GaussianIDM,
    SigmaOnlyGaussianIDM,
)
from distributional_idm.phase1 import (
    BackboneConfig,
    RandomFeatureTarget,
    train_backbone_with_restarts,
    train_frozen_head,
)
from distributional_idm.training import tensors_from
from distributional_idm.world_model import LATENT_DIM, ArmConfig, WorldModel

RESULTS_DIR = Path("results/phase1a")
SEEDS = (0, 1, 2, 3, 4)
BACKBONE = BackboneConfig()

DATASETS: dict[str, QuadraticConfig] = {
    "e1_symmetric": QuadraticConfig(name="e1_symmetric", seed=0),
    "e10_control": QuadraticConfig(name="e10_control", nonnegative_actions=True, seed=0),
    "e2_asymmetric": QuadraticConfig(
        name="e2_asymmetric", positive_probabilities=(0.7, 0.7), seed=0
    ),
    "e1_mixed": QuadraticConfig(
        name="e1_mixed", forced_positive_magnitudes=(1.0,), seed=0
    ),
}

ARMS: dict[str, ArmConfig] = {
    "no_idm": ArmConfig(name="no_idm"),
    "det_idm": ArmConfig(name="det_idm", idm_kind="deterministic", idm_weight=1.0),
    "gauss_idm": ArmConfig(name="gauss_idm", idm_kind="gaussian", idm_weight=0.3),
    "mdn_idm": ArmConfig(name="mdn_idm", idm_kind="mdn", idm_weight=0.3),
    "aux_task": ArmConfig(name="aux_task", aux_task=True),
    "coverage": ArmConfig(name="coverage", coverage=True),
    "hybrid": ArmConfig(name="hybrid", idm_kind="mdn", coverage=True, idm_weight=0.3),
    "mdn_pred": ArmConfig(
        name="mdn_pred", idm_kind="mdn", idm_conditioning="predictor", idm_weight=0.3
    ),
    "flow_idm": ArmConfig(name="flow_idm", idm_kind="flow", idm_weight=0.3),
    "mdn_cycle": ArmConfig(
        name="mdn_cycle", idm_kind="mdn", cycle=True, idm_weight=0.3, cycle_weight=1.0
    ),
    "flow_cycle": ArmConfig(
        name="flow_cycle", idm_kind="flow", cycle=True, idm_weight=0.3, cycle_weight=1.0
    ),
    "sigma_only": ArmConfig(name="sigma_only", idm_kind="sigma_gaussian", idm_weight=0.3),
    "gauss_bnnl": ArmConfig(
        name="gauss_bnnl", idm_kind="gaussian", idm_loss_variant="beta", idm_weight=0.3
    ),
    "gauss_clampnll": ArmConfig(
        name="gauss_clampnll", idm_kind="gaussian", idm_loss_variant="clamped", idm_weight=0.3
    ),
}
HEAD_BUILDERS = {
    "deterministic": lambda: DeterministicIDM(2 * LATENT_DIM),
    "gaussian": lambda: GaussianIDM(2 * LATENT_DIM),
    "mdn": lambda: MDNIDM(2 * LATENT_DIM, n_components=2),
    "sigma": lambda: SigmaOnlyGaussianIDM(2 * LATENT_DIM),
    "sigma_gaussian": lambda: SigmaOnlyGaussianIDM(2 * LATENT_DIM),
    "flow": lambda: FlowIDM(2 * LATENT_DIM, n_bins=24),
}


def state_tensor(dataset) -> torch.Tensor:
    return torch.from_numpy(dataset.states.astype(np.float32))


@torch.no_grad()
def probe_metrics(world_model: WorldModel, splits) -> dict[str, float]:
    latents_train = encode_dataset(world_model.encoder, state_tensor(splits["train"]))
    latents_test = encode_dataset(world_model.encoder, state_tensor(splits["test"]))
    r2 = ridge_probe_r2(
        latents_train, splits["train"].next_states_clean,
        latents_test, splits["test"].next_states_clean,
    )
    health = latent_health(latents_test)
    return {"probe_r2_controllable": round(r2, 4), **{k: round(v, 4) for k, v in health.items()}}


def make_twin_head(kind: str):
    return HEAD_BUILDERS[kind]()


@torch.no_grad()
def attached_predictor_head_metrics(
    world_model: WorldModel,
    states: torch.Tensor,
    actions: torch.Tensor,
    next_states_clean: torch.Tensor,
    generator: torch.Generator,
) -> dict[str, float]:
    """Evaluate an attached predictor-conditioned head teacher-forced: the
    head sees [f(s_t), P(f(s_t), a_true)] on held-out data."""

    from distributional_idm.evaluation.metrics import (
        gaussian_nll,
        mdn_nll,
        sample_mdn,
    )

    world_model.encoder.eval()
    world_model.predictor.eval()
    features = world_model.idm_features(actions, states, next_states_clean)
    outputs = world_model.idm.predict(features)
    if "logits" in outputs:
        nll = mdn_nll(outputs["logits"], outputs["means"], outputs["stds"], actions)
        sampled = sample_mdn(outputs["logits"], outputs["means"], outputs["stds"],
                             16, generator)
    else:
        std = outputs.get("std")
        if std is None:
            return {}
        nll = gaussian_nll(outputs["mean"], std, actions)
        noise = torch.randn(len(states), 16, generator=generator)
        sampled = outputs["mean"].unsqueeze(-1) + std.unsqueeze(-1) * noise
    predicted = states.unsqueeze(-1) + sampled**2
    target = next_states_clean.unsqueeze(-1)
    return {
        "attached_nll": round(nll, 4),
        "attached_cycle": round(float((predicted - target).abs().mean()), 4),
    }


def evaluate_arm(world_model: WorldModel, splits, seed: int) -> dict[str, float]:

    standardizer = state_standardizer(splits["train"])
    tr = tensors_from(splits["train"], "state_displacement", standardizer)
    te = tensors_from(splits["test"], "state_displacement", standardizer)

    metrics: dict[str, float] = {}
    metrics.update(probe_metrics(world_model, splits))
    metrics.update(latent_planning_success(
        world_model.encoder, world_model.predictor, te["states"], te["next_clean"]
    ))

    generator = torch.Generator().manual_seed(4242 + seed)

    if world_model.idm is not None:
        kind = world_model.config.idm_kind
        assert kind is not None
        if world_model.config.idm_conditioning == "predictor":
            metrics.update(attached_predictor_head_metrics(
                world_model, te["states"], te["actions"], te["next_clean"], generator
            ))
        else:
            metrics["attached_nll"] = posthoc_head_nll(
                world_model.idm, world_model.encoder, te["states"], te["actions"], te["next_clean"]
            )
            metrics["attached_cycle"] = posthoc_cycle_validity(
                world_model.idm, world_model.encoder, te["states"], te["next_clean"],
                n_samples=16, generator=generator,
            )
        frozen_twin = train_frozen_head(make_twin_head(kind), world_model.encoder, tr, BACKBONE, seed + 777)
        metrics["frozen_nll"] = posthoc_head_nll(
            frozen_twin, world_model.encoder, te["states"], te["actions"], te["next_clean"]
        )
        metrics["frozen_cycle"] = posthoc_cycle_validity(
            frozen_twin, world_model.encoder, te["states"], te["next_clean"],
            n_samples=16, generator=generator,
        )

    posthoc_head = train_frozen_head(
        MDNIDM(2 * LATENT_DIM, n_components=2), world_model.encoder, tr, BACKBONE, seed + 555
    )
    metrics["posthoc_mdn_nll"] = posthoc_head_nll(
        posthoc_head, world_model.encoder, te["states"], te["actions"], te["next_clean"]
    )
    metrics["posthoc_mdn_cycle"] = posthoc_cycle_validity(
        posthoc_head, world_model.encoder, te["states"], te["next_clean"],
        n_samples=16, generator=generator,
    )
    return metrics


def aggregate(per_seed: list[dict[str, float]]) -> dict[str, float]:
    aggregated: dict[str, float] = {}
    for key in per_seed[0]:
        values = np.array([float(row[key]) for row in per_seed if row.get(key) is not None])
        if values.size == 0:
            continue
        aggregated[key] = round(float(values.mean()), 4)
        aggregated[f"{key}_std"] = round(float(values.std()), 4)
    return aggregated


def standardize_against_no_idm(arms: dict[str, dict]) -> None:
    """Standardized deltas where POSITIVE always means better.

    Error-style components (cycle errors, NLLs) flip sign before dividing,
    so a lower error yields a positive delta. Components are also reported
    with their raw delta because pooled SDs near zero make standardized
    values explode; the |d| >= 0.2 rule uses the raw-delta direction check
    downstream.
    """

    baseline = arms.get("no_idm")
    if baseline is None:
        return
    components = {
        "probe_r2_controllable": 1.0,
        "planning_success": 1.0,
        "posthoc_mdn_cycle": -1.0,
        "posthoc_mdn_nll": -1.0,
    }
    for agg in arms.values():
        scores: dict[str, float] = {}
        for component, direction in components.items():
            if component not in agg:
                continue
            raw_delta = agg[component] - baseline[component]
            scores[f"{component}_delta"] = round(raw_delta, 5)
            pooled_sd = max(baseline.get(f"{component}_std", 0.0), 1e-6)
            scores[f"{component}_d_vs_no_idm"] = round(direction * raw_delta / pooled_sd, 3)
        active = [v for k, v in scores.items() if k.endswith("_d_vs_no_idm")]
        scores["endpoint_d_mean"] = round(sum(active) / len(active), 3)
        agg["standardized"] = scores


def run(arm_filter: set[str] | None = None, dataset_filter: set[str] | None = None) -> dict:
    results: dict = {"seeds": list(SEEDS), "backbone": vars(BACKBONE), "datasets": {}}
    selected_arms = {k: v for k, v in ARMS.items() if arm_filter is None or k in arm_filter}
    for dataset_name, config in DATASETS.items():
        if dataset_filter is not None and dataset_name not in dataset_filter:
            continue
        splits = generate_dataset(config)
        dataset_entry: dict[str, dict] = {"arms": {}}
        feature_target = RandomFeatureTarget(seed=1234)

        for arm_name, arm_config in selected_arms.items():
            per_seed: list[dict[str, float]] = []
            for seed in SEEDS:
                start_time = time.time()
                world_model = train_backbone_with_restarts(
                    arm_config,
                    tensors_from(
                        splits["train"],
                        "state_displacement",
                        state_standardizer(splits["train"]),
                    ),
                    tensors_from(
                        splits["val"], "state_displacement", state_standardizer(splits["train"])
                    ),
                    BACKBONE,
                    seed,
                    feature_target,
                )
                entry = evaluate_arm(world_model, splits, seed)
                entry["seconds"] = round(time.time() - start_time, 1)
                per_seed.append(entry)
                print(
                    f"[{dataset_name}] {arm_name} seed{seed}: "
                    f"probe={entry['probe_r2_controllable']:.3f} "
                    f"plan={entry['planning_success']:.3f} "
                    f"posthoc_cyc={entry['posthoc_mdn_cycle']:.3f} "
                    f"({entry['seconds']}s)"
                )

            dataset_entry["arms"][arm_name] = aggregate(per_seed)
            dataset_entry.setdefault("per_seed", []).extend(
                [{"arm": arm_name, "seed": seed, **row} for seed, row in zip(SEEDS, per_seed)]
            )

        standardize_against_no_idm(dataset_entry["arms"])
        results["datasets"][dataset_name] = dataset_entry

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / "results.json"
    if arm_filter is not None or dataset_filter is not None and output.exists():
        try:
            existing = json.loads(output.read_text())
            for dataset_name, entry in results["datasets"].items():
                target = existing.setdefault("datasets", {}).setdefault(dataset_name, {"arms": {}, "per_seed": []})
                target["arms"].update(entry["arms"])
                seen = {
                    (row["arm"], row["seed"]) for row in target.get("per_seed", [])
                }
                target["per_seed"] = [
                    row for row in target.get("per_seed", [])
                    if (row["arm"], row["seed"]) not in
                    {(r["arm"], r["seed"]) for r in entry.get("per_seed", [])}
                ] + entry.get("per_seed", [])
                del seen
        except (json.JSONDecodeError, KeyError):
            pass
        output.write_text(json.dumps(existing, indent=2))
    else:
        output.write_text(json.dumps(results, indent=2))
    print(f"wrote {output}")
    return results


def state_standardizer(dataset):
    from distributional_idm.training import Standardizer

    states = dataset.states.astype(np.float32)
    return Standardizer(mean=np.array([states.mean()]), std=np.array([max(states.std(), 1e-6)]))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", type=str, default=None, help="comma-separated arm names")
    parser.add_argument("--datasets", type=str, default=None, help="comma-separated dataset names")
    args = parser.parse_args()
    run(
        set(args.arms.split(",")) if args.arms else None,
        set(args.datasets.split(",")) if args.datasets else None,
    )
