"""Instrument per-bucket losses during e1_mixed training.

Distinguishes three explanations for the ambiguous-bucket planning failure:
1. improves then degrades  -> interference / active forgetting
2. never improves          -> the head cannot express the solution
3. improves while planning fails -> metric mismatch

Logs validation NLL per displacement bucket (ambiguous: ds=0.25, bimodal;
deterministic: ds=1.00, forced +a=1) and ambiguous-bucket latent-planning
success every epoch for gauss/det/mdn arms.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "scripts")

from run_phase1a import state_standardizer

from distributional_idm.envs.quadratic import QuadraticConfig, generate_dataset
from distributional_idm.evaluation.planning import latent_planning_success
from distributional_idm.phase1 import BackboneConfig, train_backbone
from distributional_idm.training import tensors_from
from distributional_idm.world_model import ArmConfig, WorldModel

OUTPUT = Path("results/phase1a/e1_mixed_loss_traces.json")
SEEDS = (0, 1, 2)


def per_row_nll(world_model: WorldModel, features: torch.Tensor, actions: torch.Tensor):
    head = world_model.idm
    with torch.no_grad():
        if hasattr(head, "flow_nll_tensor"):
            return head.flow_nll_tensor(features, actions)
        outputs = head.predict(features)
        if "logits" in outputs:
            import torch.nn.functional as F

            from distributional_idm.evaluation.metrics import mdn_nll as _  # noqa: F401

            log_probs = -0.5 * (actions.unsqueeze(-1) - outputs["means"]) ** 2 / outputs["stds"] ** 2 - outputs["stds"].log()
            log_mix = torch.logsumexp(F.log_softmax(outputs["logits"], dim=-1) + log_probs, dim=-1)
            return -log_mix
        std = outputs.get("std")
        if std is None:
            return (actions - outputs["mean"]) ** 2
        return 0.5 * (actions - outputs["mean"]) ** 2 / std**2 + std.log()


def bucket_losses(world_model, val_tensors, masks) -> dict[str, float]:
    with torch.no_grad():
        features = torch.cat(
            [world_model.encoder(val_tensors["states"]), world_model.encoder(val_tensors["next_clean"])],
            dim=-1,
        )
    nll = per_row_nll(world_model, features, val_tensors["actions"])
    out = {}
    for name, mask in masks.items():
        rows = nll[mask]
        out[f"nll_{name}"] = float(rows.mean()) if rows.numel() else float("nan")
    amb_desired = (val_tensors["next_clean"] - val_tensors["states"]).round(decimals=3)
    amb_mask = torch.isclose(amb_desired, torch.tensor(0.25), atol=0.01)
    if amb_mask.any():
        plan_amb = latent_planning_success(
            world_model.encoder, world_model.predictor,
            val_tensors["states"][amb_mask], val_tensors["next_clean"][amb_mask],
        )
        out["plan_ambiguous"] = plan_amb["planning_success"]
        det_mask = torch.isclose(amb_desired, torch.tensor(1.0), atol=0.01)
        if det_mask.any():
            plan_det = latent_planning_success(
                world_model.encoder, world_model.predictor,
                val_tensors["states"][det_mask], val_tensors["next_clean"][det_mask],
            )
            out["plan_deterministic"] = plan_det["planning_success"]
    return out


def main() -> None:
    config = QuadraticConfig(name="e1_mixed", forced_positive_magnitudes=(1.0,), seed=0)
    splits = generate_dataset(config)
    standardizer = state_standardizer(splits["train"])
    tr = tensors_from(splits["train"], "state_displacement", standardizer)
    va = tensors_from(splits["val"], "state_displacement", standardizer)

    desired = (va["next_clean"] - va["states"]).round(decimals=3)
    masks = {
        "ambiguous": torch.isclose(desired, torch.tensor(0.25), atol=0.01),
        "deterministic": torch.isclose(desired, torch.tensor(1.0), atol=0.01),
    }

    arms = {
        "gauss": ArmConfig(name="gauss", idm_kind="gaussian", idm_weight=0.3),
        "det": ArmConfig(name="det", idm_kind="deterministic", idm_weight=1.0),
        "mdn": ArmConfig(name="mdn", idm_kind="mdn", idm_weight=0.3),
    }
    traces: dict = {}
    for arm_name, arm_config in arms.items():
        for seed in SEEDS:
            torch.manual_seed(seed)
            model = WorldModel(arm_config)
            history: list[dict] = []

            def callback(wm, _epoch, history=history):
                entry = {"epoch": len(history)}
                entry.update(bucket_losses(wm, va, masks))
                history.append(entry)

            train_backbone(model, tr, BackboneConfig(), seed, None, callback)
            key = f"{arm_name}_seed{seed}"
            traces[key] = history
            first, last = history[0], history[-1]
            best_amb = min(h["nll_ambiguous"] for h in history)
            print(
                f"{key}: nll_amb {first['nll_ambiguous']:.3f} -> {last['nll_ambiguous']:.3f} "
                f"(best {best_amb:.3f}) | nll_det {first['nll_deterministic']:.3f} -> "
                f"{last['nll_deterministic']:.3f} | plan_amb {first.get('plan_ambiguous', float('nan')):.3f} -> "
                f"{last.get('plan_ambiguous', float('nan')):.3f}"
            )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(traces, indent=2))
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
