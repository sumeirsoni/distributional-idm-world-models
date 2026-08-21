"""Phase 0A implementation check: E1 data conditions times Phase 0A models."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from distributional_idm.envs.quadratic import (
    QuadraticConfig,
    conditional_law_table,
    generate_dataset,
)
from distributional_idm.evaluation.metrics import (
    gaussian_nll,
    known_forward_cycle_error_quadratic,
    mode_coverage_from_samples,
    mode_precision_from_samples,
    sample_mdn,
)
from distributional_idm.models.idm import (
    MDNIDM,
    DeterministicIDM,
    FixedVarianceGaussianIDM,
    GaussianIDM,
)
from distributional_idm.training import (
    TrainConfig,
    fit_standardizer,
    tensors_from,
    train_model,
)

RESULTS_DIR = Path("results/phase0a")
TOLERANCE = 0.05
N_SAMPLES = 16
SEEDS = (0, 1, 2)


def dataset_configs() -> list[QuadraticConfig]:
    return [
        QuadraticConfig(name="e1_symmetric", seed=0),
        QuadraticConfig(name="e1_asymmetric", positive_probabilities=(0.7, 0.7), seed=0),
        QuadraticConfig(name="e1_single_magnitude", magnitudes=(1.0,), seed=0),
        QuadraticConfig(name="e1_noisy", observation_noise_std=0.1, seed=0),
        QuadraticConfig(name="e10_control", nonnegative_actions=True, seed=0),
    ]


def make_model(name: str, input_dim: int) -> torch.nn.Module:
    if name == "deterministic":
        return DeterministicIDM(input_dim)
    if name == "state_only":
        return DeterministicIDM(input_dim=1)
    if name == "fixed_variance":
        return FixedVarianceGaussianIDM(input_dim)
    if name == "gaussian":
        return GaussianIDM(input_dim)
    if name == "mdn":
        return MDNIDM(input_dim, n_components=2)
    raise ValueError(name)


def train_config_for(name: str) -> TrainConfig:
    if name == "mdn":
        return TrainConfig(learning_rate=3e-4, patience=20, epochs=200)
    return TrainConfig()


def evaluate_model(model, tensors, conditioning: str, two_sided: bool = True) -> dict:
    model.eval()
    with torch.no_grad():
        outputs = model.predict(tensors["features"])
        actions = tensors["actions"]
        metrics: dict[str, float] = {}
        if "logits" in outputs:
            from distributional_idm.evaluation.metrics import mdn_component_probabilities

            weights = mdn_component_probabilities(outputs["logits"])
            mean = (weights * outputs["means"]).sum(dim=-1)
        else:
            mean = outputs["mean"]
        metrics["mean_abs_error"] = float((mean - actions).abs().mean())
        metrics["mean_prediction_abs_mean"] = float(mean.abs().mean())

        modes = actions.abs().clamp_min(TOLERANCE * 2)
        low_modes, high_modes = (-modes, modes) if two_sided else (modes, modes)
        tol = torch.full_like(modes, TOLERANCE).unsqueeze(-1)

        if "std" in outputs and outputs["std"].dim() == 1 and "logits" not in outputs:
            from distributional_idm.evaluation.metrics import heteroscedastic_invalid_mass

            metrics["nll"] = gaussian_nll(outputs["mean"], outputs["std"], actions)
            metrics["invalid_mass"] = heteroscedastic_invalid_mass(
                outputs["mean"].unsqueeze(-1),
                outputs["std"].unsqueeze(-1),
                low_modes.unsqueeze(-1),
                high_modes.unsqueeze(-1),
                tol,
                second_band=two_sided,
            )

        if "logits" in outputs:
            from distributional_idm.evaluation.metrics import (
                invalid_between_mode_mass,
                mdn_nll,
            )

            logits = outputs["logits"]
            means = outputs["means"]
            stds = outputs["stds"]
            metrics["nll"] = mdn_nll(logits, means, stds, actions)
            metrics["invalid_mass"] = invalid_between_mode_mass(
                logits,
                means,
                stds,
                low_modes.unsqueeze(-1),
                high_modes.unsqueeze(-1),
                tol,
                second_band=two_sided,
            )
            generator = torch.Generator().manual_seed(1234)
            samples = sample_mdn(logits, means, stds, N_SAMPLES, generator)
            metrics["mode_coverage"] = mode_coverage_from_samples(
                samples,
                low_modes.unsqueeze(-1),
                high_modes.unsqueeze(-1),
                tol,
                second_band=two_sided,
            )
            metrics["mode_precision"] = mode_precision_from_samples(
                samples,
                low_modes.unsqueeze(-1),
                high_modes.unsqueeze(-1),
                tol,
                second_band=two_sided,
            )
            metrics["cycle_error_samples"] = known_forward_cycle_error_quadratic(
                tensors["states"], samples, tensors["next_clean"]
            )

        metrics["cycle_error_point"] = known_forward_cycle_error_quadratic(
            tensors["states"], mean.unsqueeze(-1), tensors["next_clean"]
        )
        del conditioning
        return metrics


MDN_RESTARTS = 5


def fit_model(model_name: str, input_dim: int, train_tensors, val_tensors, seed: int):
    torch.manual_seed(seed)
    model = make_model(model_name, input_dim)
    config = train_config_for(model_name)
    restarts = MDN_RESTARTS if model_name == "mdn" else 1
    best_model, best_val = model, float("inf")
    history = []
    for restart in range(restarts):
        candidate = make_model(model_name, input_dim) if restart else model
        hist = train_model(candidate, train_tensors, val_tensors, config, seed * 100 + restart)
        if hist[-1] < best_val:
            best_val = hist[-1]
            best_model = candidate
        history = hist
    return best_model, history


def run() -> dict:
    all_results: dict = {"tolerance": TOLERANCE, "seeds": list(SEEDS), "datasets": {}}
    for config in dataset_configs():
        splits = generate_dataset(config)
        standardizer_source = splits["train"]
        law = conditional_law_table(splits["test"])
        dataset_entry: dict = {
            "conditional_law_test": {str(k): v for k, v in law.items()},
            "models": {},
        }
        for model_name in ("deterministic", "state_only", "fixed_variance", "gaussian", "mdn"):
            conditioning = "state_only" if model_name == "state_only" else "state_displacement"
            standardizer = fit_standardizer(standardizer_source, conditioning)
            train_tensors = tensors_from(splits["train"], conditioning, standardizer)
            val_tensors = tensors_from(splits["val"], conditioning, standardizer)
            test_tensors = tensors_from(splits["test"], conditioning, standardizer)

            per_seed = []
            for seed in SEEDS:
                start_time = time.time()
                model, _ = fit_model(model_name, train_tensors["features"].shape[-1], train_tensors, val_tensors, seed)
                entry = evaluate_model(
                    model, test_tensors, conditioning, two_sided=not config.nonnegative_actions
                )
                entry["seconds"] = round(time.time() - start_time, 2)
                per_seed.append(entry)

            aggregated: dict[str, float] = {}
            for key in per_seed[0]:
                values = [seed_result[key] for seed_result in per_seed]
                aggregated[key] = sum(values) / len(values)
                aggregated[f"{key}_std"] = (sum((v - aggregated[key]) ** 2 for v in values) / len(values)) ** 0.5
            dataset_entry["models"][model_name] = aggregated
            print(f"[{config.name}] {model_name}: " + ", ".join(f"{k}={v:.4f}" for k, v in aggregated.items()))
        all_results["datasets"][config.name] = dataset_entry

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "results.json"
    output_path.write_text(json.dumps(all_results, indent=2))
    print(f"wrote {output_path}")
    return all_results


if __name__ == "__main__":
    run()
