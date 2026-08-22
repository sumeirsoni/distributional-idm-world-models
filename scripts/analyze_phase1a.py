"""Gate 2 analysis: paired t-tests with Holm correction over the decisive core.

Family for multiplicity correction: every (arm, component, dataset) test in
the decisive core - six arms times four endpoints (three components plus the
composite mean) times two datasets. Positive difference means the arm beats
the no-IDM arm; error components are sign-flipped before testing so positive
always means better. Advance requires Holm-adjusted p < 0.05 and a
standardized effect of at least 0.2 (delta divided by the no-IDM arm's seed
SD), per the pre-registration.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

RESULTS_PATH = Path("results/phase1a/results.json")
OUTPUT_PATH = Path("results/phase1a/gate_analysis.json")

COMPONENTS = {
    "probe_r2_controllable": 1.0,
    "planning_success": 1.0,
    "posthoc_mdn_cycle": -1.0,
}
ARMS_EXCLUDED = ("no_idm",)


def paired_differences(rows: list[dict], arm: str, key_fn) -> np.ndarray:
    baseline = {row["seed"]: key_fn(row) for row in rows if row["arm"] == "no_idm"}
    target = {row["seed"]: key_fn(row) for row in rows if row["arm"] == arm}
    return np.array(
        [key_direction(target[s]) - key_direction(baseline[s]) for s in sorted(baseline)]
    )


def key_direction(value: float | None) -> float:
    return 0.0 if value is None else float(value)


def endpoint_per_seed(row: dict) -> float:
    values = []
    for component, direction in COMPONENTS.items():
        value = row.get(component)
        values.append(direction * key_direction(value))
    return float(np.mean(values))


def holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = [0.0] * len(p_values)
    running = 0.0
    n = len(p_values)
    for rank, index in enumerate(order):
        corrected = (n - rank) * p_values[index]
        running = max(running, corrected)
        adjusted[index] = min(1.0, running)
    return adjusted


def main() -> None:
    results = json.loads(RESULTS_PATH.read_text())
    tests: list[dict] = []

    for dataset_name, entry in results["datasets"].items():
        rows = entry["per_seed"]
        baseline_sd = {}
        for component in list(COMPONENTS) + ["endpoint"]:
            key_fn = (
                (lambda row, c=component: row.get(c))
                if component != "endpoint"
                else endpoint_per_seed
            )
            base_values = np.array(
                [key_direction(key_fn(row)) for row in rows if row["arm"] == "no_idm"]
            )
            baseline_sd[component] = max(float(np.std(base_values, ddof=1)), 1e-6)

        for arm in entry["arms"]:
            if arm in ARMS_EXCLUDED:
                continue
            for component, direction in list(COMPONENTS.items()) + [("endpoint", None)]:
                key_fn = (
                    (lambda row, c=component: row.get(c))
                    if component != "endpoint"
                    else endpoint_per_seed
                )
                diffs = paired_differences(rows, arm, key_fn)
                if diffs.size < 2:
                    continue
                t_stat, p_value = stats.ttest_1samp(diffs, 0.0)
                signed_direction = 1.0 if direction is None else direction
                effect = signed_direction * diffs.mean() / baseline_sd[component]
                tests.append(
                    {
                        "dataset": dataset_name,
                        "arm": arm,
                        "component": component,
                        "mean_delta": round(float(diffs.mean()), 5),
                        "t": round(float(t_stat), 3),
                        "p": round(float(p_value), 5),
                        "effect_d": round(float(effect), 3),
                    }
                )

    adjusted = holm([test["p"] for test in tests])
    verdicts = []
    for test, adj in zip(tests, adjusted):
        test["p_holm"] = round(adj, 5)
        test["significant"] = bool(adj < 0.05)
        test["meets_effect_floor"] = bool(abs(test["effect_d"]) >= 0.2)
        test["advances"] = bool(
            test["effect_d"] > 0 and test["significant"] and abs(test["effect_d"]) >= 0.2
        )
        if test["advances"] or (test["significant"] and not test["advances"]):
            verdicts.append(test)

    output = {"family_size": len(tests), "tests": tests, "notable": verdicts}
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"wrote {OUTPUT_PATH} ({len(tests)} tests)")

    print("\nTests that clear both bars (Holm p<0.05, d>=+0.2):")
    for test in tests:
        if test["advances"]:
            print(f"  {test['dataset']}/{test['arm']}/{test['component']}: "
                  f"d={test['effect_d']} p_holm={test['p_holm']}")
    print("\nSignificant but failing a bar:")
    for test in verdicts:
        if not test["advances"]:
            print(f"  {test['dataset']}/{test['arm']}/{test['component']}: "
                  f"d={test['effect_d']} p_holm={test['p_holm']}")


if __name__ == "__main__":
    main()
