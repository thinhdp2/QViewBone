from collections import defaultdict
import json
from pathlib import Path
import random
import re
import unicodedata
from .io import digest, read_jsonl, write_csv, write_json


def normalize_answer(value):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).casefold()).strip(" \t\r\n.,;:")


def parse_prediction(raw):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    try:
        value = json.loads(raw)["answer"]
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list) or not value or not all(isinstance(v, str) and v.strip() for v in value):
            return None
        return value
    except (ValueError, TypeError, KeyError):
        return None


def exact_set_match(prediction, gold, aliases=None):
    if prediction is None:
        return False
    aliases = {normalize_answer(k): normalize_answer(v) for k, v in (aliases or {}).items()}
    def normalized(values):
        return {aliases.get(normalize_answer(v), normalize_answer(v)) for v in values}
    return normalized(prediction) == normalized(gold)


def cluster_ci(rows, key, samples, seed):
    clusters = defaultdict(list)
    for row in rows:
        clusters[row["patient_id"]].append(row[key])
    patients = sorted(clusters)
    if len(patients) < 2 or samples < 1:
        return None
    rng = random.Random(seed)
    results = []
    for _ in range(samples):
        sample = [value for p in rng.choices(patients, k=len(patients)) for value in clusters[p]]
        results.append(sum(sample) / len(sample))
    results.sort()
    return [results[int(.025 * (len(results) - 1))], results[int(.975 * (len(results) - 1))]]


def analyze(run_dir, bootstrap_samples=1000, seed=42, aliases_path=None):
    run_dir = Path(run_dir)
    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    rows = read_jsonl(run_dir / "predictions.jsonl")
    if not rows:
        raise ValueError("No predictions to analyze")
    aliases = json.loads(Path(aliases_path).read_text(encoding="utf-8")) if aliases_path else {}
    groups = defaultdict(dict)
    scored = []
    for r in rows:
        if r["run_fingerprint"] != metadata["fingerprint"]:
            raise ValueError("Mismatched run fingerprint")
        if r["condition"] in groups[r["task_id"]]:
            raise ValueError("Duplicate task/condition")
        parsed = parse_prediction(r["raw_prediction"])
        score = {**r, "prediction": parsed, "parse_valid": parsed is not None, "correct": exact_set_match(parsed, r["gold"], aliases)}
        groups[r["task_id"]][r["condition"]] = score
        scored.append(score)
    required = set(metadata["conditions"])
    if "A_to_B" in required:
        required.add("A")
    if "B_to_A" in required:
        required.add("B")
    if any(not required <= set(g) for g in groups.values()):
        raise ValueError("Some tasks are incomplete. Resume the same run before analyzing")
    # Detect missing whole tasks as well as missing conditions.
    expected_task_count = metadata.get("task_count")
    if expected_task_count is not None and len(groups) != expected_task_count:
        raise ValueError("Run is incomplete; resume before analyzing")
    metrics = []
    for category in ["ALL"] + sorted({r["category"] for r in scored}):
        for condition in sorted(required):
            subset = [r for r in scored if r["condition"] == condition and (category == "ALL" or r["category"] == category)]
            metrics.append({"category": category, "condition": condition, "n": len(subset),
                            "exact_set_accuracy": sum(r["correct"] for r in subset) / len(subset),
                            "parse_failure_rate": sum(not r["parse_valid"] for r in subset) / len(subset),
                            "mean_views": sum(r["views"] for r in subset) / len(subset)})
    utility = []
    for task_id, conditions in groups.items():
        for before, after in (("A", "A_to_B"), ("B", "B_to_A")):
            if before not in conditions or after not in conditions:
                continue
            initial, final = conditions[before], conditions[after]
            a, b = int(initial["correct"]), int(final["correct"])
            utility.append({"task_id": task_id, "pair_id": initial["pair_id"], "patient_id": initial["patient_id"],
                            "category": initial["category"], "direction": after, "correct_before": a, "correct_after": b,
                            "delta": b - a, "outcome": {(0, 1): "rescue", (1, 0): "harm", (1, 1): "preserved", (0, 0): "unresolved"}[a, b]})
    utility_summary = []
    for direction in sorted({r["direction"] for r in utility}):
        for category in ["ALL"] + sorted({r["category"] for r in utility}):
            subset = [r for r in utility if r["direction"] == direction and (category == "ALL" or r["category"] == category)]
            if not subset:
                continue
            wrong = sum(not r["correct_before"] for r in subset)
            right = len(subset) - wrong
            rescue = sum(r["outcome"] == "rescue" for r in subset)
            harm = sum(r["outcome"] == "harm" for r in subset)
            utility_summary.append({"direction": direction, "category": category, "n": len(subset),
                                    "delta_accuracy": sum(r["delta"] for r in subset) / len(subset),
                                    "rescue_rate_all": rescue / len(subset), "harm_rate_all": harm / len(subset),
                                    "rescue_given_initial_wrong": rescue / wrong if wrong else None,
                                    "harm_given_initial_right": harm / right if right else None,
                                    "delta_patient_bootstrap_ci95": cluster_ci(subset, "delta", bootstrap_samples, seed)})
    order = []
    for task_id, conditions in groups.items():
        for first, second in (("AB", "BA"), ("A_to_B", "B_to_A")):
            if first in conditions and second in conditions:
                left, right = conditions[first], conditions[second]
                comparable = left["prediction"] is not None and right["prediction"] is not None
                def answer_set(values):
                    lookup = {normalize_answer(k): normalize_answer(v) for k, v in aliases.items()}
                    return {lookup.get(normalize_answer(v), normalize_answer(v)) for v in values}
                order.append({"task_id": task_id, "patient_id": left["patient_id"], "category": left["category"],
                              "comparison": f"{first}_vs_{second}", "correct_first": left["correct"], "correct_second": right["correct"],
                              "answer_comparable": comparable,
                              "answer_changed": answer_set(left["prediction"]) != answer_set(right["prediction"]) if comparable else None})
    write_csv(run_dir / "metrics.csv", metrics, ["category", "condition", "n", "exact_set_accuracy", "parse_failure_rate", "mean_views"])
    write_csv(run_dir / "view_utility.csv", utility, ["task_id", "pair_id", "patient_id", "category", "direction", "correct_before", "correct_after", "delta", "outcome"])
    write_csv(run_dir / "order_effects.csv", order, ["task_id", "patient_id", "category", "comparison", "correct_first", "correct_second", "answer_comparable", "answer_changed"])
    summary = {"result_kind": metadata["result_kind"], "scorer": "exact_normalized_answer_set_v1", "aliases_fingerprint": digest(aliases),
               "tasks": len(groups), "patients": len({r["patient_id"] for r in scored}), "utility": utility_summary,
               "interpretation": "Smoke-test outputs; not model evaluation" if metadata["result_kind"] == "dry_run" else "Lexical matching only; diagnosis synonyms require adjudication"}
    write_json(run_dir / "summary.json", summary)
    return summary
