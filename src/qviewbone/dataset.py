from collections import Counter
import hashlib
from pathlib import Path
from .io import digest, file_hash, read_csv, read_jsonl, write_csv, write_json, write_jsonl

IMAGE_FILE = "qviewbone_approved_image_records.jsonl"
PAIR_FILE = "qviewbone_candidate_pairs.csv"
CROSS_FILE = "qviewbone_crossview_candidates.jsonl"


def is_true(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


def patient_split(patient, seed, train_fraction=.7, val_fraction=.15):
    # Independent of row order, categories and which patients are later added.
    value = int(hashlib.sha256(f"{seed}:{patient}".encode()).hexdigest()[:16], 16) / 2**64
    return "train" if value < train_fraction else "val" if value < train_fraction + val_fraction else "test"


def pair_is_verified(row):
    return (is_true(row.get("same_examination_verified"))
            and is_true(row.get("pair_confirmed_same_lesion"))
            and row.get("status", "").strip().lower() in {"verified", "approved"}
            and bool(row.get("approved_categories", "").strip()))


def checked_image(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Image path leaves image_dir")
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {relative}")
    return path


def prepare(settings, allow_unverified=False, max_pairs=None):
    images = read_jsonl(settings.data_dir / IMAGE_FILE)
    pairs = read_csv(settings.data_dir / PAIR_FILE)
    episodes = read_jsonl(settings.data_dir / CROSS_FILE)
    settings.prepared_dir.mkdir(parents=True, exist_ok=True)
    review_fields = list(pairs[0]) + ["approved_categories", "reviewer", "reviewed_at", "review_note"] if pairs else []
    if not settings.pair_review.exists():
        write_csv(settings.pair_review, pairs, review_fields)
    reviews = read_csv(settings.pair_review)
    review_map = {r["pair_id"]: r for r in reviews}
    if len(review_map) != len(reviews):
        raise ValueError("Duplicate pair_id in pair review file")
    image_map = {str(r["image_id"]): r for r in images}
    if len(image_map) != len(images):
        raise ValueError("Duplicate image_id")
    manifest = []
    for image_id, r in image_map.items():
        relative = next((f"{image_id}{ext}" for ext in (".jpg", ".jpeg", ".png", ".webp") if (settings.image_dir / f"{image_id}{ext}").is_file()), "")
        manifest.append({"image_id": image_id, "patient_id": str(r["patient_id"]), "case_id": str(r["case_id"]), "relative_path": relative, "sha256": file_hash(settings.image_dir / relative) if relative else ""})
    manifest_map = {r["image_id"]: r for r in manifest}
    write_csv(settings.prepared_dir / "images_manifest.csv", manifest, ["image_id", "patient_id", "case_id", "relative_path", "sha256"])
    pair_index = {tuple(sorted((r["image_a"], r["image_b"]))): r for r in pairs}
    if len(pair_index) != len(pairs):
        raise ValueError("Duplicate image pair in source CSV")
    allowed_pairs = sorted(pairs, key=lambda r: digest([settings.experiment["seed"], r["pair_id"]]))
    allowed_pairs = [p for p in allowed_pairs if allow_unverified or pair_is_verified(review_map.get(p["pair_id"], {}))]
    limit = settings.experiment["max_pairs"] if max_pairs is None else max_pairs
    if limit < 0:
        raise ValueError("max_pairs must be nonnegative")
    selected_pairs = {p["pair_id"] for p in (allowed_pairs[:limit] if limit else allowed_pairs)}
    tasks, seen, rejected = [], {}, Counter()
    for episode in episodes:
        category = episode["question_category"]
        if category not in settings.experiment["categories"]:
            rejected["category_excluded"] += 1
            continue
        pair = pair_index.get(tuple(sorted(map(str, episode["image_ids"]))))
        if not pair:
            raise ValueError("Episode references an unknown image pair")
        if pair["pair_id"] not in selected_pairs:
            rejected["pair_unverified_or_outside_limit"] += 1
            continue
        review = review_map.get(pair["pair_id"], {})
        for key in ("image_a", "image_b", "patient_id", "case_id"):
            if review.get(key) != pair[key]:
                raise ValueError(f"Review identity changed for {pair['pair_id']}: {key}")
        approved = {c.strip() for c in review.get("approved_categories", "").split(";") if c.strip()}
        verified = pair_is_verified(review) and category in approved
        if not verified and not allow_unverified:
            rejected["question_not_approved"] += 1
            continue
        a, b = str(pair["image_a"]), str(pair["image_b"])
        for image_id in (a, b):
            if image_id not in image_map:
                raise ValueError("Pair references an unknown image")
            source = image_map[image_id]
            if str(source["patient_id"]) != pair["patient_id"] or str(source["case_id"]) != pair["case_id"]:
                raise ValueError("Image/pair patient or case mismatch")
        if str(episode["patient_id"]) != pair["patient_id"] or str(episode["case_id"]) != pair["case_id"]:
            raise ValueError("Episode/pair identity mismatch")
        if not manifest_map[a]["relative_path"] or not manifest_map[b]["relative_path"]:
            rejected["missing_image"] += 1
            continue
        key = (pair["pair_id"], category)
        signature = digest([episode["question"], episode["gold"]])
        if key in seen:
            if seen[key] != signature:
                raise ValueError("Conflicting gold/question between forward and reverse episodes")
            rejected["reverse_episode_deduplicated"] += 1
            continue
        seen[key] = signature
        patient = pair["patient_id"]
        task = {"task_id": f"{pair['pair_id']}_{category}", "pair_id": pair["pair_id"], "patient_id": patient,
                "case_id": pair["case_id"], "image_ids": [a, b], "category": category,
                "question": episode["question"], "gold": episode["gold"], "pathology_group": pair["pathology_group"],
                "images": [manifest_map[i]["relative_path"] for i in (a, b)],
                "image_sha256": [manifest_map[i]["sha256"] for i in (a, b)],
                "verified": verified, "split": patient_split(patient, settings.experiment["seed"], settings.experiment["train_fraction"], settings.experiment["val_fraction"])}
        tasks.append(task)
    tasks.sort(key=lambda r: r["task_id"])
    write_jsonl(settings.prepared_dir / "eval_dataset.jsonl", tasks)
    patients = {str(r["patient_id"]) for r in images}
    write_json(settings.prepared_dir / "patient_splits.json", {p: patient_split(p, settings.experiment["seed"], settings.experiment["train_fraction"], settings.experiment["val_fraction"]) for p in sorted(patients)})
    report = {"source_images": len(images), "available_images": sum(bool(r["relative_path"]) for r in manifest),
              "source_pairs": len(pairs), "verified_pairs": sum(pair_is_verified(r) for r in reviews),
              "source_episodes": len(episodes), "tasks": len(tasks), "unique_pairs": len({r["pair_id"] for r in tasks}),
              "patients": len({r["patient_id"] for r in tasks}), "category_counts": dict(Counter(r["category"] for r in tasks)),
              "exploratory": any(not r["verified"] for r in tasks), "rejected_episode_counts": dict(rejected),
              "dataset_fingerprint": digest(tasks)}
    write_json(settings.prepared_dir / "preparation_report.json", report)
    return report
