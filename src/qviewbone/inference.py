from datetime import datetime, timezone
from importlib import metadata
import json
import platform
from pathlib import Path
import re
import shutil
import subprocess
import time
from .dataset import checked_image
from .io import append_jsonl, digest, file_hash, read_jsonl, recover_predictions, write_json
from .prompts import PROMPT_VERSION, first_turn, sequential


def hardware_settings(model):
    import torch
    device = model["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; select a GPU runtime in Colab")
    dtype = model["dtype"]
    if dtype == "auto":
        dtype = "bfloat16" if device == "cuda" and torch.cuda.is_bf16_supported() else "float16" if device == "cuda" else "float32"
    if device == "cpu" and dtype != "float32":
        raise ValueError("Use float32 for the supported CPU path")
    if device == "cuda" and dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support bfloat16; use auto or float16")
    if model["quantization"] == "4bit" and device != "cuda":
        raise ValueError("The supported 4-bit path requires CUDA")
    return {"device": device, "dtype": dtype, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2) if device == "cuda" else None}


class DummyBackend:
    """Smoke testing only; receives prompts, never task rows or reference answers."""
    def generate(self, messages):
        return '{"answer": ["dry-run"]}'


class QwenBackend:
    def __init__(self, model, runtime, seed):
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
        self.torch = torch
        torch.manual_seed(seed)
        dtype = getattr(torch, runtime["dtype"])
        kwargs = {"revision": model["revision"], "dtype": dtype, "attn_implementation": model["attention"],
                  "device_map": {"": 0} if runtime["device"] == "cuda" else {"": "cpu"}, "trust_remote_code": False}
        if model["quantization"] == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model["id"], **kwargs).eval()
        self.processor = AutoProcessor.from_pretrained(model["id"], revision=model["revision"], min_pixels=model["min_pixels"], max_pixels=model["max_pixels"], trust_remote_code=False)
        self.max_new_tokens = model["max_new_tokens"]
        self.resolved_revision = getattr(self.model.config, "_commit_hash", None)

    def generate(self, messages):
        # apply_chat_template reads each local image path, including earlier turns.
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False, use_cache=True)
        generated = outputs[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def run(settings, backend="dummy", run_name="smoke", split="all", limit=0, allow_unverified=False):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_name):
        raise ValueError("run_name must be a simple folder name")
    if backend not in {"dummy", "qwen"}:
        raise ValueError("Unknown backend")
    tasks = read_jsonl(settings.prepared_dir / "eval_dataset.jsonl")
    tasks = [r for r in tasks if split == "all" or r["split"] == split]
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    if limit:
        tasks = tasks[:limit]
    if not tasks:
        raise ValueError("No eligible tasks. Review pair_review.csv and run prepare, or explicitly prepare exploratory data with --allow-unverified")
    if any(not r["verified"] for r in tasks) and not allow_unverified:
        raise ValueError("Unverified tasks require --allow-unverified; results will be exploratory")
    for task in tasks:
        for relative, expected_hash in zip(task["images"], task["image_sha256"]):
            if file_hash(checked_image(settings.image_dir, relative)) != expected_hash:
                raise ValueError("Image changed after preparation; prepare again and use a new run name")
    runtime = hardware_settings(settings.model) if backend == "qwen" else {"device": "cpu", "dtype": "none", "gpu": None}
    versions = {"python": platform.python_version()}
    for name in ("torch", "transformers", "accelerate", "Pillow", "bitsandbytes"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    source_hash = digest({name: file_hash(Path(__file__).with_name(name)) for name in ("inference.py", "prompts.py")})
    signature = {"tasks": digest(tasks), "inference_source_hash": source_hash, "model": settings.model if backend == "qwen" else {}, "backend": backend,
                 "seed": settings.experiment["seed"], "conditions": settings.experiment["conditions"],
                 "prompt_version": PROMPT_VERSION, "runtime": runtime, "versions": versions}
    fingerprint = digest(signature)
    folder = settings.output_dir / run_name
    folder.mkdir(parents=True, exist_ok=True)
    run_file, prediction_file = folder / "run.json", folder / "predictions.jsonl"
    if run_file.exists():
        previous = json.loads(run_file.read_text(encoding="utf-8"))
        if previous["fingerprint"] != fingerprint:
            raise ValueError("This run has a different dataset/config/runtime. Use a new --run-name; never mix experiments")
    elif prediction_file.exists():
        raise ValueError("Predictions exist without run.json; use a new run name")
    else:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=settings.root, capture_output=True, text=True, check=False).stdout.strip() if shutil.which("git") else None
        write_json(run_file, {"fingerprint": fingerprint, **signature, "task_count": len(tasks), "created_at": datetime.now(timezone.utc).isoformat(),
                             "git_commit": commit, "result_kind": "dry_run" if backend == "dummy" else "exploratory" if any(not t["verified"] for t in tasks) else "verified_pairs",
                             "paths": {"data": str(settings.data_dir), "images": str(settings.image_dir)}, "resolved_model_revision": None})
    existing = recover_predictions(prediction_file)
    done = {(r["task_id"], r["condition"]): r for r in existing}
    if len(done) != len(existing) or any(r["run_fingerprint"] != fingerprint for r in existing):
        raise ValueError("Duplicate or mismatched prediction rows")
    conditions = list(settings.experiment["conditions"])
    # Sequential experiments reuse the actual single-image model outputs.
    if "A_to_B" in conditions and "A" not in conditions:
        conditions.insert(0, "A")
    if "B_to_A" in conditions and "B" not in conditions:
        conditions.insert(0, "B")
    conditions = sorted(conditions, key=lambda c: ["A", "B", "AB", "BA", "A_to_B", "B_to_A"].index(c))
    engine = None
    generated = 0
    for task in tasks:
        a, b = [checked_image(settings.image_dir, relative) for relative in task["images"]]
        for condition in conditions:
            key = (task["task_id"], condition)
            if key in done:
                continue
            if engine is None:
                engine = DummyBackend() if backend == "dummy" else QwenBackend(settings.model, runtime, settings.experiment["seed"])
                if backend == "qwen":
                    stored = json.loads(run_file.read_text(encoding="utf-8"))
                    if stored["resolved_model_revision"] and stored["resolved_model_revision"] != engine.resolved_revision:
                        raise ValueError("Model revision changed on the Hub. Pin the original commit or start a new run")
                    stored["resolved_model_revision"] = engine.resolved_revision
                    write_json(run_file, stored)
            if condition in {"A_to_B", "B_to_A"}:
                first = "A" if condition == "A_to_B" else "B"
                messages = sequential(task["question"], a if first == "A" else b, b if first == "A" else a, done[(task["task_id"], first)]["raw_prediction"])
            else:
                paths = {"A": [a], "B": [b], "AB": [a, b], "BA": [b, a]}[condition]
                messages = first_turn(task["question"], paths)
            start = time.perf_counter()
            raw = engine.generate(messages)
            row = {**task, "condition": condition, "raw_prediction": raw, "seconds": time.perf_counter() - start,
                   "views": 1 if condition in {"A", "B"} else 2, "run_fingerprint": fingerprint, "backend": backend}
            append_jsonl(prediction_file, row)
            done[key] = row
            generated += 1
            print(f"Completed {len(done)}/{len(tasks) * len(conditions)} outputs", flush=True)
    return {"run_dir": str(folder), "new_predictions": generated, "total_predictions": len(done), "dry_run": backend == "dummy"}
