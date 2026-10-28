from dataclasses import dataclass
from pathlib import Path
import os
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


@dataclass
class Settings:
    root: Path
    data_dir: Path
    image_dir: Path
    prepared_dir: Path
    output_dir: Path
    pair_review: Path
    model: dict
    experiment: dict


def load_settings(config=None, overrides=None):
    root = Path(os.environ.get("QVB_ROOT", Path(__file__).resolve().parents[2])).expanduser().resolve()
    config_path = Path(config).expanduser() if config else root / "configs/experiment.toml"
    if not config_path.is_absolute():
        config_path = root / config_path
    with config_path.open("rb") as stream:
        values = tomllib.load(stream)
    values = {k: dict(v) for k, v in values.items()}
    mapping = {
        "DATA_DIR": ("paths", "data_dir", str), "IMAGE_DIR": ("paths", "image_dir", str),
        "PREPARED_DIR": ("paths", "prepared_dir", str), "OUTPUT_DIR": ("paths", "output_dir", str),
        "PAIR_REVIEW": ("paths", "pair_review", str), "MODEL_ID": ("model", "id", str),
        "DEVICE": ("model", "device", str), "DTYPE": ("model", "dtype", str),
        "QUANTIZATION": ("model", "quantization", str), "MAX_PIXELS": ("model", "max_pixels", int),
        "MAX_NEW_TOKENS": ("model", "max_new_tokens", int),
    }
    for name, (section, key, cast) in mapping.items():
        value = os.environ.get("QVB_" + name)
        if value:
            values[section][key] = cast(value)
    # Explicit API/CLI overrides take precedence over environment and TOML.
    for section, items in (overrides or {}).items():
        values[section].update({k: v for k, v in items.items() if v is not None})
    paths, model, experiment = values["paths"], values["model"], values["experiment"]
    def resolve(value):
        path = Path(os.path.expandvars(str(value))).expanduser()
        return (path if path.is_absolute() else root / path).resolve()
    sibling = root.parent / "vqa-dataset"
    data_dir = resolve(paths["data_dir"]) if paths["data_dir"] else sibling if sibling.is_dir() else root / "data/raw"
    image_dir = resolve(paths["image_dir"]) if paths["image_dir"] else data_dir / "image-cache"
    prepared_dir = resolve(paths["prepared_dir"])
    pair_review = resolve(paths["pair_review"]) if paths["pair_review"] else prepared_dir / "pair_review.csv"
    if model["device"] not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu or cuda")
    if model["dtype"] not in {"auto", "float32", "float16", "bfloat16"}:
        raise ValueError("Unsupported dtype")
    if model["quantization"] not in {"none", "4bit"}:
        raise ValueError("quantization must be none or 4bit")
    if not 0 < model["min_pixels"] <= model["max_pixels"] or model["max_new_tokens"] <= 0:
        raise ValueError("Invalid image/token budget")
    if not 0 < experiment["train_fraction"] < 1 or not 0 <= experiment["val_fraction"] < 1 - experiment["train_fraction"]:
        raise ValueError("Invalid patient split fractions")
    if "C1.4" in experiment["categories"]:
        raise ValueError("C1.4 needs per-image grounding evaluation and is excluded from this text runner")
    allowed = {"A", "B", "AB", "BA", "A_to_B", "B_to_A"}
    if not set(experiment["conditions"]) <= allowed or not experiment["conditions"]:
        raise ValueError("Invalid conditions")
    return Settings(root, data_dir, image_dir, prepared_dir, resolve(paths["output_dir"]), pair_review, model, experiment)
