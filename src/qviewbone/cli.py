import argparse
from dataclasses import asdict
import json
from .analysis import analyze
from .dataset import prepare
from .inference import run
from .settings import load_settings


def main():
    parser = argparse.ArgumentParser(prog="qviewbone")
    parser.add_argument("--config")
    for name in ("data-dir", "image-dir", "prepared-dir", "output-dir", "pair-review"):
        parser.add_argument("--" + name)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("settings", help="Print resolved paths/settings without downloading weights")
    prep = sub.add_parser("prepare")
    prep.add_argument("--allow-unverified", action="store_true")
    prep.add_argument("--max-pairs", type=int)
    infer = sub.add_parser("run")
    infer.add_argument("--backend", choices=["dummy", "qwen"], default="dummy")
    infer.add_argument("--run-name", default="smoke")
    infer.add_argument("--split", choices=["all", "train", "val", "test"], default="all")
    infer.add_argument("--limit", type=int, default=0)
    infer.add_argument("--allow-unverified", action="store_true")
    result = sub.add_parser("analyze")
    result.add_argument("run_dir")
    result.add_argument("--aliases")
    args = parser.parse_args()
    try:
        settings = load_settings(args.config, {"paths": {k: getattr(args, k) for k in ("data_dir", "image_dir", "prepared_dir", "output_dir", "pair_review")}})
        if args.command == "settings":
            output = asdict(settings)
        elif args.command == "prepare":
            output = prepare(settings, args.allow_unverified, args.max_pairs)
        elif args.command == "run":
            output = run(settings, args.backend, args.run_name, args.split, args.limit, args.allow_unverified)
        else:
            output = analyze(args.run_dir, settings.experiment["bootstrap_samples"], settings.experiment["seed"], args.aliases)
        print(json.dumps(output, ensure_ascii=False, indent=2, default=str))
    except (ValueError, FileNotFoundError, ImportError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
