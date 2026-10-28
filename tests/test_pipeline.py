from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from qviewbone.analysis import analyze, exact_set_match, parse_prediction
from qviewbone.dataset import checked_image, pair_is_verified, patient_split, prepare
from qviewbone.inference import hardware_settings, run
from qviewbone.io import read_csv, read_jsonl, recover_predictions, write_csv, write_json, write_jsonl
from qviewbone.settings import load_settings

ROOT = Path(__file__).resolve().parents[1]


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.raw = root / "raw"
        self.raw.mkdir()
        self.image_dir = self.raw / "image-cache"
        self.image_dir.mkdir()
        for name in ("1.jpg", "2.jpg"):
            (self.image_dir / name).write_bytes(b"fixture-image-content")
        images = [{"image_id": i, "patient_id": "p1", "case_id": "c1"} for i in ("1", "2")]
        write_jsonl(self.raw / "qviewbone_approved_image_records.jsonl", images)
        pair = {"pair_id": "pair_1", "patient_id": "p1", "case_id": "c1", "image_a": "1", "image_b": "2", "pathology_group": "U xương", "same_examination_verified": "False", "pair_confirmed_same_lesion": "False", "status": "candidate"}
        write_csv(self.raw / "qviewbone_candidate_pairs.csv", [pair], list(pair))
        episodes = [{"image_ids": ids, "patient_id": "p1", "case_id": "c1", "question_category": "C1.1", "question": "Cấu trúc nào?", "gold": ["GOLD_NEVER_IN_PROMPT"]} for ids in (["1", "2"], ["2", "1"])]
        write_jsonl(self.raw / "qviewbone_crossview_candidates.jsonl", episodes)
        self.cfg = load_settings(ROOT / "configs/experiment.toml", {"paths": {"data_dir": str(self.raw), "image_dir": str(self.image_dir), "prepared_dir": str(root / "prepared"), "output_dir": str(root / "outputs"), "pair_review": str(root / "prepared/pair_review.csv")}})

    def tearDown(self):
        self.temp.cleanup()

    def test_default_gates_unverified_and_deduplicates_reverse(self):
        report = prepare(self.cfg)
        self.assertEqual(report["tasks"], 0)
        report = prepare(self.cfg, allow_unverified=True)
        self.assertEqual(report["tasks"], 1)
        self.assertEqual(report["rejected_episode_counts"]["reverse_episode_deduplicated"], 1)
        with self.assertRaisesRegex(ValueError, "Unverified"):
            run(self.cfg)

    def test_question_level_approval_and_review_identity(self):
        prepare(self.cfg)
        rows = read_csv(self.cfg.pair_review)
        rows[0].update(same_examination_verified="True", pair_confirmed_same_lesion="True", status="verified", approved_categories="C1.2")
        write_csv(self.cfg.pair_review, rows, list(rows[0]))
        self.assertTrue(pair_is_verified(rows[0]))
        self.assertEqual(prepare(self.cfg)["tasks"], 0)
        rows[0]["approved_categories"] = "C1.1"
        write_csv(self.cfg.pair_review, rows, list(rows[0]))
        self.assertEqual(prepare(self.cfg)["tasks"], 1)
        rows[0]["image_a"] = "999"
        write_csv(self.cfg.pair_review, rows, list(rows[0]))
        with self.assertRaisesRegex(ValueError, "identity changed"):
            prepare(self.cfg)

    def test_sequential_history_no_gold_and_resume(self):
        prepare(self.cfg, allow_unverified=True)
        calls = []
        def generate(engine, messages):
            calls.append(messages)
            return '{"answer": ["model prediction"]}'
        with patch("qviewbone.inference.DummyBackend.generate", generate):
            result = run(self.cfg, allow_unverified=True)
        self.assertEqual(result["new_predictions"], 6)
        self.assertEqual(len(calls), 6)
        self.assertNotIn("GOLD_NEVER_IN_PROMPT", json.dumps(calls))
        self.assertEqual(calls[4][2]["role"], "assistant")
        self.assertIn("model prediction", calls[4][2]["content"][0]["text"])
        self.assertTrue(calls[4][1]["content"][0]["path"].endswith("1.jpg"))
        self.assertTrue(calls[4][3]["content"][0]["path"].endswith("2.jpg"))
        self.assertEqual(run(self.cfg, allow_unverified=True)["new_predictions"], 0)
        summary = analyze(result["run_dir"], bootstrap_samples=10)
        self.assertEqual(summary["result_kind"], "dry_run")
        self.assertEqual(summary["tasks"], 1)
        changed = replace(self.cfg, experiment={**self.cfg.experiment, "seed": 43})
        with self.assertRaisesRegex(ValueError, "different dataset/config/runtime"):
            run(changed, allow_unverified=True)

    def test_missing_and_changed_images(self):
        prepare(self.cfg, allow_unverified=True)
        (self.image_dir / "1.jpg").write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "Image changed"):
            run(self.cfg, allow_unverified=True)
        (self.image_dir / "2.jpg").unlink()
        self.assertEqual(prepare(self.cfg, allow_unverified=True)["tasks"], 0)
        with self.assertRaises(ValueError):
            checked_image(self.image_dir, "../outside.jpg")

    def test_conflicting_reverse_episode_is_rejected(self):
        path = self.raw / "qviewbone_crossview_candidates.jsonl"
        rows = read_jsonl(path)
        rows[1]["gold"] = ["different"]
        write_jsonl(path, rows)
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            prepare(self.cfg, allow_unverified=True)

    def test_partial_checkpoint_recovery(self):
        path = Path(self.temp.name) / "partial.jsonl"
        path.write_bytes(b'{"a":1}\n{"a":')
        self.assertEqual(recover_predictions(path), [{"a": 1}])
        self.assertEqual(path.read_bytes(), b'{"a":1}\n')
        path.write_bytes(b'{bad}\n{"a":1}\n')
        with self.assertRaises(ValueError):
            recover_predictions(path)

    def test_splits_and_config_not_dependent_on_cwd(self):
        before = patient_split("p1", 42)
        self.assertEqual(before, patient_split("p1", 42))
        with patch.dict(os.environ, {"QVB_DATA_DIR": str(self.raw)}):
            cfg = load_settings(ROOT / "configs/experiment.toml")
            self.assertEqual(cfg.data_dir, self.raw.resolve())
            cfg = load_settings(ROOT / "configs/experiment.toml", {"paths": {"data_dir": str(self.image_dir)}})
            self.assertEqual(cfg.data_dir, self.image_dir.resolve())

    def test_scoring_does_not_match_substrings_or_negations(self):
        self.assertTrue(exact_set_match([" Phải. "], ["phải"]))
        self.assertFalse(exact_set_match(["không phải"], ["phải"]))
        self.assertFalse(exact_set_match(["xương chày và xương đùi"], ["xương chày"]))
        self.assertTrue(exact_set_match(["xương đùi", "xương chày"], ["xương chày", "xương đùi"]))
        self.assertIsNone(parse_prediction("the answer is phải"))
        self.assertEqual(parse_prediction('```json\n{"answer": ["phải"]}\n```'), ["phải"])

    def test_gpu_and_dtype_are_selected_at_runtime(self):
        cuda = SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: False,
                               get_device_name=lambda i: "T4", get_device_properties=lambda i: SimpleNamespace(total_memory=16 * 2**30))
        with patch.dict("sys.modules", {"torch": SimpleNamespace(cuda=cuda)}):
            self.assertEqual(hardware_settings(self.cfg.model)["dtype"], "float16")
            cuda.is_bf16_supported = lambda: True
            self.assertEqual(hardware_settings(self.cfg.model)["dtype"], "bfloat16")
            cuda.is_available = lambda: False
            self.assertEqual(hardware_settings(self.cfg.model)["dtype"], "float32")
            with self.assertRaisesRegex(ValueError, "requires CUDA"):
                hardware_settings({**self.cfg.model, "quantization": "4bit"})

    def test_all_utility_outcomes_and_partial_run_rejection(self):
        folder = Path(self.temp.name) / "utility"
        folder.mkdir()
        conditions = ["A", "B", "A_to_B", "B_to_A"]
        write_json(folder / "run.json", {"fingerprint": "test", "conditions": conditions, "task_count": 2, "result_kind": "dry_run"})
        rows = []
        for task, answers in enumerate(([False, True, True, False], [True, False, True, False])):
            for condition, correct in zip(conditions, answers):
                rows.append({"task_id": str(task), "pair_id": str(task), "patient_id": str(task), "category": "C1.1",
                             "gold": ["correct"], "raw_prediction": json.dumps({"answer": ["correct" if correct else "wrong"]}),
                             "condition": condition, "run_fingerprint": "test", "views": 1 if condition in {"A", "B"} else 2})
        write_jsonl(folder / "predictions.jsonl", rows)
        result = analyze(folder, bootstrap_samples=20)
        outcomes = {r["outcome"] for r in read_csv(folder / "view_utility.csv")}
        self.assertEqual(outcomes, {"rescue", "harm", "preserved", "unresolved"})
        forward = next(r for r in result["utility"] if r["direction"] == "A_to_B" and r["category"] == "ALL")
        self.assertEqual(forward["delta_accuracy"], .5)
        self.assertEqual(forward["rescue_given_initial_wrong"], 1)
        write_jsonl(folder / "predictions.jsonl", rows[:4])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            analyze(folder)


if __name__ == "__main__":
    unittest.main()
