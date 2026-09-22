import tempfile
import unittest
from pathlib import Path

import torch

from train import (
    MIN_CALIBRATION_EXAMPLES,
    fit_temperatures,
    load_records,
    target_indices,
)


class TargetIndicesTest(unittest.TestCase):
    def test_choice_dict_preserves_label_order(self) -> None:
        record = {
            "question": {
                "type": "choice",
                "criteria": {"billing": "payments", "technical": "errors"},
            },
            "target": {"choice": "technical", "action": "escalate"},
        }
        self.assertEqual(target_indices(record), (1, 1))

    def test_choice_list_maps_answer_to_zero(self) -> None:
        record = {
            "question": {
                "type": "choice",
                "criteria": ["billing", "technical", "sales"],
            },
            "target": {"choice": "billing", "action": "answer"},
        }
        self.assertEqual(target_indices(record), (0, 0))

    def test_score_and_boolean_targets(self) -> None:
        score = {
            "question": {"type": "score", "criteria": ["low", "high"]},
            "target": {"score": 1},
        }
        boolean = {
            "question": {"type": "noul"},
            "target": {"noul": True},
        }
        self.assertEqual(target_indices(score), (1, 0))
        self.assertEqual(target_indices(boolean), (1, 0))


class DatasetTest(unittest.TestCase):
    def test_requires_validation_split(self) -> None:
        content = (
            '{"id":"one","state":"hello","question":{"type":"choice",'
            '"instructions":"route","criteria":["a","b"]},'
            '"target":{"choice":"a"},"split":"train"}\n'
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text(content)
            with self.assertRaisesRegex(ValueError, "validation"):
                load_records(path)


class CalibrationTest(unittest.TestCase):
    def test_small_validation_set_keeps_neutral_temperature(self) -> None:
        predictions = [
            {
                "logits": torch.tensor([2.0, 0.0]),
                "label": 0,
                "qtype": 0,
                "option_count": 2,
            }
        ]
        temperatures, buckets = fit_temperatures(predictions)
        self.assertEqual(temperatures, [1.0, 1.0, 1.0])
        self.assertEqual(buckets, {})

    def test_sufficient_consistent_predictions_fit_temperature(self) -> None:
        predictions = [
            {
                "logits": torch.tensor([2.0, 0.0]),
                "label": 0,
                "qtype": 0,
                "option_count": 2,
            }
            for _ in range(MIN_CALIBRATION_EXAMPLES)
        ]
        temperatures, buckets = fit_temperatures(predictions)
        self.assertAlmostEqual(temperatures[0], 0.25)
        self.assertAlmostEqual(buckets["choice:2"], 0.25)


if __name__ == "__main__":
    unittest.main()
