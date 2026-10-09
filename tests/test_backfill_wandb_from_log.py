import tempfile
import unittest
from pathlib import Path

from scripts.backfill_wandb_from_log import load_metrics, parse_train_line


class BackfillWandbFromLogTest(unittest.TestCase):
    def test_parse_train_line(self) -> None:
        line = (
            "\x1b[32m07/22 [12:00:00]\x1b[0m INFO | >> "
            "[train] epoch=2 step=10/1000 loss=0.1234 "
            "loss_action=0.0234 loss_video=0.1000 lr=1.00e-04 "
            "speed=1.25 step/s, 10.00 samples/s eta=00:13:12"
        )

        self.assertEqual(
            parse_train_line(line),
            (
                10,
                {
                    "global_step": 10,
                    "train/epoch": 2,
                    "train/loss": 0.1234,
                    "train/max_steps": 1000,
                    "train/loss_action": 0.0234,
                    "train/loss_video": 0.1,
                    "train/lr": 0.0001,
                    "performance/steps_per_sec": 1.25,
                    "performance/samples_per_sec": 10.0,
                },
            ),
        )

    def test_load_metrics_sorts_steps_and_latest_duplicate_wins(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "train.log"
            log_path.write_text(
                "[train] epoch=0 step=20/100 loss=0.2000 lr=2e-4\n"
                "unrelated line\n"
                "[train] epoch=0 step=10/100 loss=0.1000 lr=1e-4\n"
                "[train] epoch=1 step=20/100 loss=0.1500 lr=5e-5\n",
                encoding="utf-8",
            )

            metrics, matched_lines = load_metrics(log_path)

        self.assertEqual(matched_lines, 3)
        self.assertEqual([step for step, _ in metrics], [10, 20])
        self.assertEqual(metrics[-1][1]["train/loss"], 0.15)
        self.assertEqual(metrics[-1][1]["train/epoch"], 1)


if __name__ == "__main__":
    unittest.main()
