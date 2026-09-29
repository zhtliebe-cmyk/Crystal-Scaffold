from __future__ import annotations

import argparse
from training_utils import add_common_train_args, config_from_args, run_training

STAGE_DEFAULTS = {
    "resume_mode": "base",
    "max_seq_length": 512,
    "num_train_epochs": 1.0,
    "learning_rate": 1.0e-4,
    "per_device_train_batch_size": 16,
    "gradient_accumulation_steps": 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage I parser-oriented warm-up for V4 Core->Full generation.")
    add_common_train_args(
        parser,
        default_train_file="/workspace/crystal/test-time/data/parse_curriculum_train_v4.csv",
    )
    parser.set_defaults(
        logging_steps=20,
        save_steps=400,
        eval_steps=200,
        early_stopping_patience=2,
        max_steps=800,
        max_train_samples=12000,
        max_eval_samples=2000,
        dataloader_num_workers=4,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = config_from_args(args, stage_name="parse", stage_defaults=STAGE_DEFAULTS)
    run_training(cfg)
