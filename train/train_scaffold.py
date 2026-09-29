from __future__ import annotations

import argparse
from training_utils import add_common_train_args, config_from_args, run_training

STAGE_DEFAULTS = {
    "resume_mode": "previous",
    "max_seq_length": 768,
    "num_train_epochs": 5.0,
    "learning_rate": 5.0e-5,
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage II V4 scaffold-core planner.")
    add_common_train_args(
        parser,
        default_train_file="/workspace/crystal/test-time/data/scaffold_train_v4.csv",
    )
    parser.set_defaults(
        logging_steps=20,
        save_steps=1000,
        eval_steps=500,
        early_stopping_patience=4,
        dataloader_num_workers=4,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = config_from_args(args, stage_name="scaffold", stage_defaults=STAGE_DEFAULTS)
    run_training(cfg)
