from __future__ import annotations

import argparse
from training_utils import add_common_train_args, config_from_args, run_training

STAGE_DEFAULTS = {
    "resume_mode": "previous",
    "max_seq_length": 3072,
    "num_train_epochs": 10.0,
    "learning_rate": 2.0e-5,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage III V4 scaffold-full conditioned CIF generator.")
    add_common_train_args(
        parser,
        default_train_file="/workspace/crystal/test-time/data/cif_with_scaffold_train_v4.csv",
    )
    parser.set_defaults(
        logging_steps=10,
        save_steps=2500,
        eval_steps=2500,
        early_stopping_patience=3,
        warmup_ratio=0.05,
        dataloader_num_workers=4,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = config_from_args(args, stage_name="cif", stage_defaults=STAGE_DEFAULTS)
    run_training(cfg)
