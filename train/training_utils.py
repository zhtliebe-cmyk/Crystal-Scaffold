from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input that provides "
    "further context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input_text}\n\n"
    "### Response:\n{response}"
)

DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

SCAFFOLD_CORE_BEGIN = "<CRYSTAL_SCAFFOLD_CORE_V4>"
SCAFFOLD_FULL_BEGIN = "<CRYSTAL_SCAFFOLD_FULL_V4>"
PREVIOUS_STAGE = {"parse": None, "scaffold": "parse", "cif": "scaffold"}
CKPT_DIRNAME = "ckpt"

DEFAULT_PARSE_CAPS = {
    "dual_symmetry_summary": 2500,
    "formula_z_closure": 3000,
    "cell_archetype": 2500,
    "lattice_bin_correction": 2500,
    "coord_profile_parse": 2500,
    "scaffold_core_parse": 2500,
    "failure_bank_diagnose": 3000,
}


@dataclass
class TrainConfig:
    stage_name: str
    model_name: str = "/workspace/Llama-3.1-8B"
    train_file: str = ""
    val_file: Optional[str] = None
    pipeline_root: str = "/workspace/crystal/test-time"
    output_dir: str = ""
    adapter_dir: str = ""
    resume_mode: str = "base"
    resume_adapter_path: Optional[str] = None
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    dtype: Optional[str] = None
    lora_rank: int = 16
    lora_alpha: Optional[int] = None
    lora_dropout: float = 0.05
    num_train_epochs: float = 3.0
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.03
    warmup_steps: int = 0
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 0
    save_total_limit: int = 3
    early_stopping_patience: int = 0
    seed: int = 3407
    report_to: str = "none"
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_8bit"
    gradient_checkpointing: str = "unsloth"
    response_only_loss: bool = True
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    length_audit_samples: int = 128
    truncation_warn_threshold: float = 0.05
    add_eos_token: bool = True
    max_steps: int = 0
    dataloader_num_workers: int = 2
    group_by_length: bool = True
    parse_task_caps: Optional[str] = None

    def finalize(self) -> "TrainConfig":
        if self.lora_alpha is None:
            self.lora_alpha = int(self.lora_rank) * 2
        if not self.output_dir:
            self.output_dir = os.path.join(self.pipeline_root, CKPT_DIRNAME, self.stage_name)
        if not self.adapter_dir:
            self.adapter_dir = os.path.join(self.output_dir, "adapter")
        self.resume_mode = str(self.resume_mode or "base").lower()
        if self.resume_mode not in {"base", "previous", "path"}:
            raise ValueError(f"Unsupported resume_mode={self.resume_mode!r}")
        if self.resume_adapter_path:
            self.resume_mode = "path"
        if self.resume_mode == "previous":
            prev = PREVIOUS_STAGE.get(self.stage_name)
            if prev is None:
                self.resume_mode = "base"
                self.resume_adapter_path = None
            else:
                self.resume_adapter_path = os.path.join(self.pipeline_root, CKPT_DIRNAME, prev, "adapter")
        if self.resume_mode == "base":
            self.resume_adapter_path = None
        return self


def add_common_train_args(parser: argparse.ArgumentParser, *, default_train_file: str) -> None:
    parser.add_argument("--model-name", default="/workspace/Llama-3.1-8B")
    parser.add_argument("--train-file", default=default_train_file)
    parser.add_argument("--val-file", default=None)
    parser.add_argument("--pipeline-root", default="/workspace/crystal/test-time")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--resume-mode", default=None, choices=["base", "previous", "path"])
    parser.add_argument("--resume-adapter-path", default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--no-load-in-4bit", action="store_false", dest="load_in_4bit")
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--num-train-epochs", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--per-device-train-batch-size", type=int, default=None)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=0)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--optim", default="adamw_8bit")
    parser.add_argument("--gradient-checkpointing", default="unsloth")
    parser.add_argument("--full-sequence-loss", action="store_true", help="Disable response-only loss masking")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--length-audit-samples", type=int, default=128)
    parser.add_argument("--truncation-warn-threshold", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--no-group-by-length", action="store_false", dest="group_by_length")
    parser.add_argument("--group-by-length", action="store_true", dest="group_by_length")
    parser.set_defaults(group_by_length=True)
    parser.add_argument(
        "--parse-task-caps",
        default=None,
        help="JSON dict task->cap used only for parse-stage downsampling.",
    )


def config_from_args(args: argparse.Namespace, *, stage_name: str, stage_defaults: Dict[str, Any]) -> TrainConfig:
    output_dir = args.output_dir or os.path.join(args.pipeline_root, CKPT_DIRNAME, stage_name)
    cfg = TrainConfig(
        stage_name=stage_name,
        model_name=args.model_name,
        train_file=args.train_file,
        val_file=args.val_file,
        pipeline_root=args.pipeline_root,
        output_dir=output_dir,
        adapter_dir=args.adapter_dir or os.path.join(output_dir, "adapter"),
        resume_mode=args.resume_mode or stage_defaults.get("resume_mode", "base"),
        resume_adapter_path=args.resume_adapter_path,
        max_seq_length=args.max_seq_length or stage_defaults["max_seq_length"],
        load_in_4bit=args.load_in_4bit,
        dtype=args.dtype,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        num_train_epochs=args.num_train_epochs if args.num_train_epochs is not None else stage_defaults["num_train_epochs"],
        learning_rate=args.learning_rate if args.learning_rate is not None else stage_defaults["learning_rate"],
        per_device_train_batch_size=args.per_device_train_batch_size if args.per_device_train_batch_size is not None else stage_defaults["per_device_train_batch_size"],
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps if args.gradient_accumulation_steps is not None else stage_defaults["gradient_accumulation_steps"],
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        early_stopping_patience=args.early_stopping_patience,
        seed=args.seed,
        report_to=args.report_to,
        lr_scheduler_type=args.lr_scheduler_type,
        optim=args.optim,
        gradient_checkpointing=args.gradient_checkpointing,
        response_only_loss=not bool(args.full_sequence_loss),
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        length_audit_samples=args.length_audit_samples,
        truncation_warn_threshold=args.truncation_warn_threshold,
        max_steps=args.max_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        group_by_length=args.group_by_length,
        parse_task_caps=args.parse_task_caps,
    )
    return cfg.finalize()


def lazy_import_training_stack() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    errors: List[str] = []

    try:
        from unsloth import FastLanguageModel, is_bfloat16_supported
        out["FastLanguageModel"] = FastLanguageModel
        out["is_bfloat16_supported"] = is_bfloat16_supported
        out["backend"] = "unsloth"
    except Exception as e:
        errors.append(f"unsloth unavailable: {e}")
        out["backend"] = "transformers"

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig, EarlyStoppingCallback
        from transformers.trainer_utils import get_last_checkpoint
        out["AutoModelForCausalLM"] = AutoModelForCausalLM
        out["AutoTokenizer"] = AutoTokenizer
        out["Trainer"] = Trainer
        out["TrainingArguments"] = TrainingArguments
        out["BitsAndBytesConfig"] = BitsAndBytesConfig
        out["EarlyStoppingCallback"] = EarlyStoppingCallback
        out["get_last_checkpoint"] = get_last_checkpoint
    except Exception as e:
        errors.append(f"transformers unavailable: {e}")

    try:
        from peft import LoraConfig, get_peft_model, PeftModel
        out["LoraConfig"] = LoraConfig
        out["get_peft_model"] = get_peft_model
        out["PeftModel"] = PeftModel
    except Exception as e:
        errors.append(f"peft unavailable: {e}")

    if "Trainer" not in out:
        raise ImportError(
            "Required training stack is not available. Install transformers, peft, and optionally unsloth. "
            + " | ".join(errors)
        )
    return out


def ensure_tokenizer_padding(tokenizer: Any) -> None:
    if getattr(tokenizer, "pad_token", None) is None:
        if getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
    tokenizer.padding_side = "right"


def _bf16_supported(stack: Dict[str, Any]) -> bool:
    if "is_bfloat16_supported" in stack:
        try:
            return bool(stack["is_bfloat16_supported"]())
        except Exception:
            pass
    return bool(torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)())


def load_model_and_tokenizer(cfg: TrainConfig) -> Tuple[Any, Any]:
    stack = lazy_import_training_stack()
    backend = stack.get("backend", "transformers")

    if backend == "unsloth":
        FastLanguageModel = stack["FastLanguageModel"]
        model_name = cfg.resume_adapter_path if cfg.resume_adapter_path else cfg.model_name
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=cfg.max_seq_length,
            dtype=cfg.dtype,
            load_in_4bit=cfg.load_in_4bit,
        )
        ensure_tokenizer_padding(tokenizer)
        if not cfg.resume_adapter_path:
            model = FastLanguageModel.get_peft_model(
                model,
                r=cfg.lora_rank,
                target_modules=DEFAULT_TARGET_MODULES,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                bias="none",
                use_gradient_checkpointing=cfg.gradient_checkpointing,
                random_state=cfg.seed,
                use_rslora=False,
                loftq_config=None,
            )
        return model, tokenizer

    AutoModelForCausalLM = stack["AutoModelForCausalLM"]
    AutoTokenizer = stack["AutoTokenizer"]
    BitsAndBytesConfig = stack["BitsAndBytesConfig"]
    LoraConfig = stack["LoraConfig"]
    get_peft_model = stack["get_peft_model"]
    PeftModel = stack["PeftModel"]

    quantization_config = None
    if cfg.load_in_4bit:
        try:
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        except Exception:
            quantization_config = None

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=False)
    ensure_tokenizer_padding(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=quantization_config,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if cfg.resume_adapter_path:
        model = PeftModel.from_pretrained(model, cfg.resume_adapter_path, is_trainable=True)
    else:
        peft_cfg = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            target_modules=DEFAULT_TARGET_MODULES,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_cfg)
    return model, tokenizer


def build_prompt_text(instruction: str, input_text: str, response: str) -> str:
    return PROMPT_TEMPLATE.format(instruction=instruction, input_text=input_text, response=response)


def build_prompt_prefix(instruction: str, input_text: str) -> str:
    return build_prompt_text(instruction=instruction, input_text=input_text, response="")


def load_table(path: str, *, max_rows: Optional[int] = None) -> pd.DataFrame:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path_obj.suffix.lower() == ".jsonl":
        rows = []
        with open(path_obj, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        df = pd.DataFrame(rows)
    else:
        df = pd.read_csv(path_obj)
    required = {"instruction", "input", "output"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    df = df.copy()
    for col in ["instruction", "input", "output"]:
        df[col] = df[col].fillna("").astype(str)
    if max_rows is not None:
        df = df.head(max_rows).copy()
    return df


def validate_stage_table(df: pd.DataFrame, cfg: TrainConfig) -> None:
    if cfg.stage_name == "parse":
        if "task" in df.columns:
            allowed = {
                "dual_symmetry_summary",
                "formula_z_closure",
                "cell_archetype",
                "lattice_bin_correction",
                "coord_profile_parse",
                "scaffold_core_parse",
                "failure_bank_diagnose",
            }
            bad = sorted(set(df["task"].astype(str)) - allowed)
            if bad:
                raise ValueError(f"Unexpected parse tasks: {bad}")
    elif cfg.stage_name == "scaffold":
        sample = "\n".join(df["output"].head(5).tolist())
        if SCAFFOLD_CORE_BEGIN not in sample:
            raise ValueError("Scaffold stage expects outputs tagged with <CRYSTAL_SCAFFOLD_CORE_V4>.")
    elif cfg.stage_name == "cif":
        sample_in = "\n".join(df["input"].head(5).tolist())
        if SCAFFOLD_FULL_BEGIN not in sample_in:
            raise ValueError("CIF stage expects compiled scaffold input tagged with <CRYSTAL_SCAFFOLD_FULL_V4>.")
        sample_out = "\n".join(df["output"].head(5).tolist())
        if "_atom_site_type_symbol" not in sample_out:
            raise ValueError("CIF stage outputs do not look like CIF text.")


def summarize_stage_dataset(df: pd.DataFrame, cfg: TrainConfig) -> None:
    print(f"[INFO] stage={cfg.stage_name} rows={len(df)}")
    if cfg.stage_name == "parse" and "task" in df.columns:
        vc = df["task"].value_counts().to_dict()
        print("[INFO] parse task distribution=" + json.dumps(vc, ensure_ascii=False))
    if cfg.stage_name in {"scaffold", "cif"} and "condition_sg_number" in df.columns:
        top = df["condition_sg_number"].value_counts().head(10).to_dict()
        print("[INFO] top condition_sg_number=" + json.dumps(top, ensure_ascii=False))


def _safe_parse_caps(text: Optional[str]) -> Dict[str, int]:
    caps = dict(DEFAULT_PARSE_CAPS)
    if not text:
        return caps
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            for k, v in obj.items():
                caps[str(k)] = int(v)
    except Exception as e:
        print(f"[WARN] could not parse --parse-task-caps JSON: {e}")
    return caps


def optimize_parse_dataset(df: pd.DataFrame, cfg: TrainConfig) -> pd.DataFrame:
    if cfg.stage_name != "parse" or "task" not in df.columns:
        return df

    caps = _safe_parse_caps(cfg.parse_task_caps)
    df = df.copy()
    out_parts = []
    for task, cap in caps.items():
        part = df[df["task"].astype(str) == task].copy()
        if len(part) == 0 or cap <= 0:
            continue
        part = part.drop_duplicates(subset=["instruction", "input", "output"], keep="first")
        if "material_id" in part.columns:
            part = part.sort_values(["material_id", "task"]).groupby("material_id", as_index=False).head(2)
        if len(part) > cap:
            part = part.head(cap).copy()
        out_parts.append(part)

    if not out_parts:
        return df

    out = pd.concat(out_parts, axis=0, ignore_index=True)
    out = out.drop_duplicates(subset=["instruction", "input", "output"], keep="first").reset_index(drop=True)
    print(f"[INFO] parse warm-up reduced from {len(df)} -> {len(out)} rows")
    return out


class PromptResponseDataset(Dataset):
    def __init__(self, records: Sequence[Dict[str, str]], tokenizer: Any, *, max_seq_length: int, response_only_loss: bool = True, add_eos_token: bool = True) -> None:
        self.records = list(records)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.response_only_loss = response_only_loss
        self.add_eos_token = add_eos_token
        self.eos_token = getattr(tokenizer, "eos_token", None) or ""

    def __len__(self) -> int:
        return len(self.records)

    def _encode(self, text: str) -> Dict[str, List[int]]:
        return self.tokenizer(text, truncation=True, max_length=self.max_seq_length, add_special_tokens=False)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.records[idx]
        response_text = ex["output"] + (self.eos_token if self.add_eos_token else "")
        full_text = build_prompt_text(ex["instruction"], ex["input"], response_text)
        enc = self._encode(full_text)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        labels = list(input_ids)
        if self.response_only_loss:
            prefix_text = build_prompt_prefix(ex["instruction"], ex["input"])
            prefix_ids = self._encode(prefix_text)["input_ids"]
            prefix_len = min(len(prefix_ids), len(labels))
            labels[:prefix_len] = [-100] * prefix_len
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class Seq2SeqStyleCollator:
    def __init__(self, tokenizer: Any):
        self.pad_id = int(tokenizer.pad_token_id)

    def __call__(self, features: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(int(f["input_ids"].shape[0]) for f in features)
        batch_input_ids, batch_attention, batch_labels = [], [], []
        for feat in features:
            seq_len = int(feat["input_ids"].shape[0])
            pad_len = max_len - seq_len
            batch_input_ids.append(torch.nn.functional.pad(feat["input_ids"], (0, pad_len), value=self.pad_id))
            batch_attention.append(torch.nn.functional.pad(feat["attention_mask"], (0, pad_len), value=0))
            batch_labels.append(torch.nn.functional.pad(feat["labels"], (0, pad_len), value=-100))
        return {
            "input_ids": torch.stack(batch_input_ids, dim=0),
            "attention_mask": torch.stack(batch_attention, dim=0),
            "labels": torch.stack(batch_labels, dim=0),
        }


def audit_sequence_lengths(df: pd.DataFrame, tokenizer: Any, cfg: TrainConfig, *, title: str) -> None:
    n = min(len(df), max(int(cfg.length_audit_samples), 1))
    if n == 0:
        return
    sample = df.head(n)
    lens: List[int] = []
    for _, row in sample.iterrows():
        text = build_prompt_text(str(row["instruction"]), str(row["input"]), str(row["output"]))
        toks = tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"]
        lens.append(len(toks))
    lens = sorted(lens)

    def pct(q: float) -> int:
        idx = min(len(lens) - 1, max(0, int(round((len(lens) - 1) * q))))
        return int(lens[idx])

    trunc_frac = sum(1 for x in lens if x > cfg.max_seq_length) / max(len(lens), 1)
    print(
        f"[INFO] {title} token lengths: p50={pct(0.50)} p90={pct(0.90)} p95={pct(0.95)} "
        f"p99={pct(0.99)} max={max(lens)} limit={cfg.max_seq_length} trunc_frac={trunc_frac:.3%}"
    )
    if trunc_frac > cfg.truncation_warn_threshold:
        print("[WARN] sequence truncation exceeds threshold. Increase --max-seq-length or shorten the stage inputs.")


def build_trainer(cfg: TrainConfig, model: Any, tokenizer: Any, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None) -> Any:
    stack = lazy_import_training_stack()
    Trainer = stack["Trainer"]
    TrainingArguments = stack["TrainingArguments"]
    EarlyStoppingCallback = stack.get("EarlyStoppingCallback")

    train_records = train_df[["instruction", "input", "output"]].to_dict(orient="records")
    eval_records = None if val_df is None else val_df[["instruction", "input", "output"]].to_dict(orient="records")

    train_dataset = PromptResponseDataset(
        train_records,
        tokenizer,
        max_seq_length=cfg.max_seq_length,
        response_only_loss=cfg.response_only_loss,
        add_eos_token=cfg.add_eos_token,
    )
    eval_dataset = None
    if eval_records is not None and len(eval_records) > 0:
        eval_dataset = PromptResponseDataset(
            eval_records,
            tokenizer,
            max_seq_length=cfg.max_seq_length,
            response_only_loss=cfg.response_only_loss,
            add_eos_token=cfg.add_eos_token,
        )

    bf16 = _bf16_supported(stack)
    use_eval = eval_dataset is not None and cfg.eval_steps and cfg.eval_steps > 0

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_steps=cfg.warmup_steps,
        warmup_ratio=0.0 if cfg.warmup_steps > 0 else cfg.warmup_ratio,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        fp16=not bf16,
        bf16=bf16,
        logging_steps=cfg.logging_steps,
        optim=cfg.optim,
        report_to=cfg.report_to,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type=cfg.lr_scheduler_type,
        seed=cfg.seed,
        save_steps=cfg.save_steps,
        save_strategy="steps",
        save_total_limit=cfg.save_total_limit,
        eval_strategy="steps" if use_eval else "no",
        eval_steps=cfg.eval_steps if use_eval else None,
        remove_unused_columns=False,
        dataloader_pin_memory=torch.cuda.is_available(),
        load_best_model_at_end=bool(use_eval),
        metric_for_best_model="eval_loss" if use_eval else None,
        greater_is_better=False if use_eval else None,
        max_steps=cfg.max_steps if cfg.max_steps and cfg.max_steps > 0 else -1,
        dataloader_num_workers=cfg.dataloader_num_workers,
        group_by_length=cfg.group_by_length,
    )

    callbacks = []
    if use_eval and cfg.early_stopping_patience > 0 and EarlyStoppingCallback is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience))

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=Seq2SeqStyleCollator(tokenizer),
        callbacks=callbacks,
    )
    return trainer


def save_training_artifacts(cfg: TrainConfig, model: Any, tokenizer: Any, *, train_rows: int, val_rows: int) -> None:
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.adapter_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(cfg.adapter_dir)
    tokenizer.save_pretrained(cfg.adapter_dir)
    manifest = asdict(cfg)
    manifest.update({
        "train_rows": int(train_rows),
        "val_rows": int(val_rows),
        "global_batch_size": int(cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps),
    })
    with open(Path(cfg.output_dir) / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def _maybe_get_last_checkpoint(cfg: TrainConfig) -> Optional[str]:
    stack = lazy_import_training_stack()
    get_last_checkpoint = stack.get("get_last_checkpoint")
    if get_last_checkpoint is None:
        return None
    out_dir = Path(cfg.output_dir)
    if not out_dir.exists():
        return None
    try:
        return get_last_checkpoint(str(out_dir))
    except Exception:
        return None


def run_training(cfg: TrainConfig) -> None:
    cfg.finalize()

    train_df = load_table(cfg.train_file, max_rows=cfg.max_train_samples)
    if cfg.stage_name == "parse":
        train_df = optimize_parse_dataset(train_df, cfg)
        if cfg.max_train_samples is not None and len(train_df) > cfg.max_train_samples:
            train_df = train_df.head(cfg.max_train_samples).copy()

    val_df = load_table(cfg.val_file, max_rows=cfg.max_eval_samples) if cfg.val_file else None

    validate_stage_table(train_df, cfg)
    summarize_stage_dataset(train_df, cfg)
    if val_df is not None:
        validate_stage_table(val_df, cfg)

    if cfg.resume_adapter_path and not Path(cfg.resume_adapter_path).exists():
        raise FileNotFoundError(
            f"resume adapter not found: {cfg.resume_adapter_path}. Run the previous stage first or pass --resume-mode base."
        )

    model, tokenizer = load_model_and_tokenizer(cfg)
    if hasattr(model, "print_trainable_parameters"):
        try:
            model.print_trainable_parameters()
        except Exception:
            pass

    audit_sequence_lengths(train_df, tokenizer, cfg, title=f"{cfg.stage_name}/train")
    if val_df is not None and len(val_df) > 0:
        audit_sequence_lengths(val_df, tokenizer, cfg, title=f"{cfg.stage_name}/val")

    trainer = build_trainer(cfg, model, tokenizer, train_df, val_df)
    resume_ckpt = _maybe_get_last_checkpoint(cfg)
    if resume_ckpt:
        print(f"[INFO] resuming stage={cfg.stage_name} from {resume_ckpt}")
        trainer.train(resume_from_checkpoint=resume_ckpt)
    else:
        trainer.train()

    save_training_artifacts(cfg, model, tokenizer, train_rows=len(train_df), val_rows=0 if val_df is None else len(val_df))
    print(f"[OK] finished training stage={cfg.stage_name}")
    print(f"[OK] adapter saved to {cfg.adapter_dir}")
