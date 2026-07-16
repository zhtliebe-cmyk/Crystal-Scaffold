from __future__ import annotations

import json
import os
import random
import re
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Sequence

from v4_preprocess import schema
from v4_preprocess.cif_utils import (
    angle_bin,
    angle_pattern,
    build_prompt_input,
    canonicalize_cif_text,
    c_over_a_bin_family,
    compile_full_scaffold,
    crystal_system_from_sg,
    length_pattern,
    metric_family,
    parse_cif_text,
    parse_formula_counts,
    parse_scaffold_text,
    position_family,
    ratio_bin,
    reduced_counts,
    reduced_ratio_string,
    stack_family,
    vol_per_atom_bin,
)

PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input that provides "
    "further context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input_text}\n\n"
    "### Response:\n{response}"
)
DEFAULT_MODEL_NAME = os.environ.get("V4_MODEL_NAME", "meta-llama/Llama-3.1-8B")


@dataclass
class GenerationConfig:
    model_name: str = DEFAULT_MODEL_NAME
    adapter_path: Optional[str] = None
    load_in_4bit: bool = True
    dtype: Optional[str] = None
    max_seq_length: int = 3072
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.92
    top_k: int = 40
    num_return_sequences: int = 1
    repetition_penalty: float = 1.02
    do_sample: bool = True
    device: Optional[str] = None

    def validate(self) -> "GenerationConfig":
        if not self.model_name:
            raise ValueError("model_name must not be empty")
        if self.dtype not in {None, "float16", "bfloat16", "float32"}:
            raise ValueError("dtype must be float16, bfloat16, float32, or omitted")
        if self.max_seq_length <= 0 or self.max_new_tokens <= 0:
            raise ValueError("max_seq_length and max_new_tokens must be positive")
        if self.num_return_sequences <= 0:
            raise ValueError("num_return_sequences must be positive")
        if not self.do_sample and self.num_return_sequences != 1:
            raise ValueError("greedy decoding requires num_return_sequences=1")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if self.do_sample and self.temperature <= 0:
            raise ValueError("temperature must be positive when sampling is enabled")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        return self


def build_prompt_text(instruction: str, input_text: str, response: str = "") -> str:
    return PROMPT_TEMPLATE.format(
        instruction=str(instruction).strip(),
        input_text=str(input_text).strip(),
        response=response,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        return


def write_text(path: str | Path, text: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def write_json(path: str | Path, value: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def dedupe_texts(texts: Sequence[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for text in texts:
        normalized = normalize_text(text)
        if normalized and normalized not in seen:
            output.append(str(text).strip())
            seen.add(normalized)
    return output


def validate_space_group(space_group: int) -> int:
    value = int(space_group)
    if not 1 <= value <= 230:
        raise ValueError(f"space_group must be between 1 and 230, got {value}")
    return value


def material_prompt_input(composition: str, space_group: int) -> str:
    return build_prompt_input(str(composition).strip(), validate_space_group(space_group))


def formula_reduced_ratio(formula: str) -> str:
    return reduced_ratio_string(parse_formula_counts(formula))


def extract_scaffold_core_from_response(text: str) -> str:
    return _extract_tagged_block(text, schema.SCAFFOLD_CORE_BEGIN, schema.SCAFFOLD_CORE_END)


def extract_cif_from_response(text: str) -> str:
    cleaned = str(text).strip().replace("</s>", "").replace("<|end_of_text|>", "").strip()
    if "data_" in cleaned:
        cleaned = cleaned[cleaned.index("data_") :]
    return cleaned.strip()


def _extract_tagged_block(text: str, begin_tag: str, end_tag: str) -> str:
    value = str(text).strip()
    if begin_tag not in value or end_tag not in value:
        return value
    start = value.index(begin_tag)
    end = value.index(end_tag, start) + len(end_tag)
    return value[start:end].strip()


def _resolve_dtype(dtype_name: Optional[str]) -> Any:
    if dtype_name is None:
        return None
    import torch

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def _ensure_tokenizer_padding(tokenizer: Any) -> bool:
    added_token = False
    if getattr(tokenizer, "pad_token", None) is None:
        if getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
            added_token = True
    tokenizer.padding_side = "right"
    return added_token


def _load_inference_stack() -> dict[str, Any]:
    stack: dict[str, Any] = {}
    try:
        from unsloth import FastLanguageModel

        stack["FastLanguageModel"] = FastLanguageModel
    except Exception as exc:
        warnings.warn(f"Unsloth is unavailable; using Transformers fallback: {exc}")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        stack.update(
            AutoModelForCausalLM=AutoModelForCausalLM,
            AutoTokenizer=AutoTokenizer,
            BitsAndBytesConfig=BitsAndBytesConfig,
        )
    except ImportError:
        if "FastLanguageModel" not in stack:
            raise ImportError(
                "Install transformers and peft, or install unsloth, before running inference."
            )

    try:
        from peft import PeftModel

        stack["PeftModel"] = PeftModel
    except ImportError:
        pass
    return stack


class ModelRunner:
    def __init__(self, config: GenerationConfig):
        self.config = config.validate()
        self.model: Any = None
        self.tokenizer: Any = None

    def load(self) -> None:
        stack = _load_inference_stack()
        if "FastLanguageModel" in stack:
            self._load_unsloth(stack["FastLanguageModel"])
            return
        self._load_transformers(stack)

    def _load_unsloth(self, fast_language_model: Any) -> None:
        load_target = self.config.adapter_path or self.config.model_name
        self.model, self.tokenizer = fast_language_model.from_pretrained(
            model_name=load_target,
            max_seq_length=self.config.max_seq_length,
            dtype=_resolve_dtype(self.config.dtype),
            load_in_4bit=self.config.load_in_4bit,
        )
        added_token = _ensure_tokenizer_padding(self.tokenizer)
        if added_token and hasattr(self.model, "resize_token_embeddings"):
            self.model.resize_token_embeddings(len(self.tokenizer))
        fast_language_model.for_inference(self.model)

    def _load_transformers(self, stack: dict[str, Any]) -> None:
        import torch

        use_4bit = self.config.load_in_4bit and torch.cuda.is_available()
        if self.config.load_in_4bit and not torch.cuda.is_available():
            warnings.warn("4-bit loading requires CUDA; falling back to non-quantized loading.")

        tokenizer_source = self.config.adapter_path or self.config.model_name
        try:
            self.tokenizer = stack["AutoTokenizer"].from_pretrained(
                tokenizer_source,
                use_fast=False,
            )
        except (OSError, ValueError):
            self.tokenizer = stack["AutoTokenizer"].from_pretrained(
                self.config.model_name,
                use_fast=False,
            )
        added_token = _ensure_tokenizer_padding(self.tokenizer)

        model_kwargs: dict[str, Any] = {}
        if use_4bit:
            model_kwargs["quantization_config"] = stack["BitsAndBytesConfig"](
                load_in_4bit=True,
                bnb_4bit_compute_dtype=_resolve_dtype(self.config.dtype) or torch.float16,
            )
            model_kwargs["device_map"] = self.config.device or "auto"
        else:
            resolved_dtype = _resolve_dtype(self.config.dtype)
            if resolved_dtype is not None:
                model_kwargs["torch_dtype"] = resolved_dtype
            if self.config.device:
                model_kwargs["device_map"] = self.config.device
            elif torch.cuda.is_available():
                model_kwargs["device_map"] = "auto"

        self.model = stack["AutoModelForCausalLM"].from_pretrained(
            self.config.model_name,
            **model_kwargs,
        )
        if self.config.adapter_path:
            if "PeftModel" not in stack:
                raise ImportError("peft is required when --adapter-path is provided")
            self.model = stack["PeftModel"].from_pretrained(
                self.model,
                self.config.adapter_path,
            )
        if added_token and hasattr(self.model, "resize_token_embeddings"):
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.eval()

    def generate(self, instruction: str, input_text: str, response_prefix: str = "") -> list[str]:
        if self.model is None or self.tokenizer is None:
            self.load()

        prompt = build_prompt_text(instruction, input_text, response_prefix)
        tokens = self.tokenizer(
            [prompt],
            return_tensors="pt",
            add_special_tokens=False,
            truncation=False,
        )
        prompt_length = int(tokens["input_ids"].shape[-1])
        max_new_tokens = self._available_generation_tokens(prompt_length)
        input_device = self._input_device()
        tokens = {name: tensor.to(input_device) for name, tensor in tokens.items()}

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "num_return_sequences": self.config.num_return_sequences,
            "repetition_penalty": self.config.repetition_penalty,
            "do_sample": self.config.do_sample,
            "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.config.do_sample:
            generation_kwargs.update(
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
            )

        import torch

        with torch.inference_mode():
            outputs = self.model.generate(**tokens, **generation_kwargs)
        generated_ids = outputs[:, prompt_length:]
        return [
            text.strip()
            for text in self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        ]

    def _available_generation_tokens(self, prompt_length: int, safety_margin: int = 16) -> int:
        available = self.config.max_seq_length - prompt_length - safety_margin
        if available <= 0:
            raise ValueError(
                f"Prompt length {prompt_length} leaves no generation space within "
                f"max_seq_length={self.config.max_seq_length}."
            )
        return min(self.config.max_new_tokens, available)

    def _input_device(self) -> Any:
        try:
            return self.model.get_input_embeddings().weight.device
        except (AttributeError, TypeError):
            return self.model.device


def scaffold_quality_checks(
    scaffold_text: str,
    *,
    formula: str,
    space_group: int,
    expect_full: bool = False,
) -> tuple[bool, list[str], dict[str, str]]:
    fields = dict(parse_scaffold_text(scaffold_text))
    if not fields:
        return False, ["empty_scaffold"], {}

    errors: list[str] = []
    normalized_formula = re.sub(r"\s+", "", str(formula))
    if re.sub(r"\s+", "", fields.get("formula", "")) != normalized_formula:
        errors.append("formula_mismatch")
    if fields.get("reduced_ratio", "") != formula_reduced_ratio(formula):
        errors.append("reduced_ratio_mismatch")
    if fields.get("condition_sg_number", "") != str(validate_space_group(space_group)):
        errors.append("condition_sg_mismatch")
    expected_system = crystal_system_from_sg(space_group)
    if fields.get("condition_crystal_system", "") != expected_system:
        errors.append("crystal_system_mismatch")
    if fields.get("target_cell_mode", "") != "primitive":
        errors.append("bad_target_cell_mode")

    required_core = (
        "formula",
        "reduced_ratio",
        "condition_sg_number",
        "condition_crystal_system",
        "target_cell_mode",
        "output_Z_raw",
        "metric_family",
        "length_pattern",
        "angle_pattern",
        "vol_per_atom_bin",
        "position_family",
        "stack_family",
    )
    errors.extend(f"missing_{key}" for key in required_core if not fields.get(key))
    try:
        if int(fields.get("output_Z_raw", "0")) <= 0:
            errors.append("bad_output_Z_raw")
    except ValueError:
        errors.append("bad_output_Z_raw")

    if expect_full:
        required_full = (
            "site_schema",
            "n_listed_sites",
            "formula_sum_expected",
            "geometry_signature_v4",
            "prototype_bucket_v4",
        )
        errors.extend(f"missing_{key}" for key in required_full if not fields.get(key))
        try:
            if int(fields.get("n_listed_sites", "0")) <= 0:
                errors.append("bad_n_listed_sites")
        except ValueError:
            errors.append("bad_n_listed_sites")

    errors = list(dict.fromkeys(errors))
    return not errors, errors, fields


def compile_generated_scaffold(
    response_text: str,
    *,
    formula: str,
    space_group: int,
) -> dict[str, Any]:
    core_text = extract_scaffold_core_from_response(response_text)
    valid, errors, fields = scaffold_quality_checks(
        core_text,
        formula=formula,
        space_group=space_group,
    )
    full_text = ""
    if valid:
        try:
            full_text = compile_full_scaffold(core_text, formula=formula)
            valid, full_errors, fields = scaffold_quality_checks(
                full_text,
                formula=formula,
                space_group=space_group,
                expect_full=True,
            )
            errors.extend(full_errors)
        except (KeyError, TypeError, ValueError) as exc:
            valid = False
            errors.append(f"compile_full_failed:{exc}")
    return {
        "scaffold_core": core_text,
        "scaffold_full": full_text,
        "valid_scaffold": valid,
        "scaffold_errors": list(dict.fromkeys(errors)),
        "fields": fields,
    }


def scaffold_candidate_score(candidate: dict[str, Any]) -> tuple[int, int, int, int]:
    fields = candidate.get("fields", {})
    return (
        int(bool(candidate.get("valid_scaffold"))),
        -len(candidate.get("scaffold_errors", [])),
        int(fields.get("position_family") == "GENERAL_DOMINANT"),
        len(candidate.get("scaffold_full", "")),
    )


def generate_scaffold_candidates(
    runner: ModelRunner,
    *,
    formula: str,
    space_group: int,
    num_samples: int,
) -> list[dict[str, Any]]:
    raw_outputs = dedupe_texts(
        runner.generate(
            schema.SCAFFOLD_CORE_INSTRUCTION,
            material_prompt_input(formula, space_group),
        )
    )
    candidates: list[dict[str, Any]] = []
    seen_full: set[str] = set()
    for rank, response in enumerate(raw_outputs[:num_samples]):
        candidate = compile_generated_scaffold(
            response,
            formula=formula,
            space_group=space_group,
        )
        full_key = normalize_text(candidate["scaffold_full"])
        if full_key and full_key in seen_full:
            continue
        if full_key:
            seen_full.add(full_key)
        candidate["rank"] = rank
        candidates.append(candidate)
    return sorted(candidates, key=scaffold_candidate_score, reverse=True)


def select_best_scaffold(candidates: Sequence[dict[str, Any]]) -> Optional[dict[str, Any]]:
    return candidates[0] if candidates else None


def _looks_degenerate_cif(cif_text: str) -> bool:
    value = str(cif_text)
    return any(
        marker in value
        for marker in (
            "data_unknown",
            "_chemical_formula_structural unknown",
            "_chemical_formula_sum ''",
            '_chemical_formula_sum ""',
        )
    ) or all(
        marker in value
        for marker in (
            "_cell_length_a 0.000000",
            "_cell_length_b 0.000000",
            "_cell_length_c 0.000000",
        )
    )


def canonicalize_generated_cif(text: str) -> tuple[str, bool, Optional[str]]:
    cif_text = extract_cif_from_response(text)
    try:
        canonical = canonicalize_cif_text(cif_text)
    except (KeyError, TypeError, ValueError) as exc:
        return cif_text, False, str(exc)
    if _looks_degenerate_cif(canonical):
        return canonical, False, "degenerate_canonicalized_cif"
    return canonical, True, None


def _site_schema_counts(record: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for site in record.atom_sites:
        counts[site.element] = counts.get(site.element, 0) + 1
    return counts


def _counts_to_schema(counts: dict[str, int]) -> str:
    return "|".join(f"{element}:{count}" for element, count in counts.items())


def _b_over_a_bin(record: Any) -> str:
    if length_pattern(record) == "ALL_DIFF":
        return ratio_bin(record.cell_b / max(record.cell_a, 1e-8))
    return ""


def collapse_penalty(record: Any, scaffold_fields: dict[str, str]) -> int:
    penalty = 0
    expected_position = scaffold_fields.get("position_family", "")
    expected_stack = scaffold_fields.get("stack_family", "")
    candidate_position = position_family(record)
    candidate_stack = stack_family(record)

    if expected_position == "GENERAL_DOMINANT" and candidate_position in {
        "HIGH_SPECIAL",
        "MIXED_SPECIAL",
    }:
        penalty += 2
    if expected_position == "MIXED_SPECIAL" and candidate_position == "HIGH_SPECIAL":
        penalty += 1
    if expected_stack == "NO_LAYER" and candidate_stack.startswith("LAYERED"):
        penalty += 2
    if expected_stack == "SKEW_DIAGONAL" and candidate_stack != "SKEW_DIAGONAL":
        penalty += 1
    return penalty


def verify_cif_against_scaffold(
    cif_text: str,
    scaffold_full: str,
    *,
    formula: str,
    space_group: int,
) -> dict[str, Any]:
    scaffold_valid, scaffold_errors, scaffold_fields = scaffold_quality_checks(
        scaffold_full,
        formula=formula,
        space_group=space_group,
        expect_full=True,
    )
    if not scaffold_valid:
        return {
            "valid_cif": False,
            "verify_errors": [f"invalid_scaffold:{error}" for error in scaffold_errors],
            "consistency_score": -999,
            "collapse_penalty": 999,
        }

    try:
        record = parse_cif_text(cif_text)
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "valid_cif": False,
            "verify_errors": [f"parse_failed:{exc}"],
            "consistency_score": -999,
            "collapse_penalty": 999,
        }

    errors: list[str] = []
    if not record.atom_sites:
        errors.append("empty_atom_sites")
    if min(float(record.cell_a), float(record.cell_b), float(record.cell_c)) <= 0:
        errors.append("zero_lattice")
    try:
        expected_formula = dict(reduced_counts(parse_formula_counts(formula)))
        candidate_formula = dict(
            reduced_counts(parse_formula_counts(record.formula_structural or ""))
        )
    except ValueError:
        errors.append("formula_structural_mismatch")
    else:
        if candidate_formula != expected_formula:
            errors.append("formula_structural_mismatch")

    expected_z = int(scaffold_fields["output_Z_raw"])
    expected_sites = int(scaffold_fields["n_listed_sites"])
    if int(record.z) != expected_z:
        errors.append("output_Z_raw_mismatch")
    if len(record.atom_sites) != expected_sites:
        errors.append("n_listed_sites_mismatch")

    site_schema = _counts_to_schema(_site_schema_counts(record))
    checks = {
        "site_schema": site_schema,
        "metric_family": metric_family(record),
        "length_pattern": length_pattern(record),
        "angle_pattern": angle_pattern(record),
        "vol_per_atom_bin": vol_per_atom_bin(record),
        "position_family": position_family(record),
        "stack_family": stack_family(record),
    }
    for field, candidate_value in checks.items():
        if candidate_value != scaffold_fields.get(field, candidate_value):
            errors.append(f"{field}_mismatch")

    angle_pattern_value = angle_pattern(record)
    if "eq_angle_bin" in scaffold_fields:
        if angle_pattern_value in {"ALPHA_EQ_BETA_NE_GAMMA", "ALPHA_EQ_GAMMA_NE_BETA"}:
            equal_angle_bin = angle_bin(record.alpha)
        elif angle_pattern_value == "BETA_EQ_GAMMA_NE_ALPHA":
            equal_angle_bin = angle_bin(record.beta)
        else:
            equal_angle_bin = ""
        if equal_angle_bin != scaffold_fields["eq_angle_bin"]:
            errors.append("eq_angle_bin_mismatch")
    else:
        equal_angle_bin = ""

    gamma_bin = angle_bin(record.gamma)
    b_over_a_bin = _b_over_a_bin(record)
    c_over_a = c_over_a_bin_family(record)
    optional_checks = {
        "gamma_bin": gamma_bin,
        "b_over_a_bin": b_over_a_bin,
        "c_over_a_bin_family": c_over_a,
    }
    for field, candidate_value in optional_checks.items():
        if field in scaffold_fields and candidate_value != scaffold_fields[field]:
            errors.append(f"{field}_mismatch")

    penalty = collapse_penalty(record, scaffold_fields)
    hard_errors = {
        "empty_atom_sites",
        "zero_lattice",
        "formula_structural_mismatch",
        "output_Z_raw_mismatch",
        "n_listed_sites_mismatch",
        "site_schema_mismatch",
    }
    score = 100
    score -= 12 * sum(error in hard_errors for error in errors)
    score -= 5 * sum(error not in hard_errors for error in errors)
    score -= 6 * penalty

    return {
        "valid_cif": not errors,
        "verify_errors": errors,
        "consistency_score": score,
        "collapse_penalty": penalty,
        "cif_fields": {
            "formula": record.formula_structural,
            "output_Z_raw": int(record.z),
            "n_listed_sites": len(record.atom_sites),
            **checks,
            "eq_angle_bin": equal_angle_bin,
            "gamma_bin": gamma_bin,
            "b_over_a_bin": b_over_a_bin,
            "c_over_a_bin_family": c_over_a,
        },
    }


def cif_candidate_score(candidate: dict[str, Any]) -> tuple[int, float, float, int]:
    return (
        int(bool(candidate.get("valid_cif"))),
        float(candidate.get("consistency_score", -999.0)),
        -float(candidate.get("collapse_penalty", 999.0)),
        int(bool(candidate.get("canonicalized"))),
    )


def generate_cif_candidates(
    runner: ModelRunner,
    *,
    formula: str,
    space_group: int,
    scaffold_full: str,
    num_samples: int,
    prototype_hint_file: Optional[str] = None,
    prototype_index_json: Optional[str] = None,
) -> list[dict[str, Any]]:
    input_text = material_prompt_input(formula, space_group) + "\n\n" + scaffold_full.strip()
    hints = load_prototype_hints(
        scaffold_full,
        prototype_hint_file=prototype_hint_file,
        prototype_index_json=prototype_index_json,
    )
    if hints:
        input_text += "\n\n" + hints

    raw_outputs = dedupe_texts(
        runner.generate(schema.CIF_WITH_SCAFFOLD_INSTRUCTION, input_text)
    )
    candidates: list[dict[str, Any]] = []
    for rank, response in enumerate(raw_outputs[:num_samples]):
        cif_text, canonicalized, canonicalize_error = canonicalize_generated_cif(response)
        verification = verify_cif_against_scaffold(
            cif_text,
            scaffold_full,
            formula=formula,
            space_group=space_group,
        )
        candidates.append(
            {
                "rank": rank,
                "canonicalized": canonicalized,
                "canonicalize_error": canonicalize_error,
                "cif": cif_text,
                **verification,
            }
        )
    return sorted(candidates, key=cif_candidate_score, reverse=True)


def select_best_cif(candidates: Sequence[dict[str, Any]]) -> Optional[dict[str, Any]]:
    return candidates[0] if candidates else None


@lru_cache(maxsize=8)
def _read_text_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


@lru_cache(maxsize=4)
def _read_json_file(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Prototype index must be a JSON object: {path}")
    return value


def load_prototype_hints(
    scaffold_full: str,
    *,
    prototype_hint_file: Optional[str],
    prototype_index_json: Optional[str],
) -> str:
    if prototype_hint_file:
        path = Path(prototype_hint_file)
        if not path.is_file():
            raise FileNotFoundError(f"Prototype hint file not found: {path}")
        return _read_text_file(str(path.resolve()))
    if prototype_index_json:
        path = Path(prototype_index_json)
        if not path.is_file():
            raise FileNotFoundError(f"Prototype index not found: {path}")
        return build_prototype_hints(dict(parse_scaffold_text(scaffold_full)), path)
    return ""


def build_scaffold_key_candidates(fields: dict[str, str]) -> list[tuple[str, str]]:
    ratio = fields.get("reduced_ratio", "")
    space_group = fields.get("condition_sg_number", "")
    output_z = fields.get("output_Z_raw", "")
    n_sites = fields.get("n_listed_sites", "")
    metric = fields.get("metric_family", "")
    position = fields.get("position_family", "")
    stack = fields.get("stack_family", "")
    bucket = fields.get("prototype_bucket_v4", "")
    return [
        ("level1", bucket),
        (
            "level2",
            "|".join(
                [ratio, space_group, f"z{output_z}", f"n{n_sites}", metric, position, stack]
            ),
        ),
        ("level3", "|".join([ratio, space_group, f"z{output_z}", f"n{n_sites}"])),
    ]


def build_prototype_hints(
    fields: dict[str, str],
    prototype_index_path: str | Path,
    *,
    top_k: int = 3,
) -> str:
    path = str(Path(prototype_index_path).resolve())
    index = _read_json_file(path)

    selected: list[dict[str, Any]] = []
    for level, key in build_scaffold_key_candidates(fields):
        for row in index.get(level, {}).get(key, []):
            if row not in selected:
                selected.append(row)
            if len(selected) >= top_k:
                break
        if len(selected) >= top_k:
            break
    if not selected:
        return ""

    lines = ["<PROTOTYPE_HINTS_V4>"]
    for index_value, row in enumerate(selected):
        lines.extend((f"[hint_{index_value}]", str(row.get("scaffold_full", "")).strip()))
    lines.append("</PROTOTYPE_HINTS_V4>")
    return "\n".join(lines)


def safe_file_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip()).strip("._")
    return stem or "sample"
