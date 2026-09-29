
from __future__ import annotations


from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

LLAMA_BOS = "<|begin_of_text|>"
LLAMA_EOT = "<|eot_id|>"
LLAMA_SYSTEM_HEADER = "<|start_header_id|>system<|end_header_id|>"
LLAMA_USER_HEADER = "<|start_header_id|>user<|end_header_id|>"
LLAMA_ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>"
SYSTEM_PROMPT = (
    "You are a crystallography model specialized in crystal-structure reasoning, scaffold planning, and CIF generation. "
    "Follow the user instruction exactly. Treat the provided chemical composition, space-group condition, and scaffold "
    "fields as constraints when present. Preserve required crystallographic notation, tags, delimiters, and CIF syntax. "
    "Return only the requested output without commentary or Markdown."
)


def _ensure_path(path: Path) -> None:
    import sys
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _load_training_helpers() -> Tuple[Any, Any, Any]:
    try:
        from ..train.training_utils import (
            build_prompt_text,
            lazy_import_training_stack,
            ensure_tokenizer_padding,
        )
        return build_prompt_text, lazy_import_training_stack, ensure_tokenizer_padding
    except Exception:
        THIS_DIR = Path(__file__).resolve().parent
        candidates = [
            THIS_DIR.parent / "train",
            THIS_DIR.parent / "v4_train",
            Path("/mnt/data/v4_train"),
        ]
        for cand in candidates:
            if (cand / "training_utils.py").exists():
                _ensure_path(cand)
                try:
                    from training_utils import (
                        build_prompt_text,
                        lazy_import_training_stack,
                        ensure_tokenizer_padding,
                    )
                    return build_prompt_text, lazy_import_training_stack, ensure_tokenizer_padding
                except Exception:
                    continue

    def build_prompt_text(instruction: str, input_text: str, response: str) -> str:
        instruction = str(instruction).strip()
        input_text = str(input_text).strip()
        user_message = f"{instruction}\n\nInput:\n{input_text}" if input_text else instruction
        return (
            f"{LLAMA_BOS}{LLAMA_SYSTEM_HEADER}\n\n{SYSTEM_PROMPT}{LLAMA_EOT}"
            f"{LLAMA_USER_HEADER}\n\n{user_message}{LLAMA_EOT}"
            f"{LLAMA_ASSISTANT_HEADER}\n\n{response}"
        )

    def lazy_import_training_stack() -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        errors: List[str] = []
        try:
            from unsloth import FastLanguageModel
            out["FastLanguageModel"] = FastLanguageModel
            out["backend"] = "unsloth"
        except Exception as e:
            errors.append(f"unsloth unavailable: {e}")
            out["backend"] = "transformers"
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            out["AutoModelForCausalLM"] = AutoModelForCausalLM
            out["AutoTokenizer"] = AutoTokenizer
            out["BitsAndBytesConfig"] = BitsAndBytesConfig
        except Exception as e:
            errors.append(f"transformers unavailable: {e}")
        try:
            from peft import PeftModel
            out["PeftModel"] = PeftModel
        except Exception as e:
            errors.append(f"peft unavailable: {e}")
        if "AutoTokenizer" not in out and "FastLanguageModel" not in out:
            raise ImportError("Could not import inference stack. " + " | ".join(errors))
        return out

    def ensure_tokenizer_padding(tokenizer: Any) -> None:
        if getattr(tokenizer, "pad_token", None) is None:
            if getattr(tokenizer, "eos_token", None) is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "<pad>"})
        tokenizer.padding_side = "right"

    return build_prompt_text, lazy_import_training_stack, ensure_tokenizer_padding


build_prompt_text, lazy_import_training_stack, ensure_tokenizer_padding = _load_training_helpers()


def _load_preprocess_v4():
    try:
        from ..preprocess_v4 import schema
        from ..preprocess_v4.cif_utils import (
            build_prompt_input,
            canonicalize_cif_text,
            parse_cif_text,
            crystal_system_from_sg,
            length_pattern,
            angle_pattern,
            ratio_bin,
            c_over_a_bin_family,
            angle_bin,
            vol_per_atom_bin,
            metric_family,
            position_family,
            stack_family,
            compile_full_scaffold,
            parse_scaffold_text,
        )
        return locals()
    except Exception:
        THIS_DIR = Path(__file__).resolve().parent
        candidates = [
            THIS_DIR.parent / "preprocess_v4",
            THIS_DIR.parent / "preprocess",
            Path("/mnt/data/v4_preprocess"),
        ]
        for cand in candidates:
            if (cand / "schema.py").exists() and (cand / "cif_utils.py").exists():
                _ensure_path(cand)
                try:
                    import schema
                    from cif_utils import (
                        build_prompt_input,
                        canonicalize_cif_text,
                        parse_cif_text,
                                                crystal_system_from_sg,
                        length_pattern,
                        angle_pattern,
                        ratio_bin,
                        c_over_a_bin_family,
                        angle_bin,
                        vol_per_atom_bin,
                        metric_family,
                        position_family,
                        stack_family,
                                                            compile_full_scaffold,
                        parse_scaffold_text,
                                )
                    return locals()
                except Exception:
                    continue
        raise ImportError("Could not locate V4 preprocess schema/cif_utils.")


_pp = _load_preprocess_v4()
schema = _pp["schema"]
build_prompt_input = _pp["build_prompt_input"]
canonicalize_cif_text = _pp["canonicalize_cif_text"]
parse_cif_text = _pp["parse_cif_text"]
crystal_system_from_sg = _pp["crystal_system_from_sg"]
length_pattern = _pp["length_pattern"]
angle_pattern = _pp["angle_pattern"]
ratio_bin = _pp["ratio_bin"]
c_over_a_bin_family = _pp["c_over_a_bin_family"]
angle_bin = _pp["angle_bin"]
vol_per_atom_bin = _pp["vol_per_atom_bin"]
metric_family = _pp["metric_family"]
position_family = _pp["position_family"]
stack_family = _pp["stack_family"]
compile_full_scaffold = _pp["compile_full_scaffold"]
parse_scaffold_text = _pp["parse_scaffold_text"]

DEFAULT_MAX_SEQ_LEN = 3072


@dataclass
class GenerationConfig:
    model_name: str = "/workspace/Llama-3.1-8B"
    adapter_path: Optional[str] = None
    load_in_4bit: bool = True
    dtype: Optional[str] = None
    max_seq_length: int = DEFAULT_MAX_SEQ_LEN
    max_new_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 0.92
    top_k: int = 40
    num_return_sequences: int = 1
    repetition_penalty: float = 1.02
    do_sample: bool = True


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def first_existing(row: Dict[str, str], keys: Sequence[str]) -> str:
    for key in keys:
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return ""


def write_text(path: str, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def score_scaffold_candidate(row: Dict[str, Any]) -> Tuple[int, int, int, int]:
    fields = row.get("fields", {})
    return (
        1 if row.get("valid_scaffold") else 0,
        -len(row.get("scaffold_errors", [])),
        1 if fields.get("position_family") == "GENERAL_DOMINANT" else 0,
        len(row.get("scaffold_full", "")),
    )


def score_cif_candidate(row: Dict[str, Any]) -> Tuple[int, float, float, int]:
    return (
        1 if row.get("valid_cif") else 0,
        float(row.get("consistency_score", -999.0)),
        -float(row.get("collapse_penalty", 999.0)),
        1 if row.get("canonicalized") else 0,
    )


def dedupe_texts(texts: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for txt in texts:
        key = " ".join(str(txt).strip().split())
        if key and key not in seen:
            out.append(str(txt).strip())
            seen.add(key)
    return out


def _extract_tagged_block(text: str, begin_tag: str, end_tag: str) -> str:
    if begin_tag in text and end_tag in text:
        start = text.index(begin_tag)
        end = text.index(end_tag, start) + len(end_tag)
        return text[start:end].strip()
    return str(text).strip()


def extract_scaffold_core_from_response(text: str) -> str:
    return _extract_tagged_block(text, schema.SCAFFOLD_CORE_BEGIN, schema.SCAFFOLD_CORE_END)


def extract_cif_from_response(text: str) -> str:
    text = str(text).strip().replace("</s>", "").replace("<|end_of_text|>", "").strip()
    if "data_" in text:
        text = text[text.index("data_"):]
    return text.strip()


def material_prompt_input(composition: str, space_group_number: int) -> str:
    return build_prompt_input(composition, int(space_group_number))


def _effective_max_new_tokens(*, input_ids, requested_max_new_tokens: int, max_seq_length: int, safety_margin: int = 16) -> int:
    prompt_len = int(input_ids.shape[-1])
    available = max_seq_length - prompt_len - safety_margin
    if available <= 0:
        return 1
    return max(1, min(int(requested_max_new_tokens), int(available)))


class ModelRunner:
    def __init__(self, cfg: GenerationConfig):
        self.cfg = cfg
        self.model = None
        self.tokenizer = None
        self.backend = None

    def load(self) -> None:
        stack = lazy_import_training_stack()
        self.backend = stack.get("backend", "transformers")
        model_name = self.cfg.model_name
        adapter_path = self.cfg.adapter_path

        if self.backend == "unsloth" and "FastLanguageModel" in stack:
            FastLanguageModel = stack["FastLanguageModel"]
            load_target = adapter_path if adapter_path else model_name
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=load_target,
                max_seq_length=self.cfg.max_seq_length,
                dtype=self.cfg.dtype,
                load_in_4bit=self.cfg.load_in_4bit,
            )
            ensure_tokenizer_padding(self.tokenizer)
            FastLanguageModel.for_inference(self.model)
            return

        AutoModelForCausalLM = stack["AutoModelForCausalLM"]
        AutoTokenizer = stack["AutoTokenizer"]
        BitsAndBytesConfig = stack.get("BitsAndBytesConfig")
        PeftModel = stack.get("PeftModel")

        quantization_config = None
        if self.cfg.load_in_4bit and BitsAndBytesConfig is not None:
            try:
                quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            except Exception:
                quantization_config = None

        tokenizer_source = adapter_path if adapter_path else model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=False)
        ensure_tokenizer_padding(self.tokenizer)

        import torch
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        if adapter_path and PeftModel is not None:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    def generate(self, instruction: str, input_text: str, *, response_prefix: str = "") -> List[str]:
        if self.model is None or self.tokenizer is None:
            self.load()
        prompt_text = build_prompt_text(instruction, input_text, response_prefix)
        toks = self.tokenizer(
            [prompt_text],
            return_tensors="pt",
            truncation=True,
            max_length=self.cfg.max_seq_length,
            add_special_tokens=False,
        )
        if hasattr(self.model, "device"):
            toks = {k: v.to(self.model.device) for k, v in toks.items()}
        max_new_tokens = _effective_max_new_tokens(
            input_ids=toks["input_ids"],
            requested_max_new_tokens=self.cfg.max_new_tokens,
            max_seq_length=self.cfg.max_seq_length,
            safety_margin=16,
        )
        eos_token_ids = []
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if isinstance(eos_id, int) and eos_id >= 0:
            eos_token_ids.append(eos_id)
        try:
            eot_id = self.tokenizer.convert_tokens_to_ids(LLAMA_EOT)
            unk_id = getattr(self.tokenizer, "unk_token_id", None)
            if isinstance(eot_id, int) and eot_id >= 0 and eot_id != unk_id and eot_id not in eos_token_ids:
                eos_token_ids.append(eot_id)
        except Exception:
            pass
        eos_token_id = eos_token_ids[0] if len(eos_token_ids) == 1 else eos_token_ids or None
        outputs = self.model.generate(
            **toks,
            max_new_tokens=max_new_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            top_k=self.cfg.top_k,
            num_return_sequences=self.cfg.num_return_sequences,
            repetition_penalty=self.cfg.repetition_penalty,
            do_sample=self.cfg.do_sample,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=eos_token_id,
        )
        prompt_len = int(toks["input_ids"].shape[-1])
        generated = outputs[:, prompt_len:]
        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=False)
        out: List[str] = []
        for text in decoded:
            text = str(text).split(LLAMA_EOT, 1)[0].split("<|end_of_text|>", 1)[0].strip()
            out.append((response_prefix + text).strip() if response_prefix else text)
        return out


def scaffold_quality_checks_v4(scaffold_text: str, *, formula: str, condition_sg_number: int, expect_full: bool = False) -> Tuple[bool, List[str], Dict[str, str]]:
    errors: List[str] = []
    fields = dict(parse_scaffold_text(scaffold_text))
    if not fields:
        return False, ["empty_scaffold"], {}

    if fields.get("formula", "") != formula:
        errors.append("formula_mismatch")
    if str(fields.get("condition_sg_number", "")) != str(int(condition_sg_number)):
        errors.append("condition_sg_mismatch")
    expected_cs = crystal_system_from_sg(int(condition_sg_number))
    if fields.get("condition_crystal_system", "") not in {"", expected_cs}:
        errors.append("crystal_system_mismatch")

    required_core = [
        "formula", "reduced_ratio", "condition_sg_number", "condition_crystal_system",
        "target_cell_mode", "output_Z_raw", "metric_family", "length_pattern",
        "angle_pattern", "vol_per_atom_bin", "position_family", "stack_family",
    ]
    for key in required_core:
        if key not in fields or fields[key] == "":
            errors.append(f"missing_{key}")

    if fields.get("target_cell_mode", "") not in {"", "primitive"}:
        errors.append("bad_target_cell_mode")

    if expect_full:
        for key in ["site_schema", "n_listed_sites", "formula_sum_expected", "geometry_signature_v4", "prototype_bucket_v4"]:
            if key not in fields or fields[key] == "":
                errors.append(f"missing_{key}")

    return len(errors) == 0, errors, fields


def maybe_compile_generated_core(response_text: str, *, formula: str, condition_sg_number: int) -> Tuple[str, str, bool, List[str], Dict[str, str]]:
    core_text = extract_scaffold_core_from_response(response_text)
    ok, errors, fields = scaffold_quality_checks_v4(core_text, formula=formula, condition_sg_number=condition_sg_number, expect_full=False)
    full_text = ""
    if ok:
        try:
            full_text = compile_full_scaffold(core_text, formula=formula)
            ok2, errors2, full_fields = scaffold_quality_checks_v4(full_text, formula=formula, condition_sg_number=condition_sg_number, expect_full=True)
            ok = ok and ok2
            errors = list(dict.fromkeys(errors + errors2))
            fields = full_fields
        except Exception as e:
            ok = False
            errors.append(f"compile_full_failed:{e}")
    return core_text, full_text, ok, errors, fields


def _looks_degenerate_cif(cif_text: str) -> bool:
    low = str(cif_text)
    if "data_unknown" in low:
        return True
    if "_chemical_formula_structural unknown" in low:
        return True
    if "_chemical_formula_sum ''" in low or "_chemical_formula_sum \"\"" in low:
        return True
    if "_cell_length_a 0.000000" in low and "_cell_length_b 0.000000" in low and "_cell_length_c 0.000000" in low:
        return True
    return False


def maybe_canonicalize_generated_cif(text: str) -> Tuple[str, bool, Optional[str]]:
    cif_text = extract_cif_from_response(text)
    try:
        can = canonicalize_cif_text(cif_text)
        if _looks_degenerate_cif(can):
            return can, False, "degenerate_canonicalized_cif"
        return can, True, None
    except Exception as e:
        return cif_text, False, str(e)


def _site_schema(rec) -> str:
    counts: Dict[str, int] = {}
    for site in rec.atom_sites:
        counts[site.element] = counts.get(site.element, 0) + 1
    return "|".join(f"{el}:{cnt}" for el, cnt in counts.items())


def _b_over_a_bin_if_needed(rec) -> str:
    if length_pattern(rec) == "ALL_DIFF":
        return ratio_bin(rec.cell_b / max(rec.cell_a, 1e-8))
    return ""


def _collapse_penalty(rec, scaffold_fields: Dict[str, str]) -> int:
    penalty = 0
    exp_pos = scaffold_fields.get("position_family", "")
    exp_stack = scaffold_fields.get("stack_family", "")
    cand_pos = position_family(rec)
    cand_stack = stack_family(rec)

    if exp_pos == "GENERAL_DOMINANT" and cand_pos in {"HIGH_SPECIAL", "MIXED_SPECIAL"}:
        penalty += 2
    if exp_pos == "MIXED_SPECIAL" and cand_pos == "HIGH_SPECIAL":
        penalty += 1
    if exp_stack == "NO_LAYER" and cand_stack.startswith("LAYERED"):
        penalty += 2
    if exp_stack == "SKEW_DIAGONAL" and cand_stack != "SKEW_DIAGONAL":
        penalty += 1
    return penalty


def verify_cif_against_scaffold(cif_text: str, scaffold_full_text: str, *, formula: str) -> Dict[str, Any]:
    scaffold_fields = dict(parse_scaffold_text(scaffold_full_text))
    out: Dict[str, Any] = {
        "valid_cif": False,
        "verify_errors": [],
        "consistency_score": -999,
        "collapse_penalty": 999,
    }
    try:
        rec = parse_cif_text(cif_text)
    except Exception as e:
        out["verify_errors"] = [f"parse_failed:{e}"]
        return out

    errors: List[str] = []
    if not rec.atom_sites:
        errors.append("empty_atom_sites")
    if float(rec.cell_a) <= 0 or float(rec.cell_b) <= 0 or float(rec.cell_c) <= 0:
        errors.append("zero_lattice")
    if str((rec.formula_structural or "").strip()) != str(formula).strip():
        errors.append("formula_structural_mismatch")
    if int(rec.z) != int(scaffold_fields.get("output_Z_raw", "1")):
        errors.append("output_Z_raw_mismatch")
    if len(rec.atom_sites) != int(scaffold_fields.get("n_listed_sites", str(len(rec.atom_sites)))):
        errors.append("n_listed_sites_mismatch")
    site_schema = _site_schema(rec)
    if site_schema != scaffold_fields.get("site_schema", site_schema):
        errors.append("site_schema_mismatch")
    if metric_family(rec) != scaffold_fields.get("metric_family", metric_family(rec)):
        errors.append("metric_family_mismatch")
    if length_pattern(rec) != scaffold_fields.get("length_pattern", length_pattern(rec)):
        errors.append("length_pattern_mismatch")
    if angle_pattern(rec) != scaffold_fields.get("angle_pattern", angle_pattern(rec)):
        errors.append("angle_pattern_mismatch")
    if "eq_angle_bin" in scaffold_fields:
        ap = angle_pattern(rec)
        if ap == "ALPHA_EQ_BETA_NE_GAMMA":
            cand_eq = angle_bin(rec.alpha)
        elif ap == "BETA_EQ_GAMMA_NE_ALPHA":
            cand_eq = angle_bin(rec.beta)
        elif ap == "ALPHA_EQ_GAMMA_NE_BETA":
            cand_eq = angle_bin(rec.alpha)
        else:
            cand_eq = ""
        if cand_eq != scaffold_fields.get("eq_angle_bin", cand_eq):
            errors.append("eq_angle_bin_mismatch")
    if "gamma_bin" in scaffold_fields and angle_bin(rec.gamma) != scaffold_fields.get("gamma_bin", angle_bin(rec.gamma)):
        errors.append("gamma_bin_mismatch")
    if "b_over_a_bin" in scaffold_fields and _b_over_a_bin_if_needed(rec) != scaffold_fields.get("b_over_a_bin", _b_over_a_bin_if_needed(rec)):
        errors.append("b_over_a_bin_mismatch")
    if "c_over_a_bin_family" in scaffold_fields and c_over_a_bin_family(rec) != scaffold_fields.get("c_over_a_bin_family", c_over_a_bin_family(rec)):
        errors.append("c_over_a_bin_family_mismatch")
    if vol_per_atom_bin(rec) != scaffold_fields.get("vol_per_atom_bin", vol_per_atom_bin(rec)):
        errors.append("vol_per_atom_bin_mismatch")
    if position_family(rec) != scaffold_fields.get("position_family", position_family(rec)):
        errors.append("position_family_mismatch")
    if stack_family(rec) != scaffold_fields.get("stack_family", stack_family(rec)):
        errors.append("stack_family_mismatch")

    penalty = _collapse_penalty(rec, scaffold_fields)
    score = 100
    hard = {
        "empty_atom_sites", "zero_lattice", "formula_structural_mismatch", "output_Z_raw_mismatch",
        "n_listed_sites_mismatch", "site_schema_mismatch",
    }
    score -= 12 * sum(1 for e in errors if e in hard)
    score -= 5 * sum(1 for e in errors if e not in hard)
    score -= 6 * penalty

    out.update({
        "valid_cif": len(errors) == 0,
        "verify_errors": errors,
        "consistency_score": score,
        "collapse_penalty": penalty,
        "cif_fields": {
            "formula": rec.formula_structural,
            "output_Z_raw": int(rec.z),
            "n_listed_sites": len(rec.atom_sites),
            "site_schema": site_schema,
            "metric_family": metric_family(rec),
            "length_pattern": length_pattern(rec),
            "angle_pattern": angle_pattern(rec),
            "eq_angle_bin": angle_bin(rec.alpha) if angle_pattern(rec) in {"ALPHA_EQ_BETA_NE_GAMMA","ALPHA_EQ_GAMMA_NE_BETA"} else (angle_bin(rec.beta) if angle_pattern(rec) == "BETA_EQ_GAMMA_NE_ALPHA" else ""),
            "gamma_bin": angle_bin(rec.gamma),
            "b_over_a_bin": _b_over_a_bin_if_needed(rec),
            "c_over_a_bin_family": c_over_a_bin_family(rec),
            "vol_per_atom_bin": vol_per_atom_bin(rec),
            "position_family": position_family(rec),
            "stack_family": stack_family(rec),
        },
    })
    return out



def build_prototype_hints(fields: Dict[str, str], prototype_index_path: str, *, top_k: int = 3) -> str:
    with open(prototype_index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    ratio = fields.get("reduced_ratio", "")
    sg = fields.get("condition_sg_number", "")
    z = fields.get("output_Z_raw", "")
    n = fields.get("n_listed_sites", "")
    keys = [
        ("level1", fields.get("prototype_bucket_v4", "")),
        ("level2", "|".join([ratio, sg, f"z{z}", f"n{n}", fields.get("metric_family", ""), fields.get("position_family", ""), fields.get("stack_family", "")])),
        ("level3", "|".join([ratio, sg, f"z{z}", f"n{n}"])),
    ]
    chosen: List[Dict[str, Any]] = []
    for level, key in keys:
        for row in index.get(level, {}).get(key, []):
            if row not in chosen:
                chosen.append(row)
            if len(chosen) >= top_k:
                break
        if len(chosen) >= top_k:
            break
    if not chosen:
        return ""
    lines = ["<PROTOTYPE_HINTS_V4>"]
    for i, row in enumerate(chosen):
        lines.extend([f"[hint_{i}]", str(row.get("scaffold_full", "")).strip()])
    lines.append("</PROTOTYPE_HINTS_V4>")
    return "\n".join(lines)
