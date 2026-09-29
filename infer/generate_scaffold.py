from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from inference_utils import (
    GenerationConfig,
    ModelRunner,
    build_prototype_hints,
    dedupe_texts,
    material_prompt_input,
    maybe_compile_generated_core,
    schema,
    score_scaffold_candidate,
    set_seed,
    write_text,
)


def _write_json(path: str, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one V4 scaffold from composition + space group.")
    parser.add_argument("--model-name", default="/workspace/Llama-3.1-8B")
    parser.add_argument("--adapter-path", default="/workspace/crystal/test-time/ckpt/scaffold/adapter")
    parser.add_argument("--composition", required=True)
    parser.add_argument("--space-group", type=int, required=True)
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-scaffold-core-txt", default=None)
    parser.add_argument("--output-scaffold-full-txt", default=None)
    parser.add_argument("--prototype-index-json", default=None)
    args = parser.parse_args()

    set_seed(args.seed)

    cfg = GenerationConfig(
        model_name=args.model_name,
        adapter_path=args.adapter_path,
        num_return_sequences=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=1.03,
        do_sample=True,
        max_seq_length=1536,
    )
    runner = ModelRunner(cfg)
    input_text = material_prompt_input(args.composition, args.space_group)

    raw = dedupe_texts(runner.generate(schema.SCAFFOLD_CORE_INSTRUCTION, input_text))
    rows: List[Dict] = []
    seen_full = set()
    for i, resp in enumerate(raw):
        core_text, full_text, ok, errors, fields = maybe_compile_generated_core(
            resp,
            formula=args.composition,
            condition_sg_number=args.space_group,
        )
        full_key = " ".join(full_text.split()) if full_text else ""
        if full_key and full_key in seen_full:
            continue
        if full_key:
            seen_full.add(full_key)

        prototype_hints = ""
        if ok and args.prototype_index_json and Path(args.prototype_index_json).exists():
            try:
                prototype_hints = build_prototype_hints(fields, args.prototype_index_json)
            except Exception:
                prototype_hints = ""

        rows.append({
            "rank": i,
            "scaffold_core": core_text,
            "scaffold_full": full_text,
            "valid_scaffold": ok,
            "scaffold_errors": errors,
            "fields": fields,
            "prototype_hints": prototype_hints,
        })

    rows = sorted(rows, key=score_scaffold_candidate, reverse=True)
    best_row = rows[0] if rows else None

    if args.output_json:
        _write_json(args.output_json, rows)

    if best_row and args.output_scaffold_core_txt:
        write_text(args.output_scaffold_core_txt, best_row.get("scaffold_core", "").strip() + "\n")
    if best_row and args.output_scaffold_full_txt:
        write_text(args.output_scaffold_full_txt, best_row.get("scaffold_full", "").strip() + "\n")

    print(json.dumps(best_row if best_row else {"error": "no_scaffold_generated"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
