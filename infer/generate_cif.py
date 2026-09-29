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
    maybe_canonicalize_generated_cif,
    parse_scaffold_text,
    schema,
    score_cif_candidate,
    set_seed,
    verify_cif_against_scaffold,
    write_text,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one V4 CIF from composition + space group + scaffold_full.")
    parser.add_argument("--model-name", default="/workspace/Llama-3.1-8B")
    parser.add_argument("--adapter-path", default="/workspace/crystal/test-time/ckpt/cif/adapter")
    parser.add_argument("--composition", required=True)
    parser.add_argument("--space-group", type=int, required=True)
    parser.add_argument("--scaffold-full-file", required=True)
    parser.add_argument("--prototype-index-json", default=None)
    parser.add_argument("--prototype-hint-file", default=None)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--max-new-tokens", type=int, default=1800)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output-cif", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--strict-valid-only", action="store_true", help="Do not write CIF if all candidates are invalid.")
    args = parser.parse_args()

    set_seed(args.seed)

    if args.greedy:
        args.num_samples = 1

    scaffold_full = Path(args.scaffold_full_file).read_text(encoding="utf-8").strip()
    input_text = material_prompt_input(args.composition, args.space_group) + "\n\n" + scaffold_full

    if args.prototype_hint_file and Path(args.prototype_hint_file).exists():
        input_text += "\n\n" + Path(args.prototype_hint_file).read_text(encoding="utf-8").strip()
    elif args.prototype_index_json and Path(args.prototype_index_json).exists():
        try:
            fields = dict(parse_scaffold_text(scaffold_full))
            hints = build_prototype_hints(fields, args.prototype_index_json)
            if hints:
                input_text += "\n\n" + hints
        except Exception:
            pass

    cfg = GenerationConfig(
        model_name=args.model_name,
        adapter_path=args.adapter_path,
        num_return_sequences=args.num_samples,
        temperature=1.0 if args.greedy else args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=1.05,
        do_sample=not args.greedy,
        max_seq_length=4096,
    )
    runner = ModelRunner(cfg)
    raw = dedupe_texts(runner.generate(schema.CIF_WITH_SCAFFOLD_INSTRUCTION, input_text))

    rows: List[Dict] = []
    for i, resp in enumerate(raw):
        cif_text, canonicalized, can_err = maybe_canonicalize_generated_cif(resp)
        verify = verify_cif_against_scaffold(
            cif_text,
            scaffold_full,
            formula=args.composition,
        )
        rows.append({
            "rank": i,
            "canonicalized": canonicalized,
            "canonicalize_error": can_err,
            "cif": cif_text,
            **verify,
        })

    rows = sorted(rows, key=score_cif_candidate, reverse=True)
    best = rows[0] if rows else None

    if best is None:
        raise RuntimeError("No CIF candidate generated.")

    if args.strict_valid_only and not best.get("valid_cif", False):
        print(json.dumps({"error": "no_valid_cif_candidate", "top_candidate": best}, indent=2, ensure_ascii=False))
        return

    write_text(args.output_cif, best["cif"].strip() + "\n")
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({"best": best, "all": rows}, f, indent=2, ensure_ascii=False)

    print(json.dumps(best, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
