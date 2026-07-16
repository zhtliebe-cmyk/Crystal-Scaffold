from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .inference_utils import (
        DEFAULT_MODEL_NAME,
        GenerationConfig,
        ModelRunner,
        generate_cif_candidates,
        select_best_cif,
        set_seed,
        write_json,
        write_text,
    )
except ImportError:
    from inference_utils import (
        DEFAULT_MODEL_NAME,
        GenerationConfig,
        ModelRunner,
        generate_cif_candidates,
        select_best_cif,
        set_seed,
        write_json,
        write_text,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate, verify, and rank V4 CIF candidates for one compiled scaffold."
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--adapter-path", default="ckpt/cif/adapter")
    parser.add_argument("--composition", required=True)
    parser.add_argument("--space-group", type=int, required=True)
    parser.add_argument("--scaffold-full-file", required=True)
    parser.add_argument("--prototype-index-json", default=None)
    parser.add_argument("--prototype-hint-file", default=None)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.66)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--max-new-tokens", type=int, default=1800)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default=None)
    parser.add_argument("--load-in-4bit", action="store_true", dest="load_in_4bit")
    parser.add_argument("--no-load-in-4bit", action="store_false", dest="load_in_4bit")
    parser.set_defaults(load_in_4bit=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output-cif", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--strict-valid-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.greedy:
        args.num_samples = 1
    set_seed(args.seed)

    scaffold_path = Path(args.scaffold_full_file)
    if not scaffold_path.is_file():
        raise FileNotFoundError(f"Scaffold file not found: {scaffold_path}")
    scaffold_full = scaffold_path.read_text(encoding="utf-8").strip()

    runner = ModelRunner(
        GenerationConfig(
            model_name=args.model_name,
            adapter_path=args.adapter_path,
            load_in_4bit=args.load_in_4bit,
            dtype=args.dtype,
            max_seq_length=args.max_seq_length,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0 if args.greedy else args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            num_return_sequences=args.num_samples,
            repetition_penalty=1.05,
            do_sample=not args.greedy,
        )
    )
    candidates = generate_cif_candidates(
        runner,
        formula=args.composition,
        space_group=args.space_group,
        scaffold_full=scaffold_full,
        num_samples=args.num_samples,
        prototype_hint_file=args.prototype_hint_file,
        prototype_index_json=args.prototype_index_json,
    )
    best = select_best_cif(candidates)
    if best is None:
        raise RuntimeError("No CIF candidate was generated")
    if args.strict_valid_only and not best["valid_cif"]:
        print(
            json.dumps(
                {"error": "no_valid_cif_candidate", "top_candidate": best},
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    write_text(args.output_cif, best["cif"].strip() + "\n")
    if args.output_json:
        write_json(args.output_json, {"best": best, "all": candidates})
    print(json.dumps(best, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
