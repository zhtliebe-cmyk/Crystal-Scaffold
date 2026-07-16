from __future__ import annotations

import argparse
import json

try:
    from .inference_utils import (
        DEFAULT_MODEL_NAME,
        GenerationConfig,
        ModelRunner,
        generate_scaffold_candidates,
        select_best_scaffold,
        set_seed,
        write_json,
        write_text,
    )
except ImportError:
    from inference_utils import (
        DEFAULT_MODEL_NAME,
        GenerationConfig,
        ModelRunner,
        generate_scaffold_candidates,
        select_best_scaffold,
        set_seed,
        write_json,
        write_text,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and rank V4 scaffold candidates for one material condition."
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--adapter-path", default="ckpt/scaffold/adapter")
    parser.add_argument("--composition", required=True)
    parser.add_argument("--space-group", type=int, required=True)
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.28)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-seq-length", type=int, default=1536)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default=None)
    parser.add_argument("--load-in-4bit", action="store_true", dest="load_in_4bit")
    parser.add_argument("--no-load-in-4bit", action="store_false", dest="load_in_4bit")
    parser.set_defaults(load_in_4bit=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-scaffold-core", default=None)
    parser.add_argument("--output-scaffold-full", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    runner = ModelRunner(
        GenerationConfig(
            model_name=args.model_name,
            adapter_path=args.adapter_path,
            load_in_4bit=args.load_in_4bit,
            dtype=args.dtype,
            max_seq_length=args.max_seq_length,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            num_return_sequences=args.num_samples,
            repetition_penalty=1.03,
        )
    )
    candidates = generate_scaffold_candidates(
        runner,
        formula=args.composition,
        space_group=args.space_group,
        num_samples=args.num_samples,
    )
    best = select_best_scaffold(candidates)
    if best is None:
        raise RuntimeError("No scaffold candidate was generated")

    if args.output_json:
        write_json(args.output_json, {"best": best, "all": candidates})
    if args.output_scaffold_core:
        write_text(args.output_scaffold_core, best["scaffold_core"].strip() + "\n")
    if args.output_scaffold_full:
        write_text(args.output_scaffold_full, best["scaffold_full"].strip() + "\n")
    print(json.dumps(best, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
