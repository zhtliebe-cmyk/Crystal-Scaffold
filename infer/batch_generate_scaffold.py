from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional

try:
    from .inference_utils import (
        DEFAULT_MODEL_NAME,
        GenerationConfig,
        ModelRunner,
        generate_scaffold_candidates,
        select_best_scaffold,
        set_seed,
        validate_space_group,
    )
except ImportError:
    from inference_utils import (
        DEFAULT_MODEL_NAME,
        GenerationConfig,
        ModelRunner,
        generate_scaffold_candidates,
        select_best_scaffold,
        set_seed,
        validate_space_group,
    )

INPUT_COLUMNS = ("material_id", "pretty_formula", "space_group")
OUTPUT_COLUMNS = (
    "material_id",
    "pretty_formula",
    "space_group",
    "scaffold_core",
    "scaffold_full",
    "valid_scaffold",
    "scaffold_errors",
)


def load_rows(input_csv: str, first_n: Optional[int]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with Path(input_csv).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in INPUT_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing columns in {input_csv}: {missing}")
        for line_number, row in enumerate(reader, start=2):
            material_id = str(row["material_id"]).strip()
            formula = str(row["pretty_formula"]).strip()
            if not material_id or not formula:
                raise ValueError(f"Empty material_id or pretty_formula at CSV line {line_number}")
            try:
                space_group = validate_space_group(int(float(row["space_group"])))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid space_group at CSV line {line_number}: {exc}") from exc
            rows.append(
                {
                    "material_id": material_id,
                    "pretty_formula": formula,
                    "space_group": str(space_group),
                }
            )
            if first_n is not None and len(rows) >= first_n:
                break
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate one ranked V4 scaffold for every row in the four-column dataset."
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--adapter-path", default="ckpt/scaffold/adapter")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--first-n", type=int, default=None)
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.first_n is not None and args.first_n <= 0:
        raise ValueError("first_n must be positive")
    set_seed(args.seed)
    input_rows = load_rows(args.input_csv, args.first_n)
    if not input_rows:
        raise ValueError(f"No rows found in {args.input_csv}")

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

    output_rows: list[dict[str, str]] = []
    for index, row in enumerate(input_rows, start=1):
        print(
            f"[INFO] {index}/{len(input_rows)} material_id={row['material_id']} "
            f"formula={row['pretty_formula']} sg={row['space_group']}",
            flush=True,
        )
        candidates = generate_scaffold_candidates(
            runner,
            formula=row["pretty_formula"],
            space_group=int(row["space_group"]),
            num_samples=args.num_samples,
        )
        best = select_best_scaffold(candidates)
        output_rows.append(
            {
                **row,
                "scaffold_core": "" if best is None else best["scaffold_core"].strip(),
                "scaffold_full": "" if best is None else best["scaffold_full"].strip(),
                "valid_scaffold": str(bool(best and best["valid_scaffold"])),
                "scaffold_errors": json.dumps(
                    ["generation_failed"] if best is None else best["scaffold_errors"],
                    ensure_ascii=False,
                ),
            }
        )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"[OK] wrote {len(output_rows)} rows to {output_path}", flush=True)


if __name__ == "__main__":
    main()
