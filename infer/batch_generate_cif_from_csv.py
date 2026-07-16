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
        generate_cif_candidates,
        safe_file_stem,
        select_best_cif,
        set_seed,
        validate_space_group,
        write_text,
    )
except ImportError:
    from inference_utils import (
        DEFAULT_MODEL_NAME,
        GenerationConfig,
        ModelRunner,
        generate_cif_candidates,
        safe_file_stem,
        select_best_cif,
        set_seed,
        validate_space_group,
        write_text,
    )

INPUT_COLUMNS = ("material_id", "pretty_formula", "space_group", "scaffold_full")
OUTPUT_COLUMNS = (
    "material_id",
    "pretty_formula",
    "space_group",
    "scaffold_full",
    "cif_path",
    "selected_rank",
    "valid_cif",
    "canonicalized",
    "canonicalize_error",
    "consistency_score",
    "collapse_penalty",
    "verify_errors",
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
            scaffold_full = str(row["scaffold_full"]).strip()
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
                    "scaffold_full": scaffold_full,
                }
            )
            if first_n is not None and len(rows) >= first_n:
                break
    return rows


def failed_summary(row: dict[str, str], error: str) -> dict[str, str]:
    return {
        **row,
        "cif_path": "",
        "selected_rank": "",
        "valid_cif": "False",
        "canonicalized": "False",
        "canonicalize_error": error,
        "consistency_score": "",
        "collapse_penalty": "",
        "verify_errors": json.dumps([error], ensure_ascii=False),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate one selected V4 CIF for every scaffold row and preserve material_id."
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--adapter-path", default="ckpt/cif/adapter")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--first-n", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.66)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--max-new-tokens", type=int, default=1800)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--prototype-hint-file", default=None)
    parser.add_argument("--prototype-index-json", default=None)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default=None)
    parser.add_argument("--load-in-4bit", action="store_true", dest="load_in_4bit")
    parser.add_argument("--no-load-in-4bit", action="store_false", dest="load_in_4bit")
    parser.set_defaults(load_in_4bit=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--strict-valid-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.first_n is not None and args.first_n <= 0:
        raise ValueError("first_n must be positive")
    if args.greedy:
        args.num_samples = 1
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
            temperature=0.0 if args.greedy else args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            num_return_sequences=args.num_samples,
            repetition_penalty=1.05,
            do_sample=not args.greedy,
        )
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, str]] = []

    for index, row in enumerate(input_rows, start=1):
        print(
            f"[INFO] {index}/{len(input_rows)} material_id={row['material_id']} "
            f"formula={row['pretty_formula']} sg={row['space_group']}",
            flush=True,
        )
        if not row["scaffold_full"]:
            summary_rows.append(failed_summary(row, "missing_scaffold_full"))
            continue

        candidates = generate_cif_candidates(
            runner,
            formula=row["pretty_formula"],
            space_group=int(row["space_group"]),
            scaffold_full=row["scaffold_full"],
            num_samples=args.num_samples,
            prototype_hint_file=args.prototype_hint_file,
            prototype_index_json=args.prototype_index_json,
        )
        best = select_best_cif(candidates)
        if best is None:
            summary_rows.append(failed_summary(row, "generation_failed"))
            continue
        if args.strict_valid_only and not best["valid_cif"]:
            summary_rows.append(
                {
                    **failed_summary(row, "no_valid_cif_candidate"),
                    "selected_rank": str(best["rank"]),
                    "canonicalized": str(bool(best["canonicalized"])),
                    "canonicalize_error": str(best.get("canonicalize_error") or ""),
                    "consistency_score": str(best["consistency_score"]),
                    "collapse_penalty": str(best["collapse_penalty"]),
                    "verify_errors": json.dumps(best["verify_errors"], ensure_ascii=False),
                }
            )
            continue

        cif_path = output_dir / f"{safe_file_stem(row['material_id'])}.cif"
        write_text(cif_path, best["cif"].strip() + "\n")
        summary_rows.append(
            {
                **row,
                "cif_path": str(cif_path),
                "selected_rank": str(best["rank"]),
                "valid_cif": str(bool(best["valid_cif"])),
                "canonicalized": str(bool(best["canonicalized"])),
                "canonicalize_error": str(best.get("canonicalize_error") or ""),
                "consistency_score": str(best["consistency_score"]),
                "collapse_penalty": str(best["collapse_penalty"]),
                "verify_errors": json.dumps(best["verify_errors"], ensure_ascii=False),
            }
        )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[OK] wrote {len(summary_rows)} rows to {output_csv}", flush=True)


if __name__ == "__main__":
    main()
