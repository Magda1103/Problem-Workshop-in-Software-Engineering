from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import vipy


def extract_scene_payload(record: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(record, dict) or not record:
        raise ValueError("record is empty or not a dictionary")
    payload = next(iter(record.values()))
    if not isinstance(payload, dict):
        raise ValueError("record does not contain a scene dictionary payload")
    return payload


def resolve_video_path(filename: str, dataset_root: Path) -> Path:
    p = Path(filename)
    if p.is_absolute() and p.exists():
        return p
    candidate = dataset_root / p
    if candidate.exists():
        return candidate

    # PIP metadata may include a leading "videos/" while local data roots often
    # point directly at the videos directory (e.g., ./data).
    if p.parts and p.parts[0].lower() == "videos":
        candidate = dataset_root / Path(*p.parts[1:])
        if candidate.exists():
            return candidate

    return p


def is_valid_record(line: str, dataset_root: Path, require_existing_video: bool) -> bool:
    record = json.loads(line)
    payload = extract_scene_payload(record)
    scene = vipy.video.Scene.from_json(json.dumps(payload))
    video_path = resolve_video_path(scene.filename(), dataset_root)
    if require_existing_video and not video_path.exists():
        return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean JSONL by keeping only valid vipy scene entries"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("pip_370k_filtered.jsonl"),
        help="Input JSONL file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pip_370k_filtered.valid.jsonl"),
        help="Output JSONL file for valid entries",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data"),
        help="Root directory that contains category folders (default: ./data)",
    )
    parser.add_argument(
        "--allow-missing-video",
        action="store_true",
        help="Keep entries even when referenced video file is missing",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Replace input with cleaned content and create .bak backup",
    )
    return parser.parse_args()


def normalize_dataset_root(dataset_root: Path) -> Path:
    if dataset_root.is_absolute():
        return dataset_root
    project_root = Path(__file__).resolve().parent.parent
    return (project_root / dataset_root).resolve()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"input JSONL not found: {input_path}")

    require_existing_video = not args.allow_missing_video
    dataset_root = normalize_dataset_root(args.dataset_root)

    total = 0
    kept = 0
    removed_blank = 0
    removed_invalid = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            total += 1
            stripped = line.strip()
            if not stripped:
                removed_blank += 1
                continue

            try:
                if is_valid_record(stripped, dataset_root, require_existing_video=require_existing_video):
                    dst.write(stripped + "\n")
                    kept += 1
                else:
                    removed_invalid += 1
            except Exception:
                removed_invalid += 1

    if args.in_place:
        backup = input_path.with_suffix(input_path.suffix + ".bak")
        shutil.copy2(input_path, backup)
        output_path.replace(input_path)
        final_output = input_path
        print(f"Backup created: {backup}")
    else:
        final_output = output_path

    print(f"Input: {input_path}")
    print(f"Output: {final_output}")
    print(f"Dataset root: {dataset_root}")
    print(f"Total lines read: {total}")
    print(f"Kept valid entries: {kept}")
    print(f"Removed blank lines: {removed_blank}")
    print(f"Removed invalid entries: {removed_invalid}")


if __name__ == "__main__":
    main()
