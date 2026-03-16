from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Validate annotation JSONL paths against local video files"
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=project_root / "data" / "annotations.jsonl",
        help="Path to annotations JSONL file",
    )
    parser.add_argument(
        "--videos-root",
        type=Path,
        default=project_root / "data" / "videos",
        help="Root folder containing category subfolders with videos",
    )
    parser.add_argument(
        "--missing-report",
        type=Path,
        default=project_root / "data" / "annotations.missing.jsonl",
        help="Output JSONL report with missing entries",
    )
    parser.add_argument(
        "--scan-on-miss",
        action="store_true",
        help="If set, search videos-root recursively by basename when direct mapping fails",
    )
    return parser.parse_args()


def normalize_category(raw: str | None) -> str | None:
    if not raw:
        return None

    # PIP category metadata may include multiple classes and variants.
    # Example: "person_enters_car#Viewpoint=...,person_exits_car#...".
    token = raw.split(",", 1)[0].split("#", 1)[0].strip()
    if not token:
        return None

    if token.startswith("person_exits_car"):
        return "person_enters_car"

    return token


def extract_payload(record: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(record, dict) or not record:
        raise ValueError("record is empty or not a dictionary")

    payload = next(iter(record.values()))
    if not isinstance(payload, dict):
        raise ValueError("record does not contain a scene dictionary payload")

    return payload


def resolve_video_candidates(filename: str, videos_root: Path, category: str | None) -> list[Path]:
    p = Path(filename)
    candidates: list[Path] = []

    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(videos_root / p)

    if p.parts and p.parts[0].lower() == "videos" and len(p.parts) >= 2:
        candidates.append(videos_root / Path(*p.parts[1:]))

    if category:
        candidates.append(videos_root / category / p.name)

    # Remove duplicates while preserving order.
    unique: list[Path] = []
    seen: set[str] = set()
    for c in candidates:
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)

    return unique


def maybe_find_by_basename(videos_root: Path, filename: str, category: str | None) -> Path | None:
    basename = Path(filename).name
    if not basename:
        return None

    matches = list(videos_root.rglob(basename))
    if not matches:
        return None

    if category:
        for match in matches:
            if category.lower() in [part.lower() for part in match.parts]:
                return match

    return matches[0]


def main() -> None:
    args = parse_args()

    annotations_path = args.annotations.resolve()
    videos_root = args.videos_root.resolve()
    missing_report_path = args.missing_report.resolve()

    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_path}")
    if not videos_root.exists():
        raise FileNotFoundError(f"Videos root not found: {videos_root}")

    total = 0
    valid = 0
    invalid_json = 0
    missing = 0

    missing_report_path.parent.mkdir(parents=True, exist_ok=True)

    with annotations_path.open("r", encoding="utf-8") as src, missing_report_path.open(
        "w", encoding="utf-8"
    ) as missing_out:
        for line_no, line in enumerate(src, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            total += 1

            try:
                record = json.loads(stripped)
                payload = extract_payload(record)
                filename = str(payload.get("_filename", "")).strip()
                if not filename:
                    raise ValueError("missing _filename")

                raw_category = payload.get("_category")
                if not isinstance(raw_category, str):
                    attributes = payload.get("attributes")
                    if isinstance(attributes, dict):
                        cat_value = attributes.get("category")
                        raw_category = cat_value if isinstance(cat_value, str) else None
                    else:
                        raw_category = None

                category = normalize_category(raw_category)

                candidates = resolve_video_candidates(filename, videos_root, category)
                resolved = next((c for c in candidates if c.exists()), None)

                if resolved is None and args.scan_on_miss:
                    resolved = maybe_find_by_basename(videos_root, filename, category)

                if resolved is not None and resolved.exists():
                    valid += 1
                    continue

                missing += 1
                report = {
                    "line": line_no,
                    "filename": filename,
                    "category": category,
                    "candidate_paths": [str(c) for c in candidates],
                }
                missing_out.write(json.dumps(report, ensure_ascii=True) + "\n")

            except Exception as exc:
                invalid_json += 1
                report = {
                    "line": line_no,
                    "error": str(exc),
                }
                missing_out.write(json.dumps(report, ensure_ascii=True) + "\n")

    print(f"Annotations: {annotations_path}")
    print(f"Videos root: {videos_root}")
    print(f"Total records processed: {total}")
    print(f"Valid video paths: {valid}")
    print(f"Missing videos: {missing}")
    print(f"Invalid records: {invalid_json}")
    print(f"Missing report: {missing_report_path}")


if __name__ == "__main__":
    main()
