from __future__ import annotations

import json
from collections import Counter
from decimal import Decimal
from pathlib import Path

import ijson


def json_default(value: object) -> object:
    if isinstance(value, Decimal):
        # ijson may parse numbers as Decimal; convert for JSONL output.
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def load_categories(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip() and not line.strip().startswith("#")}


def extract_category(record: object) -> str | None:
    if not isinstance(record, dict):
        return None

    # Records are shaped like {"<class 'vipy.video.Scene'>": {...scene...}}
    for value in record.values():
        if not isinstance(value, dict):
            continue

        scene_category = value.get("_category")
        if isinstance(scene_category, str) and scene_category:
            return scene_category

        attributes = value.get("attributes")
        if isinstance(attributes, dict):
            raw = attributes.get("category")
            if isinstance(raw, str) and raw:
                # Keep the first category base (strip any '#key=value' suffix)
                first = raw.split(",", 1)[0]
                return first.split("#", 1)[0].strip()

    return None


def main() -> None:
    workspace = Path(__file__).resolve().parent
    categories_file = workspace / "category_list.txt"
    input_file = Path("E:/pip370k/pip_370k/pip_370k.json")
    output_file = workspace / "pip_370k_filtered.jsonl"

    categories = load_categories(categories_file)
    if not categories:
        raise RuntimeError(f"No categories found in {categories_file}")

    total = 0
    kept = 0
    kept_by_category: Counter[str] = Counter()

    with input_file.open("rb") as src, output_file.open("w", encoding="utf-8") as dst:
        for item in ijson.items(src, "item"):
            total += 1
            category = extract_category(item)
            if category in categories:
                dst.write(json.dumps(item, ensure_ascii=False, default=json_default) + "\n")
                kept += 1
                kept_by_category[category] += 1

            if total % 100000 == 0:
                print(f"Processed {total:,} items, kept {kept:,}...")

    print(f"Done. Processed {total:,} items, kept {kept:,} items.")
    print(f"Output: {output_file}")
    if kept_by_category:
        print("Kept by category:")
        for category, count in sorted(kept_by_category.items()):
            print(f"  {category}: {count:,}")


if __name__ == "__main__":
    main()
