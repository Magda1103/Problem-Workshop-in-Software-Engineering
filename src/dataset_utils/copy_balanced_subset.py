from __future__ import annotations

import argparse
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoFile:
    category: str
    source: Path
    size_bytes: int


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    parser = argparse.ArgumentParser(
        description=(
            "Copy a balanced subset of videos from source categories into "
            "../data/videos/<category> while staying under a total size cap."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(r"E:\pip370k\pip_370k\videos"),
        help="Root source folder that contains category subfolders",
    )
    parser.add_argument(
        "--dest-root",
        type=Path,
        default=project_root / "data" / "videos",
        help="Destination root folder (default: ../data/videos)",
    )
    parser.add_argument(
        "--category-list",
        type=Path,
        default=project_root / "category_list.txt",
        help="Path to category list file",
    )
    parser.add_argument(
        "--max-gb",
        type=float,
        default=6.0,
        help="Maximum total destination size in GB",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=711,
        help="Seed used for tie-break randomization",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan selections but do not copy files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination files if they already exist",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".mp4"],
        help="Allowed file extensions (default: .mp4)",
    )
    return parser.parse_args()


def read_categories(path: Path) -> list[str]:
    categories = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    categories = [c for c in categories if c and not c.startswith("#")]
    if not categories:
        raise ValueError(f"No categories found in {path}")
    return categories


def bytes_from_gb(max_gb: float) -> int:
    if max_gb <= 0:
        raise ValueError("--max-gb must be greater than 0")
    return int(max_gb * (1024**3))


def collect_candidates(
    source_root: Path,
    categories: list[str],
    allowed_exts: set[str],
    dest_root: Path,
    overwrite: bool,
) -> dict[str, list[VideoFile]]:
    candidates: dict[str, list[VideoFile]] = {}

    for category in categories:
        src_dir = source_root / category
        if not src_dir.exists():
            raise FileNotFoundError(f"Missing source category folder: {src_dir}")

        files: list[VideoFile] = []
        for p in src_dir.iterdir():
            if not p.is_file():
                continue
            if p.suffix.lower() not in allowed_exts:
                continue

            dest_file = dest_root / category / p.name
            if dest_file.exists() and not overwrite:
                continue

            files.append(VideoFile(category=category, source=p, size_bytes=p.stat().st_size))

        files.sort(key=lambda v: (v.size_bytes, v.source.name.lower()))
        candidates[category] = files

    return candidates


def get_existing_destination_stats(
    dest_root: Path,
    categories: list[str],
    allowed_exts: set[str],
) -> tuple[int, dict[str, int], dict[str, int]]:
    total_bytes = 0
    bytes_per_category: dict[str, int] = defaultdict(int)
    count_per_category: dict[str, int] = defaultdict(int)

    for category in categories:
        d = dest_root / category
        if not d.exists():
            continue

        for p in d.iterdir():
            if not p.is_file() or p.suffix.lower() not in allowed_exts:
                continue
            size = p.stat().st_size
            total_bytes += size
            bytes_per_category[category] += size
            count_per_category[category] += 1

    return total_bytes, bytes_per_category, count_per_category


def choose_balanced_subset(
    candidates: dict[str, list[VideoFile]],
    max_total_bytes: int,
    initial_total_bytes: int,
    initial_counts: dict[str, int],
    seed: int,
) -> list[VideoFile]:
    rng = random.Random(seed)

    selected: list[VideoFile] = []
    total_bytes = initial_total_bytes
    offsets = {category: 0 for category in candidates}
    counts = defaultdict(int, initial_counts)

    while True:
        viable: list[tuple[int, float, str]] = []
        for category, files in candidates.items():
            idx = offsets[category]
            if idx >= len(files):
                continue

            next_file = files[idx]
            if total_bytes + next_file.size_bytes > max_total_bytes:
                continue

            viable.append((counts[category], rng.random(), category))

        if not viable:
            break

        viable.sort(key=lambda x: (x[0], x[1]))
        _, _, chosen_category = viable[0]

        file_to_add = candidates[chosen_category][offsets[chosen_category]]
        offsets[chosen_category] += 1

        selected.append(file_to_add)
        counts[chosen_category] += 1
        total_bytes += file_to_add.size_bytes

    return selected


def copy_selected(
    files: list[VideoFile],
    dest_root: Path,
    dry_run: bool,
    overwrite: bool,
) -> tuple[int, dict[str, int], dict[str, int]]:
    copied_bytes = 0
    copied_counts: dict[str, int] = defaultdict(int)
    copied_bytes_per_category: dict[str, int] = defaultdict(int)

    for vf in files:
        dest_dir = dest_root / vf.category
        dest_file = dest_dir / vf.source.name

        if dest_file.exists() and not overwrite:
            continue

        if not dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(vf.source, dest_file)

        copied_bytes += vf.size_bytes
        copied_counts[vf.category] += 1
        copied_bytes_per_category[vf.category] += vf.size_bytes

    return copied_bytes, copied_counts, copied_bytes_per_category


def format_gb(size_bytes: int) -> str:
    return f"{size_bytes / (1024**3):.3f} GB"


def main() -> None:
    args = parse_args()

    source_root = args.source_root.resolve()
    dest_root = args.dest_root.resolve()
    category_list = args.category_list.resolve()
    max_total_bytes = bytes_from_gb(args.max_gb)
    allowed_exts = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in args.extensions}

    categories = read_categories(category_list)

    if not source_root.exists():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")

    initial_total, initial_bytes_per_cat, initial_counts = get_existing_destination_stats(
        dest_root=dest_root,
        categories=categories,
        allowed_exts=allowed_exts,
    )

    if initial_total >= max_total_bytes:
        raise RuntimeError(
            "Destination already meets or exceeds size limit: "
            f"{format_gb(initial_total)} >= {format_gb(max_total_bytes)}"
        )

    candidates = collect_candidates(
        source_root=source_root,
        categories=categories,
        allowed_exts=allowed_exts,
        dest_root=dest_root,
        overwrite=args.overwrite,
    )

    selected = choose_balanced_subset(
        candidates=candidates,
        max_total_bytes=max_total_bytes,
        initial_total_bytes=initial_total,
        initial_counts=initial_counts,
        seed=args.seed,
    )

    copied_bytes, copied_counts, copied_bytes_per_cat = copy_selected(
        files=selected,
        dest_root=dest_root,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )

    final_total = initial_total + copied_bytes

    mode = "DRY RUN" if args.dry_run else "COPY"
    print(f"[{mode}] Source root: {source_root}")
    print(f"[{mode}] Destination root: {dest_root}")
    print(f"[{mode}] Categories: {len(categories)}")
    print(f"[{mode}] Initial size: {format_gb(initial_total)}")
    print(f"[{mode}] Added size:   {format_gb(copied_bytes)}")
    print(f"[{mode}] Final size:   {format_gb(final_total)} / {format_gb(max_total_bytes)}")
    print(f"[{mode}] Added files:  {sum(copied_counts.values())}")
    print()
    print("Per-category totals (existing + added):")

    final_counts = dict(initial_counts)
    final_bytes = dict(initial_bytes_per_cat)
    for category in categories:
        final_counts[category] = final_counts.get(category, 0) + copied_counts.get(category, 0)
        final_bytes[category] = final_bytes.get(category, 0) + copied_bytes_per_cat.get(category, 0)
        print(
            f"- {category}: "
            f"count={final_counts[category]} "
            f"size={format_gb(final_bytes[category])} "
            f"(added={copied_counts.get(category, 0)})"
        )


if __name__ == "__main__":
    main()