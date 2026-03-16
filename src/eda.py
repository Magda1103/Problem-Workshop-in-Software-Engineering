from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import vipy


def maybe_import_pycollector() -> str:
	try:
		import pycollector.version as pyc_version  # type: ignore

		version = getattr(pyc_version, "__version__", None)
		return f"pycollector available (version={version or 'unknown'})"
	except Exception as exc:  # pragma: no cover - informational path
		return f"pycollector import warning: {exc}"


def reservoir_sample_jsonl(path: Path, seed: int | None = None) -> dict[str, Any]:
	if seed is not None:
		random.seed(seed)

	chosen: dict[str, Any] | None = None
	with path.open("r", encoding="utf-8") as f:
		for i, line in enumerate(f, start=1):
			line = line.strip()
			if not line:
				continue
			obj = json.loads(line)
			if random.randrange(i) == 0:
				chosen = obj

	if chosen is None:
		raise RuntimeError(f"No JSON objects found in {path}")

	return chosen


def extract_scene_payload(record: dict[str, Any]) -> dict[str, Any]:
	if not isinstance(record, dict) or not record:
		raise ValueError("Record is empty or not a dictionary")

	payload = next(iter(record.values()))
	if not isinstance(payload, dict):
		raise ValueError("Record does not contain a scene dictionary payload")
	return payload


def resolve_video_path(filename: str, dataset_root: Path) -> Path:
	p = Path(filename)
	if p.is_absolute() and p.exists():
		return p

	candidate = dataset_root / p
	if candidate.exists():
		return candidate

	# Annotation metadata often stores filenames as videos/<category>/<video>.
	# Local subset is rooted at data/videos, so strip the leading videos/ segment.
	if p.parts and p.parts[0].lower() == "videos":
		candidate = dataset_root / Path(*p.parts[1:])
		if candidate.exists():
			return candidate

	return p


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="EDA: load one random PIP-370K sample and loop annotated playback"
	)
	parser.add_argument(
		"--jsonl",
		type=Path,
		default=Path("data/annotations.jsonl"),
		help="Path to filtered JSONL file",
	)
	parser.add_argument(
		"--dataset-root",
		type=Path,
		default=Path("data/videos"),
		help="Root directory that contains <category>/<video> under data/videos",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=None,
		help="Optional RNG seed for reproducible random pick",
	)
	parser.add_argument(
		"--fps",
		type=float,
		default=10.0,
		help="Playback FPS for visualization",
	)
	return parser.parse_args()


def _set_window_title(figure: int, title: str) -> None:
	"""Best-effort window title update for matplotlib-backed vipy display."""
	fig = plt.figure(figure)
	manager = getattr(fig.canvas, "manager", None)
	if manager is not None and hasattr(manager, "set_window_title"):
		manager.set_window_title(title)


def play_scene_loop(scene: vipy.video.Scene, fps: float, figure: int = 1) -> None:
	if fps <= 0:
		raise ValueError("FPS must be positive")

	target_fps = min(fps, scene.framerate()) if scene.framerate() is not None else fps
	if target_fps <= 0:
		raise ValueError("Invalid target FPS")

	mutator = vipy.image.mutator_show_noun_verb()
	last_tick = time.perf_counter()
	window_closed = False

	fig = plt.figure(figure)

	def _on_close(_event: Any) -> None:
		nonlocal window_closed
		window_closed = True

	fig.canvas.mpl_connect("close_event", _on_close)

	while True:
		with vipy.util.Stopwatch() as sw:
			for k, im in enumerate(scene.load() if scene.isloaded() else scene.stream()):
				if window_closed or not plt.fignum_exists(figure):
					return

				time.sleep(max(0.0, (1.0 / target_fps) - sw.since()))
				mutator(im, k).show(figure=figure, theme="dark")

				if window_closed or not plt.fignum_exists(figure):
					return

				now = time.perf_counter()
				actual_fps = 1.0 / max(now - last_tick, 1e-9)
				last_tick = now
				_set_window_title(figure, f"EDA Playback - {actual_fps:.1f} FPS")

				if vipy.globals._user_hit_escape():
					return


def main() -> None:
	args = parse_args()

	print(maybe_import_pycollector())

	jsonl_path = args.jsonl.resolve()
	if not jsonl_path.exists():
		raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

	record = reservoir_sample_jsonl(jsonl_path, seed=args.seed)
	scene_payload = extract_scene_payload(record)

	scene = vipy.video.Scene.from_json(json.dumps(scene_payload))

	original_filename = scene.filename()
	resolved_video = resolve_video_path(original_filename, args.dataset_root)
	scene.filename(str(resolved_video))

	print(f"Selected video: {scene.filename()}")
	print(f"Track count: {len(scene.tracklist())}")
	print("Looping annotated playback in one window. Press Ctrl+C to stop.")

	try:
		play_scene_loop(scene, fps=args.fps)
	except KeyboardInterrupt:
		print("Stopped.")


if __name__ == "__main__":
	main()

