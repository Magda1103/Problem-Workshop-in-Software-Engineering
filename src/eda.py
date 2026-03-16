from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import cv2
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
		default=60.0,
		help="Playback FPS for visualization",
	)
	parser.add_argument(
		"--display-width",
		type=int,
		default=0,
		help="Resize display to this width for faster playback (<=0 disables resize)",
	)
	return parser.parse_args()


def _activity_label_text(category: str) -> str:
	labels = {
		"person_embraces_person": "person embracing person",
		"person_enters_car": "person entering car",
		"person_holds_hand": "person holding hand",
		"person_picks_up_object": "person picking up object",
		"person_reads_document": "person reading a book or document",
		"person_rides_bicycle": "person riding bicycle",
		"person_shakes_hand": "person shaking hands",
		"person_steals_object": "person stealing object",
		"person_talks_on_phone": "person talking on phone",
	}
	return labels.get(category, category.replace("_", " "))


def _build_track_activity_index(scene: vipy.video.Scene) -> dict[str, list[Any]]:
	track_to_activities: dict[str, list[Any]] = {}
	if not hasattr(scene, "activitylist"):
		return track_to_activities

	for activity in scene.activitylist():
		for track_id in activity.trackids():
			track_to_activities.setdefault(track_id, []).append(activity)

	return track_to_activities


def _draw_tracks(
	frame: Any,
	frame_index: int,
	tracks: list[Any],
	track_activities: dict[str, list[Any]],
) -> Any:
	for track in tracks:
		if frame_index < track.startframe() or frame_index > track.endframe():
			continue

		det = track[frame_index]
		if det is None:
			continue

		x1 = max(0, int(round(det.xmin())))
		y1 = max(0, int(round(det.ymin())))
		x2 = max(x1 + 1, int(round(det.xmax())))
		y2 = max(y1 + 1, int(round(det.ymax())))

		cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 220, 40), 2)

		label = (track.category() or "object").lower()
		track_id = track.id() if hasattr(track, "id") else None
		if track_id is not None:
			active = [
				_activity_label_text(activity.category())
				for activity in track_activities.get(track_id, [])
				if activity.during(frame_index)
			]
			if active:
				label = " / ".join(dict.fromkeys(active))

		cv2.putText(
			frame,
			label,
			(x1, max(18, y1 - 8)),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.5,
			(40, 220, 40),
			1,
			cv2.LINE_AA,
		)

	return frame


def _resize_frame(frame: Any, display_width: int | None) -> Any:
	if not display_width or display_width <= 0:
		return frame

	h, w = frame.shape[:2]
	if w <= display_width:
		return frame

	scale = display_width / float(w)
	new_h = max(1, int(round(h * scale)))
	return cv2.resize(frame, (display_width, new_h), interpolation=cv2.INTER_AREA)


def play_scene_loop(scene: vipy.video.Scene, fps: float, display_width: int) -> None:
	if fps <= 0:
		raise ValueError("FPS must be positive")

	source_fps = scene.framerate() if scene.framerate() is not None else fps
	target_fps = min(fps, source_fps)
	if target_fps <= 0:
		raise ValueError("Invalid target FPS")

	window_name = "EDA Playback"
	tracks = scene.tracklist()
	track_activities = _build_track_activity_index(scene)
	cap = cv2.VideoCapture(scene.filename())
	if not cap.isOpened():
		raise RuntimeError(f"Unable to open video for playback: {scene.filename()}")

	cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
	frame_index = 0
	last_tick = time.perf_counter()
	fps_smooth = target_fps

	try:
		while True:
			# If the user closed the window, exit before imshow can recreate it.
			try:
				if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
					return
			except cv2.error:
				return

			frame_start = time.perf_counter()
			ok, frame = cap.read()
			if not ok:
				cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
				frame_index = 0
				continue

			frame = _draw_tracks(frame, frame_index, tracks, track_activities)
			frame = _resize_frame(frame, display_width)

			now = time.perf_counter()
			inst_fps = 1.0 / max(now - last_tick, 1e-9)
			last_tick = now
			fps_smooth = (0.9 * fps_smooth) + (0.1 * inst_fps)
			cv2.setWindowTitle(window_name, f"EDA Playback - {fps_smooth:.1f} FPS")
			cv2.imshow(window_name, frame)

			elapsed = time.perf_counter() - frame_start
			wait_ms = max(1, int(round((1.0 / target_fps - elapsed) * 1000)))
			key = cv2.waitKey(wait_ms) & 0xFF
			if key in (27, ord("q")):
				return

			try:
				if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
					return
			except cv2.error:
				return

			frame_index += 1
	finally:
		cap.release()
		cv2.destroyAllWindows()


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
	print("Looping annotated playback in one window. Close window or press Q/Esc to stop.")

	try:
		play_scene_loop(scene, fps=args.fps, display_width=args.display_width)
	except KeyboardInterrupt:
		print("Stopped.")


if __name__ == "__main__":
	main()

