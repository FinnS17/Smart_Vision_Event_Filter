from pathlib import Path

import cv2

from smart_vision_event_filter.events import MotionEvent


def export_event_clips(video_path: str, events: list[MotionEvent], output_dir: str, padding_seconds: float = 0.0) -> list[str]:
    """Export one video clip for every completed motion event."""
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Could not open Video File: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    padding_frames = round(padding_seconds * fps) # eg: pad_s = 1.0, fps=30 -> 30 Frames
    codec = cv2.VideoWriter_fourcc(*"mp4v")

    clip_paths = []
    for index, event in enumerate(events, start=1):
        if event.end_frame is None:
            continue
        clip_start_frame = max(0, event.start_frame - padding_frames)
        clip_end_frame = min(total_frames - 1, event.end_frame + padding_frames)
        clip_path = output_directory / f"event_{index:03d}.mp4" #index 12 -> event_012.mp4
        capture.set(cv2.CAP_PROP_POS_FRAMES, clip_start_frame) # next read frame to be start frame of this event
        writer = cv2.VideoWriter(str(clip_path), codec, fps, (width, height))
        frame_count = clip_end_frame - clip_start_frame + 1
        for _ in range(frame_count):
            was_read, frame = capture.read()
            if not was_read:
                break
            writer.write(frame)
        writer.release()
        clip_paths.append(str(clip_path))
    capture.release()
    return clip_paths
