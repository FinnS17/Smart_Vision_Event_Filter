from pathlib import Path

import cv2

from smart_vision_event_filter.events import MotionEvent


def export_event_clips(video_path: str, events: list[MotionEvent], output_dir: str) -> list[str]:
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Could not open Video File: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = cv2.VideoWriter_fourcc(*"mp4v")

    clip_paths = []
    for index, event in enumerate(events, start=1):
        if event.end_frame is None:
            continue
        clip_path = output_directory / f"event_{index:03d}.mp4" #index 12 -> event_012.mp4
        capture.set(cv2.CAP_PROP_POS_FRAMES, event.start_frame) # next read frame to be start frame of this event
        writer = cv2.VideoWriter(str(clip_path), codec, fps, (width, height))
        frame_count = event.end_frame - event.start_frame + 1
        for _ in range(frame_count):
            was_read, frame = capture.read()
            if not was_read:
                break
            writer.write(frame)
        writer.release()
        clip_paths.append(str(clip_path))
    capture.release()
    return clip_paths
