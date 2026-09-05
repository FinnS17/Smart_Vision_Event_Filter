import cv2
import numpy as np

from smart_vision_event_filter.clips import export_event_clips
from smart_vision_event_filter.events import MotionEvent


def test_export_event_clips_writes_expected_frames(tmp_path):
    source_path = tmp_path / "source.mp4"
    width = 64
    height = 48
    fps = 10
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    source_writer = cv2.VideoWriter(str(source_path), codec, fps, (width, height))
    assert source_writer.isOpened()
    for frame_number in range(10):
        frame = np.full((height, width, 3), frame_number * 20, dtype=np.uint8)
        source_writer.write(frame)
    source_writer.release()
    events = [MotionEvent(start_frame=2, end_frame=5)]
    output_dir = tmp_path / "clips"
    clip_paths = export_event_clips(str(source_path), events, str(output_dir))
    assert len(clip_paths) == 1
    assert output_dir.joinpath("event_001.mp4").exists()
    exported_capture = cv2.VideoCapture(clip_paths[0])
    assert exported_capture.isOpened()
    exported_frame_count = 0
    while True:
        was_read, _ = exported_capture.read()
        if not was_read:
            break
        exported_frame_count += 1
    exported_capture.release()
    assert exported_frame_count == 4
