import json

from smart_vision_event_filter.events import MotionEvent

def save_events_to_json(events: list[MotionEvent], output_path: str, fps: float) -> None:
    """Save completed motion events with frame and time data as JSON."""
    event_out = []
    for event in events:
        if event.end_frame is None:
            continue
        start_second = event.start_frame / fps
        end_second = (event.end_frame + 1) / fps
        durations_seconds = end_second - start_second
        event_out.append(
            {"start_frame": event.start_frame,
             "end_frame": event.end_frame,
             "start_seconds": round(start_second, 2),
             "end_seconds": round(end_second, 2),
             "duration_seconds": round(durations_seconds, 2)}
        )
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(event_out, output_file, indent=2)
