import json

from smart_vision_event_filter.events import MotionEvent

def save_events_to_json(events: list[MotionEvent], output_path: str, fps: float) -> None:
    """Save each finished event with its times and object names as JSON."""
    event_out = []
    for event in events:
        if event.end_frame is None:
            continue
        # Example: at 10 FPS, frame 20 begins at 2.0 seconds.
        start_second = event.start_frame / fps
        # Add 1 because end_frame is included in the event.
        end_second = (event.end_frame + 1) / fps
        durations_seconds = end_second - start_second
        event_out.append(
            {"start_frame": event.start_frame,
             "end_frame": event.end_frame,
             "start_seconds": round(start_second, 2),
             "end_seconds": round(end_second, 2),
             "duration_seconds": round(durations_seconds, 2),
             "detected_objects": sorted(event.detected_objects)}
        )
    # event_out is now a list of JSON-ready dictionaries.
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(event_out, output_file, indent=2)
