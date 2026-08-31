import json

from smart_vision_event_filter.events import MotionEvent
from smart_vision_event_filter.reporting import save_events_to_json

def test_save_events_to_json(tmp_path):
    output_path = tmp_path / "events.json"
    events = [
        MotionEvent(start_frame=10, end_frame=19)
    ]
    save_events_to_json(events, output_path, fps=10.0)

    with open(output_path, "r", encoding="utf-8") as input_file:
        saved_data = json.load(input_file)
        assert saved_data == [
            {
                "start_frame": 10,
                "end_frame": 19,
                "start_seconds": 1.0,
                "end_seconds": 2.0,
                "duration_seconds": 1.0
            }
        ]