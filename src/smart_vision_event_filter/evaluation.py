import json

from smart_vision_event_filter.events import MotionEvent


def frames_from_events(events: list[MotionEvent]) -> set[int]:
    """Convert motion event ranges into a set of individual frame numbers."""
    frames: set[int] = set()
    for event in events:
        if event.end_frame is None:
            continue
        frames.update(range(event.start_frame, event.end_frame + 1))
    return frames


def load_predicted_events(path: str) -> list[MotionEvent]:
    """Load predictet motion events from a JSON report"""
    with open(path, encoding="utf-8") as input_file:
        event_data = json.load(input_file)
        events = []
        for item in event_data:
            event = MotionEvent(start_frame=item["start_frame"], end_frame=item["end_frame"])
            events.append(event)
        return events


def load_ground_truth_events(path: str, fps: float) -> list[MotionEvent]:
    """Load grund truth times and convert them to motion events."""
    with open(path, encoding="utf-8") as input_file:
        ground_truth_data = json.load(input_file)
        event_data = ground_truth_data["events"]
        events = []
        for item in event_data:
            start_frame = round(item["start_seconds"] * fps)
            end_frame = round(item["end_seconds"] * fps) - 1
            event = MotionEvent(start_frame=start_frame, end_frame=end_frame)
            events.append(event)
        return events

def evaluate_events(predicted_events: list[MotionEvent], ground_truth_events: list[MotionEvent]) -> dict[str, int | float]:
    """Compare predicted events with ground truth and calculate frame metrics."""
    predicted_frames = frames_from_events(predicted_events)
    ground_truth_frames = frames_from_events(ground_truth_events)
    true_positives = len(predicted_frames & ground_truth_frames)
    false_positives = len(predicted_frames - ground_truth_frames)
    false_negatives = len(ground_truth_frames - predicted_frames)

    predicted_positive_count = true_positives + false_positives
    if predicted_positive_count == 0:
        precision = 0.0
    else:
        precision = true_positives / predicted_positive_count

    actual_positive_count = true_positives + false_negatives
    if actual_positive_count == 0:
        recall = 0.0
    else: 
        recall = true_positives / actual_positive_count

    return {
        "true_positive_frames": true_positives,
        "false_positive_frames": false_positives,
        "false_negative_frames": false_negatives,
        "precision": precision,
        "recall": recall,
    }
