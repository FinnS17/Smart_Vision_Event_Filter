from smart_vision_event_filter.events import MotionEvent


def frames_from_events(events: list[MotionEvent]) -> set[int]:
    frames: set[int] = set()
    for event in events:
        if event.end_frame is None:
            continue
        frames.update(range(event.start_frame, event.end_frame + 1))
    return frames


def evaluate_events(predicted_events: list[MotionEvent], ground_truth_events: list[MotionEvent]) -> dict[str, int | float]:
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