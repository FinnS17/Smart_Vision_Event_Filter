from smart_vision_event_filter.evaluation import evaluate_events
from smart_vision_event_filter.events import MotionEvent


def test_evaluate_events_calculates_frame_metrics():
    """Check frame-level precision and recall for overlapping event ranges."""
    ground_truth_events = [MotionEvent(start_frame=10, end_frame=19),]
    predicted_events = [MotionEvent(start_frame=15, end_frame=24),]
    metrics = evaluate_events(predicted_events, ground_truth_events)
    assert metrics["true_positive_frames"] == 5
    assert metrics["false_positive_frames"] == 5
    assert metrics["false_negative_frames"] == 5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
