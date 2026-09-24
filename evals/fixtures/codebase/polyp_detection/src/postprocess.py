"""Temporal smoothing of per-frame detections (fictional evaluation fixture)."""

WINDOW = 5


def smooth(frame_detections: list[bool]) -> list[bool]:
    # Majority vote over a sliding window: the only temporal information the system uses
    smoothed = []
    for i in range(len(frame_detections)):
        window = frame_detections[max(0, i - WINDOW // 2): i + WINDOW // 2 + 1]
        smoothed.append(sum(window) > len(window) / 2)
    return smoothed
