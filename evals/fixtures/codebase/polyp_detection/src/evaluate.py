"""Frame-level evaluation (fictional evaluation fixture)."""

IOU_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.25  # replaced the old 0.5 policy


def precision_recall_f1(predictions, ground_truth):
    true_positives = sum(1 for p, g in zip(predictions, ground_truth) if p.iou(g) >= IOU_THRESHOLD and p.score >= CONFIDENCE_THRESHOLD)
    precision = true_positives / max(len(predictions), 1)
    recall = true_positives / max(len(ground_truth), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return precision, recall, f1
