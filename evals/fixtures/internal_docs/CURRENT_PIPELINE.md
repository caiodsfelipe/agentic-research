---
title: "Polyp detector - current production pipeline"
status: current
---

# Current production pipeline (fictional)

The production polyp detector is detector v3: a YOLO-style frame detector trained with focal loss at 640x640 on ColonSet-A (12,000 annotated frames). Augmentation is limited to horizontal and vertical flips.

Post-processing applies a majority vote over a sliding window of 5 frames. The confidence threshold is 0.25 and detections are matched to ground truth at IoU 0.5.

Reported test metrics (frame level): precision 0.64, recall 0.71, F1 0.67.

Known issue: flat (sessile) polyps are frequently missed; most false negatives in the error review are flat polyps. Most false positives are specular reflections on the mucosa.
