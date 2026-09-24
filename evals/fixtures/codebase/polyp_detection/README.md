# Polyp Detection (fictional evaluation fixture)

Frame-level polyp detector for colonoscopy videos. This project is fictional and exists only to evaluate the research agents.

- `src/train.py`: trains a YOLO-style detector with focal loss at 640x640, using horizontal/vertical flips as the only augmentation.
- `src/postprocess.py`: temporal smoothing, a majority vote over a sliding window of 5 frames.
- `src/evaluate.py`: frame-level precision, recall and F1 at IoU 0.5, with a fixed confidence threshold of 0.25.
- `src/dataset.py`: loads ColonSet-A and splits it randomly by frame (80/10/10).
