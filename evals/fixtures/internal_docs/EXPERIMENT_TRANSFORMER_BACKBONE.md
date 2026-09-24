---
title: "Experiment: transformer backbone"
status: historical
---

# Experiment: transformer backbone (fictional, abandoned)

We replaced the convolutional backbone with a Swin transformer backbone, keeping the frame-level detection setup.

Result: training loss kept decreasing but validation recall dropped from 0.71 to 0.66; the model overfit ColonSet-A. With only 12,000 frames the dataset was judged too small for this backbone. The experiment was abandoned.
