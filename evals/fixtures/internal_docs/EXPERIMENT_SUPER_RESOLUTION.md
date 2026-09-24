---
title: "Experiment: super-resolution preprocessing"
status: historical
---

# Experiment: super-resolution preprocessing (fictional, abandoned)

We added a 2x super-resolution network before the detector to recover detail on small and flat polyps.

Result: F1 changed by +0.003, within run-to-run noise, while inference time doubled. Recall on flat polyps did not improve. The experiment was abandoned and super-resolution is not part of the production pipeline.
