# Clip-Level Temporal Context for Polyp Detection (fictional paper, evaluation fixture)

Abstract. Frame-level detectors process each colonoscopy frame independently and rely on post-hoc smoothing. We propose a clip-level detector that aggregates features over 16 consecutive frames with a temporal attention module.

Results. On a benchmark with patient-level splits, the clip-level model improved recall on flat (sessile) polyps from 0.52 to 0.64 (+12 points) compared with a frame-level detector with majority-vote smoothing, while overall precision was unchanged. Flat polyps benefit most because they are only visible from certain angles, which temporal context captures.

Conclusion. Temporal feature aggregation, not post-hoc smoothing of frame outputs, drives the recall gain on hard polyps.
