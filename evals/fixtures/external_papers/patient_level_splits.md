# Data Leakage in Endoscopy Benchmarks (fictional paper, evaluation fixture)

Abstract. Many endoscopy detection studies split datasets randomly by frame. Consecutive frames of the same video are nearly identical, so frame-level splits leak test information into training.

Results. Re-evaluating five published detectors with patient-level splits reduced their F1 by 10 to 15 points on average. Models that looked best under frame-level splits were not the best under patient-level splits.

Recommendation. Split by patient (or at least by video), and report metrics at the lesion level in addition to the frame level. Frame-level splits make results incomparable with patient-level benchmarks.
