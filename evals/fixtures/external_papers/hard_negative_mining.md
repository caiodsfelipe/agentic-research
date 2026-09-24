# Hard Negative Mining Against Specular Reflections (fictional paper, evaluation fixture)

Abstract. Specular reflections on the mucosa are the most common source of false positives in polyp detection.

Method. We mine frames containing reflections that the detector wrongly flags, add them as hard negatives, and add synthetic specular-reflection augmentation during training.

Results. Precision improved from 0.66 to 0.75 (+9 points) with no loss of recall. Color and blur augmentation alone gave smaller gains (+2 points).
