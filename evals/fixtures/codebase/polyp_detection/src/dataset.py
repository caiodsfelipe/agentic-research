"""ColonSet-A loading and splitting (fictional evaluation fixture)."""

import random

SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}


def split_frames(frames: list, seed: int = 0) -> dict:
    # Random split by frame: frames of the same patient can end up in both train and test
    random.Random(seed).shuffle(frames)
    n_train = int(len(frames) * SPLITS["train"])
    n_val = int(len(frames) * SPLITS["val"])
    return {
        "train": frames[:n_train],
        "val": frames[n_train:n_train + n_val],
        "test": frames[n_train + n_val:],
    }
