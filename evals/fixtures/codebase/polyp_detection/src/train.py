"""Trains the frame-level polyp detector (fictional evaluation fixture)."""

IMAGE_SIZE = 640
EPOCHS = 80
LOSS = "focal"  # focal loss to handle the background/polyp imbalance
AUGMENTATIONS = ["horizontal_flip", "vertical_flip"]  # no color, blur or specular-reflection augmentation


def train(dataset, model):
    for epoch in range(EPOCHS):
        for frames, boxes in dataset.train_batches(image_size=IMAGE_SIZE, augmentations=AUGMENTATIONS):
            loss = model.loss(frames, boxes, kind=LOSS)
            loss.backward()
    return model
