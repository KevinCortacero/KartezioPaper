import numpy as np
import pandas as pd
from DiO.train_model import preprocessing as preprocessing_DiO
from kartezio.dataset import read_dataset
from kartezio.easy import print_stats
from kartezio.fitness import FitnessAP
from kartezio.inference import ModelPool
from WGA.train_model import preprocessing as preprocessing_WGA

scores_all = {}
for channel, preprocessing in zip(
    ["WGA", "DiO"], [preprocessing_WGA, preprocessing_DiO]
):
    pool = ModelPool(f"./{channel}/models", FitnessAP(), regex="*/elite.json")
    dataset = read_dataset(f"./{channel}/dataset", counting=True)
    annotations_test = 0
    annotations_training = 0
    roi_pixel_areas = []
    for y_true in dataset.train_y:
        n_annotations = y_true[1]
        annotations_training += n_annotations
    for y_true in dataset.test_y:
        annotations = y_true[0]
        n_annotations = y_true[1]
        annotations_test += n_annotations
        for i in range(1, n_annotations + 1):
            roi_pixel_areas.append(np.count_nonzero(annotations[annotations == i]))
    print(f"\n{channel}:")
    print(f"Total annotations for {channel} training set: {annotations_training}")
    print(f"Total annotations for {channel} test set: {annotations_test}")
    print(f"Mean pixel area for {channel} test set: {np.mean(roi_pixel_areas)}")
    scores_test = []
    scores_training = []
    for model in pool.models:
        # Test set
        _, fitness, _ = model.eval(dataset, subset="test", preprocessing=preprocessing)
        scores_test.append(1.0 - fitness)
        # Training set
        _, fitness, _ = model.eval(dataset, subset="train", preprocessing=preprocessing)
        scores_training.append(1.0 - fitness)
    scores_all[f"training_{channel}"] = scores_training
    scores_all[f"test_{channel}"] = scores_test
    print_stats(scores_training, "AP50", f"{channel} training set")
    print_stats(scores_test, "AP50", f"{channel} test set")
pd.DataFrame(scores_all).to_csv("./results/results.csv", index=False)
