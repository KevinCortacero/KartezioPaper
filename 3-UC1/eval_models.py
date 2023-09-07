import numpy as np
import pandas as pd

from kartezio.easy import print_stats
from kartezio.dataset import read_dataset
from kartezio.fitness import FitnessIOU
from kartezio.inference import ModelPool

scores_all = {}
pool = ModelPool(f"./models", FitnessIOU(), regex="*/elite.json").to_ensemble()
dataset = read_dataset(f"./dataset", counting=True)
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
print(f"Total annotations for training set: {annotations_training}")
print(f"Total annotations for test set: {annotations_test}")
print(f"Mean pixel area for test set: {np.mean(roi_pixel_areas)}")


scores_test = []
scores_training = []
for i, model in enumerate(pool.models):
    # Test set
    _, fitness, _ = model.eval(dataset, subset="test")
    scores_test.append(1.0 - fitness)

    # Training set
    _, fitness, _ = model.eval(dataset, subset="train")
    scores_training.append(1.0 - fitness)


scores_all[f"training"] = scores_training
scores_all[f"test"] = scores_test
print_stats(scores_training, "AP50", "training set")
print_stats(scores_test, "AP50", "test set")

pd.DataFrame(scores_all).to_csv("./results/results.csv", index=False)