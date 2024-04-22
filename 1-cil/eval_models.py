"""Evaluates the models on CIL dataset"""

import pandas as pd
from kartezio.dataset import read_dataset
from kartezio.easy import print_stats
from kartezio.fitness import FitnessAP
from kartezio.inference import ModelPool
from numena.io.drive import Directory
from train_model import preprocessing

print(
    "Warning! This evaluation must be done with Python 3.7 to reproduce experimental results!"
)
print(
    "Using Python 3.8+ might results in small differences due to a change of scipy stats module."
)
print("The training has been done with Python 3.7, evaluation might use the same.")
print("Exception: for n = 30/40/60/70/80, these are trained with python 3.8")


VERSION = "38"

DATASET = "./dataset"
MODELS = "./models"
if VERSION == "38":
    N_IMAGES = [30, 40, 60, 70, 80]
elif VERSION == "37":
    N_IMAGES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 50, 89]


SCORES_TEST = f"./results/results_test_py{VERSION}.csv"
SCORES_TRAINING = f"./results/results_training_py{VERSION}.csv"
SCORES_DIFFERENCE = f"./results/results_difference_py{VERSION}.csv"
ANNOTATIONS_TEST = f"./results/annotations_test_py{VERSION}.csv"
ANNOTATIONS_TRAINING = f"./results/annotations_training_py{VERSION}.csv"


scores_test_all = {}
scores_training_all = {}
scores_difference_all = {}
annotations_test_all = {}
annotations_training_all = {}

for n in N_IMAGES:
    pool_path = Directory(MODELS).next(str(n))
    pool = ModelPool(pool_path, FitnessAP(), regex="GRP_*/*/elite.json")
    scores_test = []
    scores_training = []
    scores_difference = []
    annotations_test = []
    annotations_training = []
    for model in pool.models:
        dataset = read_dataset(DATASET, indices=model.indices, counting=True)
        # Test set
        total_annotations = 0
        for annots in dataset.test_y:
            total_annotations += annots[1]
        annotations_test.append(total_annotations)
        _, fitness, _ = model.eval(dataset, subset="test", preprocessing=preprocessing)
        scores_test.append(1.0 - fitness)

        # Training set
        total_annotations = 0
        for annots in dataset.train_y:
            total_annotations += annots[1]
        annotations_training.append(total_annotations)
        _, fitness, _ = model.eval(dataset, subset="train", preprocessing=preprocessing)
        scores_training.append(1.0 - fitness)

        # Difference
        scores_difference.append(scores_training[-1] - scores_test[-1])

    print_stats(scores_training, "AP50", f"{n} images on training set")
    print_stats(scores_test, "AP50", f"{n} images on test set")
    scores_test_all[n] = scores_test
    scores_training_all[n] = scores_training
    scores_difference_all[n] = scores_difference
    annotations_test_all[n] = annotations_test
    annotations_training_all[n] = annotations_training


pd.DataFrame(scores_test_all).to_csv(SCORES_TEST, index=False)
pd.DataFrame(scores_training_all).to_csv(SCORES_TRAINING, index=False)
pd.DataFrame(scores_difference_all).to_csv(SCORES_DIFFERENCE, index=False)
pd.DataFrame(annotations_test_all).to_csv(ANNOTATIONS_TEST, index=False)
pd.DataFrame(annotations_training_all).to_csv(ANNOTATIONS_TRAINING, index=False)


print("done!")
