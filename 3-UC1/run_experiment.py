import cv2
import numpy as np
from numena.image.basics import image_normalize
from numena.io.drive import Directory

from kartezio.dataset import read_dataset
from kartezio.fitness import FitnessIOU
from kartezio.inference import EnsembleModel, ModelPool
import matplotlib.pyplot as plt

THRESHOLD = 0.35000000000000003
image_threshold = 0
threshold_list = [0., 0.35000000000000003, 0.5, 0.6, 0.75]
fitness = FitnessIOU()
ensemble = ModelPool(f"./models", fitness, regex="*/elite.json").to_ensemble()
print(len(ensemble.models))
dataset = read_dataset(f"./dataset", counting=True)
threshold_range = np.linspace(0., 1., 101)  # np.arange(0., 1.01, 0.01)
print(threshold_range)
scores_train = []
scores_test = []
p_train = ensemble.predict(dataset.train_x)


for i in range(12):
    scores_per_image = []
    mask_list = [image_normalize(pi[0][i]["mask"]) for pi in p_train]
    heatmap = np.array(mask_list).mean(axis=0)

    for threshold in threshold_range:
        heatmap_cp = heatmap.copy()
        heatmap_cp[heatmap_cp < threshold] = 0
        y_pred = {"mask": (heatmap_cp * 255.0).astype(np.uint8)}
        s = fitness.compute_one([dataset.train_y[i]], [y_pred])
        scores_per_image.append(1. - s)

        if i == image_threshold and threshold in threshold_list:
            heatmap_cp_thresh = (heatmap_cp * 255.0).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap_cp_thresh, cv2.COLORMAP_JET)
            heatmap_color[heatmap_cp_thresh == 0] = dataset.train_v[i][heatmap_cp_thresh == 0]
            overlayed_heatmap = cv2.addWeighted(
                heatmap_color, 0.5, dataset.train_v[i].copy(), 1 - 0.5, 0, dataset.train_v[i].copy()
            )
            cv2.imwrite(f"heatmap_threshold_{int(threshold*100)}.png", overlayed_heatmap)

        if threshold == THRESHOLD:
            heatmap_cp = (heatmap_cp * 255.0).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap_cp, cv2.COLORMAP_JET)
            heatmap_color[heatmap_cp == 0] = dataset.train_v[i][heatmap_cp == 0]

            overlayed_heatmap = cv2.addWeighted(
                heatmap_color, 0.5, dataset.train_v[i].copy(), 1 - 0.5, 0, dataset.train_v[i].copy()
            )
            cv2.imwrite(f"heatmap_train_{i}.png", overlayed_heatmap)
    scores_train.append(scores_per_image)

p_test = ensemble.predict(dataset.test_x)
best_image_scores = []


for i in range(12):
    scores_per_image = []
    mask_list = [image_normalize(pi[0][i]["mask"]) for pi in p_test]
    heatmap = np.array(mask_list).mean(axis=0)
    for threshold in threshold_range:
        heatmap_cp = heatmap.copy()
        heatmap_cp[heatmap_cp < threshold] = 0
        y_pred = {"mask": (heatmap_cp * 255.0).astype(np.uint8)}
        s = fitness.compute_one([dataset.test_y[i]], [y_pred])
        scores_per_image.append(1. - s)

        if threshold == THRESHOLD:
            best_image_scores.append(1. - s)
            heatmap_cp = (heatmap_cp * 255.0).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap_cp, cv2.COLORMAP_JET)
            heatmap_color[heatmap_cp == 0] = dataset.test_v[i][heatmap_cp == 0]

            overlayed_heatmap = cv2.addWeighted(
                heatmap_color, 0.5, dataset.test_v[i], 1 - 0.5, 0, dataset.test_v[i]
            )
            print(f"creating test image {i}, threshold = {threshold}")
            cv2.imwrite(f"heatmap_test_{i}.png", overlayed_heatmap)
    scores_test.append(scores_per_image)
best_image_scores = np.array(best_image_scores)
print(best_image_scores)
print(best_image_scores.min(), best_image_scores.max(), np.std(best_image_scores), np.mean(best_image_scores))
scores_train = np.array(scores_train).mean(axis=0)
scores_test = np.array(scores_test).mean(axis=0)


GREEN = "#3E8948"
GREEN_RGB = (0, 176, 80)
BLUE = "#124E89"
PURPLE_RGB = (32, 56, 100)
MAGENTA = "#B55088"
GRAY = "#5A6988"
print(scores_train[35])
with plt.style.context(["nature"]):
    fig, ax = plt.subplots()
    ax.set_xlim([-0.005, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.plot(threshold_range, scores_train, label="Training", color=GREEN, zorder=2)
    ax.plot(threshold_range, scores_test, label="Test", color=MAGENTA, zorder=1)
    ax.scatter(threshold_list, scores_train[[0, 35, 50, 60, 75]], marker=".", color=GREEN, zorder=3, s=20)
    ax.axline((0.35, scores_train[35]), (0.35, 0.0), linestyle="dotted", color=GREEN)
    ax.legend(frameon=False)
    fig.savefig("scores.png", dpi=300)

# plt.title("Fitness of Model Ensemble vs threshold")
# plt.xlabel("Threshold")
# plt.ylabel("IoU Fitness")
