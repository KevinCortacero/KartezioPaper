import numpy as np
import pandas as pd
import cv2
from kartezio.dataset import read_dataset
from kartezio.fitness import FitnessAP
from kartezio.inference import ModelPool
from kartezio.preprocessing import TransformToHSV, TransformToHED
from kartezio.easy import print_stats, get_model_size, node_histogram, python_class
from numena.image.contour import contours_find, contours_draw
from numena.image.basics import image_normalize
from numena.image.color import rgb2bgr
from melanoma_test.train_model import COLORS_SCALES
from kartezio.export import KartezioInsight

MODES = ["MCW", "LMW", "ELLIPSE", "HCT", "LABELS"]
dataset = read_dataset(f"./dataset", counting=True, preview=True)

for mode in MODES:
    scores_all = {}
    for color_scale in COLORS_SCALES:
        preprocessing = None
        if color_scale == "HSV":
            preprocessing = TransformToHSV()
        elif color_scale == "HED":
            preprocessing = TransformToHED()
        pool = ModelPool(f"./models/{mode}/{color_scale}", FitnessAP(), regex="*/elite.json")
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

            p_test, fitness, t = model.eval(dataset, subset="test", preprocessing=preprocessing)

            scores_test.append(1.0 - fitness)
            for j, p_test_i in enumerate(p_test):
                visual = dataset.test_v[j]
                labels = p_test_i["labels"]
                n_labels = labels.max()
                for k in range(n_labels):
                    labels_unique = labels.copy()
                    labels_unique[labels_unique != k+ 1] = 0
                    visual = contours_draw(visual, contours_find(labels_unique.astype(np.uint8)), color=[72, 137, 62], thickness=2)
                # visual = cv2.applyColorMap(p_test_i["labels"].astype(np.uint8)*5, cv2.COLORMAP_VIRIDIS)
                # cv2.imwrite(f"{mode}_{color_scale}_{i}_test_{j}.png", rgb2bgr(visual))
                if "mask_raw" in p_test_i:
                    cm_mask = cv2.applyColorMap((image_normalize(p_test_i["mask_raw"]) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
                else:
                    cm_mask = cv2.applyColorMap((image_normalize(p_test_i["mask"]) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
                # cv2.imwrite(f"{mode}_{color_scale}_{i}_test_{j}_mask.png", cm_mask)
                '''
                                if "markers" in p_test_i:
                    cm_markers = cv2.applyColorMap((image_normalize(p_test_i["markers"]) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
                    if i == 5 and mode == "MCW" and color_scale == "HSV":
                        cv2.imwrite(f"{mode}_{color_scale}_{i}_test_{j}_markers.png", cm_markers)
                        if j == 1:
                            insight = KartezioInsight(model._model.parser, preprocessing=preprocessing)
                            insight.create_node_images(model._model.genome, dataset.test_x[j], prefix=f"{mode}_{color_scale}_{i}_test_{j}")
                            exit()
                '''


                # cm_labels = cv2.applyColorMap(p_test_i["labels"].astype(np.uint8) * 8, cv2.COLORMAP_VIRIDIS)
                # cv2.imwrite(f"{mode}_{color_scale}_{i}_test_{j}_labels.png", cm_labels)

            # Training set
            p_train, fitness, _ = model.eval(dataset, subset="train", preprocessing=preprocessing)
            scores_training.append(1.0 - fitness)


        scores_all[f"training_{color_scale}"] = scores_training
        scores_all[f"test_{color_scale}"] = scores_test
        print_stats(scores_training, "AP50", f"{mode} {color_scale} training set")
        print_stats(scores_test, "AP50", f"{mode} {color_scale} test set")

    pd.DataFrame(scores_all).to_csv(f"./results/{mode}_results.csv", index=False)