import cv2
import numpy as np
from kartezio.dataset import read_dataset
from kartezio.fitness import FitnessIOU
from kartezio.inference import ModelPool
from kartezio.preprocessing import TransformToHSV
from numena.image.basics import image_normalize

preprocessing = TransformToHSV().call

THRESHOLD = 0.35
ensemble = ModelPool(f"./models", FitnessIOU(), regex="*/elite.json").to_ensemble()
dataset = read_dataset(f"./dataset", counting=True, filename="dataset_upsample.csv")
p_test = ensemble.predict(dataset.test_x, reformat_x=preprocessing)


for i in range(15):
    mask_list = [image_normalize(pi[0][i]["mask"]) for pi in p_test]
    heatmap = np.array(mask_list).mean(axis=0)
    heatmap[heatmap < THRESHOLD] = 0
    heatmap = (heatmap * 255.0).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color[heatmap == 0] = dataset.test_v[i][heatmap == 0]

    overlayed_heatmap = cv2.addWeighted(
        heatmap_color, 0.5, dataset.test_v[i], 1 - 0.5, 0, dataset.test_v[i]
    )

    cv2.imwrite(f"heatmap_upscaled_{i}.png", overlayed_heatmap)
    cv2.imwrite(f"raw_{i}.png", dataset.test_v[i])
