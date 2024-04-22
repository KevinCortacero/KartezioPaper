import cv2
import numpy as np
from kartezio.apps.instance_segmentation import create_instance_segmentation_model
from kartezio.dataset import read_dataset
from kartezio.preprocessing import SelectChannels
from kartezio.training import train_model

DATASET = "./dataset"
MODELS = "./models"
CHANNELS = [1, 2]
preprocessing = SelectChannels(CHANNELS)


if __name__ == "__main__":
    generations = 100
    _lambda = 5
    frequency = 5
    indices = None  # [12, 26, 76, 59, 58, 37, 11, 79, 34, 35, 36, 81, 67, 17, 13]
    model = create_instance_segmentation_model(
        generations,
        _lambda,
        inputs=2,
    )
    dataset = read_dataset(DATASET, indices=indices)
    elite, _ = train_model(
        model,
        dataset,
        MODELS,
        preprocessing=preprocessing,
        callback_frequency=frequency,
    )
