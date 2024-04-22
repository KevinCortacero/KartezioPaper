from kartezio.apps.instance_segmentation import create_instance_segmentation_model
from kartezio.dataset import read_dataset
from kartezio.endpoint import LocalMaxWatershed
from kartezio.preprocessing import SelectChannels
from kartezio.training import train_model

CHANNEL_DiO = 1
preprocessing = SelectChannels([CHANNEL_DiO])
DATASET = "./dataset"
OUTPUT = "./new_trained_models"


if __name__ == "__main__":
    generations = 100
    _lambda = 5
    frequency = 5
    model = create_instance_segmentation_model(
        generations,
        _lambda,
        inputs=1,
        outputs=1,
        endpoint=LocalMaxWatershed(threshold=1, markers_distance=5),
    )
    dataset = read_dataset(DATASET)
    elite, _ = train_model(
        model,
        dataset,
        OUTPUT,
        preprocessing=preprocessing,
        callback_frequency=frequency,
    )
