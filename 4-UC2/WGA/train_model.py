from kartezio.apps.instance_segmentation import create_instance_segmentation_model
from kartezio.endpoint import LocalMaxWatershed
from kartezio.dataset import read_dataset
from kartezio.training import train_model
from kartezio.preprocessing import SelectChannels


CHANNEL_WGA = 0
preprocessing = SelectChannels([CHANNEL_WGA])
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
    elite, _ = train_model(model, dataset, OUTPUT, preprocessing=preprocessing, callback_frequency=frequency)
