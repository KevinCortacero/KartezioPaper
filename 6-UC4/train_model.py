from kartezio.apps.instance_segmentation import create_instance_segmentation_model
from kartezio.dataset import read_dataset
from kartezio.endpoint import EndpointWatershed
from kartezio.preprocessing import Format3D
from kartezio.stacker import StackerMean
from kartezio.training import train_model

DATASET = "./dataset"
OUTPUT = "./new_trained_models"
CHANNELS = [1, 2]
preprocessing = Format3D(channels=CHANNELS)


if __name__ == "__main__":
    generations = 100
    _lambda = 5
    frequency = 5
    model = create_instance_segmentation_model(
        generations,
        _lambda,
        inputs=2,
        outputs=2,
        series_mode=True,
        series_stacker=StackerMean(arity=2),
        endpoint=EndpointWatershed(),
    )

    dataset = read_dataset(DATASET)
    elite, _ = train_model(
        model,
        dataset,
        OUTPUT,
        preprocessing=preprocessing,
        callback_frequency=frequency,
    )
