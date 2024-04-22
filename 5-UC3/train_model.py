from kartezio.apps.instance_segmentation import create_instance_segmentation_model
from kartezio.dataset import read_dataset
from kartezio.endpoint import LocalMaxWatershed
from kartezio.fitness import FitnessAP
from kartezio.model.registry import registry
from kartezio.preprocessing import Format3D
from kartezio.training import train_model

DATASET = "./dataset"
OUTPUT = "./new_trained_models"
preprocessing = Format3D()


if __name__ == "__main__":
    generations = 100
    _lambda = 5
    frequency = 5
    model = create_instance_segmentation_model(
        generations,
        _lambda,
        inputs=1,
        outputs=1,
        series_mode=True,
        series_stacker=registry.stackers.instantiate("MEAN"),
        endpoint=LocalMaxWatershed(),
        fitness=FitnessAP(thresholds=0.7),
    )
    dataset = read_dataset(DATASET)
    elite, _ = train_model(
        model,
        dataset,
        OUTPUT,
        preprocessing=preprocessing,
        callback_frequency=frequency,
    )
