import cv2
import numpy as np
from kartezio.apps.instance_segmentation import create_instance_segmentation_model
from kartezio.dataset import read_dataset
from kartezio.endpoint import LocalMaxWatershed, EndpointThreshold
from kartezio.fitness import FitnessIOU
from kartezio.image.bundle import BUNDLE_OPENCV
from kartezio.image.nodes import NodeImageProcessing
from kartezio.model.evolution import KartezioFitness
from kartezio.model.registry import registry
from kartezio.training import train_model
from kartezio.preprocessing import SelectChannels


@registry.fitness.add("APIOU")
class FitnessAPIOU(KartezioFitness):
    def __init__(self, thresholds=0.5):
        super().__init__(
            name=f"Average Precision ({thresholds}) + IoU",
            symbol="APIOU",
            arity=1,
            default_metric=registry.metrics.instantiate("CAP", thresholds=thresholds),
        )
        self.add_metric(registry.metrics.instantiate("IOU"))
    """
    def compute_ones(self, y_true, y_pred):
        for i in range(len(y_pred)):
            y_pred[i]["mask"] = cv2.resize(y_pred[i]["mask"], (0, 0), fx=scale_up, fy=scale_up)
            labels = y_pred[i]["labels"]
            labels[labels > 255] = 0
            resized = cv2.resize(labels.astype(np.uint8), (0, 0), fx=scale_up, fy=scale_up,  interpolation=cv2.INTER_NEAREST)
            y_pred[i]["labels"] = resized
        return super().compute_one(y_true, y_pred)
    """


"""
class Resize(KartezioPreprocessing):
    def __init__(self, scale_factor):
        super().__init__("Resize", "SIZE")
        self.scale_factor = scale_factor

    def call(self, x, args=None):
        new_x = []
        for i in range(len(x)):
            one_item = [cv2.resize(x[i][j], (0, 0), fx=self.scale_factor, fy=self.scale_factor, interpolation=cv2.INTER_AREA) for j in range(len(x[i]))]
            # one_item = [x[i][channel] for channel in self.channels]
            new_x.append(one_item)
        return new_x

    def _to_json_kwargs(self) -> dict:
        pass
"""


KERNEL_EMBOSS = np.array(([-2, -1, 0],
                          [-1, 1, 1],
                          [0, 1, 2]), dtype="int")

KERNEL_KIRSCH_N = np.array(([5, 5, 5],
                          [-3, 0, -3],
                          [-3, -3, -3]), dtype="int")

KERNEL_KIRSCH_NE = np.array(([-3, 5, 5],
                          [-3, 0, 5],
                          [-3, -3, -3]), dtype="int")

KERNEL_KIRSCH_E = np.array(([-3, -3, 5],
                          [-3, 0, 5],
                          [-3, -3, 5]), dtype="int")

KERNEL_KIRSCH_SE = np.array(([-3, -3, -3],
                          [-3, 0, 5],
                          [-3, 5, 5]), dtype="int")

KERNEL_KIRSCH_S = np.array(([-3, -3, -3],
                          [-3, 0, -3],
                          [5, 5, 5]), dtype="int")

KERNEL_KIRSCH_SW = np.array(([-3, -3, -3],
                          [5, 0, -3],
                          [5, 5, -3]), dtype="int")

KERNEL_KIRSCH_W = np.array(([5, -3, -3],
                          [5, 0, -3],
                          [5, -3, -3]), dtype="int")

KERNEL_KIRSCH_NW = np.array(([5, 5, -3],
                          [5, 0, -3],
                          [-3, -3, -3]), dtype="int")

KERNEL_KIRSCH_COMPASS = [
    KERNEL_KIRSCH_N,
    KERNEL_KIRSCH_NE,
    KERNEL_KIRSCH_E,
    KERNEL_KIRSCH_SE,
    KERNEL_KIRSCH_S,
    KERNEL_KIRSCH_SW,
    KERNEL_KIRSCH_W,
    KERNEL_KIRSCH_NW
]


@registry.nodes.add("kirsch")
class Kirsch(NodeImageProcessing):
    def __init__(self):
        super().__init__("kirsch", "KIR", 1, 0, sources="OpenCV")

    def call(self, x, args=None):
        compass_gradients = [
            cv2.filter2D(x[0], ddepth=cv2.CV_32F, kernel=kernel/5)
            for kernel in KERNEL_KIRSCH_COMPASS
        ]
        res = np.max(compass_gradients, axis=0)
        res[res > 255] = 255
        return res.astype(np.uint8)


@registry.nodes.add("embossing")
class Embossing(NodeImageProcessing):
    def __init__(self):
        super().__init__("embossing", "EMBO", 1, 0, sources="OpenCV")

    def call(self, x, args=None):
        res = cv2.filter2D(x[0], ddepth=cv2.CV_32F, kernel=KERNEL_EMBOSS)
        res[res > 255] = 255
        return res.astype(np.uint8)


@registry.nodes.add("norm")
class Normalization(NodeImageProcessing):
    def __init__(self):
        super().__init__("norm", "NORM", 1, 0, sources="OpenCV")

    def call(self, x, args=None):
        return cv2.normalize(x[0],  None, 0, 255, cv2.NORM_MINMAX)


@registry.nodes.add("otsu")
class Threshold(NodeImageProcessing):
    def __init__(self):
        super().__init__("otsu", "OTSU", 1, 0, sources="OpenCV")

    def call(self, x, args=None):
        return cv2.threshold(x[0], 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)[1]


@registry.nodes.add("denoizing")
class Denoizing(NodeImageProcessing):
    def __init__(self):
        super().__init__("denoizing", "DNOZ", 1, 1, sources="OpenCV")

    def call(self, x, args=None):
        return cv2.fastNlMeansDenoising(x[0], None, h=int(args[0]))


@registry.nodes.add("pyr")
class Pyr(NodeImageProcessing):
    def __init__(self):
        super().__init__("pyr", "PYR", 1, 1, sources="OpenCV")

    def call(self, x, args=None):
        if args[0] < 128:
            return self.pyrdown(x[0])
        return self.pyrup(x[0])

    def pyrup(self, x):
        h, w = x.shape
        scaled_twice = cv2.pyrUp(x[0])
        return cv2.resize(scaled_twice, (w, h))

    def pyrdown(self, x):
        h, w = x.shape
        scaled_half = cv2.pyrDown(x[0])
        return cv2.resize(scaled_half, (w, h))


bundle = BUNDLE_OPENCV
bundle.add_node("otsu")
bundle.add_node("denoizing")
bundle.add_node("norm")
bundle.add_node("pyr")
bundle.add_node("embossing")
bundle.add_node("kirsch")

DATASET = "./dataset"
MODELS = "./models"
CHANNELS = [1, 2]
preprocessing = SelectChannels(CHANNELS)


if __name__ == "__main__":
    generations = 2000
    _lambda = 5
    frequency = 5
    indices = None  # [12, 26, 76, 59, 58, 37, 11, 79, 34, 35, 36, 81, 67, 17, 13]
    model = create_instance_segmentation_model(
        generations, _lambda, inputs=2, outputs=1,
        # fitness=FitnessAPIOU(thresholds=0.5),
        fitness=FitnessIOU(),
        bundle=bundle, nodes=30,
        # endpoint=LocalMaxWatershed()
        endpoint=EndpointThreshold(128)
    )
    dataset = read_dataset(DATASET, indices=indices)
    elite, _ = train_model(model, dataset, MODELS, preprocessing=preprocessing, callback_frequency=frequency)
