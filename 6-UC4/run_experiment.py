import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kartezio.dataset import read_dataset
from kartezio.endpoint import EndpointWatershed
from kartezio.fitness import FitnessAP
from kartezio.inference import KartezioModel
from kartezio.model.components import KartezioStacker
from kartezio.plot import plot_watershed
from numena.features.profiling import CellStainingProfile, ProfilingInfo
from numena.figure import Figure
from numena.geometry import Cell2D, Synapse
from numena.image.color import rgb2bgr
from numena.image.drawing import draw_overlay
from numena.image.morphology import get_kernel, morph_dilate
from numena.image.threshold import threshold_tozero
from numena.io.drive import Directory
from numena.io.image import imread_color, imread_tiff
from sklearn import mixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from train_model import preprocessing

infos = ProfilingInfo("./dataset/experiment/INFOS.json")
profiling = CellStainingProfile(infos)


def _read_image(filepath):
    img = imread_tiff(filepath)
    channels = [
        img[:, 0],
        img[:, 1],
        img[:, 2],
    ]
    return channels


def read_batch(directory: Directory):
    batch = {}
    for filepath in directory.ls("*.tif", ordered=True):
        image = _read_image(filepath)
        batch[filepath.name] = image
    return batch


def get_cell_feature(cell, channels, lst):
    threshold = 8
    inside_cell = cell.mask > 0
    labels = []
    profile = []
    channels_true = {}
    channels_false = {}
    for i, channel_name in enumerate(["Perf", "Tub", "DAPI"]):
        channel_data = channels[i]
        channel_true = channel_data >= threshold
        channel_false = ~channel_true
        channels_true[channel_name] = channel_true & inside_cell
        channels_false[channel_name] = channel_false & inside_cell

    for combination in lst:
        combination_names = []
        combination_values = []
        for j, channel_name in enumerate(["Perf", "Tub", "DAPI"]):
            if combination[j] == 1:
                combination_names.append(channel_name)
                combination_values.append(channels_true[channel_name])
            else:
                combination_values.append(channels_false[channel_name])
        s = np.count_nonzero(np.logical_and.reduce(combination_values))
        label_name = f"-".join(combination_names)
        labels.append(label_name)
        profile.append(s)

    for i, channel_name in enumerate(["Perf", "Tub", "DAPI"]):
        labels.append(f"{channel_name}+")
        profile.append(np.count_nonzero(channels_true[channel_name]))
        labels.append(f"{channel_name}-SUM")
        profile.append(np.sum(channels[i], where=channels_true[channel_name]))
        labels.append(f"{channel_name}-MEAN")
        mean_value = np.mean(channels[i], where=channels_true[channel_name])
        if np.isnan(mean_value):
            mean_value = 0
        mean_value = int(round(mean_value))
        profile.append(mean_value)

    return profile, labels


def image_dilate(image, half_kernel_size=1):
    kernel_size = half_kernel_size * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.dilate(image, kernel)


def classify_data(X, n_class, scale=False):
    if scale:
        scaler = StandardScaler().fit(X)
        X = scaler.transform(X)
    X_emb = PCA(n_components="mle", svd_solver="full").fit_transform(X)
    model_classifier = mixture.GaussianMixture(
        n_components=n_class, covariance_type="tied"
    ).fit(X_emb)
    p = model_classifier.predict(X_emb)
    return X_emb, p, model_classifier


def labels_to_cells(labels_image, image_name, condition):
    kernel = get_kernel("circle", 10)
    label_numbers = np.unique(labels_image)
    cells = []
    for n in label_numbers:
        if n == 0:
            continue
        mask = (labels_image == n).astype("uint8")
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        new_cell = Cell2D.from_mask(
            mask, f"cell_{n}", custom_data={"image": image_name, "condition": condition}
        )
        cells.append(new_cell)
    return cells


class MeanStackerSaver(KartezioStacker):
    def _to_json_kwargs(self) -> dict:
        pass

    def stack(self, Y):
        return np.mean(np.array(Y), axis=0).astype(np.uint8)

    def post_stack(self, x, index):
        yi = x.copy()
        output = threshold_tozero(yi, self.threshold)
        return output

    def __init__(
        self, name="mean_stacker_saver", abbv="MEAN-saver", arity=1, threshold=4
    ):
        super().__init__(name, abbv, arity)
        self.threshold = threshold

    def call(self, x, args=None):
        y = []
        for i in range(self.arity):
            Y = [xi[i] for xi in x]
            for j, yj in enumerate(Y):
                heatmap_color = cv2.applyColorMap(yj, cv2.COLORMAP_VIRIDIS)
                cv2.imwrite(f"output_{i}_z{j}.png", heatmap_color)
            stacked_output = self.post_stack(self.stack(Y), i)
            y.append(stacked_output)
            heatmap_color = cv2.applyColorMap(stacked_output, cv2.COLORMAP_VIRIDIS)
            cv2.imwrite(f"output_{i}_mean.png", heatmap_color)
        return y


if __name__ == "__main__":
    dataset = read_dataset("./dataset")
    # Load Kartezio Model
    model = KartezioModel(
        "./models/55429-b827ed9a-b3ea-4cb7-b451-07ea04d48179/elite.json", FitnessAP()
    )
    model._model.parser.endpoint = EndpointWatershed(
        use_dt=False, markers_distance=21, markers_area=(16, 4096)
    )

    model_introspection = KartezioModel(
        "./models/55429-b827ed9a-b3ea-4cb7-b451-07ea04d48179/elite.json", FitnessAP()
    )
    model_introspection._model.parser.stacker = MeanStackerSaver(arity=2)
    image_idx = 0
    """ SAVE sub-predictions """
    print(dataset.test_x[image_idx][0].shape, len(dataset.test_x[image_idx]))
    y_hat, _ = model_introspection.predict(
        [dataset.test_x[image_idx]], preprocessing=preprocessing
    )

    print("image to save")

    # Load raw images
    batch_directory = Directory("./dataset/experiment")
    batch = read_batch(batch_directory)

    y_pred, _ = model.predict(list(batch.values()), preprocessing=preprocessing)

    pred_batch = {list(batch.keys())[i]: y_pred[i] for i in range(len(y_pred))}

    cell_features = []
    cells_for_features = []
    visuals = []

    for image_name, p in pred_batch.items():
        if "NP" in image_name:
            condition = "NP"
        else:
            condition = "P"
        labels = p["labels"]
        visual_image = imread_color(
            "./dataset/experiment/" + image_name.replace(".tif", ".png"),
            rgb=True,
        )
        visuals.append(visual_image)
        plot_watershed(
            visual_image,
            p["mask"],
            p["markers"],
            p["labels"],
            filename=f"kartezio_{image_name.replace('.tif', '.png')}",
        )

        cells = labels_to_cells(labels, image_name, condition)
        for cell in cells:
            if cell is not None:
                cell_feature = get_cell_feature(cell, batch[image_name], profiling.lst)
                cell_features.append(cell_feature[0])
                cells_for_features.append(cell)
            else:
                print(f"cell is None in {image_name}")

    df = pd.DataFrame(cell_features, columns=cell_feature[1])
    df["area"] = [c.area for c in cells_for_features]
    print(f"area max: {np.max(df['area'])}  - area min: {np.min(df['area'])}")
    df = df[(df.area > 350) & (df.area < 6000)]
    print(df.values.shape)
    X_emb, classes, model_classifier = classify_data(df.values, 2, scale=True)

    dataset_out = df.copy()
    dataset_out["Cell Class"] = classes
    dataset_out["PCA-1"] = X_emb[:, 0]
    dataset_out["PCA-2"] = X_emb[:, 1]

    count_0 = np.count_nonzero(classes == 0)
    count_1 = np.count_nonzero(classes == 1)

    if count_1 > count_0:
        CLASS_CTL = 1
        CLASS_TARGET = 0
    else:
        CLASS_CTL = 0
        CLASS_TARGET = 1

    dataset_out["Cell Class"] = dataset_out["Cell Class"].apply(
        lambda x: "CTL" if x == CLASS_CTL else "Target"
    )
    pparam = dict(xlabel="PCA-1", ylabel="PCA-2")
    with plt.style.context(["science", "nature"]):
        fig, ax = plt.subplots()
        ax.tick_params(
            axis="both",  # changes apply to the x-axis
            which="both",  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            right=False,
            left=False,  # ticks along the top edge are off
            labelbottom=False,
            labelleft=False,
        )  # labels along the bottom edge are off

        df_Target = dataset_out[dataset_out["Cell Class"] == "Target"]
        ax.scatter(
            df_Target["PCA-1"],
            df_Target["PCA-2"],
            c="#0099DB",
            alpha=0.75,
            edgecolors="#0099DB",
            label="Target",
        )

        df_CTL = dataset_out[dataset_out["Cell Class"] == "CTL"]
        ax.scatter(
            df_CTL["PCA-1"],
            df_CTL["PCA-2"],
            c="#F77622",
            alpha=0.75,
            edgecolors="#F77622",
            label="CTL",
        )

        ax.legend()
        ax.set(**pparam)
        fig.savefig("./results/CTL_Target.png", dpi=300)

    figures = {}
    overlay_images = {}
    for image_name, visual in zip(list(batch.keys()), visuals):
        overlay_images[image_name] = visual
        figures[image_name] = Figure("Cell Classification and Interaction", size=(8, 4))
        figures[image_name].create_panels(1, 2, show_axis=False)
        figures[image_name].get_panel(0).imshow(visual)

    cells_CTL = []
    cells_target = []

    for index, row in dataset_out.iterrows():
        one_cell = cells_for_features[index]
        image_name = one_cell.get_data("image")
        cell_mask = one_cell.mask
        if row["Cell Class"] == "CTL":
            cells_CTL.append(one_cell)
            color = [247, 118, 34]
        elif row["Cell Class"] == "Target":
            cells_target.append(one_cell)
            color = [0, 153, 219]
        overlay_images[image_name] = draw_overlay(
            overlay_images[image_name], cell_mask, color=color, alpha=1.0
        )

    CONTACT_PX = 1
    DISTANCE = 70
    synapses = []
    for one_ctl in cells_CTL:
        for one_target in cells_target:
            if one_ctl.get_data("image") == one_target.get_data("image"):
                if one_ctl.distance(one_target) < DISTANCE:
                    image_name = one_ctl.get_data("image")
                    if (
                        np.count_nonzero(
                            one_ctl.mask & morph_dilate(one_target.mask, 3)
                        )
                        >= CONTACT_PX
                    ):
                        custom_data = {
                            "image": one_ctl.get_data("image"),
                            "condition": one_ctl.get_data("condition"),
                            "name": f"{one_ctl.name}-{one_target.name}",
                        }
                        synapse = Synapse(one_ctl, one_target, custom_data=custom_data)
                        synapses.append(synapse)

                        figures[one_ctl.get_data("image")].get_panel(1).plot(
                            [one_ctl.x, one_target.x],
                            [one_ctl.y, one_target.y],
                            c="green",
                        )

    for image_name in list(batch.keys()):
        filename = f"./results/kartezio_classes_{image_name.replace('.tif', '.png')}"
        cv2.imwrite(filename, rgb2bgr(overlay_images[image_name]))
