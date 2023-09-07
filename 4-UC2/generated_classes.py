from kartezio.easy import generate_python_class
from kartezio.endpoint import LocalMaxWatershed
from kartezio.inference import CodeModel


MODEL_WGA = "973700-c4d70092-f9da-4832-85d1-564e23741d66"
MODEL_DIO = "375414-784bc9d8-43d7-4c07-934d-487f4fcc4f42"


# ============================== GENERATED CODE TO COPY ================================
class ModelWGA(CodeModel):
    def __init__(self):
        super().__init__(endpoint=LocalMaxWatershed(**{'threshold': 1, 'markers_distance': 5}))
    def _parse(self, X):
        x_0 = X[0]
        node_1 = self.call_node("morph_tophat", [x_0], [82, 174])
        node_2 = self.call_node("close", [node_1], [98, 97])
        node_5 = self.call_node("canny", [node_2], [141, 73])
        node_15 = self.call_node("sharpen", [x_0], [195, 116])
        node_16 = self.call_node("subtract", [node_5, node_15], [36, 133])
        node_18 = self.call_node("laplacian", [node_16], [122, 18])
        node_19 = self.call_node("fill_holes", [node_18], [235, 133])
        y_0 = node_19
        Y = [y_0]
        return Y
# ======================================================================================


# ============================== GENERATED CODE TO COPY ================================
class ModelDiO(CodeModel):
    def __init__(self):
        super().__init__(endpoint=LocalMaxWatershed(**{'threshold': 1, 'markers_distance': 5}))
    def _parse(self, X):
        x_0 = X[0]
        node_1 = self.call_node("canny", [x_0], [33, 69])
        node_3 = self.call_node("mean", [x_0, node_1], [175, 167])
        node_9 = self.call_node("canny", [node_3], [123, 44])
        node_12 = self.call_node("laplacian", [x_0], [207, 243])
        node_13 = self.call_node("sobel", [node_12], [65, 207])
        node_17 = self.call_node("close", [node_9], [25, 178])
        node_28 = self.call_node("bitwise_and_mask", [node_13, node_17], [227, 124])
        y_0 = node_28
        Y = [y_0]
        return Y
# ======================================================================================


if __name__ == "__main__":
    generate_python_class(f"./WGA/models/{MODEL_WGA}/elite.json", "ModelWGA")
    generate_python_class(f"./DiO/models/{MODEL_DIO}/elite.json", "ModelDiO")
