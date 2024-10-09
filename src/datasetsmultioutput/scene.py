from river.datasets import base
import stream


class Scene(base.RemoteDataset):
    """
    It is a image dataset, that contains 2407 images, annotated in up to 6 classes: beach, sunset, fall foliage, field, mountain and urban. Each image is described with 294 visual numeric features corresponding to spatial colour moments in the LUV space.

    Matthew R. Boutell, Jiebo Luo, Xipeng Shen, and Christopher M. Brown. Learning multi-label scene classification. Pattern Recognition, 37(9):1757-1771, September 2004.
    """
    def __init__(self):
        super().__init__(
            task=base.MO_BINARY_CLF,
            n_samples=2407,
            n_features=294,
            n_outputs=6,
            url="http://www.uco.es/grupos/kdis/MLLResources/ucobigfiles/Datasets/Full/Scene_Meka.zip",
            unpack=True,
            filename="Scene.arff",
            size=6_210_993,
        )

    def _iter(self):
        return stream.iter_arff(
            self.path,
            target=["Beach", "Sunset", "FallFoliage", "Field", "Mountain", "Urban"],
        )
