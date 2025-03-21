from river.datasets import base
import stream


class Nuswidec(base.RemoteDataset):
    """
    We provide two versions of the full NUS-WIDE dataset. In the first version, images are represented using 500-D bag of visual words features provided by the creators of the dataset [Chua et al. 2009]. In the second version, images are represented using 128-D cVLAD+ features described in [Spyromitros et al. 2014]. In both cases, the 1st attribute is the image id.

    [Chua et al. 2009]: Tat-Seng Chua, Jinhui Tang, Richang Hong, Haojie Li, Zhiping Luo, and Yan-Tao Zheng. “NUS-WIDE: A Real-World Web Image Database from National University of Singapore”, ACM International Conference on Image and Video Retrieval. Greece. Jul. 8-10, 2009.
    [Spyromitros et al. 2014]: E. Spyromitros-Xioufis, S. Papadopoulos, Y. Kompatsiaris, G. Tsoumakas, I. Vlahavas, “A Comprehensive Study over VLAD and Product Quantization in Large-scale Image Retrieval”, IEEE Transactions on Multimedia, 2014.
    """
    def __init__(self) -> None:
        super().__init__(
            task=base.MO_BINARY_CLF,
            n_samples=269648,
            n_features=129,
            n_outputs=81,
            url="http://www.uco.es/grupos/kdis/MLLResources/ucobigfiles/Datasets/Full/Nuswide_cVLADplus_Meka.zip",
            unpack=True,
            filename="Nuswide_cVLADplus.arff",
            size=464_257_823,
        )

    def _iter(self):
        return stream.iter_arff(
            self.path,
            sparse=False,
            drop=["image_name"],
            target=[
                "zebra",
                "window",
                "whales",
                "wedding",
                "waterfall",
                "water",
                "vehicle",
                "valley",
                "tree",
                "train",
                "toy",
                "town",
                "tower",
                "tiger",
                "temple",
                "tattoo",
                "swimmers",
                "surf",
                "sunset",
                "sun",
                "street",
                "statue",
                "sports",
                "soccer",
                "snow",
                "sky",
                "sign",
                "sand",
                "running",
                "rocks",
                "road",
                "reflection",
                "rainbow",
                "railroad",
                "protest",
                "police",
                "plants",
                "plane",
                "person",
                "ocean",
                "nighttime",
                "mountain",
                "moon",
                "military",
                "map",
                "leaf",
                "lake",
                "house",
                "horses",
                "harbor",
                "grass",
                "glacier",
                "garden",
                "frost",
                "fox",
                "food",
                "flowers",
                "flags",
                "fish",
                "fire",
                "elk",
                "earthquake",
                "dog",
                "dancing",
                "cow",
                "coral",
                "computer",
                "clouds",
                "cityscape",
                "cat",
                "castle",
                "cars",
                "buildings",
                "bridge",
                "book",
                "boats",
                "birds",
                "bear",
                "beach",
                "animal",
                "airport",
            ],
        )