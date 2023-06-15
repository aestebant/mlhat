from river.datasets import base
import stream


class WaterQuality(base.RemoteDataset):
    """
    This dataset is used to predict the quality of water of Slovenian rivers, knowing 16 characteristics such as the temperature, ph, hardness, NO2 or C02.

    H. Blockeel, S. Džeroski, and J. Grbovic. Simultaneous prediction of multiple chemical parameters of river water quality with tilde. Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics), 1704:32-40, 1999.
    """
    def __init__(self):
        super().__init__(
            task=base.MO_BINARY_CLF,
            n_samples=1060,
            n_features=16,
            n_outputs=14,
            url="http://www.uco.es/grupos/kdis/MLLResources/ucobigfiles/Datasets/Full/Water-quality_Meka.zip",
            unpack=True,
            filename="Water-quality.arff",
            size=336_783,
        )

    def _iter(self):
        return stream.iter_arff(
            self.path,
            target=[
                "25400",
                "29600",
                "30400",
                "33400",
                "17300",
                "19400",
                "34500",
                "38100",
                "49700",
                "50390",
                "55800",
                "57500",
                "59300",
                "37880",
            ],
        )
