from river import stream
from river.datasets import base


class Yeast(base.RemoteDataset):
    """
    This dataset contains micro-array expressions and phylogenetic profiles for 2417 yeast genes. Each gen is annotated with a subset of 14 functional categories (e.g. Metabolism, energy, etc.) of the top level of the functional catalogue.

    Andre Elisseeff and Jason Weston. A kernel method for multi-labelled classification. In In Advances in Neural Information Processing Systems 14, volume 14, pages 681-687, 2001.
    """
    def __init__(self):
        super().__init__(
            task=base.MO_BINARY_CLF,
            n_samples=2417,
            n_features=103,
            n_outputs=14,
            url="http://www.uco.es/grupos/kdis/MLLResources/ucobigfiles/Datasets/Full/Yeast_Meka.zip",
            unpack=True,
            filename="Yeast.arff",
            size=2_416_740,
        )

    def _iter(self):
        return stream.iter_arff(
            self.path,
            sparse=False,
            target=[
                "Class1",
                "Class2",
                "Class3",
                "Class4",
                "Class5",
                "Class6",
                "Class7",
                "Class8",
                "Class9",
                "Class10",
                "Class11",
                "Class12",
                "Class13",
                "Class14",
            ],
        )
