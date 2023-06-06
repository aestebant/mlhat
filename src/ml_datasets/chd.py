from river import stream
from river.datasets import base


class Chd(base.RemoteDataset):
    """
    This dataset has information of coronary heart disease (CHD) in traditional Chinese medicine (TCM). This dataset has been filtered by specialist removing irrelevant features, keeping only 49 features.

    H. Shao, G.Z. Li, G.P. Liu, and Y.Q. Wang. Symptom selection for multi-label data of inquiry diagnosis in traditional chinese medicine. Science China Information Sciences, 56(5):1-13, 2013.
    """
    def __init__(self) -> None:
        super().__init__(
            task=base.MO_BINARY_CLF,
            n_samples=555,
            n_features=49,
            n_outputs=6,
            url="http://www.uco.es/grupos/kdis/MLLResources/ucobigfiles/Datasets/Full/CHD_49_Meka.zip",
            unpack=True,
            filename="CHD_49.arff",
            size=79_523,
        )
    
    def _iter(self):
        return stream.iter_arff(
            self.path,
            sparse=False,
            target=[
                "label1",
                "label2",
                "label3",
                "label4",
                "label5",
                "label6",
            ],
        )