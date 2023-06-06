from river import stream
from river.datasets import base


class Gnegative(base.RemoteDataset):
    """
    This dataset is used to predict the sub-cellular locations of proteins according to their sequences. It contains 1392 sequences for Gram negative bacterial (Gnegative) species. Both the GO (Gene ontology) features and PseAAC (including 20 amino acid, 20 pseudo-amino acid and 400 diptide components) are provided. There are 8 subcellular locations (cell inner membrane, cell outer membrane, cytoplasm, extracellular, fimbrium, flagellum, nucleoid and periplasm).

    Jianhua Xu, Jiali Liu, Jing Yin, and Chengyu Sun. A multi-label feature extraction algorithm via maximizing feature variance and feature-label dependence simultaneously. Knowledge-Based Systems, 98:172 — 184, 2016
    """
    def __init__(self):
        super().__init__(
            task=base.MO_BINARY_CLF,
            n_samples=1392,
            n_features=440,
            n_outputs=8,
            url="http://www.uco.es/grupos/kdis/MLLResources/ucobigfiles/Datasets/Full/GnegativePseAAC_Meka.zip",
            unpack=True,
            filename="GnegativePseAAC.arff",
            size=3_041_079,
        )
    
    def _iter(self):
        return stream.iter_arff(
            self.path,
            sparse=True,
            target=[
                "Label1",
                "Label2",
                "Label3",
                "Label4",
                "Label5",
                "Label6",
                "Label7",
                "Label8",
            ],
        )