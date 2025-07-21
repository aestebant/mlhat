from river.datasets import base
import stream


class VirusGO(base.RemoteDataset):
    """
    This dataset is used to predict the sub-cellular locations of proteins according to their sequences. It contains 207 sequences for Virus species. Both the GO (Gene ontology) features and PseAAC (including 20 amino acid, 20 pseudo-amino acid and 400 diptide components) are provided. There are 6 subcellular locations (viral capsid, host cell membrane, host endoplasm reticulum, host cytoplasm, host nucleus and secreted).

    Jianhua Xu, Jiali Liu, Jing Yin, and Chengyu Sun. A multi-label feature extraction algorithm via maximizing feature variance and feature-label dependence simultaneously. Knowledge-Based Systems, 98:172 — 184, 2016
    """
    def __init__(self):
        super().__init__(
            task=base.MO_BINARY_CLF,
            n_samples=207,
            n_features=749,
            n_outputs=6,
            url="http://www.uco.es/grupos/kdis/MLLResources/ucobigfiles/Datasets/Full/VirusGO_Meka.zip",
            unpack=True,
            filename="VirusGO.arff",
            size=35_304,
        )

    def _iter(self):
        return stream.iter_arff(
            self.path,
            sparse=True,
            target=["Label1", "Label2", "Label3", "Label4", "Label5", "Label6"],
        )
