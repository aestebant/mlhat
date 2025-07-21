from river.datasets import base
import stream


class HumanPseAac(base.RemoteDataset):
    """
    This dataset is used to predict the sub-cellular locations of proteins according to their sequences. It contains 3106 sequences for Human species. Both the GO (Gene ontology) features and PseAAC (including 20 amino acid, 20 pseudo-amino acid and 400 diptide components) are provided. There are 14 subcellular locations (centriole, cytoplasm, cytoskeleton, endoplasm reticulum, endosome, extracell, golgi apparatus, lysosome, microsome, mitochondrion, nucleus, peroxisome, plasma membrace, and synapse).

    Jianhua Xu, Jiali Liu, Jing Yin, and Chengyu Sun. A multi-label feature extraction algorithm via maximizing feature variance and feature-label dependence simultaneously. Knowledge-Based Systems, 98:172 — 184, 2016
    """
    def __init__(self):
        super().__init__(
            task=base.MO_BINARY_CLF,
            n_samples=3106,
            n_features=440,
            n_outputs=14,
            url="http://www.uco.es/grupos/kdis/MLLResources/ucobigfiles/Datasets/Full/HumanPseAAC_Meka.zip",
            unpack=True,
            filename="HumanPseAAC.arff",
            size=7_869_568,
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
                "Label9",
                "Label10",
                "Label11",
                "Label12",
                "Label13",
                "Label14",
            ],
        )
