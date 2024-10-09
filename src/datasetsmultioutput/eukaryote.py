from river.datasets import base
import stream


class Eukaryote(base.RemoteDataset):
    """
    This dataset is used to predict the sub-cellular locations of proteins according to their sequences. It contains 7766 sequences for Eukaryote species. Both the GO (Gene ontology) features and PseAAC (including 20 amino acid, 20 pseudo-amino acid and 400 diptide components) are provided. There are 22 subcellular locations (acrosome, cell membrane, cell wall, centrosome, chloroplast, cyanelle, cytoplasm, cytoeskeleton, endoplasmatic reticulum, endosome, extracell, golgi apparatus, hydrogenosome, lysosome, melanosome, microsome, mitochondrion, nucleus, peroxisome, spindle pole body, synapse and vacuole).

    Jianhua Xu, Jiali Liu, Jing Yin, and Chengyu Sun. A multi-label feature extraction algorithm via maximizing feature variance and feature-label dependence simultaneously. Knowledge-Based Systems, 98:172 — 184, 2016
    """
    def __init__(self):
        super().__init__(
            task=base.MO_BINARY_CLF,
            n_samples=7766,
            n_features=440,
            n_outputs=22,
            url="http://www.uco.es/grupos/kdis/MLLResources/ucobigfiles/Datasets/Full/EukaryotePseAAC_Meka.zip",
            unpack=True,
            filename="EukaryotePseAAC.arff",
            size=18_877_490,
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
                "Label15",
                "Label16",
                "Label17",
                "Label18",
                "Label19",
                "Label20",
                "Label21",
                "Label22",
            ],
        )
