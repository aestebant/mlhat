from river.datasets import base
import stream


class Mediamill(base.RemoteDataset):
    """
    It is a multimedia dataset for generic video indexing, which was extracted tom the TRECVID 2005/2006 benchmark. This dataset contains 85 hours of international broadcast news data categorized into 100 labels and each video instance is represented as a 120-dimensional feature vector of numeric features.

    C.G.M. Snoek, M.Worring, J.C. van Gemert, J.-M. Geusebroek, A.W.M. Smeulders. 2006. The Challenge Problem for Automated Detection of 101 Semantic Concepts in Multimedia, In Proceedings of ACM Multimedia, 421-430.
    """
    def __init__(self):
        super().__init__(
            task=base.MO_BINARY_CLF,
            n_samples=43907,
            n_features=120,
            n_outputs=101,
            url="http://www.uco.es/grupos/kdis/MLLResources/ucobigfiles/Datasets/Full/Mediamill_Meka.zip",
            unpack=True,
            filename="Mediamill.arff",
            size=55_708_970,
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
                "Class15",
                "Class16",
                "Class17",
                "Class18",
                "Class19",
                "Class20",
                "Class21",
                "Class22",
                "Class23",
                "Class24",
                "Class25",
                "Class26",
                "Class27",
                "Class28",
                "Class29",
                "Class30",
                "Class31",
                "Class32",
                "Class33",
                "Class34",
                "Class35",
                "Class36",
                "Class37",
                "Class38",
                "Class39",
                "Class40",
                "Class41",
                "Class42",
                "Class43",
                "Class44",
                "Class45",
                "Class46",
                "Class47",
                "Class48",
                "Class49",
                "Class50",
                "Class51",
                "Class52",
                "Class53",
                "Class54",
                "Class55",
                "Class56",
                "Class57",
                "Class58",
                "Class59",
                "Class60",
                "Class61",
                "Class62",
                "Class63",
                "Class64",
                "Class65",
                "Class66",
                "Class67",
                "Class68",
                "Class69",
                "Class70",
                "Class71",
                "Class72",
                "Class73",
                "Class74",
                "Class75",
                "Class76",
                "Class77",
                "Class78",
                "Class79",
                "Class80",
                "Class81",
                "Class82",
                "Class83",
                "Class84",
                "Class85",
                "Class86",
                "Class87",
                "Class88",
                "Class89",
                "Class90",
                "Class91",
                "Class92",
                "Class93",
                "Class94",
                "Class95",
                "Class96",
                "Class97",
                "Class98",
                "Class99",
                "Class100",
                "Class101",
            ],
        )