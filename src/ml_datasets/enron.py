from river.datasets import base
import stream


class Enron(base.RemoteDataset):
    """
    The Enron dataset is a subset of Enron email Corpus, labelled with a set of categories. It is based in a collection of email messages that were categorized into 53 topic categories, such as company strategy, humour and legal advice.

    Jesse Read, Bernhard Pfahringer, and Geoff Holmes. Multi-label Classification Using Ensembles of Pruned Sets. In ICDM'08: Proceedings of the 2008 Eighth IEEE International Conference on Data Mining, volume 0, pages 995-1000, Washington, DC, USA, 2008. IEEE Computer Society.
    """
    def __init__(self):
        super().__init__(
            task=base.MO_BINARY_CLF,
            n_samples=1702,
            n_features=1001,
            n_outputs=53,
            url="http://www.uco.es/grupos/kdis/MLLResources/ucobigfiles/Datasets/Full/Enron_Meka.zip",
            unpack=True,
            filename="Enron.arff",
            size=908_718,
        )

    def _iter(self):
        return stream.iter_arff(
            self.path,
            sparse=True,
            target=[
                "A.A8",
                "C.C9",
                "B.B12",
                "C.C11",
                "C.C5",
                "C.C7",
                "B.B2",
                "B.B3",
                "D.D16",
                "A.A7",
                "D.D1",
                "A.A4",
                "C.C2",
                "A.A3",
                "A.A1",
                "D.D9",
                "D.D19",
                "B.B8",
                "D.D12",
                "D.D6",
                "C.C8",
                "A.A6",
                "B.B9",
                "A.A5",
                "C.C10",
                "B.B1",
                "D.D5",
                "B.B11",
                "D.D2",
                "B.B4",
                "D.D15",
                "C.C4",
                "D.D8",
                "B.B6",
                "D.D3",
                "D.D13",
                "D.D7",
                "C.C12",
                "B.B7",
                "C.C6",
                "B.B5",
                "D.D11",
                "A.A2",
                "C.C3",
                "D.D10",
                "D.D18",
                "B.B13",
                "D.D17",
                "B.B10",
                "C.C1",
                "D.D4",
                "C.C13",
                "D.D14",
            ],
        )
