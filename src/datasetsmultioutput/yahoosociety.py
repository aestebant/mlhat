from river.datasets import base
import stream


class YahooSociety(base.RemoteDataset):
    """
    It is a dataset to categorize web pages and consists of 14 top-level categories, each one is classified into a number of second-level categories. By focusing in second-level categories, there were used 11 out of the 14 independent text categorization problems.

    N. Ueda, K. Saito: Parametric mixture models for multi-labeled text, In Neural Information Processing Systems 15 (NIPS 15), MIT Press, pp. 737-744, 2002.
    """
    def __init__(self) -> None:
        super().__init__(
            task=base.MO_BINARY_CLF,
            n_samples=14512,
            n_features=31802,
            n_outputs=27,
            url="http://www.uco.es/grupos/kdis/MLLResources/ucobigfiles/Datasets/Full/Yahoo_Society_Meka.zip",
            unpack=True,
            filename="Yahoo_Society.arff",
            size=18_885_044,
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
                "Label23",
                "Label24",
                "Label25",
                "Label26",
                "Label27",
            ],
        )