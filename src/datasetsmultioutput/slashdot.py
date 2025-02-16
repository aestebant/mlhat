from river.datasets import base
import stream


class Slashdot(base.RemoteDataset):
    """
    It consists of article blurbs with subject categories representing the label space, mined from http://slashdot.org.

    Jesse Read. Scalable multi-label classification. PhD Thesis, University of Waikato, 2010.
    """
    def __init__(self):
        super().__init__(
            task=base.MO_BINARY_CLF,
            n_samples=3782,
            n_features=1079,
            n_outputs=22,
            url="http://www.uco.es/grupos/kdis/MLLResources/ucobigfiles/Datasets/Full/Slashdot_Meka.zip",
            unpack=True,
            filename="Slashdot.arff",
            size=253_679,
        )

    def _iter(self):
        return stream.iter_arff(
            self.path,
            sparse=True,
            target=[
                "Entertainment",
                "Interviews",
                "Main",
                "Developers",
                "Apache",
                "News",
                "Search",
                "Mobile",
                "Science",
                "IT",
                "BSD",
                "Idle",
                "Games",
                "YourRightsOnline",
                "AskSlashdot",
                "Apple",
                "BookReviews",
                "Hardware",
                "Meta",
                "Linux",
                "Politics",
                "Technology",
            ],
        )
