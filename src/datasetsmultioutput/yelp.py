from river.datasets import base
import stream


class Yelp(base.RemoteDataset):
    """
    This dataset has been obtained from the user’s reviews and ratings about business and services on Yelp. It is used in order to categorize if the food, service, ambiance, deals and price of one of these business are good or not. It contains more than 10000 reviews of users.

    H. Sajnani, V. Saini, K. Kumar , E. Gabrielova , P. Choudary, C. Lopes. 2013. Classifying Yelp reviews into relevant categories.
    """
    def __init__(self):
        super().__init__(
            task=base.MO_BINARY_CLF,
            n_samples=10806,
            n_features=671,
            n_outputs=5,
            url="http://www.uco.es/grupos/kdis/MLLResources/ucobigfiles/Datasets/Full/Yelp_Meka.zip",
            unpack=True,
            filename="Yelp.arff",
            size=14_627_346,
        )

    def _iter(self):
        return stream.iter_arff(
            self.path,
            sparse=False,
            target=[
                "IsFoodGood",
                "IsServiceGood",
                "IsAmbianceGood",
                "IsDealsGood",
                "IsPriceGood",
            ],
        )
