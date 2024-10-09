from river.datasets import base
import stream


class Emotions(base.RemoteDataset):
    """
    Also called Music. Is a small dataset to classify music into emotions that it evokes according to the Tellegen-Watson-Clark model of mood: amazed-suprised, happy-pleased, relaxing-calm, quiet-still, sad-lonely and angry-aggresive. It consists of 593 songs with 6 classes.

    G. Tsoumakas, I. Katakis, and I. Vlahavas. Effective and Efficient Multilabel Classification in Domains with Large Number of Labels. In Proc. ECML/PKDD 2008 Workshop on Mining Multidimensional Data (MMD'08), 2008.
    """
    def __init__(self):
        super().__init__(
            task=base.MO_BINARY_CLF,
            n_samples=593,
            n_features=72,
            n_outputs=6,
            url="http://www.uco.es/grupos/kdis/MLLResources/ucobigfiles/Datasets/Full/Emotions_Meka.zip",
            unpack=True,
            filename="Emotions.arff",
            size=380_486,
        )

    def _iter(self):
        return stream.iter_arff(
            self.path,
            target=[
                "amazed-suprised",
                "happy-pleased",
                "relaxing-calm",
                "quiet-still",
                "sad-lonely",
                "angry-aggresive",
            ],
        )
