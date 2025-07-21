from river.datasets import base
import stream


class Tmc2007500(base.RemoteDataset):
    """
    It is a subset of the Aviation Safety Reporting System dataset. It contains 28596 aviation safety free text reports that the fligth crew submit after each flight about events that took place during the flight. The goal is to label the documents with respect to what types of problem they describe. The dataset has 49060 discrete attributes corresponding to terms in the collection. The safety reports are provided with 22 labels, each of them representing a problem type that appears during a flight.

    A. Srivastava, B. Zane-Ulman: Discovering recurring anomalies in text reports regarding complex space systems. In: 2005 IEEE Aerospace Conference. (2005)
    """
    def __init__(self):
        super().__init__(
            task=base.MO_BINARY_CLF,
            n_samples=28600,
            n_features=500,
            n_outputs=22,
            url="http://www.uco.es/grupos/kdis/MLLResources/ucobigfiles/Datasets/Full/tmc2007-500_Meka.zip",
            unpack=True,
            filename="tmc2007-500.arff",
            size=4_152_538,
        )

    def _iter(self):
        return stream.iter_arff(
            self.path,
            sparse=True,
            target=[
                "class01",
                "class02",
                "class03",
                "class04",
                "class05",
                "class06",
                "class07",
                "class08",
                "class09",
                "class10",
                "class11",
                "class12",
                "class13",
                "class14",
                "class15",
                "class16",
                "class17",
                "class18",
                "class19",
                "class20",
                "class21",
                "class22",
            ],
        )