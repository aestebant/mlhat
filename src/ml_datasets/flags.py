from river.datasets import base
import stream


class Flags(base.RemoteDataset):
    """
    This dataset contains details of some countries and their flags, and the goal is to predict some of the features. 

    E.C. Goncalves, Alexandre Plastino, and Alex A. Freitas. A genetic algorithm for optimizing the label ordering in multi-label classifier chains. In IEEE 25th International Conference on Tools with Artificial Intelligence, pages 469-476. IEEE Computer Society Conference Publishing Services (CPS), 2013.
    """
    def __init__(self):
        super().__init__(
            task=base.MO_BINARY_CLF,
            n_samples=194,
            n_features=19,
            n_outputs=7,
            url="http://www.uco.es/grupos/kdis/MLLResources/ucobigfiles/Datasets/Full/Flags_Meka.zip",
            unpack=True,
            filename="Flags.arff",
            size=11_241,
        )

    def _iter(self):
        return stream.iter_arff(
            self.path,
            target=["red", "green", "blue", "yellow", "white", "black", "orange"],
        )
