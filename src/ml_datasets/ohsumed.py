from river import stream
from river.datasets import base


class Ohsumed(base.RemoteDataset):
    """
    This collection includes medical abstracts from the MeSH categories of the year 1991. The specific task was to categorize the 23 cardiovascular diseases categories.

    Thorsten Joachims, Text Categorization with Support Vector Machines: Learning with Many Relevant Features. In: Nédellec C., Rouveirol C. (eds) Machine Learning: ECML-98. ECML 1998. Lecture Notes in Computer Science (Lecture Notes in Artificial Intelligence), vol 1398.
    """
    def __init__(self):
        super().__init__(
            task=base.MO_BINARY_CLF,
            n_samples=13929,
            n_features=1002,
            n_outputs=23,
            url="http://www.uco.es/grupos/kdis/MLLResources/ucobigfiles/Datasets/Full/Ohsumed_Meka.zip",
            unpack=True,
            filename="Ohsumed.arff",
            size=3_423_747,
        )

    def _iter(self):
        return stream.iter_arff(
            self.path,
            sparse=True,
            target=[
                "C12",
                "C04",
                "C19",
                "C03",
                "C18",
                "C11",
                "C05",
                "C14",
                "C02",
                "C07",
                "C16",
                "C10",
                "C13",
                "C09",
                "C20",
                "C17",
                "C23",
                "C22",
                "C15",
                "C21",
                "C06",
                "C08",
                "C01",
            ],
        )
