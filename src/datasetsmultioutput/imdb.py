from river.datasets import base
import stream


class Imdb(base.RemoteDataset):
    """
    It contains 120919 movie plot tex summaries from the Internet Movie Database (www.imdb.com), labelled with one or more genres.

    Jesse Read. Scalable multi-label classification. PhD Thesis, University of Waikato, 2010.
    """
    def __init__(self) -> None:
        super().__init__(
            task=base.MO_BINARY_CLF,
            n_samples=120919,
            n_features=1001,
            n_outputs=28,
            url="http://www.uco.es/grupos/kdis/MLLResources/ucobigfiles/Datasets/Full/Imdb_Meka.zip",
            unpack=True,
            filename="Imdb.arff",
            size=15_763_518,
        )

    def _iter(self):
        return stream.iter_arff(
            self.path,
            sparse=True,
            target=[
                "Sci-Fi",
                "Crime",
                "Romance",
                "Animation",
                "Music",
                "Comedy",
                "War",
                "Horror",
                "Film-Noir",
                "Adventure",
                "News",
                "Western",
                "Thriller",
                "Adult",
                "Mystery",
                "Short",
                "Talk-Show",
                "Drama",
                "Action",
                "Documentary",
                "Musical",
                "History",
                "Family",
                "Reality-TV",
                "Fantasy",
                "Game-Show",
                "Sport",
                "Biography",
            ],
        )