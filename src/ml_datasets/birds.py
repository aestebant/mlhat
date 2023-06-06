from river.datasets import base
import stream


class Birds(base.RemoteDataset):
    """
    It is a dataset to predict the set of birds species that are present, given a ten-second audio clip.

    Forrest Briggs, Yonghong Huang, Raviv Raich, Konstantinos Eftaxias, Zhong Lei, William Cukierski, Sarah Frey Hadley, Adam Hadley, Matthew Betts, Xiaoli Z. Fern, Jed Irvine, Lawrence Neal, Anil Thomas, Gábor Fodor, Grigorios Tsoumakas, Hong Wei Ng, Thi Ngoc Tho Nguyen, Heikki Huttunen, Pekka Ruusuvuori, Tapio Manninen, Aleksandr Diment, Tuomas Virtanen, Julien Marzat, Joseph Defretin, Dave Callender, Chris Hurlburt, Ken Larrey, and Maxim Milakov. The 9th annual MLSP competition: New methods for acoustic classification of multiple simultaneous bird species in a noisy environment. In IEEE International Workshop on Machine Learning for Signal Processing, MLSP 2013, Southampton, United Kingdom, September 22-25, 2013, pages 1-8, 2013.
    """
    def __init__(self):
        super().__init__(
            task=base.MO_BINARY_CLF,
            n_samples=645,
            n_features=260,
            n_outputs=19,
            url="http://www.uco.es/grupos/kdis/MLLResources/ucobigfiles/Datasets/Full/Birds_Meka.zip",
            unpack=True,
            filename="Birds.arff",
            size=1_071_429,
        )

    def _iter(self):
        return stream.iter_arff(
            self.path,
            target=[
                "BrownCreeper",
                "PacificWren",
                "PacificSlopeFlycatcher",
                "RedBreastedNuthatch",
                "DarkEyedJunco",
                "OliveSidedFlycatcher",
                "HermitThrush",
                "ChestnutBackedChickadee",
                "VariedThrush",
                "HermitWarbler",
                "SwainsonsThrush",
                "HammondsFlycatcher",
                "WesternTanager",
                "BlackHeadedGrosbeak",
                "GoldenCrownedKinglet",
                "WarblingVireo",
                "MacGillivraysWarbler",
                "StellarsJay",
                "CommonNighthawk",
            ],
        )
