from river.datasets import base
import stream


class D20ng(base.RemoteDataset):
    """
    It is a compilation of around 20000 post to 20Newsgroups. Around 1000 posts are available for each group.

    K. Lang. 2008. The 20 newsgroup dataset. http://people.csail.mit.edu/jrennie/20Newsgroups/.
    """
    def __init__(self):
        super().__init__(
            task=base.MO_BINARY_CLF,
            n_samples=19300,
            n_features=1006,
            n_outputs=20,
            url="http://www.uco.es/grupos/kdis/MLLResources/ucobigfiles/Datasets/Full/20NG_Meka.zip",
            unpack=True,
            filename="20NG.arff",
            size=3_842_021,
        )
    
    def _iter(self):
        return stream.iter_arff(
            self.path,
            sparse=True,
            target=[
                "comp.os_ms_windows_misc",
                "religion.rmisc",
                "rec.sport.baseball",
                "sci.space",
                "comp.sys.mac_hardware",
                "sci.med",
                "politics.pmisc",
                "rec.autos",
                "misc_forsale",
                "politics.mideast",
                "rec.motorcycles",
                "politics.guns",
                "rec.sport.hockey",
                "comp.sys.ibm_pc_hardware",
                "comp.graphics",
                "sci.crypt",
                "sci.electronics",
                "religion.christian",
                "religion.atheism",
                "comp.windows_x",
            ]
        )