from pathlib import Path
from river.datasets import base
from river import stream
import re

class SynthLoader(base.FileDataset):
    """
    Load data generated with the Java package MultiLabel-DataStreams
    """

    def __init__(self, filename):
        super().__init__(
            task=base.MO_BINARY_CLF,
            n_samples=-1,
            n_features=-1,
            n_outputs=-1,
            filename=filename,
            directory=Path(__file__).parent / 'synth'
        )
        self.n_samples = int(re.search('L([0-9]+)', self.filename).group(1))
        self.n_features = int(re.search('A([0-9]+)', self.filename).group(1))
        self.n_outputs = int(re.search('C([0-9]+)', self.filename).group(1))

    def __iter__(self):
        return stream.iter_arff(
            self.path,
            target=[f'class{i}' for i in range(self.n_outputs)]
        )
