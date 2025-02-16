from pathlib import Path
from river.datasets import base
import stream


class Hypersphere(base.FileDataset):
    """
    Hyperspherical Learning in Multi-Label Classification
    """
    def __init__(self) -> None:
        super().__init__(
            task=base.MO_BINARY_CLF,
            n_samples=100000,
            n_features=100,
            n_outputs=10,
            directory=Path(__file__).parent,
            filename="Hypersphere.arff",
        )

    def _iter(self):
        return stream.iter_arff(
            self.path,
            sparse=False,
            target=[
                "y1",
                "y2",
                "y3",
                "y4",
                "y5",
                "y6",
                "y7",
                "y8",
                "y9",
                "y10",
            ],
        )

    def __iter__(self):
        return self._iter()