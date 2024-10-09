import copy
from collections import defaultdict
from typing import DefaultDict, Dict, List

from river import base


class MajorityLabelset(base.MultiLabelClassifier):

    _BINARY = 1
    _CATEGORICAL = 2
    _ORDINAL = 3
    _NEUTRAL: Dict[int, base.typing.ClfTarget] = {_BINARY: False, _CATEGORICAL: "", _ORDINAL: 0}

    def __init__(self) -> None:
        super().__init__()
        self.label_space: Dict[str, int] = dict()
        self.labelsets: List[Dict[str, base.typing.ClfTarget]] = list()
        self.count_ls: DefaultDict[int, int] = defaultdict(lambda: 0)

    @property
    def _multiclass(self):
        return True

    def learn_one(self, x, y, sample_weight=1):
        self._update_labelsets(y)
        self.count_ls[self.labelsets.index(y)] += sample_weight

    def predict_proba_one(self, x):
        mls = self.predict_one(x)
        if not mls:
            return {}
        proba: DefaultDict[str, Dict[base.typing.ClfTarget, float]] = defaultdict(lambda: dict())
        for label, value in mls.items():
            proba[label][value] = 1.0
            proba[label][not value] = 0.0
        # TODO: multi-class multi-label
        return dict(proba)

    def predict_one(self, x):
        if not self.labelsets:
            return None
        # Invoking __getitem__ instead of dict.get to make MyPy happy.
        return self.labelsets[max(self.count_ls, key=lambda k: self.count_ls[k])]

    def _update_label_space(self, y) -> set:
        new_labels = set(y.keys()).difference(self.label_space.keys())
        for label in new_labels:
            if isinstance(y[label], bool):  # Binary multi-label
                self.label_space[label] = self._BINARY
            elif isinstance(y[label], (int, float)):  # Ordinal multi-label
                self.label_space[label] = self._ORDINAL
            else:  # Categorical multi-label
                self.label_space[label] = self._CATEGORICAL
        return new_labels

    def _update_labelsets(self, y):
        if y in self.labelsets:
            return

        new_labels = self._update_label_space(y)
        for label in new_labels:
            for ls in self.labelsets:
                ls[label] = self._NEUTRAL[self.label_space[label]]

        uncomplete_labels = set(self.label_space.keys()).difference(y.keys())
        if uncomplete_labels:
            y = copy.copy(y)
            for label in uncomplete_labels:
                y[label] = self._NEUTRAL[self.label_space[label]]

        self.labelsets.append(y)
