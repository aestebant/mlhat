import collections
import copy

from river import base, linear_model


class BinaryRelevance(base.Wrapper, collections.UserDict, base.MultiLabelClassifier):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    @property
    def _wrapped_model(self):
        return self.model

    def __getitem__(self, key):
        try:
            return collections.UserDict.__getitem__(self, key)
        except KeyError:
            collections.UserDict.__setitem__(self, key, copy.deepcopy(self.model))
            return self[key]

    @classmethod
    def _unit_test_params(cls):
        yield {"model": linear_model.LogisticRegression()}

    @property
    def _multiclass(self):
        return self.model._multiclass

    def learn_one(self, x, y, **kwargs):
        for label in y:
            clf = self[label]
            y_l = y[label]
            clf.learn_one(x, y_l, **kwargs)

    def predict_proba_one(self, x):
        y_pred = dict()
        for label, clf in self.items():
            y_pred[label] = clf.predict_proba_one(x)
        return y_pred

    def predict_one(self, x):
        y_pred = self.predict_proba_one(x)
        return {c: max(y_pred[c], key=y_pred[c].get) for c in y_pred}
