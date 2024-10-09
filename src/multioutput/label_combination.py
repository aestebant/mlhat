from collections import defaultdict

from river import base, tree

from .majority import MajorityLabelset


class LabelCombination(MajorityLabelset, base.Wrapper):
    def __init__(self, model: base.Classifier = tree.HoeffdingTreeClassifier()) -> None:
        super().__init__()
        self.model = model

    @property
    def _wrapped_model(self):
        return self.model

    @property
    def _multiclass(self):
        return self.model._multiclass

    def learn_one(self, x, y, **kwargs):
        super().learn_one(x, y)
        self.model.learn_one(x, self.labelsets.index(y), **kwargs)

    def predict_proba_one(self, x):
        y_pred = defaultdict(lambda: defaultdict(lambda: 0.0))
        map_y_probs = self.model.predict_proba_one(x)
        for map_y, prob in map_y_probs.items():
            for label, value in self.labelsets[map_y].items():
                y_pred[label][value] += prob

        return {item[0]: dict(item[1]) for item in y_pred.items()}

    def predict_one(self, x):
        y_pred = self.predict_proba_one(x)
        return {label: max(y_pred[label], key=y_pred[label].get) for label in y_pred}
