from river import metrics
from river.metrics.multioutput.base import MultiOutputClassificationMetric


class HammingLoss(metrics.base.MeanMetric, MultiOutputClassificationMetric):

    _fmt = ".4"

    def _eval(self, y_true, y_pred):
        return len([label for label, val in y_true.items() if y_pred[label] != val]) / len(y_true)

    @property
    def bigger_is_better(self):
        return False
