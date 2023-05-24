import abc
from river import drift


class MultiLabelNode(abc.ABC):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @property
    def total_weight(self):
        if self.stats and "total" in self.stats.keys():
            return self.stats["total"]
        else:
            return 0

    def update_stats(self, y, sample_weight):
        # TODO: only considers the multi-label binary classification
        for label, value in y.items():
            if value:
                try:
                    self.stats[label] += sample_weight
                except KeyError:
                    self.stats[label] = sample_weight
        try:
            self.stats["total"] += sample_weight
        except KeyError:
            self.stats["total"] = sample_weight


class MultiLabelAdaptiveNode(MultiLabelNode):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def update_drift_detector(self, y, y_pred, tree) -> bool:
        warning = False
        if isinstance(self.drift_detector, drift.ADWIN):
            if tree.adwin_metric == tree._ADWIN_PER_LABEL:
                for label in y_pred:
                    error = 0 if y[label] == y_pred[label] else 1
                    self.drift_detector.update(error)
                    self._mean_error.update(error)
                    warning = warning or bool(error)
            else:
                if tree.adwin_metric == tree._ADWIN_SUBSET_ACC:
                    y_aux = {label: val for label, val in y.items() if label in y_pred}
                    error = 0 if y_aux == y_pred else 1
                    warning = bool(error)
                elif tree.adwin_metric == tree._ADWIN_HAMMING_LOSS:
                    error = len([label for label, val in y_pred.items() if y[label] != val]) / len(
                        y_pred
                    )
                    warning = error > 0.1
                self._mean_error.update(error)
                self.drift_detector.update(error)
        elif isinstance(self.drift_detector, drift.LD3):
            self.drift_detector.update(y_pred)
            warning = self.drift_detector.warning_detected
        else:
            raise TypeError
        return warning

    def move_to_frontground(self):
        self._is_background = False