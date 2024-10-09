import abc
import copy
import numbers

from river import drift
from river.tree.nodes import htc_nodes, branch
from river.tree.utils import BranchFactory, round_sig_fig


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
        empty = True
        for label, value in y.items():
            if value:
                empty = False
                if label in self.stats.keys():
                    self.stats[label] += sample_weight
                else:
                    self.stats[label] = sample_weight
        if empty:
            if 'empty' in self.stats.keys():
                self.stats['empty'] += sample_weight
            else:
                self.stats['empty'] = sample_weight
        if 'total' in self.stats.keys():
            self.stats["total"] += sample_weight
        else:
            self.stats["total"] = sample_weight


class MultiLabelAdaptiveNode(MultiLabelNode):
    def __init__(self, drift_detector, perf_metric, *args, **kwargs) -> None:
        self.drift_detector = drift_detector
        self._mean_perf = perf_metric
        self._post_drift_weight = 0
        self.warning = False
        super().__init__(*args, **kwargs)

    def update_drift_detector(self, y, y_pred, sample_weigth, tree):
        if isinstance(self.drift_detector, drift.ADWIN):
            old_perf = self._mean_perf.get()
            if tree.drift_method == tree._ADWIN_PER_LABELSET:
                self._mean_perf.update(y, y_pred, sample_weigth)
                self.drift_detector.update(self._mean_perf.get())
            elif tree.drift_method == tree._ADWIN_PER_LABEL:
                for label in y_pred:
                    error = 0 if y[label] == y_pred[label] else 1
                    self.drift_detector.update(error)
                    self._mean_perf.update(error)
            curr_perf = self._mean_perf.get()
            self.warning = (self._mean_perf.bigger_is_better and old_perf > curr_perf) or (not self._mean_perf.bigger_is_better and curr_perf > old_perf)
        elif isinstance(self.drift_detector, drift.LD3):
            self.drift_detector.update(y_pred)
            self.warning = self.drift_detector.warning_detected
        else:
            raise TypeError

    def move_to_frontground(self):
        self._is_background = False


class LeafMultiLabel(MultiLabelNode, htc_nodes.LeafMajorityClass):
    def __init__(self, stats, depth, splitter, ml_clf, parent_ml_clf, **kwargs):
        super().__init__(stats, depth, splitter, **kwargs)
        self.ml_clf = copy.deepcopy(ml_clf)
        self.parent_ml_clf = parent_ml_clf  # Produce predictions if the classifier at leaf is not initialized yet (any instance received)

    @property
    def n_active_leaves(self):
        if self.is_active():
            return 1
        else:
            return 0

    @property
    def n_inactive_leaves(self):
        if self.is_active():
            return 0
        else:
            return 1

    def update_splitters(self, x, y, sample_weight, nominal_attributes):
        for att_id, att_val in self._iter_features(x):
            if att_id in self._disabled_attrs:
                continue
            try:
                splitter = self.splitters[att_id]
            except KeyError:
                if (
                    nominal_attributes is not None and att_id in nominal_attributes
                ) or not isinstance(att_val, numbers.Number):
                    splitter = self.new_nominal_splitter()
                else:
                    splitter = copy.deepcopy(self.splitter)

                self.splitters[att_id] = splitter
            for label, val in y.items():
                if val:
                    splitter.update(att_val, label, sample_weight)
            splitter.update(att_val, "total", sample_weight)

    def best_split_suggestions(self, criterion, tree) -> list[BranchFactory]:
        maj_class = max([count for label, count in self.stats.items() if label != 'total'])
        # Only perform split attempts when the majority class does not dominate
        # the amount of observed instances
        if maj_class and maj_class / self.total_weight > tree.max_share_to_split:
            return [BranchFactory()]
        return super().best_split_suggestions(criterion, tree)

    def calculate_promise(self):
        if self.total_weight > 0:
            return self.total_weight - max(
                {k: v for (k, v) in self.stats.items() if k != "total"}.values()
            )
        else:
            return 0

    def observed_class_distribution_is_pure(self):
        # TODO: only works with classifiers that save label sets combinations
        return len(self.ml_clf.labelsets) <= 1

    def learn_one(self, x, y, *, sample_weight=1, tree=None):
        super().learn_one(x, y, sample_weight=sample_weight, tree=tree)
        if self.is_active():
            self.ml_clf.learn_one(x, y)

    def prediction(self, x, *, tree=None):
        current_pred = self.ml_clf.predict_proba_one(x)
        if current_pred:
            return current_pred
        elif self.parent_ml_clf is not None:
            return self.parent_ml_clf.predict_proba_one(x)
        else:
            return {}

    def __repr__(self):
        if not self.stats:
            return ""
        if self.ml_clf.count_ls:
            mls = self.ml_clf.labelsets[
                max(self.ml_clf.count_ls, key=lambda k: self.ml_clf.count_ls[k])
            ]
        else:
            mls = self.parent_ml_clf.labelsets[
                max(self.parent_ml_clf.count_ls, key=lambda k: self.parent_ml_clf.count_ls[k])
            ]
        text = f"Majority LS: {sorted(k for (k,v) in mls.items() if v)}"
        for label, proba in sorted({k: v for (k, v) in self.stats.items() if k != "total"}.items()):
            text += f"\n\tP({label}) = {round_sig_fig(proba/self.stats['total'])}"
        return text


class MLBranch(branch.DTBranch):
    @property
    def n_active_leaves(self):
        return sum(child.n_active_leaves for child in self.children)

    @property
    def n_inactive_leaves(self):
        return sum(child.n_inactive_leaves for child in self.children)
