from river.base.ensemble import Ensemble
from river.tree.nodes import htc_nodes
from river.tree.utils import BranchFactory, round_sig_fig

from multioutput.majority import MajorityLabelset
from multioutput.mlhat_nodes.base import MultiLabelAdaptiveNode

import copy
import numbers


class MLHAToCLeaf(MultiLabelAdaptiveNode, htc_nodes.LeafMajorityClass):
    def __init__(
        self,
        stats,
        depth,
        low_card_clf,
        high_card_clf,
        splitter,
        drift_detector,
        perf_metric,
        is_background,
        **kwargs,
    ):
        super().__init__(drift_detector, perf_metric, stats, depth, splitter, **kwargs)
        self.current_weight = 0
        self._is_background = is_background
        self.pure_clf = MajorityLabelset()
        self.entropy = 0
        self.low_card_clf = low_card_clf
        self.high_card_clf = high_card_clf

    @property
    def n_current_alternate_trees(self):
        return 0

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

    def calculate_promise(self):
        if self.current_weight > 0 and self.pure_clf.count_ls:
            return self.current_weight - max(self.pure_clf.count_ls.values())
        else:
            return 0

    def observed_class_distribution_is_pure(self):
        return len(self.pure_clf.labelsets) <= 1

    def best_split_suggestions(self, criterion, tree) -> list[BranchFactory]:
        maj_class = max([count for label, count in self.stats.items() if label != 'total'])
        # Only perform split attempts when the majority class does not dominate
        # the amount of observed instances
        if maj_class and maj_class / self.total_weight > tree.max_share_to_split:
            return [BranchFactory()]

        best_suggestions = []
        pre_split_dist = self.stats
        if tree.merit_preprune:
            # Add null split as an option
            null_split = BranchFactory()
            best_suggestions.append(null_split)
        for att_id, splitter in self.splitters.items():
            best_suggestion = splitter.best_evaluated_split_suggestion(
                criterion, pre_split_dist, att_id, tree.binary_split
            )
            best_suggestions.append(best_suggestion)
        return best_suggestions

    def update_splitters(self, x, y, sample_weight, nominal_attributes):
        for att_id, att_val in self._iter_features(x):
            if att_id in self._disabled_attrs:
                continue
            if att_id in self.splitters.keys():
                splitter = self.splitters[att_id]
            else:
                if (
                    nominal_attributes is not None and att_id in nominal_attributes
                ) or not isinstance(att_val, numbers.Number):
                    splitter = self.new_nominal_splitter()
                else:
                    splitter = copy.deepcopy(self.splitter)
                self.splitters[att_id] = splitter
            empty = True
            for label, val in y.items():
                if val:
                    empty = False
                    splitter.update(att_val, label, sample_weight)
            if empty:
                splitter.update(att_val, "empty", sample_weight)
            splitter.update(att_val, "total", sample_weight)

    def learn_one(self, x, y, *, sample_weight=1, tree=None, p_node=None, p_branch=None):
        # Update the drift detector
        if p_node is None:
            y_pred = self.prediction_one(x, tree=tree)
            if y_pred:
                old_perf = self._mean_perf.get()
                self.update_drift_detector(y, y_pred, sample_weight, tree)
                self._post_drift_weight += sample_weight
                if self.drift_detector.drift_detected:
                    curr_perf = self._mean_perf.get()
                    # Error is decreasing
                    if (self._mean_perf.bigger_is_better and curr_perf > old_perf) or (not self._mean_perf.bigger_is_better and curr_perf < old_perf):
                        self._mean_perf = self._mean_perf.clone()
                        self._post_drift_weight = 0
        else:
            self._mean_perf = p_node._mean_perf
            self._post_drift_weight = p_node._post_drift_weight

        self.current_weight += sample_weight
        self.update_stats(y, sample_weight)
        if self.is_active():
            self.update_splitters(x, y, sample_weight, tree.nominal_attributes)

        self.pure_clf.learn_one(x, y, sample_weight=sample_weight)
        # I guess in ensembles the inst weight shouldn't spread to this level -> they have their own poisson distribution
        if isinstance(self.low_card_clf, Ensemble):
            self.low_card_clf.learn_one(x, y, sample_weight=1)
        else:
            self.low_card_clf.learn_one(x, y, sample_weight=sample_weight)
        if isinstance(self.high_card_clf, Ensemble):
            self.high_card_clf.learn_one(x, y, sample_weight=1)
        else:
            self.high_card_clf.learn_one(x, y, sample_weight=sample_weight)

        weight_diff = self.current_weight - self.last_split_attempt_at
        if weight_diff >= tree.grace_period:
            if self.depth >= tree.max_depth:
                # Depth-based pre-pruning
                self.deactivate()
            elif self.is_active():
                tree._attempt_to_split(
                    self,
                    p_node,
                    p_branch,
                    drift_detector=tree.drift_detector.clone(),
                    perf_metric=tree.perf_metric.clone(),
                    is_background=self._is_background,
                    x=x,
                    y=y,
                    sample_weight=sample_weight,
                )
                self.last_split_attempt_at = self.current_weight

    def prediction(self, x, *, tree=None):
        if self.observed_class_distribution_is_pure():
            dist = self.pure_clf.predict_proba_one(x)
        else:
            compute_entropy = tree._new_split_criterion()
            entropy = compute_entropy.compute_entropy(self.stats)
            if self.current_weight < tree.cardinality_th or (self.current_weight >= tree.cardinality_th and entropy < tree.entropy_th):
                dist = self.low_card_clf.predict_proba_one(x)
            else:
                dist = self.high_card_clf.predict_proba_one(x)

        if dist:
            for label, vals in dist.items():
                if self._mean_perf.bigger_is_better:
                    normalization_factor = sum(vals.values()) * (1-self._mean_perf.get()) ** 2
                else:
                    normalization_factor = sum(vals.values()) * self._mean_perf.get() ** 2
                if normalization_factor > 0:
                    for val in vals:
                        dist[label][val] /= normalization_factor
        return dist

    def prediction_one(self, x, *, tree=None):
        y_pred_proba = self.prediction(x, tree=tree)
        if y_pred_proba:
            y_pred = {
                label: max(y_pred_proba[label], key=y_pred_proba[label].get)
                for label in y_pred_proba
            }
        else:
            y_pred = {}
        return y_pred

    def kill_tree_children(self, tree):
        pass

    def __repr__(self):
        if not self.stats:
            return ""
        mls = self.pure_clf.predict_one(None)
        if mls:
            to_show = [k for k, v in mls.items() if v]
        else:
            to_show = None
        text = f"Majority LS: {to_show}\n"

        if self.observed_class_distribution_is_pure():
            text += "Classifier: majority"
        else: text += "Classifier: adaptive"

        for label, proba in sorted({k: v for (k, v) in self.stats.items() if k != "total"}.items()):
            text += f"\n\tP({label}) = {round_sig_fig(proba/self.stats['total'])}"

        return text
