import copy
import numbers
from river import ensemble, linear_model, neighbors
from river import stats as st
from river.tree.nodes import htc_nodes
from river.utils.random import poisson
from river.tree.utils import round_sig_fig

from .ml_node import MultiLabelAdaptiveNode, MultiLabelNode
from leaf_classifiers import BinaryRelevance, MajorityLabelset


class LeafMultiLabel(MultiLabelNode, htc_nodes.LeafMajorityClass):
    def __init__(self, stats, depth, splitter, ml_clf, parent_ml_clf, **kwargs):
        super().__init__(stats, depth, splitter, **kwargs)
        self.ml_clf = copy.deepcopy(ml_clf)
        self.parent_ml_clf = parent_ml_clf  # Produce predictions if the classifier at leaf is not initialized yet (any instance received)

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

    def calculate_promise(self):
        total_seen = self.stats["total"]
        if total_seen > 0:
            return total_seen - max(
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


class AdaLeafMultiLabel(MultiLabelAdaptiveNode, LeafMultiLabel):
    def __init__(
        self, stats, depth, splitter, ml_clf, parent_ml_clf, drift_detector, rng, **kwargs
    ):
        super().__init__(stats, depth, splitter, ml_clf, parent_ml_clf, **kwargs)
        self.drift_detector = drift_detector
        self.rng = rng
        self._mean_error = st.Mean()

    def learn_one(self, x, y, *, sample_weight=1, tree=None, p_node=None, p_branch=None):
        if tree.bootstrap_sampling:
            # Perform bootstrap-sampling
            k = poisson(rate=1, rng=self.rng)
            if k > 0:
                sample_weight *= k

        # Update the drift detector
        y_pred_proba = self.prediction(x, tree=tree)
        if y_pred_proba:
            y_pred = {
                label: max(y_pred_proba[label], key=y_pred_proba[label].get)
                for label in y_pred_proba
            }
            old_error = self._mean_error.get()
            self.update_drift_detector(y, y_pred)
            # Error is decreasing
            if self.drift_detector.drift_detected and old_error > self._mean_error.get():
                self._mean_error = self._mean_error.clone()
        # Update stats and multi-label classifier
        super().learn_one(x, y, sample_weight=sample_weight, tree=tree)

        weight_seen = self.total_weight
        weight_diff = weight_seen - self.last_split_attempt_at
        if weight_diff >= tree.grace_period:
            if self.depth >= tree.max_depth:
                # Depth-based pre-pruning
                self.deactivate()
                tree._n_inactive_leaves += 1
                tree._n_active_leaves -= 1
            elif self.is_active():
                tree._attempt_to_split(
                    self,
                    p_node,
                    p_branch,
                    drift_detector=tree.drift_detector.clone(),
                    ml_clf=self.ml_clf,
                )
                self.last_split_attempt_at = weight_seen

    def prediction(self, x, *, tree=None):
        dist = super().prediction(x, tree=tree)
        if dist:
            for label, vals in dist.items():
                normalization_factor = sum(vals.values()) * self._mean_error.get() ** 2
                if normalization_factor > 0:
                    for val in vals:
                        dist[label][val] /= normalization_factor
        return dist
        # TODO: naive bayes approach and weighted combination of predictions according to ADWIN for multi-label

    def kill_tree_children(self, tree):
        pass


class MLHATLeaf(MultiLabelAdaptiveNode, htc_nodes.LeafMajorityClass):
    def __init__(self, stats, depth, splitter, drift_detector, rng, is_background, **kwargs):
        self.current_obs = 0
        MultiLabelAdaptiveNode.__init__(self, stats, depth, splitter, **kwargs)
        self.drift_detector = drift_detector
        self._rng = rng
        self._is_background = is_background
        self._mean_error = st.Mean()
        self.mls = MajorityLabelset()
        self.low_card_clf = BinaryRelevance(neighbors.KNNClassifier(window_size=100))
        self.high_card_low_entr_clf = BinaryRelevance(ensemble.SRPClassifier(model=linear_model.LogisticRegression(), n_models=10, subspace_size=0.6, disable_detector="drift", seed=self._rng.randint(0, 100)))
    
    @property
    def total_weight(self):
        return self.current_obs
    
    def calculate_promise(self):
        if self.total_weight > 0 and self.mls.count_ls:
            return self.total_weight - max(self.mls.count_ls.values())
        else:
            return 0
    
    def observed_class_distribution_is_pure(self):
        return len(self.mls.labelsets) <= 1
    
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
    
    def learn_one(self, x, y, *, sample_weight=1, tree=None, p_node=None, p_branch=None):
        self.current_obs += sample_weight
        
        # Update the drift detector
        if p_node is None:
            y_pred = self.prediction_one(x, tree=tree)
            if y_pred:
                old_error = self._mean_error.get()
                self.update_drift_detector(y, y_pred, tree)
                # Error is decreasing
                if self.drift_detector.drift_detected and old_error > self._mean_error.get():
                    self._mean_error = self._mean_error.clone()
        else:
            self._mean_error = p_node._mean_error

        self.update_stats(y, sample_weight)
        if self.is_active():
            self.update_splitters(x, y, sample_weight, tree.nominal_attributes)

        self.mls.learn_one(x, y)
        if self.total_weight < tree.grace_period:
            self.low_card_clf.learn_one(x, y)
        self.high_card_low_entr_clf.learn_one(x, y)

        weight_diff = self.total_weight - self.last_split_attempt_at
        if weight_diff >= tree.grace_period:
            if self.depth >= tree.max_depth:
                # Depth-based pre-pruning
                self.deactivate()
                tree._n_inactive_leaves += 1
                tree._n_active_leaves -= 1
            elif self.is_active():
                tree._attempt_to_split(
                    self,
                    p_node,
                    p_branch,
                    drift_detector=tree.drift_detector.clone(),
                    is_background=self._is_background,
                    x=x,
                    y=y,
                    sample_weight=sample_weight,
                )
                self.last_split_attempt_at = self.total_weight

    def prediction(self, x, *, tree=None):
        if self.observed_class_distribution_is_pure():
            dist = self.mls.predict_proba_one(x)
        elif self.total_weight < tree.grace_period:
            dist = self.low_card_clf.predict_proba_one(x)
        else:
            dist = self.high_card_low_entr_clf.predict_proba_one(x)

        if dist:
            for label, vals in dist.items():
                normalization_factor = sum(vals.values()) * self._mean_error.get() ** 2
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
