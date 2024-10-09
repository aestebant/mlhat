import math
import random
from collections import defaultdict

from river import base
from river.drift import ADWIN
from river.ensemble.bagging import BaggingClassifier
from river.metrics.base import Metric
from river.tree.splitter import GaussianSplitter, Splitter
from river.neighbors import LazySearch
from river.optim.sgd import SGD
from river.utils.random import poisson

from multioutput import BinaryRelevance, LabelCombination
from multioutput.mlht import MultiLabelHoeffdingTree
from metrics.multioutput import HammingLoss
from multioutput.mlhat_nodes.mlhatoc_branch import (
    MLHAToCBranch,
    MLHAToCBranchNomBinary,
    MLHAToCBranchNomMultiway,
    MLHAToCBranchNumBinary,
    MLHAToCBranchNumMultiway,
)
from multioutput.mlhat_nodes.mlhatoc_leaf import MLHAToCLeaf
from neighbors.knn_classifier import KNNClassifier
from linear_model.log_reg import LogisticRegression

class MLHAT(MultiLabelHoeffdingTree):
    _BUILD_ON_WARNING = "on warning"
    _BUILD_ON_DRIFT = "on drift"
    _VALID_BUILD_ALTERNATE_TREE = {_BUILD_ON_WARNING, _BUILD_ON_DRIFT}

    _ADWIN_PER_LABEL = "per label"
    _ADWIN_PER_LABELSET = "per labelset"
    _VALID_ADWIN_STRATEGIES = {_ADWIN_PER_LABEL, _ADWIN_PER_LABELSET}

    def __init__(
        self,
        grace_period: int = 200,
        delta: float = 1.6587784772850855e-7,
        drift_window_threshold: int = 200,
        switch_significance: float = 0.05,
        cardinality_th: int = 750,
        poisson_rate: float = 1.0,

        tau: float = 0.05,
        splitter: Splitter = GaussianSplitter(),

        entropy_th: float = 0.3,
        low_card_clf: base.MultiLabelClassifier = LabelCombination(KNNClassifier(n_neighbors=4, engine=LazySearch(window_size=170))),
        high_card_clf: base.MultiLabelClassifier = BinaryRelevance(BaggingClassifier(LogisticRegression(SGD(0.3)), n_models=10, seed=random.randrange(100))),
        bootstrap_sampling: bool = True,

        drift_detector: base.DriftDetector = ADWIN(),
        drift_method: str = _ADWIN_PER_LABELSET,
        perf_metric: Metric = HammingLoss(),
        alternate_strategy: str = _BUILD_ON_WARNING,

        combined_prediction: bool = True,
        min_branch_fraction: float = 0.01,
        max_share_to_split: float = 0.99,
        nominal_attributes: list = None,
        max_depth: int = None,
        max_size: float = 100,
        memory_estimate_period: int = 1000000,
        stop_mem_management: bool = False,
        remove_poor_attrs: bool = False,
        merit_preprune: bool = True,
        seed: int = 0,
    ):
        super().__init__(
            model=None,
            grace_period=grace_period,
            max_depth=max_depth,
            delta=delta,
            tau=tau,
            nominal_attributes=nominal_attributes,
            splitter=splitter,
            binary_split=True,
            min_branch_fraction=min_branch_fraction,
            max_share_to_split=max_share_to_split,
            max_size=max_size,
            memory_estimate_period=memory_estimate_period,
            stop_mem_management=stop_mem_management,
            remove_poor_attrs=remove_poor_attrs,
            merit_preprune=merit_preprune,
        )
        self.cardinality_th = cardinality_th
        self.entropy_th = entropy_th
        self.low_card_clf = low_card_clf
        self.high_card_clf = high_card_clf
        self.bootstrap_sampling = bootstrap_sampling
        self.poisson_rate = poisson_rate
        self.drift_window_threshold = drift_window_threshold
        self.drift_detector = drift_detector
        if drift_method not in self._VALID_ADWIN_STRATEGIES:
            raise ValueError("Not valid strategy for computing ADWIN", drift_method)
        else:
            self.drift_method = drift_method
        self.perf_metric = perf_metric
        if alternate_strategy not in self._VALID_BUILD_ALTERNATE_TREE:
            raise ValueError("Not valid strategy for building alternate tree", alternate_strategy)
        else:
            self.alternate_strategy = alternate_strategy
        self.switch_significance = switch_significance
        self.combined_prediction = combined_prediction
        self.seed = seed
        self.n_pruned_alternate_trees = 0
        self.n_switched_alternate_trees = 0
        self._rng = random.Random(self.seed)

    def learn_one(self, x, y, sample_weight=1):
        for label, val in y.items():
            self.label_space[label].add(val)

        if self.bootstrap_sampling:
            # Perform bootstrap-sampling
            k = poisson(rate=self.poisson_rate, rng=self._rng)
            if k > 0:
                sample_weight *= k

        self._train_weight_seen_by_model += sample_weight

        if self._root is None:
            self._root = self._new_leaf(is_background=False)
            self._n_active_leaves = 1

        self._root.learn_one(x, y, sample_weight=sample_weight, tree=self)

        if self._train_weight_seen_by_model % self.memory_estimate_period == 0:
            self._estimate_model_size()

        return self

    def predict_proba_one(self, x):
        proba = defaultdict(lambda: dict())
        for label, vals in self.label_space.items():
            for val in vals:
                proba[label][val] = 0.0
        if self._root is not None:
            found_nodes = [self._root]
            if isinstance(self._root, MLHAToCBranch):
                found_nodes = self._root.traverse(x, until_leaf=True)
            # Combine the response of all leaves reached (weighted according concept drift detector)
            if self.combined_prediction:
                for leaf in found_nodes:
                    if not leaf._is_background or leaf.total_weight > self.drift_window_threshold:
                        partial_proba = leaf.prediction(x, tree=self)
                        for label, vals in partial_proba.items():
                            for val, prob in vals.items():
                                try:
                                    proba[label][val] += prob
                                except KeyError:
                                    proba[label][val] = prob
            else:
                for leaf in found_nodes:
                    if not leaf._is_background: # It should only be one
                        partial_proba = leaf.prediction(x, tree=self)
                        for label, vals in partial_proba.items():
                            for val, prob in vals.items():
                                try:
                                    proba[label][val] += prob
                                except KeyError:
                                    proba[label][val] = prob
            for label, vals in proba.items():
                total = sum(vals.values())
                if total > 0:
                    proba[label] = {val: prob / total for val, prob in vals.items()}
                else:
                    proba[label] = {val: 0.0 for val in vals}
        return dict(proba)

    def _attempt_to_split(
        self,
        leaf: MLHAToCLeaf,
        parent: MLHAToCBranch,
        parent_branch: int,
        drift_detector: base.DriftDetector,
        perf_metric: Metric,
        is_background: bool,
        x,
        y,
        sample_weight,
    ):
        if leaf.observed_class_distribution_is_pure():  # type: ignore
            return

        split_criterion = self._new_split_criterion()

        best_split_suggestions = leaf.best_split_suggestions(split_criterion, self)
        best_split_suggestions.sort()
        should_split = False
        if len(best_split_suggestions) < 2:
            should_split = len(best_split_suggestions) > 0
        else:
            num_subsets = len(leaf.pure_clf.count_ls)
            if num_subsets <= 2:
                merit_range = 1.0  # log_2(2) = 1
            else:
                merit_range = math.log2(num_subsets)
            hoeffding_bound = self._hoeffding_bound(merit_range, self.delta, leaf.current_weight)
            best_suggestion = best_split_suggestions[-1]
            second_best_suggestion = best_split_suggestions[-2]
            if (
                best_suggestion.merit - second_best_suggestion.merit > hoeffding_bound
                or hoeffding_bound < self.tau
            ):
                should_split = True
            if self.remove_poor_attrs:
                poor_atts = set()
                # Add any poor attribute to set
                for suggestion in best_split_suggestions:
                    if (
                        suggestion.feature
                        and best_suggestion.merit - suggestion.merit > hoeffding_bound
                    ):
                        poor_atts.add(suggestion.feature)
                for poor_att in poor_atts:
                    leaf.disable_attribute(poor_att)
        if should_split:
            split_decision = best_split_suggestions[-1]
            if split_decision.feature is None:
                # Pre-pruning - null wins
                leaf.deactivate()
            else:
                branch = self._branch_selector(
                    split_decision.numerical_feature, split_decision.multiway_split
                )
                leaves = tuple(
                    self._new_leaf(initial_stats, leaf, leaf._is_background)
                    for initial_stats in split_decision.children_stats  # type: ignore
                )
                new_split = split_decision.assemble(
                    branch,
                    leaf.stats,
                    leaf.depth,
                    *leaves,
                    drift_detector=drift_detector.clone(),
                    perf_metric=perf_metric.clone(),
                    is_background=is_background,
                )
                new_split.learn_one(
                    x,
                    y,
                    sample_weight=sample_weight,
                    tree=self,
                    p_node=parent,
                    p_branch=parent_branch,
                )
                if parent is None:
                    if not is_background:
                        self._root = new_split
                    else:
                        self._root._alternate_tree = new_split
                else:
                    parent.children[parent_branch] = new_split

            # Manage memory
            self._enforce_size_limit()

    def _new_leaf(self, initial_stats=None, parent=None, is_background=False):
        if initial_stats is None:
            initial_stats = {}
        if parent is None:
            depth = 0
        else:
            depth = parent.depth + 1
        return MLHAToCLeaf(
            initial_stats,
            depth,
            self.low_card_clf.clone(),
            self.high_card_clf.clone(),
            self.splitter,
            self.drift_detector.clone(),
            self.perf_metric.clone(),
            is_background,
        )

    def _branch_selector(self, numerical_feature=True, multiway_split=False):
        if numerical_feature and multiway_split:
            return MLHAToCBranchNumMultiway
        elif numerical_feature and not multiway_split:
            return MLHAToCBranchNumBinary
        elif not numerical_feature and multiway_split:
            return MLHAToCBranchNomMultiway
        elif not numerical_feature and not multiway_split:
            return MLHAToCBranchNomBinary

    @property
    def n_current_alternate_trees(self):
        if self._root:
            return self._root.n_current_alternate_trees

    @property
    def summary(self):
        summ = super().summary
        summ.update(
            {
                "n_current_alternate_trees": self.n_current_alternate_trees,
                "n_pruned_alternate_trees": self.n_pruned_alternate_trees,
                "n_switched_alternate_trees": self.n_switched_alternate_trees,
            }
        )
        return summ
