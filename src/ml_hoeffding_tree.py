import math
import random
from collections import defaultdict
import typing

from river import base
from river.drift import ADWIN
from river.tree import HoeffdingTreeClassifier
from river.tree.hoeffding_tree import HoeffdingTree
from river.tree.splitter import GaussianSplitter, Splitter
from river.utils.random import poisson

from nodes import MLHATBranch, MLHATBranchNomBinary, MLHATBranchNomMultiway, MLHATBranchNumBinary, MLHATBranchNumMultiway, MLHATLeaf
from split_criterion import MultiLabelSplitCriterion


class MultiLabelHoeffdingAdaptiveTree(HoeffdingTreeClassifier, base.MultiLabelClassifier):
    _BUILD_ON_WARNING = "on warning"
    _BUILD_ON_DRIFT = "on drift"
    _VALID_BUILD_ALTERNATE_TREE = {_BUILD_ON_WARNING, _BUILD_ON_DRIFT}

    _ADWIN_PER_LABEL = "adwin per label"
    _ADWIN_SUBSET_ACC = "adwin based on subset acc"
    _ADWIN_HAMMING_LOSS = "adwin based on hamming loss"
    _VALID_ADWIN_STRATEGIES = {_ADWIN_PER_LABEL, _ADWIN_SUBSET_ACC, _ADWIN_HAMMING_LOSS}

    def __init__(
        self,
        grace_period: int = 200,
        max_depth: int = None,
        delta: float = 1e-7,
        tau: float = 0.05,
        nominal_attributes: list = None,
        splitter: Splitter = GaussianSplitter(),
        bootstrap_sampling: bool = True,
        poisson_rate: float = 1.0,
        drift_window_threshold: int = 100,
        drift_detector: base.DriftDetector = ADWIN(),
        adwin_metric: str = _ADWIN_PER_LABEL,
        alternate_strategy: str = _BUILD_ON_WARNING,
        switch_significance: float = 0.05,
        binary_split: bool = False,
        max_size: float = 100,
        memory_estimate_period: int = 1000000,
        stop_mem_management: bool = False,
        remove_poor_attrs: bool = False,
        merit_preprune: bool = True,
        seed: int = 0,
    ):
        super().__init__(
            grace_period=grace_period,
            max_depth=max_depth,
            split_criterion="",
            delta=delta,
            tau=tau,
            leaf_prediction="",
            nb_threshold=0,
            nominal_attributes=nominal_attributes,
            splitter=splitter,
            binary_split=binary_split,
            max_size=max_size,
            memory_estimate_period=memory_estimate_period,
            stop_mem_management=stop_mem_management,
            remove_poor_attrs=remove_poor_attrs,
            merit_preprune=merit_preprune,
        )
        self.label_space: typing.DefaultDict[str, typing.Set[base.typing.ClfTarget]] = defaultdict(lambda: set())
        self.bootstrap_sampling = bootstrap_sampling
        self.poisson_rate = poisson_rate
        self.drift_window_threshold = drift_window_threshold
        self.drift_detector = drift_detector
        if adwin_metric not in self._VALID_ADWIN_STRATEGIES:
            raise ValueError("Not valid strategy for computing ADWIN", adwin_metric)
        else:
            self.adwin_metric = adwin_metric
        if alternate_strategy not in self._VALID_BUILD_ALTERNATE_TREE:
            raise ValueError("Not valid strategy for building alternate tree", alternate_strategy)
        else:
            self.alternate_strategy = alternate_strategy
        self.switch_significance = switch_significance
        self.seed = seed
        self._n_alternate_trees = 0
        self._n_pruned_alternate_trees = 0
        self._n_switch_alternate_trees = 0
        self._rng = random.Random(self.seed)

    @HoeffdingTree.split_criterion.setter  # type: ignore
    def split_criterion(self, split_criterion):
        pass

    @HoeffdingTree.leaf_prediction.setter  # type: ignore
    def leaf_prediction(self, leaf_prediction):
        pass

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
            if isinstance(self._root, MLHATBranch):
                found_nodes = self._root.traverse(x, until_leaf=True)
            # Combine the response of all leaves reached (weighted according concept drift detector)
            for leaf in found_nodes:
                if not leaf._is_background or leaf._mean_error.n > self.drift_window_threshold:
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
    
    def predict_one(self, x: dict):
        y_pred = self.predict_proba_one(x)
        return {label: max(y_pred[label], key=y_pred[label].get) for label in y_pred}

    def draw(self, max_depth: int = None):
        print(self.label_space.keys())
        self.classes = set(self.label_space.keys())
        return super().draw(max_depth)
    
    def _new_split_criterion(self):
        return MultiLabelSplitCriterion()

    def _attempt_to_split(
        self,
        leaf: MLHATLeaf,
        parent: MLHATBranch,
        parent_branch: int,
        drift_detector: base.DriftDetector,
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
            num_subsets = len(leaf.mls.count_ls)
            if num_subsets <= 2:
                merit_range = 1.0  # log_2(2) = 1
            else:
                merit_range = math.log2(num_subsets)
            hoeffding_bound = self._hoeffding_bound(merit_range, self.delta, leaf.total_weight)
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
                self._n_inactive_leaves += 1
                self._n_active_leaves -= 1
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
                    drift_detector=drift_detector,
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

                self._n_active_leaves -= 1
                self._n_active_leaves += len(leaves)
                if parent is None:
                    self._root = new_split
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
        return MLHATLeaf(
            initial_stats,
            depth,
            self.splitter,
            self.drift_detector.clone(),
            self._rng,
            is_background,
        )

    def _branch_selector(self, numerical_feature=True, multiway_split=False):
        if numerical_feature and multiway_split:
            return MLHATBranchNumMultiway
        elif numerical_feature and not multiway_split:
            return MLHATBranchNumBinary
        elif not numerical_feature and multiway_split:
            return MLHATBranchNomMultiway
        elif not numerical_feature and not multiway_split:
            return MLHATBranchNomBinary

    @property
    def summary(self):
        summ = super().summary
        summ.update(
            {
                "n_alternate_trees": self._n_alternate_trees,
                "n_pruned_alternate_trees": self._n_pruned_alternate_trees,
                "n_switch_alternate_trees": self._n_switch_alternate_trees,
            }
        )
        return summ

