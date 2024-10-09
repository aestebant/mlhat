import math
import typing
from collections import defaultdict

from river import base
from river.tree import HoeffdingTreeClassifier, split_criterion, splitter
from river.tree.hoeffding_tree import HoeffdingTree
from river.tree.nodes import branch

from multioutput.majority import MajorityLabelset
from multioutput.mlhat_nodes.base import LeafMultiLabel, MLBranch


class MultiLabelSplitCriterion(split_criterion.InfoGainSplitCriterion):
    def merit_of_split(self, pre_split_dist, post_split_dist):
        aux_post = [
            {k: v for (k, v) in subdist.items() if k != "total"} for subdist in post_split_dist
        ]
        if self.num_subsets_greater_than_frac(aux_post, self.min_branch_fraction) < 2:
            return -math.inf
        return self.compute_entropy(pre_split_dist) - self.compute_entropy(post_split_dist)

    @staticmethod
    def range_of_merit(pre_split_dist):
        num_classes = len([k for k in pre_split_dist if k != "total"])
        if num_classes <= 2:
            return 1.0  # log_2(2) = 1.0
        return math.log2(num_classes)

    def compute_entropy(self, dist):
        if isinstance(dist, dict):
            return self._compute_entropy_single(dist)
        elif isinstance(dist, list):
            return self._compute_entropy_comp(dist)

    def _compute_entropy_single(self, dist: dict) -> float:
        if len(dist) == 2:  # a label and the 'total' count
            return 0.0
        entropy_ml = 0.0
        # Sometimes 'total' can be less than other labels due to approximation errors, so we use it or the highest
        total = max(dist.values())
        for label_count in [v for (k, v) in dist.items() if k != "total" and v > 0.0]:
            p_l = label_count / total
            if p_l == 1:
                continue
            q_l = 1 - p_l
            entropy_ml -= p_l * math.log2(p_l) + q_l * math.log2(q_l)
        return entropy_ml

    def _compute_entropy_comp(self, dists: list) -> float:
        entropy = 0.0
        total = 0.0
        for dist in dists:
            entropy += dist["total"] * self._compute_entropy_single(dist)
            total += dist["total"]
        entropy /= total
        return entropy


class MultiLabelHoeffdingTree(HoeffdingTreeClassifier, base.MultiLabelClassifier):
    def __init__(
        self,
        model: base.MultiLabelClassifier = MajorityLabelset(),
        grace_period: int = 200,
        max_depth: int = None,
        delta: float = 1e-7,
        tau: float = 0.05,
        nominal_attributes: list = None,
        splitter: splitter.Splitter = splitter.GaussianSplitter(),
        binary_split: bool = False,
        min_branch_fraction: float = 0.01,
        max_share_to_split: float = 0.99,
        max_size: float = 100,
        memory_estimate_period: int = 1000000,
        stop_mem_management: bool = False,
        remove_poor_attrs: bool = False,
        merit_preprune: bool = True,
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
            min_branch_fraction=min_branch_fraction,
            max_share_to_split=max_share_to_split,
            max_size=max_size,
            memory_estimate_period=memory_estimate_period,
            stop_mem_management=stop_mem_management,
            remove_poor_attrs=remove_poor_attrs,
            merit_preprune=merit_preprune,
        )

        self.label_space: typing.DefaultDict[str, typing.Set[base.typing.ClfTarget]] = defaultdict(
            lambda: set()
        )
        self.model = model

    @HoeffdingTree.split_criterion.setter  # type: ignore
    def split_criterion(self, split_criterion):
        pass

    @HoeffdingTree.leaf_prediction.setter  # type: ignore
    def leaf_prediction(self, leaf_prediction):
        pass

    @property
    def n_active_leaves(self):
        if self._root:
            return self._root.n_active_leaves

    @property
    def n_inactive_leaves(self):
        if self._root:
            return self._root.n_inactive_leaves

    def _new_leaf(self, initial_stats=None, parent=None):
        if initial_stats is None:
            initial_stats = {}
        if parent is None:
            depth = 0
            p_model = None
        else:
            depth = parent.depth + 1
            p_model = parent.ml_clf
        return LeafMultiLabel(initial_stats, depth, self.splitter, self.model, p_model)

    def _new_split_criterion(self):
        return MultiLabelSplitCriterion(self.min_branch_fraction)

    def _branch_selector(self, numerical_feature=True, multiway_split=False):
        if numerical_feature and multiway_split:
            return MLHTBranchNumMultiway
        elif numerical_feature and not multiway_split:
            return MLHTBranchNumBinary
        elif not numerical_feature and multiway_split:
            return MLHTBranchNomMultiway
        elif not numerical_feature and not multiway_split:
            return MLHTBranchNomBinary

    def learn_one(self, x, y, sample_weight=1):
        for label, val in y.items():
            self.label_space[label].add(val)

        self._train_weight_seen_by_model += sample_weight

        if self._root is None:
            self._root = self._new_leaf()

        p_node = None
        node = None
        if isinstance(self._root, branch.DTBranch):
            path = iter(self._root.walk(x, until_leaf=False))
            while True:
                aux = next(path, None)
                if aux is None:
                    break
                p_node = node
                node = aux
        else:
            node = self._root

        if isinstance(node, LeafMultiLabel):
            node.learn_one(x, y, sample_weight=sample_weight, tree=self)
            if self._growth_allowed and node.is_active():
                if node.depth >= self.max_depth:  # Max depth reached
                    node.deactivate()
                else:
                    weight_seen = node.total_weight
                    weight_diff = weight_seen - node.last_split_attempt_at
                    if weight_diff >= self.grace_period:
                        p_branch = (
                            p_node.branch_no(x) if isinstance(p_node, branch.DTBranch) else None
                        )
                        self._attempt_to_split(node, p_node, p_branch)
                        node.last_split_attempt_at = weight_seen
        else:
            while True:
                # Instance contains a categorical value previously unseen by the split node
                if node.max_branches() == -1 and node.feature in x:
                    leaf = self._new_leaf(parent=node)
                    node.add_child(x[node.feature], leaf)
                    node = leaf
                # Split feature missing so the instance is passed to the most traversed path
                else:
                    _, node = node.most_common_path()
                    if isinstance(node, branch.DTBranch):
                        node = node.traverse(x, until_leaf=False)
                # Once a leaf is reached, the traversal can stop
                if isinstance(node, LeafMultiLabel):
                    break
            node.learn_one(x, y, sample_weight=sample_weight, tree=self)

        if self._train_weight_seen_by_model % self.memory_estimate_period == 0:
            self._estimate_model_size()

    def predict_proba_one(self, x):
        proba = defaultdict(lambda: dict())
        for label, vals in self.label_space.items():
            for val in vals:
                proba[label][val] = 0.0
        if self._root is not None:
            if isinstance(self._root, branch.DTBranch):
                leaf = self._root.traverse(x, until_leaf=True)
            else:
                leaf = self._root
            partial_proba = leaf.prediction(x, tree=self)
            for label, vals in partial_proba.items():
                proba[label].update(vals)
        return dict(proba)

    def predict_one(self, x: dict):
        y_pred = self.predict_proba_one(x)
        return {label: max(y_pred[label], key=y_pred[label].get) for label in y_pred}

    def draw(self, max_depth: int = None):
        self.classes = set(self.label_space.keys())
        return super().draw(max_depth)


class MLHTBranchNomBinary(MLBranch, branch.NominalBinaryBranch):
    def __init__(self, stats, feature, value, depth, left, right, **attributes):
        super().__init__(stats, feature, value, depth, left, right, **attributes)


class MLHTBranchNumBinary(MLBranch, branch.NumericBinaryBranch):
    def __init__(self, stats, feature, threshold, depth, left, right, **attributes):
        super().__init__(stats, feature, threshold, depth, left, right, **attributes)


class MLHTBranchNomMultiway(MLBranch, branch.NominalMultiwayBranch):
    def __init__(self, stats, feature, feature_values, depth, *children, **attributes):
        super().__init__(stats, feature, feature_values, depth, *children, **attributes)


class MLHTBranchNumMultiway(MLBranch, branch.NumericMultiwayBranch):
    def __init__(self, stats, feature, radius_and_slots, depth, *children, **attributes):
        super().__init__(stats, feature, radius_and_slots, depth, *children, **attributes)
