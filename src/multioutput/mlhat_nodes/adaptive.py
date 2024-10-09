import math
import typing
from river import stats as st
from river.multioutput.mlhat_nodes.base import LeafMultiLabel, MultiLabelAdaptiveNode
from river.tree.nodes import branch
from river.utils.random import poisson


class AdaLeafMultiLabel(MultiLabelAdaptiveNode, LeafMultiLabel):
    def __init__(
        self, stats, depth, splitter, ml_clf, parent_ml_clf, drift_detector, rng, **kwargs
    ):
        super().__init__(stats, depth, splitter, ml_clf, parent_ml_clf, **kwargs)
        self.drift_detector = drift_detector
        self.rng = rng
        self._mean_error = st.Mean()

    @property
    def n_current_alternate_trees(self):
        return 0

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


class AdaBranchMultiLabel(MultiLabelAdaptiveNode, branch.DTBranch):
    def __init__(self, stats, *children, drift_detector, ml_clf, **attributes):
        super().__init__(stats, *children, **attributes)
        self.ml_clf = ml_clf
        self.drift_detector = drift_detector
        self._alternate_tree = None
        self._mean_error = st.Mean()

    @property
    def n_nodes(self):
        nodes = 1 + sum(child.n_nodes for child in self.children)
        if self._alternate_tree:
            nodes += self._alternate_tree.n_nodes
        return nodes
    
    @property
    def n_current_alternate_trees(self):
        if self._alternate_tree:
            return 1 + sum(child.n_current_alternate_trees for child in self.children)
        else:
            return sum(child.n_current_alternate_trees for child in self.children)

    def traverse(self, x, until_leaf=True):
        found_nodes: typing.List[AdaLeafMultiLabel] = []
        for node in self.walk(x, until_leaf=until_leaf):
            if isinstance(node, AdaBranchMultiLabel) and node._alternate_tree:
                if isinstance(node._alternate_tree, AdaBranchMultiLabel):
                    found_nodes.append(node._alternate_tree.traverse(x, until_leaf=until_leaf))
                else:
                    found_nodes.append(node._alternate_tree)
        found_nodes.append(node)  # TODO falla indentación?
        return found_nodes

    def iter_leaves(self):
        for child in self.children:
            yield from child.iter_leaves()

            if isinstance(child, AdaBranchMultiLabel) and child._alternate_tree:
                yield from child._alternate_tree.iter_leaves()

    def learn_one(self, x, y, *, sample_weight=1.0, tree=None, p_node=None, p_branch=None):

        # Update stats as traverse tree to improve predictions
        # (in case split nodes are used to provide responses)
        self.update_stats(y, sample_weight)

        # Update the drift detector
        old_error = self._mean_error.get()
        leaf = super().traverse(x, until_leaf=True)
        y_pred_proba = leaf.prediction(x, tree=tree)
        if y_pred_proba:
            y_pred = {
                label: max(y_pred_proba[label], key=y_pred_proba[label].get)
                for label in y_pred_proba
            }
            warning = self.update_drift_detector(y, y_pred)
            new_error = self._mean_error.get()

            if tree.alternate_strategy == tree._BUILD_ON_WARNING:
                error_change = warning
            elif tree.alternate_strategy == tree._BUILD_ON_DRIFT:
                error_change = self.drift_detector.drift_detected

            # Error is decreasing
            if error_change and old_error > new_error:
                self._mean_error = self._mean_error.clone()
            # Build a new alternate tree in background
            elif error_change and new_error > old_error:
                self._mean_error = self._mean_error.clone()
                self._alternate_tree = tree._new_leaf(parent=self)
                self._alternate_tree.depth -= 1  # To ensure we do not skip a tree level

        # Replace an alternate tree
        if self._alternate_tree:
            alt_n_obs = self._alternate_tree._mean_error.n
            n_obs = self._mean_error.n
            if alt_n_obs > tree.drift_window_threshold and n_obs > tree.drift_window_threshold:
                old_error_rate = self._mean_error.get()
                alt_error_rate = self._alternate_tree._mean_error.get()

                n = 1 / alt_n_obs + 1 / n_obs
                bound = math.sqrt(
                    2
                    * old_error_rate
                    * (1 - old_error_rate)
                    * math.log(2 / tree.switch_significance)
                    * n
                )
                if old_error_rate - alt_error_rate > bound:
                    self.kill_tree_children(tree)
                    if p_node is not None:
                        p_node.children[p_branch] = self._alternate_tree
                        self._alternate_tree = None
                    else:
                        tree._root = tree._root._alternate_tree
                    tree._n_switch_alternate_trees += 1
                elif alt_error_rate - old_error_rate > bound:
                    if isinstance(self._alternate_tree, AdaBranchMultiLabel):
                        self._alternate_tree.kill_tree_children(tree)
                    self._alternate_tree = None
                    tree._n_pruned_alternate_trees += 1

        # Learning in alternate tree and child nodes
        if self._alternate_tree:
            self._alternate_tree.learn_one(
                x, y, sample_weight=sample_weight, tree=tree, p_node=p_node, p_branch=p_branch
            )
        try:
            child = self.next(x)
        except KeyError:
            child = None
        if child:
            child.learn_one(
                x,
                y,
                sample_weight=sample_weight,
                tree=tree,
                p_node=self,
                p_branch=self.branch_no(x),
            )
        else:
            # Instance contains a categorical value previously unseen by the split node
            if self.max_branches() == -1 and self.feature in x:
                leaf = tree._new_leaf(parent=self)
                self.add_child(x[self.feature], leaf)
                tree._n_active_leaves += 1
                leaf.learn_one(
                    x,
                    y,
                    sample_weight=sample_weight,
                    tree=tree,
                    p_node=self,
                    p_branch=self.branch_no(x),
                )
            # Split feature missing so the instance is passed to the most traversed path
            else:
                child_id, child = self.most_common_path()
                child.learn_one(
                    x, y, sample_weight=sample_weight, tree=tree, p_node=self, p_branch=child_id
                )

    def kill_tree_children(self, tree):
        for child in self.children:
            if isinstance(child, branch.DTBranch):
                if child._alternate_tree:
                    child._alternate_tree.kill_tree_children(tree)
                    tree._n_pruned_alternate_trees += 1
                    child._alternate_tree = None
                child.kill_tree_children(tree)


class AdaNomBinaryBranchML(AdaBranchMultiLabel, branch.NominalBinaryBranch):
    def __init__(self, stats, feature, value, depth, left, right, **attributes):
        super().__init__(stats, feature, value, depth, left, right, **attributes)


class AdaNumBinaryBranchML(AdaBranchMultiLabel, branch.NumericBinaryBranch):
    def __init__(self, stats, feature, threshold, depth, left, right, **attributes):
        super().__init__(stats, feature, threshold, depth, left, right, **attributes)


class AdaNomMultiwayBranchML(AdaBranchMultiLabel, branch.NominalMultiwayBranch):
    def __init__(self, stats, feature, feature_values, depth, *children, **attributes):
        super().__init__(stats, feature, feature_values, depth, *children, **attributes)


class AdaNumMultiwayBranchML(AdaBranchMultiLabel, branch.NumericMultiwayBranch):
    def __init__(self, stats, feature, radius_and_slots, depth, *children, **attributes):
        super().__init__(stats, feature, radius_and_slots, depth, *children, **attributes)
