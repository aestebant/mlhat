import math
import typing
from river import stats as st
from river.tree.nodes import branch

from .ml_leaves import AdaLeafMultiLabel
from .ml_node import MultiLabelAdaptiveNode


class MLHAToCBranch(MultiLabelAdaptiveNode, branch.DTBranch):
    def __init__(self, stats, *children, drift_detector, is_background, **attributes):
        super().__init__(stats, *children, **attributes)
        self.drift_detector = drift_detector
        self._is_background = is_background
        self._alternate_tree = None
        self._mean_error = st.Mean()

    def traverse(self, x, until_leaf=True):
        found_nodes: typing.List[AdaLeafMultiLabel] = []
        for node in self.walk(x, until_leaf=until_leaf):
            if isinstance(node, MLHAToCBranch) and node._alternate_tree:
                if isinstance(node._alternate_tree, MLHAToCBranch):
                    found_nodes.append(node._alternate_tree.traverse(x, until_leaf=until_leaf))
                else:
                    found_nodes.append(node._alternate_tree)
        found_nodes.append(node)
        return found_nodes

    def iter_leaves(self):
        for child in self.children:
            yield from child.iter_leaves()
            if isinstance(child, MLHAToCBranch) and child._alternate_tree:
                yield from child._alternate_tree.iter_leaves()

    def manage_alternate_tree(self, x, y, tree, p_node, p_branch):
        # Update the drift detector
        old_error = self._mean_error.get()
        leaf = super().traverse(x, until_leaf=True)
        y_pred = leaf.prediction_one(x, tree=tree)
        if not y_pred:
            return

        warning = self.update_drift_detector(y, y_pred, tree)
        new_error = self._mean_error.get()

        # Error is decreasing
        if self.drift_detector.drift_detected and old_error > new_error:
            self._mean_error = self._mean_error.clone()
            return

        # Build a new alternate tree in background
        if tree.alternate_strategy == tree._BUILD_ON_WARNING:
            error_change = warning
        elif tree.alternate_strategy == tree._BUILD_ON_DRIFT:
            error_change = self.drift_detector.drift_detected
        if error_change and not self._alternate_tree and not self._is_background:
            self._mean_error = self._mean_error.clone()
            self._alternate_tree = tree._new_leaf(parent=self, is_background=True)
            self._alternate_tree.depth -= 1  # To ensure we do not skip a tree level
            tree._n_alternate_trees += 1
            return

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
                # Move an alternate tree to frontground
                if old_error_rate - alt_error_rate > bound:
                    tree._n_active_leaves -= self.n_leaves
                    tree._n_active_leaves += self._alternate_tree.n_leaves

                    self._alternate_tree.move_to_frontground()
                    self.kill_tree_children(tree)
                    if p_node:
                        p_node.children[p_branch] = self._alternate_tree
                        self._alternate_tree = None
                    else:
                        tree._root = tree._root._alternate_tree
                    tree._n_switch_alternate_trees += 1
                # Prune an alternate tree
                elif alt_error_rate - old_error_rate > bound:
                    if isinstance(self._alternate_tree, MLHAToCBranch):
                        self._alternate_tree.kill_tree_children(tree)
                    self._alternate_tree = None
                    tree._n_pruned_alternate_trees += 1

    def learn_one(self, x, y, *, sample_weight=1.0, tree=None, p_node=None, p_branch=None):
        # Update stats as traverse tree to improve predictions
        # (in case split nodes are used to provide responses)
        self.update_stats(y, sample_weight)

        self.manage_alternate_tree(x, y, tree, p_node, p_branch)

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
                leaf = tree._new_leaf(parent=self, is_background=self._is_background)
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
            elif child.is_active():
                tree._n_active_leaves -= 1
            else:
                tree._n_inactive_leaves -= 1

    def move_to_frontground(self):
        super().move_to_frontground()
        for child in self.children:
            child.move_to_frontground()


class MLHAToCBranchNomBinary(MLHAToCBranch, branch.NominalBinaryBranch):
    def __init__(self, stats, feature, value, depth, left, right, **kwargs):
        super().__init__(stats, feature, value, depth, left, right, **kwargs)


class MLHAToCBranchNumBinary(MLHAToCBranch, branch.NumericBinaryBranch):
    def __init__(self, stats, feature, threshold, depth, left, right, **kwargs):
        super().__init__(stats, feature, threshold, depth, left, right, **kwargs)


class MLHAToCBranchNomMultiway(MLHAToCBranch, branch.NominalMultiwayBranch):
    def __init__(self, stats, feature, feature_values, depth, *children, **kwargs):
        super().__init__(stats, feature, feature_values, depth, *children, **kwargs)


class MLHAToCBranchNumMultiway(MLHAToCBranch, branch.NumericMultiwayBranch):
    def __init__(self, stats, feature, radius_and_slots, depth, *children, **kwargs):
        super().__init__(stats, feature, radius_and_slots, depth, *children, **kwargs)