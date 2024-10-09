from .base import MultiLabelAdaptiveNode, MLBranch
from .mlhatoc_leaf import MLHAToCLeaf
from river.tree.base import Leaf
from river.tree.nodes import branch
import math
import typing
import pandas as pd
from queue import Queue
from collections import defaultdict


class MLHAToCBranch(MultiLabelAdaptiveNode, MLBranch):
    def __init__(self, stats, *children,  drift_detector, perf_metric, is_background, **kwargs):
        super().__init__(drift_detector, perf_metric, stats, *children, **kwargs)
        self._is_background = is_background
        self._alternate_tree:MultiLabelAdaptiveNode = None

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
        found_nodes: typing.List[MLHAToCLeaf] = []
        for node in self.walk(x, until_leaf=until_leaf):
            if isinstance(node, MLHAToCBranch) and node._alternate_tree:
                if isinstance(node._alternate_tree, MLHAToCBranch):
                    found_nodes += node._alternate_tree.traverse(x, until_leaf=until_leaf)
                else:
                    found_nodes.append(node._alternate_tree)
        found_nodes.append(node)
        return found_nodes

    def iter_leaves(self):
        for child in self.children:
            yield from child.iter_leaves()
            if isinstance(child, MLHAToCBranch) and child._alternate_tree:
                yield from child._alternate_tree.iter_leaves()

    def manage_alternate_tree(self, x, y, sample_weight, tree, p_node, p_branch):
        # Update the drift detector
        old_perf = self._mean_perf.get()
        leaf = super().traverse(x, until_leaf=True)
        y_pred = leaf.prediction_one(x, tree=tree)
        if not y_pred:
            return
        self.update_drift_detector(y, y_pred, sample_weight, tree)
        curr_perf = self._mean_perf.get()

        # Build a new alternate tree in background
        if not self._alternate_tree and not self._is_background:
            if (tree.alternate_strategy == tree._BUILD_ON_WARNING and self.warning) or (tree.alternate_strategy == tree._BUILD_ON_DRIFT and self.drift_detector.drift_detected):
                self._alternate_tree = tree._new_leaf(parent=self, is_background=True)
                self._alternate_tree.depth -= 1  # To ensure we do not skip a tree level

        # Manage existing alternate tree
        if self._alternate_tree and self._alternate_tree.total_weight > tree.drift_window_threshold:
            if self._post_drift_weight > tree.drift_window_threshold:
                n = 1 / self._alternate_tree.total_weight + 1 / self._post_drift_weight
                bound = math.sqrt(
                    2
                    * old_perf
                    * (1 - old_perf)
                    * math.log(2 / tree.switch_significance)
                    * n
                )
                alt_error = self._alternate_tree._mean_perf.get()
                alt_tree_to_front = False
                prune_alt_tree = False
                if self._mean_perf.bigger_is_better:
                    if alt_error - old_perf > bound:
                        alt_tree_to_front = True
                    elif old_perf - alt_error > bound:
                        prune_alt_tree = True
                else:
                    if alt_error - old_perf > bound:
                        prune_alt_tree = True
                    elif old_perf - alt_error > bound:
                        alt_tree_to_front = True
                # Move alternate tree to frontground
                if alt_tree_to_front:
                    self._alternate_tree.move_to_frontground()
                    self.kill_tree_children(tree)
                    if p_node:
                        p_node.children[p_branch] = self._alternate_tree
                        self._alternate_tree = None
                    else:
                        tree._root = tree._root._alternate_tree
                    tree.n_switched_alternate_trees += 1
                # Prune alternate tree
                elif prune_alt_tree:
                    if isinstance(self._alternate_tree, MLHAToCBranch):
                        self._alternate_tree.kill_tree_children(tree)
                    self._alternate_tree = None
                    tree.n_pruned_alternate_trees += 1

            # Error is decreasing in main tree
            if self.drift_detector.drift_detected:
                if (self._mean_perf.bigger_is_better and curr_perf > old_perf) or (not self._mean_perf.bigger_is_better and curr_perf < old_perf):
                    self._mean_perf = self._mean_perf.clone()
                    self._post_drift_weight = 0
                    # TODO: should we prune the alt tree?

    def learn_one(self, x, y, *, sample_weight=1.0, tree=None, p_node=None, p_branch=None):
        # Update stats as traverse tree to improve predictions
        # (in case split nodes are used to provide responses)
        self.update_stats(y, sample_weight)
        self._post_drift_weight += sample_weight

        self.manage_alternate_tree(x, y, sample_weight, tree, p_node, p_branch)

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
            if isinstance(child, MLHAToCBranch):
                if child._alternate_tree:
                    child._alternate_tree.kill_tree_children(tree)
                    tree.n_pruned_alternate_trees += 1
                    child._alternate_tree = None
                child.kill_tree_children(tree)

    def move_to_frontground(self):
        super().move_to_frontground()
        for child in self.children:
            child.move_to_frontground()

    def to_dataframe(self) -> pd.DataFrame:
        """Build a DataFrame containing one record for each node."""
        node_ids: defaultdict[typing.Hashable, int] = defaultdict(lambda: len(node_ids))  # type: ignore
        nodes = []

        queue: Queue = Queue()
        queue.put((self, None, 0))
        if self._alternate_tree:
            queue.put((self._alternate_tree, None, 0))

        while not queue.empty():
            node, parent, depth = queue.get()
            nodes.append(
                {
                    "node": node_ids[id(node)],
                    "parent": node_ids[id(parent)] if parent else pd.NA,
                    "is_leaf": isinstance(node, Leaf),
                    "depth": depth,
                    **{k: v for k, v in node.__dict__.items() if k != "children"},
                }
            )
            try:
                for child in node.children:
                    queue.put((child, node, depth + 1))
            except AttributeError:
                pass

        return pd.DataFrame.from_records(nodes).set_index("node")


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
