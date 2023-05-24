import math
from river.tree.split_criterion import InfoGainSplitCriterion


class MultiLabelSplitCriterion(InfoGainSplitCriterion):
    def merit_of_split(self, pre_split_dist, post_split_dist):
        aux_post = [
            {k: v for (k, v) in subdist.items() if k != "total"} for subdist in post_split_dist
        ]
        if self.num_subsets_greater_than_frac(aux_post, self.min_branch_frac_option) < 2:
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