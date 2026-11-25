"""
Product Ownership Matrix

Represents the ownership structure of a differentiated
products market before and after a merger.
"""

import numpy as np


def ownership_matrix(firm_ids):
    """Construct the product ownership matrix."""

    firm_ids = np.asarray(firm_ids)

    if firm_ids.ndim != 1 or firm_ids.size == 0:
        raise ValueError("Firm IDs must be a non-empty vector.")

    return np.equal.outer(firm_ids, firm_ids).astype(float)


def merge_firms(firm_ids, acquiring_firm, target_firm):
    """Construct the post-merger ownership structure."""

    firm_ids = np.asarray(firm_ids).copy()

    if acquiring_firm == target_firm:
        raise ValueError("Select two different firms.")

    if acquiring_firm not in firm_ids:
        raise ValueError("Acquiring firm does not exist.")

    if target_firm not in firm_ids:
        raise ValueError("Target firm does not exist.")

    firm_ids[firm_ids == target_firm] = acquiring_firm

    return firm_ids


if __name__ == "__main__":

    pre_merger = [0, 1, 2]

    post_merger = merge_firms(pre_merger, 0, 1)

    print("PRE-MERGER OWNERSHIP MATRIX")
    print(ownership_matrix(pre_merger))

    print("\nPOST-MERGER OWNERSHIP MATRIX")
    print(ownership_matrix(post_merger))
