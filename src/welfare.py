"""
Welfare Analysis and Merger Efficiency Thresholds

Computes the marginal cost efficiencies required to
offset the adverse welfare effects of a horizontal merger.

Distinguishes between consumer welfare and total welfare
standards.
"""

import numpy as np
from scipy.optimize import brentq

from demand import LogitDemand
from merger_simulation import simulate_horizontal_merger


def find_efficiency_threshold(
    demand,
    costs,
    firm_ids,
    acquiring_firm,
    target_firm,
    welfare_standard="consumer"
):
    """Find a break-even efficiency between zero and one."""

    standards = {
        "consumer": "delta_consumer_surplus",
        "total": "delta_welfare"
    }

    if welfare_standard not in standards:
        raise ValueError(
            "Welfare standard must be 'consumer' or 'total'."
        )

    welfare_key = standards[welfare_standard]

    def welfare_change(efficiency):

        results = simulate_horizontal_merger(
            demand,
            costs,
            firm_ids,
            acquiring_firm,
            target_firm,
            efficiency
        )

        return results[welfare_key]

    lower = welfare_change(0.0)
    upper = welfare_change(1.0)

    if lower >= 0:
        return 0.0

    if upper == 0:
        return 1.0

    if upper < 0:
        return None

    threshold = brentq(
        welfare_change,
        0.0,
        1.0,
        xtol=1e-10
    )

    return threshold


def compare_welfare_standards(
    demand,
    costs,
    firm_ids,
    acquiring_firm,
    target_firm
):
    """Compare efficiency thresholds under two welfare standards."""

    consumer_threshold = find_efficiency_threshold(
        demand,
        costs,
        firm_ids,
        acquiring_firm,
        target_firm,
        welfare_standard="consumer"
    )

    total_threshold = find_efficiency_threshold(
        demand,
        costs,
        firm_ids,
        acquiring_firm,
        target_firm,
        welfare_standard="total"
    )

    return {
        "consumer_threshold": consumer_threshold,
        "total_threshold": total_threshold
    }


if __name__ == "__main__":

    demand = LogitDemand(
        mean_utilities=[2.5, 2.0, 1.5],
        alpha=0.8,
        market_size=10000
    )

    costs = [1.2, 1.0, 0.8]
    firm_ids = [0, 1, 2]

    thresholds = compare_welfare_standards(
        demand,
        costs,
        firm_ids,
        acquiring_firm=0,
        target_firm=1
    )

    print("\nMERGER EFFICIENCY THRESHOLDS")
    print("=" * 50)

    for standard, threshold in thresholds.items():

        print("\n", standard.replace("_", " ").title())

        if threshold is None:

            print("No break-even threshold bracketed.")

        else:

            print(f"Required efficiency: {100 * threshold:.4f}%")

    print("\n" + "=" * 50)
