"""
Horizontal Merger Simulation

Evaluates unilateral pricing effects, merger-specific
cost efficiencies and welfare consequences under
differentiated Bertrand competition.
"""

import numpy as np
import pandas as pd

from demand import LogitDemand
from equilibrium import BertrandMarket
from ownership import merge_firms


def simulate_horizontal_merger(
    demand,
    marginal_costs,
    firm_ids,
    acquiring_firm,
    target_firm,
    efficiency=0.0
):
    """Simulate a horizontal merger with cost efficiencies."""

    if not np.isfinite(efficiency) or not 0 <= efficiency <= 1:
        raise ValueError("Efficiency must be between zero and one.")

    costs = np.asarray(marginal_costs, dtype=float)
    firm_ids = np.asarray(firm_ids)

    pre_market = BertrandMarket(
        demand, costs, firm_ids
    )

    pre_prices = pre_market.solve_equilibrium()
    pre = pre_market.market_outcomes(pre_prices)

    post_firm_ids = merge_firms(
        firm_ids, acquiring_firm, target_firm
    )

    post_costs = costs.copy()

    merging_products = np.isin(
        firm_ids, [acquiring_firm, target_firm]
    )

    post_costs[merging_products] *= (1 - efficiency)

    post_market = BertrandMarket(
        demand, post_costs, post_firm_ids
    )

    post_prices = post_market.solve_equilibrium(pre_prices)
    post = post_market.market_outcomes(post_prices)

    product_results = pd.DataFrame({
        "product": np.arange(len(costs)) + 1,
        "pre_price": pre_prices,
        "post_price": post_prices,
        "price_change_pct": 100 * (
            post_prices / pre_prices - 1
        ),
        "pre_share": pre["market_shares"],
        "post_share": post["market_shares"],
        "pre_cost": costs,
        "post_cost": post_costs,
        "pre_profit": pre["product_profits"],
        "post_profit": post["product_profits"]
    })

    pre_cs = pre["consumer_surplus"]
    post_cs = post["consumer_surplus"]

    pre_profit = pre["total_profit"]
    post_profit = post["total_profit"]

    pre_welfare = pre_cs + pre_profit
    post_welfare = post_cs + post_profit

    return {
        "efficiency": efficiency,
        "product_results": product_results,
        "pre_consumer_surplus": pre_cs,
        "post_consumer_surplus": post_cs,
        "delta_consumer_surplus": post_cs - pre_cs,
        "pre_industry_profit": pre_profit,
        "post_industry_profit": post_profit,
        "delta_industry_profit": post_profit - pre_profit,
        "pre_welfare": pre_welfare,
        "post_welfare": post_welfare,
        "delta_welfare": post_welfare - pre_welfare,
        "pre_foc_residual": pre["maximum_foc_residual"],
        "post_foc_residual": post["maximum_foc_residual"]
    }


def efficiency_sensitivity(
    demand,
    costs,
    firm_ids,
    acquiring_firm,
    target_firm,
    efficiency_grid
):
    """Evaluate merger outcomes across efficiency scenarios."""

    results = []

    for efficiency in efficiency_grid:

        simulation = simulate_horizontal_merger(
            demand,
            costs,
            firm_ids,
            acquiring_firm,
            target_firm,
            efficiency
        )

        results.append({
            "efficiency_pct": 100 * efficiency,
            "delta_consumer_surplus":
                simulation["delta_consumer_surplus"],
            "delta_industry_profit":
                simulation["delta_industry_profit"],
            "delta_welfare":
                simulation["delta_welfare"]
        })

    return pd.DataFrame(results)


if __name__ == "__main__":

    demand = LogitDemand(
        mean_utilities=[2.5, 2.0, 1.5],
        alpha=0.8,
        market_size=10000
    )

    costs = [1.2, 1.0, 0.8]
    firm_ids = [0, 1, 2]

    simulation = simulate_horizontal_merger(
        demand,
        costs,
        firm_ids,
        acquiring_firm=0,
        target_firm=1,
        efficiency=0.10
    )

    print("\nMERGER SIMULATION WITH COST EFFICIENCIES")
    print("=" * 60)

    print("\nProduct-level results:")
    print(
        simulation["product_results"]
        .round(4)
        .to_string(index=False)
    )

    print("\nChange in consumer surplus:")
    print(round(simulation["delta_consumer_surplus"], 2))

    print("\nChange in industry profit:")
    print(round(simulation["delta_industry_profit"], 2))

    print("\nChange in total welfare:")
    print(round(simulation["delta_welfare"], 2))

    print("\nEFFICIENCY SENSITIVITY ANALYSIS")

    sensitivity = efficiency_sensitivity(
        demand,
        costs,
        firm_ids,
        acquiring_firm=0,
        target_firm=1,
        efficiency_grid=np.linspace(0, 0.5, 11)
    )

    print(sensitivity.round(2).to_string(index=False))
