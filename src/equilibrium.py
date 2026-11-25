"""
Differentiated Bertrand Equilibrium Solver

Computes candidate equilibrium prices for single-product
and multiproduct firms under multinomial logit demand.
"""

import numpy as np
from scipy.optimize import least_squares

from demand import LogitDemand
from ownership import ownership_matrix, merge_firms


class BertrandMarket:

    def __init__(self, demand_model, marginal_costs, firm_ids):

        self.demand = demand_model
        self.costs = np.asarray(marginal_costs, dtype=float)
        self.firm_ids = np.asarray(firm_ids)
        self.ownership = ownership_matrix(firm_ids)

        if self.costs.shape != self.demand.delta.shape:
            raise ValueError("Each product requires a marginal cost.")

        if self.firm_ids.shape != self.costs.shape:
            raise ValueError("Each product requires a firm ID.")

        if not np.all(np.isfinite(self.costs)):
            raise ValueError("Marginal costs must be finite.")

        if np.any(self.costs < 0):
            raise ValueError("Marginal costs cannot be negative.")

    def first_order_conditions(self, prices):
        """Evaluate firms' pricing first-order conditions."""

        prices = self.demand.validate_prices(prices)

        shares = self.demand.market_shares(prices)
        jacobian = self.demand.demand_jacobian(prices)

        markups = prices - self.costs

        internalised_effects = self.ownership * jacobian.T

        return shares + internalised_effects @ markups

    def solve_equilibrium(self, initial_prices=None):
        """Numerically solve the pricing first-order conditions."""

        if initial_prices is None:
            initial_prices = self.costs + 1 / self.demand.alpha

        initial_prices = np.asarray(initial_prices, dtype=float)

        solution = least_squares(
            self.first_order_conditions,
            initial_prices,
            bounds=(self.costs, np.inf),
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
            max_nfev=5000
        )

        prices = solution.x

        residual = np.max(
            np.abs(self.first_order_conditions(prices))
        )

        if not solution.success or residual > 1e-8:
            raise RuntimeError(
                "Equilibrium solver failed to converge."
            )

        return prices

    def product_profits(self, prices):
        """Calculate product-level operating profits."""

        quantities = self.demand.quantities(prices)

        return (prices - self.costs) * quantities

    def firm_profits(self, prices):
        """Aggregate operating profits by firm."""

        product_profits = self.product_profits(prices)

        return {
            firm: float(
                product_profits[self.firm_ids == firm].sum()
            )
            for firm in np.unique(self.firm_ids)
        }

    def market_outcomes(self, prices):
        """Report equilibrium market outcomes."""

        prices = self.demand.validate_prices(prices)

        return {
            "prices": prices,
            "market_shares": self.demand.market_shares(prices),
            "outside_share": self.demand.outside_share(prices),
            "quantities": self.demand.quantities(prices),
            "markups": prices - self.costs,
            "product_profits": self.product_profits(prices),
            "firm_profits": self.firm_profits(prices),
            "consumer_surplus": self.demand.consumer_surplus(prices),
            "total_profit": self.product_profits(prices).sum(),
            "maximum_foc_residual": np.max(
                np.abs(self.first_order_conditions(prices))
            )
        }


if __name__ == "__main__":

    np.set_printoptions(precision=4, suppress=True)

    demand = LogitDemand(
        mean_utilities=[2.5, 2.0, 1.5],
        alpha=0.8,
        market_size=10000
    )

    costs = [1.2, 1.0, 0.8]

    pre_merger_ids = [0, 1, 2]

    post_merger_ids = merge_firms(
        pre_merger_ids, 0, 1
    )

    pre_market = BertrandMarket(
        demand, costs, pre_merger_ids
    )

    post_market = BertrandMarket(
        demand, costs, post_merger_ids
    )

    pre_prices = pre_market.solve_equilibrium()

    post_prices = post_market.solve_equilibrium()

    pre_results = pre_market.market_outcomes(pre_prices)

    post_results = post_market.market_outcomes(post_prices)

    print("\nBERTRAND MERGER SIMULATION")
    print("=" * 50)

    print("\nPre-merger equilibrium prices:")
    print(pre_prices)

    print("\nPost-merger equilibrium prices:")
    print(post_prices)

    print("\nPercentage price changes:")
    print(100 * (post_prices / pre_prices - 1))

    print("\nPre-merger market shares:")
    print(pre_results["market_shares"])

    print("\nPost-merger market shares:")
    print(post_results["market_shares"])

    print("\nPre-merger consumer surplus:")
    print(round(pre_results["consumer_surplus"], 2))

    print("\nPost-merger consumer surplus:")
    print(round(post_results["consumer_surplus"], 2))

    print("\nChange in consumer surplus:")
    print(round(
        post_results["consumer_surplus"]
        - pre_results["consumer_surplus"], 2
    ))

    print("\nPre-merger total profit:")
    print(round(pre_results["total_profit"], 2))

    print("\nPost-merger total profit:")
    print(round(post_results["total_profit"], 2))

    print("\nMaximum pricing FOC residuals:")
    print(pre_results["maximum_foc_residual"])
    print(post_results["maximum_foc_residual"])

    print("\n" + "=" * 50)
