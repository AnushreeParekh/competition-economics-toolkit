"""
Multinomial Logit Demand Model

A differentiated-products demand system with an outside option,
own-price and cross-price elasticities, and consumer surplus.

Assumptions:
- Homogeneous price sensitivity
- Type I extreme value utility shocks
- Outside option utility normalised to zero
- Positive price sensitivity parameter
"""

import numpy as np
from scipy.special import logsumexp


class LogitDemand:

    def __init__(self, mean_utilities, alpha, market_size=1):

        self.delta = np.asarray(mean_utilities, dtype=float)
        self.alpha = float(alpha)
        self.market_size = float(market_size)

        if self.delta.ndim != 1 or self.delta.size == 0:
            raise ValueError("Mean utilities must be a non-empty vector.")

        if not np.all(np.isfinite(self.delta)):
            raise ValueError("Mean utilities must be finite.")

        if not np.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("Price sensitivity must be positive and finite.")

        if not np.isfinite(self.market_size) or self.market_size <= 0:
            raise ValueError("Market size must be positive and finite.")

    def validate_prices(self, prices):

        prices = np.asarray(prices, dtype=float)

        if prices.shape != self.delta.shape:
            raise ValueError("Each product must have a corresponding price.")

        if not np.all(np.isfinite(prices)) or np.any(prices < 0):
            raise ValueError("Prices must be finite and non-negative.")

        return prices

    def utility(self, prices):
        """Calculate deterministic indirect utility."""

        prices = self.validate_prices(prices)

        return self.delta - self.alpha * prices

    def market_shares(self, prices):
        """Calculate product market shares."""

        utility = self.utility(prices)

        denominator = logsumexp(
            np.concatenate(([0.0], utility))
        )

        return np.exp(utility - denominator)

    def outside_share(self, prices):
        """Calculate the share choosing the outside option."""

        utility = self.utility(prices)

        denominator = logsumexp(
            np.concatenate(([0.0], utility))
        )

        return np.exp(-denominator)

    def quantities(self, prices):
        """Calculate product-level quantities demanded."""

        return self.market_size * self.market_shares(prices)

    def demand_jacobian(self, prices):
        """Calculate derivatives of shares with respect to prices."""

        shares = self.market_shares(prices)

        jacobian = -self.alpha * (
            np.diag(shares) - np.outer(shares, shares)
        )

        return jacobian

    def elasticities(self, prices):
        """Calculate own-price and cross-price elasticities."""

        prices = self.validate_prices(prices)
        shares = self.market_shares(prices)

        elasticity_matrix = self.alpha * np.tile(
            shares * prices, (len(shares), 1)
        )

        elasticity_matrix -= self.alpha * np.diag(prices)

        return elasticity_matrix

    def diversion_ratios(self, prices):
        """Calculate product-to-product diversion ratios."""

        shares = self.market_shares(prices)
        denominator = 1 - shares

        if np.any(denominator <= 0):
            raise ValueError("Diversion is undefined for a unit market share.")

        diversion = shares[None, :] / denominator[:, None]

        np.fill_diagonal(diversion, 0)

        return diversion

    def consumer_surplus(self, prices):
        """Calculate expected aggregate consumer surplus."""

        utility = self.utility(prices)

        surplus = logsumexp(
            np.concatenate(([0.0], utility))
        ) / self.alpha

        return self.market_size * surplus


# Example application

if __name__ == "__main__":

    np.set_printoptions(precision=4, suppress=True)

    model = LogitDemand(
        mean_utilities=[2.5, 2.0, 1.5],
        alpha=0.8,
        market_size=10000
    )

    prices = np.array([3.0, 2.5, 2.0])

    print("\nMULTINOMIAL LOGIT DEMAND MODEL")
    print("=" * 50)

    print("\nPrices:")
    print(prices)

    print("\nMarket shares:")
    print(model.market_shares(prices))

    print("\nOutside option share:")
    print(model.outside_share(prices))

    print("\nQuantities demanded:")
    print(model.quantities(prices))

    print("\nDemand Jacobian:")
    print(model.demand_jacobian(prices))

    print("\nPrice elasticity matrix:")
    print(model.elasticities(prices))

    print("\nDiversion ratios:")
    print(model.diversion_ratios(prices))

    print("\nAggregate consumer surplus:")
    print(round(model.consumer_surplus(prices), 2))

    print("\n" + "=" * 50)
