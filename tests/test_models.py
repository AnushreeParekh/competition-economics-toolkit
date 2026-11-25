"""
Automated Tests for the Competition Economics Toolkit.

Tests economic identities, demand properties,
numerical equilibrium conditions and merger outcomes.
"""

import sys
from pathlib import Path

import numpy as np
import pytest


sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src")
)

from concentration import (
    calculate_hhi,
    simulate_merger,
    merger_delta_hhi
)

from demand import LogitDemand
from ownership import ownership_matrix, merge_firms
from equilibrium import BertrandMarket
from merger_simulation import simulate_horizontal_merger
from welfare import find_efficiency_threshold


@pytest.fixture
def demand():

    return LogitDemand(
        mean_utilities=[2.5, 2.0, 1.5],
        alpha=0.8,
        market_size=10000
    )


# MARKET CONCENTRATION TESTS

def test_monopoly_hhi():

    assert calculate_hhi([100]) == 10000


def test_symmetric_duopoly_hhi():

    assert calculate_hhi([50, 50]) == 5000


def test_invalid_market_shares():

    with pytest.raises(ValueError):
        calculate_hhi([40, 30])


def test_merger_hhi_identity():

    shares = [40, 25, 15, 10, 10]

    post = simulate_merger(shares, 0, 1)

    numerical_delta = (
        calculate_hhi(post) - calculate_hhi(shares)
    )

    analytical_delta = merger_delta_hhi(40, 25)

    assert numerical_delta == analytical_delta


# DEMAND SYSTEM TESTS

def test_market_shares_sum_to_one(demand):

    prices = np.array([3.0, 2.5, 2.0])

    total = (
        demand.market_shares(prices).sum()
        + demand.outside_share(prices)
    )

    assert np.isclose(total, 1.0)


def test_own_price_elasticities(demand):

    elasticities = demand.elasticities(
        [3.0, 2.5, 2.0]
    )

    assert np.all(np.diag(elasticities) < 0)


def test_cross_price_elasticities(demand):

    elasticities = demand.elasticities(
        [3.0, 2.5, 2.0]
    )

    off_diagonal = ~np.eye(3, dtype=bool)

    assert np.all(elasticities[off_diagonal] > 0)


def test_demand_jacobian(demand):

    prices = np.array([3.0, 2.5, 2.0])

    analytical = demand.demand_jacobian(prices)

    numerical = np.zeros((3, 3))

    epsilon = 1e-5

    for j in range(3):

        upper = prices.copy()
        lower = prices.copy()

        upper[j] += epsilon
        lower[j] -= epsilon

        numerical[:, j] = (
            demand.market_shares(upper)
            - demand.market_shares(lower)
        ) / (2 * epsilon)

    assert np.allclose(
        analytical,
        numerical,
        atol=1e-7
    )


def test_consumer_surplus_decreases_with_prices(demand):

    initial = demand.consumer_surplus(
        [3.0, 2.5, 2.0]
    )

    higher_prices = demand.consumer_surplus(
        [3.5, 3.0, 2.5]
    )

    assert higher_prices < initial


# OWNERSHIP TESTS

def test_independent_firms():

    matrix = ownership_matrix([0, 1, 2])

    assert np.array_equal(matrix, np.eye(3))


def test_post_merger_ownership():

    merged_ids = merge_firms([0, 1, 2], 0, 1)

    expected = np.array([
        [1, 1, 0],
        [1, 1, 0],
        [0, 0, 1]
    ])

    assert np.array_equal(
        ownership_matrix(merged_ids),
        expected
    )


# BERTRAND EQUILIBRIUM TESTS

def test_equilibrium_residual(demand):

    market = BertrandMarket(
        demand,
        [1.2, 1.0, 0.8],
        [0, 1, 2]
    )

    prices = market.solve_equilibrium()

    residuals = market.first_order_conditions(prices)

    assert np.max(np.abs(residuals)) < 1e-8


def test_positive_equilibrium_markups(demand):

    costs = np.array([1.2, 1.0, 0.8])

    market = BertrandMarket(
        demand,
        costs,
        [0, 1, 2]
    )

    prices = market.solve_equilibrium()

    assert np.all(prices > costs)


def test_single_product_markup_identity(demand):

    costs = np.array([1.2, 1.0, 0.8])

    market = BertrandMarket(
        demand,
        costs,
        [0, 1, 2]
    )

    prices = market.solve_equilibrium()

    shares = demand.market_shares(prices)

    theoretical_markups = (
        1 / (demand.alpha * (1 - shares))
    )

    assert np.allclose(
        prices - costs,
        theoretical_markups,
        atol=1e-7
    )


# MERGER SIMULATION TESTS

def test_merger_price_effect(demand):

    results = simulate_horizontal_merger(
        demand,
        [1.2, 1.0, 0.8],
        [0, 1, 2],
        0,
        1,
        efficiency=0
    )

    changes = results["product_results"]["price_change_pct"]

    assert changes.iloc[0] > 0
    assert changes.iloc[1] > 0


def test_merger_efficiency_improves_welfare(demand):

    baseline = simulate_horizontal_merger(
        demand,
        [1.2, 1.0, 0.8],
        [0, 1, 2],
        0,
        1,
        efficiency=0
    )

    efficient = simulate_horizontal_merger(
        demand,
        [1.2, 1.0, 0.8],
        [0, 1, 2],
        0,
        1,
        efficiency=0.20
    )

    assert (
        efficient["delta_welfare"]
        > baseline["delta_welfare"]
    )


def test_total_welfare_threshold(demand):

    threshold = find_efficiency_threshold(
        demand,
        [1.2, 1.0, 0.8],
        [0, 1, 2],
        0,
        1,
        welfare_standard="total"
    )

    assert threshold is not None
    assert 0.19 < threshold < 0.21

    results = simulate_horizontal_merger(
        demand,
        [1.2, 1.0, 0.8],
        [0, 1, 2],
        0,
        1,
        efficiency=threshold
    )

    assert abs(results["delta_welfare"]) < 1e-5
