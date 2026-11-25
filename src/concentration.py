"""
Market Concentration and Merger Analysis

Computational tools for measuring market concentration
and simulating changes in market structure following mergers.

Market shares are expressed as percentages.
"""

import math


def validate_market_shares(market_shares):
    """Check whether market shares are economically valid."""

    if len(market_shares) == 0:
        raise ValueError("Market shares cannot be empty.")

    if any(not math.isfinite(share) for share in market_shares):
        raise ValueError("Market shares must be finite.")

    if any(share < 0 for share in market_shares):
        raise ValueError("Market shares cannot be negative.")

    if not math.isclose(
        sum(market_shares), 100, rel_tol=0, abs_tol=0.000001
    ):
        raise ValueError("Market shares must sum to 100.")


def calculate_hhi(market_shares):
    """Calculate the Herfindahl-Hirschman Index."""

    validate_market_shares(market_shares)

    return sum(share ** 2 for share in market_shares)


def concentration_ratio(market_shares, n=4):
    """Calculate the concentration ratio of the n largest firms."""

    validate_market_shares(market_shares)

    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer.")

    sorted_shares = sorted(market_shares, reverse=True)

    return sum(sorted_shares[:n])


def effective_number_of_firms(market_shares):
    """Calculate the equivalent number of equally sized firms."""

    hhi = calculate_hhi(market_shares)

    return 10000 / hhi


def simulate_merger(market_shares, firm_a, firm_b):
    """Construct post-merger market shares."""

    validate_market_shares(market_shares)

    if not isinstance(firm_a, int) or not isinstance(firm_b, int):
        raise ValueError("Firm indices must be integers.")

    if firm_a == firm_b:
        raise ValueError("Select two different firms.")

    if not 0 <= firm_a < len(market_shares):
        raise IndexError("Invalid acquiring firm index.")

    if not 0 <= firm_b < len(market_shares):
        raise IndexError("Invalid target firm index.")

    post_merger = market_shares.copy()

    post_merger[firm_a] += post_merger[firm_b]
    post_merger.pop(firm_b)

    return post_merger


def merger_delta_hhi(share_a, share_b):
    """Calculate the analytical change in HHI."""

    if not math.isfinite(share_a) or not math.isfinite(share_b):
        raise ValueError("Market shares must be finite.")

    if share_a < 0 or share_b < 0 or share_a + share_b > 100:
        raise ValueError("Invalid merging firms' market shares.")

    return 2 * share_a * share_b


def merger_analysis(market_shares, firm_a, firm_b):
    """Conduct a market concentration analysis of a merger."""

    post_merger = simulate_merger(
        market_shares, firm_a, firm_b
    )

    hhi_pre = calculate_hhi(market_shares)
    hhi_post = calculate_hhi(post_merger)

    return {
        "pre_merger_shares": market_shares.copy(),
        "post_merger_shares": post_merger,
        "pre_merger_hhi": hhi_pre,
        "post_merger_hhi": hhi_post,
        "delta_hhi": hhi_post - hhi_pre,
        "analytical_delta_hhi": merger_delta_hhi(
            market_shares[firm_a],
            market_shares[firm_b]
        ),
        "pre_merger_cr4": concentration_ratio(market_shares),
        "post_merger_cr4": concentration_ratio(post_merger),
        "pre_merger_effective_firms": effective_number_of_firms(
            market_shares
        ),
        "post_merger_effective_firms": effective_number_of_firms(
            post_merger
        )
    }


# Example application

if __name__ == "__main__":

    market_shares = [40, 25, 15, 10, 10]

    results = merger_analysis(market_shares, 0, 1)

    print("\nMARKET CONCENTRATION AND MERGER ANALYSIS")
    print("=" * 50)

    print("Pre-merger shares:", results["pre_merger_shares"])
    print("Post-merger shares:", results["post_merger_shares"])

    print("\nHERFINDAHL-HIRSCHMAN INDEX")
    print("Pre-merger HHI:", results["pre_merger_hhi"])
    print("Post-merger HHI:", results["post_merger_hhi"])
    print("Change in HHI:", results["delta_hhi"])
    print("Analytical change:", results["analytical_delta_hhi"])

    print("\nCONCENTRATION RATIOS")
    print("Pre-merger CR4:", results["pre_merger_cr4"])
    print("Post-merger CR4:", results["post_merger_cr4"])

    print("\nEFFECTIVE NUMBER OF FIRMS")
    print(
        "Pre-merger:",
        round(results["pre_merger_effective_firms"], 2)
    )
    print(
        "Post-merger:",
        round(results["post_merger_effective_firms"], 2)
    )

    print("\n" + "=" * 50)
