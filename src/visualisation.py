"""
Merger Simulation Visualisation

Generates figures for equilibrium prices, market shares,
welfare changes and merger efficiency sensitivity.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from demand import LogitDemand
from merger_simulation import (
    simulate_horizontal_merger,
    efficiency_sensitivity
)


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (9, 6),
    "figure.dpi": 150,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.2
})


def plot_price_comparison(results):

    data = results["product_results"]

    x = np.arange(len(data))
    width = 0.35

    fig, ax = plt.subplots()

    ax.bar(
        x - width / 2,
        data["pre_price"],
        width,
        label="Pre-merger",
        color="#4477AA"
    )

    ax.bar(
        x + width / 2,
        data["post_price"],
        width,
        label="Post-merger",
        color="#CC6677"
    )

    ax.set_xticks(x)

    ax.set_xticklabels([
        f"Product {i}" for i in data["product"]
    ])

    ax.set_ylabel("Equilibrium price")
    ax.set_title("Unilateral Pricing Effects of a Merger")
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "equilibrium_prices.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


def plot_market_shares(results):

    data = results["product_results"]

    x = np.arange(len(data))
    width = 0.35

    fig, ax = plt.subplots()

    ax.bar(
        x - width / 2,
        100 * data["pre_share"],
        width,
        label="Pre-merger",
        color="#4477AA"
    )

    ax.bar(
        x + width / 2,
        100 * data["post_share"],
        width,
        label="Post-merger",
        color="#CC6677"
    )

    ax.set_xticks(x)

    ax.set_xticklabels([
        f"Product {i}" for i in data["product"]
    ])

    ax.set_ylabel("Market share (%)")
    ax.set_title("Market Share Reallocation Following a Merger")
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "market_shares.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


def plot_efficiency_sensitivity(sensitivity):

    fig, ax = plt.subplots()

    ax.plot(
        sensitivity["efficiency_pct"],
        sensitivity["delta_consumer_surplus"],
        label="Consumer surplus",
        color="#4477AA",
        linewidth=2.5
    )

    ax.plot(
        sensitivity["efficiency_pct"],
        sensitivity["delta_industry_profit"],
        label="Industry profit",
        color="#228833",
        linewidth=2.5
    )

    ax.plot(
        sensitivity["efficiency_pct"],
        sensitivity["delta_welfare"],
        label="Total welfare",
        color="#CC6677",
        linewidth=2.5
    )

    ax.axhline(
        0,
        color="black",
        linestyle="--",
        linewidth=1
    )

    ax.set_xlabel("Marginal Cost Reduction (%)")
    ax.set_ylabel("Change Relative to Pre-Merger Equilibrium")
    ax.set_title("Merger Efficiencies and Welfare Trade-offs")

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "efficiency_sensitivity.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


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
        efficiency=0.0
    )

    sensitivity = efficiency_sensitivity(
        demand,
        costs,
        firm_ids,
        acquiring_firm=0,
        target_firm=1,
        efficiency_grid=np.linspace(0, 1, 101)
    )

    plot_price_comparison(simulation)
    plot_market_shares(simulation)
    plot_efficiency_sensitivity(sensitivity)

    sensitivity.to_csv(
        OUTPUT_DIR / "efficiency_sensitivity.csv",
        index=False
    )

    simulation["product_results"].to_csv(
        OUTPUT_DIR / "merger_results.csv",
        index=False
    )

    print("All figures and results generated successfully.")
    print(f"Output directory: {OUTPUT_DIR}")
