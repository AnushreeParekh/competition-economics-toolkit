"""
Interactive Competition Economics Toolkit

Structural simulation of horizontal mergers under
multinomial logit demand and differentiated Bertrand
competition.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent / "src")
)

from demand import LogitDemand

from merger_simulation import (
    simulate_horizontal_merger,
    efficiency_sensitivity
)

from welfare import compare_welfare_standards


# PAGE CONFIGURATION

st.set_page_config(
    page_title="Competition Economics Toolkit",
    layout="wide",
    page_icon="📊"
)

st.title("Competition Economics Toolkit")

st.subheader(
    "Structural Merger Simulation and Welfare Analysis"
)

st.markdown(
    "An interactive framework for analysing horizontal "
    "mergers under differentiated Bertrand competition "
    "and multinomial logit demand."
)

st.caption(
    "Developed by Anushree Parekh, PhD Economics"
)


# SIDEBAR PARAMETERS

st.sidebar.header("Model Parameters")

alpha = st.sidebar.slider(
    "Price sensitivity",
    min_value=0.2,
    max_value=2.0,
    value=0.8,
    step=0.1
)

market_size = st.sidebar.number_input(
    "Potential market size",
    min_value=100,
    max_value=1000000,
    value=10000,
    step=1000
)

st.sidebar.header("Product Characteristics")

mean_utilities = []
costs = []

products = ["Product A", "Product B", "Product C"]

default_utilities = [2.5, 2.0, 1.5]
default_costs = [1.2, 1.0, 0.8]

for i, product in enumerate(products):

    with st.sidebar.expander(product):

        utility = st.slider(
            "Mean utility",
            min_value=0.0,
            max_value=5.0,
            value=default_utilities[i],
            step=0.1,
            key=f"utility_{i}"
        )

        cost = st.number_input(
            "Marginal cost",
            min_value=0.0,
            max_value=10.0,
            value=default_costs[i],
            step=0.1,
            key=f"cost_{i}"
        )

        mean_utilities.append(utility)
        costs.append(cost)


# MERGER PARAMETERS

st.sidebar.header("Merger Configuration")

acquiring_firm = st.sidebar.selectbox(
    "Acquiring firm",
    options=[0, 1, 2],
    format_func=lambda x: f"Firm {chr(65 + x)}"
)

target_firm = st.sidebar.selectbox(
    "Target firm",
    options=[
        i for i in range(3)
        if i != acquiring_firm
    ],
    format_func=lambda x: f"Firm {chr(65 + x)}"
)

efficiency_pct = st.sidebar.slider(
    "Marginal cost efficiency (%)",
    min_value=0,
    max_value=100,
    value=0,
    step=1
)

efficiency = efficiency_pct / 100


# SIMULATE THE MERGER

try:

    demand = LogitDemand(
        mean_utilities=mean_utilities,
        alpha=alpha,
        market_size=market_size
    )

    simulation = simulate_horizontal_merger(
        demand=demand,
        marginal_costs=costs,
        firm_ids=[0, 1, 2],
        acquiring_firm=acquiring_firm,
        target_firm=target_firm,
        efficiency=efficiency
    )

except (ValueError, RuntimeError) as error:

    st.error(f"Simulation failed: {error}")
    st.stop()


data = simulation["product_results"].copy()

data["product"] = products

data = data.set_index("product")


# SUMMARY METRICS

st.header("Merger Impact Summary")

col1, col2, col3 = st.columns(3)

cs_change = simulation["delta_consumer_surplus"]

profit_change = simulation["delta_industry_profit"]

welfare_change = simulation["delta_welfare"]

col1.metric(
    "Consumer Surplus Change",
    f"{cs_change:,.2f}"
)

col2.metric(
    "Industry Profit Change",
    f"{profit_change:,.2f}"
)

col3.metric(
    "Total Welfare Change",
    f"{welfare_change:,.2f}"
)

if cs_change < 0 and welfare_change > 0:

    st.info(
        "The merger increases total welfare but "
        "reduces consumer surplus under the "
        "selected parameters."
    )

elif cs_change < 0 and welfare_change < 0:

    st.warning(
        "The merger reduces both consumer surplus "
        "and total welfare under the selected parameters."
    )


# ANALYSIS TABS

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Pricing Effects",
    "Market Shares",
    "Welfare Analysis",
    "Efficiency Thresholds",
    "Methodology"
])


# TAB 1: PRICING EFFECTS

with tab1:

    st.subheader("Equilibrium Price Comparison")

    price_data = data[
        ["pre_price", "post_price"]
    ].rename(columns={
        "pre_price": "Pre-merger",
        "post_price": "Post-merger"
    })

    st.bar_chart(price_data, stack=False)

    st.subheader("Product-Level Results")

    st.dataframe(
        data[[
            "pre_price",
            "post_price",
            "price_change_pct",
            "pre_cost",
            "post_cost"
        ]].round(4)
    )


# TAB 2: MARKET SHARES

with tab2:

    st.subheader("Market Share Reallocation")

    shares = 100 * data[
        ["pre_share", "post_share"]
    ]

    shares.loc["Outside option"] = (
        100 - shares.sum()
    )

    shares = shares.rename(columns={
        "pre_share": "Pre-merger",
        "post_share": "Post-merger"
    })

    st.bar_chart(shares, stack=False)

    st.dataframe(shares.round(2))

    st.caption(
        "Shares are percentages of the potential "
        "market, including the outside option."
    )


# TAB 3: WELFARE ANALYSIS

with tab3:

    st.subheader("Welfare Decomposition")

    welfare_table = pd.DataFrame({
        "Pre-merger": [
            simulation["pre_consumer_surplus"],
            simulation["pre_industry_profit"],
            simulation["pre_welfare"]
        ],
        "Post-merger": [
            simulation["post_consumer_surplus"],
            simulation["post_industry_profit"],
            simulation["post_welfare"]
        ]
    }, index=[
        "Consumer surplus",
        "Industry profit",
        "Total welfare"
    ])

    welfare_table["Change"] = (
        welfare_table["Post-merger"]
        - welfare_table["Pre-merger"]
    )

    st.dataframe(welfare_table.round(2))

    st.bar_chart(welfare_table["Change"])

    st.subheader("Numerical Diagnostics")

    st.write(
        "Pre-merger maximum FOC residual:",
        simulation["pre_foc_residual"]
    )

    st.write(
        "Post-merger maximum FOC residual:",
        simulation["post_foc_residual"]
    )


# TAB 4: EFFICIENCY THRESHOLDS

with tab4:

    st.subheader("Merger Efficiencies and Welfare")

    if st.button("Run Efficiency Sensitivity Analysis"):

        try:

            with st.spinner("Solving counterfactual equilibria..."):

                sensitivity = efficiency_sensitivity(
                    demand,
                    costs,
                    [0, 1, 2],
                    acquiring_firm,
                    target_firm,
                    np.linspace(0, 1, 51)
                )

                thresholds = compare_welfare_standards(
                    demand,
                    costs,
                    [0, 1, 2],
                    acquiring_firm,
                    target_firm
                )

        except (ValueError, RuntimeError) as error:

            st.error(f"Sensitivity analysis failed: {error}")

        else:

            chart_data = sensitivity.set_index(
                "efficiency_pct"
            ).rename(columns={
                "delta_consumer_surplus": "Consumer surplus",
                "delta_industry_profit": "Industry profit",
                "delta_welfare": "Total welfare"
            })

            st.line_chart(chart_data)

            col1, col2 = st.columns(2)

            consumer_threshold = thresholds[
                "consumer_threshold"
            ]

            total_threshold = thresholds[
                "total_threshold"
            ]

            col1.metric(
                "Consumer Welfare Break-even",
                (
                    f"{100 * consumer_threshold:.2f}%"
                    if consumer_threshold is not None
                    else "Not bracketed"
                )
            )

            col2.metric(
                "Total Welfare Break-even",
                (
                    f"{100 * total_threshold:.2f}%"
                    if total_threshold is not None
                    else "Not bracketed"
                )
            )

            st.dataframe(sensitivity.round(2))

            st.download_button(
                "Download Sensitivity Results",
                sensitivity.to_csv(index=False),
                file_name="efficiency_sensitivity.csv",
                mime="text/csv"
            )


# TAB 5: METHODOLOGY

with tab5:

    st.subheader("Economic Framework")

    st.markdown("### 1. Consumer Demand")

    st.latex(
        r"u_{ij}=\delta_j-\alpha p_j+\varepsilon_{ij}"
    )

    st.latex(
        r"s_j(\mathbf{p})="
        r"\frac{\exp(\delta_j-\alpha p_j)}"
        r"{1+\sum_{k=1}^{J}\exp(\delta_k-\alpha p_k)}"
    )

    st.markdown("### 2. Firm Profit Maximisation")

    st.latex(
        r"\pi_f=M\sum_{j\in\mathcal{J}_f}"
        r"(p_j-c_j)s_j(\mathbf{p})"
    )

    st.markdown("### 3. Bertrand Pricing Conditions")

    st.latex(
        r"s_j+\sum_{k\in\mathcal{J}_f}"
        r"(p_k-c_k)\frac{\partial s_k}{\partial p_j}=0"
    )

    st.markdown("### 4. Consumer Surplus")

    st.latex(
        r"CS=\frac{M}{\alpha}"
        r"\log\left(1+\sum_j"
        r"\exp(\delta_j-\alpha p_j)\right)"
    )

    st.markdown("### 5. Total Welfare")

    st.latex(
        r"W=CS+\sum_f\pi_f"
    )

    st.markdown("### Model Limitations")

    st.markdown("""
    - Demand parameters are calibrated, not estimated.
    - Consumers have homogeneous price sensitivity.
    - Substitution follows the logit IIA property.
    - Marginal costs are constant.
    - Merger efficiencies are hypothetical.
    - Entry, exit and dynamic competition are excluded.
    - Numerical solutions satisfy pricing first-order
      conditions but global optimality and uniqueness
      are not independently established.
    - Efficiency threshold searches rely on bracketed roots.
    - The model is illustrative and does not establish
      actual competitive harm in a real market.
    """)


# EXPORT RESULTS

st.divider()

st.download_button(
    "Download Product-Level Simulation Results",
    data.to_csv(),
    file_name="merger_simulation_results.csv",
    mime="text/csv"
)

st.caption(
    "Competition Economics Toolkit | "
    "Computational Industrial Organisation | "
    "Research and educational use"
)
