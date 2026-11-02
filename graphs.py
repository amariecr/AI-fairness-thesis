"""
graphs.py

Creates final visualizations for the thesis experiment.

Requires:
results/all_seed_results.csv

Outputs:
graphs/predictive_performance.png
graphs/fairness_comparison.png
graphs/accuracy_fairness_tradeoff.png
graphs/accuracy_change.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------
# Create graphs directory
# ---------------------------------------------------

os.makedirs(
    "graphs",
    exist_ok=True,
)


# ---------------------------------------------------
# Load results
# ---------------------------------------------------

results_df = pd.read_csv(
    "results/all_seed_results.csv"
)


# ---------------------------------------------------
# Calculate mean results across seeds
# ---------------------------------------------------

mean_results = (
    results_df
    .groupby(
        "Technique"
    )
    .mean(
        numeric_only=True
    )
)

technique_order = [
    "Baseline",
    "Reweighing",
    "Resampling",
    "Equalized Odds",
]

mean_results = (
    mean_results.loc[
        technique_order
    ]
)


# ===================================================
# GRAPH 1:
# PREDICTIVE PERFORMANCE
# ===================================================

performance_metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
]

performance_data = (
    mean_results[
        performance_metrics
    ]
)

performance_data.plot(
    kind="bar",
    figsize=(10, 6),
)

plt.title(
    "Predictive Performance Across Bias-Reduction Techniques"
)

plt.xlabel(
    "Technique"
)

plt.ylabel(
    "Average Score"
)

plt.ylim(
    0,
    1
)

plt.xticks(
    rotation=0
)

plt.legend(
    title="Metric"
)

plt.tight_layout()

plt.savefig(
    "graphs/predictive_performance.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ===================================================
# GRAPH 2:
# FAIRNESS COMPARISON
# ===================================================

fairness_metrics = [
    "DP Difference",
    "EOpp Difference",
    "EOdds Difference",
]

fairness_data = (
    mean_results[
        fairness_metrics
    ]
)

fairness_data.plot(
    kind="bar",
    figsize=(10, 6),
)

plt.title(
    "Fairness Differences Across Bias-Reduction Techniques"
)

plt.xlabel(
    "Technique"
)

plt.ylabel(
    "Average Group Difference"
)

plt.xticks(
    rotation=0
)

plt.legend(
    [
        "Demographic Parity",
        "Equal Opportunity",
        "Equalized Odds",
    ],
    title="Fairness Metric",
)

plt.tight_layout()

plt.savefig(
    "graphs/fairness_comparison.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ===================================================
# GRAPH 3:
# ACCURACY VS DEMOGRAPHIC PARITY
# ===================================================

plt.figure(
    figsize=(9, 6)
)

for technique in technique_order:

    accuracy = (
        mean_results.loc[
            technique,
            "Accuracy",
        ]
    )

    dp_difference = (
        mean_results.loc[
            technique,
            "DP Difference",
        ]
    )

    plt.scatter(
        dp_difference,
        accuracy,
        s=120,
    )

    plt.annotate(
        technique,
        (
            dp_difference,
            accuracy,
        ),
        xytext=(7, 5),
        textcoords="offset points",
    )

plt.title(
    "Accuracy vs. Demographic Parity Difference"
)

plt.xlabel(
    "Demographic Parity Difference "
    "(Lower = Smaller Group Gap)"
)

plt.ylabel(
    "Accuracy"
)

plt.tight_layout()

plt.savefig(
    "graphs/accuracy_fairness_tradeoff.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ===================================================
# GRAPH 4:
# ACCURACY CHANGE FROM BASELINE
# ===================================================

baseline_accuracy = (
    mean_results.loc[
        "Baseline",
        "Accuracy",
    ]
)

accuracy_change = (
    mean_results[
        "Accuracy"
    ]
    - baseline_accuracy
) * 100


plt.figure(
    figsize=(9, 6)
)

accuracy_change.plot(
    kind="bar"
)

plt.axhline(
    y=-2,
    linestyle="--",
    label="2 Percentage-Point Guideline",
)

plt.title(
    "Accuracy Change Compared with Baseline"
)

plt.xlabel(
    "Technique"
)

plt.ylabel(
    "Accuracy Change "
    "(Percentage Points)"
)

plt.xticks(
    rotation=0
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "graphs/accuracy_change.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


print(
    "Graphs created successfully:"
)

print(
    "- graphs/predictive_performance.png"
)

print(
    "- graphs/fairness_comparison.png"
)

print(
    "- graphs/accuracy_fairness_tradeoff.png"
)

print(
    "- graphs/accuracy_change.png"
)