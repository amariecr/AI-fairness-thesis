"""
experiment.py

Main thesis experiment.

Evaluates four conditions:

1. Baseline logistic regression
2. Reweighing
3. Resampling
4. Equalized odds post-processing

The complete experiment is repeated using seeds:
42, 43, and 44.

Outputs:
- results/all_seed_results.csv
- results/summary_results.csv
"""

import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.utils import resample

from fairlearn.metrics import (
    demographic_parity_difference,
    equal_opportunity_difference,
    equalized_odds_difference,
)

from fairlearn.postprocessing import ThresholdOptimizer

from data_setup import (
    load_clean_data,
    split_data,
    create_preprocessor,
)


# ---------------------------------------------------
# Result folder
# ---------------------------------------------------

os.makedirs(
    "results",
    exist_ok=True,
)


# ---------------------------------------------------
# Helper function for evaluation
# ---------------------------------------------------

def calculate_metrics(
    y_true,
    y_pred,
    sensitive_features,
):
    """
    Calculate predictive-performance and fairness metrics.

    Lower fairness-difference values indicate smaller
    differences between demographic groups.
    """

    return {
        "Accuracy": accuracy_score(
            y_true,
            y_pred,
        ),

        "Precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),

        "Recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),

        "F1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),

        "DP Difference":
            demographic_parity_difference(
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive_features,
            ),

        "EOpp Difference":
            equal_opportunity_difference(
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive_features,
            ),

        "EOdds Difference":
            equalized_odds_difference(
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive_features,
            ),
    }


def add_result(
    results,
    seed,
    technique,
    y_true,
    y_pred,
    sensitive_features,
):
    """
    Calculate metrics and add one experiment row
    to the results list.
    """

    metrics = calculate_metrics(
        y_true,
        y_pred,
        sensitive_features,
    )

    metrics["Seed"] = seed
    metrics["Technique"] = technique

    results.append(metrics)


# ---------------------------------------------------
# Load dataset once
# ---------------------------------------------------

X, y, A = load_clean_data()


# ---------------------------------------------------
# Experiment settings
# ---------------------------------------------------

seeds = [
    42,
    43,
    44,
]

all_results = []


# ===================================================
# RUN FULL EXPERIMENT
# ===================================================

for seed in seeds:

    print("\n")
    print("=" * 70)
    print(
        f"RUNNING EXPERIMENT WITH SEED {seed}"
    )
    print("=" * 70)

    # ---------------------------------------------------
    # Split data
    # ---------------------------------------------------

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        A_train,
        A_val,
        A_test,
    ) = split_data(
        X,
        y,
        A,
        seed,
    )

    print("\nDataset split sizes:")
    print(
        f"Training:   {len(X_train)}"
    )
    print(
        f"Validation: {len(X_val)}"
    )
    print(
        f"Testing:    {len(X_test)}"
    )

    # ---------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------

    preprocessor = create_preprocessor(
        X_train
    )

    # Fit ONLY on training data
    preprocessor.fit(
        X_train
    )

    X_train_proc = preprocessor.transform(
        X_train
    )

    X_val_proc = preprocessor.transform(
        X_val
    )

    X_test_proc = preprocessor.transform(
        X_test
    )

    print("\nProcessed feature shapes:")
    print(
        "Training:",
        X_train_proc.shape,
    )
    print(
        "Validation:",
        X_val_proc.shape,
    )
    print(
        "Testing:",
        X_test_proc.shape,
    )

    # ===================================================
    # 1. BASELINE
    # ===================================================

    baseline_model = LogisticRegression(
        max_iter=1000,
        random_state=seed,
    )

    baseline_model.fit(
        X_train_proc,
        y_train,
    )

    baseline_predictions = (
        baseline_model.predict(
            X_test_proc
        )
    )

    add_result(
        all_results,
        seed,
        "Baseline",
        y_test,
        baseline_predictions,
        A_test,
    )

    print("\nBaseline complete.")

    # ===================================================
    # 2. REWEIGHING
    # ===================================================

    # Reset indexes so sensitive attributes and
    # targets align row-by-row.
    A_rw = (
        A_train
        .reset_index(drop=True)
    )

    y_rw = (
        y_train
        .reset_index(drop=True)
    )

    reweight_df = pd.DataFrame(
        {
            "sex": A_rw,
            "income": y_rw,
        }
    )

    total_count = len(
        reweight_df
    )

    sex_counts = (
        reweight_df[
            "sex"
        ].value_counts()
    )

    income_counts = (
        reweight_df[
            "income"
        ].value_counts()
    )

    joint_counts = (
        reweight_df.groupby(
            [
                "sex",
                "income",
            ]
        ).size()
    )

    sample_weights = []

    for _, row in reweight_df.iterrows():

        sex_value = row[
            "sex"
        ]

        income_value = row[
            "income"
        ]

        n_sex = sex_counts[
            sex_value
        ]

        n_income = income_counts[
            income_value
        ]

        n_joint = joint_counts[
            (
                sex_value,
                income_value,
            )
        ]

        weight = (
            n_sex
            * n_income
        ) / (
            total_count
            * n_joint
        )

        sample_weights.append(
            weight
        )

    sample_weights = np.array(
        sample_weights
    )

    reweighted_model = LogisticRegression(
        max_iter=1000,
        random_state=seed,
    )

    reweighted_model.fit(
        X_train_proc,
        y_train,
        sample_weight=sample_weights,
    )

    reweighted_predictions = (
        reweighted_model.predict(
            X_test_proc
        )
    )

    add_result(
        all_results,
        seed,
        "Reweighing",
        y_test,
        reweighted_predictions,
        A_test,
    )

    print("Reweighing complete.")

    # ===================================================
    # 3. RESAMPLING
    # ===================================================

    X_rs = (
        X_train
        .reset_index(drop=True)
    )

    y_rs = (
        y_train
        .reset_index(drop=True)
    )

    A_rs = (
        A_train
        .reset_index(drop=True)
    )

    resample_df = X_rs.copy()

    resample_df[
        "income"
    ] = y_rs

    resample_df[
        "sex"
    ] = A_rs

    group_counts = (
        resample_df.groupby(
            [
                "sex",
                "income",
            ]
        ).size()
    )

    max_group_size = (
        group_counts.max()
    )

    balanced_groups = []

    # Balance all four sex/income combinations
    for sex_value in sorted(
        resample_df[
            "sex"
        ].unique()
    ):

        for income_value in sorted(
            resample_df[
                "income"
            ].unique()
        ):

            group = resample_df[
                (
                    resample_df[
                        "sex"
                    ]
                    == sex_value
                )
                &
                (
                    resample_df[
                        "income"
                    ]
                    == income_value
                )
            ]

            balanced_group = resample(
                group,
                replace=True,
                n_samples=max_group_size,
                random_state=seed,
            )

            balanced_groups.append(
                balanced_group
            )

    balanced_df = pd.concat(
        balanced_groups
    )

    # Shuffle balanced training data
    balanced_df = (
        balanced_df.sample(
            frac=1,
            random_state=seed,
        )
        .reset_index(
            drop=True
        )
    )

    y_balanced = (
        balanced_df[
            "income"
        ]
    )

    X_balanced = (
        balanced_df.drop(
            columns=[
                "income",
                "sex",
            ]
        )
    )

    # Use SAME preprocessor fitted
    # on original training data.
    X_balanced_proc = (
        preprocessor.transform(
            X_balanced
        )
    )

    resampled_model = LogisticRegression(
        max_iter=1000,
        random_state=seed,
    )

    resampled_model.fit(
        X_balanced_proc,
        y_balanced,
    )

    resampled_predictions = (
        resampled_model.predict(
            X_test_proc
        )
    )

    add_result(
        all_results,
        seed,
        "Resampling",
        y_test,
        resampled_predictions,
        A_test,
    )

    print("Resampling complete.")

    # ===================================================
    # 4. EQUALIZED ODDS
    # ===================================================

    # Fairlearn ThresholdOptimizer requires dense data
    # in the installed configuration.
    X_val_dense = (
        X_val_proc.toarray()
        if hasattr(
            X_val_proc,
            "toarray"
        )
        else X_val_proc
    )

    X_test_dense = (
        X_test_proc.toarray()
        if hasattr(
            X_test_proc,
            "toarray"
        )
        else X_test_proc
    )

    eo_postprocessor = ThresholdOptimizer(
        estimator=baseline_model,
        constraints="equalized_odds",
        objective="accuracy_score",
        prefit=True,
        predict_method="predict_proba",
    )

    # Use validation data to fit the fairness thresholds.
    eo_postprocessor.fit(
        X_val_dense,
        y_val,
        sensitive_features=A_val,
    )

    # Final evaluation occurs only on test data.
    eo_predictions = (
        eo_postprocessor.predict(
            X_test_dense,
            sensitive_features=A_test,
            random_state=seed,
        )
    )

    add_result(
        all_results,
        seed,
        "Equalized Odds",
        y_test,
        eo_predictions,
        A_test,
    )

    print(
        "Equalized odds complete."
    )


# ===================================================
# CREATE FINAL RESULTS TABLE
# ===================================================

results_df = pd.DataFrame(
    all_results
)

# Arrange columns in logical order
results_df = results_df[
    [
        "Seed",
        "Technique",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "DP Difference",
        "EOpp Difference",
        "EOdds Difference",
    ]
]

# Sort rows by seed and experiment order
technique_order = {
    "Baseline": 0,
    "Reweighing": 1,
    "Resampling": 2,
    "Equalized Odds": 3,
}

results_df[
    "_order"
] = results_df[
    "Technique"
].map(
    technique_order
)

results_df = (
    results_df.sort_values(
        [
            "Seed",
            "_order",
        ]
    )
    .drop(
        columns="_order"
    )
    .reset_index(
        drop=True
    )
)

print("\n")
print("=" * 70)
print("ALL EXPERIMENT RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ===================================================
# CALCULATE SUMMARY STATISTICS
# ===================================================

metric_columns = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "DP Difference",
    "EOpp Difference",
    "EOdds Difference",
]

summary = (
    results_df
    .groupby(
        "Technique"
    )[
        metric_columns
    ]
    .agg(
        [
            "mean",
            "std",
        ]
    )
)

# Flatten MultiIndex column names
summary.columns = [
    f"{metric}_{stat}"
    for metric, stat
    in summary.columns
]

# Put techniques into logical order
summary = summary.reindex(
    [
        "Baseline",
        "Reweighing",
        "Resampling",
        "Equalized Odds",
    ]
)

print("\n")
print("=" * 70)
print(
    "AVERAGE RESULTS ACROSS "
    "SEEDS 42, 43, AND 44"
)
print("=" * 70)

print(summary)


# ===================================================
# SAVE RESULTS
# ===================================================

results_df.to_csv(
    "results/all_seed_results.csv",
    index=False,
)

summary.to_csv(
    "results/summary_results.csv"
)

print("\n")
print(
    "Experiment complete."
)

print(
    "Results saved to:"
)

print(
    "results/all_seed_results.csv"
)

print(
    "results/summary_results.csv"
)