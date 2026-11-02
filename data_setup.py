"""
data_setup.py

Loads and prepares the UCI Adult Income dataset for the thesis experiment.

Responsibilities:
- Download the Adult Income dataset.
- Clean missing values.
- Convert income to a binary target.
- Separate sex as the sensitive attribute.
- Create train/validation/test splits.
- Create the preprocessing pipeline.
"""

import pandas as pd

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_clean_data():
    """
    Download and clean the UCI Adult Income dataset.

    Returns
    -------
    X : pandas.DataFrame
        Predictor variables. The sensitive attribute 'sex'
        is excluded from model predictors.

    y : pandas.Series
        Binary income target:
        0 = <= $50K
        1 = > $50K

    A : pandas.Series
        Sensitive attribute containing sex.
    """

    # ---------------------------------------------------
    # Load Adult Income Dataset
    # ---------------------------------------------------

    adult = fetch_ucirepo(id=2)

    X = adult.data.features.copy()
    y = adult.data.targets.copy()

    # Combine features and target temporarily
    df = pd.concat([X, y], axis=1)

    print("Original dataset shape:")
    print(df.shape)

    # ---------------------------------------------------
    # Handle missing values
    # ---------------------------------------------------

    # UCI Adult may represent missing values using "?"
    df = df.replace("?", pd.NA)

    print("\nMissing values before cleaning:")
    print(df.isna().sum())

    # Remove rows containing missing values
    df = df.dropna().reset_index(drop=True)

    print("\nDataset shape after removing missing values:")
    print(df.shape)

    # ---------------------------------------------------
    # Clean income target
    # ---------------------------------------------------

    df["income"] = (
        df["income"]
        .astype(str)
        .str.strip()
        .str.replace(".", "", regex=False)
    )

    # Binary target:
    # <=50K = 0
    # >50K  = 1
    df["income_binary"] = df["income"].map(
        {
            "<=50K": 0,
            ">50K": 1,
        }
    )

    # Verify mapping worked correctly
    if df["income_binary"].isna().any():
        raise ValueError(
            "Some income values could not be converted "
            "to the binary target."
        )

    df["income_binary"] = df["income_binary"].astype(int)

    # ---------------------------------------------------
    # Clean sensitive attribute
    # ---------------------------------------------------

    df["sex"] = (
        df["sex"]
        .astype(str)
        .str.strip()
    )

    # Save sensitive attribute separately
    A = df["sex"].copy()

    # Save binary target
    y_clean = df["income_binary"].copy()

    # Remove target and sensitive attribute from model features
    X_model = df.drop(
        columns=[
            "income",
            "income_binary",
            "sex",
        ]
    )

    # ---------------------------------------------------
    # Display dataset information
    # ---------------------------------------------------

    print("\nIncome target distribution:")
    print(y_clean.value_counts())

    print("\nSensitive attribute distribution:")
    print(A.value_counts())

    print("\nPredictor columns:")
    print(X_model.columns.tolist())

    return X_model, y_clean, A


def split_data(X, y, A, seed):
    """
    Split the dataset into:

    60% training
    20% validation
    20% testing

    Stratification considers both income class and sex so
    that each split has similar group proportions.

    Parameters
    ----------
    X : pandas.DataFrame
    y : pandas.Series
    A : pandas.Series
    seed : int

    Returns
    -------
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    A_train, A_val, A_test
    """

    # Combine sex and income to preserve the distribution
    # of all four sex/income combinations.
    stratify_group = (
        A.astype(str)
        + "_"
        + y.astype(str)
    )

    # ---------------------------------------------------
    # First split:
    # 60% training
    # 40% temporary
    # ---------------------------------------------------

    (
        X_train,
        X_temp,
        y_train,
        y_temp,
        A_train,
        A_temp,
    ) = train_test_split(
        X,
        y,
        A,
        test_size=0.40,
        random_state=seed,
        stratify=stratify_group,
    )

    # ---------------------------------------------------
    # Second split:
    # Temporary 40% becomes
    # 20% validation
    # 20% testing
    # ---------------------------------------------------

    temp_stratify = (
        A_temp.astype(str)
        + "_"
        + y_temp.astype(str)
    )

    (
        X_val,
        X_test,
        y_val,
        y_test,
        A_val,
        A_test,
    ) = train_test_split(
        X_temp,
        y_temp,
        A_temp,
        test_size=0.50,
        random_state=seed,
        stratify=temp_stratify,
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        A_train,
        A_val,
        A_test,
    )


def create_preprocessor(X_train):
    """
    Create the preprocessing pipeline.

    Numerical variables:
    - StandardScaler

    Categorical variables:
    - OneHotEncoder

    The preprocessor should only be fitted on training data.
    """

    numeric_features = (
        X_train.select_dtypes(
            include=["number"]
        )
        .columns
        .tolist()
    )

    categorical_features = (
        X_train.select_dtypes(
            include=[
                "object",
                "string",
                "category",
            ]
        )
        .columns
        .tolist()
    )

    print("\nNumeric features:")
    print(numeric_features)

    print("\nCategorical features:")
    print(categorical_features)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                numeric_features,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features,
            ),
        ]
    )

    return preprocessor