from __future__ import annotations

import numpy as np
import pandas as pd


def make_customer_agg_features(trans: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    trans = trans.sort_values(["customer", "step"]).copy()

    customer_step_stats = (
        trans
        .groupby(["customer", "step"])["amount"]
        .agg(
            customer_transaction_count_prev_step="count",
            customer_avg_amount_prev_step="mean",
            customer_std_amount_prev_step="std"
        )
        .reset_index()
    )

    customer_step_stats = customer_step_stats.sort_values(["customer", "step"])

    customer_step_stats[
        [
            "customer_transaction_count_prev_step",
            "customer_avg_amount_prev_step",
            "customer_std_amount_prev_step",
        ]
    ] = (
        customer_step_stats
        .groupby("customer")[
            [
                "customer_transaction_count_prev_step",
                "customer_avg_amount_prev_step",
                "customer_std_amount_prev_step",
            ]
        ]
        .shift(1)
    )

    customer_step_stats["prev_transaction_step"] = (
        customer_step_stats
        .groupby("customer")["step"]
        .shift(1)
    )

    trans = trans.merge(
        customer_step_stats,
        on=["customer", "step"],
        how="left"
    )

    trans["customer_time_since_last_transaction"] = (
        trans["step"] - trans["prev_transaction_step"]
    )
    trans = trans.drop(columns=["prev_transaction_step"])

    avg = trans["customer_avg_amount_prev_step"]
    trans["amount_to_customer_avg_amount"] = np.where(
        avg > 0,
        trans["amount"] / avg,
        np.nan
    )

    trans["amount_to_customer_avg_amount"] = (
        trans["amount_to_customer_avg_amount"]
        .replace([np.inf, -np.inf], np.nan)
    )

    trans["has_customer_history"] = (
        trans["customer_transaction_count_prev_step"].notna().astype(int)
    )

    customer_history_features = [
        "customer_transaction_count_prev_step",
        "customer_avg_amount_prev_step",
        "customer_std_amount_prev_step",
        "customer_time_since_last_transaction",
        "amount_to_customer_avg_amount",
    ]

    trans[customer_history_features] = trans[customer_history_features].fillna(0)

    customer_history_features.append("has_customer_history")
    return trans, customer_history_features


def make_merchant_agg_features(trans: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    trans = trans.sort_values(["merchant", "step"]).copy()
    merchant_step_stats = (
        trans
        .groupby(["merchant", "step"])["amount"]
        .agg(
            merchant_transaction_count_prev_step="count",
            merchant_avg_amount_prev_step="mean",
            merchant_std_amount_prev_step="std"
        )
        .reset_index()
    )

    merchant_step_stats = merchant_step_stats.sort_values(
        ["merchant", "step"]
    )

    merchant_step_stats[
        [
            "merchant_transaction_count_prev_step",
            "merchant_avg_amount_prev_step",
            "merchant_std_amount_prev_step",
        ]
    ] = (
        merchant_step_stats
        .groupby("merchant")[
            [
                "merchant_transaction_count_prev_step",
                "merchant_avg_amount_prev_step",
                "merchant_std_amount_prev_step",
            ]
        ]
        .shift(1)
    )

    merchant_step_stats["prev_transaction_step"] = (
        merchant_step_stats
        .groupby("merchant")["step"]
        .shift(1)
    )

    trans = trans.merge(
        merchant_step_stats,
        on=["merchant", "step"],
        how="left"
    )

    trans["merchant_time_since_last_transaction"] = (
        trans["step"] - trans["prev_transaction_step"]
    )
    trans = trans.drop(
        columns=["prev_transaction_step"]
    )

    merchant_customer_stats = (
        trans
        .groupby(["merchant", "step"])["customer"]
        .nunique()
        .reset_index(
            name="merchant_unique_customers_prev_step"
        )
    )
    merchant_customer_stats = merchant_customer_stats.sort_values(
        ["merchant", "step"]
    )
    merchant_customer_stats[
        "merchant_unique_customers_prev_step"
    ] = (
        merchant_customer_stats
        .groupby("merchant")[
            "merchant_unique_customers_prev_step"
        ]
        .shift(1)
    )
    trans = trans.merge(
        merchant_customer_stats,
        on=["merchant", "step"],
        how="left"
    )

    avg = trans["merchant_avg_amount_prev_step"]
    trans["amount_to_merchant_avg_amount"] = np.where(
        avg > 0,
        trans["amount"] / avg,
        np.nan
    )

    trans["has_merchant_history"] = (
        trans["merchant_transaction_count_prev_step"]
        .notna()
        .astype(int)
    )

    trans["amount_to_merchant_avg_amount"] = (
        trans["amount_to_merchant_avg_amount"]
        .replace([np.inf, -np.inf], np.nan)
    )

    merchant_history_features = [
        "merchant_transaction_count_prev_step",
        "merchant_avg_amount_prev_step",
        "merchant_std_amount_prev_step",
        "merchant_time_since_last_transaction",
        "merchant_unique_customers_prev_step",
        "amount_to_merchant_avg_amount",
    ]

    trans[merchant_history_features] = (
        trans[merchant_history_features]
        .fillna(0)
    )

    merchant_history_features.append("has_merchant_history")
    return trans, merchant_history_features


def make_features(trans: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    trans, customer_history_features = make_customer_agg_features(trans)
    trans, merchant_history_features = make_merchant_agg_features(trans)

    history_features = customer_history_features + merchant_history_features

    return trans, history_features