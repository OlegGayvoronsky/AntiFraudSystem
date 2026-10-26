"""
MLFLOW_TRACKING_URI=http://localhost:5000 uv run python -m antifraud.train

через временный докер контейнер
docker compose run --rm `
  -e MLFLOW_TRACKING_URI=http://mlflow:5000 `
  api uv run --no-sync python -m antifraud.train
"""
import copy
import hashlib
import json
import os
from pathlib import Path

import matplotlib
from tqdm.auto import tqdm

matplotlib.use("Agg")
import re

import catboost
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import sklearn
from catboost import CatBoostClassifier
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.base import clone as sklearn_clone
from sklearn.metrics import average_precision_score, confusion_matrix, roc_curve
from sklearn.model_selection import TimeSeriesSplit

from antifraud.service.preprocessing import FeatureBuilder, InMemoryStore

DATA_PATH = Path(os.getenv("DATA_PATH", "data/transactions.csv"))
CHUNK = int(os.getenv("CHUNK", "20000"))
MODEL_NAME = os.getenv("MODEL_NAME", "antifraud_catboost")
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "antifraud")

TRAIN_SIZE = float(os.getenv("TRAIN_SIZE", "0.8"))
TARGET = os.getenv("TARGET", "fraud")
TIME = os.getenv("TIME", "step")
TARGET_SPECIFICITY = float(os.getenv("TARGET_SPECIFICITY", "0.9985"))
N_SPLITS = int(os.getenv("N_SPLITS", "4"))
MIN_TRAIN_SIZE = float(os.getenv("MIN_TRAIN_SIZE", "0.5"))

ITERATIONS = int(os.getenv("ITERATIONS", "1300"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.03"))
DEPTH = int(os.getenv("DEPTH", "8"))
USE_EARLY_STOPPING = os.getenv("USE_EARLY_STOPPING", "False").lower() in ("1", "true", "yes")
EARLY_STOPPING_ROUNDS = int(os.getenv("EARLY_STOPPING_ROUNDS", "150"))
TARGET_REVIEW_RATE = float(os.getenv("REVIEW_RATE", "0.1"))
N_THRESHOLDS = int(os.getenv("N_THRESHOLDS", "100"))
FALSE_BLOCK_COST = float(os.getenv("FALSE_BLOCK_COST", "1.22"))
REVIEW_COST = float(os.getenv("REVIEW_COST", "0.12"))
SEED = 42

GATE_MIN_GAIN = float(os.getenv("GATE_MIN_GAIN", "0.0032"))

DROP = ["zipcodeOri", "zipMerchant"]
NUMERIC = [
    'amount',
    'customer_transaction_count_prev_step',
    'customer_avg_amount_prev_step',
    'customer_std_amount_prev_step',
    'customer_time_since_last_transaction',
    'amount_to_customer_avg_amount',
    'has_customer_history',
    'merchant_transaction_count_prev_step',
    'merchant_avg_amount_prev_step',
    'merchant_std_amount_prev_step',
    'merchant_time_since_last_transaction',
    'merchant_unique_customers_prev_step',
    'amount_to_merchant_avg_amount',
    'has_merchant_history'
]
CATEGORICAL = ['age', 'gender', 'category']


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Исторические признаки тем же кодом, что и на инференсе (FeatureBuilder),
    но на InMemoryStore и с commit=False: в БД ничего не пишется, состояние не копится.
    Индекс сохраняется: create_oot_splits работает через train.index / .loc.
    """
    builder = FeatureBuilder(InMemoryStore())
    enriched = builder.build(df.to_dict("records"), commit=False)
    return pd.DataFrame(enriched, index=df.index)


def load_and_validate(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=DROP, errors='ignore')
    
    required = {"step", "customer", "merchant", "age", "gender", "category", "amount", "fraud"}
    missing = required - set(df.columns)
    
    if missing:
        raise ValueError(f"в источнике нет колонок: {missing}")

    if len(df) < 1000:
            raise ValueError(f"слишком мало строк: {len(df)}")

    target_values = set(df["fraud"].unique())
    if not target_values <= {0, 1}:
        raise ValueError(f"неожиданные значения таргета: {sorted(target_values)[:5]}")

    return add_history_features(df)


def build_model(
    iterations: int = ITERATIONS,
    learning_rate: float = LEARNING_RATE,
    depth: int = DEPTH,
    cat_features: list = CATEGORICAL,
    seed: int = SEED,
) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        cat_features=cat_features,
        auto_class_weights="Balanced",
        random_seed=seed,
        verbose=False,
    )


def _fit_model(
    model: CatBoostClassifier,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame = None,
    y_val: pd.DataFrame = None,
    early_stopping_rounds=EARLY_STOPPING_ROUNDS
):
    """
    Обучает модель. Для CatBoostClassifier с переданным
    eval_set включает early stopping и возвращает
    best_iteration (None, если модель не CatBoost или
    eval_set не передан).
    """

    if X_val is not None:
        model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=early_stopping_rounds,
            use_best_model=True,
            verbose=False
        )
        best_iteration = model.get_best_iteration()
    else:
        model.fit(
            X_train,
            y_train
        )
        best_iteration = None

    return model, best_iteration


def compute_fraud_amount_recall(
        y_true,
        y_pred,
        amounts
):
    """
    Доля суммарного объёма (amount) фрод-транзакций,
    которые модель верно определила как fraud.
    Отличается от recall по количеству транзакций тем,
    что взвешивает каждую транзакцию по её сумме —
    релевантно, когда пропустить крупный fraud
    дороже, чем пропустить мелкий.
    """
    fraud_mask = (y_true == 1)

    total_fraud_amount = amounts[fraud_mask].sum()

    if total_fraud_amount == 0:
        return np.nan

    caught_fraud_amount = amounts[
        fraud_mask & (y_pred == 1)
    ].sum()

    return caught_fraud_amount / total_fraud_amount


def evaluate_experiment(
    y_train,
    train_proba,
    y_test,
    test_proba,
    test_amounts,
    T_hight,
    T_low,
    false_block_cost=FALSE_BLOCK_COST,
    review_cost=REVIEW_COST
):
    """
    Считает метрики эксперимента:
      - core-метрики (recall/specificity/precision на test, pr_auc)
      - overfitting-диагностика (train vs cv/test gap)
      - бизнес-метрика (fraud amount recall, missed fraud loss, review_rate)
    """

    y_train = np.asarray(y_train)
    train_proba = np.asarray(train_proba)

    y_test = np.asarray(y_test)
    test_proba = np.asarray(test_proba)
    test_amounts = np.asarray(test_amounts)

    pr_auc = float(average_precision_score(y_test, test_proba))

    train_pred = (train_proba >= T_hight).astype(int)

    _, _, train_fn, train_tp = confusion_matrix(
        y_train,
        train_pred,
        labels=[0, 1]
    ).ravel()

    train_recall = (
        train_tp / (train_tp + train_fn)
        if (train_tp + train_fn) > 0
        else 0
    )

    test_pred = (test_proba >= T_hight).astype(int)

    test_tn, test_fp, test_fn, test_tp = confusion_matrix(
        y_test,
        test_pred,
        labels=[0, 1]
    ).ravel()

    test_recall = (
        test_tp / (test_tp + test_fn)
        if (test_tp + test_fn) > 0
        else 0
    )

    test_specificity = (
        test_tn / (test_tn + test_fp)
        if (test_tn + test_fp) > 0
        else 0
    )

    test_precision = (
        test_tp / (test_tp + test_fp)
        if (test_tp + test_fp) > 0
        else 0
    )

    train_test_recall_gap = train_recall - test_recall

    fraud_amount_recall = compute_fraud_amount_recall(
        y_test,
        test_pred,
        test_amounts
    )

    allow_mask = test_proba < T_low
    
    review_mask = (
        (test_proba >= T_low)
        & (test_proba < T_hight)
    )

    block_mask = test_proba >= T_hight

    review_rate = float(review_mask.mean())

    missed_fraud_mask = allow_mask & (y_test == 1)

    missed_fraud_loss = float(
        test_amounts[missed_fraud_mask].sum()
    )

    false_block_mask = block_mask & (y_test == 0)

    false_block_count = int(false_block_mask.sum())
    false_block_total_cost = (
        false_block_count * false_block_cost
    )

    review_count = int(review_mask.sum())
    review_total_cost = review_count * review_cost

    total_cost_proxy = (
        missed_fraud_loss
        + false_block_total_cost
        + review_total_cost
    )

    return {
        "train_recall": train_recall,
        "train_test_recall_gap": train_test_recall_gap,

        "test_recall": test_recall,
        "test_specificity": test_specificity,
        "pr_auc": pr_auc,
        "test_precision": test_precision,

        "fraud_amount_recall": fraud_amount_recall,

        "T_hight": T_hight,
        "T_low": T_low,

        "total_cost_proxy": total_cost_proxy,

        "review_rate": review_rate,
        "review_count": review_count,
        "review_cost": review_cost,

        "false_block_total_cost": false_block_total_cost,
        "false_block_count": false_block_count,
        "false_block_cost": false_block_cost,

        "missed_fraud_loss": missed_fraud_loss,
    }


def make_train_test_split(df, train_size=TRAIN_SIZE):
    TEST_START_STEP = int(df.step.max() * train_size)

    train = df[df[TIME] < TEST_START_STEP].copy()
    test = df[df[TIME] >= TEST_START_STEP].copy()

    print(f"Train: {len(train):,}")
    print(f"Test:  {len(test):,}")

    print(
        f"Train steps: {train[TIME].min()} - {train[TIME].max()}"
    )
    print(
        f"Test steps:  {test[TIME].min()} - {test[TIME].max()}"
    )
    return train, test


def create_oot_splits(
    train,
    n_splits=N_SPLITS,
):
    steps = np.sort(train[TIME].unique())
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    oot_splits = []
    
    for fold, (train_step_idx, val_step_idx) in enumerate(tscv.split(steps), start=1):
        train_steps = steps[train_step_idx]
        val_steps = steps[val_step_idx]
    
        train_idx = train.index[
            train[TIME].isin(train_steps)
        ]
    
        val_idx = train.index[
            train[TIME].isin(val_steps)
        ]
    
        oot_splits.append((train_idx, val_idx))
    
        print(
            f"Fold {fold}: "
            f"train step {train_steps[0]}-{train_steps[-1]} "
            f"({len(train_idx):,} transactions), "
            f"val step {val_steps[0]}-{val_steps[-1]} "
            f"({len(val_idx):,} transactions)"
        )

    return oot_splits


def safe_clone(estimator):
    """
    Аналог sklearn.clone, но использует deepcopy вместо
    реконструкции через get_params/set_params.
    Нужен для CatBoost и других моделей, у которых
    get_params() не идентичен параметрам конструктора
    (из-за чего sklearn.clone падает с RuntimeError).
    """
    try:
        return sklearn_clone(estimator)
    except RuntimeError:
        return copy.deepcopy(estimator)


def recall_at_specificity(y_true, y_proba, target_specificity=TARGET_SPECIFICITY):
    """
    Максимизирует Recall при условии:
        Specificity >= target_specificity

    Возвращает:
        best_recall
        T_hight
        best_specificity
    """

    fpr, tpr, thresholds = roc_curve(y_true, y_proba)

    specificity = 1 - fpr
    recall = tpr

    valid = specificity >= target_specificity

    if not np.any(valid):
        return 0.0, 1.0, 0.0

    valid_recall = recall[valid]
    valid_specificity = specificity[valid]
    valid_thresholds = thresholds[valid]

    best_idx = np.argmax(valid_recall)

    best_recall = valid_recall[best_idx]
    T_hight = valid_thresholds[best_idx]
    best_specificity = valid_specificity[best_idx]

    return best_recall, T_hight, best_specificity


def find_T_low_array(
    y_true,
    y_proba,
    T_hight,
    train_amounts,
    max_review_share=TARGET_REVIEW_RATE,
    n_thresholds=N_THRESHOLDS,
    false_block_cost=FALSE_BLOCK_COST,
    review_cost=REVIEW_COST
):
    """
    Подбирает T_low для REVIEW-зоны [T_low, T_high).
    
    Допущение:
        REVIEW после проверки разрешается правильно:
        fraud останавливается, legitimate-транзакция разрешается.

    Стоимость:
        Total Cost Proxy =
            Missed Fraud Loss
            + False Block Cost
            + Review Cost
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    candidates = np.linspace(0.0, T_hight, n_thresholds)

    valid_thresholds = []

    for t_low in candidates:
        review_mask = (y_proba >= t_low) & (y_proba < T_hight)
        review_share = review_mask.mean()

        if review_share <= max_review_share:
            valid_thresholds.append(t_low)

    results = []

    for t_low in valid_thresholds:
        if not (0 <= t_low < T_hight):
            continue

        allow_mask = y_proba < t_low

        review_mask = (
            (y_proba >= t_low)
            & (y_proba < T_hight)
        )

        block_mask = y_proba >= T_hight

        review_rate = float(review_mask.mean())

        missed_fraud_mask = allow_mask & (y_true == 1)

        missed_fraud_loss = float(
            train_amounts[missed_fraud_mask].sum()
        )

        false_block_mask = block_mask & (y_true == 0)

        false_block_count = int(false_block_mask.sum())
        false_block_total_cost = (
            false_block_count * false_block_cost
        )

        review_count = int(review_mask.sum())
        review_total_cost = review_count * review_cost

        total_cost_proxy = (
            missed_fraud_loss
            + false_block_total_cost
            + review_total_cost
        )

        results.append({
            "T_low": float(t_low),
            "review_rate": review_rate,
            "review_count": review_count,
            "missed_fraud_loss": missed_fraud_loss,
            "false_block_count": false_block_count,
            "false_block_cost": float(false_block_total_cost),
            "review_cost": float(review_total_cost),
            "total_cost_proxy": float(total_cost_proxy),
        })

    if not results:
        raise ValueError(
            "Нет допустимых T_low"
        )

    best_result = min(
        results,
        key=lambda row: (
            row["total_cost_proxy"],
            row["review_rate"],
        ),
    )

    return best_result["T_low"]


def make_experiment(
    df,
    model,
    data_md5,
    num_features=NUMERIC,
    cat_features=CATEGORICAL,
    train_size=TRAIN_SIZE,
    n_splits=N_SPLITS,
    use_early_stopping=USE_EARLY_STOPPING,
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    amount_col="amount",
):
    features = num_features + cat_features

    train, test = make_train_test_split(
        df,
        train_size
    )

    X_train = train.loc[:, features].copy()
    X_test = test.loc[:, features].copy()
    y_train = train["fraud"]
    y_test = test["fraud"]

    test_amounts = test[amount_col]

    oot_splits = create_oot_splits(train, n_splits)

    oof_predictions = pd.Series(
        index=X_train.index,
        dtype=float
    )

    oof_best_iterations = []
    for train_idx, val_idx in tqdm(
        oot_splits,
        desc="OOF predictions",
        total=len(oot_splits)
    ):
        X_fold_train = X_train.loc[train_idx]
        y_fold_train = y_train.loc[train_idx]

        X_fold_val = X_train.loc[val_idx]
        y_fold_val = y_train.loc[val_idx]

        fold_model = safe_clone(model)
        fold_model, best_iteration = _fit_model(
            fold_model,
            X_fold_train,
            y_fold_train,
            X_val=X_fold_val if use_early_stopping else None,
            y_val=y_fold_val if use_early_stopping else None,
            early_stopping_rounds=early_stopping_rounds
        )

        if best_iteration is not None:
            oof_best_iterations.append(best_iteration)

        oof_predictions.loc[val_idx] = (
            fold_model.predict_proba(
                X_fold_val
            )[:, 1]
        )

    valid_oof = oof_predictions.notna()

    y_oof = y_train.loc[valid_oof]
    p_oof = oof_predictions.loc[valid_oof]
    amounts_oof = train.loc[y_oof.index, amount_col]
    
    _, T_hight, _ = (
        recall_at_specificity(
            y_oof,
            p_oof
        )
    )

    T_low = find_T_low_array(
        y_true=y_oof,
        y_proba=p_oof,
        T_hight=T_hight,
        train_amounts=amounts_oof
    )

    final_model = safe_clone(model)
    final_iterations = None

    if use_early_stopping and len(oof_best_iterations) > 0:
        final_iterations = int(
            np.median(oof_best_iterations)
        )
        final_model.set_params(
            iterations=final_iterations
        )

    final_model, _ = _fit_model(
        final_model,
        X_train,
        y_train
    )

    train_proba = final_model.predict_proba(
        X_train
    )[:, 1]

    test_proba = final_model.predict_proba(
        X_test
    )[:, 1]

    metrics = evaluate_experiment(
        y_train=y_train,
        train_proba=train_proba,
        y_test=y_test,
        test_proba=test_proba,
        test_amounts=test_amounts,
        T_hight=T_hight,
        T_low=T_low
    )

    metadata = {"features": features, "T_hight": float(T_hight), "T_low": float(T_low), "n_train": len(X_train),
                "numeric_cols": num_features, "categorical_cols": cat_features,
                "data_rows": len(df), "data_md5": data_md5,
                "sklearn": sklearn.__version__, "catboost": catboost.__version__,
                "pandas": pd.__version__, "numpy": np.__version__}
    
    return final_model, metadata, metrics, final_iterations


def champion_metrics(client: MlflowClient) -> tuple[str | None, float | None]:
    try:
        mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
    except MlflowException:
        return None, None, None
    return (
        mv.version,
        client.get_run(mv.run_id).data.metrics.get("test_recall"),
        client.get_run(mv.run_id).data.metrics.get("total_cost_proxy")
    )
    

def importance_figure(names, values):
    idxs = np.argsort(values)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(np.array(names)[idxs], np.array(values)[idxs])
    
    ax.set_xlabel("Важность (PredictionValuesChange)")
    ax.set_title("Важности признаков CatBoost")
    fig.tight_layout()
    
    return fig


def make_error_analysis(model, df, T_hight, train_size=TRAIN_SIZE, group_col="category", top_n=100):
    """Error analysis на test-части"""
    _, test = make_train_test_split(df, train_size)
    features = list(model.feature_names_)

    proba = model.predict_proba(test.loc[:, features])[:, 1]
    pred = proba >= T_hight
    fraud = test[TARGET].to_numpy().astype(bool)
    amount = test["amount"].to_numpy(dtype=float)

    res = pd.DataFrame({
        group_col: test[group_col].to_numpy(),
        "n_transactions": 1,
        "tp": (fraud & pred).astype(int),
        "fn": (fraud & ~pred).astype(int),
        "fp": (~fraud & pred).astype(int),
        "tn": (~fraud & ~pred).astype(int),
        "fraud_amount": np.where(fraud, amount, 0.0),
        "caught_amount": np.where(fraud & pred, amount, 0.0),
        "missed_amount": np.where(fraud & ~pred, amount, 0.0),
    })

    by_cat = res.groupby(group_col).sum().sort_values("missed_amount", ascending=False)
    by_cat = pd.concat([by_cat, by_cat.sum().to_frame("ALL").T])
    by_cat.index.name = group_col

    count_cols = ["n_transactions", "tp", "fn", "fp", "tn"]
    by_cat[count_cols] = by_cat[count_cols].astype(int)
    by_cat["n_fraud"] = by_cat["tp"] + by_cat["fn"]
    by_cat["fraud_rate"] = by_cat["n_fraud"] / by_cat["n_transactions"]
    by_cat["fnr"] = by_cat["fn"] / by_cat["n_fraud"].replace(0, np.nan)
    by_cat["fpr"] = by_cat["fp"] / (by_cat["fp"] + by_cat["tn"]).replace(0, np.nan)
    by_cat["fraud_amount_recall"] = by_cat["caught_amount"] / \
        by_cat["fraud_amount"].replace(0, np.nan)

    missed = fraud & ~pred
    keep = [c for c in ("transaction_id", "step", "customer", "merchant") if c in test.columns]
    top_missed = (
        test.loc[missed, keep + features]
        .assign(fraud_score=proba[missed])
        .nlargest(top_n, "amount")
        .reset_index(drop=True)
    )
    return by_cat, top_missed


def error_figure(by_cat):
    t = by_cat.drop(index="ALL").sort_values("fraud_amount")
    names = t.index.astype(str)

    fig, ax = plt.subplots(figsize=(7, 0.4 * len(t) + 1.5))
    ax.barh(names, t["caught_amount"], label="поймано")
    ax.barh(names, t["missed_amount"], left=t["caught_amount"], label="пропущено")
    ax.set_xlabel("Сумма фродовых транзакций")
    ax.set_title("Фрод по категориям: пойманный и пропущенный")
    ax.legend()
    fig.tight_layout()
    return fig


def log_error_analysis(model, df, T_hight):
    """Вызывать внутри активного mlflow run."""
    by_cat, top_missed = make_error_analysis(model, df, T_hight)

    total = by_cat.loc["ALL"]
    mlflow.log_metrics({
        "fraud_amount_total": float(total["fraud_amount"]),
        "fraud_amount_caught": float(total["caught_amount"]),
        "fraud_amount_missed": float(total["missed_amount"]),
    })

    per_cat = {}
    for cat, row in by_cat.drop(index="ALL").iterrows():
        name = re.sub(r"[^\w\-\. /]", "", str(cat)).strip()
        per_cat[f"cat/{name}/fraud_amount_missed"] = float(row["missed_amount"])
        if pd.notna(row["fraud_amount_recall"]):
            per_cat[f"cat/{name}/fraud_amount_recall"] = float(row["fraud_amount_recall"])
    mlflow.log_metrics(per_cat)

    table = by_cat.reset_index()
    mlflow.log_table(table, "error_analysis/by_category.json")
    mlflow.log_text(table.to_csv(index=False), "error_analysis/by_category.csv")
    mlflow.log_table(top_missed, "error_analysis/top_missed_fraud.json")

    fig = error_figure(by_cat)
    mlflow.log_figure(fig, "error_analysis/missed_by_category.png")
    plt.close(fig)

    print(table.round(4).to_string(index=False))


def main() -> dict:
    df = load_and_validate(DATA_PATH)
    data_md5 = file_md5(DATA_PATH)
    features = NUMERIC + CATEGORICAL

    model = build_model()
    model, metadata, metrics, final_iterations = make_experiment(df, model, data_md5)

    importances = dict(zip(features, map(float, model.get_feature_importance()), strict=True))

    mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient()

    with mlflow.start_run() as run:
        mlflow.log_params({
            "iterations": ITERATIONS,
            "final_iterations": final_iterations,
            "learning_rate": LEARNING_RATE,
            "depth": DEPTH,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "use_early_stopping": USE_EARLY_STOPPING,
            "model": "CatBoostClassifier",
            "seed": SEED,
            "data": str(DATA_PATH),
            "data_md5": data_md5,
            "gate_metric": "recall",
            "gate_min_gain": GATE_MIN_GAIN,
            "target_specificity": TARGET_SPECIFICITY,
            "target_review_rate": TARGET_REVIEW_RATE,
            "n_thresholds": N_THRESHOLDS,
            "false_block_cost": FALSE_BLOCK_COST,
            "review_cost": REVIEW_COST
        })

        mlflow.log_metrics({**metrics})
        mlflow.log_dict(metadata, "metadata.json")

        fig = importance_figure(list(importances), list(importances.values()))
        mlflow.log_figure(fig, "feature_importance.png")
        plt.close(fig)
        mlflow.log_dict(importances, "feature_importance.json")

        log_error_analysis(model, df, metrics["T_hight"])
        
        info = mlflow.catboost.log_model(
            model,
            name="model",
            registered_model_name=MODEL_NAME
        )
        version = info.registered_model_version

    old_version, old_recall, old_total_cost_proxy = champion_metrics(client)
    promoted = bool((old_recall is None) or (
        (metrics["test_recall"] > old_recall + GATE_MIN_GAIN)
        and
        (metrics["total_cost_proxy"] < old_total_cost_proxy)
    ))
    client.set_registered_model_alias(MODEL_NAME, "challenger", version)

    if promoted:
        client.set_registered_model_alias(MODEL_NAME, "champion", version)

    result = {
        "run_id": run.info.run_id, "version": version,
        "recall": round(metrics["test_recall"], 4),
        "total_cost_proxy": metrics["total_cost_proxy"],
        "data_md5": data_md5, "champion_before": old_version,
        "champion_recall_before": round(old_recall, 4) if old_recall is not None else None,
        "champion_total_cost_proxy_before": round(
            old_total_cost_proxy,
            4
        ) if old_total_cost_proxy is not None else None,
        "promoted": promoted
    }
    print(json.dumps(result, ensure_ascii=False))

    xcom = Path("/airflow/xcom")
    if xcom.is_dir():
        (xcom / "return.json").write_text(json.dumps(result))

    return result


if __name__ == "__main__":
    main()