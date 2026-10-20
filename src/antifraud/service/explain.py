"""
Объяснения предсказаний через встроенный SHAP CatBoost
"""
from __future__ import annotations

from catboost import CatBoostClassifier, Pool

FEATURE_LABELS: dict[str, str] = {
    "amount": "Сумма транзакции",
    "step": "Временной шаг",
    "category": "Категория",
    "age": "Возраст",
    "gender": "Пол",
    "amount_to_customer_avg_amount": "Сумма к среднему клиента за всю историю",
    "amount_to_merchant_avg_amount": "Сумма к среднему мерчанта за всю историю",
    "customer_avg_amount_prev_step": "Средняя сумма по истории клиента",
    "customer_std_amount_prev_step": "Разброс сумм по истории клиента",
    "customer_transaction_count_prev_step": "Число транзакций клиента за всю историю",
    "customer_time_since_last_transaction": "Время с прошлой транзакции клиента",
    "has_customer_history": "Есть история клиента",
    "merchant_avg_amount_prev_step": "Среднее мерчанта за всю историю",
    "merchant_std_amount_prev_step": "Разброс сумм мерчанта за всю историю",
    "merchant_transaction_count_prev_step": "Число транзакций мерчанта за всю историю",
    "merchant_unique_customers_prev_step": "Число уникальных клиентов мерчанта за всю историю",
    "merchant_time_since_last_transaction": "Время с прошлой транзакции мерчанта",
    "has_merchant_history": "Есть история мерчанта",
}


def feature_label(name: str) -> str:
    return FEATURE_LABELS.get(name, name)


class Explainer:
    """
    Обёртка над CatBoostClassifier для расчёта SHAP-вкладов признаков.
    Создаётся один раз при старте приложения.
    """

    def __init__(self, model: CatBoostClassifier, feature_names: list[str]):
        self.model = model
        self.feature_names = feature_names
        self.cat_feature_idx = model.get_cat_feature_indices()

    def explain(self, frame, top_n: int = 5) -> list[list[tuple[str, float]]]:
        """
        frame: pd.DataFrame с колонками == self.feature_names, в том же
        порядке, что подавался в predict_proba.
        Возвращает список (по строке frame) списков (feature, shap_value),
        top_n по |shap_value|, отсортированных по убыванию модуля.
        Положительное значение повышает fraud-score.
        """
        pool = Pool(frame[self.feature_names], cat_features=self.cat_feature_idx)
        raw = self.model.get_feature_importance(pool, type="ShapValues")
        contrib = raw[:, :-1]

        results = []
        for row in contrib:
            pairs = sorted(
                zip(self.feature_names, row, strict=True),
                key=lambda pair: -abs(pair[1]),
            )[:top_n]
            results.append(pairs)
        return results

    def format_reasons(self, pairs: list[tuple[str, float]]) -> str:
        return "; ".join(
            f"{feature_label(name)}: {value:+.3f}" for name, value in pairs
        )
