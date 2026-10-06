"""
Объяснения предсказаний через встроенный SHAP CatBoost
(get_feature_importance(type="ShapValues")) — та же величина (log-odds),
что и обучающий SHAP summary plot, без отдельной зависимости shap.
"""
from __future__ import annotations

from catboost import CatBoostClassifier, Pool

# Человекочитаемые подписи для самых частых признаков; для остальных
# используется имя признака как есть.
FEATURE_LABELS: dict[str, str] = {
    "amount": "Сумма транзакции",
    "step": "Временной шаг",
    "category": "Категория",
    "age": "Возраст",
    "gender": "Пол",
    "amount_to_customer_avg_amount": "Сумма к среднему клиента",
    "amount_to_merchant_avg_amount": "Сумма к среднему мерчанта",
    "customer_avg_amount_prev_step": "Среднее клиента (пред. шаг)",
    "customer_std_amount_prev_step": "Разброс сумм клиента",
    "customer_transaction_count_prev_step": "Число транзакций клиента",
    "customer_time_since_last_transaction": "Время с прошлой транзакции клиента",
    "has_customer_history": "Есть история клиента",
    "merchant_avg_amount_prev_step": "Среднее мерчанта (пред. шаг)",
    "merchant_std_amount_prev_step": "Разброс сумм мерчанта",
    "merchant_transaction_count_prev_step": "Число транзакций мерчанта",
    "merchant_unique_customers_prev_step": "Уникальных клиентов мерчанта",
    "merchant_time_since_last_transaction": "Время с прошлой транзакции мерчанта",
    "has_merchant_history": "Есть история мерчанта",
}


def feature_label(name: str) -> str:
    return FEATURE_LABELS.get(name, name)


class Explainer:
    """
    Обёртка над CatBoostClassifier для расчёта SHAP-вкладов признаков.
    Создаётся один раз при старте приложения (cat_features уже знает
    сама обученная модель — model.get_cat_feature_indices()).
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
        топ top_n по |shap_value|, отсортированных по убыванию модуля.
        Положительное значение повышает risk-score (логиты класса fraud).
        """
        pool = Pool(frame[self.feature_names], cat_features=self.cat_feature_idx)
        raw = self.model.get_feature_importance(pool, type="ShapValues")
        contrib = raw[:, :-1]  # последняя колонка — base_value, не нужен для топа причин

        results = []
        for row in contrib:
            pairs = sorted(
                zip(self.feature_names, row),
                key=lambda pair: -abs(pair[1]),
            )[:top_n]
            results.append(pairs)
        return results

    def format_reasons(self, pairs: list[tuple[str, float]]) -> str:
        return "; ".join(
            f"{feature_label(name)}: {value:+.3f}" for name, value in pairs
        )
