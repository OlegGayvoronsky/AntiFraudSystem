import joblib
import mlflow
from mlflow import MlflowClient

from antifraud.config import settings


def load_model() -> tuple[object, dict, str]:
    """Модель из реестра MLflow по алиасу, а без MODEL_NAME из файла, как раньше."""
    if not settings.model_name:
        bundle = joblib.load(settings.model_path)
        return bundle["model"], bundle["metadata"], bundle["metadata"]["model_version"]

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mv = MlflowClient().get_model_version_by_alias(settings.model_name, settings.model_alias)
    model = mlflow.catboost.load_model(f"models:/{settings.model_name}/{mv.version}")
    meta = mlflow.artifacts.load_dict(f"runs:/{mv.run_id}/metadata.json")
    return model, meta, f"{settings.model_name}-v{mv.version}"