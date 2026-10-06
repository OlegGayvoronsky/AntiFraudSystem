from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_path: str = "artifacts/catboost_10.joblib"
    explainer_top_n: int = 5
    manual_review_threshold: float = 0.2
    uploads_dir: str = "uploads"
    database_url: str | None = None
    log_level: str = "INFO"
    model_config = {"env_file": ".env"}

settings = Settings()