from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_path: str = "artifacts/catboost_10.joblib"
    explainer_top_n: int = 5
    uploads_dir: str = "uploads"
    database_url: str | None = None
    log_level: str = "INFO"
    model_name: str | None = None
    model_alias: str = "champion"
    mlflow_tracking_uri: str = "http://127.0.0.1:5000"
    
    model_config = {"env_file": ".env"}

settings = Settings()