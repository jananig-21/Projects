from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    seed: int = Field(default=42)
    log_level: str = Field(default="INFO")
    model_artifact_path: str = Field(default="./models_artifacts")
    mlflow_tracking_uri: str = Field(default="http://localhost:5000")
    batch_size: int = Field(default=256)
    max_epochs: int = Field(default=200)
    learning_rate: float = Field(default=1e-3)
    grid_size: int = Field(default=5)
    spline_order: int = Field(default=3)


settings = Settings()
