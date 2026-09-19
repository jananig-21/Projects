from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    seed: int = Field(default=42)
    log_level: str = Field(default="INFO")
    model_artifact_path: str = Field(default="./models_artifacts")
    mlflow_tracking_uri: str = Field(default="http://localhost:5000")
    database_url: str = Field(default="sqlite:///./cps_faults.db")
    redis_url: str = Field(default="redis://localhost:6379/0")
    batch_size: int = Field(default=64)
    max_epochs: int = Field(default=100)
    learning_rate: float = Field(default=1e-3)
    window_size: int = Field(default=50)
    num_sensors: int = Field(default=20)
    fault_classes: int = Field(default=5)


settings = Settings()
