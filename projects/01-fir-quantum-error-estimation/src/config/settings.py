from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    seed: int = Field(default=42)
    log_level: str = Field(default="INFO")
    model_artifact_path: str = Field(default="./models_artifacts")
    database_url: str = Field(default="sqlite:///./quantum_fir.db")
    redis_url: str = Field(default="redis://localhost:6379/0")
    mlflow_tracking_uri: str = Field(default="http://localhost:5000")
    num_qubits: int = Field(default=4)
    fir_order: int = Field(default=32)
    ibm_quantum_token: str = Field(default="")
    batch_size: int = Field(default=64)
    max_epochs: int = Field(default=100)
    learning_rate: float = Field(default=1e-3)
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

settings = Settings()
