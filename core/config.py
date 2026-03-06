import logging
import os

# Initialize logger for the config module
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    GEMINI_API_KEY: str
    DB_URL: str
    DB_NAME: str
    PORT: int = 8000

    REDIS_URL: str
    SECRET_KEY: str = os.getenv("SECRET_KEY")
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")

    LLMSHERPA_API_URL: str
    LLAMA_CLOUD_API_KEY: str

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
