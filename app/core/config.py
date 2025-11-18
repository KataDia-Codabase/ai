from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import Optional

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = True
    
    # Google Cloud Settings
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = None
    GOOGLE_CLOUD_PROJECT: Optional[str] = None
    
    # Database Settings
    MYSQL_HOST: str = "localhost"
    MYSQL_PORT: int = 3306
    MYSQL_USER: str = "root"
    MYSQL_PASSWORD: str = "password"
    MYSQL_DATABASE: str = "katadia_ml"
    MYSQL_URI: str = "mysql+pymysql://root:password@localhost:3306/katadia_ml"
    REDIS_URL: str = "redis://localhost:6379"
    
    # OpenAI Settings
    OPENAI_API_KEY: Optional[str] = None
    
    # Model Settings
    WAV2VEC_MODEL_PATH: str = "./models/wav2vec2-indonesian"
    WAV2VEC_ENGLISH_MODEL_PATH: str = "./models/wav2vec2-english"
    MFA_MODEL_PATH: str = "./models/mfa"
    WHISPER_MODEL_SIZE: str = "base"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/app.log"
    
    # Performance Settings
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    MAX_CONCURRENT_REQUESTS: int = 100

    # For pydantic v2, use model_config to allow extra env vars (some env files
    # may contain dataset paths or other entries that aren't defined here).
    model_config = ConfigDict(extra="ignore", env_file=".env")

settings = Settings()
