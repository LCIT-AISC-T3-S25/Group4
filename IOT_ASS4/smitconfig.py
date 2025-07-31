import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

class Settings(BaseSettings):
    # Groq API
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama3-8b-8192")
    
    # Server
    host: str = os.getenv("HOST", "localhost")
    port: int = int(os.getenv("PORT", "8000"))
    websocket_port: int = int(os.getenv("WEBSOCKET_PORT", "8001"))
    
    # RL Training
    training_episodes: int = int(os.getenv("TRAINING_EPISODES", "1000"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "32"))
    learning_rate: float = float(os.getenv("LEARNING_RATE", "0.0003"))
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

settings = Settings()