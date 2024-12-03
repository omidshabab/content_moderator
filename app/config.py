from pydantic_settings import BaseSettings

class Settings(BaseSettings):
     APP_NAME: str = "Content Moderator"
     DEBUG: bool = False
     MODEL_PATH: str = "models/ml"

     class Config:
          env_file = ".env"

settings = Settings()