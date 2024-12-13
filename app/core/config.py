from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = " "
    # DATABASE_URL: str = " "
    OPENAI_API_KEY: str = ""
    CALLBACK_URL: str = ""
    ACCESSTOKEN: str = ""
    class Config:
        env_file = ".env"

settings = Settings()