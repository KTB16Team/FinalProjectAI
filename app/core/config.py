from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = " "
    # DATABASE_URL: str = " "
    OPENAI_API_KEY: str = ""
    CALLBACK_URL: str = ""
    ACCESSTOKEN: str = ""
    #MQ
    RABBITMQ_URL: str = ""
    RABBITMQ_USER: str = ""
    RABBITMQ_PASS: str = ""
    # AWS S3 관련 추가 설정
    AWS_S3_BUCKET_NAME: str = ""
    AWS_REGION: str = ""
    AWS_ACCESS_KEY: str = ""
    AWS_SECRET_KEY: str = ""
    #logstash
    LOGSTASH_HOST: str = ""
    class Config:
        env_file = ".env"

settings = Settings()