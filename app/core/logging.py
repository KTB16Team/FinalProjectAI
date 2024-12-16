import logging
from logging.handlers import RotatingFileHandler

# 로그 파일 경로
LOG_FILE = "logs/app.log"

# 로깅 설정 함수
def setup_logger():
    # 로거 생성
    logger = logging.getLogger("my_fastapi_app")
    logger.setLevel(logging.INFO)

    # 파일 핸들러 추가 (RotatingFileHandler로 로그 크기 관리)
    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 콘솔 핸들러 추가
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

# 글로벌 로거 설정
logger = setup_logger()