import logging
from logging.handlers import RotatingFileHandler
#from logstash import TCPLogstashHandler
from core.config import settings
# 로그 파일 경로
LOG_FILE = "logs/app.log"

# Logstash 서버 정보
#LOGSTASH_HOST = settings.LOGSTASH_HOST  # Logstash 서버 IP 또는 도메인
#LOGSTASH_PORT = 5045                 # Logstash에서 수신하는 TCP 포트

# 로깅 설정 함수
def setup_logger():
    # 로거 생성
    logger = logging.getLogger("my_fastapi_app")
    logger.setLevel(logging.INFO)

    # 파일 핸들러 추가
    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5)
    formatter = logging.Formatter(
    "%(asctime)s [%(threadName)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S"
)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 콘솔 핸들러 추가
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Logstash 핸들러 추가
    #logstash_handler = TCPLogstashHandler(host=LOGSTASH_HOST, port=LOGSTASH_PORT, version=1)
    #logger.addHandler(logstash_handler)

    # LoggerAdapter 생성
    extra = {'application_name': 'python_server'}
    logger_adapter = logging.LoggerAdapter(logger, extra)

    return logger_adapter

# 글로벌 로거 설정
logger = setup_logger()
