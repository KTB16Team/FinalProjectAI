from fastapi import FastAPI
from api.endpoint.mediation_service import router as mediation_router
from api.endpoint.prometheus import router as prometheus_router
from core.config import settings
import asyncio
import threading
from api.endpoint.mediation_service import start_worker, start_second_worker
from contextlib import asynccontextmanager
from core.logging import logger

def run_worker_in_thread():
    """
    RabbitMQ 워커를 별도 스레드에서 실행
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    start_worker()

def run_second_worker_in_thread():
    """
    두 번째 RabbitMQ 워커를 별도 스레드에서 실행
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    start_second_worker()  # 두 번째 워커 실행

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan 핸들러로 RabbitMQ 워커 스레드 관리
    """
    logger.info("Initializing application...")
    worker_thread = threading.Thread(target=run_worker_in_thread, daemon=True)
    worker_thread.start()
    logger.info("RabbitMQ worker thread started.")

    # 두 번째 워커 실행
    second_worker_thread = threading.Thread(target=run_second_worker_in_thread, daemon=True)
    second_worker_thread.start()
    logger.info("Second OCR RabbitMQ worker thread started.")

    yield  # 애플리케이션 실행 유지

    logger.info("Shutting down application...")
    worker_thread.join(timeout=5)  # 스레드 종료 대기

app = FastAPI(title=settings.PROJECT_NAME, lifespan=lifespan)

# 라우터 포함
app.include_router(mediation_router, prefix="/api/v1/private-posts")
app.include_router(prometheus_router, prefix="")

if __name__ == "__main__":
    import uvicorn
    logger.info("FastAPI application is starting...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

