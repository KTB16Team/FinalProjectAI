from fastapi import FastAPI
from api.endpoint.mediation_service import router as mediation_router
from api.endpoint.prometheus import router as prometheus_router
from core.config import settings
import asyncio
import threading
from api.endpoint.mediation_service import start_worker

app = FastAPI(title=settings.PROJECT_NAME)

# mediation_router를 포함
app.include_router(mediation_router, prefix="/api/v1/private-posts")
app.include_router(prometheus_router, prefix="")
def run_worker_in_thread():
    """
    RabbitMQ 워커를 별도 스레드에서 실행
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    start_worker()

if __name__ == "__main__":
    import uvicorn

    # RabbitMQ 워커 실행
    worker_thread = threading.Thread(target=run_worker_in_thread, daemon=True)
    worker_thread.start()

    # FastAPI 서버 실행
    uvicorn.run(app, host="0.0.0.0", port=8000)