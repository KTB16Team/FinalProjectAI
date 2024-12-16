import random
import time
from fastapi import FastAPI, HTTPException, APIRouter
from prometheus_client import generate_latest, REGISTRY, Counter, Gauge, Histogram
from starlette.responses import PlainTextResponse
from core.logging import setup_logger


app = FastAPI()
router = APIRouter()
# logger = logging.getLogger("uvicorn")
# logger = setup_logger()
REQUESTS = Counter('http_requests_total', 'Total HTTP Requests (count)', ['method', 'endpoint', 'status_code'])
IN_PROGRESS = Gauge('http_requests_inprogress', 'Number of in-progress HTTP requests')
TIMINGS = Histogram('http_request_duration_seconds', 'HTTP request latency (seconds)')


@app.middleware("http")
async def prometheus_middleware(request, call_next):
    """요청 전후로 메트릭을 기록하는 미들웨어"""
    with IN_PROGRESS.track_inprogress():
        start_time = time.time()
        try:
            response = await call_next(request)
            status_code = response.status_code
        except HTTPException as e:
            status_code = e.status_code
            raise
        except Exception:
            status_code = 500
            raise
        finally:
            REQUESTS.labels(method=request.method, endpoint=request.url.path, status_code=status_code).inc()
            TIMINGS.observe(time.time() - start_time)
    return response

@router.get("/slow")
async def slow_request():
    v = random.expovariate(1.0 / 1.3)
    if v > 3:
        REQUESTS.labels(method='GET', endpoint="/slow", status_code=500).inc()
        raise HTTPException(status_code=500, detail="Simulated slow request failure")
    time.sleep(v)
    REQUESTS.labels(method='GET', endpoint="/slow", status_code=200).inc()
    return {"message": f"Wow, that took {v:.2f} seconds!"}

@router.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """Prometheus 메트릭을 반환하는 엔드포인트"""
    REQUESTS.labels(method='GET', endpoint="/metrics", status_code=200).inc()
    return PlainTextResponse(generate_latest(REGISTRY))
