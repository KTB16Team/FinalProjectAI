from fastapi import FastAPI

from app.api.endpoint.mediation_service import router as mediation_router
from app.core.config import settings


app = FastAPI(title=settings.PROJECT_NAME)

# mediation_router를 포함
app.include_router(mediation_router, prefix="/api/v1/private-posts")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)