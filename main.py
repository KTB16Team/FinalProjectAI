from fastapi import FastAPI

from app.api.endpoint.mediation_service import process_judge, get_voice
from app.core.config import settings


app = FastAPI(title=settings.PROJECT_NAME)

app.include_router(process_judge.router, prefix="/api/v1/ai/private-posts")
app.include_router(get_voice.router, prefix="/api/v1/ai/private-posts")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)