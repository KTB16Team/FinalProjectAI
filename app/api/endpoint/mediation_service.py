import json

from fastapi import APIRouter
from sqlalchemy import text

from app.db.database import engine
from app.models.info import DataInfoSummary, voice_info
from app.services.situation_summary import situation_summary_GPT,stt_model,generate_response
import logging

router = APIRouter()
logger = logging.getLogger("uvicorn")

situation_summary
@router.post("/stt", response_model=STTResponse, status_code=201)
async def get_voice(request: STTRequest, authorization: str = Header(...)):
    # 인증 헤더 검사
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="AUTH-001")

    logger.info("get_infos start")
    logger.info(f"audio URL : {request.audio}")

    # STT 처리 예시
    try:
        entities = json.loads(stt_model(request.audio))
        logger.info(f"entities : {entities}")
    except Exception as e:
        logger.error(f"Error processing STT request: {e}")
        raise HTTPException(status_code=500, detail="COMMON-003")

    # 응답 데이터 생성
    response = STTResponse(
        status="Created",
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        data=STTResponseData(
            title="제목",
            ai_stt=entities.get("ai_stt", "AI STT 결과")
        )
    )

    logger.info(f"response : {response}")
    return response

# 첫 번째 엔드포인트 /api/v1/ai/private-posts/judge
@router.post("/judge", response_model=DataInfoSummary)
async def process_judge(request: DataInfoSummary, authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="AUTH-001")

    logger.info("Starting judge processing")
    try:
        entities = json.loads(situation_summary_GPT(request.summary_ai))
        logger.info(f"Entities processed: {entities}")
    except Exception as e:
        logger.error(f"Error in GPT processing: {e}")
        raise HTTPException(status_code=500, detail="COMMON-003")

    response = DataInfoSummary(
        title=request.title,
        stance_plaintiff=entities.get("stance_plaintiff"),
        stance_defendant=entities.get("stance_defendant"),
        summary_ai=entities.get("summary_ai"),
        judgement=entities.get("judgement"),
        fault_rate=entities.get("fault_rate")
    )

    # /api/v1/private-posts로 결과 전송
    async with httpx.AsyncClient() as client:
        try:
            await client.post(
                "http://localhost:8000/api/v1/private-posts",
                json=response.dict(),
                headers={"Authorization": authorization}
            )
        except httpx.HTTPError as err:
            logger.error(f"Failed to forward to /api/v1/private-posts: {err}")
            raise HTTPException(status_code=500, detail="COMMON-003")

    return response

async def read_user_info():
    query = text("SELECT user_info FROM user")
    async with engine.connect() as conn:
        result = await conn.execute(query)
        user_info = [row[0] for row in result.fetchall()]
    return user_info
