import json
from fastapi import APIRouter, HTTPException, Header, status
import httpx
from sqlalchemy import text
from datetime import datetime
# from app.db.database import engine
from app.models.info import DataInfoSummary, VoiceInfo, DataInfoSTT,JudgeRequest,STTRequest
from app.services.situation_summary import situation_summary_GPT,stt_model,generate_response,test_response
import logging

router = APIRouter()
logger = logging.getLogger("uvicorn")

@router.post("/speech-to-text", response_model=VoiceInfo, status_code=201)
async def get_voice(request: STTRequest, authorization: str = Header(...)):
    # 인증 헤더 검사
    if not authorization.startswith("Bear "):
        raise HTTPException(status_code=401, detail="AUTH-001")

    logger.info("get_infos start")
    logger.info(f"audio URL : {request.url}")

    # STT 처리 예시
    try:
        entities = stt_model(request.url)
        logger.info(f"entities : {entities}")
    except Exception as e:
        logger.error(f"Error processing STT request: {e}")
        raise HTTPException(status_code=500, detail="COMMON-003")

    # 응답 데이터 생성
    response = VoiceInfo(
        status="Created",
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        data=DataInfoSTT(
            script=entities.get("ai_stt")
        )
    )

    logger.info(f"response : {response}")
    return response

# 첫 번째 엔드포인트 /api/v1/ai/private-posts/judgement
@router.post("/judgement", response_model=DataInfoSummary,status_code=201)
async def process_judge(request: JudgeRequest, authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="AUTH-001")

    logger.info("Starting judge processing")
    try:
        entities = test_response(request.content)
        logger.info(f"Entities processed: {entities}")
    except Exception as e:
        logger.error(f"Error in GPT processing: {e}")
        raise HTTPException(status_code=500, detail="COMMON-003")

    response = DataInfoSummary(
        title=entities.get("title"),
        stancePlaintiff=entities.get("stance_plaintiff"),
        stanceDefendant=entities.get("stance_defendant"),
        summaryAi=entities.get("situation_summary"),
        judgement=entities.get("judgement"),
        faultRate=entities.get("fault_rate")
    )
    return response

# async def read_user_info():
#     query = text("SELECT user_info FROM user")
#     async with engine.connect() as conn:
#         result = await conn.execute(query)
#         user_info = [row[0] for row in result.fetchall()]
#     return user_info
